#include <mpi.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <memory>

using namespace std;

constexpr int N_BODIES = 3;
constexpr double THETA = 0.5;
constexpr double G = 1.0;
constexpr double DT = 0.01;
constexpr int N_ITERATIONS = 100;
constexpr int OUTPUT_INTERVAL = 10;

struct Body {
    double x, y;
    double vx, vy;
    double m;
};

struct Node {
    double x_min, x_max;
    double y_min, y_max;

    double mass = 0.0;
    double cmx  = 0.0;
    double cmy  = 0.0;

    int body = -1;  // 如果是 leaf 且只有一个粒子，则记录 index
    unique_ptr<Node> children[4];

    Node(double xmin, double xmax, double ymin, double ymax)
        : x_min(xmin), x_max(xmax), y_min(ymin), y_max(ymax) {}
};

vector<Body> bodies(N_BODIES);

int get_quadrant(Node* node, const Body& b) {
    double midx = (node->x_min + node->x_max)/2;
    double midy = (node->y_min + node->y_max)/2;
    int q = 0;
    if (b.x >= midx) q += 1;
    if (b.y >= midy) q += 2;
    return q;
}

void child_bounds(Node* node, int q,
                  double& xmin, double& xmax,
                  double& ymin, double& ymax) {
    double midx = (node->x_min + node->x_max)/2;
    double midy = (node->y_min + node->y_max)/2;

    switch(q) {
        case 0: xmin=node->x_min; xmax=midx; ymin=node->y_min; ymax=midy; break; // 左下
        case 1: xmin=midx; xmax=node->x_max; ymin=node->y_min; ymax=midy; break; // 右下
        case 2: xmin=node->x_min; xmax=midx; ymin=midy; ymax=node->y_max; break; // 左上
        case 3: xmin=midx; xmax=node->x_max; ymin=midy; ymax=node->y_max; break; // 右上
    }
}

void update_mass(Node* node, const Body& b) {
    double total = node->mass + b.m;
    if (total == 0) return;

    node->cmx = (node->cmx * node->mass + b.x * b.m) / total;
    node->cmy = (node->cmy * node->mass + b.y * b.m) / total;
    node->mass = total;
}

// 插入粒子
void insert_body(Node* node, int idx) {
    Body& b = bodies[idx];

    // 空节点（leaf）
    if (node->mass == 0.0 && node->children[0] == nullptr) {
        node->body = idx;
        update_mass(node, b);
        return;
    }

    // leaf 且已有一个粒子 → 分裂成 4 子节点
    if (node->children[0] == nullptr && node->body != -1) {
        int old = node->body;
        node->body = -1;

        for (int i = 0; i < 4; i++) {
            double xmin, xmax, ymin, ymax;
            child_bounds(node, i, xmin, xmax, ymin, ymax);
            node->children[i] = make_unique<Node>(xmin, xmax, ymin, ymax);
        }

        int q_old = get_quadrant(node, bodies[old]);
        insert_body(node->children[q_old].get(), old);
    }

    int q = get_quadrant(node, b);
    insert_body(node->children[q].get(), idx);

    update_mass(node, b);
}

// 计算粒子引力
void compute_force(Node* node, int idx, double& fx, double& fy) {
    if (!node || node->mass == 0.0) return;
    Body& bi = bodies[idx];

    if (node->body == idx && node->children[0] == nullptr) return;

    double dx = node->cmx - bi.x;
    double dy = node->cmy - bi.y;
    double dist = sqrt(dx*dx + dy*dy + 1e-9);
    double size = node->x_max - node->x_min;

    // Barnes–Hut 判断
    if (node->children[0] == nullptr || (size / dist < THETA)) {
        double F = G * bi.m * node->mass / (dist*dist + 1e-9);
        fx += F * dx / dist;
        fy += F * dy / dist;
        return;
    }

    for (auto& ch : node->children)
        if (ch) compute_force(ch.get(), idx, fx, fy);
}

void init_bodies() {
    bodies[0] = {-0.5,  0.0,  0.0,  0.4, 1.0};
    bodies[1] = { 0.5,  0.0,  0.0, -0.4, 1.0};
    bodies[2] = { 0.0,  0.5, -0.4,  0.0, 0.5};
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 定义 MPI 结构体
    MPI_Datatype MPI_BODY;
    int blocklen[1] = {5};
    MPI_Aint disp[1] = {0};
    MPI_Datatype types[1] = {MPI_DOUBLE};
    MPI_Type_create_struct(1, blocklen, disp, types, &MPI_BODY);
    MPI_Type_commit(&MPI_BODY);

    if (rank == 0)
        init_bodies();

    MPI_Bcast(bodies.data(), N_BODIES, MPI_BODY, 0, MPI_COMM_WORLD);

    // 时间迭代循环
    for (int iter = 0; iter < N_ITERATIONS; iter++) {
        // 建树：所有进程本地构一棵一样的树
        auto root = make_unique<Node>(-1, 1, -1, 1);

        for (int i = 0; i < N_BODIES; i++)
            insert_body(root.get(), i);

        // 分片计算：只计算 i % size == rank 的粒子
        for (int i = 0; i < N_BODIES; i++) {
            if (i % size != rank) continue;

            double fx = 0, fy = 0;
            compute_force(root.get(), i, fx, fy);

            double ax = fx / bodies[i].m;
            double ay = fy / bodies[i].m;

            bodies[i].vx += ax * DT;
            bodies[i].vy += ay * DT;
            bodies[i].x  += bodies[i].vx * DT;
            bodies[i].y  += bodies[i].vy * DT;
        }

        // 同步所有进程的 bodies 数据：每个进程发送自己更新的部分
        // 使用 Allgather 收集所有更新
        for (int i = 0; i < N_BODIES; i++) {
            if (i % size == rank) {
                // 这个进程负责更新粒子 i，广播给所有进程
                MPI_Bcast(&bodies[i], 1, MPI_BODY, rank, MPI_COMM_WORLD);
            } else {
                // 接收其他进程更新的粒子
                MPI_Bcast(&bodies[i], 1, MPI_BODY, i % size, MPI_COMM_WORLD);
            }
        }

        // 输出（只在某些迭代时输出，避免输出过多）
        if (iter % OUTPUT_INTERVAL == 0) {
            MPI_Barrier(MPI_COMM_WORLD);
            for (int i = 0; i < N_BODIES; i++) {
                if (i % size == rank) {
                    cout << "Rank " << rank << " body " << i
                         << "   x=" << bodies[i].x << " y=" << bodies[i].y
                         << "   vx=" << bodies[i].vx << " vy=" << bodies[i].vy << endl;
                }
            }
            
            // 计算并输出质心位置（在 rank 0 上计算）
            if (rank == 0) {
                double total_mass = 0.0;
                double cmx = 0.0, cmy = 0.0;
                for (int i = 0; i < N_BODIES; i++) {
                    total_mass += bodies[i].m;
                    cmx += bodies[i].m * bodies[i].x;
                    cmy += bodies[i].m * bodies[i].y;
                }
                if (total_mass > 0) {
                    cmx /= total_mass;
                    cmy /= total_mass;
                }
                cout << "\nCenter of Mass: (" << cmx << ", " << cmy << ")" << endl;
                cout << "=== Iteration " << iter << " ===\n" << endl;
            }
        }
    }

    MPI_Type_free(&MPI_BODY);
    MPI_Finalize();
    return 0;
}

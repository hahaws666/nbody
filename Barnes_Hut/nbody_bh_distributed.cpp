#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>

using namespace std;

// ==== 配置参数 ====
constexpr double THETA           = 0.3;   // Barnes–Hut 阈值
constexpr double G               = 1.0;   // 万有引力常数
constexpr double DT              = 0.01;  // 时间步长
constexpr int    N_ITERATIONS    = 100;   // 迭代次数
constexpr int    OUTPUT_INTERVAL = 10;    // 监控输出间隔
constexpr int    N_TRACKED       = 500;   // 每步追踪的最大天体数（按全局 index 前 N_TRACKED 个）

int N_BODIES_GLOBAL = 1000;             // 总粒子数（命令行可改）

// ==== 数据结构 ====
struct Body {
    double x, y;
    double vx, vy;
    double m;
    double ax, ay;

    Body(): x(0), y(0), vx(0), vy(0), m(0), ax(0), ay(0) {}
    Body(double x_, double y_, double vx_, double vy_, double m_)
        : x(x_), y(y_), vx(vx_), vy(vy_), m(m_), ax(0), ay(0) {}
};

// Barnes–Hut 四叉树节点（全局树，所有 rank 上一致）
struct Node {
    double x_min, x_max;
    double y_min, y_max;

    double mass  = 0.0;
    double cmx   = 0.0;
    double cmy   = 0.0;

    int body = -1;                        // 叶子且只包含一个粒子时的全局 index
    unique_ptr<Node> children[4];

    Node(double xmin, double xmax, double ymin, double ymax)
        : x_min(xmin), x_max(xmax), y_min(ymin), y_max(ymax) {}
};

// ==== 全局 MPI 类型 ====
MPI_Datatype MPI_BODY;

// ==== 工具函数 ====
int get_quadrant(const Node* node, const Body& b) {
    double midx = 0.5 * (node->x_min + node->x_max);
    double midy = 0.5 * (node->y_min + node->y_max);
    int q = 0;
    if (b.x >= midx) q += 1;
    if (b.y >= midy) q += 2;
    return q;  // 0: 左下, 1: 右下, 2: 左上, 3: 右上
}

void child_bounds(const Node* node, int q,
                  double& xmin, double& xmax,
                  double& ymin, double& ymax) {
    double midx = 0.5 * (node->x_min + node->x_max);
    double midy = 0.5 * (node->y_min + node->y_max);
    switch (q) {
        case 0: xmin = node->x_min; xmax = midx;  ymin = node->y_min; ymax = midy;  break;
        case 1: xmin = midx;        xmax = node->x_max; ymin = node->y_min; ymax = midy;  break;
        case 2: xmin = node->x_min; xmax = midx;  ymin = midy;        ymax = node->y_max; break;
        case 3: xmin = midx;        xmax = node->x_max; ymin = midy;        ymax = node->y_max; break;
    }
}

void update_mass(Node* node, const Body& b) {
    double total = node->mass + b.m;
    if (total == 0.0) return;
    node->cmx = (node->cmx * node->mass + b.x * b.m) / total;
    node->cmy = (node->cmy * node->mass + b.y * b.m) / total;
    node->mass = total;
}

// 把“全局 bodies 数组中的第 idx 个粒子”插入到树中
void insert_body(Node* node, int idx, const vector<Body>& bodies) {
    const Body& b = bodies[idx];

    // 空节点
    if (node->mass == 0.0 && node->children[0] == nullptr) {
        node->body = idx;
        update_mass(node, b);
        return;
    }

    // 叶子且已有一个粒子 → 拆分
    if (node->children[0] == nullptr && node->body != -1) {
        int old = node->body;
        node->body = -1;

        for (int i = 0; i < 4; ++i) {
            double xmin, xmax, ymin, ymax;
            child_bounds(node, i, xmin, xmax, ymin, ymax);
            node->children[i] = make_unique<Node>(xmin, xmax, ymin, ymax);
        }

        int q_old = get_quadrant(node, bodies[old]);
        insert_body(node->children[q_old].get(), old, bodies);
    }

    int q = get_quadrant(node, b);
    insert_body(node->children[q].get(), idx, bodies);

    update_mass(node, b);
}

// 基于“全局 bodies 数组”构建 Barnes–Hut 树（所有 rank 上完全一样）
unique_ptr<Node> build_tree_global(const vector<Body>& bodies) {
    if (bodies.empty()) return nullptr;

    double x_min =  1e10, x_max = -1e10;
    double y_min =  1e10, y_max = -1e10;

    for (const auto& b : bodies) {
        x_min = min(x_min, b.x);
        x_max = max(x_max, b.x);
        y_min = min(y_min, b.y);
        y_max = max(y_max, b.y);
    }

    double margin = 0.1;
    x_min -= margin; x_max += margin;
    y_min -= margin; y_max += margin;

    auto root = make_unique<Node>(x_min, x_max, y_min, y_max);
    for (int i = 0; i < (int)bodies.size(); ++i) {
        insert_body(root.get(), i, bodies);
    }
    return root;
}

// 对“全局树”递归遍历，计算 global_index = gi 对应粒子所受的引力
void compute_force_from_tree(Node* node,
                             int gi,
                             const vector<Body>& global_bodies,
                             double& fx, double& fy)
{
    if (!node || node->mass == 0.0) return;

    const Body& bi = global_bodies[gi];

    // 叶子且只有一个粒子，且就是自己 → 不产生自引力
    if (node->children[0] == nullptr && node->body == gi) {
        return;
    }

    double dx = node->cmx - bi.x;
    double dy = node->cmy - bi.y;
    double r2 = dx*dx + dy*dy + 1e-9;
    double dist = sqrt(r2);
    double size = node->x_max - node->x_min;

    // Barnes–Hut 判定：足够远 → 用节点 multipole 近似
    if (node->children[0] == nullptr || (size / dist < THETA)) {
        double inv_r  = 1.0 / dist;
        double inv_r3 = inv_r * inv_r * inv_r;
        double s = G * bi.m * node->mass * inv_r3;
        fx += s * dx;
        fy += s * dy;
        return;
    }

    // 否则递归子节点
    for (int i = 0; i < 4; ++i) {
        if (node->children[i]) {
            compute_force_from_tree(node->children[i].get(), gi, global_bodies, fx, fy);
        }
    }
}

// 计算“引力势 φ”用的树遍历：φ_i = sum_j -G m_j / r_ij
void compute_potential_from_tree(Node* node,
                                 int gi,
                                 const vector<Body>& global_bodies,
                                 double& phi)
{
    if (!node || node->mass == 0.0) return;

    const Body& bi = global_bodies[gi];

    // 叶子且只有一个粒子，且就是自己 → 不产生自作用势
    if (node->children[0] == nullptr && node->body == gi) {
        return;
    }

    double dx = node->cmx - bi.x;
    double dy = node->cmy - bi.y;
    double r2 = dx*dx + dy*dy + 1e-9;
    double dist = sqrt(r2);
    double size = node->x_max - node->x_min;

    // Barnes–Hut 判定：足够远 → 用节点 multipole 近似势能
    if (node->children[0] == nullptr || (size / dist < THETA)) {
        phi += -G * node->mass / dist;
        return;
    }

    // 否则递归子节点
    for (int i = 0; i < 4; ++i) {
        if (node->children[i]) {
            compute_potential_from_tree(node->children[i].get(), gi, global_bodies, phi);
        }
    }
}

// 计算全局质心（用于监控）
void compute_global_center_of_mass(const vector<Body>& local_bodies,
                                   double& cmx, double& cmy, double& total_mass)
{
    double local_mass = 0.0;
    double local_mx   = 0.0;
    double local_my   = 0.0;

    for (const auto& b : local_bodies) {
        local_mass += b.m;
        local_mx   += b.m * b.x;
        local_my   += b.m * b.y;
    }

    double global_mass = 0.0;
    double global_mx   = 0.0;
    double global_my   = 0.0;

    MPI_Allreduce(&local_mass, &global_mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_mx,   &global_mx,   1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_my,   &global_my,   1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    total_mass = global_mass;
    if (global_mass > 0.0) {
        cmx = global_mx / global_mass;
        cmy = global_my / global_mass;
    } else {
        cmx = cmy = 0.0;
    }
}

// ====== 工具：把全局 N 平均切成 size 份，得到 counts / displs ======
void compute_counts_displs(int N, int size, vector<int>& counts, vector<int>& displs) {
    counts.assign(size, 0);
    displs.assign(size, 0);
    int base = N / size;
    int rem  = N % size;
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = offset;
        offset   += counts[r];
    }
}

// ====== 初始化：在 rank 0 生成所有粒子，然后 Scatterv 到各 rank ======
void init_bodies_block(int rank, int size,
                       vector<Body>& local_bodies,
                       vector<int>&  counts,
                       vector<int>&  displs)
{
    vector<Body> all_init;

    if (rank == 0) {
        all_init.resize(N_BODIES_GLOBAL);

        // Two-body test case:
        // m1 = m2 = 1
        // x1 = -1, y1 = 0
        // x2 =  1, y2 = 0
        // v1 = v2 = 0
        if (N_BODIES_GLOBAL == 2) {
            double R = 1.0;
            double v = 0.5;   // sqrt(G/(4R)) when G=1, R=1
        
            all_init[0] = Body(-R, 0.0,  0.0,  v, 1.0);   // 逆时针方向
            all_init[1] = Body( R, 0.0,  0.0, -v, 1.0);   // 顺时针方向
        } else {
            mt19937 gen(114514);
            uniform_real_distribution<double> pos_dis(-1.0, 1.0);
            uniform_real_distribution<double> vel_dis(-0.1, 0.1);
            uniform_real_distribution<double> mass_dis(0.1, 1.0);

            for (int i = 0; i < N_BODIES_GLOBAL; ++i) {
                all_init[i] = Body(
                    pos_dis(gen),
                    pos_dis(gen),
                    vel_dis(gen),
                    vel_dis(gen),
                    mass_dis(gen)
                );
            }
        }
    }

    compute_counts_displs(N_BODIES_GLOBAL, size, counts, displs);
    int local_n = counts[rank];
    local_bodies.resize(local_n);

    MPI_Scatterv(
        rank == 0 ? all_init.data() : nullptr, // sendbuf
        counts.data(),
        displs.data(),
        MPI_BODY,
        local_bodies.data(),
        local_n,
        MPI_BODY,
        0,
        MPI_COMM_WORLD
    );
}

// ====== 计算初始加速度：需要全局 tree ======
void compute_initial_accelerations(const vector<int>& counts,
                                   const vector<int>& displs,
                                   vector<Body>& local_bodies)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_n = counts[rank];

    // 1. 收集所有粒子到各个 rank（每步都 Allgatherv，保证所有人有同一份 global_bodies）
    vector<Body> global_bodies(N_BODIES_GLOBAL);
    MPI_Allgatherv(
        local_bodies.data(), local_n, MPI_BODY,
        global_bodies.data(), counts.data(), displs.data(),
        MPI_BODY, MPI_COMM_WORLD
    );

    // 2. 每个 rank 独立构建同一棵 Barnes–Hut 树
    auto root = build_tree_global(global_bodies);

    // 3. 为本 rank 负责的每个粒子计算加速度
    #pragma omp parallel for
    for (int i = 0; i < local_n; ++i) {
        double fx = 0.0, fy = 0.0;
        int gi = displs[rank] + i;  // 该粒子在全局 array 中的 index
        compute_force_from_tree(root.get(), gi, global_bodies, fx, fy);
        local_bodies[i].ax = fx / local_bodies[i].m;
        local_bodies[i].ay = fy / local_bodies[i].m;
    }
}

// ====== 时间步内：更新位置 → Allgatherv → 新树 → 新加速度并修正速度 ======
void update_acc_and_velocity(const vector<int>& counts,
                             const vector<int>& displs,
                             vector<Body>& local_bodies)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_n = counts[rank];

    // 1. Allgatherv 同步所有粒子位置（global_bodies 用来建树）
    vector<Body> global_bodies(N_BODIES_GLOBAL);
    MPI_Allgatherv(
        local_bodies.data(), local_n, MPI_BODY,
        global_bodies.data(), counts.data(), displs.data(),
        MPI_BODY, MPI_COMM_WORLD
    );

    // 2. 每个 rank 独立建同一棵树
    auto root = build_tree_global(global_bodies);

    // 3. 计算新加速度 + 修正速度（velocity Verlet）
    #pragma omp parallel for
    for (int i = 0; i < local_n; ++i) {
        double fx = 0.0, fy = 0.0;
        int gi = displs[rank] + i;
        compute_force_from_tree(root.get(), gi, global_bodies, fx, fy);

        double ax_new = fx / local_bodies[i].m;
        double ay_new = fy / local_bodies[i].m;

        local_bodies[i].vx += 0.5 * (local_bodies[i].ax + ax_new) * DT;
        local_bodies[i].vy += 0.5 * (local_bodies[i].ay + ay_new) * DT;

        local_bodies[i].ax = ax_new;
        local_bodies[i].ay = ay_new;
    }
}

// ====== 计算总能量（动能 + 势能，BH 近似） ======
double compute_total_energy(const vector<int>& counts,
                            const vector<int>& displs,
                            const vector<Body>& local_bodies)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_n = counts[rank];

    // 1) 收集所有粒子（和建树步骤类似）
    vector<Body> global_bodies(N_BODIES_GLOBAL);
    MPI_Allgatherv(
        local_bodies.data(), local_n, MPI_BODY,
        global_bodies.data(), counts.data(), displs.data(),
        MPI_BODY, MPI_COMM_WORLD
    );

    auto root = build_tree_global(global_bodies);

    // 2) 本地动能
    double local_K = 0.0;
    for (int i = 0; i < local_n; ++i) {
        double v2 = local_bodies[i].vx * local_bodies[i].vx
                  + local_bodies[i].vy * local_bodies[i].vy;
        local_K += 0.5 * local_bodies[i].m * v2;
    }

    // 3) 本地势能：U_loc_partial = 0.5 * sum_i m_i * φ_i
    double local_U = 0.0;
    for (int i = 0; i < local_n; ++i) {
        int gi = displs[rank] + i;
        double phi = 0.0;
        compute_potential_from_tree(root.get(), gi, global_bodies, phi);
        local_U += 0.5 * local_bodies[i].m * phi;
    }

    // 4) 全局归约
    double global_K = 0.0, global_U = 0.0;
    MPI_Allreduce(&local_K, &global_K, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_U, &global_U, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return global_K + global_U;
}

// ====== 每步输出前 N_TRACKED 个天体的轨迹（全局 index 0..N_TRACKED-1）到单个文件 ======
void output_tracked_bodies(const vector<int>& counts,
                           const vector<int>& displs,
                           const vector<Body>& local_bodies,
                           int iter)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_n = counts[rank];

    // 收集所有粒子到各个 rank（与建树时类似），这样 rank 0 拥有完整的 global_bodies
    vector<Body> global_bodies(N_BODIES_GLOBAL);
    MPI_Allgatherv(
        local_bodies.data(), local_n, MPI_BODY,
        global_bodies.data(), counts.data(), displs.data(),
        MPI_BODY, MPI_COMM_WORLD
    );

    if (rank == 0) {
        static bool header_written = false;

        ofstream ofs;
        if (!header_written) {
            ofs.open("tracked_bodies.dat", ios::out | ios::trunc);
            // 所有字段名用英语
            ofs << "iter time body_index x y vx vy mass\n";
            header_written = true;
        } else {
            ofs.open("tracked_bodies.dat", ios::out | ios::app);
        }

        double time = iter * DT;
        int n_tracked = std::min(N_BODIES_GLOBAL, N_TRACKED);
        for (int gi = 0; gi < n_tracked; ++gi) {
            const Body& b = global_bodies[gi];
            ofs << iter << ' ' << time << ' ' << gi << ' '
                << b.x  << ' ' << b.y  << ' '
                << b.vx << ' ' << b.vy << ' '
                << b.m  << '\n';
        }
    }
}

int main(int argc, char** argv) {
    if (argc > 1) {
        N_BODIES_GLOBAL = atoi(argv[1]);
    }

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 定义 MPI_BODY：7 个 double（假定结构无 padding）
    MPI_Type_contiguous(7, MPI_DOUBLE, &MPI_BODY);
    MPI_Type_commit(&MPI_BODY);

    int num_threads = omp_get_max_threads();
    if (rank == 0) {
        cout << "N_BODIES_GLOBAL: " << N_BODIES_GLOBAL << endl;
        cout << "MPI processes: " << size << endl;
        cout << "OpenMP threads per process: " << num_threads << endl;
    }

    // 每个 rank 拥有一段粒子（按 index 块划分）
    vector<int> counts, displs;
    vector<Body> local_bodies;
    init_bodies_block(rank, size, local_bodies, counts, displs);

    if (rank == 0) {
        int total_local = 0;
        int local = (int)local_bodies.size();
        MPI_Reduce(&local, &total_local, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        cout << "Initial local bodies on rank 0: " << local << endl;
        cout << "Total bodies summed over ranks: " << total_local << endl;
    } else {
        int local = (int)local_bodies.size();
        MPI_Reduce(&local, nullptr, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    // 计算初始加速度
    compute_initial_accelerations(counts, displs, local_bodies);

    // 时间迭代
    double t_start = MPI_Wtime();
    int local_n = counts[rank];

    for (int iter = 0; iter < N_ITERATIONS; ++iter) {
        // 1) 用当前加速度更新位置（Verlet 位置部分）
        #pragma omp parallel for
        for (int i = 0; i < local_n; ++i) {
            local_bodies[i].x += local_bodies[i].vx * DT
                               + 0.5 * local_bodies[i].ax * DT * DT;
            local_bodies[i].y += local_bodies[i].vy * DT
                               + 0.5 * local_bodies[i].ay * DT * DT;
        }

        // 2) 基于新位置建全局树，计算新加速度并更新速度
        update_acc_and_velocity(counts, displs, local_bodies);

        // 3) 每步追踪前 N_TRACKED 个天体的轨迹
        output_tracked_bodies(counts, displs, local_bodies, iter);

        // 4) 输出监控（中心质量、总质量、总能量）
        if (iter % OUTPUT_INTERVAL == 0) {
            double cmx, cmy, total_mass;
            compute_global_center_of_mass(local_bodies, cmx, cmy, total_mass);
            double E = compute_total_energy(counts, displs, local_bodies);

            if (rank == 0) {
                cout << "\n=== Iteration " << iter << " ===" << endl;
                cout << "Global Center of Mass: (" << cmx << ", " << cmy << ")\n";
                cout << "Total Mass: " << total_mass << endl;
                cout << "Total Energy (K+U): " << E << endl;
            }
        }
    }

    double t_end = MPI_Wtime();
    if (rank == 0) {
        cout << "\n=== Simulation Complete ===" << endl;
        cout << "Total time: " << (t_end - t_start) << " seconds\n";
        cout << "Time per iteration: " << (t_end - t_start) / N_ITERATIONS << " seconds\n";
    }

    MPI_Type_free(&MPI_BODY);
    MPI_Finalize();
    return 0;
}

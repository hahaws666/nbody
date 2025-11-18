#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <memory>
#include <random>
#include <algorithm>

using namespace std;

// 配置参数
constexpr double THETA = 0.5;        // Barnes-Hut 阈值
constexpr double G = 1.0;            // 万有引力常数
constexpr double DT = 0.01;          // 时间步长
constexpr int N_ITERATIONS = 100;    // 迭代次数
constexpr int OUTPUT_INTERVAL = 10;  // 输出间隔

// 从命令行参数读取
int N_BODIES = 1000;                 // 粒子数量（默认值）

struct Body {
    double x, y;
    double vx, vy;
    double m;
    double ax, ay;  // 加速度（用于 leapfrog 方法）
    
    Body() : x(0), y(0), vx(0), vy(0), m(0), ax(0), ay(0) {}
    Body(double x, double y, double vx, double vy, double m)
        : x(x), y(y), vx(vx), vy(vy), m(m), ax(0), ay(0) {}
};

struct Node {
    double x_min, x_max;
    double y_min, y_max;
    
    double mass = 0.0;
    double cmx = 0.0;
    double cmy = 0.0;
    
    int body = -1;  // 如果是 leaf 且只有一个粒子，则记录 index
    unique_ptr<Node> children[4];
    
    Node(double xmin, double xmax, double ymin, double ymax)
        : x_min(xmin), x_max(xmax), y_min(ymin), y_max(ymax) {}
};

vector<Body> bodies;

// 获取粒子所在的象限
int get_quadrant(Node* node, const Body& b) {
    double midx = (node->x_min + node->x_max) / 2;
    double midy = (node->y_min + node->y_max) / 2;
    int q = 0;
    if (b.x >= midx) q += 1;
    if (b.y >= midy) q += 2;
    return q;
}

// 获取子节点的边界
void child_bounds(Node* node, int q,
                  double& xmin, double& xmax,
                  double& ymin, double& ymax) {
    double midx = (node->x_min + node->x_max) / 2;
    double midy = (node->y_min + node->y_max) / 2;
    
    switch(q) {
        case 0: xmin=node->x_min; xmax=midx; ymin=node->y_min; ymax=midy; break; // 左下
        case 1: xmin=midx; xmax=node->x_max; ymin=node->y_min; ymax=midy; break; // 右下
        case 2: xmin=node->x_min; xmax=midx; ymin=midy; ymax=node->y_max; break; // 左上
        case 3: xmin=midx; xmax=node->x_max; ymin=midy; ymax=node->y_max; break; // 右上
    }
}

// 更新节点的质量和质心
void update_mass(Node* node, const Body& b) {
    double total = node->mass + b.m;
    if (total == 0) return;
    
    node->cmx = (node->cmx * node->mass + b.x * b.m) / total;
    node->cmy = (node->cmy * node->mass + b.y * b.m) / total;
    node->mass = total;
}

// 插入粒子到树中
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

// 收集树中所有粒子的索引
void collect_bodies(Node* node, vector<int>& indices) {
    if (!node || node->mass == 0.0) return;
    
    if (node->children[0] == nullptr && node->body != -1) {
        // 叶子节点，只有一个粒子
        indices.push_back(node->body);
    } else {
        // 有子节点，递归收集
        for (int i = 0; i < 4; i++) {
            if (node->children[i]) {
                collect_bodies(node->children[i].get(), indices);
            }
        }
    }
}

// 合并两个节点：将 src 的所有粒子重新插入到 dst
void merge_nodes(Node* dst, Node* src) {
    if (!src || src->mass == 0.0) return;
    
    // 收集 src 树中的所有粒子索引
    vector<int> body_indices;
    collect_bodies(src, body_indices);
    
    // 将所有粒子重新插入到 dst
    for (int idx : body_indices) {
        if (idx >= 0 && idx < N_BODIES) {
            insert_body(dst, idx);
        }
    }
}

// 并行建树：使用网格划分和局部子树合并
unique_ptr<Node> build_tree_parallel(double x_min, double x_max, 
                                      double y_min, double y_max) {
    // 计算网格大小（根据线程数和粒子数）
    int num_threads = omp_get_max_threads();
    int grid_size = max(2, (int)sqrt(num_threads * 4));  // 至少 2x2 网格
    
    double dx = (x_max - x_min) / grid_size;
    double dy = (y_max - y_min) / grid_size;
    
    // 为每个网格单元分配粒子索引
    vector<vector<int>> buckets(grid_size * grid_size);
    
    #pragma omp parallel
    {
        vector<vector<int>> local_buckets(grid_size * grid_size);
        
        #pragma omp for
        for (int i = 0; i < N_BODIES; i++) {
            int gx = (int)((bodies[i].x - x_min) / dx);
            int gy = (int)((bodies[i].y - y_min) / dy);
            // 确保在有效范围内
            gx = max(0, min(gx, grid_size - 1));
            gy = max(0, min(gy, grid_size - 1));
            int bucket_idx = gy * grid_size + gx;
            if (bucket_idx >= 0 && bucket_idx < grid_size * grid_size) {
                local_buckets[bucket_idx].push_back(i);
            }
        }
        
        // 合并到全局 buckets
        #pragma omp critical
        {
            for (int i = 0; i < grid_size * grid_size; i++) {
                buckets[i].insert(buckets[i].end(), 
                                  local_buckets[i].begin(), 
                                  local_buckets[i].end());
            }
        }
    }
    
    // 每个线程构建局部子树
    vector<unique_ptr<Node>> local_trees(num_threads);
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        local_trees[tid] = make_unique<Node>(x_min, x_max, y_min, y_max);
        
        // 每个线程处理一部分 buckets
        int buckets_per_thread = (grid_size * grid_size + num_threads - 1) / num_threads;
        int start_bucket = tid * buckets_per_thread;
        int end_bucket = min(start_bucket + buckets_per_thread, grid_size * grid_size);
        
        for (int b = start_bucket; b < end_bucket; b++) {
            if (buckets[b].empty()) continue;
            
            // 为这个 bucket 创建局部子树
            int gx = b % grid_size;
            int gy = b / grid_size;
            double bx_min = x_min + gx * dx;
            double bx_max = x_min + (gx + 1) * dx;
            double by_min = y_min + gy * dy;
            double by_max = y_min + (gy + 1) * dy;
            
            auto bucket_tree = make_unique<Node>(bx_min, bx_max, by_min, by_max);
            for (int idx : buckets[b]) {
                if (idx >= 0 && idx < N_BODIES) {
                    insert_body(bucket_tree.get(), idx);
                }
            }
            
            // 合并到线程的局部树
            if (bucket_tree->mass > 0) {
                merge_nodes(local_trees[tid].get(), bucket_tree.get());
            }
        }
    }
    
    // 合并所有线程的局部子树
    auto root = make_unique<Node>(x_min, x_max, y_min, y_max);
    for (int t = 0; t < num_threads; t++) {
        if (local_trees[t]) {
            merge_nodes(root.get(), local_trees[t].get());
        }
    }
    
    return root;
}

// 计算粒子受到的引力（Barnes-Hut 算法）
void compute_force(Node* node, int idx, double& fx, double& fy) {
    if (!node || node->mass == 0.0) return;
    Body& bi = bodies[idx];
    
    // 跳过自身
    if (node->body == idx && node->children[0] == nullptr) return;
    
    double dx = node->cmx - bi.x;
    double dy = node->cmy - bi.y;
    double dist = sqrt(dx*dx + dy*dy + 1e-9);
    double size = node->x_max - node->x_min;
    
    // Barnes-Hut 判断：如果节点足够远，使用质心近似
    if (node->children[0] == nullptr || (size / dist < THETA)) {
        double F = G * bi.m * node->mass / (dist*dist + 1e-9);
        fx += F * dx / dist;
        fy += F * dy / dist;
        return;
    }
    
    // 否则递归计算子节点
    for (auto& ch : node->children) {
        if (ch) compute_force(ch.get(), idx, fx, fy);
    }
}

// 初始化粒子（随机分布）
void init_bodies(int rank) {
    bodies.resize(N_BODIES);
    
    if (rank == 0) {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<double> pos_dis(-1.0, 1.0);
        uniform_real_distribution<double> vel_dis(-0.1, 0.1);
        uniform_real_distribution<double> mass_dis(0.1, 1.0);
        
        for (int i = 0; i < N_BODIES; i++) {
            bodies[i] = Body(
                pos_dis(gen),
                pos_dis(gen),
                vel_dis(gen),
                vel_dis(gen),
                mass_dis(gen)
            );
        }
    }
}

// 计算质心
void compute_center_of_mass(double& cmx, double& cmy, double& total_mass) {
    cmx = 0.0;
    cmy = 0.0;
    total_mass = 0.0;
    
    #pragma omp parallel for reduction(+:cmx,cmy,total_mass)
    for (int i = 0; i < N_BODIES; i++) {
        cmx += bodies[i].m * bodies[i].x;
        cmy += bodies[i].m * bodies[i].y;
        total_mass += bodies[i].m;
    }
    
    if (total_mass > 0) {
        cmx /= total_mass;
        cmy /= total_mass;
    }
}

int main(int argc, char** argv) {
    // 解析命令行参数
    if (argc > 1) {
        N_BODIES = atoi(argv[1]);
    }
    
    // 初始化 MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // 设置 OpenMP 线程数
    int num_threads = omp_get_max_threads();
    if (rank == 0) {
        cout << "N_BODIES: " << N_BODIES << endl;
        cout << "MPI processes: " << size << endl;
        cout << "OpenMP threads per process: " << num_threads << endl;
        cout << "Total threads: " << size * num_threads << endl;
    }
    
    // 定义 MPI 结构体（包含加速度，共 7 个 double）
    // 使用 MPI_Type_contiguous 更安全，避免 struct padding 问题
    MPI_Datatype MPI_BODY;
    MPI_Type_contiguous(7, MPI_DOUBLE, &MPI_BODY);
    MPI_Type_commit(&MPI_BODY);
    
    // 初始化粒子
    init_bodies(rank);
    MPI_Bcast(bodies.data(), N_BODIES, MPI_BODY, 0, MPI_COMM_WORLD);
    
    // 计算初始边界
    double x_min = 1e10, x_max = -1e10;
    double y_min = 1e10, y_max = -1e10;
    
    #pragma omp parallel for reduction(min:x_min,y_min) reduction(max:x_max,y_max)
    for (int i = 0; i < N_BODIES; i++) {
        x_min = min(x_min, bodies[i].x);
        x_max = max(x_max, bodies[i].x);
        y_min = min(y_min, bodies[i].y);
        y_max = max(y_max, bodies[i].y);
    }
    
    // 添加一些边距
    double margin = 0.1;
    x_min -= margin;
    x_max += margin;
    y_min -= margin;
    y_max += margin;
    
    // 计算初始加速度（leapfrog 方法需要）
    auto root_init = build_tree_parallel(x_min, x_max, y_min, y_max);
    #pragma omp parallel for
    for (int i = 0; i < N_BODIES; i++) {
        if (i % size != rank) continue;
        
        double fx = 0, fy = 0;
        compute_force(root_init.get(), i, fx, fy);
        
        bodies[i].ax = fx / bodies[i].m;
        bodies[i].ay = fy / bodies[i].m;
    }
    
    // 同步初始加速度
    vector<Body> temp_bodies_init(size * N_BODIES);
    MPI_Allgather(bodies.data(), N_BODIES, MPI_BODY,
                 temp_bodies_init.data(), N_BODIES, MPI_BODY, MPI_COMM_WORLD);
    for (int i = 0; i < N_BODIES; i++) {
        int owner_rank = i % size;
        bodies[i].ax = temp_bodies_init[owner_rank * N_BODIES + i].ax;
        bodies[i].ay = temp_bodies_init[owner_rank * N_BODIES + i].ay;
    }
    
    // 时间迭代循环
    double start_time = MPI_Wtime();
    
    for (int iter = 0; iter < N_ITERATIONS; iter++) {
        // 使用 Leapfrog 方法进行时间积分
        // Leapfrog: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
        //           v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        
        // 第一步：使用当前加速度更新位置
        #pragma omp parallel for
        for (int i = 0; i < N_BODIES; i++) {
            if (i % size != rank) continue;
            
            // 使用 leapfrog 更新位置
            bodies[i].x += bodies[i].vx * DT + 0.5 * bodies[i].ax * DT * DT;
            bodies[i].y += bodies[i].vy * DT + 0.5 * bodies[i].ay * DT * DT;
        }
        
        // 同步位置（所有进程需要知道新位置）
        vector<Body> temp_bodies_pos(size * N_BODIES);
        MPI_Allgather(bodies.data(), N_BODIES, MPI_BODY,
                     temp_bodies_pos.data(), N_BODIES, MPI_BODY, MPI_COMM_WORLD);
        for (int i = 0; i < N_BODIES; i++) {
            int owner_rank = i % size;
            bodies[i].x = temp_bodies_pos[owner_rank * N_BODIES + i].x;
            bodies[i].y = temp_bodies_pos[owner_rank * N_BODIES + i].y;
        }
        
        // 第二步：使用新位置计算边界并建树（只在位置更新后计算一次）
        x_min = 1e10; x_max = -1e10;
        y_min = 1e10; y_max = -1e10;
        
        #pragma omp parallel for reduction(min:x_min,y_min) reduction(max:x_max,y_max)
        for (int i = 0; i < N_BODIES; i++) {
            x_min = min(x_min, bodies[i].x);
            x_max = max(x_max, bodies[i].x);
            y_min = min(y_min, bodies[i].y);
            y_max = max(y_max, bodies[i].y);
        }
        
        x_min -= margin;
        x_max += margin;
        y_min -= margin;
        y_max += margin;
        
        // 使用新位置重新建树（关键：必须用新位置建树！）
        auto root = build_tree_parallel(x_min, x_max, y_min, y_max);
        
        // 第三步：基于新位置建的树计算新的加速度
        #pragma omp parallel for
        for (int i = 0; i < N_BODIES; i++) {
            if (i % size != rank) continue;
            
            double fx = 0, fy = 0;
            compute_force(root.get(), i, fx, fy);
            
            double ax_new = fx / bodies[i].m;
            double ay_new = fy / bodies[i].m;
            
            // 使用 leapfrog 更新速度：v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
            bodies[i].vx += 0.5 * (bodies[i].ax + ax_new) * DT;
            bodies[i].vy += 0.5 * (bodies[i].ay + ay_new) * DT;
            
            // 保存新加速度供下次迭代使用
            bodies[i].ax = ax_new;
            bodies[i].ay = ay_new;
        }
        
        // 同步速度和加速度（位置已经在之前同步了）
        vector<Body> temp_bodies(size * N_BODIES);
        MPI_Allgather(bodies.data(), N_BODIES, MPI_BODY,
                     temp_bodies.data(), N_BODIES, MPI_BODY, MPI_COMM_WORLD);
        
        // 从 temp_bodies 中提取每个进程更新的速度和加速度
        for (int i = 0; i < N_BODIES; i++) {
            int owner_rank = i % size;
            bodies[i].vx = temp_bodies[owner_rank * N_BODIES + i].vx;
            bodies[i].vy = temp_bodies[owner_rank * N_BODIES + i].vy;
            bodies[i].ax = temp_bodies[owner_rank * N_BODIES + i].ax;
            bodies[i].ay = temp_bodies[owner_rank * N_BODIES + i].ay;
        }
        
        // 输出（只在某些迭代时输出）
        if (iter % OUTPUT_INTERVAL == 0) {
            MPI_Barrier(MPI_COMM_WORLD);
            
            if (rank == 0) {
                double cmx, cmy, total_mass;
                compute_center_of_mass(cmx, cmy, total_mass);
                
                cout << "\n=== Iteration " << iter << " ===" << endl;
                cout << "Center of Mass: (" << cmx << ", " << cmy << ")" << endl;
                cout << "Total Mass: " << total_mass << endl;
            }
        }
    }
    
    double end_time = MPI_Wtime();
    
    if (rank == 0) {
        cout << "\n=== Simulation Complete ===" << endl;
        cout << "Total time: " << end_time - start_time << " seconds" << endl;
        cout << "Time per iteration: " << (end_time - start_time) / N_ITERATIONS << " seconds" << endl;
    }
    
    MPI_Type_free(&MPI_BODY);
    MPI_Finalize();
    
    return 0;
}


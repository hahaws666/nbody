#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <memory>
#include <random>
#include <algorithm>

using namespace std;

// ==== 配置参数 ====
constexpr double THETA = 0.1;        // Barnes-Hut 阈值
constexpr double G = 1.0;            // 万有引力常数
constexpr double DT = 0.01;          // 时间步长
constexpr int N_ITERATIONS = 10000;    // 迭代次数
constexpr int OUTPUT_INTERVAL = 10;  // 输出间隔

int N_BODIES_GLOBAL = 1000;          // 总粒子数（从命令行读取）

struct NodeMultipole {
    double mass;
    double cmx, cmy;
    double x_min, x_max;
    double y_min, y_max;
};


// 分布式全局 Barnes–Hut 树节点（所有 rank 共享同一拓扑）
struct DistributedNode {
    double mass;
    double cmx, cmy;
    double xmin, xmax, ymin, ymax;
    int parent;        // parent node index
    int child[4];      // 4 child node indices
    int level;         // tree level
    int owner_rank;    // 哪个 rank 概念上“拥有”该节点（按 x 方向划分）
};

// 全局 BH 树结构，在所有 rank 中完全一致
vector<DistributedNode> global_tree;
constexpr int MAX_BH_LEVEL = 7;  // 全局 BH 树最大深度


// ==== 数据结构 ====
struct Body {
    double x, y;
    double vx, vy;
    double m;
    double ax, ay;

    Body() : x(0), y(0), vx(0), vy(0), m(0), ax(0), ay(0) {}
    Body(double x_, double y_, double vx_, double vy_, double m_)
        : x(x_), y(y_), vx(vx_), vy(vy_), m(m_), ax(0), ay(0) {}
};

// 简化版四叉树节点（只对本 rank 的粒子建树）
struct Node {
    double x_min, x_max;
    double y_min, y_max;

    double mass = 0.0;
    double cmx = 0.0;
    double cmy = 0.0;

    int body = -1;  // 如果是叶子且只有一个粒子，存 index
    unique_ptr<Node> children[4];

    Node(double xmin, double xmax, double ymin, double ymax)
        : x_min(xmin), x_max(xmax), y_min(ymin), y_max(ymax) {}
};

// 每个 rank 对外暴露的 multipole（远场近似用）
struct Multipole {
    double mass;
    double cmx, cmy;
    double x_min, x_max;
    double y_min, y_max;
};

// ==== 全局变量：每个 rank 只存自己的粒子 ====
vector<Body> bodies;   // local bodies

// ==== MPI 类型 ====
MPI_Datatype MPI_BODY;
MPI_Datatype MPI_MULTIPOLE;
MPI_Datatype MPI_NODE_MP;

// ==== 工具函数 ====
int get_quadrant(Node* node, const Body& b) {
    double midx = 0.5 * (node->x_min + node->x_max);
    double midy = 0.5 * (node->y_min + node->y_max);
    int q = 0;
    if (b.x >= midx) q += 1;
    if (b.y >= midy) q += 2;
    return q;
}

void flatten_tree(Node* node, vector<NodeMultipole>& out) {
    if (!node || node->mass == 0.0) return;

    NodeMultipole nm;
    nm.mass = node->mass;
    nm.cmx = node->cmx;
    nm.cmy = node->cmy;
    nm.x_min = node->x_min;
    nm.x_max = node->x_max;
    nm.y_min = node->y_min;
    nm.y_max = node->y_max;

    out.push_back(nm);

    for (int i = 0; i < 4; i++) {
        if (node->children[i])
            flatten_tree(node->children[i].get(), out);
    }
}


// ============ 全局 Distributed BH 树构建 ============ //

inline bool is_leaf(const DistributedNode& nd) {
    return nd.child[0] == -1 && nd.child[1] == -1 &&
           nd.child[2] == -1 && nd.child[3] == -1;
}

void subdivide_node(int node_id, int max_level) {
    DistributedNode& nd = global_tree[node_id];
    if (nd.level >= max_level) return;

    double midx = 0.5 * (nd.xmin + nd.xmax);
    double midy = 0.5 * (nd.ymin + nd.ymax);

    for (int q = 0; q < 4; q++) {
        DistributedNode child;
        child.level = nd.level + 1;
        child.parent = node_id;
        child.owner_rank = -1;

        switch (q) {
            case 0: // 左下
                child.xmin = nd.xmin; child.xmax = midx;
                child.ymin = nd.ymin; child.ymax = midy;
                break;
            case 1: // 右下
                child.xmin = midx;    child.xmax = nd.xmax;
                child.ymin = nd.ymin; child.ymax = midy;
                break;
            case 2: // 左上
                child.xmin = nd.xmin; child.xmax = midx;
                child.ymin = midy;    child.ymax = nd.ymax;
                break;
            case 3: // 右上
                child.xmin = midx;    child.xmax = nd.xmax;
                child.ymin = midy;    child.ymax = nd.ymax;
                break;
        }

        child.mass = 0.0;
        child.cmx = child.cmy = 0.0;
        for (int k = 0; k < 4; ++k) child.child[k] = -1;

        int new_id = (int)global_tree.size();
        global_tree.push_back(child);
        nd.child[q] = new_id;

        subdivide_node(new_id, max_level);
    }
}

// 构建所有 rank 共享的全局 BH 树拓扑（仅几何 & 拓扑，不含 multipole）
void build_global_tree(double global_xmin, double global_xmax,
                       double global_ymin, double global_ymax,
                       int max_level) {
    global_tree.clear();
    global_tree.reserve(1 << (2 * max_level)); // 粗略上界

    DistributedNode root;
    root.xmin = global_xmin;
    root.xmax = global_xmax;
    root.ymin = global_ymin;
    root.ymax = global_ymax;
    root.level = 0;
    root.parent = -1;
    root.owner_rank = -1;
    root.mass = 0.0;
    root.cmx = root.cmy = 0.0;
    for (int i = 0; i < 4; ++i) root.child[i] = -1;

    global_tree.push_back(root);
    subdivide_node(0, max_level);
}

// 为每个 DistributedNode 按 x 方向 domain decomposition 分配 owner_rank
void assign_node_ownership(int rank, int size,
                           double global_xmin, double global_xmax) {
    double dx = (global_xmax - global_xmin) / size;
    double dom_min = global_xmin + rank * dx;
    double dom_max = dom_min + dx;

    for (auto& nd : global_tree) {
        if (nd.xmin >= dom_min && nd.xmax <= dom_max) {
            nd.owner_rank = rank;
        }
    }
}

// 根据粒子位置在全局树中找到叶子节点 id
int find_leaf_for_body(const Body& b) {
    int node_id = 0;
    while (true) {
        DistributedNode& nd = global_tree[node_id];
        if (is_leaf(nd)) break;

        double midx = 0.5 * (nd.xmin + nd.xmax);
        double midy = 0.5 * (nd.ymin + nd.ymax);
        int q = 0;
        if (b.x >= midx) q += 1;
        if (b.y >= midy) q += 2;
        int child_id = nd.child[q];
        if (child_id < 0) break; // 防御性处理
        node_id = child_id;
    }
    return node_id;
}

// 基于本地 bodies，构建 DistributedNode multipole，并用 Allreduce 合并为全局 multipole
void compute_global_multipoles(int rank, int size) {
    int N = (int)global_tree.size();
    static vector<double> mass, mx, my;
    mass.assign(N, 0.0);
    mx.assign(N, 0.0);
    my.assign(N, 0.0);

    // 1) 每个 rank 在自身持有的粒子上累加贡献到叶子
    for (const auto& b : bodies) {
        int leaf_id = find_leaf_for_body(b);
        mass[leaf_id] += b.m;
        mx[leaf_id]   += b.m * b.x;
        my[leaf_id]   += b.m * b.y;
    }

    // 2) 自底向上聚合到父节点（本地）
    for (int i = N - 1; i >= 0; --i) {
        int p = global_tree[i].parent;
        if (p >= 0) {
            mass[p] += mass[i];
            mx[p]   += mx[i];
            my[p]   += my[i];
        }
    }

    // 3) Allreduce 合并所有 rank 的贡献
    MPI_Allreduce(MPI_IN_PLACE, mass.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, mx.data(),   N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, my.data(),   N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // 4) 回写到 global_tree 中
    for (int i = 0; i < N; ++i) {
        global_tree[i].mass = mass[i];
        if (mass[i] > 0.0) {
            global_tree[i].cmx = mx[i] / mass[i];
            global_tree[i].cmy = my[i] / mass[i];
        } else {
            global_tree[i].cmx = global_tree[i].cmy = 0.0;
        }
    }
}

// 使用全局 BH 树对单个粒子做 Barnes–Hut 遍历
void bh_traverse(int node_id, int bi, double& fx, double& fy) {
    const DistributedNode& nd = global_tree[node_id];
    if (nd.mass == 0.0) return;

    const Body& b = bodies[bi];
    double dx = nd.cmx - b.x;
    double dy = nd.cmy - b.y;
    double dist2 = dx * dx + dy * dy + 1e-12;
    double dist = sqrt(dist2);
    double size = nd.xmax - nd.xmin;

    // Barnes–Hut 判据，或已到叶子
    if (is_leaf(nd) || (size / dist < THETA)) {
        double inv_r3 = 1.0 / (dist2 * dist); // 1 / r^3
        double s = G * b.m * nd.mass * inv_r3;
        fx += s * dx;
        fy += s * dy;
        return;
    }

    // 否则递归遍历子节点
    for (int q = 0; q < 4; ++q) {
        int cid = nd.child[q];
        if (cid != -1) {
            bh_traverse(cid, bi, fx, fy);
        }
    }
}


void child_bounds(Node* node, int q,
                  double& xmin, double& xmax,
                  double& ymin, double& ymax) {
    double midx = 0.5 * (node->x_min + node->x_max);
    double midy = 0.5 * (node->y_min + node->y_max);
    switch (q) {
        case 0: xmin = node->x_min; xmax = midx;  ymin = node->y_min; ymax = midy;  break; // 左下
        case 1: xmin = midx;        xmax = node->x_max; ymin = node->y_min; ymax = midy;  break; // 右下
        case 2: xmin = node->x_min; xmax = midx;  ymin = midy;        ymax = node->y_max; break; // 左上
        case 3: xmin = midx;        xmax = node->x_max; ymin = midy;        ymax = node->y_max; break; // 右上
    }
}

void update_mass(Node* node, const Body& b) {
    double total = node->mass + b.m;
    if (total == 0.0) return;
    node->cmx = (node->cmx * node->mass + b.x * b.m) / total;
    node->cmy = (node->cmy * node->mass + b.y * b.m) / total;
    node->mass = total;
}

void insert_body(Node* node, int idx) {
    Body& b = bodies[idx];

    // 空节点
    if (node->mass == 0.0 && node->children[0] == nullptr) {
        node->body = idx;
        update_mass(node, b);
        return;
    }

    // 叶子且已有一个粒子 → 分裂
    if (node->children[0] == nullptr && node->body != -1) {
        int old = node->body;
        node->body = -1;

        for (int i = 0; i < 4; ++i) {
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

// 串行建树：对本 rank 的 bodies 建立 Barnes–Hut 树
unique_ptr<Node> build_tree_local() {
    if (bodies.empty()) return nullptr;

    double x_min = 1e10, x_max = -1e10;
    double y_min = 1e10, y_max = -1e10;

    for (const auto& b : bodies) {
        x_min = min(x_min, b.x);
        x_max = max(x_max, b.x);
        y_min = min(y_min, b.y);
        y_max = max(y_max, b.y);
    }

    // 稍微加点边距
    double margin = 0.1;
    x_min -= margin; x_max += margin;
    y_min -= margin; y_max += margin;

    auto root = make_unique<Node>(x_min, x_max, y_min, y_max);
    for (int i = 0; i < (int)bodies.size(); ++i) {
        insert_body(root.get(), i);
    }
    return root;
}

// 计算粒子受到的引力（来自本地树）
void compute_force_local(Node* node, int idx, double& fx, double& fy) {
    if (!node || node->mass == 0.0) return;
    Body& bi = bodies[idx];

    // 跳过自身
    if (node->body == idx && node->children[0] == nullptr) return;

    double dx = node->cmx - bi.x;
    double dy = node->cmy - bi.y;
    double r2 = dx * dx + dy * dy + 1e-9;
    double dist = sqrt(r2);
    double size = node->x_max - node->x_min;

    // Barnes-Hut 判定
    if (node->children[0] == nullptr || (size / dist < THETA)) {
        double inv_r = 1.0 / dist;
        double inv_r3 = inv_r * inv_r * inv_r;
        double s = G * bi.m * node->mass * inv_r3;  // = G*m1*m2/r^3
        fx += s * dx;
        fy += s * dy;
        return;
    }

    // 否则递归子节点
    for (auto& ch : node->children) {
        if (ch) compute_force_local(ch.get(), idx, fx, fy);
    }
}

// 计算来自其他 rank 的远场 multipole 的引力（把每个 rank 当成一个大质点）
void compute_force_remote(const vector<NodeMultipole>& all_nodes,
    int my_rank,
    int idx,
    double& fx, double& fy)
{
    Body& bi = bodies[idx];

    for (const auto& mp : all_nodes) {
        // 跳过空节点
        if (mp.mass == 0.0) continue;
        

        double dx = mp.cmx - bi.x;
        double dy = mp.cmy - bi.y;
        double r2 = dx*dx + dy*dy + 1e-9;
        double dist = sqrt(r2);

        double size = max(mp.x_max - mp.x_min, mp.y_max - mp.y_min);
        // Barnes-Hut 判定：如果 size/dist < THETA，使用多极近似
        if (size / dist < THETA) {
            double inv_r3 = 1.0 / (dist * dist * dist);
            double s = G * bi.m * mp.mass * inv_r3;
            fx += s * dx;
            fy += s * dy;
        }
        // NOTE: 不递归，因为树已经被 flatten
    }
}


// 计算本 rank 的 multipole（简单用所有 local bodies 的质心）
Multipole compute_local_multipole() {
    Multipole mp{};
    if (bodies.empty()) {
        mp.mass = 0.0;
        return mp;
    }

    double mass = 0.0;
    double cmx = 0.0, cmy = 0.0;
    double x_min = 1e10, x_max = -1e10;
    double y_min = 1e10, y_max = -1e10;

    for (const auto& b : bodies) {
        mass += b.m;
        cmx += b.m * b.x;
        cmy += b.m * b.y;
        x_min = min(x_min, b.x);
        x_max = max(x_max, b.x);
        y_min = min(y_min, b.y);
        y_max = max(y_max, b.y);
    }
    if (mass > 0.0) {
        cmx /= mass;
        cmy /= mass;
    }

    mp.mass = mass;
    mp.cmx = cmx;
    mp.cmy = cmy;
    mp.x_min = x_min;
    mp.x_max = x_max;
    mp.y_min = y_min;
    mp.y_max = y_max;
    return mp;
}

// 计算全局质心（用于输出），通过 MPI_Allreduce 合并
void compute_global_center_of_mass(int rank, int size, double& cmx, double& cmy, double& total_mass) {
    double local_mass = 0.0;
    double local_mx = 0.0;
    double local_my = 0.0;

    for (const auto& b : bodies) {
        local_mass += b.m;
        local_mx += b.m * b.x;
        local_my += b.m * b.y;
    }

    double global_mass = 0.0;
    double global_mx = 0.0;
    double global_my = 0.0;

    MPI_Allreduce(&local_mass, &global_mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_mx, &global_mx, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_my, &global_my, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    total_mass = global_mass;
    if (global_mass > 0.0) {
        cmx = global_mx / global_mass;
        cmy = global_my / global_mass;
    } else {
        cmx = cmy = 0.0;
    }
}

// 简单的 1D domain decomposition：按 x 方向把 [-1,1] 划分为 size 份
struct Domain {
    double x_min, x_max;
};

Domain get_domain_for_rank(int rank, int size, double global_xmin, double global_xmax) {
    double dx = (global_xmax - global_xmin) / size;
    Domain d;
    d.x_min = global_xmin + rank * dx;
    d.x_max = d.x_min + dx;
    return d;
}

bool in_domain(const Body& b, const Domain& d) {
    return (b.x >= d.x_min && b.x < d.x_max);
}


// compute the total energy of the system,for debugging
// double compute_total_energy() {
//     double local_E = 0.0;

//     // kinetic energy 0.5 * m * v^2
//     for (auto &b : bodies) {
//         double v2 = b.vx * b.vx + b.vy * b.vy;
//         local_E += 0.5 * b.m * v2;
//     }

//     // potential energy: -G * m1*m2 / r
//     // only compute local-local & local-remote
//     // NOTE: we must avoid double-counting, so only count potential
//     // for pairs where i belongs to this rank, but j>i globally
//     // Simplest: compute potential only for local bodies interacting
//     // with ALL multipoles (except itself).
//     // This is approximate but acceptable for checking drift trend.

//     // You already have multipole info here:
//     // But we only call this after computing all_mps in update_acc_and_velocity
//     // So we store last all_mps globally.
//     extern vector<Multipole> last_all_mps;  // you add this global variable

//     for (auto &bi : bodies) {
//         for (int r = 0; r < last_all_mps.size(); r++) {
//             const auto &mp = last_all_mps[r];
//             if (mp.mass == 0) continue;

//             double dx = mp.cmx - bi.x;
//             double dy = mp.cmy - bi.y;
//             double dist = sqrt(dx*dx + dy*dy + 1e-9);

//             local_E -= G * bi.m * mp.mass / dist;
//         }
//     }

//     double global_E = 0.0;
//     MPI_Allreduce(&local_E, &global_E, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

//     return global_E;
// }


// 初始粒子：在 rank 0 上生成，然后广播给所有 rank，再依据 domain 选择本地粒子
void init_bodies(int rank, int size, const Domain& my_dom) {
    vector<Body> all_bodies;
    if (rank == 0) {
        all_bodies.resize(N_BODIES_GLOBAL);
        mt19937 gen(114514);
        uniform_real_distribution<double> pos_dis(-1.0, 1.0);
        uniform_real_distribution<double> vel_dis(-0.1, 0.1);
        uniform_real_distribution<double> mass_dis(0.1, 1.0);
        for (int i = 0; i < N_BODIES_GLOBAL; ++i) {
            all_bodies[i] = Body(
                pos_dis(gen),
                pos_dis(gen),
                vel_dis(gen),
                vel_dis(gen),
                mass_dis(gen)
            );
        }
    }

    // 先广播所有粒子
    if (rank != 0) {
        all_bodies.resize(N_BODIES_GLOBAL);
    }
    MPI_Bcast(all_bodies.data(), N_BODIES_GLOBAL, MPI_BODY, 0, MPI_COMM_WORLD);

    // 每个 rank 拿到属于自己 domain 的粒子
    bodies.clear();
    bodies.reserve(N_BODIES_GLOBAL / size + 10);
    for (int i = 0; i < N_BODIES_GLOBAL; ++i) {
        if (in_domain(all_bodies[i], my_dom)) {
            bodies.push_back(all_bodies[i]);
        }
    }
}

// 粒子跨域迁移（只允许每步跨相邻 domain，假设 DT 不大）
// 简化版本：只和左右邻居交换粒子
// 粒子跨域迁移（只允许每步跨相邻 domain，假设 DT 不大）
// 简化版本：只和左右邻居交换粒子
void migrate_bodies(int rank, int size, const Domain& my_dom) {
    vector<Body> stay;
    vector<Body> send_left;
    vector<Body> send_right;

    stay.reserve(bodies.size());
    send_left.reserve(bodies.size() / 10 + 1);
    send_right.reserve(bodies.size() / 10 + 1);

    // 按 x 位置分类要发送和保留的粒子
    for (const auto& b : bodies) {
        if (b.x < my_dom.x_min && rank > 0) {
            // 往左边 domain 迁移
            send_left.push_back(b);
        } else if (b.x >= my_dom.x_max && rank < size - 1) {
            // 往右边 domain 迁移
            send_right.push_back(b);
        } else {
            // 仍然留在本地 domain
            stay.push_back(b);
        }
    }

    // 替换成本地留下的
    bodies.swap(stay);

    vector<Body> recv_from_left;   // 从左邻居收到的粒子（它们是从右往左移动）
    vector<Body> recv_from_right;  // 从右邻居收到的粒子（它们是从左往右移动）

    // ==========================
    // 1) 处理“往右移动”的粒子：rank -> rank+1
    // ==========================
    {
        const int TAG_CNT_R = 100;
        const int TAG_DAT_R = 101;

        int send_count = (int)send_right.size();
        int recv_count = 0;

        // 先发送数量到右邻居
        if (rank < size - 1) {
            MPI_Send(&send_count, 1, MPI_INT, rank + 1, TAG_CNT_R, MPI_COMM_WORLD);
        }
        // 再从左邻居接收数量
        if (rank > 0) {
            MPI_Recv(&recv_count, 1, MPI_INT, rank - 1, TAG_CNT_R,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // 接收具体粒子数据（从左邻居）
        if (rank > 0 && recv_count > 0) {
            recv_from_left.resize(recv_count);
            MPI_Recv(recv_from_left.data(), recv_count, MPI_BODY,
                     rank - 1, TAG_DAT_R, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // 发送具体粒子数据（到右邻居）
        if (rank < size - 1 && send_count > 0) {
            MPI_Send(send_right.data(), send_count, MPI_BODY,
                     rank + 1, TAG_DAT_R, MPI_COMM_WORLD);
        }
    }

    // ==========================
    // 2) 处理“往左移动”的粒子：rank -> rank-1
    // ==========================
    {
        const int TAG_CNT_L = 200;
        const int TAG_DAT_L = 201;

        int send_count = (int)send_left.size();
        int recv_count = 0;

        // 先发送数量到左邻居
        if (rank > 0) {
            MPI_Send(&send_count, 1, MPI_INT, rank - 1, TAG_CNT_L, MPI_COMM_WORLD);
        }
        // 再从右邻居接收数量
        if (rank < size - 1) {
            MPI_Recv(&recv_count, 1, MPI_INT, rank + 1, TAG_CNT_L,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // 接收具体粒子数据（从右邻居）
        if (rank < size - 1 && recv_count > 0) {
            recv_from_right.resize(recv_count);
            MPI_Recv(recv_from_right.data(), recv_count, MPI_BODY,
                     rank + 1, TAG_DAT_L, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // 发送具体粒子数据（到左邻居）
        if (rank > 0 && send_count > 0) {
            MPI_Send(send_left.data(), send_count, MPI_BODY,
                     rank - 1, TAG_DAT_L, MPI_COMM_WORLD);
        }
    }

    // 把收到的粒子并入本地
    bodies.insert(bodies.end(), recv_from_left.begin(),  recv_from_left.end());
    bodies.insert(bodies.end(), recv_from_right.begin(), recv_from_right.end());
}


// 计算初始加速度（只更新 ax, ay，不更新速度）
void compute_initial_accelerations(int rank, int size) {
    // 基于全局 Distributed BH 树计算初始加速度
    compute_global_multipoles(rank, size);

    #pragma omp parallel for
    for (int i = 0; i < (int)bodies.size(); ++i) {
        double fx = 0.0, fy = 0.0;
        // 直接在全局树上做 Barnes–Hut 遍历（textbook BH）
        bh_traverse(0, i, fx, fy);
        bodies[i].ax = fx / bodies[i].m;
        bodies[i].ay = fy / bodies[i].m;
    }
}

// 在 time step 内部：用新位置建树、计算新加速度并更新速度（Leapfrog）
void update_acc_and_velocity(int rank, int size) {
    // 基于全局 Distributed BH 树重新计算 multipole 并更新速度
    compute_global_multipoles(rank, size);

    #pragma omp parallel for
    for (int i = 0; i < (int)bodies.size(); ++i) {
        double fx = 0.0, fy = 0.0;
        // 直接在全局树上做 Barnes–Hut 遍历
        bh_traverse(0, i, fx, fy);

        double ax_new = fx / bodies[i].m;
        double ay_new = fy / bodies[i].m;

        bodies[i].vx += 0.5 * (bodies[i].ax + ax_new) * DT;
        bodies[i].vy += 0.5 * (bodies[i].ay + ay_new) * DT;

        bodies[i].ax = ax_new;
        bodies[i].ay = ay_new;
    }
}

int main(int argc, char** argv) {
    // 解析命令行参数
    if (argc > 1) {
        N_BODIES_GLOBAL = atoi(argv[1]);
    }

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 定义 MPI_BODY：7 个 double
    MPI_Type_contiguous(7, MPI_DOUBLE, &MPI_BODY);
    MPI_Type_commit(&MPI_BODY);

    // 定义 MPI_MULTIPOLE：7 个 double
    MPI_Type_contiguous(7, MPI_DOUBLE, &MPI_MULTIPOLE);
    MPI_Type_commit(&MPI_MULTIPOLE);
    
    // 定义 MPI_NODE_MP：7 个 double (NodeMultipole)
    MPI_Type_contiguous(7, MPI_DOUBLE, &MPI_NODE_MP);
    MPI_Type_commit(&MPI_NODE_MP);

    int num_threads = omp_get_max_threads();
    if (rank == 0) {
        cout << "N_BODIES_GLOBAL: " << N_BODIES_GLOBAL << endl;
        cout << "MPI processes: " << size << endl;
        cout << "OpenMP threads per process: " << num_threads << endl;
    }

    // 全局 x 范围（和初始化保持一致）
    double global_xmin = -1.0;
    double global_xmax =  1.0;

    // 构建所有 rank 共享的全局 BH 树拓扑，并分配节点 ownership
    double global_ymin = -1.0;
    double global_ymax =  1.0;
    build_global_tree(global_xmin, global_xmax, global_ymin, global_ymax, MAX_BH_LEVEL);
    assign_node_ownership(rank, size, global_xmin, global_xmax);

    Domain my_dom = get_domain_for_rank(rank, size, global_xmin, global_xmax);

    // 初始化粒子并做 domain decomposition
    init_bodies(rank, size, my_dom);

    if (rank == 0) {
        int total_local = 0;
        int local = (int)bodies.size();
        MPI_Reduce(&local, &total_local, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        cout << "Initial local bodies on rank 0: " << local << endl;
        cout << "Total bodies summed over ranks: " << total_local << endl;
    } else {
        int local = (int)bodies.size();
        MPI_Reduce(&local, nullptr, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    // 计算初始加速度
    // printf("from rank %d, computing initial accelerations\n", rank);
    compute_initial_accelerations(rank, size);
    // printf("from rank %d, computed initial accelerations\n", rank);

    // 时间迭代
    double t_start = MPI_Wtime();
    for (int iter = 0; iter < N_ITERATIONS; ++iter) {
        // printf("from rank %d, starting iteration %d\n", rank, iter);
        // 1) 用当前加速度更新位置（Leapfrog 位置部分）
        #pragma omp parallel for
        for (int i = 0; i < (int)bodies.size(); ++i) {
            bodies[i].x += bodies[i].vx * DT + 0.5 * bodies[i].ax * DT * DT;
            bodies[i].y += bodies[i].vy * DT + 0.5 * bodies[i].ay * DT * DT;
        }
        // printf("from rank %d, updated positions\n", rank);
        // 2) 粒子跨域迁移
        // printf("from rank %d, migrating bodies\n", rank);
        migrate_bodies(rank, size, my_dom);
        // printf("from rank %d, migrated bodies\n", rank);
        // 3) 基于新位置建树 + 远场 multipole，计算新加速度并更新速度
        update_acc_and_velocity(rank, size);

        // 4) 输出
        if (iter % OUTPUT_INTERVAL == 0) {
            double cmx, cmy, total_mass;
            compute_global_center_of_mass(rank, size, cmx, cmy, total_mass);
            if (rank == 0) {
                cout << "\n=== Iteration " << iter << " ===" << endl;
                cout << "Global Center of Mass: (" << cmx << ", " << cmy << ")\n";
                cout << "Total Mass: " << total_mass << endl;
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
    MPI_Type_free(&MPI_MULTIPOLE);
    MPI_Type_free(&MPI_NODE_MP);
    MPI_Finalize();
    return 0;
}

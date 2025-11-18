#include <bits/stdc++.h>
#include <mpi.h>
#include <omp.h>
using namespace std;

using Complex = complex<double>;

struct Body {
    Complex z;  // 位置
    double q;   // 电荷
    double phi; // 势
    
    // MPI 传输用的辅助结构
    double z_real() const { return z.real(); }
    double z_imag() const { return z.imag(); }
    void set_z(double re, double im) { z = Complex(re, im); }
};

// MPI 传输用的结构体（只包含基础类型）
struct BodyMPI {
    double z_real, z_imag;
    double q;
    double phi;
};

// ------------------------
// Barnes–Hut 四叉树节点
// ------------------------
struct Node {
    // 正方形区域中心和半边长
    double cx, cy;
    double half;

    // 这个节点里的总电荷 (节点近似成一个点电荷)
    double mass;
    // “质心”（电荷加权中心）
    double com_x, com_y;

    int firstChild;          // 第一个子节点索引（4 个连续），-1 表示叶子
    vector<int> bodies;      // 叶子里包含的粒子索引

    Node() {
        cx = cy = half = 0.0;
        mass = 0.0;
        com_x = com_y = 0.0;
        firstChild = -1;
    }
};

struct QuadTree {
    vector<Node> nodes;
    vector<Body> &bodies;
    int maxLeafSize;

    QuadTree(vector<Body> &b, int maxLeaf = 8)
        : bodies(b), maxLeafSize(maxLeaf) {
        nodes.reserve(4 * (int)bodies.size());
    }

    void build() {
        if (bodies.empty()) return;

        double minx = 1e30, miny = 1e30;
        double maxx = -1e30, maxy = -1e30;
        for (auto &b : bodies) {
            minx = min(minx, b.z.real());
            miny = min(miny, b.z.imag());
            maxx = max(maxx, b.z.real());
            maxy = max(maxy, b.z.imag());
        }
        double cx = 0.5 * (minx + maxx);
        double cy = 0.5 * (miny + maxy);
        double half = 0.5 * max(maxx - minx, maxy - miny);
        if (half == 0.0) half = 1.0;

        Node root;
        root.cx = cx;
        root.cy = cy;
        root.half = half;
        root.firstChild = -1;
        root.bodies.reserve(bodies.size());
        for (int i = 0; i < (int)bodies.size(); ++i)
            root.bodies.push_back(i);

        nodes.clear();
        nodes.push_back(root);

        buildNode(0);
        computeMass(0);
    }

    void buildNode(int idx) {
        Node &node = nodes[idx];
        if ((int)node.bodies.size() <= maxLeafSize) {
            // 叶子
            return;
        }

        node.firstChild = (int)nodes.size();
        double h2 = node.half * 0.5;

        // 建立 4 个子节点
        for (int k = 0; k < 4; ++k) {
            Node child;
            child.half = h2;
            child.mass = 0.0;
            child.com_x = child.com_y = 0.0;
            child.firstChild = -1;
            double dx = (k & 1) ? 0.5 : -0.5;
            double dy = (k & 2) ? 0.5 : -0.5;
            child.cx = node.cx + dx * node.half;
            child.cy = node.cy + dy * node.half;
            nodes.push_back(child);
        }

        // 将粒子分配到子节点
        for (int bi : node.bodies) {
            Body &b = bodies[bi];
            int q = 0;
            if (b.z.real() > node.cx) q |= 1;
            if (b.z.imag() > node.cy) q |= 2;
            int cidx = node.firstChild + q;
            nodes[cidx].bodies.push_back(bi);
        }
        node.bodies.clear();

        // 递归构建子树
        for (int k = 0; k < 4; ++k) {
            int cidx = node.firstChild + k;
            if (!nodes[cidx].bodies.empty()) {
                buildNode(cidx);
            }
        }
    }

    void computeMass(int idx) {
        Node &node = nodes[idx];
        if (node.firstChild == -1) {
            // 叶子：直接统计电荷和质心
            node.mass = 0.0;
            node.com_x = node.com_y = 0.0;
            for (int bi : node.bodies) {
                Body &b = bodies[bi];
                node.mass += b.q;
                node.com_x += b.q * b.z.real();
                node.com_y += b.q * b.z.imag();
            }
            if (node.mass > 0.0) {
                node.com_x /= node.mass;
                node.com_y /= node.mass;
            } else {
                node.com_x = node.cx;
                node.com_y = node.cy;
            }
        } else {
            node.mass = 0.0;
            node.com_x = node.com_y = 0.0;
            for (int k = 0; k < 4; ++k) {
                int cidx = node.firstChild + k;
                computeMass(cidx);
                Node &ch = nodes[cidx];
                node.mass += ch.mass;
                node.com_x += ch.mass * ch.com_x;
                node.com_y += ch.mass * ch.com_y;
            }
            if (node.mass > 0.0) {
                node.com_x /= node.mass;
                node.com_y /= node.mass;
            } else {
                node.com_x = node.cx;
                node.com_y = node.cy;
            }
        }
    }
};

// ----------------------------------------
// Barnes–Hut 近似（当作 FMM 使用）
// ----------------------------------------

double bhPotentialOnBody(const QuadTree &tree, int nodeIdx, int bi,
                         double theta)
{
    const Node &node = tree.nodes[nodeIdx];
    if (node.mass == 0.0) return 0.0;

    const Body &b = tree.bodies[bi];
    double bx = b.z.real();
    double by = b.z.imag();

    double dx = node.com_x - bx;
    double dy = node.com_y - by;
    double dist = std::sqrt(dx*dx + dy*dy);

    // 叶子：对节点里所有粒子做直接相互作用
    if (node.firstChild == -1) {
        double phi = 0.0;
        for (int bj : node.bodies) {
            if (bj == bi) continue;
            const Body &sb = tree.bodies[bj];
            double dx2 = sb.z.real() - bx;
            double dy2 = sb.z.imag() - by;
            double r = std::sqrt(dx2*dx2 + dy2*dy2);
            if (r > 0.0) {
                phi += -sb.q * std::log(r);
            }
        }
        return phi;
    }

    double s = node.half * 2.0; // 节点边长

    // 如果 target 在该节点“中心”上，或者不满足 s/dist < theta，就继续细分
    if (dist == 0.0 || s / dist >= theta) {
        double phi = 0.0;
        for (int k = 0; k < 4; ++k) {
            int cidx = node.firstChild + k;
            phi += bhPotentialOnBody(tree, cidx, bi, theta);
        }
        return phi;
    } else {
        // 远场：把整个节点当成一个点电荷
        return -node.mass * std::log(dist);
    }
}

void computeFMM(vector<Body> &bodies,
                double theta = 0.3,
                int maxLeaf = 8,
                int rank = 0,
                int size = 1)
{
    if (bodies.empty()) return;

    QuadTree tree(bodies, maxLeaf);
    tree.build();

    int N = (int)bodies.size();
    // MPI 并行化：每个进程计算分配给它的粒子
    vector<double> local_phi(N, 0.0);
    #pragma omp parallel for
    for (int i = rank; i < N; i += size) {
        local_phi[i] = bhPotentialOnBody(tree, 0, i, theta);
    }
    
    // 收集所有进程的势值：使用 Allreduce 来合并（每个粒子只被一个进程计算）
    vector<double> global_phi(N);
    MPI_Allreduce(local_phi.data(), global_phi.data(), N, MPI_DOUBLE, 
                  MPI_SUM, MPI_COMM_WORLD);
    
    // 将结果写回 bodies
    for (int i = 0; i < N; ++i) {
        bodies[i].phi = global_phi[i];
    }
}

// -----------------------
// main：FMM 计算
// -----------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int N = 500000;      // 粒子数量（默认值）
    int iterations = 100; // 迭代次数（默认值）
    double theta = 0.2; // BH 开角参数，越小越精确但越慢

    // 解析命令行参数：第一个参数是粒子数量，第二个参数是迭代次数
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    if (argc > 2) {
        iterations = atoi(argv[2]);
    }

    // 定义 MPI 数据类型用于 BodyMPI
    MPI_Datatype MPI_BODY_MPI;
    MPI_Type_contiguous(4, MPI_DOUBLE, &MPI_BODY_MPI);
    MPI_Type_commit(&MPI_BODY_MPI);

    vector<Body> bodies(N);
    vector<BodyMPI> bodies_mpi(N);
    mt19937_64 rng(123 + rank); // 每个进程使用不同的随机种子
    uniform_real_distribution<double> U(-1.0, 1.0);

    // 只在 rank 0 生成初始数据，然后广播到所有进程
    if (rank == 0) {
        rng.seed(123); // 确保可重现性
        for (int i = 0; i < N; ++i) {
            Complex z = Complex(U(rng), U(rng));
            bodies_mpi[i].z_real = z.real();
            bodies_mpi[i].z_imag = z.imag();
            bodies_mpi[i].q = 1.0;
            bodies_mpi[i].phi = 0.0;
        }
    }
    
    // 广播粒子数据到所有进程
    MPI_Bcast(bodies_mpi.data(), N, MPI_BODY_MPI, 0, MPI_COMM_WORLD);
    
    // 转换为 Body 结构
    for (int i = 0; i < N; ++i) {
        bodies[i].z = Complex(bodies_mpi[i].z_real, bodies_mpi[i].z_imag);
        bodies[i].q = bodies_mpi[i].q;
        bodies[i].phi = bodies_mpi[i].phi;
    }

    double totalFMM = 0.0;

    for (int iter = 0; iter < iterations; ++iter) {
        // 重置势值
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            bodies[i].phi = 0.0;
        }

        double start_time = MPI_Wtime();
        computeFMM(bodies, theta, 8, rank, size);
        double end_time = MPI_Wtime();

        double dtFMM = end_time - start_time;
        totalFMM += dtFMM;
    }

    double avgFMM = totalFMM / iterations;

    if (rank == 0) {
        cout << "N = " << N << ", iterations = " << iterations << ", theta = " << theta << "\n";
        cout << "MPI processes: " << size << "\n";
        cout << "FMM (avg)    = " << avgFMM << " s\n";
    }

    MPI_Type_free(&MPI_BODY_MPI);
    MPI_Finalize();

    return 0;
}

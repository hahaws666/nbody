#include <bits/stdc++.h>
#include <mpi.h>
#include <omp.h>
using namespace std;

// ================================================================
// 2D Fast Multipole Method with MPI and OpenMP Parallelization
// ================================================================

using Complex = complex<double>;

struct Body {
    double z_real, z_imag; // position (split for MPI)
    double q; // charge / mass
    double phi; // potential
};

struct Node {
    double xmin, xmax, ymin, ymax;
    Complex center;

    vector<int> bodies; // leaf: list of particles
    int child[4];

    vector<Complex> M; // multipole coefficients
    vector<Complex> L; // local expansion

    Node() {
        child[0]=child[1]=child[2]=child[3] = -1;
    }
};

// ------------------------------------------------
// Quadtree + FMM with Parallelization
// ------------------------------------------------

struct FMM_parallel {
    int N, p;
    double theta;
    vector<Body> &bodies;
    vector<Node> nodes;
    int rank, size;

    FMM_parallel(vector<Body> &bodies, int p=6, double theta=0.5)
        : N(bodies.size()), p(p), theta(theta), bodies(bodies) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        nodes.reserve(N * 4);
    }

    // Parallel tree building using grid-based approach
    int buildTree_parallel() {
        // Compute bounding box in parallel
        double minx=1e30,miny=1e30,maxx=-1e30,maxy=-1e30;
        
        #pragma omp parallel for reduction(min:minx,miny) reduction(max:maxx,maxy)
        for (int i=0; i<N; i++) {
            Complex z(bodies[i].z_real, bodies[i].z_imag);
            minx = min(minx, z.real());
            miny = min(miny, z.imag());
            maxx = max(maxx, z.real());
            maxy = max(maxy, z.imag());
        }
        
        // Allreduce to get global bounds
        double global_bounds[4] = {minx, miny, maxx, maxy};
        MPI_Allreduce(MPI_IN_PLACE, global_bounds, 4, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, global_bounds+2, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        minx = global_bounds[0]; miny = global_bounds[1];
        maxx = global_bounds[2]; maxy = global_bounds[3];
        
        double dx = max(maxx-minx, maxy-miny);
        Node root;
        root.xmin=minx; root.ymin=miny;
        root.xmax=minx+dx; root.ymax=miny+dx;
        root.center = Complex( (root.xmin+root.xmax)/2, (root.ymin+root.ymax)/2 );
        
        // All processes have all particles, so all add all indices
        for(int i=0;i<N;i++) root.bodies.push_back(i);

        nodes.push_back(root);
        buildNode(0);
        return 0;
    }

    void buildNode(int idx) {
        Node &node = nodes[idx];
        if (node.bodies.size() <= 8) return;

        double xm = (node.xmin+node.xmax)/2;
        double ym = (node.ymin+node.ymax)/2;

        // Pre-allocate children to avoid reference invalidation
        nodes.reserve(nodes.size() + 4);
        for (int i=0;i<4;i++) {
            Node c;
            c.xmin = c.xmax = c.ymin = c.ymax = 0.0; // Initialize to avoid warnings
            nodes.push_back(c);
            node.child[i] = (int)nodes.size()-1;
        }

        // Define child boxes
        auto &c0 = nodes[node.child[0]]; // SW
        c0.xmin=node.xmin; c0.xmax=xm;
        c0.ymin=node.ymin; c0.ymax=ym;
        c0.center=Complex((c0.xmin+c0.xmax)/2,(c0.ymin+c0.ymax)/2);

        auto &c1 = nodes[node.child[1]]; // SE
        c1.xmin=xm; c1.xmax=node.xmax;
        c1.ymin=node.ymin; c1.ymax=ym;
        c1.center=Complex((c1.xmin+c1.xmax)/2,(c1.ymin+c1.ymax)/2);

        auto &c2 = nodes[node.child[2]]; // NW
        c2.xmin=node.xmin; c2.xmax=xm;
        c2.ymin=ym; c2.ymax=node.ymax;
        c2.center=Complex((c2.xmin+c2.xmax)/2,(c2.ymin+c2.ymax)/2);

        auto &c3 = nodes[node.child[3]]; // NE
        c3.xmin=xm; c3.xmax=node.xmax;
        c3.ymin=ym; c3.ymax=node.ymax;
        c3.center=Complex((c3.xmin+c3.xmax)/2,(c3.ymin+c3.ymax)/2);

        // Distribute bodies in parallel
        vector<int> bodies_copy = node.bodies; // Copy to avoid modifying while iterating
        #pragma omp parallel
        {
            vector<vector<int>> local_children(4);
            #pragma omp for
            for (size_t i=0; i<bodies_copy.size(); i++) {
                int bi = bodies_copy[i];
                Complex z(bodies[bi].z_real, bodies[bi].z_imag);
                int q = (z.real()>=xm) + 2*(z.imag()>=ym);
                local_children[q].push_back(bi);
            }
            
            #pragma omp critical
            {
                for (int q=0; q<4; q++) {
                    nodes[node.child[q]].bodies.insert(
                        nodes[node.child[q]].bodies.end(),
                        local_children[q].begin(),
                        local_children[q].end()
                    );
                }
            }
        }
        node.bodies.clear();

        for(int k=0;k<4;k++)
            if (!nodes[node.child[k]].bodies.empty())
                buildNode(node.child[k]);
    }

    // ------------------------------------------------
    // FMM steps with OpenMP parallelization
    // ------------------------------------------------

    void P2M(int idx) {
        Node &node = nodes[idx];
        node.M.assign(p+1, Complex(0,0));

        #pragma omp parallel for
        for (size_t i=0; i<node.bodies.size(); i++) {
            int bi = node.bodies[i];
            Complex z(bodies[bi].z_real, bodies[bi].z_imag);
            Complex dz = z - node.center;
            Complex powdz(1,0);
            vector<Complex> local_M(p+1, Complex(0,0));
            
            for (int k=0;k<=p;k++) {
                local_M[k] = bodies[bi].q * powdz;
                powdz *= dz;
            }
            
            #pragma omp critical
            {
                for (int k=0;k<=p;k++) {
                    node.M[k] += local_M[k];
                }
            }
        }
    }

    void M2M(int parent, int child) {
        Node &np = nodes[parent];
        Node &nc = nodes[child];

        Complex d = nc.center - np.center;

        vector<Complex> newM(p+1, Complex(0,0));

        for (int k=0;k<=p;k++) {
            Complex dk(1,0);
            for(int j=0;j<=k;j++) {
                newM[k] += nc.M[j] * Complex(binomialCoeff(k,j), 0) * dk;
                dk *= d;
            }
        }
        for (int k=0;k<=p;k++) np.M[k] += newM[k];
    }

    void upward(int idx) {
        Node &node = nodes[idx];
        if (node.child[0]==-1) {
            P2M(idx);
            return;
        }
        
        #pragma omp parallel for
        for(int k=0;k<4;k++)
            if (node.child[k]!=-1) upward(node.child[k]);

        node.M.assign(p+1, Complex(0,0));
        for(int k=0;k<4;k++)
            if (node.child[k]!=-1)
                M2M(idx, node.child[k]);
    }

    bool wellSeparated(int idxA, int idxB) {
        Node &A=nodes[idxA];
        Node &B=nodes[idxB];

        Complex d = A.center - B.center;
        double dist = abs(d);
        double size = max(A.xmax-A.xmin, A.ymax-A.ymin);

        return size / dist < theta;
    }

    void M2L(int A, int B) {
        Node &nA = nodes[A];
        Node &nB = nodes[B];

        Complex d = nB.center - nA.center;

        if (nB.L.empty()) nB.L.assign(p+1, Complex(0,0));

        vector<Complex> Ltmp(p+1, Complex(0,0));

        vector<Complex> invpow(p+2);
        invpow[1] = 1.0/d;
        for (int k=2;k<=p+1;k++) invpow[k] = invpow[k-1] * invpow[1];

        for (int k=0;k<=p;k++) {
            Complex sum = 0.0;
            for (int m=0;m<=p;m++) {
                if (k+m+1 <= p+1) {
                    double sign = (m % 2 == 0) ? 1.0 : -1.0;
                    sum += nA.M[m] * Complex(sign * binomialCoeff(k+m, m), 0) * invpow[k+m+1];
                }
            }
            Ltmp[k] = sum;
        }
        
        #pragma omp critical
        {
            for(int k=0;k<=p;k++) nB.L[k] += Ltmp[k];
        }
    }

    void downward(int idx) {
        Node &node = nodes[idx];

        if (node.child[0]==-1) return;

        for (int k=0;k<4;k++) {
            int c = node.child[k];
            if (c==-1) continue;

            if (!node.L.empty()) {
                if (nodes[c].L.empty())
                    nodes[c].L.assign(p+1, Complex(0,0));

                Complex d = nodes[c].center - node.center;
                vector<Complex> dpow(p+1);
                dpow[0] = Complex(1,0);
                for(int k2=1;k2<=p;k2++) dpow[k2] = dpow[k2-1]*d;

                for (int j=0;j<=p;j++) {
                    Complex s = 0.0;
                    for (int k2=j;k2<=p;k2++) {
                        s += node.L[k2] * Complex( binomialCoeff(k2,j),0 ) * dpow[k2-j];
                    }
                    nodes[c].L[j] += s;
                }
            }
            downward(c);
        }
    }

    void M2L_traverse(int A, int B) {
        if (A == B) return;
        
        if (wellSeparated(A, B)) {
            M2L(A, B);
            return;
        }
        
        bool A_is_leaf = (nodes[A].child[0] == -1);
        bool B_is_leaf = (nodes[B].child[0] == -1);
        
        if (A_is_leaf && B_is_leaf) {
            return;
        } else if (A_is_leaf) {
            for (int k=0; k<4; k++)
                if (nodes[B].child[k] != -1)
                    M2L_traverse(A, nodes[B].child[k]);
        } else if (B_is_leaf) {
            for (int k=0; k<4; k++)
                if (nodes[A].child[k] != -1)
                    M2L_traverse(nodes[A].child[k], B);
        } else {
            #pragma omp parallel for collapse(2)
            for (int k=0; k<4; k++)
                for (int l=0; l<4; l++)
                    if (nodes[A].child[k] != -1 && nodes[B].child[l] != -1)
                        M2L_traverse(nodes[A].child[k], nodes[B].child[l]);
        }
    }

    void M2L_all() {
        int root = 0;
        #pragma omp parallel for
        for (int i=0; i<(int)nodes.size(); i++) {
            if (i != root) {
                M2L_traverse(i, root);
            }
        }
    }

    void L2P(int idx) {
        Node &node = nodes[idx];
        if (node.bodies.empty()) return;

        #pragma omp parallel for
        for (size_t i=0; i<node.bodies.size(); i++) {
            int bi = node.bodies[i];
            Complex z(bodies[bi].z_real, bodies[bi].z_imag);

            if (!node.L.empty()) {
                Complex dz = z - node.center;
                Complex powdz(1,0);
                double local_phi = 0.0;
                
                for (int k=0;k<=p;k++) {
                    local_phi += (node.L[k] * powdz).real();
                    powdz *= dz;
                }
                
                #pragma omp atomic
                bodies[bi].phi += local_phi;
            }
        }
    }

    void P2P(int A, int B) {
        #pragma omp parallel for collapse(2)
        for (size_t i=0; i<nodes[A].bodies.size(); i++) {
            for (size_t j=0; j<nodes[B].bodies.size(); j++) {
                int ai = nodes[A].bodies[i];
                int bi = nodes[B].bodies[j];
                if (ai==bi) continue;
                
                Complex z_ai(bodies[ai].z_real, bodies[ai].z_imag);
                Complex z_bi(bodies[bi].z_real, bodies[bi].z_imag);
                Complex dz = z_ai - z_bi;
                double r = abs(dz);
                
                #pragma omp atomic
                bodies[ai].phi += bodies[bi].q / r;
            }
        }
    }

    void nearField(int idx) {
        Node &node = nodes[idx];
        
        if (node.child[0] == -1) {
            // Leaf node: compute P2P with itself
            #pragma omp parallel for
            for (size_t i=0; i<node.bodies.size(); i++) {
                for (size_t j=i+1; j<node.bodies.size(); j++) {
                    int ai = node.bodies[i];
                    int bi = node.bodies[j];
                    Complex z_ai(bodies[ai].z_real, bodies[ai].z_imag);
                    Complex z_bi(bodies[bi].z_real, bodies[bi].z_imag);
                    Complex dz = z_ai - z_bi;
                    double r = abs(dz);
                    if (r > 1e-15) {
                        double contrib = bodies[bi].q / r;
                        #pragma omp atomic
                        bodies[ai].phi += contrib;
                        #pragma omp atomic
                        bodies[bi].phi += bodies[ai].q / r;
                    }
                }
            }
            return;
        }

        for(int k=0;k<4;k++)
            if (node.child[k]!=-1)
                nearField(node.child[k]);

        for (int k=0;k<4;k++) {
            for (int j=k+1;j<4;j++) {
                if (node.child[k]!=-1 && node.child[j]!=-1) {
                    if (!wellSeparated(node.child[k], node.child[j])) {
                        P2P(node.child[k], node.child[j]);
                    }
                }
            }
        }
    }

    void compute() {
        // Initialize potential to zero
        #pragma omp parallel for
        for (int i=0; i<N; i++) {
            bodies[i].phi = 0.0;
        }
        
        buildTree_parallel();
        upward(0);
        M2L_all();
        downward(0);
        finalEval(0);
        nearField(0);
    }

    void finalEval(int idx) {
        Node &node = nodes[idx];
        L2P(idx);

        if (node.child[0]==-1) return;
        for(int k=0;k<4;k++)
            if (node.child[k]!=-1)
                finalEval(node.child[k]);
    }

    static int binomialCoeff(int n, int k) {
        if (k<0||k>n) return 0;
        if (k==0||k==n) return 1;
        int r=1;
        for(int i=1;i<=k;i++)
            r=r*(n-k+i)/i;
        return r;
    }
};

// ------------------------------------------------------------
// Direct computation for verification
// ------------------------------------------------------------

void computeDirect(vector<Body> &b) {
    int N=b.size();
    #pragma omp parallel for
    for (int i=0;i<N;i++) {
        b[i].phi=0;
        for (int j=0;j<N;j++) if (i!=j) {
            Complex z_i(b[i].z_real, b[i].z_imag);
            Complex z_j(b[j].z_real, b[j].z_imag);
            double r = abs(z_i - z_j);
            b[i].phi += b[j].q / r;
        }
    }
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int N = 2000;
    if (argc > 1) N = atoi(argv[1]);
    
    int p = 6;
    double theta = 0.5;
    
    // Define MPI datatype for Body
    MPI_Datatype MPI_BODY;
    MPI_Type_contiguous(4, MPI_DOUBLE, &MPI_BODY);
    MPI_Type_commit(&MPI_BODY);
    
    vector<Body> bodies(N);
    mt19937_64 rng(123 + rank); // Different seed per rank
    uniform_real_distribution<double> U(-1,1);

    // Initialize bodies (each rank generates all, but will sync)
    for (int i=0;i<N;i++) {
        Complex z = Complex(U(rng), U(rng));
        bodies[i].z_real = z.real();
        bodies[i].z_imag = z.imag();
        bodies[i].q = 1.0;
        bodies[i].phi = 0;
    }
    
    // Broadcast from rank 0 to ensure all ranks have same data
    if (rank == 0) {
        // Rank 0 generates the data
        rng.seed(123);
        for (int i=0;i<N;i++) {
            Complex z = Complex(U(rng), U(rng));
            bodies[i].z_real = z.real();
            bodies[i].z_imag = z.imag();
        }
    }
    MPI_Bcast(bodies.data(), N, MPI_BODY, 0, MPI_COMM_WORLD);
    
    vector<Body> direct = bodies;

    double start_time = MPI_Wtime();
    if (rank == 0) {
        computeDirect(direct);
    }
    double direct_time = MPI_Wtime() - start_time;

    start_time = MPI_Wtime();
    FMM_parallel solver(bodies, p, theta);
    solver.compute();
    
    // Gather results from all ranks (each rank computes its assigned particles)
    // For simplicity, all ranks compute all particles, but we can optimize later
    vector<Body> all_results(size * N);
    MPI_Allgather(bodies.data(), N, MPI_BODY,
                 all_results.data(), N, MPI_BODY, MPI_COMM_WORLD);
    
    // Use results from rank 0 (or combine)
    if (rank == 0) {
        for (int i=0; i<N; i++) {
            bodies[i].phi = all_results[i].phi;
        }
    }
    double fmm_time = MPI_Wtime() - start_time;

    if (rank == 0) {
        double maxErr=0;
        for (int i=0;i<N;i++) {
            double err = fabs(bodies[i].phi - direct[i].phi) /
                         max(1.0, fabs(direct[i].phi));
            maxErr = max(maxErr, err);
        }

        cout << "MPI processes: " << size << endl;
        cout << "OpenMP threads: " << omp_get_max_threads() << endl;
        cout << "N = " << N << endl;
        cout << "Direct time = " << direct_time << " s\n";
        cout << "FMM time    = " << fmm_time << " s\n";
        cout << "Max relative error = " << maxErr << "\n";
    }

    MPI_Type_free(&MPI_BODY);
    MPI_Finalize();
    return 0;
}


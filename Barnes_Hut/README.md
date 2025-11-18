# Barnes-Hut N-Body Simulation with MPI and OpenMP

This project implements a parallel N-body simulation using the Barnes-Hut algorithm with both MPI (Message Passing Interface) and OpenMP for hybrid parallelization.

## Features

- **Barnes-Hut Algorithm**: Efficient O(N log N) force calculation using spatial tree structure
- **MPI Parallelization**: Distributed memory parallelization across multiple processes
- **OpenMP Parallelization**: Shared memory parallelization within each process
- **Hybrid Parallelization**: Combines MPI and OpenMP for maximum performance

## Algorithm Overview

The Barnes-Hut algorithm uses a quadtree (2D) to approximate gravitational forces:
- Particles are organized in a spatial tree structure
- Distant groups of particles are approximated as a single point mass at their center of mass
- The approximation is used when `size/distance < THETA` (default: 0.5)

## Compilation

```bash
make
```

This will compile with:
- C++14 standard
- OpenMP support (`-fopenmp`)
- Optimization level O3
- MPI compiler (`mpic++`)

## Running

### Basic run (1000 particles, 2 MPI processes)
```bash
make run
# or
mpirun -np 2 ./nbody_bh 1000
```

### Large simulation (10000 particles, 4 MPI processes)
```bash
make run-large
# or
mpirun -np 4 ./nbody_bh 10000
```

### Custom run
```bash
mpirun -np <num_processes> ./nbody_bh <num_particles>
```

## Configuration

You can modify the following constants in `nbody_bh.cpp`:

- `THETA`: Barnes-Hut threshold (default: 0.5)
- `G`: Gravitational constant (default: 1.0)
- `DT`: Time step (default: 0.01)
- `N_ITERATIONS`: Number of time steps (default: 100)
- `OUTPUT_INTERVAL`: Output frequency (default: 10)

## Output

The program outputs:
- Initial configuration (number of bodies, processes, threads)
- Every `OUTPUT_INTERVAL` iterations:
  - Center of mass position
  - Total mass
  - Kinetic energy
  - Potential energy
  - Total energy
- Final timing information

## Parallelization Strategy

1. **MPI Level**: Each process is responsible for computing forces for a subset of particles
   - Particle `i` is computed by process `i % size`
   - After computation, all processes synchronize particle data

2. **OpenMP Level**: Within each process, multiple threads:
   - Build the tree in parallel (with critical sections)
   - Compute forces for assigned particles in parallel
   - Calculate statistics (energy, center of mass) in parallel

## Requirements

- MPI implementation (OpenMPI or MPICH)
- C++ compiler with C++14 support
- OpenMP support

## Performance Notes

- The Barnes-Hut algorithm reduces complexity from O(N²) to O(N log N)
- Hybrid parallelization (MPI + OpenMP) is effective for large-scale simulations
- Tree construction has some serialization due to critical sections, but force computation is fully parallel


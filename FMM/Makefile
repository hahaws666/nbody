CXX = mpic++
CXXFLAGS = -std=c++14 -O3 -Wall -fopenmp
LDFLAGS = -fopenmp

# Serial version (with MPI)
fmm_test: FMM.cpp
	$(CXX) $(CXXFLAGS) -o fmm_test FMM.cpp $(LDFLAGS) -lmpi

# Parallel version (MPI + OpenMP)
fmm_parallel: FMM_parallel.cpp
	$(CXX) $(CXXFLAGS) -o fmm_parallel FMM_parallel.cpp $(LDFLAGS)

# Profile version
CXXFLAGS_PROFILE = -std=c++14 -O3 -Wall -fopenmp -g -pg
profile: FMM_parallel.cpp
	$(CXX) $(CXXFLAGS_PROFILE) -o fmm_parallel FMM_parallel.cpp $(LDFLAGS)

clean:
	rm -f fmm_test fmm_parallel gmon.out profile*.txt

.PHONY: clean profile


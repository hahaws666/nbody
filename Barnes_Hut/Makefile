CXX = mpic++
CXXFLAGS = -std=c++14 -O3 -Wall -fopenmp -lm
CXXFLAGS_PROFILE = -std=c++14 -O3 -Wall -fopenmp -g -pg -lm
TARGET = nbody_bh
TARGET_DIST = nbody_bh_distributed
SOURCE = nbody_bh.cpp
SOURCE_DIST = nbody_bh_distributed.cpp

all: $(TARGET)

distributed: $(TARGET_DIST)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE)

$(TARGET_DIST): $(SOURCE_DIST)
	$(CXX) $(CXXFLAGS) -o $(TARGET_DIST) $(SOURCE_DIST)

profile: $(SOURCE)
	$(CXX) $(CXXFLAGS_PROFILE) -o $(TARGET) $(SOURCE)
	@echo "Built with profiling support (-g -pg)"
	@echo "Run the program, then use: gprof $(TARGET) gmon.out > profile.txt"

profile-dist: $(SOURCE_DIST)
	$(CXX) $(CXXFLAGS_PROFILE) -o $(TARGET_DIST) $(SOURCE_DIST)
	@echo "Built distributed version with profiling support (-g -pg)"
	@echo "Run the program, then use: gprof $(TARGET_DIST) gmon.out > profile.txt"

clean:
	rm -f $(TARGET) $(TARGET_DIST) gmon.out profile.txt

run: $(TARGET)
	mpirun -np 4 ./$(TARGET) 100000

run-large: $(TARGET)
	mpirun -np 4 ./$(TARGET) 100000

run-dist: $(TARGET_DIST)
	mpirun -np 4 ./$(TARGET_DIST) 100000

.PHONY: all distributed clean run run-large run-dist profile profile-dist


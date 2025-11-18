CXX = mpic++
CXXFLAGS = -std=c++14 -O3 -Wall -fopenmp
CXXFLAGS_PROFILE = -std=c++14 -O3 -Wall -fopenmp -g -pg
TARGET = nbody_bh
SOURCE = nbody_bh.cpp

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE)

profile: $(SOURCE)
	$(CXX) $(CXXFLAGS_PROFILE) -o $(TARGET) $(SOURCE)
	@echo "Built with profiling support (-g -pg)"
	@echo "Run the program, then use: gprof $(TARGET) gmon.out > profile.txt"

clean:
	rm -f $(TARGET) gmon.out profile.txt

run: $(TARGET)
	mpirun -np 2 ./$(TARGET) 1000

run-large: $(TARGET)
	mpirun -np 4 ./$(TARGET) 100000

.PHONY: all clean run run-large profile


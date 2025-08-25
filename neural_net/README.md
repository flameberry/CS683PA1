# Matrix Optimisation Assignment - Task 1E

This assignment focuses on implementing and benchmarking various matrix multiplication optimisation techniques to understand the performance characteristics of different approaches.

## Overview

You will implement five different matrix multiplication algorithms and analyze their performance characteristics:

1. **Naive** - Basic triple-nested loop implementation (DO NOT MODIFY)
2. **Reordered** - Cache-friendly loop reordering
3. **Unrolled** - Loop unrolling optimisation
4. **Tiled** - Cache blocking/tiling technique
5. **Vectorized** - SIMD (Single Instruction, Multiple Data) implementation

Additionally, you need to optimize a matrix transpose function.

## Project Structure

```
neural_net/
├── bin/                    # Compiled binaries
├── include/               # Header files
│   ├── defines.h         # Common definitions and constants
│   ├── matrix.h          # Matrix data structure definitions
│   ├── matrix_operation.h # Matrix operation function declarations
│   ├── nn.h              # Neural network related functions
│   └── performance_benchmark.h # Benchmarking utilities
├── Makefile              # Build configuration
├── README.md            # This file
└── src/                 # Source code
    ├── main.cpp         # Main program entry point
    ├── matrix_operation.c # YOUR IMPLEMENTATION GOES HERE
    └── nn.c             # Neural network implementation
```

## Implementation Requirements

### Primary Task: Matrix Multiplication Optimisation

Edit `src/matrix_operation.c` and implement the following functions:

#### 1. Reordered Implementation

- Optimize loop order for better cache locality
- Consider data access patterns and cache line utilization

#### 2. Unrolled Implementation

- Implement loop unrolling to reduce loop overhead
- Balance between code size and performance gains

#### 3. Tiled Implementation

- Use cache blocking/tiling technique
- Optimize for L1/L2 cache sizes
- Choose appropriate tile sizes

#### 4. Vectorized Implementation (SIMD)

- Use SIMD instructions for parallel computation
- You may create additional helper functions
- Optimize for your target architecture (SSE, AVX, etc.)
- Feel free to implement the most optimized version possible

### Secondary Task: Matrix Transpose Optimization

Optimize the matrix transpose function using similar techniques.

## Build Instructions

bash

```bash
# Compile the project
make

# Run the benchmark
./bin/[executable_name]
```

The compiled binary will be created in the `./bin` folder.

## Performance Analysis

Your implementation should include:

1. **Speed-up Analysis**: Generate plots showing the performance improvement of each optimization technique compared to the naive implementation
2. **Matrix Size Scaling**: Test performance across different matrix sizes to understand scaling behavior
3. **Comparative Analysis**: Compare all implementations and analyze:
    - Absolute performance (operations per second)
    - Relative speed-up over naive implementation
    - Memory access patterns
    - Cache efficiency

## Expected Deliverables

Make a `neural_net.pdf`  documenting everything.

1. **Optimized Code**: Complete implementations in `src/matrix_operation.c`
2. **Performance Plots**: Speed-up charts for different matrix sizes
3. **Analysis Report**: Brief explanation of optimisation techniques used and performance results
   
Good luck with your optimisation journey! :)

# PA-1: Matrix Multiplication – *The Rancho Way*

## Objective

The goal of this assignment is to understand and optimise a naive matrix multiplication algorithm using various optimising techniques. You will be working with square matrices of size `size x size`.

---

## Given Code

You are provided with the following baseline C++ implementation of matrix multiplication:

```cpp
for (int i = 0; i < size; i++) {                 // select a row in A
    for (int j = 0; j < size; j++) {             // select a col in B
        for (int k = 0; k < size; k++) {         // number of operations for each element in C
            C[i * size + j] += A[i * size + k] * B[k * size + j];
        }
    }
}
```

## Tasks

You are required to optimise the matrix multiplication code using the following techniques:

- 1A: **Loop Unrolling and Loop Reordering**
	Apply loop unrolling to reduce loop overhead and improve instruction-level parallelism.
	Apply loop reordering to improve memory access patterns and cache performance.

- 1B: **Tiling (Blocking)**
	Use tiling to improve data locality and better utilise CPU caches.

- 1C: **SIMD (Single Instruction, Multiple Data)**
	Use SIMD intrinsic to for optimising the matrix multiplication. You can use intel intrinsic that is supported in your system.

- 1D: **Combination of Techniques**
	Combine any of the above optimisations.
	Try different combinations and compare their performance.
	Analyse and justify which combination performs best.
	In-depth performance analysis will result in scoring full marks.

## Report

Prepare a detailed report titled: `mat_mul_analysis.pdf`

The report should include:
- Execution time comparison across all versions.
- Graphs/plots showing speedup and scalability.
- Final insights and conclusion about the best-performing version.

## Submission Checklist

- Optimised code for 1A, 1B, 1C, and 1D
- `mat_mul_analysis.pdf` with complete analysis
- Graphs and plots embedded in the report
- Code is clean, modular, and well-commented

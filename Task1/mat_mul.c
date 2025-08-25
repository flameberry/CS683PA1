/*******************************************************************
 * Author: <Name1>, <Name2>
 * Date: <Date>
 * File: mat_mul.c
 * Description: This file contains implementations of matrix multiplication
 *			    algorithms using various optimization techniques.
 *******************************************************************/

// PA 1: Matrix Multiplication

// includes
#include <stdio.h>
#include <stdlib.h>         // for malloc, free, atoi
#include <time.h>           // for time()
#include <chrono>	        // for timing
#include <xmmintrin.h> 		// for SSE
#include <immintrin.h>		// for AVX

#include "helper.h"			// for helper functions

// defines
// NOTE: you can change this value as per your requirement
#define TILE_SIZE	64		// size of the tile for blocking

/**
 * @brief 		Performs matrix multiplication of two matrices.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 */
void naive_mat_mul(double *A, double *B, double *C, int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			for (int k = 0; k < size; k++) {
				C[i * size + j] += A[i * size + k] * B[k * size + j];
			}
		}
	}
}

/**
 * @brief 		Task 1A: Performs matrix multiplication of two matrices using loop optimization.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 */
void loop_opt_mat_mul(double *A, double *B, double *C, int size){
	for (int i = 0; i < size; i++) {
    for (int k = 0; k < size; k++) {
      const int a = A[i * size + k];
      for (int j = 0; j < size; j += 4) {
				C[i * size + j + 0] += a * B[k * size + j + 0];
				C[i * size + j + 1] += a * B[k * size + j + 1];
				C[i * size + j + 2] += a * B[k * size + j + 2];
				C[i * size + j + 3] += a * B[k * size + j + 3];
			}
		}
	}
}


/**
 * @brief 		Task 1B: Performs matrix muiltiplication of two matrices using tiling.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 * @param 		tile_size 	size of the tile
 * @note 		The tile size should be a multiple of the dimension of the matrices.
 * 				For example, if the dimension is 1024, then the tile size can be 32, 64, 128, etc.
 * 				You can assume that the matrices are square matrices.
*/
void tile_mat_mul(double *A, double *B, double *C, int size, int tile_size) {
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
 for (int ii = 0; ii < size; ii += tile_size) {
    for (int jj = 0; jj < size; jj += tile_size) {
      for (int kk = 0; kk < size; kk += tile_size) {
        for (int i  = ii; i < ii + tile_size; i++) {
          for (int j = jj; j < jj + tile_size; j++) {
            for (int k = kk; k < kk + tile_size; k++) {
              C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
          }
        }
      }
    }
  }    
//-------------------------------------------------------------------------------------------------------------------------------------------
    
}

/**
 * @brief 		Task 1C: Performs matrix multiplication of two matrices using SIMD instructions.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 * @note 		You can assume that the matrices are square matrices.
*/
void simd_mat_mul(double *A, double *B, double *C, int size) {
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    // Loop over rows of A and columns of B
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            __m256d c_val = _mm256_setzero_pd();  // Initialize SIMD register to zero

            // Perform the dot product for row i of A and column j of B
            for (int k = 0; k < size; k += 4) {
                // Load 4 doubles from row i of A and column j of B into SIMD registers
                __m256d a_val = _mm256_loadu_pd(&A[i * size + k]);  // Load 4 elements from A
                __m256d b_val = _mm256_loadu_pd(&B[k * size + j]);  // Load 4 elements from B

                // Perform the dot product: c_val += a_val * b_val
                c_val = _mm256_fmadd_pd(a_val, b_val, c_val);  // c_val += a_val * b_val
            }

            // Store the result in the corresponding element of C
            double result[4];
            _mm256_storeu_pd(result, c_val);  // Store the result of the SIMD computation

            // Sum the 4 elements of the result into the final value
            C[i * size + j] = result[0] + result[1] + result[2] + result[3];
        }
    }
//-------------------------------------------------------------------------------------------------------------------------------------------
    
}

/**
 * @brief 		Task 1D: Performs matrix multiplication of two matrices using combination of tiling/SIMD/loop optimization.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 * @param 		tile_size 	size of the tile
 * @note 		The tile size should be a multiple of the dimension of the matrices.
 * @note 		You can assume that the matrices are square matrices.
*/
void combination_mat_mul(double *A, double *B, double *C, int size, int tile_size) {
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
  for (int ii = 0; ii < size; ii += tile_size) {
        for (int jj = 0; jj < size; jj += tile_size) {
            for (int kk = 0; kk < size; kk += tile_size) {
                for (int i = ii; i < ii + tile_size; i++) {
                    for (int k = kk; k < kk + tile_size; k++) {
                        const double a = A[i * size + k]; // Get value of A[i][k]

                        // Using AVX2 to handle 4 values of B at a time
                        for (int j = jj; j < jj + tile_size; j += 4) {
                            // Load 4 values from B into SIMD register (256-bit register)
                            __m256d b_vals = _mm256_loadu_pd(&B[k * size + j]);

                            // Multiply a (single value) by 4 values in b_vals and accumulate in SIMD register
                            __m256d c_vals = _mm256_loadu_pd(&C[i * size + j]); // Load current values of C[i][j..j+3]
                            __m256d a_val = _mm256_set1_pd(a); // Set the value of A[i][k] in all 4 positions of the register
                            c_vals = _mm256_fmadd_pd(a_val, b_vals, c_vals); // c_vals += a * b_vals

                            // Store the result back into C
                            _mm256_storeu_pd(&C[i * size + j], c_vals);
                        }
                    }
                }
            }
        }
    }
//-------------------------------------------------------------------------------------------------------------------------------------------
    
}

// NOTE: DO NOT CHANGE ANYTHING BELOW THIS LINE
/**
 * @brief 		Main function
 * @param 		argc 		number of command line arguments
 * @param 		argv 		array of command line arguments
 * @return 		0 on success
 * @note 		DO NOT CHANGE THIS FUNCTION
 * 				DO NOT ADD OR REMOVE ANY COMMAND LINE ARGUMENTS
*/
int main(int argc, char **argv) {

	if ( argc <= 1 ) {
		printf("Usage: %s <matrix_dimension>\n", argv[0]);
		return 0;
	}

	else {
		int size = atoi(argv[1]);

		double *A = (double *)malloc(size * size * sizeof(double));
		double *B = (double *)malloc(size * size * sizeof(double));
		double *C = (double *)calloc(size * size, sizeof(double));

		// initialize random seed
		srand(time(NULL));

		// initialize matrices A and B with random values
		initialize_matrix(A, size, size);
		initialize_matrix(B, size, size);

		// perform normal matrix multiplication
		auto start = std::chrono::high_resolution_clock::now();
		naive_mat_mul(A, B, C, size);
		auto end = std::chrono::high_resolution_clock::now();
		auto time_naive_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Normal matrix multiplication took %ld ms to execute \n\n", time_naive_mat_mul);

	#ifdef OPTIMIZE_LOOP_OPT
		// Task 1a: perform matrix multiplication with loop optimization

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		start = std::chrono::high_resolution_clock::now();
		loop_opt_mat_mul(A, B, C, size);
		end = std::chrono::high_resolution_clock::now();
		auto time_loop_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Loop optimized matrix multiplication took %ld ms to execute \n", time_loop_mat_mul);
		printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_loop_mat_mul);
	#endif

	#ifdef OPTIMIZE_TILING
		// Task 1b: perform matrix multiplication with tiling

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		start = std::chrono::high_resolution_clock::now();
		tile_mat_mul(A, B, C, size, TILE_SIZE);
		end = std::chrono::high_resolution_clock::now();
		auto time_tiling_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Tiling matrix multiplication took %ld ms to execute \n", time_tiling_mat_mul);
		printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_tiling_mat_mul);
	#endif

	#ifdef OPTIMIZE_SIMD
		// Task 1c: perform matrix multiplication with SIMD instructions 

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		start = std::chrono::high_resolution_clock::now();
		simd_mat_mul(A, B, C, size);
		end = std::chrono::high_resolution_clock::now();
		auto time_simd_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

		printf("SIMD matrix multiplication took %ld ms to execute \n", time_simd_mat_mul);
		printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_simd_mat_mul);
	#endif

	#ifdef OPTIMIZE_COMBINED
		// Task 1d: perform matrix multiplication with combination of tiling, SIMD and loop optimization

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		start = std::chrono::high_resolution_clock::now();
		combination_mat_mul(A, B, C, size, TILE_SIZE);
		end = std::chrono::high_resolution_clock::now();
		auto time_combination = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Combined optimization matrix multiplication took %ld ms to execute \n", time_combination);
		printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_combination);
	#endif

		// free allocated memory
		free(A);
		free(B);
		free(C);

		return 0;
	}
}
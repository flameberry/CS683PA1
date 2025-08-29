#include "matrix_operation.h"
#include <immintrin.h>

Matrix MatrixOperation::NaiveMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
	size_t k = A.getCols();
	size_t m = B.getCols();

	if (k != B.getRows()) {
		throw std::invalid_argument("Matrix dimensions don't match for multiplication");
	}

	Matrix C(n, m);

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			for (int l = 0; l < k; l++) {
				C(i, j) += A(i, l) * B(l, j);
			}
		}
	}

	return C;
}

// Loop reordered matrix multiplication (ikj order for better cache locality)
Matrix MatrixOperation::ReorderedMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
	size_t k = A.getCols();
	size_t m = B.getCols();

	if (k != B.getRows()) {
		throw std::invalid_argument("Matrix dimensions don't match for multiplication");
	}

	Matrix C(n, m);

	//----------------------------------------------------- Write your code here ----------------------------------------------------------------

	for (int i = 0; i < n; i++) {
		for (int l = 0; l < k; l++) {
			const int a = A(i, l);
			for (int j = 0; j < m; j++) {
				C(i, j) += a * B(l, j);
			}
		}
	}

	//-------------------------------------------------------------------------------------------------------------------------------------------

	return C;
}

// Loop unrolled matrix multiplication
Matrix MatrixOperation::UnrolledMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
	size_t k = A.getCols();
	size_t m = B.getCols();

	if (k != B.getRows()) {
		throw std::invalid_argument("Matrix dimensions don't match for multiplication");
	}

	Matrix C(n, m);

	const int UNROLL = 4;
	//----------------------------------------------------- Write your code here ----------------------------------------------------------------

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			for (int l = 0; l < k; l += 4) {
				C(i, j) += A(i, l + 0) * B(l + 0, j);
				C(i, j) += A(i, l + 1) * B(l + 1, j);
				C(i, j) += A(i, l + 2) * B(l + 2, j);
				C(i, j) += A(i, l + 3) * B(l + 3, j);
			}
		}
	}

	//-------------------------------------------------------------------------------------------------------------------------------------------

	return C;
}

// Tiled (blocked) matrix multiplication for cache efficiency
Matrix MatrixOperation::TiledMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
	size_t k = A.getCols();
	size_t m = B.getCols();

	if (k != B.getRows()) {
		throw std::invalid_argument("Matrix dimensions don't match for multiplication");
	}

	Matrix C(n, m);
	const int T = 128; // tile size
	int i_max = 0;
	int k_max = 0;
	int j_max = 0;
	//----------------------------------------------------- Write your code here ----------------------------------------------------------------

	const int TILE_SIZE = 64;
	for (int ii = 0; ii < n; ii += TILE_SIZE) {
		for (int jj = 0; jj < k; jj += TILE_SIZE) {
			for (int ll = 0; ll < m; ll += TILE_SIZE) {
				int i_max = std::min(ii + TILE_SIZE, n);
				int j_max = std::min(jj + TILE_SIZE, k);
				int l_max = std::min(ll + TILE_SIZE, m);
				for (int i = ii; i < i_max; i++) {
					for (int j = jj; j < j_max; j++) {
						for (int l = ll; l < l_max; l++) {
							C(i, j) += A(i, l) * B(l, j);
						}
					}
				}
			}
		}
	}

	//-------------------------------------------------------------------------------------------------------------------------------------------

	return C;
}

// SIMD vectorized matrix multiplication (using AVX2)
Matrix MatrixOperation::VectorizedMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
	size_t k = A.getCols();
	size_t m = B.getCols();

	if (k != B.getRows()) {
		throw std::invalid_argument("Matrix dimensions don't match for multiplication");
	}

	Matrix C(n, m);
	//----------------------------------------------------- Write your code here ----------------------------------------------------------------

	for (int i = 0; i < n; i++) {
		for (int l = 0; l < m; l++) {
			const double a = A(i, l);
			__m256d a_ext = _mm256_set1_pd(a);

			int j = 0;
			for (; j + 4 <= k; j += 4) {
				__m256d b = _mm256_loadu_pd(&B(l, j));
				__m256d c = _mm256_loadu_pd(&C(i, j));
				c = _mm256_fmadd_pd(a_ext, b, c);
				_mm256_storeu_pd(&C(i, j), c);
			}

			// Handle leftover elements (k not divisible by 4)
			for (; j < k; j++) {
				C(i, j) += a * B(l, j);
			}
		}
	}

	//-------------------------------------------------------------------------------------------------------------------------------------------

	return C;
}

// Optimized matrix transpose
Matrix MatrixOperation::Transpose(const Matrix& A) {
	size_t rows = A.getRows();
	size_t cols = A.getCols();
	Matrix result(cols, rows);

	// for (size_t i = 0; i < rows; ++i) {
	// 	for (size_t j = 0; j < cols; ++j) {
	// 		result(j, i) = A(i, j);
	// 	}
	// }

	// Optimized transpose using blocking for better cache performance
	// This is a simple implementation, more advanced techniques can be applied
	// Write your code here and commnent the above code
	//----------------------------------------------------- Write your code here ----------------------------------------------------------------

	const int TILE_SIZE = 64;
	for (int ii = 0; ii < rows; ii += TILE_SIZE) {
		for (int jj = 0; jj < cols; jj += TILE_SIZE) {
			const int i_max = std::min(ii + TILE_SIZE, rows);
			const int j_max = std::min(jj + TILE_SIZE, cols);
			for (int i = ii; i < i_max; i++) {
				for (int j = jj; j < j_max; j++) {
					result(j, i) = A(i, j);
				}
			}
		}
	}

	//-------------------------------------------------------------------------------------------------------------------------------------------

	return result;
}

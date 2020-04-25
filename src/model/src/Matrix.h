

#ifndef MATRIX_H_
#define MATRIX_H_

#define SEED 42


/**
 *  A Matrix Class
 *  Assumes Row-Major
 */
template<class T>
class Matrix {
public:
	int cols;
	int rows;
	T* elems;
	bool init;

	/**
	 * Initializes a matrix with rows and cols
	 * Allocates memory if init == true, otherwise does not allocate memory for the values
	 *
	 * @param rows
	 * @param cols
	 * @param init
	 */
	__host__ __device__ Matrix(int64_t rows, int64_t cols, bool init = true) {
		this->cols = cols;
		this->rows = rows;
		this->init = init;
		if (init) {
			this->elems = new T[cols * rows];
		}

	}

	/**
	 * If memory for the matrix is allocated: delete the allocate memory
	 * Otherwise do nothing
	 */
	__host__ __device__ virtual ~Matrix() {
		if (init) {
			delete[] elems;
		}

	}

	/**
	 * Fills the Matrix with a constant value
	 */
	void fillConst(const T val) {
		for (int i = 0; i < cols * rows; ++i) {
			elems[i] = val;
		}
	}

};

#endif /* MATRIX_H_ */

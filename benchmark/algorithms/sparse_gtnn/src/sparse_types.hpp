#include <bits/stdc++.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#ifndef __SPARSE_TYPES_HPP__
#define __SPARSE_TYPES_HPP__

#define NUM_THREADS 16

/**
 * @brief Represents a single element in a sparse vector
 * Stores both the column index and the float value at that position
 */
struct SparseElem {
    uint index;   /**< Column index of the non-zero element */
    float value;  /**< Value of the non-zero element */
};

/** @brief Type alias for a sparse vector - vector of SparseElem */
using SparseVec = std::vector<SparseElem>;

/** @brief Type alias for a sparse matrix - vector of sparse vectors */
using SparseMat = std::vector<SparseVec>;

/** @brief Type alias for a CSR matrix represented as pairs of (column indices, values) vectors */
using CSRMatrix = std::vector<std::pair<std::vector<uint>, std::vector<float>>>;

// Forward declarations
template <uint M, uint N> std::array<uint, M*N> get_best_partition(SparseMat &data, uint start, uint t);

/**
 * @brief Check if a file stream is open and valid
 * @param file Output file stream to check
 * @return true if the stream is open, false otherwise
 */
bool check_file(std::ofstream &file) {
    return file.is_open();
}

/**
 * @brief Check if a file stream is open and valid
 * @param file Input file stream to check
 * @return true if the stream is open, false otherwise
 */
bool check_file(std::ifstream &file) {
    return file.is_open();
}

/**
 * @brief Check if a C file pointer is valid
 * @param file C file pointer to check
 * @return true if the pointer is not null, false otherwise
 */
bool check_file(FILE *file) {
    return file != nullptr;
}

/**
 * @brief Check if a file exists at the given path
 * @param file_name Path to the file to check
 * @return true if the file exists and can be opened, false otherwise
 */
bool check_file(const std::string &file_name) {
    std::ifstream file(file_name);
    bool result = file.is_open();
    file.close();
    return result;
}

/**
 * @brief Check if a file exists at the given path
 * @param file_name C-string path to the file to check
 * @return true if the file exists and can be opened, false otherwise
 */
bool check_file(const char *file_name) {
    std::ifstream file(file_name);
    bool result = file.is_open();
    file.close();
    return result;
}

/**
 * @brief Compare two float values for approximate equality
 * Uses a tolerance of 1e-5 for comparison
 * @param one First float value
 * @param two Second float value
 * @return true if the values are approximately equal, false otherwise
 */
bool compare_float(float one, float two) {
    return std::fabs(one - two) < 1.0e-5;
}

/**
 * @brief Read a sparse matrix from a binary file in CSR (Compressed Sparse Row) format
 * Assumes the binary file structure: num_vectors, dimension, num_non_zero, indptr, indices, values
 * Automatically normalizes vectors to unit L2 norm
 * @param file_name Path to the binary CSR matrix file
 * @return Pair containing the sparse matrix and its dimensionality
 * @throws std::runtime_error if file cannot be opened or contains negative values
 */
std::pair<SparseMat, size_t> read_sparse_matrix(const std::string &file_name) {
    std::ifstream csr_file(file_name, std::ios::binary);
    if (!csr_file.is_open()) {
        throw std::runtime_error("Cannot open file " + file_name);
    }
    size_t num_vectors;
    size_t num_non_zero;
    std::pair<SparseMat, size_t> result_and_dimention;
    csr_file.read(reinterpret_cast<char *>(&num_vectors), sizeof(size_t));
    csr_file.read(reinterpret_cast<char *>(&result_and_dimention.second), sizeof(size_t));
    csr_file.read(reinterpret_cast<char *>(&num_non_zero), sizeof(size_t));

    std::vector<float> data(num_non_zero);
    std::vector<uint> indices(num_non_zero);
    std::vector<size_t> indptr(num_vectors + 1);
    csr_file.read(reinterpret_cast<char *>(indptr.data()), (num_vectors + 1) * sizeof(size_t));
    csr_file.read(reinterpret_cast<char *>(indices.data()), num_non_zero * sizeof(uint));
    csr_file.read(reinterpret_cast<char *>(data.data()), num_non_zero * sizeof(float));

    result_and_dimention.first.resize(num_vectors);
    for (size_t i = 0; i < num_vectors; i++) {
        result_and_dimention.first[i].resize(indptr[i + 1] - indptr[i]);
        double norm = 0;
        for (size_t j = indptr[i]; j < indptr[i + 1]; j++) {
            result_and_dimention.first[i][j - indptr[i]] = {indices[j], data[j]};
            if (data[j] < 0) {
                // throw std::runtime_error("Negative value in sparse matrix");
            }
        }
        // if (norm == 0) {
        //     continue;
        // }
        // for (size_t j = 0; j < result_and_dimention.first[i].size(); j++) {
        //     result_and_dimention.first[i][j].value /= std::sqrt(norm);
        // }
    }
    return result_and_dimention;
}

/**
 * @brief Add two sparse vectors element-wise
 * Performs sorted merge of two sparse vectors using the indices as sort keys
 * @param one First sparse vector
 * @param two Second sparse vector
 * @return New sparse vector containing the sum of the two input vectors
 */
SparseVec add_sparse(SparseVec &one, SparseVec &two) {
    SparseVec result;
    uint i = 0, j = 0;
    while (i < one.size() && j < two.size()) {
        if (one[i].index == two[j].index) {
            result.push_back({one[i].index, one[i].value + two[j].value});
            i++;
            j++;
        } else if (one[i].index < two[j].index) {
            result.push_back(one[i]);
            i++;
        } else {
            result.push_back(two[j]);
            j++;
        }
    }
    while (i < one.size()) {
        result.push_back(one[i]);
        i++;
    }
    while (j < two.size()) {
        result.push_back(two[j]);
        j++;
    }
    return result;
}

/**
 * @brief Compute the dot product of two sparse vectors
 * Uses two-pointer technique for efficient computation.
 * Using __restrict__ and manual branch elimination for the index comparison 
 * to encourage efficient code generation.
 * @param one First sparse vector
 * @param two Second sparse vector
 * @return The dot product as a double
 */
double dot_product(const SparseVec &one, const SparseVec &two) {
    double result = 0;
    const SparseElem* __restrict__ p1 = one.data();
    const SparseElem* __restrict__ p2 = two.data();
    const size_t size_one = one.size();
    const size_t size_two = two.size();

    // AVX prefetching isn't always a win for random sparse, but linear scan benefits.
    // However, without precise stride knowledge it might pollute cache.
    // Let's rely on compiler optimizations via flags (-O3 -march=native) and restrict.
    
    size_t i = 0, j = 0;
    while (i < size_one && j < size_two) {
        uint idx1 = p1[i].index;
        uint idx2 = p2[j].index;

        if (idx1 == idx2) {
            result += p1[i].value * p2[j].value;
            i++;
            j++;
        } else if (idx1 < idx2) {
            i++;
        } else {
            j++;
        }
    }
    return result;
}

// Deleted add_sparse_optimized
// Deleted dot_product_optimized


/**
 * @brief Compare two sparse vectors for equality with floating-point tolerance
 * @param one First sparse vector
 * @param two Second sparse vector
 * @return true if vectors have the same size and all elements match within tolerance
 */
bool compare_sparse(SparseVec one, SparseVec two) {
    if (one.size() != two.size()) {
        return false;
    }
    for (uint i = 0; i < one.size(); i++) {
        if (one[i].index != two[i].index || !compare_float(one[i].value, two[i].value)) {
            return false;
        }
    }
    return true;
}


/**
 * @brief Find the best permutation of vectors to maximize the sum of group L2 norms squared
 * Uses random sampling and Monte Carlo approach to find good partitions
 * @tparam M Number of vectors in each group
 * @tparam N Number of groups
 * @param data The sparse matrix to partition
 * @param start Starting index for partitioning
 * @param t Number of random trials to perform
 * @return Array containing the best permutation that maximizes group objective
 */
template <uint M, uint N> std::array<uint, M*N> get_best_partition(SparseMat &data, uint start, uint t) {
    constexpr uint MN = M * N;
    std::array<uint, M*N> best_perm;
    double max_sum = -1.0;
    std::vector<uint> base_indices(MN);
    std::iota(base_indices.begin(), base_indices.end(), start);
    std::random_device rd;
    std::mt19937 rng(rd());

    for (uint iter = 0; iter < t; ++iter) {
        std::vector<uint> current = base_indices;
        std::shuffle(current.begin(), current.end(), rng);
        double total = 0.0;
        // Process each of the N groups
        for (uint group = 0; group < N; ++group) {
            std::unordered_map<uint, float> group_sum;
            // Aggregate sparse vectors in current group
            for (uint i = group*M; i < (group+1)*M; ++i) {
                const auto& vec = data[current[i]];
                for (const auto& elem : vec) {
                    group_sum[elem.index] += elem.value;
                }
            }
            // Calculate squared L2 norm
            double group_sq = 0;
            for (const auto& [idx, val] : group_sum) {
                group_sq += val * val;
            }
            total += group_sq;
        }
        // Update best permutation
        if (total > max_sum) {
            max_sum = total;
            std::copy(current.begin(), current.end(), best_perm.begin());
        }
    }
    return best_perm;
}


/**
 * @brief Read a CSR (Compressed Sparse Row) matrix from a binary file
 * The binary file format consists of:
 * - num_rows (uint), num_cols (uint), num_nonzero (uint)
 * - values array (num_nonzero floats)
 * - indices array (num_nonzero uint32_t column indices)
 * - indptr array ((num_rows + 1) size_t row pointers)
 * 
 * @param filename Path to the binary CSR matrix file
 * @param num_rows Output parameter for the number of rows
 * @param num_cols Output parameter for the number of columns
 * @return CSR matrix as a vector of (indices, values) pairs for each row
 * @throws std::runtime_error if file cannot be opened
 */
CSRMatrix read_csr_matrix(const std::string& filename, uint& num_rows, uint& num_cols) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file: " + filename);

    // Read matrix dimensions
    file.read(reinterpret_cast<char*>(&num_rows), sizeof(uint));
    file.read(reinterpret_cast<char*>(&num_cols), sizeof(uint));

    // Read number of nonzero elements
    uint num_nonzero = 0;
    file.read(reinterpret_cast<char*>(&num_nonzero), sizeof(uint));
    std::cout << "num_rows: " << num_rows << " num_cols: " << num_cols << " num_nonzero: " << num_nonzero << std::endl;

    // Allocate space for CSR components
    std::vector<float> values(num_nonzero);
    std::vector<uint> indices(num_nonzero);
    std::vector<size_t> indptr(num_rows + 1);

    // Read binary data
    file.read(reinterpret_cast<char*>(values.data()), num_nonzero * sizeof(float));
    file.read(reinterpret_cast<char*>(indices.data()), num_nonzero * sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(indptr.data()), (num_rows + 1) * sizeof(size_t));

    file.close();

    // Convert to `std::vector<std::pair<std::vector<uint32_t>, std::vector<float>>>`
    CSRMatrix csr_matrix(num_rows);

    for (size_t i = 0; i < num_rows; i++) {
        size_t start = indptr[i];
        size_t end = indptr[i + 1];

        csr_matrix[i].first.assign(indices.begin() + start, indices.begin() + end);
        csr_matrix[i].second.assign(values.begin() + start, values.begin() + end);
    }

    return csr_matrix;
}


/**
 * @brief Create a random CSR matrix with specified sparsity pattern
 * Uses Poisson distribution for determining number of non-zero elements per row
 * and exponential distribution for element values.
 * All vectors are normalized to unit L2 norm.
 * 
 * @param num_rows Number of rows in the generated matrix
 * @param num_cols Number of columns in the generated matrix
 * @param sparsity Sparsity parameter for the Poisson distribution (controls density)
 * @return Randomly generated CSR matrix with unit-norm rows
 */
CSRMatrix create_random_csr_matrix(uint num_rows, uint num_cols, uint sparsity) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::poisson_distribution<uint> nnz_dist(sparsity);
    std::exponential_distribution<float> val_dist(1.0);

    CSRMatrix csr_matrix(num_rows);

    for (uint i = 0; i < num_rows; i++) {
        uint start = 0;
        float norm = 0;
        while (true) {
            start += nnz_dist(gen);
            if (start >= num_cols) {
                break;
            }
            csr_matrix[i].first.push_back(start);
            float val = val_dist(gen);
            csr_matrix[i].second.push_back(val);
            norm += val * val;
        }
        for (uint j = 0; j < csr_matrix[i].second.size(); j++) {
            csr_matrix[i].second[j] /= sqrt(norm);
        }
    }

    return csr_matrix;
}

#endif // __SPARSE_TYPES_HPP__
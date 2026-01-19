#include <bits/stdc++.h>
using uint = unsigned int;

/** @brief Type alias for a CSR matrix represented as pairs of (column indices, values) vectors */
using CSRMatrix = std::vector<std::pair<std::vector<uint>, std::vector<float>>>;

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
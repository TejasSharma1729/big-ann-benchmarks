#pragma once

#include "sparse_types.hpp"
#include <thread>
#include <array>
#include <vector>
#include <algorithm>
#include <queue>
#include <mutex>
#include <atomic>

#define KNN_LEVELS 10
#define INVERTED_LEVELS 5

/** @brief Template for single KNN dataset index - array of sparse vectors */
template <uint N = KNN_LEVELS> using KNNIndexSingle = std::array<SparseVec, (1 << (N + 1))>;
/** @brief Template for complete KNN dataset index - vector of single indices */
template <uint N = KNN_LEVELS> using KNNIndex = std::vector<KNNIndexSingle<N>>;
/** @brief Result element type for double-group search - array of result vectors */
using KNNIndexDoubleGroupResultElem = std::array<std::vector<uint>, (1 << INVERTED_LEVELS)>;

/**
 * @brief Build KNN index from sparse matrix using hierarchical partitioning
 */
template <uint N = KNN_LEVELS>
KNNIndex<N> build_knn_index(SparseMat &matrix) {
    const uint num_vectors = matrix.size();
    const uint num_indices = (num_vectors + (1 << N) - 1) / (1 << N);
    KNNIndex<N> data_index(num_indices);
    for (uint i = 0; i < num_indices; i++) {
        const uint offset = i * (1 << N);
        for (uint j = 0; j < (1 << N); j++) {
            const uint index = offset + j;
            if (index < num_vectors) {
                data_index[i][(1 << N) + j] = matrix[index];
            }
        }
        for (int n = N - 1; n >= 0; n--) {
            for (uint j = 0; j < static_cast<uint>(1 << n); j++) {
                const uint index = (1 << n) + j;
                data_index[i][index] = add_sparse(data_index[i][2 * index], data_index[i][2 * index + 1]);
            }
        }
    }
    return data_index;
}

/**
 * @brief Update KNN index with a new sparse vector
 * @param knn_index KNN index to update
 * @param new_vector New sparse vector to add
 * @param new_index Index of the new vector in the original dataset
 */
template <uint N = KNN_LEVELS>
void update_knn_index(KNNIndex<N> &knn_index, SparseVec &new_vector, const uint new_index) {
    const uint tree_index = new_index / (1 << N);
    const uint offset = new_index % (1 << N);
    if (offset == 0) {
        knn_index.push_back(KNNIndexSingle<N>());
    }
    for (uint l = (1 << N) + offset; l > 0; l /= 2) {
        knn_index[tree_index][l] = add_sparse(knn_index[tree_index][l], new_vector);
    }
}


/**
 * @brief Node structure for KNN index
 * Holds value and dataset location
 * @param value Dot product value at this node
 * @param pool_index Index of the data pool
 * @param pool_offset Offset within the data pool
 */
struct KNNIndexNode {
    double value;
    uint pool_index;
    uint pool_offset;

    KNNIndexNode(double v, uint p_idx, uint p_off) : value(v), pool_index(p_idx), pool_offset(p_off) {}
    KNNIndexNode() : value(0), pool_index(0), pool_offset(0) {}
    ~KNNIndexNode() = default;

    bool operator==(const KNNIndexNode &other) const {
        return value == other.value && pool_index == other.pool_index && pool_offset == other.pool_offset;
    }
    std::partial_ordering operator<=>(const KNNIndexNode &other) const {
        if (auto cmp = value <=> other.value; cmp != 0) {
            return cmp;
        }
        if (auto cmp = pool_index <=> other.pool_index; cmp != 0) {
            return cmp;
        }
        return pool_offset <=> other.pool_offset;
    }
};

struct KNNDoubleIndexNode {
    double value;
    uint pool_idx;
    uint pool_offset;
    uint qpool_idx;
    uint qpool_offset;

    KNNDoubleIndexNode(double v, uint p_idx, uint p_off, uint q_idx, uint q_off) 
        : value(v), pool_idx(p_idx), pool_offset(p_off), qpool_idx(q_idx), qpool_offset(q_off) {}
    KNNDoubleIndexNode() : value(0), pool_idx(0), pool_offset(0), qpool_idx(0), qpool_offset(0) {}
    ~KNNDoubleIndexNode() = default;

    bool operator==(const KNNDoubleIndexNode &other) const {
        return value == other.value && pool_idx == other.pool_idx && pool_offset == other.pool_offset
            && qpool_idx == other.qpool_idx && qpool_offset == other.qpool_offset;
    }
    std::partial_ordering operator<=>(const KNNDoubleIndexNode &other) const {
        if (auto cmp = value <=> other.value; cmp != 0) {
            return cmp;
        }
        if (auto cmp = pool_idx <=> other.pool_idx; cmp != 0) {
            return cmp;
        }
        if (auto cmp = pool_offset <=> other.pool_offset; cmp != 0) {
            return cmp;
        }
        if (auto cmp = qpool_idx <=> other.qpool_idx; cmp != 0) {
            return cmp;
        }
        return qpool_offset <=> other.qpool_offset;
    }
};


/**
 * @brief Index structure for efficient KNN search on sparse data
 * Performs hierarchical binary partitioning for pruned KNN search
 */
class KNNIndexDataset {
public:
    /**
     * @brief Constructor - initializes with dataset and builds the index
     * @param dataset Sparse matrix reference to index
     * @param k Number of nearest neighbors to search for (default 1)
     * @param use_threading Enable multi-threaded search (default false)
     */
    KNNIndexDataset(SparseMat &dataset, size_t k = 1, bool use_threading = false);

    /**
     * @brief Update the index with a new sparse vector
     * @param new_vector New sparse vector to add
     */
    void update(SparseVec &new_vector);

    /**
     * @brief Update the index with a batch of new sparse vectors
     * @param new_vectors New sparse vectors to add
     */
    void update(SparseMat &new_vectors);
    
    /**
     * @brief Search for k-nearest neighbors to a query vector
     * @param query Query sparse vector
     * @return Pair of (result indices, number of dot products computed)
     */
    std::pair<std::vector<uint>, size_t> search(SparseVec &query);
    
    /**
     * @brief Parallel search using all system threads
     * @param query Query sparse vector
     * @return Pair of (result indices, number of dot products computed)
     */
    std::pair<std::vector<uint>, size_t> search_parallel(SparseVec &query);

    /**
     * @brief Parallel batch search using simple parallel loop (One thread per query)
     * @param queries Multiple query sparse vectors
     * @return Pair of (vector of result indices per query, total dot products computed)
     */
    std::pair<std::vector<std::vector<uint>>, size_t> search_parallel(SparseMat &queries);

    /**
     * @brief Search for k-nearest neighbors for multiple queries using double-group testing
     * @param queries Multiple query sparse vectors
     * @return Pair of (vector of result indices per query, total dot products computed)
     */
    std::pair<std::vector<std::vector<uint>>, size_t> search_double_group(SparseMat &queries);
    
    /**
     * @brief Verify search results and compute distance metrics
     * For KNN: recall == precision (returns both as same value)
     * @param query Query sparse vector
     * @param result Result indices from search
     * @return Array of [time_ms, recall, precision]
     */
    std::array<double, 3> verify_results(SparseVec &query, std::vector<uint> &result);

protected:
    SparseMat data_set;
    KNNIndex<KNN_LEVELS> data_index;
    size_t dimention;
    size_t k_val;
    bool use_threading;

    std::pair<std::vector<uint>, size_t> search_threshold(SparseVec &query, double threshold);
    std::pair<std::vector<uint>, size_t> search_pool(KNNIndexSingle<KNN_LEVELS> &pool, SparseVec &query, uint pool_index, double threshold);

    // Double-group testing for batch queries
    std::pair<std::vector<std::vector<uint>>, size_t> search_threshold_batch(const SparseMat &queries, double threshold);
    std::pair<KNNIndexDoubleGroupResultElem, size_t> search_pool_batch(
        KNNIndexSingle<KNN_LEVELS> &pool,
        KNNIndexSingle<INVERTED_LEVELS> &qpool,
        uint pool_index,
        double threshold
    );
};

#include "knn_search.hpp"
#include "knn_double_group.hpp"
#include "knn_verify.hpp"

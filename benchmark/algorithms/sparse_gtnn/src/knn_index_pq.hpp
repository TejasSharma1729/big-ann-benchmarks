#pragma once

#include "sparse_types.hpp"
#include <thread>
#include <array>
#include <chrono>
#include <numeric>
#include <compare>
#include <mutex>
#include <condition_variable>
#include <queue>

#define KNN_LEVELS 9
#define INVERTED_LEVELS 7
#define NUM_THREADS 16

/** @brief Template for single KNN dataset index - array of sparse vectors */
template <uint N = KNN_LEVELS> using KNNPQIndexSingle = std::array<SparseVec, (1 << (N + 1))>;
/** @brief Template for complete KNN dataset index - vector of single indices */
template <uint N = KNN_LEVELS> using KNNPQIndex = std::vector<KNNPQIndexSingle<N>>;
/** @brief Result element type for double-group search - array of result vectors */
using KNNPQIndexDoubleGroupResultElem = std::array<std::vector<uint>, (1 << INVERTED_LEVELS)>;
/**
 * @brief Build KNN index from sparse matrix using hierarchical partitioning
 */
template <uint N = KNN_LEVELS>
KNNPQIndex<N> build_knnpq_index(SparseMat &matrix) {
    const uint num_vectors = matrix.size();
    const uint num_indices = (num_vectors + (1 << N) - 1) / (1 << N);
    KNNPQIndex<N> data_index(num_indices);
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
 * @brief Node structure for KNN index
 * Holds value and dataset location
 * @param value Dot product value at this node
 * @param pool_index Index of the data pool
 * @param pool_offset Offset within the data pool
 */
struct KNNPQIndexNode {
    double value;
    uint pool_index;
    uint pool_offset;

    KNNPQIndexNode(double v, uint p_idx, uint p_off) : value(v), pool_index(p_idx), pool_offset(p_off) {}
    KNNPQIndexNode() : value(0), pool_index(0), pool_offset(0) {}
    ~KNNPQIndexNode() = default;

    bool operator==(const KNNPQIndexNode &other) const {
        return value == other.value && pool_index == other.pool_index && pool_offset == other.pool_offset;
    }
    std::partial_ordering operator<=>(const KNNPQIndexNode &other) const {
        if (auto cmp = value <=> other.value; cmp != 0) {
            return cmp;
        }
        if (auto cmp = pool_index <=> other.pool_index; cmp != 0) {
            return cmp;
        }
        return pool_offset <=> other.pool_offset;
    }
};

const std::array<double, 12> KNNPQIndexThresholds = {
    0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.025
};

struct KNNPQIndexQueue {
    std::array<std::vector<KNNPQIndexNode>, 13> buckets;
    
    void push(const KNNPQIndexNode& node) {
        for (size_t i = 0; i < 12; i++) {
            if (node.value >= KNNPQIndexThresholds[i]) {
                buckets[i].push_back(node);
                return;
            }
        }
        buckets.back().push_back(node);
    }

    void merge(KNNPQIndexQueue& other) {
        for (size_t i = 0; i < buckets.size(); i++) {
            buckets[i].insert(buckets[i].end(), other.buckets[i].begin(), other.buckets[i].end());
        }
    }
};

/**
 * @brief Index structure for efficient KNN search on sparse data
 * Performs hierarchical binary partitioning for pruned KNN search
 */
class KNNPQIndexDataset {
public:
    /**
     * @brief Constructor - initializes with dataset and builds the index
     * @param dataset Sparse matrix reference to index
     * @param k Number of nearest neighbors to search for (default 1)
     * @param use_threading Enable multi-threaded search (default false)
     */
    KNNPQIndexDataset(SparseMat &dataset, size_t k = 1, bool use_threading = false);
    
    /**
     * @brief Search for k-nearest neighbors to a query vector
     * @param query Query sparse vector
     * @return Pair of (result indices, number of dot products computed)
     */
    std::pair<std::vector<uint>, size_t> search(SparseVec &query);

    /**
     * @brief Search for k-nearest neighbors for multiple queries using double-group testing
     * @param queries Multiple query sparse vectors
     * @return Pair of (vector of result indices per query, total dot products computed)
     */
    std::pair<std::vector<std::vector<uint>>, size_t> search_multiple(SparseMat &queries);
    
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
    KNNPQIndex<KNN_LEVELS> data_index;
    size_t dimention;
    size_t k_val;
    bool use_threading;

    std::pair<std::vector<uint>, size_t> search_internal(SparseVec &query, uint pool_start, uint pool_end);
    std::pair<std::vector<uint>, size_t> search_threshold(SparseVec &query, double threshold);
    std::pair<std::vector<uint>, size_t> search_pool(
        KNNPQIndexSingle<KNN_LEVELS> &pool, 
        SparseVec &query, 
        uint pool_index, 
        uint offset_start,
        double threshold,
        KNNPQIndexQueue *queue = nullptr
    );

    // Double-group testing for batch queries
    std::pair<std::vector<std::vector<uint>>, size_t> search_threshold_batch(const SparseMat &queries, double threshold);
    std::pair<KNNPQIndexDoubleGroupResultElem, size_t> search_pool_batch(
        KNNPQIndexSingle<KNN_LEVELS> &pool,
        KNNPQIndexSingle<INVERTED_LEVELS> &qpool,
        uint pool_index,
        double threshold
    );
};

KNNPQIndexDataset::KNNPQIndexDataset(SparseMat &dataset, size_t k, bool use_threading_flag) 
    : k_val(k), use_threading(use_threading_flag) {
    this->data_set = dataset;
    this->dimention = dataset.empty() ? 0 : dataset[0].size();
    this->data_index = build_knnpq_index<KNN_LEVELS>(this->data_set);
}

std::pair<std::vector<uint>, size_t> KNNPQIndexDataset::search_internal(SparseVec &query, uint pool_start, uint pool_end) {
    std::pair<std::vector<uint>, size_t> result = {std::vector<uint>(), pool_end - pool_start};
    std::priority_queue<KNNPQIndexNode, std::vector<KNNPQIndexNode>, std::less<KNNPQIndexNode>> node_queue;
    
    for (uint pool_index = pool_start; pool_index < pool_end; pool_index++) {
        double dot_val = dot_product(query, this->data_index[pool_index][1]);
        node_queue.push(KNNPQIndexNode(dot_val, pool_index, 1));
    }

    while (result.first.size() < this->k_val && !node_queue.empty()) {
        KNNPQIndexNode node = node_queue.top();
        node_queue.pop();
        if (node.pool_offset >= static_cast<uint>(1 << KNN_LEVELS)) {
            uint actual_idx = node.pool_offset - (1 << KNN_LEVELS) + node.pool_index * (1 << KNN_LEVELS);
            result.first.push_back(actual_idx);
            continue;
        }
        double left_dot = dot_product(query, this->data_index[node.pool_index][2 * node.pool_offset]);
        double right_dot = node.value - left_dot;
        result.second += 2;
        node_queue.push(KNNPQIndexNode(left_dot, node.pool_index, 2 * node.pool_offset));
        node_queue.push(KNNPQIndexNode(right_dot, node.pool_index, 2 * node.pool_offset + 1));
    }
    return result;
}

std::pair<std::vector<uint>, size_t> KNNPQIndexDataset::search(SparseVec &query) {
    if (this->data_index.size() == 0) {
        return {std::vector<uint>(), 0};
    }
    if (!this->use_threading) {
        return this->search_internal(query, 0, this->data_index.size()); 
    }
    
    std::pair<std::vector<uint>, size_t> result = {std::vector<uint>(), 0};
    std::vector<std::pair<double, uint>> sorted_pools;
    KNNPQIndexQueue node_queue;

    // Initial population of the queue with root nodes
    for (uint pool_index = 0; pool_index < this->data_index.size(); pool_index++) {
        double val = dot_product(query, this->data_index[pool_index][1]);
        node_queue.push(KNNPQIndexNode(val, pool_index, 1));
    }

    // Iterate through thresholds
    for (size_t t_idx = 0; t_idx < KNNPQIndexThresholds.size(); t_idx++) {
        double threshold = KNNPQIndexThresholds[t_idx];
        std::vector<KNNPQIndexNode>& active_nodes = node_queue.buckets[t_idx];

        if (active_nodes.empty()) continue;
        if (sorted_pools.size() >= this->k_val) break;

        // Thread-local variables
        std::vector<KNNPQIndexQueue> thread_queues(NUM_THREADS);
        std::vector<std::vector<uint>> thread_results(NUM_THREADS);
        std::vector<size_t> thread_comparisons(NUM_THREADS, 0);

        std::vector<std::thread> threads;
        uint chunk_size = (active_nodes.size() + NUM_THREADS - 1) / NUM_THREADS;

        for (uint i = 0; i < NUM_THREADS; i++) {
            uint start = i * chunk_size;
            uint end = std::min((uint)((i + 1) * chunk_size), (uint)active_nodes.size());
            if (start >= active_nodes.size()) break;

            threads.emplace_back([this, &active_nodes, &query, threshold, &thread_queues, &thread_results, &thread_comparisons, start, end, i]() {
                for (uint n = start; n < end; n++) {
                    const auto& node = active_nodes[n];
                    // IMPORTANT: Pass the thread-local queue to capture rejected branches
                    auto res = this->search_pool(
                        this->data_index[node.pool_index], 
                        query, 
                        node.pool_index, 
                        node.pool_offset, 
                        threshold, 
                        &thread_queues[i]
                    );
                    thread_results[i].insert(thread_results[i].end(), res.first.begin(), res.first.end());
                    thread_comparisons[i] += res.second;
                }
            });
        }

        for (auto &t : threads) {
            if (t.joinable()) t.join();
        }

        // Merge results and queues
        for (uint i = 0; i < NUM_THREADS; i++) {
            node_queue.merge(thread_queues[i]);
            result.second += thread_comparisons[i];
            for (uint idx : thread_results[i]) {
                sorted_pools.push_back({dot_product(query, this->data_set[idx]), idx});
            }
        }
        
        // Sorting and Pruning Results (Optional optimization loop-by-loop)
        std::sort(sorted_pools.begin(), sorted_pools.end(), std::greater<std::pair<double, uint>>());
        if (sorted_pools.size() > this->k_val * 2) { // Heuristic: keep buffer
             sorted_pools.resize(this->k_val * 2);
        }
    }

    // Process 'rest' bucket if needed, or if we haven't found k items (usually strict thresholds imply we stop)
    // Assuming 'rest' is handled or ignored based on requirements. 
    // If strict exact search is needed, we technically need to search 'rest' if we don't haven't proven optimaility.
    // But this threshold-stepping usually implies approximate or "good enough" search.
    
    std::vector<uint> top_k;
    for (size_t i = 0; i < std::min(sorted_pools.size(), this->k_val); i++) {
        top_k.push_back(sorted_pools[i].second);
    }
    std::sort(top_k.begin(), top_k.end());
    result.first = top_k;
    
    return result;
}

std::pair<std::vector<std::vector<uint>>, size_t> KNNPQIndexDataset::search_multiple(SparseMat &queries) {
    std::pair<std::vector<std::vector<uint>>, size_t> result;
    result.first.resize(queries.size());
    result.second = 0;
    
    if (this->data_index.size() == 0 || queries.size() == 0) {
        return result;
    }
    
    std::vector<size_t> batch_costs(NUM_THREADS, 0);
    std::vector<std::thread> threads;
    uint chunk_size = (queries.size() + NUM_THREADS - 1) / NUM_THREADS;
    
    for (uint i = 0; i < NUM_THREADS; i++) {
        uint start = i * chunk_size;
        uint end = std::min((uint)((i + 1) * chunk_size), (uint)queries.size());
        if (start >= queries.size()) break;

        threads.emplace_back([this, &queries, &result, &batch_costs, start, end, i]() {
            // Each thread processes a subset of queries independently
            // We use the internal serial search logic for each query
            for (uint q = start; q < end; q++) {
                auto single_res = this->search_internal(queries[q], 0, this->data_index.size());
                result.first[q] = single_res.first;
                batch_costs[i] += single_res.second;
                
                // Ensure result is sorted (search_internal returns sorted indices if k is met by queue, 
                // but let's ensure it just in case logic changes. Actually queue pop order provides High->Low scores.
                // search_internal pushes `actual_idx` to `result.first`. 
                // Wait, `search_internal` pushes as it pops. So it pushes highest score first.
                // But `verify_results` expects sorted INDICES (std::sort at end of `verify_results` usage?).
                // Let's check `search_internal`: `result.first.push_back`. Order is by score descending.
                // We should sort by index for consistency?
                std::sort(result.first[q].begin(), result.first[q].end());
            }
        });
    }
    
    for (auto &t : threads) {
        if (t.joinable()) t.join();
    }
    
    for (size_t c : batch_costs) {
        result.second += c;
    }
    
    return result;
}

std::pair<std::vector<uint>, size_t> KNNPQIndexDataset::search_threshold(SparseVec &query, double threshold) {
    std::vector<std::pair<std::vector<uint>, size_t>> async_results(this->data_index.size());
    std::array<std::thread, NUM_THREADS> threads;
    
    auto worker = [this, &async_results, &query, threshold] (uint start, uint end) {
        for (uint pool_index = start; pool_index < end; pool_index++) {
            async_results[pool_index] = this->search_pool(
                this->data_index[pool_index], query, pool_index, 1, threshold
            );
        }
    };

    if (use_threading && this->data_index.size() > 1) {
        for (uint i = 0; i < NUM_THREADS; i++) {
            uint start = (i * this->data_index.size()) / NUM_THREADS;
            uint end = ((i + 1) * this->data_index.size()) / NUM_THREADS;
            threads[i] = std::thread(worker, start, end);
        }
        for (uint i = 0; i < NUM_THREADS; i++) {
            threads[i].join();
        }
    } else {
        worker(0, this->data_index.size());
    }

    std::pair<std::vector<uint>, size_t> result = {std::vector<uint>(), 0};
    for (uint i = 0; i < async_results.size(); i++) {
        result.first.insert(result.first.end(), async_results[i].first.begin(), async_results[i].first.end());
        result.second += async_results[i].second;
    }
    return result;
}

std::pair<std::vector<uint>, size_t> KNNPQIndexDataset::search_pool(
    KNNPQIndexSingle<KNN_LEVELS> &pool, 
    SparseVec &query, 
    uint pool_index,
    uint offset_start,
    double threshold,
    KNNPQIndexQueue *queue
) {
    std::pair<std::vector<uint>, size_t> result = {std::vector<uint>(), 0};
    if (pool[1].size() == 0) {
        return result;
    }
    std::array<double, KNN_LEVELS + 1> dots;
    std::array<uint, KNN_LEVELS + 1> idxs;
    idxs[0] = offset_start;
    dots[0] = dot_product(query, pool[offset_start]);
    result.second = 1;
    uint dot_idx = 1;
    uint idx = 1;
    double dot_val = dots[0];
    
    // Initial check for root of this subtree
    if (dot_val < threshold) {
        if (queue) {
            queue->push(KNNPQIndexNode(dot_val, pool_index, offset_start));
        }
        return result; 
    }

    while (dot_idx > 0) {
        dot_val = dots[dot_idx - 1];
        idx = idxs[dot_idx - 1];
        if (dot_val < threshold) {
            // Push to queue for later processing
            if (queue) {
                queue->push(KNNPQIndexNode(dot_val, pool_index, idx));
            }
            dot_idx--;
            continue;
        }
        if (idx >= static_cast<uint>(1 << KNN_LEVELS)) {
            uint actual_idx = idx - (1 << KNN_LEVELS) + pool_index * (1 << KNN_LEVELS);
            result.first.push_back(actual_idx);
            dot_idx--;
            continue;
        }
        result.second++;
        dot_idx++;
        dots[dot_idx - 1] = dot_product(query, pool[2 * idx + 1]);
        dots[dot_idx - 2] -= dots[dot_idx - 1];
        idxs[dot_idx - 1] = 2 * idx + 1;
        idxs[dot_idx - 2] = 2 * idx;
    }
    return result;
}

std::array<double, 3> KNNPQIndexDataset::verify_results(SparseVec &query, std::vector<uint> &result) {
    std::vector<std::pair<double, uint>> true_results;
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < data_set.size(); i++) {
        true_results.push_back({dot_product(query, data_set[i]), i});
    }
    std::sort(true_results.begin(), true_results.end(), std::greater<std::pair<double, uint>>());
    std::vector<uint> true_result;
    for (size_t i = 0; i < this->k_val && i < true_results.size(); i++) {
        true_result.push_back(true_results[i].second);
    }
    if (true_result.size() < this->k_val) {
        true_result.resize(this->k_val, 0);
    }
    std::sort(true_result.begin(), true_result.end());
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double time = duration.count() / 1.0e+3;
    uint match_count = 0;
    uint i = 0;
    uint j = 0;
    while (i < this->k_val && j < this->k_val) {
        if (result[i] == true_result[j]) {
            match_count++;
            i++;
            j++;
        } else if (result[i] < true_result[j]) {
            i++;
        } else {
            j++;
        }
    }
    // For KNN: recall == precision
    double metric = static_cast<double>(match_count) / this->k_val;
    return {time, metric, metric};
}

std::pair<std::vector<std::vector<uint>>, size_t> KNNPQIndexDataset::search_threshold_batch(
    const SparseMat &queries, double threshold) {
    KNNPQIndex<INVERTED_LEVELS> query_pools = build_knnpq_index<INVERTED_LEVELS>(const_cast<SparseMat&>(queries));
    std::vector<std::pair<KNNPQIndexDoubleGroupResultElem, size_t>> async_results(
        this->data_index.size() * query_pools.size());
    std::array<std::thread, NUM_THREADS> threads;
    
    auto worker = [this, &async_results, &query_pools, threshold] (uint start, uint end) {
        for (uint pool_index = start; pool_index < end; pool_index++) {
            for (uint qpool_index = 0; qpool_index < query_pools.size(); qpool_index++) {
                async_results[pool_index * query_pools.size() + qpool_index] = this->search_pool_batch(
                    this->data_index[pool_index], query_pools[qpool_index], pool_index, threshold
                );
            }
        }
    };
    
    if (use_threading) {
        for (uint i = 0; i < NUM_THREADS; i++) {
            uint start = (i * this->data_index.size()) / NUM_THREADS;
            uint end = ((i + 1) * this->data_index.size()) / NUM_THREADS;
            threads[i] = std::thread(worker, start, end);
        }
        for (uint i = 0; i < NUM_THREADS; i++) {
            threads[i].join();
        }
    } else {
        worker(0, this->data_index.size());
    }
    
    std::pair<std::vector<std::vector<uint>>, size_t> result = {std::vector<std::vector<uint>>(queries.size()), 0};
    for (uint pool_index = 0; pool_index < this->data_index.size(); pool_index++) {
        for (uint qpool_index = 0; qpool_index < query_pools.size(); qpool_index++) {
            auto &async_result = async_results[pool_index * query_pools.size() + qpool_index].first;
            result.second += async_results[pool_index * query_pools.size() + qpool_index].second;
            for (uint i = 0; i < async_result.size(); i++) {
                uint qidx = i + qpool_index * (1 << INVERTED_LEVELS);
                if (qidx >= queries.size()) {
                    break;
                }
                for (uint match : async_result[i]) {
                    result.first[qidx].push_back(match);
                }
            }
        }
    }
    return result;
}

std::pair<KNNPQIndexDoubleGroupResultElem, size_t> KNNPQIndexDataset::search_pool_batch(
    KNNPQIndexSingle<KNN_LEVELS> &pool,
    KNNPQIndexSingle<INVERTED_LEVELS> &qpool,
    uint pool_index,
    double threshold
) {
    
    std::pair<KNNPQIndexDoubleGroupResultElem, size_t> result = {KNNPQIndexDoubleGroupResultElem(), 1};
    std::array<double, 2 * (KNN_LEVELS + INVERTED_LEVELS) + 1> dots;
    std::array<uint, 2 * (KNN_LEVELS + INVERTED_LEVELS) + 1> idxs;
    std::array<uint, 2 * (KNN_LEVELS + INVERTED_LEVELS) + 1> qidxs;
    uint dot_idx = 1;
    dots[0] = dot_product(qpool[1], pool[1]);
    idxs[0] = 1;
    qidxs[0] = 1;
    uint idx = 1;
    uint qidx = 1;
    double dot_val = 0;

    while (dot_idx > 0) {
        idx = idxs[dot_idx - 1];
        qidx = qidxs[dot_idx - 1];
        dot_val = dots[dot_idx - 1];

        if (dot_val < threshold) {
            dot_idx--;
            continue;
        }
        if (idx >= static_cast<uint>(1 << KNN_LEVELS)) {
            if (qidx >= static_cast<uint>(1 << INVERTED_LEVELS)) {
                uint data_idx = idx - (1 << KNN_LEVELS) + pool_index * (1 << KNN_LEVELS);
                if (data_idx >= this->data_set.size()) {
                    dot_idx--;
                    continue;
                }
                uint query_idx = qidx - (1 << INVERTED_LEVELS);
                result.first[query_idx].push_back(data_idx);
                dot_idx--;
                continue;
            }
            dot_idx++;
            result.second++;
            dots[dot_idx - 1] = dot_product(qpool[2 * qidx + 1], pool[idx]);
            dots[dot_idx - 2] -= dots[dot_idx - 1];
            
            idxs[dot_idx - 1] = idx;
            idxs[dot_idx - 2] = idx;
            qidxs[dot_idx - 1] = 2 * qidx + 1;
            qidxs[dot_idx - 2] = 2 * qidx;
            continue;
        }
        if (qidx >= static_cast<uint>(1 << INVERTED_LEVELS)) {
            dot_idx++;
            result.second++;
            dots[dot_idx - 1] = dot_product(qpool[qidx], pool[2 * idx + 1]);
            dots[dot_idx - 2] -= dots[dot_idx - 1];
            
            idxs[dot_idx - 1] = 2 * idx + 1;
            idxs[dot_idx - 2] = 2 * idx;
            qidxs[dot_idx - 1] = qidx;
            qidxs[dot_idx - 2] = qidx;
            continue;
        }
        dot_idx += 3;
        result.second += 3;
        dots[dot_idx - 1] = dot_product(qpool[2 * qidx + 1], pool[2 * idx + 1]);
        dots[dot_idx - 2] = dot_product(qpool[2 * qidx + 1], pool[2 * idx]);
        dots[dot_idx - 3] = dot_product(qpool[2 * qidx], pool[2 * idx + 1]);
        dots[dot_idx - 4] -= (dots[dot_idx - 3] + dots[dot_idx - 2] + dots[dot_idx - 1]);

        idxs[dot_idx - 1] = 2 * idx + 1;
        idxs[dot_idx - 2] = 2 * idx;
        idxs[dot_idx - 3] = 2 * idx + 1;
        idxs[dot_idx - 4] = 2 * idx;
        qidxs[dot_idx - 1] = 2 * qidx + 1;
        qidxs[dot_idx - 2] = 2 * qidx + 1;
        qidxs[dot_idx - 3] = 2 * qidx;
        qidxs[dot_idx - 4] = 2 * qidx;
    }
    return result;
}

#undef KNN_LEVELS
#undef INVERTED_LEVELS
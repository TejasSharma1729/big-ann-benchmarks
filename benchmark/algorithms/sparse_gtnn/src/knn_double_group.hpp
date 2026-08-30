#pragma once

#include "knn_index.hpp"

// Inline helper function for processing a single query pool
// Marked always_inline to ensure no overhead
__attribute__((always_inline))
inline void process_double_group_qpool(
    const uint qpool_idx,
    const KNNIndex<INVERTED_LEVELS>& query_pools,
    const KNNIndex<KNN_LEVELS>& data_index,
    const SparseMat& queries,
    const size_t k_val,
    std::vector<std::vector<uint>>& results,
    size_t& local_dots
) {
    const uint start_query_idx = qpool_idx * (1 << INVERTED_LEVELS);
    const uint end_query_idx = std::min(
        (uint)queries.size(), 
        (uint)((qpool_idx + 1) * (1 << INVERTED_LEVELS))
    );
    const uint queries_in_this_pool = end_query_idx - start_query_idx;
    
    std::vector<bool> local_query_saturated(queries_in_this_pool, false);
    uint finished_queries_in_pool = 0;

    std::priority_queue<KNNDoubleIndexNode, 
                        std::vector<KNNDoubleIndexNode>, 
                        std::less<KNNDoubleIndexNode>
    > node_queue;

    // Initialize against all Data Pools
    for (uint pool_index = 0; pool_index < data_index.size(); pool_index++) {
        double dot_val = dot_product(query_pools[qpool_idx][1], data_index[pool_index][1]);
        node_queue.push(KNNDoubleIndexNode(dot_val, pool_index, 1, qpool_idx, 1));
    }
    local_dots += data_index.size();

    while (finished_queries_in_pool < queries_in_this_pool && !node_queue.empty()) {
        const KNNDoubleIndexNode node = node_queue.top();
        node_queue.pop();

        const bool d_is_leaf = node.pool_offset >= static_cast<uint>(1 << KNN_LEVELS);
        const bool q_is_leaf = node.qpool_offset >= static_cast<uint>(1 << INVERTED_LEVELS);

        if (d_is_leaf && q_is_leaf) {
            uint data_idx = node.pool_offset - (1 << KNN_LEVELS) + node.pool_idx * (1 << KNN_LEVELS);
            uint query_idx = node.qpool_offset - (1 << INVERTED_LEVELS) + node.qpool_idx * (1 << INVERTED_LEVELS);

            if (query_idx < queries.size()) {
                uint local_q_idx = query_idx - start_query_idx;
                if (!local_query_saturated[local_q_idx]) {
                    results[query_idx].push_back(data_idx);
                    if (results[query_idx].size() >= k_val) {
                        local_query_saturated[local_q_idx] = true;
                        finished_queries_in_pool++;
                    }
                }
            }
            continue;
        }

        if (d_is_leaf) {
            double right_dot = dot_product(
                query_pools[node.qpool_idx][2 * node.qpool_offset + 1], 
                data_index[node.pool_idx][node.pool_offset]
            );
            double left_dot = node.value - right_dot;
            local_dots++;
            
            node_queue.push(KNNDoubleIndexNode(
                left_dot, 
                node.pool_idx, node.pool_offset, 
                node.qpool_idx, 2 * node.qpool_offset
            ));
            node_queue.push(KNNDoubleIndexNode(
                right_dot, 
                node.pool_idx, node.pool_offset, 
                node.qpool_idx, 2 * node.qpool_offset + 1
            ));
        } 
        else if (q_is_leaf) {
            if (local_query_saturated[node.qpool_offset - (1 << INVERTED_LEVELS)]) {
                continue;
            }
            
            double right_dot = dot_product(
                query_pools[node.qpool_idx][node.qpool_offset], 
                data_index[node.pool_idx][2 * node.pool_offset + 1]
            );
            double left_dot = node.value - right_dot;
            local_dots++;
            
            node_queue.push(KNNDoubleIndexNode(
                left_dot, 
                node.pool_idx, 2 * node.pool_offset, 
                node.qpool_idx, node.qpool_offset
            ));
            node_queue.push(KNNDoubleIndexNode(
                right_dot, 
                node.pool_idx, 2 * node.pool_offset + 1, 
                node.qpool_idx, node.qpool_offset
            ));
        } 
        else {
            double qR_dR = dot_product(
                query_pools[node.qpool_idx][2 * node.qpool_offset + 1], 
                data_index[node.pool_idx][2 * node.pool_offset + 1]
            );
            double qR_dL = dot_product(
                query_pools[node.qpool_idx][2 * node.qpool_offset + 1], 
                data_index[node.pool_idx][2 * node.pool_offset]
            );
            double qL_dR = dot_product(
                query_pools[node.qpool_idx][2 * node.qpool_offset], 
                data_index[node.pool_idx][2 * node.pool_offset + 1]
            );
            double qL_dL = node.value - (qR_dR + qR_dL + qL_dR);
            local_dots += 3;
            
            node_queue.push(KNNDoubleIndexNode(
                qL_dL, 
                node.pool_idx, 2 * node.pool_offset, 
                node.qpool_idx, 2 * node.qpool_offset
            ));
            node_queue.push(KNNDoubleIndexNode(
                qL_dR, 
                node.pool_idx, 2 * node.pool_offset + 1, 
                node.qpool_idx, 2 * node.qpool_offset
            ));
            node_queue.push(KNNDoubleIndexNode(
                qR_dL, 
                node.pool_idx, 2 * node.pool_offset, 
                node.qpool_idx, 2 * node.qpool_offset + 1
            ));
            node_queue.push(KNNDoubleIndexNode(
                qR_dR, 
                node.pool_idx, 2 * node.pool_offset + 1, 
                node.qpool_idx, 2 * node.qpool_offset + 1
            ));
        }
    }
}


std::pair<std::vector<std::vector<uint>>, size_t> KNNIndexDataset::search_double_group(
    SparseMat &queries
) {
    if (this->data_index.empty() || queries.empty()) {
        return {{}, 0};
    }

    // Build Query Index Hierarchy
    KNNIndex<INVERTED_LEVELS> query_pools = build_knn_index<INVERTED_LEVELS>(queries);
    std::vector<std::vector<uint>> results(queries.size());

    if (!this->use_threading) {
        size_t total_dots = 0;
        for (uint qpool_idx = 0; qpool_idx < query_pools.size(); qpool_idx++) {
            process_double_group_qpool(
                qpool_idx,
                query_pools,
                this->data_index,
                queries,
                this->k_val,
                results,
                total_dots
            );
        }
        return {results, total_dots};
    }

    std::atomic<size_t> total_dots = 0;
    #pragma omp parallel for
    for (uint qpool_idx = 0; qpool_idx < query_pools.size(); qpool_idx++) {
        size_t local_dots = 0;
        process_double_group_qpool(
            qpool_idx,
            query_pools,
            this->data_index,
            queries,
            this->k_val,
            results,
            local_dots
        );
        total_dots += local_dots;
    }
    return {results, total_dots.load()};
}


// ------------------------------------------------------------------
// Legacy/Other helper methods below
// ------------------------------------------------------------------

std::pair<KNNIndexDoubleGroupResultElem, size_t> KNNIndexDataset::search_pool_batch(
    KNNIndexSingle<KNN_LEVELS> &pool,
    KNNIndexSingle<INVERTED_LEVELS> &qpool,
    uint pool_index,
    double threshold
) {
    std::pair<KNNIndexDoubleGroupResultElem, size_t> result = {KNNIndexDoubleGroupResultElem(), 1};
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

std::pair<std::vector<std::vector<uint>>, size_t> KNNIndexDataset::search_threshold_batch(
    const SparseMat &queries, 
    double threshold
) {
    KNNIndex<INVERTED_LEVELS> query_pools = build_knn_index<INVERTED_LEVELS>(const_cast<SparseMat&>(queries));
    std::vector<std::pair<KNNIndexDoubleGroupResultElem, size_t>> async_results(
        this->data_index.size() * query_pools.size());
    std::vector<std::thread> threads;
    
    // Auto-detect threads
    uint num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;

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
        uint step = this->data_index.size() / num_threads; 
        if(step == 0) step = 1;

        for (uint i = 0; i < num_threads && i * step < this->data_index.size(); i++) {
            uint start = i * step; 
            uint end = (i == num_threads - 1) ? this->data_index.size() : (i + 1) * step;
            threads.emplace_back(worker, start, end);
        }
        for (auto &t : threads) t.join();
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

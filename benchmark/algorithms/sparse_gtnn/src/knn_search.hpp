#pragma once

#include "knn_index.hpp"
#include <thread>
#include <mutex>
#include <future>
#include <atomic>
#include <algorithm>

KNNIndexDataset::KNNIndexDataset(SparseMat &dataset, size_t k, bool use_threading_flag) 
    : k_val(k), use_threading(use_threading_flag) {
    this->data_set = dataset;
    this->dimention = dataset.empty() ? 0 : dataset[0].size();
    this->data_index = build_knn_index<KNN_LEVELS>(this->data_set);
}

// Single-threaded priority-queue based search
std::pair<std::vector<uint>, size_t> KNNIndexDataset::search(SparseVec &query) {
    if (this->data_index.size() == 0) {
        return {std::vector<uint>(), 0};
    }
    std::pair<std::vector<uint>, size_t> result = {std::vector<uint>(), this->data_index.size()};
    std::priority_queue<KNNIndexNode, std::vector<KNNIndexNode>, std::less<KNNIndexNode>> node_queue;
    
    for (uint pool_index = 0; pool_index < this->data_index.size(); pool_index++) {
        double dot_val = dot_product(query, this->data_index[pool_index][1]);
        node_queue.push(KNNIndexNode(dot_val, pool_index, 1));
    }

    while (result.first.size() < this->k_val && !node_queue.empty()) {
        KNNIndexNode node = node_queue.top();
        node_queue.pop();
        if (node.pool_offset >= static_cast<uint>(1 << KNN_LEVELS)) {
            uint actual_idx = node.pool_offset - (1 << KNN_LEVELS) + node.pool_index * (1 << KNN_LEVELS);
            result.first.push_back(actual_idx);
            continue;
        }
        double left_dot = dot_product(query, this->data_index[node.pool_index][2 * node.pool_offset]);
        double right_dot = node.value - left_dot;
        result.second += 2;
        node_queue.push(KNNIndexNode(left_dot, node.pool_index, 2 * node.pool_offset));
        node_queue.push(KNNIndexNode(right_dot, node.pool_index, 2 * node.pool_offset + 1));
    }
    return result;
}

// Multi-threaded search using all available threads
std::pair<std::vector<uint>, size_t> KNNIndexDataset::search_parallel(SparseVec &query) {
    if (this->data_index.size() == 0) {
        return {{}, 0};
    }
    uint num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;

    auto worker = [&](
        uint start, 
        uint end, 
        std::vector<KNNIndexNode> &local_results, 
        size_t &local_comparisons
    ) {
        std::priority_queue<KNNIndexNode, std::vector<KNNIndexNode>, std::less<KNNIndexNode>> local_queue;
        local_comparisons = 0;

        // Initial population
        for (uint pool_index = start; pool_index < end; pool_index++) {
            if (pool_index >= this->data_index.size()) break;
            double dot_val = dot_product(query, this->data_index[pool_index][1]);
            local_queue.push(KNNIndexNode(dot_val, pool_index, 1));
            local_comparisons++;
        }
        
        while (local_results.size() < this->k_val && !local_queue.empty()) {
            KNNIndexNode node = local_queue.top();
            local_queue.pop();
            
            if (node.pool_offset >= static_cast<uint>(1 << KNN_LEVELS)) {
                local_results.push_back(node); // Storing offset temporarily
                continue;
            }

            double left_dot = dot_product(query, this->data_index[node.pool_index][2 * node.pool_offset]);
            double right_dot = node.value - left_dot;
            local_comparisons += 2;
            local_queue.push(KNNIndexNode(left_dot, node.pool_index, 2 * node.pool_offset));
            local_queue.push(KNNIndexNode(right_dot, node.pool_index, 2 * node.pool_offset + 1));
        }
        return;
    };
    
    std::pair<std::vector<uint>, size_t> result = {std::vector<uint>(), 0};
    
    // Parallel Initialization
    std::vector<std::thread> threads;
    std::vector<std::vector<KNNIndexNode>> thread_results(num_threads);
    std::vector<size_t> thread_comparisons(num_threads, 0);
    uint batch_size = (this->data_index.size() + num_threads - 1) / num_threads;

    for (uint i = 0; i < num_threads; ++i) {
        uint start = i * batch_size;
        uint end = std::min(start + batch_size, (uint)this->data_index.size());
        if (start < end) {
            threads.emplace_back(worker, start, end, std::ref(thread_results[i]), std::ref(thread_comparisons[i]));
        }
    }
    for (auto &t : threads) {
        if (t.joinable()) t.join();
    }

    // Merge results
    std::vector<std::pair<double, uint>> sorted_pools;
    for (uint i = 0; i < num_threads; i++) {
        result.second += thread_comparisons[i];
        for (const auto &node : thread_results[i]) {
            sorted_pools.push_back(std::make_pair(
                node.value, 
                node.pool_index * (1 << KNN_LEVELS) + node.pool_offset - (1 << KNN_LEVELS)
            ));
        }
    }
    std::sort(sorted_pools.begin(), sorted_pools.end(), std::greater<std::pair<double, uint>>());

    for (size_t i = 0; i < this->k_val; i++) {
        result.first.push_back(sorted_pools[i].second);
    }
    return result;
}

std::pair<std::vector<uint>, size_t> KNNIndexDataset::search_pool(KNNIndexSingle<KNN_LEVELS> &pool, SparseVec &query, uint pool_index, double threshold) {
    std::pair<std::vector<uint>, size_t> result = {std::vector<uint>(), 0};
    if (pool[1].size() == 0) {
        return result;
    }
    std::array<double, KNN_LEVELS + 1> dots;
    std::array<uint, KNN_LEVELS + 1> idxs;
    idxs[0] = 1;
    dots[0] = dot_product(query, pool[1]);
    result.second = 1;
    uint dot_idx = 1;
    uint idx = 1;
    double dot_val = dots[0];
    while (dot_idx > 0) {
        dot_val = dots[dot_idx - 1];
        idx = idxs[dot_idx - 1];
        if (dot_val < threshold) {
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

std::pair<std::vector<uint>, size_t> KNNIndexDataset::search_threshold(SparseVec &query, double threshold) {
    std::vector<std::pair<std::vector<uint>, size_t>> async_results(this->data_index.size());
    std::vector<std::thread> threads;
    uint num_threads = std::thread::hardware_concurrency();
    if(num_threads == 0) num_threads=4;

    auto worker = [this, &async_results, &query, threshold] (uint start, uint end) {
        for (uint pool_index = start; pool_index < end; pool_index++) {
            async_results[pool_index] = this->search_pool(this->data_index[pool_index], query, pool_index, threshold);
        }
    };

    if (this->use_threading && this->data_index.size() > 1) {
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

    std::pair<std::vector<uint>, size_t> result = {std::vector<uint>(), 0};
    for (uint i = 0; i < async_results.size(); i++) {
        result.first.insert(result.first.end(), async_results[i].first.begin(), async_results[i].first.end());
        result.second += async_results[i].second;
    }
    return result;
}

std::pair<std::vector<std::vector<uint>>, size_t> KNNIndexDataset::search_parallel(SparseMat &queries) {
    std::pair<std::vector<std::vector<uint>>, size_t> result;
    result.first.resize(queries.size());
    std::atomic<size_t> total_comps(0);
    std::atomic<uint> next_query_idx(0); // Atomic counter for dynamic scheduling

    uint num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    std::vector<std::thread> threads;

    auto worker = [&]() {
        size_t local_comps = 0;
        while (true) {
            uint i = next_query_idx.fetch_add(1, std::memory_order_relaxed);
            if (i >= queries.size()) break;

            auto [res, comps] = this->search(queries[i]);
            result.first[i] = std::move(res);
            local_comps += comps;
        }
        total_comps += local_comps;
    };

    for (uint i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }

    for (auto &t : threads) {
        if (t.joinable()) t.join();
    }

    result.second = total_comps.load();
    return result;
}

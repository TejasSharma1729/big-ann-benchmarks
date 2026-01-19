#pragma once

#include "knn_index.hpp"
#include <chrono>


void KNNIndexDataset::update(SparseVec &new_vector) {
    update_knn_index(this->data_index, new_vector, this->data_set.size());
    this->data_set.push_back(new_vector);
}

void KNNIndexDataset::update(SparseMat &new_vectors) {
    for (size_t i = 0; i < new_vectors.size(); i++) {
        this->update(new_vectors[i]);
    }
}


std::array<double, 3> KNNIndexDataset::verify_results(SparseVec &query, std::vector<uint> &result) {
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
    
    // Create a sorted copy of the result for intersection
    std::vector<uint> sorted_result = result;
    std::sort(sorted_result.begin(), sorted_result.end());

    uint match_count = 0;
    uint i = 0;
    uint j = 0;
    while (i < sorted_result.size() && j < true_result.size()) {
        if (sorted_result[i] == true_result[j]) {
            match_count++;
            i++;
            j++;
        } else if (sorted_result[i] < true_result[j]) {
            i++;
        } else {
            j++;
        }
    }
    // For KNN: recall == precision
    double metric = static_cast<double>(match_count) / this->k_val;
    return {time, metric, metric};
}

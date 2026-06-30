#pragma once
#include "indexed_set.hpp"
#include <vector>
#include <cassert>

// Bucket array over the gain range [-max_gain, +max_gain].
// In the FM convention used here: more-negative gain = more improvement.
// best_idx tracks the lowest non-empty bucket (best available action).
// All operations keep best_idx consistent via a small upward scan on removal.
struct BucketArray {
    std::vector<IndexedSet> buckets; // bucket[gain + max_gain]
    int32_t max_gain;
    int32_t num_actions;
    int32_t best_idx; // lowest non-empty bucket index; -1 when all empty

    BucketArray() : max_gain(0), num_actions(0), best_idx(-1) {}

    void init(int32_t max_gain_, int32_t num_actions_) {
        max_gain    = max_gain_;
        num_actions = num_actions_;
        best_idx    = -1;
        int32_t n   = 2 * max_gain + 1;
        buckets.resize(n);
        for (auto& b : buckets) b.reset(num_actions);
    }

    void clear() {
        best_idx = -1;
        for (auto& b : buckets) {
            for (int32_t i = 0; i < b.size(); ) {
                int32_t a = b.items[0];
                b.remove(a);
            }
        }
    }

    int32_t gain_to_idx(int32_t gain) const { return gain + max_gain; }
    int32_t idx_to_gain(int32_t idx)  const { return idx  - max_gain; }
    int32_t nbuckets()                const { return static_cast<int32_t>(buckets.size()); }

    void insert(int32_t action, int32_t gain) {
        int32_t idx = gain_to_idx(gain);
        buckets[idx].insert(action);
        if (best_idx == -1 || idx < best_idx) best_idx = idx;
    }

    void remove(int32_t action, int32_t gain) {
        int32_t idx = gain_to_idx(gain);
        buckets[idx].remove(action);
        if (idx == best_idx && buckets[idx].empty()) _repair_best(idx);
    }

    // Move action from old_gain to new_gain.
    void move(int32_t action, int32_t old_gain, int32_t new_gain) {
        int32_t old_idx = gain_to_idx(old_gain);
        int32_t new_idx = gain_to_idx(new_gain);
        buckets[old_idx].remove(action);
        buckets[new_idx].insert(action);
        // best_idx can only improve if new_idx < best_idx
        if (new_idx < best_idx || best_idx == -1) {
            best_idx = new_idx;
        } else if (old_idx == best_idx && buckets[old_idx].empty()) {
            // Need to scan upward from old_idx; new_idx is the other candidate
            _repair_best(old_idx);
            // new_idx is already non-empty; take whichever is lower
            if (best_idx == -1 || new_idx < best_idx) best_idx = new_idx;
        }
    }

    bool empty() const { return best_idx == -1; }

    // Returns (action_id, gain). Caller must still validate (lock, spaces).
    std::pair<int32_t, int32_t> best_pick(std::mt19937& rng) const {
        assert(!empty());
        int32_t action = buckets[best_idx].random_pick(rng);
        return {action, idx_to_gain(best_idx)};
    }

    bool action_in_bucket(int32_t action, int32_t gain) const {
        return buckets[gain_to_idx(gain)].contains(action);
    }

private:
    void _repair_best(int32_t from_idx) {
        best_idx = -1;
        for (int32_t i = from_idx + 1; i < nbuckets(); ++i) {
            if (!buckets[i].empty()) { best_idx = i; return; }
        }
    }
};

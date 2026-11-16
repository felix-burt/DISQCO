#pragma once
#include <vector>
#include <cstdint>
#include <random>
#include <cassert>
#include <algorithm>

// Bucket array over the gain range [-max_gain, +max_gain].
// In the FM convention used here: more-negative gain = more improvement.
// best_idx tracks the lowest non-empty bucket (best available action).
// All operations keep best_idx consistent via a small upward scan on removal.
//
// Each action lives in at most one bucket, so a single shared position map
// (pos / bucket_of, both sized num_actions) serves every bucket.  Memory is
// O(num_actions + max_gain) rather than O(num_actions * max_gain).
//
// Gains outside [-max_gain, +max_gain] are clamped into the end buckets; the
// true gain is kept by the caller (FMState::gains), so ordering is only
// approximate for out-of-range actions but memory stays in bounds.
struct Bucket {
    std::vector<int32_t> items;

    bool empty()    const { return items.empty(); }
    int32_t size()  const { return static_cast<int32_t>(items.size()); }
};

struct BucketArray {
    std::vector<Bucket>  buckets;    // bucket[gain + max_gain]
    std::vector<int32_t> pos;        // pos[a] = index in its bucket's items
    std::vector<int32_t> bucket_of;  // bucket_of[a] = bucket index, or -1 if absent
    int32_t max_gain;
    int32_t num_actions;
    int32_t best_idx; // lowest non-empty bucket index; -1 when all empty

    BucketArray() : max_gain(0), num_actions(0), best_idx(-1) {}

    void init(int32_t max_gain_, int32_t num_actions_) {
        max_gain    = max_gain_;
        num_actions = num_actions_;
        best_idx    = -1;
        buckets.assign(2 * max_gain + 1, Bucket{});
        pos.assign(num_actions, -1);
        bucket_of.assign(num_actions, -1);
    }

    // O(number of items currently stored), not O(num_actions).
    void clear() {
        for (auto& b : buckets) {
            for (int32_t a : b.items) { pos[a] = -1; bucket_of[a] = -1; }
            b.items.clear();
        }
        best_idx = -1;
    }

    int32_t gain_to_idx(int32_t gain) const {
        return std::clamp(gain, -max_gain, max_gain) + max_gain;
    }
    int32_t idx_to_gain(int32_t idx)  const { return idx  - max_gain; }
    int32_t nbuckets()                const { return static_cast<int32_t>(buckets.size()); }

    void insert(int32_t action, int32_t gain) {
        if (bucket_of[action] != -1) return;
        int32_t idx = gain_to_idx(gain);
        _push(action, idx);
        if (best_idx == -1 || idx < best_idx) best_idx = idx;
    }

    void remove(int32_t action, int32_t /*gain*/) {
        int32_t idx = bucket_of[action];
        if (idx == -1) return;
        _erase(action, idx);
        if (idx == best_idx && buckets[idx].empty()) _repair_best(idx);
    }

    // Move action from its current bucket to the one for new_gain.
    void move(int32_t action, int32_t /*old_gain*/, int32_t new_gain) {
        int32_t old_idx = bucket_of[action];
        int32_t new_idx = gain_to_idx(new_gain);
        if (old_idx == new_idx) return;
        if (old_idx != -1) _erase(action, old_idx);
        _push(action, new_idx);
        // best_idx can only improve if new_idx < best_idx
        if (new_idx < best_idx || best_idx == -1) {
            best_idx = new_idx;
        } else if (old_idx == best_idx && buckets[old_idx].empty()) {
            // Need to scan upward from old_idx; new_idx is the other candidate
            _repair_best(old_idx);
            if (best_idx == -1 || new_idx < best_idx) best_idx = new_idx;
        }
    }

    bool empty() const { return best_idx == -1; }

    // Returns (action_id, gain). Caller must still validate (lock, spaces).
    std::pair<int32_t, int32_t> best_pick(std::mt19937& rng) const {
        assert(!empty());
        const auto& items = buckets[best_idx].items;
        std::uniform_int_distribution<int32_t> dist(0, static_cast<int32_t>(items.size()) - 1);
        return {items[dist(rng)], idx_to_gain(best_idx)};
    }

    bool contains(int32_t action) const { return bucket_of[action] != -1; }

    bool action_in_bucket(int32_t action, int32_t /*gain*/) const {
        return contains(action);
    }

private:
    void _push(int32_t action, int32_t idx) {
        auto& items = buckets[idx].items;
        pos[action]       = static_cast<int32_t>(items.size());
        bucket_of[action] = idx;
        items.push_back(action);
    }

    void _erase(int32_t action, int32_t idx) {
        auto& items = buckets[idx].items;
        int32_t i    = pos[action];
        int32_t last = items.back();
        items[i]  = last;
        pos[last] = i;
        items.pop_back();
        pos[action]       = -1;
        bucket_of[action] = -1;
    }

    void _repair_best(int32_t from_idx) {
        best_idx = -1;
        for (int32_t i = from_idx + 1; i < nbuckets(); ++i) {
            if (!buckets[i].empty()) { best_idx = i; return; }
        }
    }
};

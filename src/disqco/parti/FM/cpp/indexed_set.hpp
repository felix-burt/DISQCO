#pragma once
#include <vector>
#include <cstdint>
#include <random>
#include <stdexcept>

// O(1) insert, remove, contains, and random-element-pick.
// Internally keeps a flat array of items alongside a position map.
// Remove works by swapping the target with the last item then popping.
// capacity must be an upper bound on any action_id ever inserted.
struct IndexedSet {
    std::vector<int32_t> items;
    std::vector<int32_t> pos;   // pos[a] = index in items, or -1 if absent

    explicit IndexedSet() = default;
    explicit IndexedSet(int32_t capacity) : pos(capacity, -1) {}

    void reset(int32_t capacity) {
        items.clear();
        pos.assign(capacity, -1);
    }

    void insert(int32_t a) {
        if (pos[a] != -1) return;
        pos[a] = static_cast<int32_t>(items.size());
        items.push_back(a);
    }

    void remove(int32_t a) {
        int32_t i = pos[a];
        if (i == -1) return;
        int32_t last = items.back();
        items[i] = last;
        pos[last] = i;
        items.pop_back();
        pos[a] = -1;
    }

    bool contains(int32_t a) const { return pos[a] != -1; }
    bool empty()             const { return items.empty(); }
    int32_t size()           const { return static_cast<int32_t>(items.size()); }

    int32_t random_pick(std::mt19937& rng) const {
        std::uniform_int_distribution<int32_t> dist(0, size() - 1);
        return items[dist(rng)];
    }
};

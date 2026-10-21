#ifndef SCHEDULER_HPP
#define SCHEDULER_HPP

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "nn/nn-quants.hpp"

// Iteration-level scheduling for dllama-api. Before every forward the policy picks which of the
// active requests (the ones holding a KV-cache slot) contribute rows to the next batch.
// A prefill row is one prompt token, a decode row is the request's next token.

enum SchedPhase {
    SCHED_PREFILL,
    SCHED_DECODE,
};

struct SchedRequest {
    NnUint id;
    SchedPhase phase;
    NnUint nPrefillLeft; // prompt tokens still to be written to the KV cache (all but the last one)
};

struct SchedItem {
    SchedRequest *request;
    NnUint nTokens;
};

// Rows a request can take in one forward: a prefill chunk or one decode token
static inline NnUint schedRequestRows(const SchedRequest *request, NnUint nRows) {
    if (request->phase == SCHED_PREFILL)
        return std::min(request->nPrefillLeft, nRows);
    return 1;
}

class SchedPolicy {
public:
    virtual ~SchedPolicy() {}
    virtual const char *name() = 0;
    // `active` is ordered by admission and is never empty; at most nRows tokens in total
    virtual void pick(const std::vector<SchedRequest *> &active, NnUint nRows, std::vector<SchedItem> &items) = 0;
    // Called before the request at `index` is removed from the active list
    virtual void onRemove(size_t index) {}
};

// Request-level FCFS: the oldest request runs until it finishes (dllama's native behavior)
class SerialPolicy : public SchedPolicy {
public:
    const char *name() { return "serial"; }
    void pick(const std::vector<SchedRequest *> &active, NnUint nRows, std::vector<SchedItem> &items) {
        SchedRequest *request = active[0];
        items.push_back(SchedItem{request, schedRequestRows(request, nRows)});
    }
};

// Each request in turn gets `quantum` consecutive forwards (a prefill chunk counts as one),
// then the next active request runs; new requests join at the end of the rotation.
class RoundRobinPolicy : public SchedPolicy {
private:
    NnUint quantum;
    size_t cursor;
    NnUint used;
public:
    RoundRobinPolicy(NnUint quantum) {
        this->quantum = quantum;
        this->cursor = 0;
        this->used = 0;
    }
    const char *name() { return "rr"; }
    void pick(const std::vector<SchedRequest *> &active, NnUint nRows, std::vector<SchedItem> &items) {
        if (used >= quantum) {
            cursor++;
            used = 0;
        }
        if (cursor >= active.size())
            cursor = 0;
        SchedRequest *request = active[cursor];
        items.push_back(SchedItem{request, schedRequestRows(request, nRows)});
        used++;
    }
    void onRemove(size_t index) {
        if (index < cursor) {
            cursor--;
        } else if (index == cursor) {
            // The next request moves into the cursor's place and starts a fresh quantum
            used = 0;
        }
    }
};

// All active requests share every forward: each decoding request gets one row, the remaining
// rows go to prefilling requests as chunks, oldest first (decode-first, stall-free batching).
class ConcurrentPolicy : public SchedPolicy {
public:
    const char *name() { return "concurrent"; }
    void pick(const std::vector<SchedRequest *> &active, NnUint nRows, std::vector<SchedItem> &items) {
        NnUint left = nRows;
        for (size_t i = 0; i < active.size() && left > 0; i++) {
            if (active[i]->phase == SCHED_DECODE) {
                items.push_back(SchedItem{active[i], 1});
                left--;
            }
        }
        for (size_t i = 0; i < active.size() && left > 0; i++) {
            if (active[i]->phase == SCHED_PREFILL) {
                const NnUint n = schedRequestRows(active[i], left);
                items.push_back(SchedItem{active[i], n});
                left -= n;
            }
        }
    }
};

static inline SchedPolicy *createSchedPolicy(const char *name, NnUint rrQuantum) {
    if (std::strcmp(name, "serial") == 0)
        return new SerialPolicy();
    if (std::strcmp(name, "rr") == 0)
        return new RoundRobinPolicy(rrQuantum);
    if (std::strcmp(name, "concurrent") == 0)
        return new ConcurrentPolicy();
    throw std::runtime_error("Unknown scheduling policy: " + std::string(name));
}

#endif

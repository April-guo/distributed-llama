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
// With pipeline parallelism several forwards (microbatches) are in flight, and a request is
// `ready` only when it can take part in one more: a decoding request has to wait for the
// token of its previous forward. Without a pipeline every active request is ready.

enum SchedPhase {
    SCHED_PREFILL,
    SCHED_DECODE,
};

struct SchedRequest {
    NnUint id;
    SchedPhase phase;
    NnUint nPrefillLeft; // prompt tokens still to be sent to the KV cache (all but the last one)
    bool ready;
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
    // `active` is ordered by admission and is never empty; at most nRows tokens in total, only of
    // ready requests. Picking nothing makes the server wait for a microbatch in flight.
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
        if (request->ready)
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
        if (!request->ready)
            return; // the turn stays with this request
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

// All ready requests share every forward: each decoding request gets one row, the remaining
// rows go to prefilling requests as chunks, oldest first (decode-first, stall-free batching).
// maxRequests > 0 caps the requests per forward, so a pipeline gets several smaller microbatches
// instead of one that leaves the other stages idle.
class ConcurrentPolicy : public SchedPolicy {
private:
    NnUint maxRequests;
public:
    ConcurrentPolicy(NnUint maxRequests) {
        this->maxRequests = maxRequests;
    }
    const char *name() { return "concurrent"; }
    void pick(const std::vector<SchedRequest *> &active, NnUint nRows, std::vector<SchedItem> &items) {
        NnUint left = nRows;
        for (size_t i = 0; i < active.size() && left > 0 && !isFull(items); i++) {
            if (active[i]->ready && active[i]->phase == SCHED_DECODE) {
                items.push_back(SchedItem{active[i], 1});
                left--;
            }
        }
        for (size_t i = 0; i < active.size() && left > 0 && !isFull(items); i++) {
            if (active[i]->ready && active[i]->phase == SCHED_PREFILL) {
                const NnUint n = schedRequestRows(active[i], left);
                items.push_back(SchedItem{active[i], n});
                left -= n;
            }
        }
    }
private:
    bool isFull(const std::vector<SchedItem> &items) {
        return maxRequests > 0 && items.size() >= maxRequests;
    }
};

static inline SchedPolicy *createSchedPolicy(const char *name, NnUint rrQuantum, NnUint mbMaxRequests) {
    if (std::strcmp(name, "serial") == 0)
        return new SerialPolicy();
    if (std::strcmp(name, "rr") == 0)
        return new RoundRobinPolicy(rrQuantum);
    if (std::strcmp(name, "concurrent") == 0)
        return new ConcurrentPolicy(mbMaxRequests);
    throw std::runtime_error("Unknown scheduling policy: " + std::string(name));
}

#endif

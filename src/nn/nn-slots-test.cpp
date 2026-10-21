#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "nn-config-builder.hpp"
#include "nn-quants.hpp"
#include "nn-cpu.hpp"
#ifdef DLLAMA_VULKAN
#include "nn-vulkan.hpp"
#endif

// KV-cache slots: a position is slot * SEQ_LEN + pos (see sliceKvCache). A request in slot 1 must
// get exactly the outputs it would get in slot 0, and rows of different slots may share a batch.
// The net is the attention part of a block: rope q/k, shift k/v into the cache, multi-head att.
//   ./nn-slots-test cpu
//   ./nn-slots-test vulkan   (DLLAMA_VULKAN builds, run from the directory with src/nn/vulkan/*.spv)

#define N_BATCHES 4
#define SEQ_LEN 16
#define N_SLOTS 2
#define N_HEADS 4
#define N_KV_HEADS 2
#define HEAD_DIM 8
#define Q_DIM (N_HEADS * HEAD_DIM)
#define KV_DIM (N_KV_HEADS * HEAD_DIM)

typedef struct {
    NnUint pos;
    NnUint q;
    NnUint k;
    NnUint v;
    NnUint y;
} TestPipes;

typedef struct {
    NnUint position;
    NnUint token;
} TestRow;

static void buildNet(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, TestPipes *p) {
    NnNetConfigBuilder netBuilder(1, N_BATCHES);
    p->pos = netBuilder.addPipe("POS", size2D(F_32, N_BATCHES, 1));
    p->q = netBuilder.addPipe("Q", size2D(F_32, N_BATCHES, Q_DIM));
    p->k = netBuilder.addPipe("K", size2D(F_32, N_BATCHES, KV_DIM));
    p->v = netBuilder.addPipe("V", size2D(F_32, N_BATCHES, KV_DIM));
    p->y = netBuilder.addPipe("Y", size2D(F_32, N_BATCHES, Q_DIM));
    netBuilder.addPreSync(p->pos);

    const NnRopeSlice ropeSlice = sliceRope(ROPE_LLAMA, Q_DIM, KV_DIM, N_KV_HEADS, 1, SEQ_LEN, HEAD_DIM, 10000.0f, 0);
    const NnKvCacheSlice kvCacheSlice = sliceKvCache(KV_DIM, SEQ_LEN, 1, N_SLOTS);
    const NnMultiHeadAttSlice multiHeadAttSlice = sliceMultiHeadAtt(N_HEADS, SEQ_LEN, 1, N_BATCHES);

    NnNodeConfigBuilder nodeBuilder(0);
    const NnUint qBufferIndex = nodeBuilder.addBuffer("q", size2D(F_32, N_BATCHES, Q_DIM));
    const NnUint kTempBufferIndex = nodeBuilder.addBuffer("k_temp", size2D(F_32, N_BATCHES, KV_DIM));
    const NnUint vTempBufferIndex = nodeBuilder.addBuffer("v_temp", size2D(F_32, N_BATCHES, KV_DIM));
    const NnUint zBufferIndex = nodeBuilder.addBuffer("z", size2D(F_32, N_BATCHES, Q_DIM));
    const NnUint kBufferIndex = nodeBuilder.addBuffer("k", kvCacheSlice.keySize);
    const NnUint vBufferIndex = nodeBuilder.addBuffer("v", kvCacheSlice.valueSize);
    const NnUint ropeCacheBufferIndex = nodeBuilder.addBuffer("rope_cache", ropeSlice.cacheSize);
    const NnUint attBufferIndex = nodeBuilder.addBuffer("att", multiHeadAttSlice.attSize);

    NnSegmentConfigBuilder segment;
    segment.addOp(OP_CAST, "cast_q", 0,
        pointerBatchConfig(SRC_PIPE, p->q), pointerBatchConfig(SRC_BUFFER, qBufferIndex), size0(), NnCastOpCodeConfig{});
    segment.addOp(OP_CAST, "cast_k", 0,
        pointerBatchConfig(SRC_PIPE, p->k), pointerBatchConfig(SRC_BUFFER, kTempBufferIndex), size0(), NnCastOpCodeConfig{});
    segment.addOp(OP_CAST, "cast_v", 0,
        pointerBatchConfig(SRC_PIPE, p->v), pointerBatchConfig(SRC_BUFFER, vTempBufferIndex), size0(), NnCastOpCodeConfig{});
    segment.addOp(OP_ROPE, "rope_q", 0,
        pointerBatchConfig(SRC_BUFFER, qBufferIndex), pointerBatchConfig(SRC_BUFFER, qBufferIndex), size0(),
        NnRopeOpConfig{ROPE_LLAMA, 1, p->pos, ropeCacheBufferIndex, 1.0f, 1.0f, 1.0f, SEQ_LEN, ropeSlice});
    segment.addOp(OP_ROPE, "rope_k", 0,
        pointerBatchConfig(SRC_BUFFER, kTempBufferIndex), pointerBatchConfig(SRC_BUFFER, kTempBufferIndex), size0(),
        NnRopeOpConfig{ROPE_LLAMA, 0, p->pos, ropeCacheBufferIndex, 1.0f, 1.0f, 1.0f, SEQ_LEN, ropeSlice});
    segment.addOp(OP_SHIFT, "shift_k", 0,
        pointerBatchConfig(SRC_BUFFER, kTempBufferIndex), pointerRawConfig(SRC_BUFFER, kBufferIndex), size0(),
        NnShiftOpCodeConfig{p->pos});
    segment.addOp(OP_SHIFT, "shift_v", 0,
        pointerBatchConfig(SRC_BUFFER, vTempBufferIndex), pointerRawConfig(SRC_BUFFER, vBufferIndex), size0(),
        NnShiftOpCodeConfig{p->pos});
    segment.addOp(OP_MULTIHEAD_ATT, "multihead_att", 0,
        pointerBatchedSliceConfig(SRC_BUFFER, zBufferIndex), pointerBatchedSliceConfig(SRC_BUFFER, zBufferIndex), size0(),
        NnMultiHeadAttOpConfig{N_HEADS, N_HEADS, N_KV_HEADS, HEAD_DIM, SEQ_LEN, Q_DIM, KV_DIM,
            p->pos, qBufferIndex, kBufferIndex, vBufferIndex, attBufferIndex});
    segment.addOp(OP_CAST, "cast_y", 0,
        pointerBatchedSliceConfig(SRC_BUFFER, zBufferIndex), pointerBatchConfig(SRC_PIPE, p->y), size0(), NnCastOpCodeConfig{});
    nodeBuilder.addSegment(segment.build());

    *netConfig = netBuilder.build();
    *nodeConfig = nodeBuilder.build();
}

// Runs one forward with the given rows and returns Y (one Q_DIM vector per row).
// The q/k/v inputs of a row depend only on its token, so equal tokens mean equal inputs.
static std::vector<float> forward(NnExecutor *executor, NnNetExecution *execution, TestPipes *p,
    const std::vector<TestRow> &rows) {
    const NnUint batchSize = (NnUint)rows.size();
    float *pos = (float *)execution->pipes[p->pos];
    float *q = (float *)execution->pipes[p->q];
    float *k = (float *)execution->pipes[p->k];
    float *v = (float *)execution->pipes[p->v];
    float *y = (float *)execution->pipes[p->y];
    for (NnUint b = 0; b < batchSize; b++) {
        const float t = (float)rows[b].token;
        pos[b] = (float)rows[b].position;
        for (NnUint i = 0; i < Q_DIM; i++)
            q[b * Q_DIM + i] = sinf(t * 0.37f + i * 0.11f);
        for (NnUint i = 0; i < KV_DIM; i++) {
            k[b * KV_DIM + i] = cosf(t * 0.53f + i * 0.07f);
            v[b * KV_DIM + i] = sinf(t * 0.29f - i * 0.13f);
        }
    }
    execution->setBatchSize(batchSize);
    executor->forward();
    return std::vector<float>(y, y + batchSize * Q_DIM);
}

static void assertRowsEqual(const char *name, const std::vector<float> &a, NnUint aRow,
    const std::vector<float> &b, NnUint bRow) {
    for (NnUint i = 0; i < Q_DIM; i++) {
        const float x = a[aRow * Q_DIM + i];
        const float e = b[bRow * Q_DIM + i];
        if (!(fabsf(x - e) <= 1e-6f)) {
            printf("❌ %s failed at [%u]: %f != %f\n", name, i, x, e);
            exit(1);
        }
    }
    printf("✅ %s passed\n", name);
}

int main(int argc, char **argv) {
    initQuants();
    const bool useVulkan = argc > 1 && std::strcmp(argv[1], "vulkan") == 0;

    TestPipes p;
    NnNetConfig netConfig;
    NnNodeConfig nodeConfig;
    buildNet(&netConfig, &nodeConfig, &p);
    std::unique_ptr<NnNetConfig, void(*)(NnNetConfig *)> netConfigPtr(&netConfig, releaseNetConfig);
    std::unique_ptr<NnNodeConfig, void(*)(NnNodeConfig *)> nodeConfigPtr(&nodeConfig, releaseNodeConfig);

    NnNetExecution execution(useVulkan ? 1 : 2, &netConfig);
    std::vector<NnExecutorDevice> devices;
    if (useVulkan) {
#ifdef DLLAMA_VULKAN
        devices.push_back(NnExecutorDevice(new NnVulkanDevice(0, &netConfig, &nodeConfig, &execution), -1, -1));
#else
        printf("This build has no Vulkan support\n");
        return 1;
#endif
    } else {
        devices.push_back(NnExecutorDevice(new NnCpuDevice(&netConfig, &nodeConfig, &execution), -1, -1));
    }
    NnFakeNodeSynchronizer synchronizer;
    NnExecutor executor(&netConfig, &nodeConfig, &devices, &execution, &synchronizer, false);
    printf("Device: %s\n", useVulkan ? "vulkan" : "cpu");

    const NnUint s1 = SEQ_LEN; // first position of slot 1

    // The same 3-token prefill in slot 0 and in slot 1 gives the same outputs
    std::vector<float> slot0 = forward(&executor, &execution, &p, {{0, 1}, {1, 2}, {2, 3}});
    std::vector<float> slot1 = forward(&executor, &execution, &p, {{s1 + 0, 1}, {s1 + 1, 2}, {s1 + 2, 3}});
    for (NnUint r = 0; r < 3; r++)
        assertRowsEqual("prefill in slot 1 equals slot 0", slot1, r, slot0, r);

    // One batch with a row of each slot: both slots hold the same history, so the rows are equal
    std::vector<float> mixed = forward(&executor, &execution, &p, {{3, 4}, {s1 + 3, 4}});
    assertRowsEqual("mixed-slot batch, row of slot 1 equals row of slot 0", mixed, 1, mixed, 0);

    // A row in a mixed batch equals the same row alone in a batch of one
    std::vector<float> alone = forward(&executor, &execution, &p, {{s1 + 3, 4}});
    assertRowsEqual("mixed-slot batch row equals the row alone", mixed, 1, alone, 0);

    // Different histories: slot 1 gets other tokens, slot 0 must not see them
    forward(&executor, &execution, &p, {{s1 + 4, 9}, {4, 5}});
    std::vector<float> slot0Next = forward(&executor, &execution, &p, {{5, 6}});
    std::vector<float> slot1Copy = forward(&executor, &execution, &p, {{s1 + 4, 5}, {s1 + 5, 6}});
    assertRowsEqual("slot 0 is isolated from slot 1", slot0Next, 0, slot1Copy, 1);
    return 0;
}

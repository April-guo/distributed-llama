#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "nn/nn-core.hpp"
#include "nn/nn-cpu.hpp"
#include "nn/nn-executor.hpp"
#include "nn/nn-quants.hpp"
#include "llm.hpp"

// Pipeline parallelism must not change any number. A small random Llama (Q40 weights, Q80 sync,
// like the real models) runs on the CPU as one node and split into stages; the stages run one
// after the other and hand X on as the workers do. Every forward must give bit-identical logits.
// Also checks the partition (every weight in exactly one stage) and the layer split.
//   ./llm-pp-test

#define N_BATCHES 4

static int nFailed = 0;

static void check(bool ok, const std::string &name) {
    printf("%s %s\n", ok ? "✅" : "❌", name.c_str());
    if (!ok)
        nFailed++;
}

static LlmHeader createTestHeader() {
    LlmHeader h;
    std::memset(&h, 0, sizeof(h));
    h.archType = LLAMA;
    h.hiddenAct = HIDDEN_ACT_SILU;
    h.dim = 256;
    h.hiddenDim = 512;
    h.nLayers = 7;
    h.nHeads = 8;
    h.nKvHeads = 2;
    h.headDim = h.dim / h.nHeads;
    h.qDim = h.headDim * h.nHeads;
    h.kvDim = h.headDim * h.nKvHeads;
    h.vocabSize = 384;
    h.seqLen = 32;
    h.origSeqLen = h.seqLen;
    h.ropeType = ROPE_LLAMA3_1;
    h.ropeTheta = 500000.0f;
    h.ropeScalingFactor = 8.0f;
    h.ropeScalingLowFreqFactor = 1.0f;
    h.ropeScalingHighFreqFactory = 4.0f;
    h.ropeScalingOrigMaxSeqLen = 8192;
    h.normEpsilon = 1e-5f;
    h.weightType = F_Q40;
    h.syncType = F_Q80;
    return h;
}

// The same tensor (by op name and layer) gets the same bytes in every net
static std::vector<NnByte> createWeight(const char *name, NnUint index, NnSize3D size) {
    uint64_t state = 1469598103934665603ull;
    for (const char *c = name; *c != '\0'; c++)
        state = (state ^ (unsigned char)*c) * 1099511628211ull;
    state = (state ^ index) * 1099511628211ull;

    const bool isNorm = std::strstr(name, "norm") != nullptr;
    const bool isEmbedding = std::strcmp(name, "embedding") == 0;
    std::vector<float> values(size.length);
    for (NnSize i = 0; i < size.length; i++) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        const float u = (float)((state >> 40) & 0xFFFFFF) / (float)0xFFFFFF * 2.0f - 1.0f;
        values[i] = isNorm ? 1.0f + 0.1f * u : (isEmbedding ? 0.5f * u : 0.08f * u);
    }

    std::vector<NnByte> bytes(size.nBytes);
    if (size.floatType == F_32) {
        std::memcpy(bytes.data(), values.data(), size.nBytes);
    } else if (size.floatType == F_Q40) {
        quantizeF32toQ40(values.data(), (NnBlockQ40 *)bytes.data(), size.length, 1, 0);
    } else {
        printf("Unsupported weight type in %s\n", name);
        exit(1);
    }
    return bytes;
}

class TestNode {
public:
    NnNodeConfig *config;
    std::unique_ptr<NnNetExecution> execution;
    std::vector<NnExecutorDevice> devices;
    std::unique_ptr<NnExecutor> executor;

    TestNode(NnNetConfig *netConfig, NnNodeConfig *config, NnNodeSynchronizer *synchronizer) {
        this->config = config;
        execution.reset(new NnNetExecution(1, netConfig));
        devices.push_back(NnExecutorDevice(new NnCpuDevice(netConfig, config, execution.get()), -1, -1));
        executor.reset(new NnExecutor(netConfig, config, &devices, execution.get(), synchronizer, false));
        for (NnUint segmentIndex = 0; segmentIndex < config->nSegments; segmentIndex++) {
            NnSegmentConfig *segment = &config->segments[segmentIndex];
            for (NnUint opIndex = 0; opIndex < segment->nOps; opIndex++) {
                NnOpConfig *op = &segment->ops[opIndex];
                if (op->weightSize.nBytes == 0)
                    continue;
                std::vector<NnByte> weight = createWeight(op->name, op->index, op->weightSize);
                executor->loadWeight(op->name, op->index, 0, weight.size(), weight.data());
            }
        }
    }

    float *pipe(NnUint pipeIndex) {
        return (float *)execution->pipes[pipeIndex];
    }

    void forward(NnUint batchSize) {
        execution->setBatchSize(batchSize);
        executor->forward();
    }
};

typedef struct {
    std::vector<NnUint> positions;
    std::vector<NnUint> tokens;
} TestBatch;

// One node runs the whole net (tensor parallelism with one slice)
static std::vector<float> forwardOneNode(LlmNet *net, TestNode *node, const TestBatch &batch) {
    const NnUint batchSize = (NnUint)batch.tokens.size();
    for (NnUint i = 0; i < batchSize; i++) {
        node->pipe(net->positionPipeIndex)[i] = (float)batch.positions[i];
        node->pipe(net->tokenPipeIndex)[i] = (float)batch.tokens[i];
    }
    node->forward(batchSize);
    const float *logits = node->pipe(net->logitsPipeIndex);
    return std::vector<float>(logits, logits + batchSize * net->header->vocabSize);
}

// The root embeds, every stage gets the positions and the previous node's X, the last stage
// returns the logits: what the root and the workers send each other
static std::vector<float> forwardPipeline(LlmNet *net, std::vector<std::unique_ptr<TestNode>> &nodes, const TestBatch &batch) {
    const NnUint batchSize = (NnUint)batch.tokens.size();
    const NnSize xBytes = batchSize * net->header->dim * sizeof(float);
    TestNode *root = nodes[0].get();
    for (NnUint i = 0; i < batchSize; i++) {
        root->pipe(net->positionPipeIndex)[i] = (float)batch.positions[i];
        root->pipe(net->tokenPipeIndex)[i] = (float)batch.tokens[i];
    }
    root->forward(batchSize);

    const float *x = root->pipe(net->xPipeIndex);
    for (size_t s = 1; s < nodes.size(); s++) {
        TestNode *stage = nodes[s].get();
        for (NnUint i = 0; i < batchSize; i++)
            stage->pipe(net->positionPipeIndex)[i] = (float)batch.positions[i];
        std::memcpy(stage->pipe(net->xPipeIndex), x, xBytes);
        stage->forward(batchSize);
        x = stage->pipe(net->xPipeIndex);
    }
    const float *logits = nodes.back()->pipe(net->logitsPipeIndex);
    return std::vector<float>(logits, logits + batchSize * net->header->vocabSize);
}

typedef std::map<std::string, NnSize> WeightMap; // "<name>:<index>" -> bytes

static void collectWeights(NnNodeConfig *config, WeightMap *weights, NnUint *nDuplicates) {
    for (NnUint segmentIndex = 0; segmentIndex < config->nSegments; segmentIndex++) {
        NnSegmentConfig *segment = &config->segments[segmentIndex];
        for (NnUint opIndex = 0; opIndex < segment->nOps; opIndex++) {
            NnOpConfig *op = &segment->ops[opIndex];
            if (op->weightSize.nBytes == 0)
                continue;
            const std::string key = std::string(op->name) + ":" + std::to_string(op->index);
            if (weights->count(key) > 0)
                (*nDuplicates)++;
            (*weights)[key] = op->weightSize.nBytes;
        }
    }
}

static void testPartition(LlmNet *oneNode, LlmNet *pipeline, const std::string &name) {
    WeightMap expected, actual;
    NnUint nDuplicates = 0;
    collectWeights(&oneNode->nodeConfigs[0], &expected, &nDuplicates);
    nDuplicates = 0;
    for (NnUint nodeIndex = 0; nodeIndex < pipeline->nNodeConfigs; nodeIndex++)
        collectWeights(&pipeline->nodeConfigs[nodeIndex], &actual, &nDuplicates);
    check(nDuplicates == 0 && actual == expected, name + ": every weight is in exactly one node, with its full size");

    bool isSlice0 = pipeline->netConfig.nNodes == 1;
    for (NnUint nodeIndex = 0; nodeIndex < pipeline->nNodeConfigs; nodeIndex++)
        isSlice0 = isSlice0 && pipeline->nodeConfigs[nodeIndex].nodeIndex == 0;
    check(isSlice0, name + ": every node is slice 0 of 1");

    NnNodeConfig *root = &pipeline->nodeConfigs[0];
    check(root->nSegments == 1 && root->segments[0].nOps == 1 && std::strcmp(root->segments[0].ops[0].name, "embedding") == 0,
        name + ": the root only embeds");
}

static void testPipeline(LlmHeader *h, LlmNet *oneNodeNet, TestNode *oneNode, const std::vector<NnUint> &stageEnds) {
    std::string name = "stages";
    for (NnUint s = 0; s < stageEnds.size(); s++)
        name += (s == 0 ? " " : ",") + std::to_string(stageEnds[s] - llmStageStart(stageEnds, s));

    LlmNet net = buildLlmPipelineNet(h, N_BATCHES, 1, stageEnds);
    std::unique_ptr<LlmNet, void(*)(LlmNet *)> netPtr(&net, releaseLlmNet);
    testPartition(oneNodeNet, &net, name);

    NnFakeNodeSynchronizer synchronizer;
    std::vector<std::unique_ptr<TestNode>> nodes;
    for (NnUint nodeIndex = 0; nodeIndex < net.nNodeConfigs; nodeIndex++)
        nodes.push_back(std::unique_ptr<TestNode>(new TestNode(&net.netConfig, &net.nodeConfigs[nodeIndex], &synchronizer)));

    // Prefill in chunks of N_BATCHES, then decode the one-node net's greedy tokens. The one-node
    // net starts from an empty KV cache too: it is rebuilt for every pipeline.
    std::vector<TestBatch> batches;
    const NnUint prompt[] = {5, 17, 300, 42, 7, 99};
    const NnUint nPrompt = sizeof(prompt) / sizeof(prompt[0]);
    for (NnUint start = 0; start < nPrompt; start += N_BATCHES) {
        TestBatch batch;
        for (NnUint i = start; i < nPrompt && i < start + N_BATCHES; i++) {
            batch.positions.push_back(i);
            batch.tokens.push_back(prompt[i]);
        }
        batches.push_back(batch);
    }

    NnUint nForwards = 0;
    NnUint nEqual = 0;
    float maxDiff = 0.0f;
    NnUint pos = nPrompt;
    NnUint token = 0;
    for (NnUint step = 0; step < batches.size() + 10; step++) {
        TestBatch batch;
        if (step < batches.size()) {
            batch = batches[step];
        } else {
            batch.positions.push_back(pos++);
            batch.tokens.push_back(token);
        }
        std::vector<float> expected = forwardOneNode(oneNodeNet, oneNode, batch);
        std::vector<float> actual = forwardPipeline(&net, nodes, batch);
        nForwards++;
        if (std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(float)) == 0)
            nEqual++;
        for (size_t i = 0; i < expected.size(); i++)
            maxDiff = std::max(maxDiff, std::fabs(expected[i] - actual[i]));

        const float *last = &expected[(batch.tokens.size() - 1) * h->vocabSize];
        token = 0;
        for (NnUint i = 1; i < h->vocabSize; i++) {
            if (last[i] > last[token])
                token = i;
        }
    }
    char detail[128];
    snprintf(detail, sizeof(detail), ": %u/%u forwards with bit-identical logits (max diff %g)", nEqual, nForwards, maxDiff);
    check(nEqual == nForwards, name + detail);
}

static LlmHeader createShapeHeader(NnUint dim, NnUint hiddenDim, NnUint nLayers, NnUint nHeads, NnUint nKvHeads, NnUint vocabSize) {
    LlmHeader h;
    std::memset(&h, 0, sizeof(h));
    h.archType = LLAMA;
    h.dim = dim;
    h.hiddenDim = hiddenDim;
    h.nLayers = nLayers;
    h.nHeads = nHeads;
    h.nKvHeads = nKvHeads;
    h.headDim = dim / nHeads;
    h.qDim = h.headDim * nHeads;
    h.kvDim = h.headDim * nKvHeads;
    h.vocabSize = vocabSize;
    return h;
}

static std::string formatSplit(const std::vector<NnUint> &stageEnds) {
    std::string text;
    for (NnUint s = 0; s < stageEnds.size(); s++)
        text += (s == 0 ? "" : ",") + std::to_string(stageEnds[s] - llmStageStart(stageEnds, s));
    return text;
}

int main() {
    initQuants();

    LlmHeader h = createTestHeader();
    // A fresh one-node net per pipeline, so both start from an empty KV cache
    std::vector<std::vector<NnUint>> splits = {
        {7},
        splitLlmLayers(&h, 2),
        parseLlmStageLayers("2,2,3", &h, 3),
        {1, 2, 3, 4, 5, 6, 7},
    };
    for (const std::vector<NnUint> &stageEnds : splits) {
        LlmNet oneNodeNet = buildLlmNet(&h, 1, N_BATCHES, 1);
        std::unique_ptr<LlmNet, void(*)(LlmNet *)> oneNodeNetPtr(&oneNodeNet, releaseLlmNet);
        NnFakeNodeSynchronizer synchronizer;
        TestNode oneNode(&oneNodeNet.netConfig, &oneNodeNet.nodeConfigs[0], &synchronizer);
        testPipeline(&h, &oneNodeNet, &oneNode, stageEnds);
    }

    // Layer splits of the real models: the last stage also runs the logits matmul
    LlmHeader llama8b = createShapeHeader(4096, 14336, 32, 32, 8, 128256);
    LlmHeader llama70b = createShapeHeader(8192, 28672, 80, 64, 8, 128256);
    check(formatSplit(splitLlmLayers(&llama8b, 2)) == "17,15", "8B in 2 stages: " + formatSplit(splitLlmLayers(&llama8b, 2)));
    for (NnUint nStages : {4u, 8u})
        printf("   8B in %u stages: %s\n", nStages, formatSplit(splitLlmLayers(&llama8b, nStages)).c_str());
    for (NnUint nStages : {2u, 4u, 8u})
        printf("   70B in %u stages: %s\n", nStages, formatSplit(splitLlmLayers(&llama70b, nStages)).c_str());

    bool rejectsBadSplit = false;
    try {
        parseLlmStageLayers("3,3", &h, 2);
    } catch (const std::exception &e) {
        rejectsBadSplit = true;
    }
    check(rejectsBadSplit, "--pp-layers must add up to all layers");

    printf("%d check(s) failed\n", nFailed);
    return nFailed == 0 ? 0 : 1;
}

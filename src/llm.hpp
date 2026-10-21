#ifndef LLM_HPP
#define LLM_HPP

#include <vector>
#include "nn/nn-core.hpp"
#include "nn/nn-executor.hpp"
#include "nn/nn-network.hpp"

enum LlmHeaderKey {
    VERSION = 0,
    ARCH_TYPE = 1,
    DIM = 2,
    HIDDEN_DIM = 3,
    N_LAYERS = 4,
    N_HEADS = 5,
    N_KV_HEADS = 6,
    N_EXPERTS = 7,
    N_ACTIVE_EXPERTS = 8,
    VOCAB_SIZE = 9,
    SEQ_LEN = 10,
    HIDDEN_ACT = 11,
    ROPE_THETA = 12,
    WEIGHT_FLOAT_TYPE = 13,
    ROPE_SCALING_FACTOR = 14,
    ROPE_SCALING_LOW_FREQ_FACTOR = 15,
    ROPE_SCALING_HIGH_FREQ_FACTORY = 16,
    ROPE_SCALING_ORIG_MAX_SEQ_LEN = 17,
    ROPE_TYPE = 18,
    HEAD_DIM = 19,
    NORM_EPSILON = 20,
    MOE_HIDDEN_DIM = 21,
};

enum LlmHiddenAct {
    HIDDEN_ACT_GELU,
    HIDDEN_ACT_SILU,
};

enum LlmArchType {
    LLAMA = 0xABCD00,
    QWEN3 = 0xABCD01,
    QWEN3_MOE = 0xABCD02,
};

typedef struct {
    NnSize headerSize;
    NnSize fileSize;
    int version;
    LlmArchType archType;
    NnUint dim;
    NnUint nLayers;
    NnUint nHeads;
    NnUint headDim;
    NnUint nKvHeads;
    NnUint nExperts;
    NnUint nActiveExperts;
    NnUint origSeqLen; // Original model context length
    NnUint seqLen; // Limited context length by the `--max-seq-len` argument
    NnUint hiddenDim;
    NnUint moeHiddenDim;
    LlmHiddenAct hiddenAct;
    NnUint qDim;
    NnUint kvDim;
    NnUint vocabSize;
    float ropeTheta;
    NnRopeType ropeType;
    float ropeScalingFactor;
    float ropeScalingLowFreqFactor;
    float ropeScalingHighFreqFactory;
    NnUint ropeScalingOrigMaxSeqLen;
    float normEpsilon;

    NnFloatType weightType;
    NnFloatType syncType;
} LlmHeader;

typedef struct {
    LlmHeader *header;
    NnNetConfig netConfig;
    NnUint nNodeConfigs;
    NnNodeConfig *nodeConfigs;
    // Pipeline parallelism: 0 = off (tensor parallelism, node i runs slice i of every layer).
    // Otherwise node 0 is the root, which only embeds tokens, and node s + 1 runs the full width
    // of layers [stageEnds[s - 1], stageEnds[s]) (stage 0 starts at layer 0).
    NnUint nStages;
    std::vector<NnUint> stageEnds;
    NnRowMatmulSlice qSlice;
    NnRowMatmulSlice kSlice;
    NnRowMatmulSlice vSlice;
    NnColMatmulSlice woSlice;
    NnRowMatmulSlice w1Slice;
    NnColMatmulSlice w2Slice;
    NnRowMatmulSlice w3Slice;
    NnRowMatmulSlice wclsSlice;
    NnUint positionPipeIndex;
    NnUint tokenPipeIndex;
    NnUint xPipeIndex;
    NnUint logitsPipeIndex;
    NnUint nSlots;
    NnSize3D tokenEmbeddingSize;
    NnSize3D rmsNormSize;
    NnSize3D qkRmsNormSize;
    NnSize3D moeGateSize;
} LlmNet;

LlmHeader loadLlmHeader(const char* path, const unsigned int maxSeqLen, NnFloatType syncType);
void printLlmHeader(LlmHeader *header);
LlmNet buildLlmNet(LlmHeader *h, NnUint nNodes, NnUint nBatches, NnUint nSlots = 1);
LlmNet buildLlmPipelineNet(LlmHeader *h, NnUint nBatches, NnUint nSlots, const std::vector<NnUint> &stageEnds);
void releaseLlmNet(LlmNet *net);
void loadLlmNetWeight(const char* path, LlmNet *net, NnRootWeightLoader *loader);

// The first layer of a pipeline stage
static inline NnUint llmStageStart(const std::vector<NnUint> &stageEnds, NnUint stageIndex) {
    return stageIndex == 0 ? 0 : stageEnds[stageIndex - 1];
}
// Splits the layers into stages with the smallest possible maximum stage cost. A layer costs its
// parameter count; the last stage also runs the logits matmul, so it gets fewer layers.
std::vector<NnUint> splitLlmLayers(LlmHeader *h, NnUint nStages);
// Parses layers per stage, e.g. "17,15"
std::vector<NnUint> parseLlmStageLayers(const char *text, LlmHeader *h, NnUint nStages);

// Pipeline parallelism: every tensor goes whole to the node that runs it. The embedding stays on
// the root, the layers go to their stages and the final norm and logits to the last stage.
class LlmPipelineWeightLoader : public NnRootWeightLoader {
private:
    std::vector<NnUint> layerNodes;
    NnUint lastNode;
    NnUint resolveNode(const char *opName, NnUint opIndex);
public:
    LlmPipelineWeightLoader(NnExecutor *executor, NnNetwork *network, const std::vector<NnUint> &stageEnds);
    NnSize loadAll(const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight) override;
    NnSize loadRowMatmulSlices(const char *opName, const NnUint opIndex, const NnUint expertIndex, NnRowMatmulSlice *slice, NnByte *weight) override;
    NnSize loadColMatmulSlices(const char *opName, const NnUint opIndex, const NnUint expertIndex, NnColMatmulSlice *slice, NnByte *weight) override;
};

#endif
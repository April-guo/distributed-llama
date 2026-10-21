#ifndef APP_HPP
#define APP_HPP

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "nn/nn-core.hpp"
#include "nn/nn-cpu.hpp"
#include "tokenizer.hpp"
#include "llm.hpp"

class AppCliArgs {
public:
    char *mode;
    NnUint nThreads;
    NnUint nBatches;
    bool info;
    bool help;

    // inference
    char *modelPath;
    char *tokenizerPath;
    char *prompt;
    NnFloatType syncType;
    NnUint nWorkers;
    char **workerHosts;
    NnUint *workerPorts;
    float temperature;
    float topp;
    NnUint steps;
    bool benchmark;
    unsigned long long seed;
    ChatTemplateType chatTemplateType;
    NnUint maxSeqLen;
    bool netTurbo;
    int gpuIndex;
    int gpuSegmentFrom;
    int gpuSegmentTo;

    // binding
    const char *host;
    NnUint port;

    // scheduling (dllama-api)
    NnUint maxSlots;
    char *policy;
    NnUint rrQuantum;
    char *schedLogPath;
    NnUint mbMaxRequests;

    // pipeline parallelism: nStages > 0 makes every worker one stage, in the order of --workers
    NnUint nStages;
    char *ppLayers;
    NnUint ppWindow;

    char *logitsLogPath;

    static AppCliArgs parse(int argc, char **argv, bool hasMode);
    ~AppCliArgs();
};

// The control packet is sent to workers before every forward:
// [batchSize (0 = stop signal), position of row 0, ..., position of row nBatches - 1].
// A position is slot * seqLen + pos, so the rows of one batch may belong to different requests.
#define LLM_CONTROL_PACKET_LENGTH(nBatches) ((nBatches) + 1)

// Pipeline parallelism: a microbatch travels root -> stage 0 -> ... -> last stage -> root as this
// header ([control packet, microbatch id]) and one row per batch item: X between the stages, the
// logits from the last stage to the root. The stop signal (batch size 0) has no rows.
#define LLM_PIPELINE_HEADER_LENGTH(nBatches) (LLM_CONTROL_PACKET_LENGTH(nBatches) + 1)

// Sent by the root to every worker after its node config
typedef struct {
    NnUint isStage; // 0: tensor-parallel worker, 1: pipeline stage
    NnUint stageIndex;
    NnUint nStages;
    NnUint layerStart;
    NnUint layerEnd;
    NnUint inSocket; // a stage reads microbatches from this socket
    NnUint outSocket; // and writes its output to this one
    NnUint positionPipeIndex;
    NnUint inputPipeIndex;
    NnUint outputPipeIndex;
} LlmWorkerRole;

// The output of a microbatch, read from the last stage
struct LlmPipelineResult {
    NnUint id;
    NnUint batchSize;
    std::vector<NnByte> logits;
};

class RootLlmInference {
public:
    float *logitsPipe;
    NnUint nStages; // 0 = tensor parallelism
private:
    float *tokenPipe;
    float *positionPipe;
    LlmHeader *header;
    NnNetExecution *execution;
    NnExecutor *executor;
    NnNetwork *network;
    NnUint nSlots;
    std::unique_ptr<NnUint[]> controlPacket;
    NnSize controlPacketSize;

    // pipeline parallelism
    NnByte *xPipe;
    NnSize xRowBytes;
    NnSize logitsRowBytes;
    NnSize pipelineHeaderSize;
    NnUint lastStageSocket;
    NnUint nextSubmitId;
    NnUint nextCollectId;
    bool isFinished;
    std::thread receiver;
    std::mutex resultsMutex;
    std::condition_variable resultsCondition;
    std::deque<std::unique_ptr<LlmPipelineResult>> results;
    bool isReceiverDone;
    std::string receiverError;
    void receiveLoop();
public:
    RootLlmInference(LlmNet *net, NnNetExecution *execution, NnExecutor *executor, NnNetwork *network);
    ~RootLlmInference();
    void setBatchSize(NnUint batchSize);
    void setPosition(NnUint position);
    void setRowPosition(NnUint batchIndex, NnUint position);
    void setToken(NnUint batchIndex, NnUint token);
    // Runs one forward and leaves the logits in logitsPipe
    void forward();
    // Starts a forward with the current rows. Tensor parallelism runs it to the end here; a
    // pipeline embeds the tokens and sends the microbatch to the first stage.
    void submit();
    // Waits up to `timeout` for the oldest submitted microbatch; on true its logits are in
    // logitsPipe (rows as submitted). Microbatches come back in submission order.
    bool tryCollect(std::chrono::microseconds timeout);
    void collect();
    void finish();
};

class WorkerLlmInference {
public:
    bool isFinished;
private:
    float *positionPipe;
    NnNetExecution *execution;
    NnNetwork *network;
    std::unique_ptr<NnUint[]> controlPacket;
    NnSize controlPacketSize;
public:
    WorkerLlmInference(NnNetExecution *execution, NnNetwork *network);
    bool tryReadControlPacket();
};

typedef struct {
    AppCliArgs *args;
    LlmHeader *header;
    RootLlmInference *inference;
    Tokenizer *tokenizer;
    Sampler *sampler;
    NnNetwork *network;
    NnExecutor *executor;
} AppInferenceContext;

void runInferenceApp(AppCliArgs *args, void (*handler)(AppInferenceContext *context));
void runWorkerApp(AppCliArgs *args);

#endif

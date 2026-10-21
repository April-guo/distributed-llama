#ifndef APP_HPP
#define APP_HPP

#include <chrono>
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

    static AppCliArgs parse(int argc, char **argv, bool hasMode);
    ~AppCliArgs();
};

// The control packet is sent to workers before every forward:
// [batchSize (0 = stop signal), position of row 0, ..., position of row nBatches - 1].
// A position is slot * seqLen + pos, so the rows of one batch may belong to different requests.
#define LLM_CONTROL_PACKET_LENGTH(nBatches) ((nBatches) + 1)

class RootLlmInference {
public:
    float *logitsPipe;
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
public:
    RootLlmInference(LlmNet *net, NnNetExecution *execution, NnExecutor *executor, NnNetwork *network);
    void setBatchSize(NnUint batchSize);
    void setPosition(NnUint position);
    void setRowPosition(NnUint batchIndex, NnUint position);
    void setToken(NnUint batchIndex, NnUint token);
    void forward();
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

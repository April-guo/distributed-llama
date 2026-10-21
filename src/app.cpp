#include "app.hpp"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <stdexcept>
#if defined(DLLAMA_VULKAN)
    #include "nn/nn-vulkan.hpp"
#endif

static NnFloatType parseFloatType(char *val) {
    if (std::strcmp(val, "f32") == 0) return F_32;
    if (std::strcmp(val, "f16") == 0) return F_16;
    if (std::strcmp(val, "q40") == 0) return F_Q40;
    if (std::strcmp(val, "q80") == 0) return F_Q80;
    throw std::runtime_error("Invalid float type: " + std::string(val));
}

static ChatTemplateType parseChatTemplateType(char *val) {
    if (std::strcmp(val, "llama2") == 0) return TEMPLATE_LLAMA2;
    if (std::strcmp(val, "llama3") == 0) return TEMPLATE_LLAMA3;
    if (std::strcmp(val, "deepSeek3") == 0) return TEMPLATE_DEEP_SEEK3;
    throw std::runtime_error("Invalid chat template type: " + std::string(val));
}

AppCliArgs AppCliArgs::parse(int argc, char* *argv, bool requireMode) {
    AppCliArgs args;
    args.info = true;
    args.help = false;
    args.mode = nullptr;
    args.nBatches = 32;
    args.nThreads = 1;
    args.modelPath = nullptr;
    args.tokenizerPath = nullptr;
    args.prompt = nullptr;
    args.syncType = F_32;
    args.nWorkers = 0;
    args.workerHosts = nullptr;
    args.workerPorts = nullptr;
    args.host = "0.0.0.0";
    args.port = 9990;
    args.temperature = 0.8f;
    args.topp = 0.9f;
    args.steps = 0;
    args.seed = (unsigned long long)time(nullptr);
    args.chatTemplateType = TEMPLATE_UNKNOWN;
    args.maxSeqLen = 0;
    args.netTurbo = true;
    args.gpuIndex = -1;
    args.gpuSegmentFrom = -1;
    args.gpuSegmentTo = -1;
    args.maxSlots = 1;
    args.policy = (char *)"serial";
    args.rrQuantum = 1;
    args.schedLogPath = nullptr;
    args.mbMaxRequests = 0;
    args.nStages = 0;
    args.ppLayers = nullptr;
    args.ppWindow = 0;
    args.logitsLogPath = nullptr;

    int i = 1;
    if (requireMode && argc > 1) {
        args.mode = argv[1];
        i++;
    }
    // First see if any of the args are asking for help/usage and fail fast
    for (int x = 0; x < argc; x++) {
        if ((std::strcmp(argv[x], "--usage") == 0) ||
            (std::strcmp(argv[x], "--help") == 0) ||
            (std::strcmp(argv[x], "-h") == 0)) {
            args.help = true;
            return args;
        }
    }
    for (; i + 1 < argc; i += 2) {
        char *name = argv[i];
        char *value = argv[i + 1];
        if (std::strcmp(name, "--model") == 0) {
            args.modelPath = value;
        } else if (std::strcmp(name, "--tokenizer") == 0) {
            args.tokenizerPath = value;
        } else if (std::strcmp(name, "--prompt") == 0) {
            args.prompt = value;
        } else if (std::strcmp(name, "--buffer-float-type") == 0) {
            args.syncType = parseFloatType(value);
        } else if (std::strcmp(name, "--workers") == 0) {
            int j = i + 1;
            for (; j < argc && argv[j][0] != '-'; j++);
            int count = j - i - 1;

            args.nWorkers = count;
            args.workerHosts = new char*[count];
            args.workerPorts = new NnUint[count];

            for (int s = 0; s < count; s++) {
                char *v = argv[i + 1 + s];
                char *separator = std::strstr(v, ":");
                if (separator == NULL) {
                    throw std::runtime_error("Invalid worker address: " + std::string(v));
                }
                int hostLen = separator - v;
                args.workerHosts[s] = new char[hostLen + 1];
                std::memcpy(args.workerHosts[s], v, hostLen);
                args.workerHosts[s][hostLen] = '\0';
                args.workerPorts[s] = std::atoi(separator + 1);
            }

            i += count - 1;
        } else if (std::strcmp(name, "--port") == 0) {
            args.port = atoi(value);
        } else if (std::strcmp(name, "--host") == 0) {
            args.host = value;
        } else if (std::strcmp(name, "--nthreads") == 0) {
            args.nThreads = atoi(value);
        } else if (std::strcmp(name, "--steps") == 0) {
            args.steps = atoi(value);
        } else if (std::strcmp(name, "--temperature") == 0) {
            args.temperature = atof(value);
        } else if (std::strcmp(name, "--topp") == 0) {
            args.topp = atof(value);
        } else if (std::strcmp(name, "--seed") == 0) {
            args.seed = atoll(value);
        } else if (std::strcmp(name, "--chat-template") == 0) {
            args.chatTemplateType = parseChatTemplateType(value);
        } else if (std::strcmp(name, "--max-seq-len") == 0) {
            args.maxSeqLen = (unsigned int)atoi(value);
        } else if (std::strcmp(name, "--gpu-index") == 0) {
            args.gpuIndex = atoi(value);
        } else if (std::strcmp(name, "--gpu-segments") == 0) {
            char *separator = std::strstr(value, ":");
            if (separator == NULL)
                throw std::runtime_error("GPU segments expected in the format <from>:<to>");
            args.gpuSegmentFrom = atoi(value);
            args.gpuSegmentTo = atoi(separator + 1);
        } else if (std::strcmp(name, "--net-turbo") == 0) {
            args.netTurbo = atoi(value) == 1;
        } else if (std::strcmp(name, "--max-slots") == 0) {
            args.maxSlots = (NnUint)atoi(value);
        } else if (std::strcmp(name, "--policy") == 0) {
            args.policy = value;
        } else if (std::strcmp(name, "--rr-quantum") == 0) {
            args.rrQuantum = (NnUint)atoi(value);
        } else if (std::strcmp(name, "--sched-log") == 0) {
            args.schedLogPath = value;
        } else if (std::strcmp(name, "--mb-max-requests") == 0) {
            args.mbMaxRequests = (NnUint)atoi(value);
        } else if (std::strcmp(name, "--pp") == 0) {
            args.nStages = (NnUint)atoi(value);
        } else if (std::strcmp(name, "--pp-layers") == 0) {
            args.ppLayers = value;
        } else if (std::strcmp(name, "--pp-window") == 0) {
            args.ppWindow = (NnUint)atoi(value);
        } else if (std::strcmp(name, "--logits-log") == 0) {
            args.logitsLogPath = value;
        } else {
            throw std::runtime_error("Unknown option: " + std::string(name));
        }
    }

    if (args.nThreads < 1)
        throw std::runtime_error("Number of threads must be at least 1");
    if (args.maxSlots < 1)
        throw std::runtime_error("Number of slots must be at least 1");
    if (args.rrQuantum < 1)
        throw std::runtime_error("Round-robin quantum must be at least 1");
    if (args.ppWindow == 0)
        args.ppWindow = args.nStages > 0 ? args.nStages : 1;
    return args;
}

AppCliArgs::~AppCliArgs() {
    if (workerHosts != nullptr) {
        for (NnUint i = 0; i < nWorkers; i++)
            delete[] workerHosts[i];
        delete[] workerHosts;
    }
    if (workerPorts != nullptr)
        delete[] workerPorts;
}

static std::vector<NnExecutorDevice> resolveDevices(AppCliArgs *args, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution) {
    std::vector<NnExecutorDevice> devices;

    if (args->gpuIndex >= 0) {
#if defined(DLLAMA_VULKAN)
        devices.push_back(NnExecutorDevice(
            new NnVulkanDevice(args->gpuIndex, netConfig, nodeConfig, netExecution),
            args->gpuSegmentFrom,
            args->gpuSegmentTo
        ));
#else
        throw std::runtime_error("This build does not support GPU");
#endif
    }

    if (args->gpuIndex < 0 || (args->gpuSegmentFrom >= 0 && args->gpuSegmentTo >= 0)) {
        devices.push_back(NnExecutorDevice(new NnCpuDevice(netConfig, nodeConfig, netExecution), -1, -1));
    }
    return devices;
}

static uint64_t appNowNs() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static NnSize getPipeRowBytes(NnNetConfig *netConfig, NnUint pipeIndex) {
    const NnSize3D *size = &netConfig->pipes[pipeIndex].size;
    return getBytes(size->floatType, size->x);
}

RootLlmInference::RootLlmInference(LlmNet *net, NnNetExecution *execution, NnExecutor *executor, NnNetwork *network) {
    this->header = net->header;
    this->tokenPipe = (float *)execution->pipes[net->tokenPipeIndex];
    this->positionPipe = (float *)execution->pipes[net->positionPipeIndex];
    this->logitsPipe = (float *)execution->pipes[net->logitsPipeIndex];
    this->execution = execution;
    this->executor = executor;
    this->network = network; // May be nullptr!
    this->nSlots = net->nSlots;
    // The control packet has room for the microbatch id of the pipeline header
    this->controlPacket.reset(new NnUint[LLM_PIPELINE_HEADER_LENGTH(execution->nBatches)]());
    this->controlPacketSize = LLM_CONTROL_PACKET_LENGTH(execution->nBatches) * sizeof(NnUint);

    this->nStages = net->nStages;
    this->xPipe = execution->pipes[net->xPipeIndex];
    this->xRowBytes = getPipeRowBytes(&net->netConfig, net->xPipeIndex);
    this->logitsRowBytes = getPipeRowBytes(&net->netConfig, net->logitsPipeIndex);
    this->pipelineHeaderSize = LLM_PIPELINE_HEADER_LENGTH(execution->nBatches) * sizeof(NnUint);
    this->lastStageSocket = nStages > 0 ? nStages - 1 : 0;
    this->nextSubmitId = 0;
    this->nextCollectId = 0;
    this->isFinished = false;
    this->isReceiverDone = false;
    assert(nStages == 0 || network != nullptr);
}

RootLlmInference::~RootLlmInference() {
    if (receiver.joinable()) {
        // Left by an error: stop the stages, then the receiver gets the stop signal back
        try {
            finish();
        } catch (const std::exception &e) {
            printf("🚨 Cannot stop the pipeline: %s\n", e.what());
        }
        if (receiver.joinable())
            receiver.join();
    }
}

void RootLlmInference::setBatchSize(NnUint batchSize) {
    execution->setBatchSize(batchSize);
    controlPacket[0] = batchSize;
}

void RootLlmInference::setPosition(NnUint position) {
    assert(position >= 0);
    assert(position + execution->batchSize - 1 < header->seqLen);

    for (NnUint i = 0; i < execution->batchSize; i++)
        setRowPosition(i, position + i);
}

void RootLlmInference::setRowPosition(NnUint batchIndex, NnUint position) {
    assert(batchIndex < execution->batchSize);
    assert(position < header->seqLen * nSlots);
    positionPipe[batchIndex] = (float)position;
    controlPacket[1 + batchIndex] = position;
}

void RootLlmInference::setToken(NnUint batchIndex, NnUint token) {
    assert(batchIndex >= 0 && batchIndex < execution->batchSize);
    tokenPipe[batchIndex] = (float)token;
}

void RootLlmInference::forward() {
    submit();
    collect();
}

void RootLlmInference::submit() {
    if (nStages == 0) {
        if (network != nullptr) 
            network->writeAll(controlPacket.get(), controlPacketSize);
        executor->forward();
        return;
    }
    if (!receiver.joinable())
        receiver = std::thread(&RootLlmInference::receiveLoop, this);
    executor->forward(); // embeds the tokens into X
    controlPacket[LLM_CONTROL_PACKET_LENGTH(execution->nBatches)] = nextSubmitId++;
    network->write(0, controlPacket.get(), pipelineHeaderSize);
    network->write(0, xPipe, execution->batchSize * xRowBytes);
}

bool RootLlmInference::tryCollect(std::chrono::microseconds timeout) {
    if (nStages == 0)
        return true;
    std::unique_ptr<LlmPipelineResult> result;
    {
        std::unique_lock<std::mutex> lock(resultsMutex);
        if (!resultsCondition.wait_for(lock, timeout, [this] { return !results.empty() || isReceiverDone; }))
            return false;
        if (results.empty())
            throw NnExecutorException("The pipeline receiver stopped: " + receiverError);
        result = std::move(results.front());
        results.pop_front();
    }
    if (result->id != nextCollectId)
        throw NnExecutorException("The pipeline returned microbatch " + std::to_string(result->id) +
            ", expected " + std::to_string(nextCollectId));
    nextCollectId++;
    std::memcpy(logitsPipe, result->logits.data(), result->logits.size());
    return true;
}

void RootLlmInference::collect() {
    while (!tryCollect(std::chrono::microseconds(1000000)));
}

// Reads every microbatch output from the last stage as soon as it arrives, so the last stage can
// always send. Otherwise the ring root -> stages -> root could deadlock when the root blocks on
// sending to stage 0 while the last stage blocks on sending to the root.
void RootLlmInference::receiveLoop() {
    const NnUint headerLength = LLM_PIPELINE_HEADER_LENGTH(execution->nBatches);
    std::vector<NnUint> packet(headerLength);
    std::string error = "stopped";
    try {
        while (true) {
            if (!network->tryReadWithMaxAttempts(lastStageSocket, packet.data(), pipelineHeaderSize, 10000))
                continue;
            const NnUint batchSize = packet[0];
            if (batchSize == 0)
                break; // the stop signal went through every stage
            if (batchSize > execution->nBatches)
                throw std::runtime_error("Invalid batch size from the last stage");
            std::unique_ptr<LlmPipelineResult> result(new LlmPipelineResult());
            result->id = packet[headerLength - 1];
            result->batchSize = batchSize;
            result->logits.resize(batchSize * logitsRowBytes);
            network->read(lastStageSocket, result->logits.data(), result->logits.size());
            {
                std::lock_guard<std::mutex> lock(resultsMutex);
                results.push_back(std::move(result));
            }
            resultsCondition.notify_one();
        }
    } catch (const std::exception &e) {
        error = e.what();
    }
    {
        std::lock_guard<std::mutex> lock(resultsMutex);
        isReceiverDone = true;
        receiverError = error;
    }
    resultsCondition.notify_all();
}

void RootLlmInference::finish() {
    if (isFinished)
        return;
    isFinished = true;
    controlPacket[0] = 0;
    if (nStages == 0) {
        if (network != nullptr)
            network->writeAll(controlPacket.get(), controlPacketSize);
        return;
    }
    network->write(0, controlPacket.get(), pipelineHeaderSize);
    if (receiver.joinable())
        receiver.join();
}

WorkerLlmInference::WorkerLlmInference(NnNetExecution *execution, NnNetwork *network) {
    this->isFinished = false;
    this->execution = execution;
    this->network = network;
    this->positionPipe = (float *)execution->pipes[0];
    const NnUint length = LLM_CONTROL_PACKET_LENGTH(execution->nBatches);
    this->controlPacket.reset(new NnUint[length]());
    this->controlPacketSize = length * sizeof(NnUint);
}

bool WorkerLlmInference::tryReadControlPacket() {
    const unsigned long maxAttempts = 10000;
    if (!network->tryReadWithMaxAttempts(ROOT_SOCKET_INDEX, controlPacket.get(), controlPacketSize, maxAttempts))
        return false;
    const NnUint batchSize = controlPacket[0];
    if (batchSize == 0) {
        printf("🛑 Stop signal\n");
        isFinished = true;
        return true;
    }
    assert(batchSize <= execution->nBatches);
    for (NnUint i = 0; i < batchSize; i++)
        positionPipe[i] = (float)controlPacket[1 + i];
    execution->setBatchSize(batchSize);
    return true;
}

void runInferenceApp(AppCliArgs *args, void (*handler)(AppInferenceContext *context)) {
    NnUint nNodes = args->nWorkers + 1;
    const bool isPipeline = args->nStages > 0;

    LlmHeader header = loadLlmHeader(args->modelPath, args->maxSeqLen, args->syncType);
    if (isPipeline) {
        if (args->nWorkers != args->nStages)
            throw std::runtime_error("--pp " + std::to_string(args->nStages) + " needs exactly " +
                std::to_string(args->nStages) + " workers, one per stage");
    } else if (nNodes > header.nKvHeads)
        // TODO: https://github.com/b4rtaz/distributed-llama/issues/70
        throw std::runtime_error("This version does not support more nodes than the number of KV heads in the model");
    if (header.weightType == F_Q40 && header.syncType != F_Q80)
        throw std::runtime_error("This version supports only Q40 weights with Q80 sync type");

    Tokenizer tokenizer(args->tokenizerPath);
    if (args->info && tokenizer.vocabSize != header.vocabSize)
        printf("Tokenizer vocab size (%d) does not match the model vocab size (%d)\n", tokenizer.vocabSize, header.vocabSize);

    Sampler sampler(tokenizer.vocabSize, args->temperature, args->topp, args->seed);

    // Positions travel as floats in the position pipe, which are exact only below 2^24
    if ((unsigned long long)header.seqLen * args->maxSlots >= (1ull << 24))
        throw std::runtime_error("seqLen * maxSlots must be below 2^24");
    LlmNet net;
    if (isPipeline) {
        // Pipeline parallelism: the root only embeds tokens and samples, worker s runs stage s
        std::vector<NnUint> stageEnds = args->ppLayers != nullptr
            ? parseLlmStageLayers(args->ppLayers, &header, args->nStages)
            : splitLlmLayers(&header, args->nStages);
        net = buildLlmPipelineNet(&header, args->nBatches, args->maxSlots, stageEnds);
    } else {
        net = buildLlmNet(&header, nNodes, args->nBatches, args->maxSlots);
    }
    if (args->info && args->maxSlots > 1)
        printf("💡 KvCacheSlots: %u\n", args->maxSlots);
    std::unique_ptr<LlmNet, void(*)(LlmNet *)> netPtr(&net, releaseLlmNet);

    NnNodeConfig *rootNodeConfig = &net.nodeConfigs[0];

    if (args->info) {
        tokenizer.printHeader();
        printLlmHeader(&header);
        printNodeRequiredMemory(&net.netConfig, rootNodeConfig);
        for (NnUint stageIndex = 0; stageIndex < net.nStages; stageIndex++) {
            printf("🧩 Stage %u: layers %u-%u%s, ", stageIndex, llmStageStart(net.stageEnds, stageIndex),
                net.stageEnds[stageIndex] - 1, stageIndex + 1 == net.nStages ? " + logits" : "");
            printNodeRequiredMemory(&net.netConfig, &net.nodeConfigs[stageIndex + 1]);
        }
    }

    NnNetExecution execution(args->nThreads, &net.netConfig);

    std::unique_ptr<NnNodeSynchronizer> synchronizer(nullptr);
    std::unique_ptr<NnNetwork> networkPtr(nullptr);
    NnNetwork *network = nullptr;

    if (nNodes == 1) {
        synchronizer.reset(new NnFakeNodeSynchronizer());
    } else {
        networkPtr = NnNetwork::connect(args->nWorkers, args->workerHosts, args->workerPorts);
        network = networkPtr.get();
        NnRootConfigWriter configWriter(network);

        if (isPipeline) {
            // The root's net is a single node: it runs the embedding only and syncs nothing
            synchronizer.reset(new NnFakeNodeSynchronizer());
            for (NnUint stageIndex = 0; stageIndex < net.nStages; stageIndex++) {
                configWriter.writeNet(stageIndex, &net.netConfig);
                configWriter.writeNode(stageIndex, &net.nodeConfigs[stageIndex + 1]);
            }
            for (NnUint stageIndex = 0; stageIndex < net.nStages; stageIndex++) {
                const bool isLast = stageIndex + 1 == net.nStages;
                LlmWorkerRole role;
                role.isStage = 1;
                role.stageIndex = stageIndex;
                role.nStages = net.nStages;
                role.layerStart = llmStageStart(net.stageEnds, stageIndex);
                role.layerEnd = net.stageEnds[stageIndex];
                // Workers form a full mesh (NnNetwork::serve): on worker s socket 0 is the root,
                // socket j + 1 is worker j < s and socket j is worker j > s. So stage s reads from
                // socket s (the root for stage 0) and writes to socket s + 1, the last one to the root.
                role.inSocket = stageIndex;
                role.outSocket = isLast ? 0 : stageIndex + 1;
                role.positionPipeIndex = net.positionPipeIndex;
                role.inputPipeIndex = net.xPipeIndex;
                role.outputPipeIndex = isLast ? net.logitsPipeIndex : net.xPipeIndex;
                network->write(stageIndex, &role, sizeof(role));
            }
        } else {
            synchronizer.reset(new NnNetworkNodeSynchronizer(network, &execution, &net.netConfig, rootNodeConfig));
            configWriter.writeToWorkers(&net.netConfig, net.nodeConfigs);
            LlmWorkerRole role;
            std::memset(&role, 0, sizeof(role));
            for (NnUint socketIndex = 0; socketIndex < network->nSockets; socketIndex++)
                network->write(socketIndex, &role, sizeof(role));
        }
    }

    std::vector<NnExecutorDevice> devices = resolveDevices(args, &net.netConfig, rootNodeConfig, &execution);
    NnExecutor executor(&net.netConfig, rootNodeConfig, &devices, &execution, synchronizer.get(), args->benchmark);

    std::unique_ptr<NnRootWeightLoader> weightLoader(isPipeline
        ? new LlmPipelineWeightLoader(&executor, network, net.stageEnds)
        : new NnRootWeightLoader(&executor, network, nNodes));
    loadLlmNetWeight(args->modelPath, &net, weightLoader.get());

    RootLlmInference inference(&net, &execution, &executor, network);

    if (network != nullptr) {
        network->resetStats();
        if (args->netTurbo) {
            network->setTurbo(true);
            printf("🚁 Network is in non-blocking mode\n");
        }
    }

    AppInferenceContext context;
    context.args = args;
    context.header = &header;
    context.inference = &inference;
    context.sampler = &sampler;
    context.tokenizer = &tokenizer;
    context.network = network;
    context.executor = &executor;

    handler(&context);

    inference.finish();
}

// A pipeline stage: reads a microbatch from the previous node, runs its layers and sends the
// result on, one microbatch at a time and in arrival order
static void runPipelineStage(AppCliArgs *args, NnNetwork *network, NnNetConfig *netConfig, NnNetExecution *execution,
    NnExecutor *executor, const LlmWorkerRole *role) {
    const NnUint headerLength = LLM_PIPELINE_HEADER_LENGTH(execution->nBatches);
    const NnSize headerSize = headerLength * sizeof(NnUint);
    std::vector<NnUint> header(headerLength);
    float *positionPipe = (float *)execution->pipes[role->positionPipeIndex];
    NnByte *inputPipe = execution->pipes[role->inputPipeIndex];
    NnByte *outputPipe = execution->pipes[role->outputPipeIndex];
    const NnSize inputRowBytes = getPipeRowBytes(netConfig, role->inputPipeIndex);
    const NnSize outputRowBytes = getPipeRowBytes(netConfig, role->outputPipeIndex);

    std::unique_ptr<FILE, int(*)(FILE *)> stageLog(nullptr, fclose);
    if (args->schedLogPath != nullptr) {
        stageLog.reset(fopen(args->schedLogPath, "w"));
        if (!stageLog)
            throw std::runtime_error("Cannot open the stage log: " + std::string(args->schedLogPath));
        fprintf(stageLog.get(), "# stage=%u n_stages=%u layers=%u-%u clock=steady_ns\n",
            role->stageIndex, role->nStages, role->layerStart, role->layerEnd - 1);
        fprintf(stageLog.get(), "# mb,<id>,<batch_size>,<t_header>,<t_input>,<t_forward>,<t_output>\n");
        fflush(stageLog.get());
    }

    bool isFirstAttempt = true;
    bool isTurboEnabled = false;
    clock_t startTime;
    while (true) {
        if (isFirstAttempt)
            startTime = clock();
        if (!network->tryReadWithMaxAttempts(role->inSocket, header.data(), headerSize, 10000)) {
            if (isTurboEnabled && !isFirstAttempt && clock() - startTime > CLOCKS_PER_SEC) {
                network->setTurbo(false);
                isTurboEnabled = false;
                printf("🚁 Network is in blocking mode\n");
            }
            isFirstAttempt = false;
            continue;
        }
        isFirstAttempt = true;
        const uint64_t tHeader = appNowNs();

        const NnUint batchSize = header[0];
        if (batchSize == 0) {
            // Pass the stop signal on, the last stage back to the root
            network->write(role->outSocket, header.data(), headerSize);
            printf("🛑 Stop signal\n");
            break;
        }
        if (batchSize > execution->nBatches)
            throw std::runtime_error("Invalid batch size: " + std::to_string(batchSize));

        if (args->netTurbo && !isTurboEnabled) {
            network->setTurbo(true);
            isTurboEnabled = true;
            printf("🚁 Network is in non-blocking mode\n");
        }

        for (NnUint i = 0; i < batchSize; i++)
            positionPipe[i] = (float)header[1 + i];
        execution->setBatchSize(batchSize);
        network->read(role->inSocket, inputPipe, batchSize * inputRowBytes);
        const uint64_t tInput = appNowNs();

        executor->forward();
        const uint64_t tForward = appNowNs();

        network->write(role->outSocket, header.data(), headerSize);
        network->write(role->outSocket, outputPipe, batchSize * outputRowBytes);
        const uint64_t tOutput = appNowNs();

        if (stageLog) {
            fprintf(stageLog.get(), "mb,%u,%u,%llu,%llu,%llu,%llu\n", header[headerLength - 1], batchSize,
                (unsigned long long)tHeader, (unsigned long long)tInput,
                (unsigned long long)tForward, (unsigned long long)tOutput);
            fflush(stageLog.get());
        }
    }
}

void runWorkerApp(AppCliArgs *args) {
    while (true) {
        std::unique_ptr<NnNetwork> networkPtr = NnNetwork::serve(args->host, args->port);
        NnNetwork *network = networkPtr.get();

        NnWorkerConfigReader configReader(network);
        NnNetConfig netConfig = configReader.readNet();
        NnNodeConfig nodeConfig = configReader.readNode();
        std::unique_ptr<NnNetConfig, void(*)(NnNetConfig *)> netConfigPtr(&netConfig, releaseNetConfig);
        std::unique_ptr<NnNodeConfig, void(*)(NnNodeConfig *)> nodeConfigPtr(&nodeConfig, releaseNodeConfig);

        LlmWorkerRole role;
        network->read(ROOT_SOCKET_INDEX, &role, sizeof(role));
        if (role.isStage)
            printf("🧩 Pipeline stage %u of %u: layers %u-%u%s\n", role.stageIndex, role.nStages,
                role.layerStart, role.layerEnd - 1, role.stageIndex + 1 == role.nStages ? " + logits" : "");

        printNodeRequiredMemory(&netConfig, &nodeConfig);

        NnNetExecution execution(args->nThreads, &netConfig);

        std::vector<NnExecutorDevice> devices = resolveDevices(args, &netConfig, &nodeConfig, &execution);
        NnNetworkNodeSynchronizer synchronizer(network, &execution, &netConfig, &nodeConfig);
        NnExecutor executor(&netConfig, &nodeConfig, &devices, &execution, &synchronizer, false);

        NnWorkerWeightReader weightReader(&executor, network);
        weightReader.read();

        if (role.isStage) {
            try {
                runPipelineStage(args, network, &netConfig, &execution, &executor, &role);
            } catch (const NnTransferSocketException &e) {
                printf("🚨 Network error: %s\n", e.what());
            } catch (const NnExecutorException &e) {
                printf("🚨 Inference error: %s\n", e.what());
            }
            continue;
        }

        WorkerLlmInference inference(&execution, network);
        bool isFirstAttempt = true;
        bool isTurboEnabled = false;
        clock_t startTime;
        while (true) {
            try {
                if (isFirstAttempt)
                    startTime = clock();

                if (!inference.tryReadControlPacket()) {
                    if (isTurboEnabled && !isFirstAttempt && clock() - startTime > CLOCKS_PER_SEC) {
                        network->setTurbo(false);
                        isTurboEnabled = false;
                        printf("🚁 Network is in blocking mode\n");
                    }
                    isFirstAttempt = false;
                    continue;
                }
                if (inference.isFinished)
                    break;

                if (args->netTurbo && !isTurboEnabled) {
                    network->setTurbo(true);
                    isTurboEnabled = true;
                    printf("🚁 Network is in non-blocking mode\n");
                }
                executor.forward();
                isFirstAttempt = true;
            } catch (const NnTransferSocketException &e) {
                printf("🚨 Network error: %s\n", e.what());
                break;
            } catch (const NnExecutorException &e) {
                printf("🚨 Inference error: %s\n", e.what());
                break;
            }
        }
    }
}

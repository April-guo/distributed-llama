#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <csignal>
#include <thread>
#include <chrono>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#endif

#include "tokenizer.hpp"
#include "app.hpp"
#include "json.hpp"
#include "api-types.hpp"
#include "nn/nn-network.hpp"
#include "scheduler.hpp"

typedef unsigned int pos_t;

using json = nlohmann::json;

enum class HttpMethod {
    METHOD_GET = 0,
    METHOD_POST = 1,
    METHOD_PUT = 2,
    METHOD_DELETE = 3,
    METHOD_OPTIONS = 4,
    METHOD_UNKNOWN = 5
};

class HttpRequest {
public:
    static HttpRequest read(int serverSocket) {
        HttpRequest req(serverSocket);

        std::vector<char> httpRequest = req.readHttpRequest();
        // Parse the HTTP request
        std::string data = std::string(httpRequest.begin(), httpRequest.end());

        // Split request into lines
        std::istringstream iss(data);
        std::string line;
        std::getline(iss, line);

        // Parse request line
        std::istringstream lineStream(line);
        std::string methodStr, path;
        lineStream >> methodStr >> path;
        req.method = parseMethod(methodStr);
        req.path = path;

        // Parse headers
        while (std::getline(iss, line) && line != "\r") {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 2); // Skip ': ' after key
                // Trim whitespace and non-printable characters from header value
                value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char c) {
                    return std::isspace(c) || !std::isprint(c);
                }), value.end());
                req.headers[key] = value;
            }
        }

        // Parse body
        std::getline(iss, req.body, '\0');

        if (req.body.size() > 0) {
            // printf("body: %s\n", req.body.c_str());
            req.parsedJson = json::parse(req.body);
        }
        return req;
    }

    static HttpMethod parseMethod(const std::string& method) {
        if (method == "GET") return HttpMethod::METHOD_GET;
        if (method == "POST") return HttpMethod::METHOD_POST;
        if (method == "PUT") return HttpMethod::METHOD_PUT;
        if (method == "DELETE") return HttpMethod::METHOD_DELETE;
        if (method == "OPTIONS") return HttpMethod::METHOD_OPTIONS;
        return HttpMethod::METHOD_UNKNOWN;
    }

private:
    int serverSocket;
public:
    std::string path;
    std::unordered_map<std::string, std::string> headers;
    std::string body;
    json parsedJson;
    HttpMethod method;

    HttpRequest(int serverSocket) {
        this->serverSocket = serverSocket;
    }

    std::vector<char> readHttpRequest() {
        std::string httpRequest;
        char buffer[1024 * 64];
        ssize_t bytesRead;

        // First, read all headers
        std::string headerData;
        std::string extraReadPastHeader;
        for (;;) {
            bytesRead = recv(serverSocket, buffer, sizeof(buffer) - 1, 0);
            if (bytesRead <= 0) {
                throw std::runtime_error("Error while reading headers from socket");
            }
            buffer[bytesRead] = '\0';
            headerData.append(buffer);

            const size_t endRnRn = headerData.find("\r\n\r\n");
            if (endRnRn != std::string::npos) {
                if (endRnRn < headerData.size() - 4)
                    extraReadPastHeader = headerData.substr(endRnRn + 4);
                break;
            }
            const size_t endNN = headerData.find("\n\n");
            if (endNN != std::string::npos) {
                if (endNN < headerData.size() - 2)
                    extraReadPastHeader = headerData.substr(endNN + 2);
                break;
            }
        }

        httpRequest.append(headerData);

        // Next, find Content-Length header for body length
        std::istringstream headerStream(headerData);
        std::string line;
        ssize_t contentLength = 0;
        while (std::getline(headerStream, line) && line != "\r" && line != "\n") {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 2); // Skip ': ' after key
                if (key == "Content-Length") {
                    try {
                      contentLength = std::stoi(value);  // stoi ignores any whitespace
                    } catch (const std::invalid_argument& e) {
                      throw std::runtime_error("Bad Content-Length header - not a number");
                    }
                    break;
                }
            }
        }

        // Now read the full content body
        if (contentLength > 0) {
            // If we read any extra past the header before, read that much less now
            // But first, sanity check to make sure Content-Length isn't lying and there is actually more
            if (extraReadPastHeader.size() > static_cast<size_t>(contentLength)) {
                throw std::runtime_error("Received more body data than Content-Length header said");
            }
            contentLength -= extraReadPastHeader.size();

            std::vector<char> body(contentLength);
            ssize_t totalRead = 0;
            while (totalRead < contentLength) {
                bytesRead = recv(serverSocket, body.data() + totalRead, contentLength - totalRead, 0);
                if (bytesRead <= 0) {
                    throw std::runtime_error("Error while reading body from socket");
                }
                totalRead += bytesRead;
            }
            if (body.size() > 0) {
              httpRequest.append(body.data(), contentLength);
            }
        }

        return std::vector<char>(httpRequest.begin(), httpRequest.end());
    }

    std::string getMethod() {
        if (method == HttpMethod::METHOD_GET) return "GET";
        if (method == HttpMethod::METHOD_POST) return "POST";
        if (method == HttpMethod::METHOD_PUT) return "PUT";
        if (method == HttpMethod::METHOD_DELETE) return "DELETE";
        if (method == HttpMethod::METHOD_OPTIONS) return "OPTIONS";
        return "UNKNOWN";
    }
 
    void writeCors() {
        std::ostringstream buffer;
        buffer << "HTTP/1.1 204 No Content\r\n"
            << "Access-Control-Allow-Origin: *\r\n"
            << "Access-Control-Allow-Methods: GET, POST, PUT, DELETE\r\n"
            << "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
            << "Connection: close\r\n"
            << "\r\n";
        std::string data = buffer.str();
        writeSocket(serverSocket, data.c_str(), data.size());
    }

    void writeNotFound() {
        std::ostringstream buffer;
        buffer << "HTTP/1.1 404 Not Found\r\n"
            << "Connection: close\r\n"
            << "Content-Length: 9\r\n"
            << "\r\n"
            << "Not Found";
        std::string data = buffer.str();
        writeSocket(serverSocket, data.c_str(), data.size());
    }

    void writeJson(std::string json) {
        std::ostringstream buffer;
        buffer << "HTTP/1.1 200 OK\r\n"
            << "Access-Control-Allow-Origin: *\r\n"
            << "Content-Type: application/json; charset=utf-8\r\n"
            << "Connection: close\r\n"
            << "Content-Length: " << json.length() << "\r\n\r\n" << json;
        std::string data = buffer.str();
        writeSocket(serverSocket, data.c_str(), data.size());
    }

    void writeStreamStartChunk() {
        std::ostringstream buffer;
        buffer << "HTTP/1.1 200 OK\r\n"
            << "Access-Control-Allow-Origin: *\r\n"
            << "Content-Type: text/event-stream; charset=utf-8\r\n"
            << "Connection: close\r\n"
            << "Transfer-Encoding: chunked\r\n\r\n";
        std::string data = buffer.str();
        writeSocket(serverSocket, data.c_str(), data.size());
    }

    void writeStreamChunk(const std::string data) {
        std::ostringstream buffer;
        buffer << std::hex << data.size() << "\r\n" << data << "\r\n";
        std::string d = buffer.str();
        writeSocket(serverSocket, d.c_str(), d.size());
    }

    void writeStreamEndChunk() {
        const char *endChunk = "0000\r\n\r\n";
        writeSocket(serverSocket, endChunk, strlen(endChunk));
    }
};

void writeChatCompletionChunk(HttpRequest &request, const std::string &delta, const bool stop){
    ChunkChoice choice;
    if (stop) {
        choice.finish_reason = "stop";
    } else {
        choice.delta = ChatMessageDelta("assistant", delta);
        choice.has_delta = true;
    }
    ChatCompletionChunk chunk = ChatCompletionChunk(choice);

    std::ostringstream buffer;
    buffer << "data: " << ((json)chunk).dump() << "\r\n\r\n";
    request.writeStreamChunk(buffer.str());

    if (stop) {
        request.writeStreamChunk("data: [DONE]");
        request.writeStreamEndChunk();
    }
}

static bool tryParseJsonObject(const std::string &text, json &out) {
    json parsed = json::parse(text, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object())
        return false;
    out = std::move(parsed);
    return true;
}

static bool tryFindJsonFragmentAtEnd(const std::string &text, json &out) {
    if (tryParseJsonObject(text, out))
        return true;

    bool inString = false;
    int depth = 0;
    size_t end = std::string::npos;

    for (size_t i = text.size(); i-- > 0;) {
        char c = text[i];
        if (c == '"') {
            size_t backslashes = 0;
            for (size_t j = i; j > 0 && text[j - 1] == '\\'; j--)
                backslashes++;
            if ((backslashes % 2) == 0)
                inString = !inString;
            continue;
        }
        if (inString)
            continue;
        if (c == '}') {
            if (end == std::string::npos)
                end = i;
            depth++;
            continue;
        }
        if (c == '{') {
            if (depth == 0)
                continue;
            depth--;
            if (depth == 0 && end != std::string::npos) {
                std::string candidate = text.substr(i, end - i + 1);
                if (tryParseJsonObject(candidate, out))
                    return true;
                end = std::string::npos;
            }
        }
    }
    return false;
}

static std::string tryBuildToolsSystemPrompt(const std::vector<Tool> &tools, const ToolChoice &choice) {
    json toolJson = tools;
    std::string prompt = "You have access to the following tools:\n";
    prompt += toolJson.dump();
    prompt += "\n\nWhen you decide to call a tool, respond with a JSON object in this format:\n";
    prompt += "{\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"tool_name\",\"arguments\":\"{...}\"}}]}";
    if (choice.kind == TOOL_CHOICE_NONE)
        prompt += "\nDo not call any tools.";
    else if (choice.kind == TOOL_CHOICE_REQUIRED)
        prompt += "\nYou must call a tool.";
    else if (choice.kind == TOOL_CHOICE_NAMED)
        prompt += "\nYou must call the tool named: " + choice.tool_name + ".";
    return prompt;
}

static uint64_t nowNs() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

void handleModelsRequest(HttpRequest& request, const char* modelPath) {
    std::string path(modelPath);
    size_t pos = path.find_last_of("/\\");
    std::string modelName = (pos == std::string::npos) ? path : path.substr(pos + 1);

    Model model(modelName);
    ModelList list(model);
    std::string response = ((json)list).dump();
    request.writeJson(response);
}

// A chat completion read by the HTTP thread, waiting for a KV-cache slot
struct PendingCompletion {
    std::unique_ptr<NnSocket> socket;
    HttpRequest request;
    uint64_t arrivalNs;

    PendingCompletion(NnSocket *socket, const HttpRequest &request, uint64_t arrivalNs)
        : socket(socket), request(request), arrivalNs(arrivalNs) {}
};

class PendingQueue {
private:
    std::mutex mutex;
    std::condition_variable condition;
    std::deque<std::unique_ptr<PendingCompletion>> items;

public:
    void push(std::unique_ptr<PendingCompletion> item) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            items.push_back(std::move(item));
        }
        condition.notify_one();
    }

    // Returns nullptr if the queue is empty and wait is false
    std::unique_ptr<PendingCompletion> pop(bool wait) {
        std::unique_lock<std::mutex> lock(mutex);
        if (wait)
            condition.wait(lock, [this] { return !items.empty(); });
        if (items.empty())
            return nullptr;
        std::unique_ptr<PendingCompletion> item = std::move(items.front());
        items.pop_front();
        return item;
    }
};

// An admitted request: it owns a KV-cache slot and its own sampling and streaming state
struct ApiRequest : SchedRequest {
    std::unique_ptr<NnSocket> socket;
    HttpRequest http;
    InferenceParams params;
    NnUint slot;
    std::vector<int> promptTokens;
    int nPromptTokens;
    pos_t pos; // tokens already in the slot's KV cache
    pos_t promptEndPos;
    pos_t maxPredPos;
    int token; // the next decode input
    std::unique_ptr<Sampler> sampler;
    std::unique_ptr<EosDetector> eosDetector;
    std::unique_ptr<TokenDecoder> decoder;
    std::string buffer;
    uint64_t arrivalNs;
    bool hasFirstToken;
    bool failed;
    bool finished;

    ApiRequest(const HttpRequest &http) : http(http) {}
};

// Runs every forward of the root node. Before each forward the scheduling policy picks which
// admitted requests contribute rows; rows of one batch may belong to different requests.
class ApiServer {
private:
    RootLlmInference *inference;
    Tokenizer *tokenizer;
    AppCliArgs *args;
    LlmHeader *header;
    TokenizerChatStops *stops;
    ChatTemplateGenerator *templateGenerator;
    PendingQueue *pending;
    std::unique_ptr<SchedPolicy> policy;
    std::vector<bool> isSlotUsed;
    std::vector<std::unique_ptr<ApiRequest>> active; // in admission order
    NnUint nextRequestId;
    NnUint step;
    FILE *schedLog;

public:
    ApiServer(RootLlmInference *inference, Tokenizer *tokenizer, AppCliArgs *args, LlmHeader *header,
        TokenizerChatStops *stops, ChatTemplateGenerator *templateGenerator, PendingQueue *pending) {
        this->inference = inference;
        this->tokenizer = tokenizer;
        this->args = args;
        this->header = header;
        this->stops = stops;
        this->templateGenerator = templateGenerator;
        this->pending = pending;
        this->policy.reset(createSchedPolicy(args->policy, args->rrQuantum));
        this->isSlotUsed.assign(args->maxSlots, false);
        this->nextRequestId = 0;
        this->step = 0;
        this->schedLog = nullptr;
        if (args->schedLogPath != nullptr) {
            schedLog = fopen(args->schedLogPath, "w");
            if (schedLog == nullptr)
                throw std::runtime_error("Cannot open the scheduler log: " + std::string(args->schedLogPath));
            fprintf(schedLog, "# policy=%s rr_quantum=%u max_slots=%u n_batches=%u seq_len=%u clock=steady_ns\n",
                policy->name(), args->rrQuantum, args->maxSlots, args->nBatches, header->seqLen);
            fprintf(schedLog, "# forward,<step>,<t_start>,<t_end>,<batch_size>,<request>:<p|d>:<rows>;...\n");
            fprintf(schedLog, "# request,<id>,<arrive|admit|first_token|finish>,<t>,<detail>\n");
            fflush(schedLog);
        }
        printf("🗓️ Scheduler: %s (rr quantum %u), %u KV-cache slots\n", policy->name(), args->rrQuantum, args->maxSlots);
        if (strcmp(policy->name(), "serial") != 0 && args->maxSlots < 2)
            printf("⚠️ With 1 KV-cache slot only one request is served at a time, use --max-slots 2 or more\n");
    }

    ~ApiServer() {
        if (schedLog != nullptr)
            fclose(schedLog);
    }

    void run() {
        std::vector<SchedRequest *> view;
        std::vector<SchedItem> items;
        while (true) {
            if (active.empty()) {
                admit(pending->pop(true));
            }
            while (hasFreeSlot()) {
                std::unique_ptr<PendingCompletion> next = pending->pop(false);
                if (!next)
                    break;
                admit(std::move(next));
            }
            removeFinished();
            if (active.empty())
                continue;

            view.clear();
            for (size_t i = 0; i < active.size(); i++)
                view.push_back(active[i].get());
            items.clear();
            policy->pick(view, args->nBatches, items);
            forward(items);
            removeFinished();
        }
    }

private:
    bool hasFreeSlot() {
        return std::find(isSlotUsed.begin(), isSlotUsed.end(), false) != isSlotUsed.end();
    }

    void admit(std::unique_ptr<PendingCompletion> next) {
        std::unique_ptr<ApiRequest> request(new ApiRequest(next->request));
        ApiRequest *r = request.get();
        r->socket = std::move(next->socket);
        r->id = nextRequestId++;
        r->arrivalNs = next->arrivalNs;
        r->hasFirstToken = false;
        r->failed = false;
        r->finished = false;
        logRequest(r, "arrive", r->arrivalNs, "");

        try {
            r->params = parseRequest(r->http);

            size_t nInputItems = r->params.messages.size();
            std::unique_ptr<ChatItem[]> inputItemsPtr(new ChatItem[nInputItems]);
            ChatItem *inputItems = inputItemsPtr.get();
            for (size_t i = 0; i < nInputItems; i++) {
                inputItems[i].role = r->params.messages[i].role;
                inputItems[i].message = r->params.messages[i].content;
            }

            GeneratedChat inputPrompt = templateGenerator->generate(nInputItems, inputItems, true);
            printf("🔹\033[34m%s", inputPrompt.content);
            fflush(stdout);

            r->promptTokens.resize(inputPrompt.length + 2);
            tokenizer->encode((char*)inputPrompt.content, r->promptTokens.data(), &r->nPromptTokens, true, true);

            r->promptEndPos = r->nPromptTokens - 1;
            if (r->promptEndPos > header->seqLen)
                r->promptEndPos = header->seqLen;
            r->maxPredPos = r->params.max_tokens > 0 ? (r->promptEndPos + r->params.max_tokens) : header->seqLen;
            if (r->maxPredPos > header->seqLen)
                r->maxPredPos = header->seqLen;

            // Prefill writes all prompt tokens but the last one, which is the first decode input
            r->pos = 0;
            r->token = r->promptTokens[0];
            r->nPrefillLeft = r->promptEndPos;
            r->phase = r->nPrefillLeft > 0 ? SCHED_PREFILL : SCHED_DECODE;

            r->sampler.reset(new Sampler(tokenizer->vocabSize, args->temperature, args->topp, r->params.seed));
            r->eosDetector.reset(new EosDetector(stops->nStops, tokenizer->eosTokenIds.data(), stops->stops, stops->maxStopLength, stops->maxStopLength));
            r->decoder.reset(new TokenDecoder(tokenizer));

            if (r->params.stream)
                r->http.writeStreamStartChunk();
            if (inputPrompt.publicPrompt != nullptr) {
                if (r->params.stream)
                    writeChatCompletionChunk(r->http, inputPrompt.publicPrompt, false);
                r->buffer += inputPrompt.publicPrompt;
            }
            printf("(%d tokens)\033\n[0m🔸", r->promptEndPos);
            fflush(stdout);
        } catch (const std::exception &e) {
            printf("🚨 Request %u rejected: %s\n", r->id, e.what());
            logRequest(r, "finish", nowNs(), "n_out=0;reason=error");
            return; // closes the socket
        }

        r->slot = (NnUint)(std::find(isSlotUsed.begin(), isSlotUsed.end(), false) - isSlotUsed.begin());
        assert(r->slot < isSlotUsed.size());
        isSlotUsed[r->slot] = true;
        char detail[128];
        snprintf(detail, sizeof(detail), "slot=%u;prompt_tokens=%d;max_pred_pos=%u", r->slot, r->nPromptTokens, r->maxPredPos);
        logRequest(r, "admit", nowNs(), detail);
        active.push_back(std::move(request));

        if (r->phase == SCHED_DECODE && r->pos >= r->maxPredPos)
            finish(r, "max");
    }

    void forward(const std::vector<SchedItem> &items) {
        std::vector<NnUint> firstRows(items.size());
        NnUint nRows = 0;
        for (size_t i = 0; i < items.size(); i++) {
            firstRows[i] = nRows;
            nRows += items[i].nTokens;
        }
        assert(nRows > 0 && nRows <= args->nBatches);

        inference->setBatchSize(nRows);
        for (size_t i = 0; i < items.size(); i++) {
            ApiRequest *r = static_cast<ApiRequest *>(items[i].request);
            const NnUint slotStart = r->slot * header->seqLen;
            for (NnUint j = 0; j < items[i].nTokens; j++) {
                const NnUint row = firstRows[i] + j;
                inference->setRowPosition(row, slotStart + r->pos + j);
                inference->setToken(row, r->phase == SCHED_PREFILL ? r->promptTokens[r->pos + j] : r->token);
            }
        }

        const uint64_t tStart = nowNs();
        inference->forward();
        const uint64_t tEnd = nowNs();
        logForward(items, nRows, tStart, tEnd);
        step++;

        for (size_t i = 0; i < items.size(); i++) {
            ApiRequest *r = static_cast<ApiRequest *>(items[i].request);
            if (r->phase == SCHED_PREFILL)
                advancePrefill(r, items[i].nTokens);
            else
                decode(r, firstRows[i]);
        }
    }

    void advancePrefill(ApiRequest *r, NnUint nTokens) {
        r->pos += nTokens;
        r->nPrefillLeft -= nTokens;
        if (r->nPrefillLeft > 0)
            return;
        r->phase = SCHED_DECODE;
        r->token = r->promptTokens[r->pos];
        if (r->pos >= r->maxPredPos)
            finish(r, "max");
    }

    void decode(ApiRequest *r, NnUint row) {
        r->token = r->sampler->sample(inference->logitsPipe + (NnSize)row * header->vocabSize);

        char *piece = r->decoder->decode(r->token);
        EosDetectorType eosType = r->eosDetector->append(r->token, piece);

        if (piece != nullptr) {
            printf("%s", piece);
            fflush(stdout);
        }

        if (eosType == NOT_EOS || eosType == EOS) {
            char *delta = r->eosDetector->getDelta();
            if (delta != nullptr) {
                std::string deltaStr(delta);
                if (r->params.stream)
                    writeChunk(r, deltaStr, false);
                r->buffer += deltaStr;
            }
            r->eosDetector->reset();
        }
        r->pos++;
        if (!r->hasFirstToken) {
            r->hasFirstToken = true;
            logRequest(r, "first_token", nowNs(), "");
        }

        if (r->failed)
            finish(r, "error");
        else if (eosType == EOS)
            finish(r, "eos");
        else if (r->pos >= r->maxPredPos)
            finish(r, "max");
    }

    void writeChunk(ApiRequest *r, const std::string &delta, bool stop) {
        if (r->failed)
            return;
        try {
            writeChatCompletionChunk(r->http, delta, stop);
        } catch (const NnTransferSocketException &e) {
            printf("🚨 Request %u: %s\n", r->id, e.what());
            r->failed = true;
        }
    }

    void finish(ApiRequest *r, const char *reason) {
        ChatMessage reply("assistant", r->buffer);
        if (r->params.stream) {
            writeChunk(r, "", true);
        } else if (!r->failed) {
            Choice choice(reply);

            if (!r->params.tools.empty()) {
                std::vector<ToolCall> parsedToolCalls;
                json tempJson;
                if (tryFindJsonFragmentAtEnd(r->buffer, tempJson) &&
                    tryParseToolCallsFromJson(tempJson, parsedToolCalls)) {
                    choice.message.tool_calls = parsedToolCalls;
                    choice.finish_reason = "tool_calls";
                }
            }

            int nCompletionTokens = r->pos - r->promptEndPos;
            ChatUsage usage(r->nPromptTokens, nCompletionTokens, r->nPromptTokens + nCompletionTokens);
            ChatCompletion completion(choice, usage);
            std::string chatJson = ((json)completion).dump();
            try {
                r->http.writeJson(chatJson);
            } catch (const NnTransferSocketException &e) {
                printf("🚨 Request %u: %s\n", r->id, e.what());
            }
        }
        printf("🔶\n");
        fflush(stdout);

        char detail[96];
        snprintf(detail, sizeof(detail), "n_out=%u;reason=%s", r->pos - r->promptEndPos, reason);
        logRequest(r, "finish", nowNs(), detail);
        r->finished = true;
    }

    void removeFinished() {
        for (size_t i = 0; i < active.size();) {
            if (active[i]->finished) {
                policy->onRemove(i);
                isSlotUsed[active[i]->slot] = false;
                active.erase(active.begin() + i);
            } else {
                i++;
            }
        }
    }

    void logRequest(ApiRequest *r, const char *event, uint64_t t, const char *detail) {
        if (schedLog == nullptr)
            return;
        fprintf(schedLog, "request,%u,%s,%llu,%s\n", r->id, event, (unsigned long long)t, detail);
        fflush(schedLog);
    }

    void logForward(const std::vector<SchedItem> &items, NnUint nRows, uint64_t tStart, uint64_t tEnd) {
        if (schedLog == nullptr)
            return;
        fprintf(schedLog, "forward,%u,%llu,%llu,%u,", step, (unsigned long long)tStart, (unsigned long long)tEnd, nRows);
        for (size_t i = 0; i < items.size(); i++)
            fprintf(schedLog, "%s%u:%c:%u", i == 0 ? "" : ";", items[i].request->id,
                items[i].request->phase == SCHED_PREFILL ? 'p' : 'd', items[i].nTokens);
        fprintf(schedLog, "\n");
        fflush(schedLog);
    }

    InferenceParams parseRequest(HttpRequest& request) {
        InferenceParams params = parseInferenceParams(
            request.parsedJson,
            args->temperature,
            args->topp,
            args->seed);
        if (!params.tools.empty()) {
            std::string prompt = tryBuildToolsSystemPrompt(params.tools, params.tool_choice);
            ChatMessage toolMessage("system", prompt);
            params.messages.insert(params.messages.begin(), toolMessage);
        }
        return params;
    }
};

// Reads HTTP requests and hands chat completions to the scheduler, so requests are accepted
// while others are being generated
static void acceptLoop(int serverSocket, PendingQueue *pending, const char *modelPath, std::atomic<bool> *isStopping) {
    while (!isStopping->load()) {
        try {
            std::unique_ptr<NnSocket> clientSocket(new NnSocket(acceptSocket(serverSocket)));
            HttpRequest request = HttpRequest::read(clientSocket->fd);
            const uint64_t arrivalNs = nowNs();
            printf("🔷 %s %s\n", request.getMethod().c_str(), request.path.c_str());

            if (request.method == HttpMethod::METHOD_OPTIONS) {
                request.writeCors();
            } else if (request.method == HttpMethod::METHOD_POST && request.path == "/v1/chat/completions") {
                pending->push(std::unique_ptr<PendingCompletion>(
                    new PendingCompletion(clientSocket.release(), request, arrivalNs)));
            } else if (request.method == HttpMethod::METHOD_GET && request.path == "/v1/models") {
                handleModelsRequest(request, modelPath);
            } else {
                request.writeNotFound();
            }
        } catch (const std::exception &e) {
            if (!isStopping->load())
                printf("Socket error: %s\n", e.what());
        }
    }
}

static void server(AppInferenceContext *context) {
    NnSocket serverSocket(createServerSocket(context->args->host, context->args->port));

    TokenizerChatStops stops(context->tokenizer);
    ChatTemplateGenerator templateGenerator(context->args->chatTemplateType, context->tokenizer->chatTemplate, stops.stops[0]);
    PendingQueue pending;
    ApiServer api(context->inference, context->tokenizer, context->args, context->header, &stops, &templateGenerator, &pending);

    if (strcmp(context->args->host, "0.0.0.0") == 0 ||
        strcmp(context->args->host, "127.0.0.1") == 0)
        printf("Server URL: http://localhost:%d/v1/\n", context->args->port);

    std::atomic<bool> isStopping(false);
    std::thread acceptor(acceptLoop, serverSocket.fd, &pending, context->args->modelPath, &isStopping);
    try {
        api.run();
    } catch (...) {
        isStopping.store(true);
        shutdown(serverSocket.fd, 2); // unblocks accept()
        acceptor.join();
        throw;
    }
}

#ifdef _WIN32
    #define EXECUTABLE_NAME "dllama-api.exe"
#else
    #define EXECUTABLE_NAME "dllama-api"
#endif

void usage() {
    fprintf(stderr, "Usage: %s {--model <path>} {--tokenizer <path>} [--host <addr>] [--port <p>]\n", EXECUTABLE_NAME);
    fprintf(stderr, "        [--buffer-float-type {f32|f16|q40|q80}]\n");
    fprintf(stderr, "        [--weights-float-type {f32|f16|q40|q80}]\n");
    fprintf(stderr, "        [--max-seq-len <max>]\n");
    fprintf(stderr, "        [--nthreads <n>]\n");
    fprintf(stderr, "        [--workers <ip:port> ...]\n");
    fprintf(stderr, "        [--temperature <temp>]\n");
    fprintf(stderr, "        [--topp <t>]\n");
    fprintf(stderr, "        [--seed <s>]\n");
    fprintf(stderr, "        [--max-slots <n>] (KV-cache slots = requests served at once, default 1)\n");
    fprintf(stderr, "        [--policy {serial|rr|concurrent}] [--rr-quantum <forwards>]\n");
    fprintf(stderr, "        [--sched-log <csv path>]\n");
    fprintf(stderr, "Example:\n");
    fprintf(stderr, "  sudo nice -n -20 ./dllama-api --port 9990 --nthreads 4 \\\n");
    fprintf(stderr, "    --model dllama_model_llama3_2_3b_instruct_q40.m \\\n");
    fprintf(stderr, "    --tokenizer dllama_tokenizer_llama3_2_3b_instruct_q40.t \\\n");
    fprintf(stderr, "    --buffer-float-type q80 --max-seq-len 8192 \\\n");
    fprintf(stderr, "    --workers 10.0.0.2:9998 10.0.0.3:9998 10.0.0.4:9998\n");
    fflush(stderr);
}

int main(int argc, char *argv[]) {
#ifdef SIGPIPE
    std::signal(SIGPIPE, SIG_IGN);
#endif

    AppCliArgs args = AppCliArgs::parse(argc, argv, false);
    if (args.help) {
        usage();
        return EXIT_SUCCESS;
    }

    initQuants();
    initSockets();

    while (true) {
        try {
            runInferenceApp(&args, server);
        } catch (const NnConnectionSocketException &e) {
            printf("🚨 Connection error: %s\n", e.what());
        } catch (const NnExecutorException &e) {
            printf("🚨 Inference error: %s\n", e.what());
        }

        printf("🔄 Retrying in 3 seconds...\n");
        std::this_thread::sleep_for(std::chrono::seconds(3));
        args.info = false;
    }

    cleanupSockets();
    return EXIT_SUCCESS;
}

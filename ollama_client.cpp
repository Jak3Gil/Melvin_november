/*
 * Ollama API Client Implementation for Melvin Curiosity Learning System
 */

#include "ollama_client.h"
#include <curl/curl.h>
#include <json/json.h>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <random>
#include <cstdlib>
#include <cstring>

namespace MelvinOllama {

// RateLimiter Implementation
RateLimiter::RateLimiter(int requests_per_minute) 
    : min_interval(std::chrono::milliseconds(60000 / requests_per_minute)) {
    last_request_time = std::chrono::steady_clock::now() - min_interval;
}

void RateLimiter::waitForNextRequest() {
    std::lock_guard<std::mutex> lock(mutex_);
    auto now = std::chrono::steady_clock::now();
    auto time_since_last = now - last_request_time;
    
    if (time_since_last < min_interval) {
        auto wait_time = min_interval - time_since_last;
        std::this_thread::sleep_for(wait_time);
    }
    
    last_request_time = std::chrono::steady_clock::now();
}

bool RateLimiter::canMakeRequest() const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto now = std::chrono::steady_clock::now();
    return (now - last_request_time) >= min_interval;
}

// HTTP Response callback for libcurl
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* userp) {
    size_t total_size = size * nmemb;
    userp->append((char*)contents, total_size);
    return total_size;
}

// OllamaHttpClient Implementation
OllamaHttpClient::OllamaHttpClient(const OllamaConfig& config) 
    : base_url(config.base_url), 
      api_key(config.api_key),
      timeout_seconds(config.request_timeout_seconds),
      max_connections(config.max_connections) {
    initializeCurl();
}

OllamaHttpClient::~OllamaHttpClient() {
    cleanupCurl();
}

void OllamaHttpClient::initializeCurl() {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    
    // Initialize connection pool
    for (int i = 0; i < max_connections; ++i) {
        CURL* handle = curl_easy_init();
        if (handle) {
            curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(handle, CURLOPT_TIMEOUT, timeout_seconds);
            curl_easy_setopt(handle, CURLOPT_CONNECTTIMEOUT, 10);
            curl_easy_setopt(handle, CURLOPT_FOLLOWLOCATION, 1L);
            curl_easy_setopt(handle, CURLOPT_SSL_VERIFYPEER, 1L);
            curl_easy_setopt(handle, CURLOPT_SSL_VERIFYHOST, 2L);
            curl_easy_setopt(handle, CURLOPT_USERAGENT, "Melvin-Ollama-Client/1.0");
            
            connection_pool.push(handle);
        }
    }
}

void OllamaHttpClient::cleanupCurl() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    while (!connection_pool.empty()) {
        CURL* handle = connection_pool.front();
        connection_pool.pop();
        curl_easy_cleanup(handle);
    }
    curl_global_cleanup();
}

CURL* OllamaHttpClient::getConnection() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (!connection_pool.empty()) {
        CURL* handle = connection_pool.front();
        connection_pool.pop();
        active_connections++;
        return handle;
    }
    return nullptr;
}

void OllamaHttpClient::returnConnection(CURL* handle) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    connection_pool.push(handle);
    active_connections--;
}

OllamaResponse OllamaHttpClient::makeRequest(const std::string& endpoint, 
                                            const std::string& json_data) {
    OllamaResponse response;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    CURL* handle = getConnection();
    if (!handle) {
        response.error_message = "No available connections";
        return response;
    }
    
    std::string response_data;
    std::string url = base_url + endpoint;
    
    // Set up the request
    curl_easy_setopt(handle, CURLOPT_URL, url.c_str());
    curl_easy_setopt(handle, CURLOPT_POSTFIELDS, json_data.c_str());
    curl_easy_setopt(handle, CURLOPT_WRITEDATA, &response_data);
    
    // Set headers
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    if (!api_key.empty()) {
        std::string auth_header = "Authorization: Bearer " + api_key;
        headers = curl_slist_append(headers, auth_header.c_str());
    }
    curl_easy_setopt(handle, CURLOPT_HTTPHEADER, headers);
    
    // Perform the request
    CURLcode res = curl_easy_perform(handle);
    
    // Get response info
    long http_code = 0;
    curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &http_code);
    response.status_code = static_cast<int>(http_code);
    
    // Clean up
    curl_slist_free_all(headers);
    returnConnection(handle);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    response.response_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    if (res == CURLE_OK && http_code == 200) {
        response.success = true;
        response.content = response_data;
    } else {
        response.error_message = curl_easy_strerror(res);
        if (response.error_message.empty()) {
            response.error_message = "HTTP " + std::to_string(http_code);
        }
    }
    
    return response;
}

bool OllamaHttpClient::isHealthy() const {
    return active_connections < max_connections;
}

void OllamaHttpClient::setApiKey(const std::string& key) {
    api_key = key;
}

// AsyncRequestManager Implementation
AsyncRequestManager::AsyncRequestManager(int max_concurrent) 
    : max_concurrent_requests(max_concurrent) {
    
    // Start worker threads
    int num_threads = std::min(max_concurrent, 4);
    for (int i = 0; i < num_threads; ++i) {
        worker_threads.emplace_back(&AsyncRequestManager::workerThread, this);
    }
}

AsyncRequestManager::~AsyncRequestManager() {
    shutdown();
}

void AsyncRequestManager::workerThread() {
    while (!shutdown_) {
        std::function<void()> task;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { 
                return !request_queue.empty() || shutdown_; 
            });
            
            if (shutdown_) break;
            
            if (!request_queue.empty()) {
                task = request_queue.front();
                request_queue.pop();
            }
        }
        
        if (task) {
            active_requests++;
            task();
            active_requests--;
        }
    }
}

std::future<OllamaResponse> AsyncRequestManager::submitRequest(
    std::function<OllamaResponse()> request_func) {
    
    auto promise = std::make_shared<std::promise<OllamaResponse>>();
    auto future = promise->get_future();
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        request_queue.push([promise, request_func]() {
            try {
                OllamaResponse result = request_func();
                promise->set_value(result);
            } catch (...) {
                OllamaResponse error_response;
                error_response.success = false;
                error_response.error_message = "Async request failed";
                promise->set_value(error_response);
            }
        });
    }
    
    queue_cv_.notify_one();
    return future;
}

void AsyncRequestManager::shutdown() {
    shutdown_ = true;
    queue_cv_.notify_all();
    
    for (auto& thread : worker_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

int AsyncRequestManager::getActiveRequestCount() const {
    return active_requests;
}

// OllamaClient Implementation
OllamaClient::OllamaClient(const OllamaConfig& config) : config_(config) {
    loadApiKeyFromEnvironment();
    
    http_client_ = std::make_unique<OllamaHttpClient>(config_);
    rate_limiter_ = std::make_unique<RateLimiter>(config_.rate_limit_requests_per_minute);
    
    if (config_.enable_async) {
        async_manager_ = std::make_unique<AsyncRequestManager>(config_.max_async_requests);
    }
}

void OllamaClient::loadApiKeyFromEnvironment() {
    if (config_.api_key.empty()) {
        config_.api_key = Utils::loadApiKeyFromEnv();
    }
}

OllamaResponse OllamaClient::makeRequestWithRetry(const std::string& prompt) {
    OllamaResponse response;
    int attempts = 0;
    
    while (attempts < config_.max_retries) {
        rate_limiter_->waitForNextRequest();
        
        std::string request_json = buildGenerateRequest(prompt);
        response = http_client_->makeRequest("/api/generate", request_json);
        
        if (response.success) {
            response = parseGenerateResponse(response.content);
            successful_requests_++;
            break;
        } else {
            attempts++;
            retry_attempts_++;
            
            if (attempts < config_.max_retries) {
                int delay = config_.retry_delay_ms * (1 << (attempts - 1)); // Exponential backoff
                std::this_thread::sleep_for(std::chrono::milliseconds(delay));
            }
        }
    }
    
    if (!response.success) {
        failed_requests_++;
        response.error_message = "Failed after " + std::to_string(config_.max_retries) + " attempts";
    }
    
    total_requests_++;
    return response;
}

std::string OllamaClient::buildGenerateRequest(const std::string& prompt) const {
    Json::Value request;
    request["model"] = config_.model;
    request["prompt"] = prompt;
    request["stream"] = false;
    request["options"]["temperature"] = 0.7;
    request["options"]["top_p"] = 0.9;
    request["options"]["max_tokens"] = 512;
    
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "";
    return Json::writeString(builder, request);
}

OllamaResponse OllamaClient::parseGenerateResponse(const std::string& json_response) const {
    OllamaResponse response;
    
    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errors;
    
    std::istringstream stream(json_response);
    if (!Json::parseFromStream(builder, stream, &root, &errors)) {
        response.error_message = "Failed to parse JSON: " + errors;
        return response;
    }
    
    if (root.isMember("response")) {
        response.success = true;
        response.content = root["response"].asString();
        response.model_used = root.get("model", config_.model).asString();
        
        if (root.isMember("eval_count")) {
            response.tokens_generated = root["eval_count"].asInt();
        }
    } else if (root.isMember("error")) {
        response.error_message = root["error"].asString();
    } else {
        response.error_message = "Invalid response format";
    }
    
    return response;
}

OllamaResponse OllamaClient::generate(const std::string& prompt) {
    return makeRequestWithRetry(prompt);
}

OllamaResponse OllamaClient::askQuestion(const std::string& question) {
    std::string prompt = "Answer the following question clearly and concisely: " + question;
    return generate(prompt);
}

std::future<OllamaResponse> OllamaClient::generateAsync(const std::string& prompt) {
    if (!async_manager_) {
        // Fallback to synchronous if async is disabled
        auto promise = std::make_shared<std::promise<OllamaResponse>>();
        auto future = promise->get_future();
        
        std::thread([promise, this, prompt]() {
            OllamaResponse response = generate(prompt);
            promise->set_value(response);
        }).detach();
        
        return future;
    }
    
    return async_manager_->submitRequest([this, prompt]() {
        return generate(prompt);
    });
}

std::future<OllamaResponse> OllamaClient::askQuestionAsync(const std::string& question) {
    std::string prompt = "Answer the following question clearly and concisely: " + question;
    return generateAsync(prompt);
}

bool OllamaClient::isHealthy() const {
    return http_client_->isHealthy();
}

std::map<std::string, std::string> OllamaClient::getStatus() const {
    std::map<std::string, std::string> status;
    status["healthy"] = isHealthy() ? "true" : "false";
    status["model"] = config_.model;
    status["base_url"] = config_.base_url;
    status["total_requests"] = std::to_string(total_requests_);
    status["successful_requests"] = std::to_string(successful_requests_);
    status["failed_requests"] = std::to_string(failed_requests_);
    
    if (async_manager_) {
        status["active_async_requests"] = std::to_string(async_manager_->getActiveRequestCount());
    }
    
    return status;
}

void OllamaClient::updateConfig(const OllamaConfig& new_config) {
    config_ = new_config;
    loadApiKeyFromEnvironment();
    
    http_client_ = std::make_unique<OllamaHttpClient>(config_);
    rate_limiter_ = std::make_unique<RateLimiter>(config_.rate_limit_requests_per_minute);
    
    if (config_.enable_async && !async_manager_) {
        async_manager_ = std::make_unique<AsyncRequestManager>(config_.max_async_requests);
    }
}

OllamaClient::Statistics OllamaClient::getStatistics() const {
    Statistics stats;
    stats.total_requests = total_requests_;
    stats.successful_requests = successful_requests_;
    stats.failed_requests = failed_requests_;
    stats.retry_attempts = retry_attempts_;
    
    if (total_requests_ > 0) {
        stats.success_rate = static_cast<double>(successful_requests_) / total_requests_;
    } else {
        stats.success_rate = 0.0;
    }
    
    // Note: Average response time would need to be tracked separately
    stats.average_response_time_ms = 0.0;
    
    return stats;
}

// Utility Functions Implementation
namespace Utils {
    std::string loadApiKeyFromEnv(const std::string& env_var) {
        const char* key = std::getenv(env_var.c_str());
        return key ? std::string(key) : "";
    }
    
    std::string escapeJsonString(const std::string& input) {
        std::string output;
        output.reserve(input.length() + 10);
        
        for (char c : input) {
            switch (c) {
                case '"': output += "\\\""; break;
                case '\\': output += "\\\\"; break;
                case '\b': output += "\\b"; break;
                case '\f': output += "\\f"; break;
                case '\n': output += "\\n"; break;
                case '\r': output += "\\r"; break;
                case '\t': output += "\\t"; break;
                default: output += c; break;
            }
        }
        return output;
    }
    
    std::string extractJsonField(const std::string& json, const std::string& field) {
        Json::Value root;
        Json::CharReaderBuilder builder;
        std::string errors;
        
        std::istringstream stream(json);
        if (Json::parseFromStream(builder, stream, &root, &errors)) {
            if (root.isMember(field)) {
                return root[field].asString();
            }
        }
        return "";
    }
    
    bool isValidJson(const std::string& json) {
        Json::Value root;
        Json::CharReaderBuilder builder;
        std::string errors;
        
        std::istringstream stream(json);
        return Json::parseFromStream(builder, stream, &root, &errors);
    }
    
    std::string formatDuration(std::chrono::milliseconds duration) {
        auto ms = duration.count();
        if (ms < 1000) {
            return std::to_string(ms) + "ms";
        } else if (ms < 60000) {
            return std::to_string(ms / 1000) + "s";
        } else {
            return std::to_string(ms / 60000) + "m";
        }
    }
}

} // namespace MelvinOllama

/*
 * Ollama API Client for Melvin Curiosity Learning System
 * 
 * Features:
 * - Real Ollama API integration with HTTP requests
 * - Retry logic with exponential backoff
 * - Rate limiting and connection pooling
 * - Secure token loading from environment
 * - JSON parsing for responses
 * - Async request handling
 */

#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>
#include <atomic>
#include <functional>

// Forward declarations for HTTP client
struct curl_slist;
typedef void CURL;

namespace MelvinOllama {

// Configuration for Ollama client
struct OllamaConfig {
    std::string base_url = "http://localhost:11434";
    std::string model = "llama2";
    std::string api_key = "";  // Load from environment
    int max_retries = 3;
    int retry_delay_ms = 1000;
    int max_connections = 10;
    int request_timeout_seconds = 30;
    int rate_limit_requests_per_minute = 60;
    bool enable_async = true;
    int max_async_requests = 5;
};

// Response from Ollama API
struct OllamaResponse {
    bool success = false;
    std::string content;
    std::string model_used;
    int tokens_generated = 0;
    double response_time_ms = 0.0;
    int status_code = 0;
    std::string error_message;
    std::chrono::system_clock::time_point timestamp;
    
    OllamaResponse() : timestamp(std::chrono::system_clock::now()) {}
};

// Rate limiter for API calls
class RateLimiter {
private:
    std::chrono::steady_clock::time_point last_request_time;
    std::chrono::milliseconds min_interval;
    std::mutex mutex_;
    
public:
    RateLimiter(int requests_per_minute);
    void waitForNextRequest();
    bool canMakeRequest() const;
};

// HTTP client wrapper for Ollama API
class OllamaHttpClient {
private:
    CURL* curl_handle;
    std::string base_url;
    std::string api_key;
    int timeout_seconds;
    std::mutex curl_mutex_;
    
    // Connection pooling
    std::queue<CURL*> connection_pool;
    std::mutex pool_mutex_;
    std::atomic<int> active_connections{0};
    int max_connections;
    
    void initializeCurl();
    void cleanupCurl();
    CURL* getConnection();
    void returnConnection(CURL* handle);
    
public:
    OllamaHttpClient(const OllamaConfig& config);
    ~OllamaHttpClient();
    
    OllamaResponse makeRequest(const std::string& endpoint, 
                              const std::string& json_data);
    bool isHealthy() const;
    void setApiKey(const std::string& key);
};

// Async request manager
class AsyncRequestManager {
private:
    std::queue<std::function<void()>> request_queue;
    std::vector<std::thread> worker_threads;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> shutdown_{false};
    std::atomic<int> active_requests{0};
    int max_concurrent_requests;
    
    void workerThread();
    
public:
    AsyncRequestManager(int max_concurrent = 5);
    ~AsyncRequestManager();
    
    std::future<OllamaResponse> submitRequest(
        std::function<OllamaResponse()> request_func);
    void shutdown();
    int getActiveRequestCount() const;
};

// Main Ollama API client
class OllamaClient {
private:
    OllamaConfig config_;
    std::unique_ptr<OllamaHttpClient> http_client_;
    std::unique_ptr<RateLimiter> rate_limiter_;
    std::unique_ptr<AsyncRequestManager> async_manager_;
    
    // Statistics
    std::atomic<uint64_t> total_requests_{0};
    std::atomic<uint64_t> successful_requests_{0};
    std::atomic<uint64_t> failed_requests_{0};
    std::atomic<uint64_t> retry_attempts_{0};
    
    OllamaResponse makeRequestWithRetry(const std::string& prompt);
    std::string buildGenerateRequest(const std::string& prompt) const;
    OllamaResponse parseGenerateResponse(const std::string& json_response) const;
    void loadApiKeyFromEnvironment();
    
public:
    explicit OllamaClient(const OllamaConfig& config = OllamaConfig{});
    ~OllamaClient() = default;
    
    // Synchronous API
    OllamaResponse generate(const std::string& prompt);
    OllamaResponse askQuestion(const std::string& question);
    
    // Asynchronous API
    std::future<OllamaResponse> generateAsync(const std::string& prompt);
    std::future<OllamaResponse> askQuestionAsync(const std::string& question);
    
    // Health and status
    bool isHealthy() const;
    std::map<std::string, std::string> getStatus() const;
    void updateConfig(const OllamaConfig& new_config);
    
    // Statistics
    struct Statistics {
        uint64_t total_requests;
        uint64_t successful_requests;
        uint64_t failed_requests;
        uint64_t retry_attempts;
        double success_rate;
        double average_response_time_ms;
    };
    Statistics getStatistics() const;
};

// Utility functions
namespace Utils {
    std::string loadApiKeyFromEnv(const std::string& env_var = "OLLAMA_API_KEY");
    std::string escapeJsonString(const std::string& input);
    std::string extractJsonField(const std::string& json, const std::string& field);
    bool isValidJson(const std::string& json);
    std::string formatDuration(std::chrono::milliseconds duration);
}

} // namespace MelvinOllama

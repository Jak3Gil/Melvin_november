/*
 * Test program for Ollama API integration
 */

#include "ollama_client.h"
#include <iostream>
#include <string>

int main() {
    std::cout << "🤖 Testing Ollama API Integration" << std::endl;
    std::cout << "=================================" << std::endl;
    
    try {
        // Configure Ollama client
        MelvinOllama::OllamaConfig config;
        config.base_url = "http://localhost:11434";
        config.model = "llama3.2";
        config.max_retries = 3;
        config.request_timeout_seconds = 30;
        config.rate_limit_requests_per_minute = 60;
        
        std::cout << "📡 Connecting to Ollama at: " << config.base_url << std::endl;
        std::cout << "🤖 Using model: " << config.model << std::endl;
        
        // Create client
        MelvinOllama::OllamaClient client(config);
        
        // Test health
        if (!client.isHealthy()) {
            std::cout << "❌ Ollama client is not healthy" << std::endl;
            return 1;
        }
        
        std::cout << "✅ Ollama client is healthy" << std::endl;
        
        // Test simple question
        std::cout << "\n🧠 Testing question: What is a cat?" << std::endl;
        auto response = client.askQuestion("What is a cat?");
        
        if (response.success) {
            std::cout << "✅ Response received:" << std::endl;
            std::cout << "📝 " << response.content << std::endl;
            std::cout << "⏱️  Response time: " << response.response_time_ms << "ms" << std::endl;
            std::cout << "🔢 Tokens generated: " << response.tokens_generated << std::endl;
        } else {
            std::cout << "❌ Request failed: " << response.error_message << std::endl;
            return 1;
        }
        
        // Test async request
        std::cout << "\n🚀 Testing async request: What is a dog?" << std::endl;
        auto future_response = client.askQuestionAsync("What is a dog?");
        auto async_response = future_response.get();
        
        if (async_response.success) {
            std::cout << "✅ Async response received:" << std::endl;
            std::cout << "📝 " << async_response.content << std::endl;
            std::cout << "⏱️  Response time: " << async_response.response_time_ms << "ms" << std::endl;
        } else {
            std::cout << "❌ Async request failed: " << async_response.error_message << std::endl;
        }
        
        // Get statistics
        auto stats = client.getStatistics();
        std::cout << "\n📊 Statistics:" << std::endl;
        std::cout << "   Total requests: " << stats.total_requests << std::endl;
        std::cout << "   Successful: " << stats.successful_requests << std::endl;
        std::cout << "   Failed: " << stats.failed_requests << std::endl;
        std::cout << "   Success rate: " << (stats.success_rate * 100) << "%" << std::endl;
        
        std::cout << "\n🎉 Ollama integration test completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

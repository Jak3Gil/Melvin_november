#include "ollama_client.h"
#include <iostream>
#include <chrono>

int main() {
    std::cout << "🧪 MINIMAL OLLAMA CLIENT TEST" << std::endl;
    std::cout << "============================" << std::endl;
    
    std::cout << "1. Creating OllamaClient..." << std::endl;
    OllamaClient client;
    
    std::cout << "2. Checking if Ollama is available..." << std::endl;
    if (!client.isAvailable()) {
        std::cout << "❌ Ollama not available" << std::endl;
        return 1;
    }
    std::cout << "✅ Ollama is available" << std::endl;
    
    std::cout << "3. Testing simple response (this might hang)..." << std::endl;
    auto start = std::chrono::steady_clock::now();
    
    std::string response = client.generateResponse("Hello");
    
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "✅ Response received in " << duration.count() << "ms" << std::endl;
    std::cout << "Response: " << response << std::endl;
    
    return 0;
}

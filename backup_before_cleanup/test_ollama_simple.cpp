#include "ollama_client.h"
#include <iostream>

int main() {
    std::cout << "Testing OllamaClient directly..." << std::endl;
    
    OllamaClient client;
    
    if (!client.isAvailable()) {
        std::cout << "❌ Ollama not available" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Ollama available, testing response..." << std::endl;
    
    std::string response = client.generateResponse("Hello, test message");
    
    std::cout << "Response: " << response << std::endl;
    
    return 0;
}

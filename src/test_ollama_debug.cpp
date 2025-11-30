#include <iostream>
#include "ollama_client.h"

int main() {
    std::cout << "🔍 Testing Ollama Client Debug" << std::endl;
    
    MelvinOllama::OllamaClient client;
    
    std::cout << "🤖 Asking Ollama: What is a cat?" << std::endl;
    MelvinOllama::OllamaResponse response = client.askQuestion("What is a cat?");
    
    std::cout << "📊 Response Details:" << std::endl;
    std::cout << "Success: " << (response.success ? "true" : "false") << std::endl;
    std::cout << "Content: '" << response.content << "'" << std::endl;
    std::cout << "Model: " << response.model_used << std::endl;
    std::cout << "Tokens: " << response.tokens_generated << std::endl;
    std::cout << "Error: " << response.error_message << std::endl;
    
    return 0;
}

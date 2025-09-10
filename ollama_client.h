#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <curl/curl.h>

// ============================================================================
// OLLAMA CLIENT FOR REAL AI RESPONSES
// ============================================================================

class OllamaClient {
private:
    std::string base_url;
    std::string model;
    CURL* curl;
    
    // Callback function for receiving data
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* data);
    
public:
    OllamaClient(const std::string& url = "http://localhost:11434", const std::string& model_name = "llama3.2");
    ~OllamaClient();
    
    // Generate a response from Ollama
    std::string generateResponse(const std::string& prompt, const std::string& context = "");
    
    // Generate a response with specific parameters
    std::string generateResponseWithParams(const std::string& prompt, 
                                         const std::string& context = "",
                                         float temperature = 0.7f,
                                         int max_tokens = 1000);
    
    // Check if Ollama is running
    bool isAvailable();
    
    // Get available models
    std::vector<std::string> getAvailableModels();
    
    // Set model
    void setModel(const std::string& model_name);
    
    // Generate autonomous thinking prompt
    std::string generateAutonomousThinkingPrompt(const std::string& input, 
                                                const std::string& driver_context,
                                                const std::string& previous_thoughts = "");
    
    // Generate curiosity-driven question
    std::string generateCuriosityQuestion(const std::string& current_knowledge);
    
    // Generate self-improvement reflection
    std::string generateSelfImprovementReflection(const std::string& recent_cycles);
    
    // Generate meta-cognitive analysis
    std::string generateMetaCognitiveAnalysis(const std::string& thought_patterns);
};

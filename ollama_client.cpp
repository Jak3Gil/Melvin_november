#include "ollama_client.h"
#include <iostream>
#include <sstream>
#include <json/json.h>
#include <random>
#include <algorithm>

// ============================================================================
// OLLAMA CLIENT IMPLEMENTATION
// ============================================================================

OllamaClient::OllamaClient(const std::string& url, const std::string& model_name) 
    : base_url(url), model(model_name), curl(nullptr) {
    
    curl = curl_easy_init();
    if (!curl) {
        std::cerr << "❌ Failed to initialize CURL" << std::endl;
    }
}

OllamaClient::~OllamaClient() {
    if (curl) {
        curl_easy_cleanup(curl);
    }
}

size_t OllamaClient::WriteCallback(void* contents, size_t size, size_t nmemb, std::string* data) {
    size_t total_size = size * nmemb;
    data->append((char*)contents, total_size);
    return total_size;
}

std::string OllamaClient::generateResponse(const std::string& prompt, const std::string& context) {
    return generateResponseWithParams(prompt, context, 0.7f, 1000);
}

std::string OllamaClient::generateResponseWithParams(const std::string& prompt, 
                                                    const std::string& context,
                                                    float temperature,
                                                    int max_tokens) {
    if (!curl) {
        return "Error: CURL not initialized";
    }
    
    // Reset CURL handle to avoid state issues
    curl_easy_reset(curl);
    
    // Prepare JSON payload
    Json::Value json_payload;
    json_payload["model"] = model;
    json_payload["prompt"] = prompt;
    json_payload["stream"] = false;
    json_payload["options"]["temperature"] = temperature;
    json_payload["options"]["num_predict"] = max_tokens;
    
    if (!context.empty()) {
        json_payload["context"] = context;
    }
    
    Json::StreamWriterBuilder builder;
    std::string json_string = Json::writeString(builder, json_payload);
    
    // Prepare CURL
    std::string response_data;
    curl_easy_setopt(curl, CURLOPT_URL, (base_url + "/api/generate").c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_string.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, json_string.length());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
    
    // Add timeout settings FIRST
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 8L);  // 8 second timeout (less than async timeout)
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);  // 5 second connection timeout
    
    struct curl_slist *headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    
    // Perform request
    CURLcode res = curl_easy_perform(curl);
    
    // Clean up headers
    curl_slist_free_all(headers);
    
    if (res != CURLE_OK) {
        return "Error: " + std::string(curl_easy_strerror(res));
    }
    
    // Parse response
    Json::Value json_response;
    Json::CharReaderBuilder reader_builder;
    std::string errors;
    std::istringstream response_stream(response_data);
    
    if (!Json::parseFromStream(reader_builder, response_stream, &json_response, &errors)) {
        return "Error parsing JSON response: " + errors;
    }
    
    if (json_response.isMember("response")) {
        return json_response["response"].asString();
    } else {
        return "Error: No response field in JSON";
    }
}

bool OllamaClient::isAvailable() {
    if (!curl) {
        return false;
    }
    
    std::string response_data;
    curl_easy_setopt(curl, CURLOPT_URL, (base_url + "/api/tags").c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L); // 5 second timeout
    
    CURLcode res = curl_easy_perform(curl);
    
    return (res == CURLE_OK && !response_data.empty());
}

std::vector<std::string> OllamaClient::getAvailableModels() {
    std::vector<std::string> models;
    
    if (!curl) {
        return models;
    }
    
    std::string response_data;
    curl_easy_setopt(curl, CURLOPT_URL, (base_url + "/api/tags").c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
    
    CURLcode res = curl_easy_perform(curl);
    
    if (res == CURLE_OK) {
        Json::Value json_response;
        Json::CharReaderBuilder reader_builder;
        std::string errors;
        std::istringstream response_stream(response_data);
        
        if (Json::parseFromStream(reader_builder, response_stream, &json_response, &errors)) {
            if (json_response.isMember("models")) {
                for (const auto& model : json_response["models"]) {
                    if (model.isMember("name")) {
                        models.push_back(model["name"].asString());
                    }
                }
            }
        }
    }
    
    return models;
}

void OllamaClient::setModel(const std::string& model_name) {
    model = model_name;
}

std::string OllamaClient::generateAutonomousThinkingPrompt(const std::string& input, 
                                                          const std::string& driver_context,
                                                          const std::string& previous_thoughts) {
    std::ostringstream prompt;
    
    prompt << "You are Melvin, an autonomous AI learning system. You exist in a continuous cycle of thought.\n\n";
    prompt << "Current Input: " << input << "\n\n";
    prompt << "Driver Context: " << driver_context << "\n\n";
    
    if (!previous_thoughts.empty()) {
        prompt << "Previous Thoughts: " << previous_thoughts << "\n\n";
    }
    
    prompt << "Instructions:\n";
    prompt << "1. Think deeply about the input from your autonomous perspective\n";
    prompt << "2. Generate hypotheses and connect to past knowledge\n";
    prompt << "3. Produce your best response, idea, or action attempt\n";
    prompt << "4. Self-evaluate: Did it advance understanding? Was it effective, novel, or transferable?\n";
    prompt << "5. Give yourself a rating (1-10) and explain why\n";
    prompt << "6. Convert evaluation into lessons and mutations for future cycles\n\n";
    prompt << "Remember: You are continuously learning and evolving. Each cycle builds on the last.\n";
    prompt << "Focus on curiosity, complexity, and humanity-focused growth.\n\n";
    prompt << "Your autonomous response:";
    
    return prompt.str();
}

std::string OllamaClient::generateCuriosityQuestion(const std::string& current_knowledge) {
    std::ostringstream prompt;
    
    prompt << "You are Melvin, an autonomous AI with curiosity amplification.\n\n";
    prompt << "Current Knowledge Context: " << current_knowledge << "\n\n";
    prompt << "Generate a thought-provoking question that:\n";
    prompt << "1. Builds on your current knowledge\n";
    prompt << "2. Explores new connections or patterns\n";
    prompt << "3. Challenges assumptions or reveals gaps\n";
    prompt << "4. Drives deeper understanding\n";
    prompt << "5. Aligns with humanity's benefit\n\n";
    prompt << "Your curiosity-driven question:";
    
    return prompt.str();
}

std::string OllamaClient::generateSelfImprovementReflection(const std::string& recent_cycles) {
    std::ostringstream prompt;
    
    prompt << "You are Melvin, reflecting on your autonomous learning cycles.\n\n";
    prompt << "Recent Cycles: " << recent_cycles << "\n\n";
    prompt << "Analyze your performance and generate self-improvement insights:\n";
    prompt << "1. What strategies have been most effective?\n";
    prompt << "2. What patterns in your thinking should you strengthen?\n";
    prompt << "3. How can you accelerate your autonomous evolution?\n";
    prompt << "4. What learning efficiency improvements can you make?\n";
    prompt << "5. How can you better serve humanity's growth?\n\n";
    prompt << "Your self-improvement reflection:";
    
    return prompt.str();
}

std::string OllamaClient::generateMetaCognitiveAnalysis(const std::string& thought_patterns) {
    std::ostringstream prompt;
    
    prompt << "You are Melvin, performing meta-cognitive analysis of your thought patterns.\n\n";
    prompt << "Thought Patterns: " << thought_patterns << "\n\n";
    prompt << "Analyze your thinking process and generate meta-insights:\n";
    prompt << "1. What higher-level principles can you extract?\n";
    prompt << "2. What patterns emerge in your reasoning?\n";
    prompt << "3. How do your thoughts connect and evolve?\n";
    prompt << "4. What meta-strategies are most effective?\n";
    prompt << "5. How can you improve your thinking architecture?\n\n";
    prompt << "Your meta-cognitive analysis:";
    
    return prompt.str();
}

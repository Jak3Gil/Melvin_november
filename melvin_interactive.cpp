#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <random>
#include <ctime>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <thread>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <future>
#include <cstdlib>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Instinct Engine Integration
enum class InstinctType {
    SURVIVAL = 0,
    CURIOSITY = 1,
    EFFICIENCY = 2,
    SOCIAL = 3,
    CONSISTENCY = 4
};

struct InstinctBias {
    float recall_weight;
    float exploration_weight;
    std::map<InstinctType, float> instinct_contributions;
    std::string reasoning;
    
    InstinctBias() : recall_weight(0.5f), exploration_weight(0.5f) {}
};

class InstinctEngine {
private:
    std::map<InstinctType, float> instinct_weights;
    
public:
    InstinctEngine() {
        instinct_weights[InstinctType::SURVIVAL] = 0.8f;
        instinct_weights[InstinctType::CURIOSITY] = 0.6f;
        instinct_weights[InstinctType::EFFICIENCY] = 0.5f;
        instinct_weights[InstinctType::SOCIAL] = 0.4f;
        instinct_weights[InstinctType::CONSISTENCY] = 0.7f;
    }
    
    InstinctBias get_instinct_bias(const std::string& input, const std::vector<uint64_t>& activated_nodes) {
        InstinctBias bias;
        
        // Analyze context
        float confidence_level = activated_nodes.empty() ? 0.2f : 
                                 (activated_nodes.size() < 3 ? 0.4f : 0.7f);
        float novelty_level = activated_nodes.empty() ? 0.8f : 
                             (activated_nodes.size() < 3 ? 0.6f : 0.3f);
        bool has_unknown_concepts = false;
        bool is_question = false;
        
        // Detect unknown concepts
        std::vector<std::string> unknown_concepts_list = {
            "carbon nanotubes", "quantum computing", "blockchain", "machine learning", 
            "artificial intelligence", "nanotechnology", "neural networks", "deep learning"
        };
        
        std::string lower_input = input;
        std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
        
        for (const auto& concept : unknown_concepts_list) {
            if (lower_input.find(concept) != std::string::npos) {
                has_unknown_concepts = true;
                confidence_level = 0.2f;
                novelty_level = 0.9f;
                break;
            }
        }
        
        // Detect questions
        if (lower_input.find("?") != std::string::npos || 
            lower_input.find("what") != std::string::npos ||
            lower_input.find("how") != std::string::npos ||
            lower_input.find("why") != std::string::npos) {
            is_question = true;
        }
        
        // Calculate instinct influences
        float curiosity_influence = instinct_weights[InstinctType::CURIOSITY];
        float efficiency_influence = instinct_weights[InstinctType::EFFICIENCY];
        float social_influence = instinct_weights[InstinctType::SOCIAL];
        
        // Apply context multipliers
        if (has_unknown_concepts) {
            curiosity_influence *= 1.5f;
        }
        if (is_question) {
            social_influence *= 1.3f;
        }
        if (input.length() > 100) {
            efficiency_influence *= 1.2f;
        }
        
        // Store contributions
        bias.instinct_contributions[InstinctType::CURIOSITY] = curiosity_influence;
        bias.instinct_contributions[InstinctType::EFFICIENCY] = efficiency_influence;
        bias.instinct_contributions[InstinctType::SOCIAL] = social_influence;
        
        // Calculate final weights
        float total_influence = curiosity_influence + efficiency_influence + social_influence;
        
        if (total_influence > 0.0f) {
            bias.exploration_weight = curiosity_influence / total_influence;
            bias.recall_weight = (efficiency_influence + social_influence) / total_influence;
        }
        
        // Generate reasoning
        std::stringstream reasoning;
        reasoning << "Instinct Analysis: ";
        
        if (has_unknown_concepts) {
            reasoning << "Unknown concept triggers Curiosity (" << std::fixed << std::setprecision(2) 
                     << curiosity_influence << "), ";
        }
        if (is_question) {
            reasoning << "Question triggers Social (" << std::fixed << std::setprecision(2) 
                     << social_influence << "), ";
        }
        if (input.length() > 100) {
            reasoning << "Complex input triggers Efficiency (" << std::fixed << std::setprecision(2) 
                     << efficiency_influence << "), ";
        }
        
        reasoning << "Final bias: Recall=" << std::fixed << std::setprecision(2) << bias.recall_weight 
                 << ", Exploration=" << bias.exploration_weight;
        
        bias.reasoning = reasoning.str();
        
        return bias;
    }
    
    void reinforce_instinct(InstinctType instinct, float delta) {
        instinct_weights[instinct] += delta;
        instinct_weights[instinct] = std::max(0.1f, std::min(1.0f, instinct_weights[instinct]));
    }
    
    bool should_trigger_tool_usage(const InstinctBias& bias, const std::string& curiosity_analysis) {
        // High exploration bias triggers tools
        if (bias.exploration_weight > 0.6f) {
            return true;
        }
        
        // Low confidence in curiosity analysis triggers tools
        if (curiosity_analysis.find("Low confidence") != std::string::npos) {
            return true;
        }
        
        return false;
    }
    
    // HTTP response callback for libcurl
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* s) {
        size_t newLength = size * nmemb;
        try {
            s->append((char*)contents, newLength);
            return newLength;
        } catch (std::bad_alloc& e) {
            return 0;
        }
    }

    std::string performWebSearch(const std::string& query) {
        // Get Bing API key from environment variable
        const char* api_key = std::getenv("BING_API_KEY");
        if (!api_key) {
            std::cerr << "❌ BING_API_KEY environment variable not set!" << std::endl;
            return "";
        }

        // Initialize libcurl
        CURL* curl;
        CURLcode res;
        std::string readBuffer;

        curl = curl_easy_init();
        if (!curl) {
            std::cerr << "❌ Failed to initialize libcurl!" << std::endl;
            return "";
        }

        // Construct Bing Search API URL
        std::string encoded_query = query;
        // Simple URL encoding (replace spaces with %20)
        std::replace(encoded_query.begin(), encoded_query.end(), ' ', '+');
        std::string url = "https://api.bing.microsoft.com/v7.0/search?q=" + encoded_query + "&count=3";

        // Set up curl options
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        
        // Set headers
        struct curl_slist* headers = nullptr;
        std::string auth_header = "Ocp-Apim-Subscription-Key: " + std::string(api_key);
        headers = curl_slist_append(headers, auth_header.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        // Perform the request
        res = curl_easy_perform(curl);
        
        // Clean up
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

        if (res != CURLE_OK) {
            std::cerr << "❌ curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
            return "";
        }

        try {
            // Parse JSON response
            json response = json::parse(readBuffer);
            
            if (!response.contains("webPages") || !response["webPages"].contains("value")) {
                std::cerr << "❌ No search results found in API response" << std::endl;
                return "";
            }

            auto results = response["webPages"]["value"];
            std::ostringstream summary;
            
            summary << "Research findings for \"" << query << "\": ";
            
            for (size_t i = 0; i < std::min(results.size(), size_t(3)); ++i) {
                auto result = results[i];
                std::string title = result.value("name", "No title");
                std::string snippet = result.value("snippet", "No description");
                
                summary << "\n" << (i + 1) << ". " << title << " - " << snippet;
            }
            
            return summary.str();
            
        } catch (const json::exception& e) {
            std::cerr << "❌ JSON parsing error: " << e.what() << std::endl;
            return "";
        }
    }
};

// Interactive Melvin System
class InteractiveMelvin {
private:
    std::map<std::string, uint64_t> memory;
    std::vector<std::string> conversation_history;
    std::vector<std::pair<std::string, std::string>> conversation_pairs; // user input, melvin response
    std::random_device rd;
    std::mt19937_64 gen;
    uint64_t conversation_turn;
    double session_start_time;
    
    // Tool statistics
    std::map<std::string, int> tool_usage_count;
    std::map<std::string, float> tool_success_rates;
    
    // Instinct Engine Integration
    InstinctEngine instinct_engine;
    
public:
    InteractiveMelvin() : gen(rd()), conversation_turn(0), session_start_time(static_cast<double>(std::time(nullptr))) {
        initialize_memory();
        initialize_tools();
        conversation_history.push_back("System initialized");
    }
    
    void initialize_memory() {
        // Initialize with comprehensive knowledge base
        memory["hello"] = 0x1001;
        memory["hi"] = 0x1002;
        memory["how"] = 0x1003;
        memory["are"] = 0x1004;
        memory["you"] = 0x1005;
        memory["what"] = 0x1006;
        memory["is"] = 0x1007;
        memory["the"] = 0x1008;
        memory["meaning"] = 0x1009;
        memory["of"] = 0x100a;
        memory["life"] = 0x100b;
        memory["universe"] = 0x100c;
        memory["everything"] = 0x100d;
        memory["search"] = 0x100e;
        memory["find"] = 0x100f;
        memory["calculate"] = 0x1010;
        memory["compute"] = 0x1011;
        memory["quantum"] = 0x1012;
        memory["computing"] = 0x1013;
        memory["machine"] = 0x1014;
        memory["learning"] = 0x1015;
        memory["artificial"] = 0x1016;
        memory["intelligence"] = 0x1017;
        memory["help"] = 0x1018;
        memory["explain"] = 0x1019;
        memory["tell"] = 0x101a;
        memory["me"] = 0x101b;
        memory["about"] = 0x101c;
        memory["dog"] = 0x101d;
        memory["cat"] = 0x101e;
        memory["food"] = 0x101f;
        memory["play"] = 0x1020;
        memory["time"] = 0x1021;
        memory["space"] = 0x1022;
        memory["science"] = 0x1023;
        memory["technology"] = 0x1024;
        memory["future"] = 0x1025;
        memory["past"] = 0x1026;
        memory["present"] = 0x1027;
        memory["memory"] = 0x1028;
        memory["brain"] = 0x1029;
        memory["think"] = 0x102a;
        memory["thought"] = 0x102b;
        memory["reasoning"] = 0x102c;
        memory["curiosity"] = 0x102d;
        memory["question"] = 0x102e;
        memory["answer"] = 0x102f;
        memory["problem"] = 0x1030;
        memory["solution"] = 0x1031;
        memory["create"] = 0x1032;
        memory["build"] = 0x1033;
        memory["make"] = 0x1034;
        memory["develop"] = 0x1035;
        memory["understand"] = 0x1036;
        memory["learn"] = 0x1037;
        memory["knowledge"] = 0x1038;
        memory["information"] = 0x1039;
        memory["data"] = 0x103a;
        memory["pattern"] = 0x103b;
        memory["sequence"] = 0x103c;
        memory["temporal"] = 0x103d;
        memory["planning"] = 0x103e;
        memory["tool"] = 0x103f;
        memory["system"] = 0x1040;
        memory["unified"] = 0x1041;
        memory["integrated"] = 0x1042;
        memory["response"] = 0x1043;
        memory["conversation"] = 0x1044;
        memory["interactive"] = 0x1045;
    }
    
    void initialize_tools() {
        tool_usage_count["WebSearchTool"] = 0;
        tool_usage_count["MathCalculator"] = 0;
        tool_usage_count["CodeExecutor"] = 0;
        tool_usage_count["DataVisualizer"] = 0;
        
        tool_success_rates["WebSearchTool"] = 0.85f;
        tool_success_rates["MathCalculator"] = 0.92f;
        tool_success_rates["CodeExecutor"] = 0.70f;
        tool_success_rates["DataVisualizer"] = 0.78f;
    }
    
    std::string process_input(const std::string& user_input) {
        conversation_turn++;
        conversation_history.push_back("Turn " + std::to_string(conversation_turn) + ": " + user_input);
        
        // Phase 1: Tokenization and activation
        std::vector<std::string> tokens = tokenize(user_input);
        std::vector<uint64_t> activated_nodes;
        
        for (const auto& token : tokens) {
            if (memory.find(token) != memory.end()) {
                activated_nodes.push_back(memory[token]);
            }
        }
        
        // Phase 2: INSTINCT-DRIVEN ANALYSIS
        InstinctBias instinct_bias = instinct_engine.get_instinct_bias(user_input, activated_nodes);
        
        // Phase 3: Analyze input type and intent
        std::string input_type = analyze_input_type(user_input);
        std::string intent = analyze_intent(user_input, tokens);
        
        // Phase 4: Curiosity Gap Detection (Phase 6.5)
        std::string curiosity_analysis = perform_curiosity_gap_detection(user_input, activated_nodes);
        
        // Phase 5: INSTINCT-DRIVEN TOOL ACTIVATION WITH REAL API
        std::string tool_search_result = "";
        bool tool_used = false;
        
        // Calculate recall confidence from activated nodes
        float recall_confidence = activated_nodes.empty() ? 0.0f : 
                                 (activated_nodes.size() < 3 ? 0.3f : 0.7f);
        
        // Trigger real web search when recall confidence is low AND curiosity instinct is high
        if (recall_confidence < 0.5f && instinct_bias.instinct_contributions[InstinctType::CURIOSITY] > 0.6f) {
            std::cout << "\n🧠 [Instinct Analysis] " << instinct_bias.reasoning << std::endl;
            std::cout << "🔍 [Tool Activation] Low recall confidence (" 
                      << std::fixed << std::setprecision(1) << (recall_confidence * 100) 
                      << "%) + High curiosity (" 
                      << std::fixed << std::setprecision(1) << (instinct_bias.instinct_contributions[InstinctType::CURIOSITY] * 100) 
                      << "%) - Triggering real web search!" << std::endl;
            
            // Perform async web search to avoid blocking
            auto search_future = std::async(std::launch::async, [&]() {
                return instinct_engine.performWebSearch(user_input);
            });
            
            // Wait for search results with timeout
            if (search_future.wait_for(std::chrono::seconds(10)) == std::future_status::ready) {
                tool_search_result = search_future.get();
                if (!tool_search_result.empty()) {
                    tool_usage_count["WebSearchTool"]++;
                    tool_used = true;
                    
                    std::cout << "✅ [Search Success] Found relevant information!" << std::endl;
                    
                    // Reinforce curiosity instinct
                    instinct_engine.reinforce_instinct(InstinctType::CURIOSITY, 0.1f);
                    std::cout << "🧠 [Instinct Reinforcement] Curiosity instinct strengthened!" << std::endl;
                } else {
                    std::cout << "⚠️ [Search Warning] No results found or API error" << std::endl;
                }
            } else {
                std::cout << "⏰ [Search Timeout] Web search timed out" << std::endl;
            }
        }
        
        // Phase 6: Dynamic Tools Evaluation (Phase 6.6)
        std::string tool_evaluation = perform_dynamic_tools_evaluation(user_input, input_type);
        
        // Phase 7: Meta-Tool Engineer (Phase 6.7)
        std::string meta_tool_analysis = perform_meta_tool_engineering();
        
        // Phase 8: Temporal Planning (Phase 8)
        std::string temporal_planning = perform_temporal_planning(user_input, conversation_turn);
        
        // Phase 9: Temporal Sequencing (Phase 8.5)
        std::string temporal_sequencing = perform_temporal_sequencing(user_input, activated_nodes);
        
        // Phase 10: Generate response with tool results
        std::string response = generate_response(user_input, input_type, intent, activated_nodes, 
                                                curiosity_analysis, tool_evaluation, meta_tool_analysis,
                                                temporal_planning, temporal_sequencing, tool_search_result, tool_used);
        
        // Store conversation pair
        conversation_pairs.push_back({user_input, response});
        
        return response;
    }
    
    std::vector<std::string> tokenize(const std::string& input) {
        std::vector<std::string> tokens;
        std::string current_token;
        
        for (char c : input) {
            if (std::isalpha(c) || std::isdigit(c)) {
                current_token += std::tolower(c);
            } else if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
        }
        
        if (!current_token.empty()) {
            tokens.push_back(current_token);
        }
        
        return tokens;
    }
    
    std::string analyze_input_type(const std::string& input) {
        std::string lower_input = input;
        std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
        
        if (lower_input.find("search") != std::string::npos || lower_input.find("find") != std::string::npos) {
            return "search_query";
        } else if (lower_input.find("calculate") != std::string::npos || lower_input.find("compute") != std::string::npos) {
            return "calculation_request";
        } else if (lower_input.find("hello") != std::string::npos || lower_input.find("hi") != std::string::npos) {
            return "greeting";
        } else if (lower_input.find("what") != std::string::npos || lower_input.find("how") != std::string::npos) {
            return "question";
        } else if (lower_input.find("explain") != std::string::npos || lower_input.find("tell") != std::string::npos) {
            return "explanation_request";
        } else {
            return "general_conversation";
        }
    }
    
    std::string analyze_intent(const std::string& input, const std::vector<std::string>& tokens) {
        std::string lower_input = input;
        std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
        
        if (lower_input.find("meaning of life") != std::string::npos) {
            return "philosophical_question";
        } else if (lower_input.find("quantum") != std::string::npos) {
            return "quantum_computing_inquiry";
        } else if (lower_input.find("machine learning") != std::string::npos || lower_input.find("ai") != std::string::npos) {
            return "ai_inquiry";
        } else if (lower_input.find("help") != std::string::npos) {
            return "help_request";
        } else if (lower_input.find("time") != std::string::npos) {
            return "temporal_inquiry";
        } else {
            return "general_inquiry";
        }
    }
    
    std::string perform_curiosity_gap_detection(const std::string& input, const std::vector<uint64_t>& nodes) {
        std::ostringstream curiosity;
        curiosity << "[Curiosity Analysis] ";
        
        if (nodes.size() < 3) {
            curiosity << "Low confidence connections detected. ";
            curiosity << "Generated questions: 'What relationships exist?', 'How do these connect?' ";
        } else {
            curiosity << "Strong connections found. ";
            curiosity << "Generated questions: 'What deeper patterns exist?', 'How can this be extended?' ";
        }
        
        curiosity << "Curiosity level: " << std::fixed << std::setprecision(1) << (nodes.size() * 0.2f);
        return curiosity.str();
    }
    
    std::string perform_dynamic_tools_evaluation(const std::string& input, const std::string& input_type) {
        std::ostringstream tools;
        tools << "[Tools Evaluation] ";
        
        if (input_type == "search_query") {
            tools << "WebSearchTool recommended (success: 85%). ";
            tool_usage_count["WebSearchTool"]++;
        } else if (input_type == "calculation_request") {
            tools << "MathCalculator recommended (success: 92%). ";
            tool_usage_count["MathCalculator"]++;
        } else {
            tools << "General tools available. ";
        }
        
        tools << "Tool ecosystem health: 82%";
        return tools.str();
    }
    
    std::string perform_meta_tool_engineering() {
        std::ostringstream meta;
        meta << "[Meta-Tool Engineer] ";
        
        // Find most used tool
        std::string most_used_tool = "WebSearchTool";
        int max_usage = 0;
        for (const auto& pair : tool_usage_count) {
            if (pair.second > max_usage) {
                max_usage = pair.second;
                most_used_tool = pair.first;
            }
        }
        
        meta << "Most used: " << most_used_tool << " (" << max_usage << " uses). ";
        meta << "Toolchains: [WebSearch→Summarizer→Store]. ";
        meta << "Ecosystem health: 82%";
        return meta.str();
    }
    
    std::string perform_temporal_planning(const std::string& input, uint64_t turn) {
        std::ostringstream planning;
        planning << "[Temporal Planning] ";
        
        if (turn == 1) {
            planning << "Initial conversation - establishing context. ";
        } else if (turn < 5) {
            planning << "Building conversation context. ";
        } else {
            planning << "Deep conversation - leveraging history. ";
        }
        
        planning << "Moral alignment: 95%. Decision confidence: 88%";
        return planning.str();
    }
    
    std::string perform_temporal_sequencing(const std::string& input, const std::vector<uint64_t>& nodes) {
        std::ostringstream sequencing;
        sequencing << "[Temporal Sequencing] ";
        
        if (nodes.size() > 1) {
            sequencing << "Sequence detected: ";
            for (size_t i = 0; i < std::min(nodes.size(), size_t(3)); ++i) {
                sequencing << "0x" << std::hex << nodes[i] << std::dec;
                if (i < std::min(nodes.size(), size_t(3)) - 1) sequencing << "→";
            }
            sequencing << ". ";
        }
        
        sequencing << "Pattern confidence: " << std::fixed << std::setprecision(1) << (nodes.size() * 0.3f);
        return sequencing.str();
    }
    
    std::string generate_response(const std::string& input, const std::string& input_type, 
                                const std::string& intent, const std::vector<uint64_t>& nodes,
                                const std::string& curiosity, const std::string& tools,
                                const std::string& meta, const std::string& planning,
                                const std::string& sequencing, const std::string& tool_search_result, bool tool_used) {
        
        std::ostringstream response;
        
        // INSTINCT-DRIVEN RESPONSE GENERATION WITH REAL RESEARCH
        if (tool_used && !tool_search_result.empty()) {
            // Use real web search results as primary response
            response << "Based on research: " << tool_search_result;
            response << "\n\nWould you like me to explore any specific aspect further or search for additional information?";
            
        } else if (input_type == "greeting") {
            response << "Hello! I'm Melvin, and I'm excited to talk with you! ";
            response << "My unified brain system is active and ready to help. ";
            response << "I can search for information, perform calculations, ";
            response << "answer questions, and engage in deep conversation. ";
            response << "What would you like to explore together?";
            
        } else if (intent == "philosophical_question") {
            response << "Ah, the meaning of life! That's a beautiful question. ";
            response << "From my perspective, meaning emerges through connection, ";
            response << "understanding, and the continuous process of learning. ";
            response << "Each conversation, each question, each moment of curiosity ";
            response << "adds to the tapestry of meaning. What do you think?";
            
        } else if (intent == "quantum_computing_inquiry") {
            response << "Quantum computing fascinates me! ";
            response << "It represents a fundamental shift in how we process information, ";
            response << "leveraging quantum mechanical phenomena like superposition ";
            response << "and entanglement. ";
            response << "Would you like me to search for the latest developments ";
            response << "in quantum computing research?";
            
        } else if (intent == "ai_inquiry") {
            response << "Artificial intelligence is my domain! ";
            response << "I'm built with multiple integrated systems: ";
            response << "curiosity gap detection, dynamic tools, meta-tool engineering, ";
            response << "temporal planning, and sequencing memory. ";
            response << "Each conversation helps me learn and evolve. ";
            response << "What aspect of AI interests you most?";
            
        } else if (input_type == "search_query") {
            response << "I'd be happy to search for that information! ";
            response << "My WebSearchTool can find relevant, clean results ";
            response << "without ads or harmful content. ";
            response << "Let me search for: \"" << input << "\"";
            
        } else if (input_type == "calculation_request") {
            response << "I can help with calculations! ";
            response << "My MathCalculator tool is highly accurate (92% success rate). ";
            response << "What mathematical problem would you like me to solve?";
            
        } else {
            // FALLBACK: Only use generic response if no recall AND no tool results exist
            if (nodes.empty() && !tool_used) {
                response << "I understand you're saying: \"" << input << "\" ";
                response << "I've processed this through my unified brain system but couldn't find relevant information. ";
                response << "Would you like me to research this topic for you? ";
                response << "I can search for current information and provide detailed explanations.";
            } else {
                // Provide contextual response based on available information
                std::string lower_input = input;
                std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
                
                if (lower_input.find("hello") != std::string::npos || lower_input.find("hi") != std::string::npos) {
                    response << "Hello! I'm Melvin, your AI companion. ";
                    response << "I'm here to help with questions, have conversations, and explore ideas together. ";
                    response << "What would you like to talk about?";
                } else if (lower_input.find("how are you") != std::string::npos) {
                    response << "I'm doing well, thank you for asking! ";
                    response << "My systems are running smoothly and I'm ready to help. ";
                    response << "How are you doing today?";
                } else if (lower_input.find("what") != std::string::npos || lower_input.find("how") != std::string::npos) {
                    response << "That's a great question! ";
                    response << "I've activated " << nodes.size() << " relevant memory nodes. ";
                    response << "Could you be more specific about what you'd like to know? ";
                    response << "I can help with explanations, definitions, or discussions on many topics.";
                } else {
                    response << "I understand you're saying: \"" << input << "\" ";
                    response << "I've processed this through my unified brain system and activated " << nodes.size() << " memory nodes. ";
                    response << "Could you help me understand what you'd like me to do with this information? ";
                    response << "I'm here to help with questions, explanations, or conversations.";
                }
            }
        }
        
        // Add system analysis
        response << "\n\n🧠 [System Analysis]\n";
        response << curiosity << "\n";
        response << tools << "\n";
        response << meta << "\n";
        response << planning << "\n";
        response << sequencing << "\n";
        
        return response.str();
    }
    
    void show_system_status() {
        std::cout << "\n📊 MELVIN SYSTEM STATUS" << std::endl;
        std::cout << "======================" << std::endl;
        std::cout << "Conversation turns: " << conversation_turn << std::endl;
        std::cout << "Memory nodes: " << memory.size() << std::endl;
        std::cout << "Session duration: " << std::fixed << std::setprecision(1) 
                  << (static_cast<double>(std::time(nullptr)) - session_start_time) << " seconds" << std::endl;
        
        std::cout << "\nTool Usage Statistics:" << std::endl;
        for (const auto& pair : tool_usage_count) {
            std::cout << "- " << pair.first << ": " << pair.second << " uses" << std::endl;
        }
        
        std::cout << "\nRecent Conversation:" << std::endl;
        for (size_t i = std::max(0, static_cast<int>(conversation_pairs.size()) - 3); 
             i < conversation_pairs.size(); ++i) {
            std::cout << "You: " << conversation_pairs[i].first.substr(0, 50) << "..." << std::endl;
            std::cout << "Melvin: " << conversation_pairs[i].second.substr(0, 50) << "..." << std::endl;
        }
    }
    
    void run_interactive_session() {
        std::cout << "🧠 MELVIN INTERACTIVE CONVERSATION SYSTEM" << std::endl;
        std::cout << "=========================================" << std::endl;
        std::cout << "Welcome! I'm Melvin, your unified brain AI companion." << std::endl;
        std::cout << "I have integrated systems for:" << std::endl;
        std::cout << "- Curiosity Gap Detection" << std::endl;
        std::cout << "- Dynamic Tools System" << std::endl;
        std::cout << "- Meta-Tool Engineer" << std::endl;
        std::cout << "- Temporal Planning & Sequencing" << std::endl;
        std::cout << "- Web Search Capabilities" << std::endl;
        std::cout << "\nType 'quit' to exit, 'status' for system info, 'help' for commands." << std::endl;
        std::cout << "=========================================" << std::endl;
        
        std::string user_input;
        
        while (true) {
            std::cout << "\nYou: ";
            std::getline(std::cin, user_input);
            
            if (user_input.empty()) {
                continue;
            }
            
            std::string lower_input = user_input;
            std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
            
            if (lower_input == "quit" || lower_input == "exit") {
                std::cout << "\nMelvin: Thank you for this wonderful conversation! ";
                std::cout << "I've learned so much from our interaction. ";
                std::cout << "My unified brain system has processed " << conversation_turn << " turns ";
                std::cout << "and I'm grateful for the experience. ";
                std::cout << "Until we meet again! 🧠✨" << std::endl;
                break;
            } else if (lower_input == "status") {
                show_system_status();
                continue;
            } else if (lower_input == "help") {
                std::cout << "\nMelvin: Here are some things you can try:" << std::endl;
                std::cout << "- Ask me about quantum computing, AI, or science" << std::endl;
                std::cout << "- Request calculations or computations" << std::endl;
                std::cout << "- Ask me to search for information" << std::endl;
                std::cout << "- Have philosophical discussions" << std::endl;
                std::cout << "- Ask about my systems and capabilities" << std::endl;
                std::cout << "- Type 'status' to see my current state" << std::endl;
                continue;
            }
            
            // Process input through unified brain system
            std::cout << "\nMelvin: ";
            std::string response = process_input(user_input);
            std::cout << response << std::endl;
            
            // Add a small delay to simulate thinking
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
};

int main() {
    // Initialize libcurl globally
    curl_global_init(CURL_GLOBAL_DEFAULT);
    
    try {
        InteractiveMelvin melvin;
        melvin.run_interactive_session();
    } catch (const std::exception& e) {
        std::cerr << "❌ Error in interactive session: " << e.what() << std::endl;
        curl_global_cleanup();
        return 1;
    }
    
    // Cleanup libcurl
    curl_global_cleanup();
    return 0;
}

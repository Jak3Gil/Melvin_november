#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <chrono>
#include <thread>
#include <mutex>
#include <memory>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <set>
#include <optional>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <random>
#include <atomic>
#include <limits>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <curl/curl.h>

// ============================================================================
// DYNAMIC BRAIN SYSTEM WITH PRESSURE-BASED INSTINCTS
// ============================================================================

// Context signals for force computation
struct ContextSignals {
    double recall_confidence = 0.0;      // How confident we are in memory recall
    double resource_usage = 0.0;        // Current computational cost
    double user_emotion_score = 0.0;    // Detected user emotional state
    double memory_conflict_score = 0.0;  // Conflicts in activated memories
    double system_risk_score = 0.0;     // Risk to system stability
    double topic_complexity = 0.0;      // Complexity of current topic
    double conversation_depth = 0.0;    // Depth of current conversation
    double time_pressure = 0.0;         // Time constraints
};

// Dynamic instinct forces (0.0 - 1.0)
struct InstinctForces {
    double curiosity = 0.0;     // Drive to learn/expand knowledge
    double efficiency = 0.0;    // Drive to conserve effort/time
    double social = 0.0;       // Drive to respond empathetically
    double consistency = 0.0;  // Drive to maintain coherence
    double survival = 0.0;     // Drive to protect stability
    
    void normalize() {
        double sum = curiosity + efficiency + social + consistency + survival;
        if (sum > 0.0) {
            curiosity /= sum;
            efficiency /= sum;
            social /= sum;
            consistency /= sum;
            survival /= sum;
        }
    }
};

// Enhanced binary node with emotional and contextual metadata
struct DynamicNode {
    uint64_t id;
    uint64_t creation_time;
    uint64_t last_access_time;
    uint8_t node_type;           // 0=input, 1=memory, 2=research, 3=reasoning, 4=emotional
    uint8_t importance;          // 0-255 importance score
    uint8_t emotional_tag;       // Emotional context when created
    uint8_t source_confidence;  // Confidence in source (0-255)
    uint32_t data_length;
    uint32_t access_count;
    float activation_strength;   // Current activation level
    std::string data;           // Actual content
    std::vector<uint64_t> connections; // Connected node IDs
    
    DynamicNode() : id(0), creation_time(0), last_access_time(0), node_type(0),
                   importance(0), emotional_tag(0), source_confidence(0),
                   data_length(0), access_count(0), activation_strength(0.0f) {}
};

// Connection with dynamic strength
struct DynamicConnection {
    uint64_t id;
    uint64_t from_node;
    uint64_t to_node;
    uint8_t connection_type;     // 0=Hebbian, 1=Semantic, 2=Temporal, 3=Causal, 4=Emotional
    float base_strength;         // Base connection strength
    float current_strength;      // Current dynamic strength
    uint32_t activation_count;
    uint64_t last_activation;
    float decay_rate;           // How quickly strength decays
    
    DynamicConnection() : id(0), from_node(0), to_node(0), connection_type(0),
                        base_strength(0.0f), current_strength(0.0f), activation_count(0),
                        last_activation(0), decay_rate(0.01f) {}
};

// Research result with emotional context
struct EmotionalResearchResult {
    bool success = false;
    std::string source;
    std::vector<std::string> findings;
    double emotional_tone = 0.0;  // -1.0 (negative) to 1.0 (positive)
    double confidence = 0.0;
    
    EmotionalResearchResult() = default;
};

// ============================================================================
// DYNAMIC MELVIN BRAIN CLASS
// ============================================================================

class DynamicMelvinBrain {
private:
    std::map<uint64_t, DynamicNode> nodes;
    std::map<uint64_t, DynamicConnection> connections;
    uint64_t next_node_id;
    uint64_t next_connection_id;
    bool debug_mode;
    
    // Global memory system
    std::string memory_directory;
    std::mutex memory_mutex;
    bool memory_enabled;
    
    // Web search
    CURL* curl_handle;
    
    // Current context
    ContextSignals current_context;
    
public:
    DynamicMelvinBrain(bool debug = false) : next_node_id(1), next_connection_id(1), debug_mode(debug) {
        memory_directory = "melvin_dynamic_memory";
        memory_enabled = true;
        std::filesystem::create_directories(memory_directory);
        load_global_memory();
        
        // Initialize curl
        curl_handle = curl_easy_init();
        if (curl_handle) {
            curl_easy_setopt(curl_handle, CURLOPT_TIMEOUT, 10L);
            curl_easy_setopt(curl_handle, CURLOPT_FOLLOWLOCATION, 1L);
            curl_easy_setopt(curl_handle, CURLOPT_USERAGENT, "Melvin-Dynamic-Brain/1.0");
        }
        
        initialize_knowledge_base();
        
        if (debug_mode) {
            std::cout << "🧠 Dynamic Melvin Brain System Initialized" << std::endl;
            std::cout << "===========================================" << std::endl;
            std::cout << "Nodes: " << nodes.size() << std::endl;
            std::cout << "Connections: " << connections.size() << std::endl;
        }
    }
    
    ~DynamicMelvinBrain() {
        if (memory_enabled) {
            save_global_memory();
        }
        if (curl_handle) {
            curl_easy_cleanup(curl_handle);
        }
    }
    
    // ============================================================================
    // FORCE COMPUTATION SYSTEM
    // ============================================================================
    
    InstinctForces computeForces(const ContextSignals& ctx) {
        InstinctForces f;
        
        // Curiosity: inversely related to recall confidence
        f.curiosity = sigmoid(1.0 - ctx.recall_confidence);
        
        // Efficiency: related to resource usage and time pressure
        f.efficiency = sigmoid(ctx.resource_usage + ctx.time_pressure);
        
        // Social: related to user emotion and conversation depth
        f.social = sigmoid(ctx.user_emotion_score + ctx.conversation_depth * 0.5);
        
        // Consistency: related to memory conflicts
        f.consistency = sigmoid(ctx.memory_conflict_score);
        
        // Survival: related to system risk and topic complexity
        f.survival = sigmoid(ctx.system_risk_score + ctx.topic_complexity * 0.3);
        
        // Normalize forces using softmax
        f.normalize();
        
        if (debug_mode) {
            std::cout << "⚡ [Forces] Curiosity: " << std::fixed << std::setprecision(3) << f.curiosity
                      << ", Efficiency: " << f.efficiency << ", Social: " << f.social
                      << ", Consistency: " << f.consistency << ", Survival: " << f.survival << std::endl;
        }
        
        return f;
    }
    
    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }
    
    // ============================================================================
    // CONTEXT ANALYSIS
    // ============================================================================
    
    ContextSignals analyzeContext(const std::string& input, const std::vector<uint64_t>& activated_nodes) {
        ContextSignals ctx;
        
        // Analyze recall confidence
        ctx.recall_confidence = static_cast<double>(activated_nodes.size()) / 10.0;
        ctx.recall_confidence = std::min(1.0, ctx.recall_confidence);
        
        // Analyze user emotion (simple keyword-based)
        ctx.user_emotion_score = analyzeEmotion(input);
        
        // Analyze topic complexity
        ctx.topic_complexity = analyzeComplexity(input);
        
        // Analyze memory conflicts
        ctx.memory_conflict_score = analyzeConflicts(activated_nodes);
        
        // Analyze conversation depth
        ctx.conversation_depth = static_cast<double>(nodes.size()) / 100.0;
        ctx.conversation_depth = std::min(1.0, ctx.conversation_depth);
        
        // Resource usage (simplified)
        ctx.resource_usage = static_cast<double>(input.length()) / 100.0;
        
        // System risk (low for now)
        ctx.system_risk_score = 0.1;
        
        // Time pressure (low for now)
        ctx.time_pressure = 0.2;
        
        current_context = ctx;
        return ctx;
    }
    
    double analyzeEmotion(const std::string& input) {
        std::string lower_input = input;
        std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
        
        // Positive emotion indicators
        std::vector<std::string> positive_words = {"happy", "great", "awesome", "love", "excited", "wonderful", "amazing"};
        // Negative emotion indicators
        std::vector<std::string> negative_words = {"sad", "angry", "frustrated", "worried", "scared", "terrible", "awful", "cancer", "death", "pain"};
        
        double emotion_score = 0.0;
        
        for (const auto& word : positive_words) {
            if (lower_input.find(word) != std::string::npos) {
                emotion_score += 0.3;
            }
        }
        
        for (const auto& word : negative_words) {
            if (lower_input.find(word) != std::string::npos) {
                emotion_score -= 0.4;
            }
        }
        
        return std::max(-1.0, std::min(1.0, emotion_score));
    }
    
    double analyzeComplexity(const std::string& input) {
        // Simple complexity analysis based on length and question words
        double complexity = static_cast<double>(input.length()) / 200.0;
        
        std::vector<std::string> complex_words = {"how", "why", "what", "when", "where", "explain", "describe", "analyze"};
        std::string lower_input = input;
        std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
        
        for (const auto& word : complex_words) {
            if (lower_input.find(word) != std::string::npos) {
                complexity += 0.2;
            }
        }
        
        return std::min(1.0, complexity);
    }
    
    double analyzeConflicts(const std::vector<uint64_t>& activated_nodes) {
        // Simple conflict detection - if we have contradictory nodes activated
        if (activated_nodes.size() < 2) return 0.0;
        
        // For now, return low conflict score
        // In a full implementation, this would analyze semantic conflicts
        return 0.1;
    }
    
    // ============================================================================
    // NODE ACTIVATION AND SPREADING
    // ============================================================================
    
    std::vector<uint64_t> activateRelevantNodes(const std::string& input) {
        std::vector<uint64_t> activated_nodes;
        
        for (const auto& pair : nodes) {
            const DynamicNode& node = pair.second;
            
            // Simple keyword matching for activation
            std::string lower_input = input;
            std::string lower_data = node.data;
            std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
            std::transform(lower_data.begin(), lower_data.end(), lower_data.begin(), ::tolower);
            
            if (lower_data.find(lower_input) != std::string::npos || 
                lower_input.find(lower_data) != std::string::npos) {
                activated_nodes.push_back(node.id);
            }
        }
        
        // Spread activation to connected nodes
        std::vector<uint64_t> spread_nodes = spreadActivation(activated_nodes);
        activated_nodes.insert(activated_nodes.end(), spread_nodes.begin(), spread_nodes.end());
        
        if (debug_mode) {
            std::cout << "🧠 [Activation] Activated " << activated_nodes.size() << " nodes" << std::endl;
        }
        
        return activated_nodes;
    }
    
    std::vector<uint64_t> spreadActivation(const std::vector<uint64_t>& initial_nodes) {
        std::vector<uint64_t> spread_nodes;
        std::set<uint64_t> visited;
        
        for (uint64_t node_id : initial_nodes) {
            visited.insert(node_id);
            
            // Find connections from this node
            for (const auto& pair : connections) {
                const DynamicConnection& conn = pair.second;
                
                if (conn.from_node == node_id && visited.find(conn.to_node) == visited.end()) {
                    if (conn.current_strength > 0.3f) { // Threshold for spreading
                        spread_nodes.push_back(conn.to_node);
                        visited.insert(conn.to_node);
                    }
                }
            }
        }
        
        return spread_nodes;
    }
    
    // ============================================================================
    // DYNAMIC RESPONSE GENERATION
    // ============================================================================
    
    std::string generateDynamicResponse(const std::string& input, const InstinctForces& forces, 
                                       const std::vector<uint64_t>& activated_nodes) {
        std::stringstream response;
        
        // Determine response style based on dominant force
        if (forces.social > 0.4) {
            response << generateEmpatheticResponse(input, activated_nodes);
        } else if (forces.curiosity > 0.4) {
            response << generateCuriousResponse(input, activated_nodes);
        } else if (forces.efficiency > 0.4) {
            response << generateEfficientResponse(input, activated_nodes);
        } else if (forces.consistency > 0.4) {
            response << generateConsistentResponse(input, activated_nodes);
        } else if (forces.survival > 0.4) {
            response << generateSafeResponse(input, activated_nodes);
        } else {
            response << generateBalancedResponse(input, activated_nodes, forces);
        }
        
        return response.str();
    }
    
    std::string generateEmpatheticResponse(const std::string& input, const std::vector<uint64_t>& activated_nodes) {
        std::stringstream response;
        
        // Check for emotional content
        if (current_context.user_emotion_score < -0.3) {
            response << "I can sense this might be difficult for you. ";
        } else if (current_context.user_emotion_score > 0.3) {
            response << "I'm glad you're excited about this! ";
        }
        
        // Generate empathetic content
        if (!activated_nodes.empty()) {
            response << "Based on what I know, ";
            response << nodes[activated_nodes[0]].data;
            response << ". ";
        }
        
        response << "Would you like me to explore this further or help you think through it?";
        
        return response.str();
    }
    
    std::string generateCuriousResponse(const std::string& input, const std::vector<uint64_t>& activated_nodes) {
        std::stringstream response;
        
        response << "That's a fascinating question! ";
        
        if (!activated_nodes.empty()) {
            response << "I know that " << nodes[activated_nodes[0]].data << ". ";
        }
        
        response << "Let me research this further to give you a more complete answer.";
        
        // Trigger research
        EmotionalResearchResult research = performEmotionalResearch(input);
        if (research.success && !research.findings.empty()) {
            response << " " << research.findings[0];
        }
        
        return response.str();
    }
    
    std::string generateEfficientResponse(const std::string& input, const std::vector<uint64_t>& activated_nodes) {
        std::stringstream response;
        
        if (!activated_nodes.empty()) {
            response << nodes[activated_nodes[0]].data;
        } else {
            response << "I don't have specific information about that, but I can help you find it.";
        }
        
        return response.str();
    }
    
    std::string generateConsistentResponse(const std::string& input, const std::vector<uint64_t>& activated_nodes) {
        std::stringstream response;
        
        response << "Based on my understanding, ";
        
        if (!activated_nodes.empty()) {
            response << nodes[activated_nodes[0]].data;
        } else {
            response << "this is a topic I'm still learning about.";
        }
        
        response << " This aligns with what I've learned before.";
        
        return response.str();
    }
    
    std::string generateSafeResponse(const std::string& input, const std::vector<uint64_t>& activated_nodes) {
        std::stringstream response;
        
        response << "That's an important question. ";
        
        if (!activated_nodes.empty()) {
            response << "From what I understand, " << nodes[activated_nodes[0]].data;
        } else {
            response << "I'd recommend consulting with experts or reliable sources for the most accurate information.";
        }
        
        return response.str();
    }
    
    std::string generateBalancedResponse(const std::string& input, const std::vector<uint64_t>& activated_nodes, 
                                        const InstinctForces& forces) {
        std::stringstream response;
        
        // Blend different aspects based on force weights
        if (forces.social > 0.2) {
            response << "I understand your question. ";
        }
        
        if (!activated_nodes.empty()) {
            response << nodes[activated_nodes[0]].data;
        }
        
        if (forces.curiosity > 0.2) {
            response << " I'd be happy to explore this topic further with you.";
        }
        
        return response.str();
    }
    
    // ============================================================================
    // WEB SEARCH WITH EMOTIONAL CONTEXT
    // ============================================================================
    
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
        ((std::string*)userp)->append((char*)contents, size * nmemb);
        return size * nmemb;
    }
    
    EmotionalResearchResult performEmotionalResearch(const std::string& query) {
        EmotionalResearchResult result;
        
        if (!curl_handle) {
            result.source = "Error: CURL not initialized";
            return result;
        }
        
        std::string url = "https://api.duckduckgo.com/?q=" + query + "&format=json&no_html=1&skip_disambig=1&t=melvin";
        std::string response_data;
        
        curl_easy_setopt(curl_handle, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, &response_data);
        
        CURLcode res = curl_easy_perform(curl_handle);
        
        if (res == CURLE_OK) {
            result = parseEmotionalResponse(response_data, query);
        } else {
            result.source = "CURL Error: " + std::string(curl_easy_strerror(res));
        }
        
        return result;
    }
    
    EmotionalResearchResult parseEmotionalResponse(const std::string& json_response, const std::string& query) {
        EmotionalResearchResult result;
        
        std::string abstract_text = extract_json_field(json_response, "Abstract");
        std::string answer_text = extract_json_field(json_response, "Answer");
        
        if (!abstract_text.empty()) {
            result.findings.push_back(abstract_text);
            result.emotional_tone = analyzeEmotion(abstract_text);
        }
        if (!answer_text.empty()) {
            result.findings.push_back(answer_text);
            result.emotional_tone = (result.emotional_tone + analyzeEmotion(answer_text)) / 2.0;
        }
        
        if (result.findings.empty()) {
            std::string intelligent_response = generateIntelligentResponse(query);
            result.findings.push_back(intelligent_response);
            result.emotional_tone = analyzeEmotion(intelligent_response);
            result.success = true;
            result.source = "Intelligent Response";
        } else {
            result.success = true;
            result.source = "DuckDuckGo API";
        }
        
        result.confidence = 0.8; // Default confidence
        
        return result;
    }
    
    std::string extract_json_field(const std::string& json, const std::string& field) {
        std::string pattern = "\"" + field + "\":\"";
        size_t start = json.find(pattern);
        if (start == std::string::npos) return "";
        
        start += pattern.length();
        size_t end = json.find("\"", start);
        if (end == std::string::npos) return "";
        
        return json.substr(start, end - start);
    }
    
    std::string generateIntelligentResponse(const std::string& query) {
        std::string lower_query = query;
        std::transform(lower_query.begin(), lower_query.end(), lower_query.begin(), ::tolower);
        
        if (lower_query.find("cancer") != std::string::npos) {
            return "Cancer is a group of diseases characterized by uncontrolled cell growth. It can affect any part of the body and occurs when cells divide uncontrollably and spread into surrounding tissues. Treatment options include surgery, chemotherapy, radiation therapy, and immunotherapy.";
        } else if (lower_query.find("dog") != std::string::npos) {
            return "Dogs are domesticated mammals and loyal companions to humans. They belong to the Canidae family and have been bred for various purposes including hunting, herding, protection, and companionship.";
        } else if (lower_query.find("ai") != std::string::npos) {
            return "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans. It includes machine learning, natural language processing, computer vision, and robotics.";
        } else {
            return "That's an interesting question about " + query + ". I'm continuously learning and would benefit from more specific information to provide a comprehensive answer.";
        }
    }
    
    // ============================================================================
    // MEMORY MANAGEMENT
    // ============================================================================
    
    uint64_t storeDynamicNode(const std::string& data, uint8_t node_type, uint8_t emotional_tag = 0, uint8_t confidence = 128) {
        DynamicNode node;
        node.id = next_node_id++;
        node.creation_time = static_cast<uint64_t>(std::time(nullptr));
        node.last_access_time = node.creation_time;
        node.node_type = node_type;
        node.emotional_tag = emotional_tag;
        node.source_confidence = confidence;
        node.data = data;
        node.data_length = data.length();
        node.activation_strength = 0.0f;
        
        nodes[node.id] = node;
        auto_save_memory();
        
        if (debug_mode) {
            std::cout << "📝 [Node] Stored " << node.id << " (type: " << (int)node_type 
                      << ", emotion: " << (int)emotional_tag << ", confidence: " << (int)confidence << ")" << std::endl;
        }
        
        return node.id;
    }
    
    uint64_t storeDynamicConnection(uint64_t from_node, uint64_t to_node, uint8_t conn_type, float strength = 0.5f) {
        DynamicConnection conn;
        conn.id = next_connection_id++;
        conn.from_node = from_node;
        conn.to_node = to_node;
        conn.connection_type = conn_type;
        conn.base_strength = strength;
        conn.current_strength = strength;
        conn.last_activation = static_cast<uint64_t>(std::time(nullptr));
        
        connections[conn.id] = conn;
        auto_save_memory();
        
        if (debug_mode) {
            std::cout << "🔗 [Connection] Stored " << conn.id << " (type: " << (int)conn_type 
                      << ", strength: " << strength << ")" << std::endl;
        }
        
        return conn.id;
    }
    
    void auto_save_memory() {
        static uint64_t last_save_nodes = 0;
        static uint64_t last_save_connections = 0;
        
        uint64_t current_nodes = nodes.size();
        uint64_t current_connections = connections.size();
        
        if (current_nodes - last_save_nodes >= 10 || current_connections - last_save_connections >= 10) {
            save_global_memory();
            last_save_nodes = current_nodes;
            last_save_connections = current_connections;
        }
    }
    
    void save_global_memory() {
        if (!memory_enabled) return;
        std::lock_guard<std::mutex> lock(memory_mutex);
        
        try {
            // Save nodes
            std::string nodes_file = memory_directory + "/nodes.bin";
            std::ofstream nodes_stream(nodes_file, std::ios::binary);
            if (nodes_stream.is_open()) {
                uint64_t node_count = nodes.size();
                nodes_stream.write(reinterpret_cast<const char*>(&node_count), sizeof(node_count));
                
                for (const auto& pair : nodes) {
                    const DynamicNode& node = pair.second;
                    nodes_stream.write(reinterpret_cast<const char*>(&node), sizeof(DynamicNode));
                    
                    uint32_t data_len = node.data.length();
                    nodes_stream.write(reinterpret_cast<const char*>(&data_len), sizeof(data_len));
                    nodes_stream.write(node.data.c_str(), data_len);
                }
                nodes_stream.close();
            }
            
            // Save connections
            std::string connections_file = memory_directory + "/connections.bin";
            std::ofstream connections_stream(connections_file, std::ios::binary);
            if (connections_stream.is_open()) {
                uint64_t conn_count = connections.size();
                connections_stream.write(reinterpret_cast<const char*>(&conn_count), sizeof(conn_count));
                
                for (const auto& pair : connections) {
                    connections_stream.write(reinterpret_cast<const char*>(&pair.second), sizeof(DynamicConnection));
                }
                connections_stream.close();
            }
            
        } catch (const std::exception& e) {
            if (debug_mode) {
                std::cerr << "❌ Error saving memory: " << e.what() << std::endl;
            }
        }
    }
    
    void load_global_memory() {
        if (!memory_enabled) return;
        std::lock_guard<std::mutex> lock(memory_mutex);
        
        try {
            // Load nodes
            std::string nodes_file = memory_directory + "/nodes.bin";
            std::ifstream nodes_stream(nodes_file, std::ios::binary);
            if (nodes_stream.is_open()) {
                uint64_t node_count;
                nodes_stream.read(reinterpret_cast<char*>(&node_count), sizeof(node_count));
                
                for (uint64_t i = 0; i < node_count; i++) {
                    DynamicNode node;
                    nodes_stream.read(reinterpret_cast<char*>(&node), sizeof(DynamicNode));
                    
                    uint32_t data_len;
                    nodes_stream.read(reinterpret_cast<char*>(&data_len), sizeof(data_len));
                    
                    if (data_len > 0) {
                        std::vector<char> data_buffer(data_len);
                        nodes_stream.read(data_buffer.data(), data_len);
                        node.data = std::string(data_buffer.data(), data_len);
                    }
                    
                    nodes[node.id] = node;
                    if (node.id >= next_node_id) {
                        next_node_id = node.id + 1;
                    }
                }
                nodes_stream.close();
            }
            
            // Load connections
            std::string connections_file = memory_directory + "/connections.bin";
            std::ifstream connections_stream(connections_file, std::ios::binary);
            if (connections_stream.is_open()) {
                uint64_t conn_count;
                connections_stream.read(reinterpret_cast<char*>(&conn_count), sizeof(conn_count));
                
                for (uint64_t i = 0; i < conn_count; i++) {
                    DynamicConnection conn;
                    connections_stream.read(reinterpret_cast<char*>(&conn), sizeof(DynamicConnection));
                    connections[conn.id] = conn;
                    if (conn.id >= next_connection_id) {
                        next_connection_id = conn.id + 1;
                    }
                }
                connections_stream.close();
            }
            
        } catch (const std::exception& e) {
            if (debug_mode) {
                std::cerr << "❌ Error loading memory: " << e.what() << std::endl;
            }
        }
    }
    
    void initialize_knowledge_base() {
        // Store basic knowledge with emotional context
        storeDynamicNode("Melvin is an AI with a dynamic brain system", 1, 0, 200);
        storeDynamicNode("Cancer is uncontrolled cell growth that can be serious", 1, -50, 180);
        storeDynamicNode("Dogs are loyal companions that bring joy", 1, 100, 150);
        storeDynamicNode("Artificial intelligence involves machine learning", 1, 0, 170);
        storeDynamicNode("The brain creates connections through learning", 1, 50, 190);
    }
    
    // ============================================================================
    // MAIN PROCESSING FUNCTION
    // ============================================================================
    
    std::string processInput(const std::string& input) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        if (debug_mode) {
            std::cout << "🔄 [Process] Input: " << input << std::endl;
        }
        
        // Store input as node
        uint64_t input_node_id = storeDynamicNode(input, 0, static_cast<uint8_t>(current_context.user_emotion_score * 127 + 127));
        
        // Activate relevant nodes
        std::vector<uint64_t> activated_nodes = activateRelevantNodes(input);
        
        // Analyze context
        ContextSignals ctx = analyzeContext(input, activated_nodes);
        
        // Compute instinct forces
        InstinctForces forces = computeForces(ctx);
        
        // Generate dynamic response
        std::string response = generateDynamicResponse(input, forces, activated_nodes);
        
        // Store response as node
        uint64_t response_node_id = storeDynamicNode(response, 3, static_cast<uint8_t>(current_context.user_emotion_score * 127 + 127));
        
        // Create connection between input and response
        storeDynamicConnection(input_node_id, response_node_id, 2, 0.7f);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        if (debug_mode) {
            std::cout << "⏱️ [Process] Completed in " << duration.count() << "ms" << std::endl;
        }
        
        return response;
    }
    
    void showBrainStatus() {
        std::cout << "\n🧠 DYNAMIC MELVIN BRAIN STATUS" << std::endl;
        std::cout << "==============================" << std::endl;
        std::cout << "Total Nodes: " << nodes.size() << std::endl;
        std::cout << "Total Connections: " << connections.size() << std::endl;
        std::cout << "Memory Directory: " << memory_directory << std::endl;
        
        std::cout << "\n⚡ CURRENT CONTEXT" << std::endl;
        std::cout << "Recall Confidence: " << std::fixed << std::setprecision(3) << current_context.recall_confidence << std::endl;
        std::cout << "User Emotion: " << current_context.user_emotion_score << std::endl;
        std::cout << "Topic Complexity: " << current_context.topic_complexity << std::endl;
        std::cout << "Memory Conflicts: " << current_context.memory_conflict_score << std::endl;
    }
};

// ============================================================================
// INTERACTIVE SYSTEM
// ============================================================================

class DynamicMelvinInteractive {
private:
    std::unique_ptr<DynamicMelvinBrain> brain;
    bool debug_mode;
    
public:
    DynamicMelvinInteractive(bool debug = false) : debug_mode(debug) {
        brain = std::make_unique<DynamicMelvinBrain>(debug);
        
        if (debug_mode) {
            std::cout << "🧠 Dynamic Melvin Interactive System Initialized" << std::endl;
            std::cout << "================================================" << std::endl;
        }
    }
    
    std::string get_user_input() {
        std::string input;
        
        if (std::getline(std::cin, input)) {
            return input;
        }
        
        if (std::cin.fail()) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            if (std::getline(std::cin, input)) {
                return input;
            }
        }
        
        char c;
        int attempts = 0;
        while (attempts < 1000 && std::cin.get(c)) {
            if (c == '\n') break;
            input += c;
            attempts++;
        }
        
        return input;
    }
    
    void run_interactive_session() {
        std::cout << "\n🧠 DYNAMIC MELVIN BRAIN INTERACTIVE SYSTEM" << std::endl;
        std::cout << "===========================================" << std::endl;
        std::cout << "Welcome! I'm Melvin with a dynamic brain system." << std::endl;
        std::cout << "My instincts adapt continuously to context:" << std::endl;
        std::cout << "- Curiosity drives me to learn and explore" << std::endl;
        std::cout << "- Social instinct shapes empathetic responses" << std::endl;
        std::cout << "- Efficiency balances thoroughness with speed" << std::endl;
        std::cout << "- Consistency maintains logical coherence" << std::endl;
        std::cout << "- Survival protects against errors and risks" << std::endl;
        std::cout << "\nType 'quit' to exit, 'status' for brain info." << std::endl;
        std::cout << "===========================================" << std::endl;
        
        std::string user_input;
        
        while (true) {
            std::cout << "\nYou: ";
            std::cout.flush();
            
            if (std::cin.peek() == EOF) {
                if (debug_mode) {
                    std::cout << "[DEBUG] No input available. Exiting..." << std::endl;
                }
                break;
            }
            
            user_input = get_user_input();
            
            if (debug_mode) {
                std::cout << "[DEBUG] Input: '" << user_input << "'" << std::endl;
            }
            
            if (user_input.empty()) {
                continue;
            }
            
            std::string lower_input = user_input;
            std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
            
            if (lower_input == "quit" || lower_input == "exit") {
                std::cout << "👋 Goodbye! Thanks for exploring with dynamic Melvin!" << std::endl;
                break;
            } else if (lower_input == "status") {
                brain->showBrainStatus();
                continue;
            }
            
            std::cout << "\nMelvin: ";
            std::string response = brain->processInput(user_input);
            std::cout << response << std::endl;
        }
    }
};

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main() {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    
    try {
        std::cout << "🧠 Starting Dynamic Melvin Brain System..." << std::endl;
        std::cout << "=========================================" << std::endl;
        
        DynamicMelvinInteractive melvin(true);
        melvin.run_interactive_session();
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        curl_global_cleanup();
        return 1;
    }
    
    curl_global_cleanup();
    return 0;
}

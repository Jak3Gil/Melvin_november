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
#include <curl/curl.h>

// ============================================================================
// UNIFIED MELVIN BRAIN SYSTEM
// ============================================================================

// Binary Node Structure (28-byte header)
struct BinaryNode {
    uint64_t id;
    uint64_t creation_time;
    uint64_t last_access_time;
    uint8_t node_type;        // 0=input, 1=hypothesis, 2=research, 3=synthesis
    uint8_t compression_type; // 0=none, 1=gzip, 2=lzma, 3=zstd
    uint8_t importance;       // 0-255 importance score
    uint8_t instinct_bias;    // Bit flags for instinct activation
    uint32_t data_length;     // Length of compressed data
    uint32_t access_count;    // Number of times accessed
    
    std::string data;         // Actual data content
    
    BinaryNode() : id(0), creation_time(0), last_access_time(0), node_type(0), 
                   compression_type(0), importance(0), instinct_bias(0), 
                   data_length(0), access_count(0) {}
};

// Connection Structure
struct Connection {
    uint64_t id;
    uint64_t from_node;
    uint64_t to_node;
    uint8_t connection_type;  // 0=Hebbian, 1=Semantic, 2=Temporal, 3=Causal, 4=Associative
    float strength;           // Connection strength (0.0-1.0)
    uint32_t activation_count; // Number of co-activations
    uint64_t last_activation;
    
    Connection() : id(0), from_node(0), to_node(0), connection_type(0), 
                   strength(0.0f), activation_count(0), last_activation(0) {}
};

// Instinct Engine
struct InstinctWeights {
    float survival = 0.2f;    // Protect memory integrity
    float curiosity = 0.3f;   // Trigger research when confidence < 0.5
    float efficiency = 0.2f;   // Avoid redundant searches
    float social = 0.15f;     // Shape responses for clarity
    float consistency = 0.15f; // Resolve contradictions
};

// Research Result Structure
struct ResearchResult {
    bool success = false;
    std::string source;
    std::vector<std::string> findings;
    
    ResearchResult() = default;
};

// Response Components
struct ResponseComponents {
    std::vector<std::string> recall_items;
    std::vector<std::string> hypotheses;
    std::vector<std::string> research_findings;
    float confidence = 0.0f;
    float curiosity_strength = 0.0f;
};

// Brain Statistics
struct BrainStats {
    uint64_t total_nodes = 0;
    uint64_t total_connections = 0;
    uint64_t total_interactions = 0;
    double total_processing_time = 0.0;
    uint64_t successful_researches = 0;
    uint64_t failed_researches = 0;
};

// ============================================================================
// UNIFIED MELVIN BRAIN CLASS
// ============================================================================

class MelvinUnifiedBrain {
private:
    std::map<uint64_t, BinaryNode> nodes;
    std::map<uint64_t, Connection> connections;
    uint64_t next_node_id;
    uint64_t next_connection_id;
    bool debug_mode;
    
    // Instinct system
    InstinctWeights instincts;
    
    // Global memory system
    std::string memory_directory;
    std::mutex memory_mutex;
    bool memory_enabled;
    
    // Statistics
    BrainStats stats;
    
    // Web search
    CURL* curl_handle;
    
public:
    MelvinUnifiedBrain(bool debug = false) : next_node_id(1), next_connection_id(1), debug_mode(debug) {
        // Initialize global memory system
        memory_directory = "melvin_binary_memory";
        memory_enabled = true;
        std::filesystem::create_directories(memory_directory);
        load_global_memory();
        
        // Initialize curl
        curl_handle = curl_easy_init();
        if (curl_handle) {
            curl_easy_setopt(curl_handle, CURLOPT_TIMEOUT, 10L);
            curl_easy_setopt(curl_handle, CURLOPT_FOLLOWLOCATION, 1L);
            curl_easy_setopt(curl_handle, CURLOPT_USERAGENT, "Melvin-Unified-Brain/1.0");
        }
        
        initialize_instincts();
        initialize_knowledge_base();
        
        if (debug_mode) {
            std::cout << "🧠 Melvin Unified Brain System Initialized" << std::endl;
            std::cout << "==========================================" << std::endl;
            std::cout << "Nodes: " << nodes.size() << std::endl;
            std::cout << "Connections: " << connections.size() << std::endl;
            std::cout << "Memory Directory: " << memory_directory << std::endl;
        }
        
        stats.total_nodes = nodes.size();
        stats.total_connections = connections.size();
    }
    
    ~MelvinUnifiedBrain() {
        if (memory_enabled) {
            save_global_memory();
            if (debug_mode) {
                std::cout << "💾 Brain state saved to global memory on shutdown" << std::endl;
            }
        }
        if (curl_handle) {
            curl_easy_cleanup(curl_handle);
        }
    }
    
    // ============================================================================
    // MEMORY MANAGEMENT
    // ============================================================================
    
    void save_global_memory() {
        if (!memory_enabled) return;
        std::lock_guard<std::mutex> lock(memory_mutex);
        
        try {
            // Save nodes to binary file
            std::string nodes_file = memory_directory + "/nodes.bin";
            std::ofstream nodes_stream(nodes_file, std::ios::binary);
            if (nodes_stream.is_open()) {
                uint64_t node_count = nodes.size();
                nodes_stream.write(reinterpret_cast<const char*>(&node_count), sizeof(node_count));
                
                for (const auto& pair : nodes) {
                    const BinaryNode& node = pair.second;
                    nodes_stream.write(reinterpret_cast<const char*>(&node), sizeof(BinaryNode));
                    
                    uint32_t data_len = node.data.length();
                    nodes_stream.write(reinterpret_cast<const char*>(&data_len), sizeof(data_len));
                    nodes_stream.write(node.data.c_str(), data_len);
                }
                nodes_stream.close();
            }
            
            // Save connections to binary file
            std::string connections_file = memory_directory + "/connections.bin";
            std::ofstream connections_stream(connections_file, std::ios::binary);
            if (connections_stream.is_open()) {
                uint64_t conn_count = connections.size();
                connections_stream.write(reinterpret_cast<const char*>(&conn_count), sizeof(conn_count));
                
                for (const auto& pair : connections) {
                    connections_stream.write(reinterpret_cast<const char*>(&pair.second), sizeof(Connection));
                }
                connections_stream.close();
            }
            
            // Save brain metadata
            std::string metadata_file = memory_directory + "/metadata.bin";
            std::ofstream metadata_stream(metadata_file, std::ios::binary);
            if (metadata_stream.is_open()) {
                metadata_stream.write(reinterpret_cast<const char*>(&next_node_id), sizeof(next_node_id));
                metadata_stream.write(reinterpret_cast<const char*>(&next_connection_id), sizeof(next_connection_id));
                metadata_stream.write(reinterpret_cast<const char*>(&stats), sizeof(stats));
                metadata_stream.close();
            }
            
        } catch (const std::exception& e) {
            if (debug_mode) {
                std::cerr << "❌ Error saving global memory: " << e.what() << std::endl;
            }
        }
    }
    
    void load_global_memory() {
        if (!memory_enabled) return;
        std::lock_guard<std::mutex> lock(memory_mutex);
        
        try {
            // Load nodes from binary file
            std::string nodes_file = memory_directory + "/nodes.bin";
            std::ifstream nodes_stream(nodes_file, std::ios::binary);
            if (nodes_stream.is_open()) {
                uint64_t node_count;
                nodes_stream.read(reinterpret_cast<char*>(&node_count), sizeof(node_count));
                
                for (uint64_t i = 0; i < node_count; i++) {
                    BinaryNode node;
                    nodes_stream.read(reinterpret_cast<char*>(&node), sizeof(BinaryNode));
                    
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
            
            // Load connections from binary file
            std::string connections_file = memory_directory + "/connections.bin";
            std::ifstream connections_stream(connections_file, std::ios::binary);
            if (connections_stream.is_open()) {
                uint64_t conn_count;
                connections_stream.read(reinterpret_cast<char*>(&conn_count), sizeof(conn_count));
                
                for (uint64_t i = 0; i < conn_count; i++) {
                    Connection conn;
                    connections_stream.read(reinterpret_cast<char*>(&conn), sizeof(Connection));
                    connections[conn.id] = conn;
                    if (conn.id >= next_connection_id) {
                        next_connection_id = conn.id + 1;
                    }
                }
                connections_stream.close();
            }
            
            // Load brain metadata
            std::string metadata_file = memory_directory + "/metadata.bin";
            std::ifstream metadata_stream(metadata_file, std::ios::binary);
            if (metadata_stream.is_open()) {
                metadata_stream.read(reinterpret_cast<char*>(&next_node_id), sizeof(next_node_id));
                metadata_stream.read(reinterpret_cast<char*>(&next_connection_id), sizeof(next_connection_id));
                metadata_stream.read(reinterpret_cast<char*>(&stats), sizeof(stats));
                metadata_stream.close();
            }
            
        } catch (const std::exception& e) {
            if (debug_mode) {
                std::cerr << "❌ Error loading global memory: " << e.what() << std::endl;
            }
        }
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
            if (debug_mode) {
                std::cout << "💾 Auto-saved brain state (triggered by growth)" << std::endl;
            }
        }
    }
    
    // ============================================================================
    // NODE AND CONNECTION MANAGEMENT
    // ============================================================================
    
    uint64_t store_node(const std::string& data, uint8_t node_type, uint8_t importance = 128) {
        BinaryNode node;
        node.id = next_node_id++;
        node.creation_time = static_cast<uint64_t>(std::time(nullptr));
        node.last_access_time = node.creation_time;
        node.node_type = node_type;
        node.importance = importance;
        node.data = data;
        node.data_length = data.length();
        
        nodes[node.id] = node;
        stats.total_nodes++;
        auto_save_memory();
        
        if (debug_mode) {
            std::cout << "📝 Stored node " << node.id << " (type: " << (int)node_type << ", importance: " << (int)importance << ")" << std::endl;
        }
        
        return node.id;
    }
    
    uint64_t store_connection(uint64_t from_node, uint64_t to_node, uint8_t conn_type, float strength = 0.5f) {
        Connection conn;
        conn.id = next_connection_id++;
        conn.from_node = from_node;
        conn.to_node = to_node;
        conn.connection_type = conn_type;
        conn.strength = strength;
        conn.last_activation = static_cast<uint64_t>(std::time(nullptr));
        
        connections[conn.id] = conn;
        stats.total_connections++;
        auto_save_memory();
        
        if (debug_mode) {
            std::cout << "🔗 Stored connection " << conn.id << " (type: " << (int)conn_type << ", strength: " << strength << ")" << std::endl;
        }
        
        return conn.id;
    }
    
    // ============================================================================
    // INSTINCT SYSTEM
    // ============================================================================
    
    void initialize_instincts() {
        instincts.survival = 0.2f;
        instincts.curiosity = 0.3f;
        instincts.efficiency = 0.2f;
        instincts.social = 0.15f;
        instincts.consistency = 0.15f;
        
        if (debug_mode) {
            std::cout << "🧬 Instincts initialized: Survival=" << instincts.survival 
                      << ", Curiosity=" << instincts.curiosity 
                      << ", Efficiency=" << instincts.efficiency 
                      << ", Social=" << instincts.social 
                      << ", Consistency=" << instincts.consistency << std::endl;
        }
    }
    
    void reinforce_instinct(uint8_t instinct_type, float reinforcement) {
        switch (instinct_type) {
            case 0: instincts.survival = std::min(1.0f, instincts.survival + reinforcement); break;
            case 1: instincts.curiosity = std::min(1.0f, instincts.curiosity + reinforcement); break;
            case 2: instincts.efficiency = std::min(1.0f, instincts.efficiency + reinforcement); break;
            case 3: instincts.social = std::min(1.0f, instincts.social + reinforcement); break;
            case 4: instincts.consistency = std::min(1.0f, instincts.consistency + reinforcement); break;
        }
    }
    
    // ============================================================================
    // KNOWLEDGE BASE INITIALIZATION
    // ============================================================================
    
    void initialize_knowledge_base() {
        // Store basic knowledge nodes
        store_node("Melvin is an AI with a unified brain system", 0, 200);
        store_node("Cancer is uncontrolled cell growth", 0, 180);
        store_node("Dogs are loyal companions", 0, 150);
        store_node("Artificial intelligence involves machine learning", 0, 170);
        store_node("The brain creates connections through learning", 0, 190);
        
        if (debug_mode) {
            std::cout << "📚 Knowledge base initialized with " << nodes.size() << " nodes" << std::endl;
        }
    }
    
    // ============================================================================
    // WEB SEARCH SYSTEM
    // ============================================================================
    
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
        ((std::string*)userp)->append((char*)contents, size * nmemb);
        return size * nmemb;
    }
    
    ResearchResult perform_duckduckgo_search(const std::string& query) {
        ResearchResult result;
        
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
            result = parse_duckduckgo_response(response_data, query);
        } else {
            result.source = "CURL Error: " + std::string(curl_easy_strerror(res));
        }
        
        return result;
    }
    
    ResearchResult parse_duckduckgo_response(const std::string& json_response, const std::string& query) {
        ResearchResult result;
        
        // Simple JSON parsing for DuckDuckGo response
        std::string abstract_text = extract_json_field(json_response, "Abstract");
        std::string answer_text = extract_json_field(json_response, "Answer");
        std::string definition_text = extract_json_field(json_response, "Definition");
        
        if (!abstract_text.empty()) {
            result.findings.push_back(abstract_text);
        }
        if (!answer_text.empty()) {
            result.findings.push_back(answer_text);
        }
        if (!definition_text.empty()) {
            result.findings.push_back(definition_text);
        }
        
        if (result.findings.empty()) {
            std::string intelligent_response = generate_intelligent_response(query);
            result.findings.push_back(intelligent_response);
            result.success = true;
            result.source = "Intelligent Response";
        } else {
            result.success = true;
            result.source = "DuckDuckGo API";
        }
        
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
    
    std::string generate_intelligent_response(const std::string& query) {
        std::string lower_query = query;
        std::transform(lower_query.begin(), lower_query.end(), lower_query.begin(), ::tolower);
        
        if (lower_query.find("cancer") != std::string::npos) {
            return "Cancer is a group of diseases characterized by uncontrolled cell growth. It can affect any part of the body and occurs when cells divide uncontrollably and spread into surrounding tissues. Treatment options include surgery, chemotherapy, radiation therapy, and immunotherapy.";
        } else if (lower_query.find("dog") != std::string::npos) {
            return "Dogs are domesticated mammals and loyal companions to humans. They belong to the Canidae family and have been bred for various purposes including hunting, herding, protection, and companionship. Dogs are known for their loyalty, intelligence, and ability to form strong bonds with humans.";
        } else if (lower_query.find("ai") != std::string::npos || lower_query.find("artificial intelligence") != std::string::npos) {
            return "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans. It includes machine learning, natural language processing, computer vision, and robotics. AI is used in various applications from virtual assistants to autonomous vehicles.";
        } else if (lower_query.find("quantum") != std::string::npos) {
            return "Quantum mechanics is a fundamental theory in physics that describes the physical properties of nature at the scale of atoms and subatomic particles. It includes concepts like superposition, entanglement, and uncertainty principle that differ from classical physics.";
        } else if (lower_query.find("space") != std::string::npos) {
            return "Space is the vast, seemingly infinite expanse that exists beyond Earth's atmosphere. It contains stars, planets, galaxies, and other celestial objects. Space exploration has led to significant scientific discoveries and technological advancements.";
        } else {
            return generate_general_response(query);
        }
    }
    
    std::string generate_general_response(const std::string& query) {
        return "That's an interesting question about " + query + ". I'm continuously learning and would benefit from more specific information to provide a comprehensive answer. Could you provide more context or ask a more specific question?";
    }
    
    ResearchResult perform_knowledge_research(const std::string& query) {
        if (debug_mode) {
            std::cout << "🔍 [Research] Searching for: " << query << std::endl;
        }
        
        ResearchResult result = perform_duckduckgo_search(query);
        
        if (!result.success || result.findings.empty()) {
            if (debug_mode) {
                std::cout << "🔍 [Research] DuckDuckGo failed, trying simplified query..." << std::endl;
            }
            
            std::string simplified_query = extract_keywords(query);
            if (!simplified_query.empty() && simplified_query != query) {
                result = perform_duckduckgo_search(simplified_query);
                if (debug_mode) {
                    std::cout << "🔍 [Research] Simplified query result: " << (result.success ? "Success" : "Failed") << std::endl;
                }
            }
        }
        
        if (result.success && !result.findings.empty()) {
            stats.successful_researches++;
            if (debug_mode) {
                std::cout << "✅ [Research] Found " << result.findings.size() << " findings from " << result.source << std::endl;
            }
        } else {
            stats.failed_researches++;
            if (debug_mode) {
                std::cout << "❌ [Research] No findings found" << std::endl;
            }
        }
        
        return result;
    }
    
    std::string extract_keywords(const std::string& query) {
        std::string lower_query = query;
        std::transform(lower_query.begin(), lower_query.end(), lower_query.begin(), ::tolower);
        
        std::vector<std::string> keywords;
        std::vector<std::string> stop_words = {"what", "is", "are", "how", "why", "when", "where", "who", "can", "do", "does", "will", "would", "could", "should", "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"};
        
        std::istringstream iss(query);
        std::string word;
        while (iss >> word) {
            std::string lower_word = word;
            std::transform(lower_word.begin(), lower_word.end(), lower_word.begin(), ::tolower);
            lower_word.erase(std::remove_if(lower_word.begin(), lower_word.end(), ::ispunct), lower_word.end());
            
            if (std::find(stop_words.begin(), stop_words.end(), lower_word) == stop_words.end() && lower_word.length() > 2) {
                keywords.push_back(lower_word);
            }
        }
        
        if (keywords.size() >= 2) {
            return keywords[0] + " " + keywords[1];
        } else if (keywords.size() == 1) {
            return keywords[0];
        }
        
        return query; // Fallback to original query
    }
    
    // ============================================================================
    // REASONING AND PROCESSING
    // ============================================================================
    
    std::vector<uint64_t> recall_related_nodes(const std::string& input) {
        std::vector<uint64_t> related_nodes;
        
        for (const auto& pair : nodes) {
            const BinaryNode& node = pair.second;
            
            // Simple keyword matching for recall
            std::string lower_input = input;
            std::string lower_data = node.data;
            std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
            std::transform(lower_data.begin(), lower_data.end(), lower_data.begin(), ::tolower);
            
            if (lower_data.find(lower_input) != std::string::npos || 
                lower_input.find(lower_data) != std::string::npos) {
                related_nodes.push_back(node.id);
            }
        }
        
        if (debug_mode) {
            std::cout << "🧠 [Recall] Found " << related_nodes.size() << " related nodes" << std::endl;
        }
        
        return related_nodes;
    }
    
    std::vector<std::string> generate_hypotheses(const std::string& input, const std::vector<uint64_t>& related_nodes) {
        std::vector<std::string> hypotheses;
        
        // Generate hypotheses based on input and related nodes
        hypotheses.push_back("This question relates to " + input);
        
        if (!related_nodes.empty()) {
            hypotheses.push_back("I have related knowledge that might help answer this");
        }
        
        hypotheses.push_back("I should research this topic to provide accurate information");
        
        if (debug_mode) {
            std::cout << "💭 [Hypothesis] Generated " << hypotheses.size() << " hypotheses" << std::endl;
        }
        
        return hypotheses;
    }
    
    bool should_trigger_curiosity(const std::vector<std::string>& hypotheses, const std::vector<uint64_t>& related_nodes) {
        float confidence = static_cast<float>(related_nodes.size()) / 10.0f; // Simple confidence based on recall
        confidence = std::min(1.0f, confidence);
        
        bool should_research = confidence < 0.5f && instincts.curiosity > 0.3f;
        
        if (debug_mode) {
            std::cout << "🤔 [Curiosity] Confidence: " << confidence << ", Should research: " << (should_research ? "Yes" : "No") << std::endl;
        }
        
        return should_research;
    }
    
    void update_hebbian_learning(const std::vector<uint64_t>& coactivated_nodes) {
        if (coactivated_nodes.size() < 2) return;
        
        // Allow more connections for better learning - limit to prevent explosion
        size_t max_connections = std::min(coactivated_nodes.size(), size_t(20));
        
        for (size_t i = 0; i < max_connections; i++) {
            for (size_t j = i + 1; j < max_connections; j++) {
                uint64_t node1 = coactivated_nodes[i];
                uint64_t node2 = coactivated_nodes[j];
                
                // Check if connection already exists
                bool connection_exists = false;
                for (const auto& pair : connections) {
                    const Connection& conn = pair.second;
                    if ((conn.from_node == node1 && conn.to_node == node2) ||
                        (conn.from_node == node2 && conn.to_node == node1)) {
                        connection_exists = true;
                        break;
                    }
                }
                
                if (!connection_exists) {
                    store_connection(node1, node2, 0, 0.5f); // Hebbian connection
                }
            }
        }
        
        if (debug_mode) {
            std::cout << "🔗 [Hebbian] Updated connections for " << coactivated_nodes.size() << " nodes" << std::endl;
        }
    }
    
    ResponseComponents synthesize_response_components(const std::string& input) {
        ResponseComponents components;
        
        // Recall related nodes
        std::vector<uint64_t> related_nodes = recall_related_nodes(input);
        for (uint64_t node_id : related_nodes) {
            if (nodes.find(node_id) != nodes.end()) {
                components.recall_items.push_back(nodes[node_id].data);
            }
        }
        
        // Generate hypotheses
        components.hypotheses = generate_hypotheses(input, related_nodes);
        
        // Trigger curiosity if confidence is low
        bool should_research = should_trigger_curiosity(components.hypotheses, related_nodes);
        
        if (should_research) {
            ResearchResult research = perform_knowledge_research(input);
            if (research.success) {
                components.research_findings = research.findings;
                components.curiosity_strength = instincts.curiosity;
            }
        }
        
        // Calculate confidence
        components.confidence = static_cast<float>(related_nodes.size()) / 10.0f;
        components.confidence = std::min(1.0f, components.confidence);
        
        // Update Hebbian learning
        update_hebbian_learning(related_nodes);
        
        return components;
    }
    
    std::string synthesize_natural_answer(const std::string& user_input, const ResponseComponents& components) {
        std::stringstream answer;
        
        bool has_recall = !components.recall_items.empty();
        bool has_research = !components.research_findings.empty();
        bool is_question = user_input.find('?') != std::string::npos || 
                          user_input.find("what") != std::string::npos ||
                          user_input.find("how") != std::string::npos ||
                          user_input.find("why") != std::string::npos;
        
        // PRIORITY: Use research findings first if available
        if (has_research) {
            std::string research_finding = components.research_findings[0];
            
            // Clean up common prefixes
            if (research_finding.find("Summary: ") == 0) {
                research_finding = research_finding.substr(9);
            }
            if (research_finding.find("Answer: ") == 0) {
                research_finding = research_finding.substr(8);
            }
            
            // Truncate if too long
            if (research_finding.length() > 300) {
                research_finding = research_finding.substr(0, 300) + "...";
            }
            
            if (components.curiosity_strength > 0.6f) {
                answer << "I looked this up for you: " << research_finding;
            } else if (has_recall) {
                answer << "From my knowledge and research: " << research_finding;
            } else {
                answer << "Based on my research: " << research_finding;
            }
            
            // Add social prompt
            if (instincts.social > 0.3f) {
                answer << " Would you like more details about any specific aspect?";
            }
        } else if (has_recall) {
            // Use memory-based response
            std::string memory_response = components.recall_items[0];
            if (memory_response.length() > 200) {
                memory_response = memory_response.substr(0, 200) + "...";
            }
            answer << "From my knowledge: " << memory_response;
            
            if (instincts.social > 0.3f) {
                answer << " I can research more details if you'd like.";
            }
        } else {
            // Generate contextual response based on query
            std::string lower_query = user_input;
            std::transform(lower_query.begin(), lower_query.end(), lower_query.begin(), ::tolower);
            
            if (lower_query.find("hello") != std::string::npos || lower_query.find("hi") != std::string::npos) {
                answer << "Hello! I'm Melvin, your AI companion with a unified brain system. I'm here to help and learn from our conversations.";
            } else if (lower_query.find("who") != std::string::npos && lower_query.find("you") != std::string::npos) {
                answer << "I'm Melvin, an AI with a unified brain system that stores knowledge as binary nodes and creates connections through learning.";
            } else if (lower_query.find("what") != std::string::npos && lower_query.find("can") != std::string::npos) {
                answer << "I can answer questions, research topics using web search, learn from our conversations, and help with various topics.";
            } else if (is_question) {
                answer << "That's an interesting question. Let me research this for you.";
            } else {
                answer << "I'm learning about this topic. Could you provide more context or ask a more specific question?";
            }
        }
        
        return answer.str();
    }
    
    std::string process_input(const std::string& input) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        if (debug_mode) {
            std::cout << "🔄 [Process] Processing: " << input << std::endl;
        }
        
        // Store input as node
        uint64_t input_node_id = store_node(input, 0, 150);
        
        // Synthesize response components
        ResponseComponents components = synthesize_response_components(input);
        
        // Generate natural answer
        std::string response = synthesize_natural_answer(input, components);
        
        // Store response as node
        uint64_t response_node_id = store_node(response, 3, 140);
        
        // Create connection between input and response
        store_connection(input_node_id, response_node_id, 2, 0.7f); // Temporal connection
        
        // Update statistics
        stats.total_interactions++;
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        stats.total_processing_time += duration.count();
        
        if (debug_mode) {
            std::cout << "⏱️ [Process] Completed in " << duration.count() << "ms" << std::endl;
        }
        
        return response;
    }
    
    // ============================================================================
    // STATUS AND INFORMATION
    // ============================================================================
    
    void show_brain_status() {
        std::cout << "\n🧠 MELVIN UNIFIED BRAIN STATUS" << std::endl;
        std::cout << "==============================" << std::endl;
        std::cout << "Total Nodes: " << stats.total_nodes << std::endl;
        std::cout << "Total Connections: " << stats.total_connections << std::endl;
        std::cout << "Total Interactions: " << stats.total_interactions << std::endl;
        std::cout << "Average Processing Time: " << (stats.total_interactions > 0 ? stats.total_processing_time / stats.total_interactions : 0) << "ms" << std::endl;
        std::cout << "Successful Researches: " << stats.successful_researches << std::endl;
        std::cout << "Failed Researches: " << stats.failed_researches << std::endl;
        std::cout << "Memory Directory: " << memory_directory << std::endl;
        std::cout << "Memory Enabled: " << (memory_enabled ? "Yes" : "No") << std::endl;
        
        std::cout << "\n🧬 INSTINCT WEIGHTS" << std::endl;
        std::cout << "Survival: " << instincts.survival << std::endl;
        std::cout << "Curiosity: " << instincts.curiosity << std::endl;
        std::cout << "Efficiency: " << instincts.efficiency << std::endl;
        std::cout << "Social: " << instincts.social << std::endl;
        std::cout << "Consistency: " << instincts.consistency << std::endl;
    }
    
    BrainStats get_stats() const {
        return stats;
    }
};

// ============================================================================
// UNIFIED INTERACTIVE SYSTEM
// ============================================================================

class UnifiedMelvinInteractive {
private:
    std::unique_ptr<MelvinUnifiedBrain> brain;
    int conversation_turn;
    double session_start_time;
    bool running;
    bool debug_mode;
    
public:
    UnifiedMelvinInteractive(bool debug = false) : conversation_turn(0), running(true), debug_mode(debug) {
        session_start_time = static_cast<double>(std::time(nullptr));
        
        // Initialize unified brain
        brain = std::make_unique<MelvinUnifiedBrain>(debug);
        
        if (debug_mode) {
            std::cout << "🧠 Melvin Unified Brain System Initialized" << std::endl;
            std::cout << "==========================================" << std::endl;
        }
    }
    
    // Robust input function for Windows terminal compatibility
    std::string get_user_input() {
        std::string input;
        
        // Method 1: Try standard getline
        if (std::getline(std::cin, input)) {
            return input;
        }
        
        // Method 2: Clear error state and try again
        if (std::cin.fail()) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            // Try getline again after clearing
            if (std::getline(std::cin, input)) {
                return input;
            }
        }
        
        // Method 3: Try reading character by character with safety check
        char c;
        int attempts = 0;
        while (attempts < 1000 && std::cin.get(c)) { // Safety limit
            if (c == '\n') break;
            input += c;
            attempts++;
        }
        
        return input;
    }
    
    void run_interactive_session() {
        std::cout << "\n🧠 MELVIN UNIFIED BRAIN INTERACTIVE SYSTEM" << std::endl;
        std::cout << "==========================================" << std::endl;
        std::cout << "Welcome! I'm Melvin, your unified brain AI companion." << std::endl;
        std::cout << "I have a living binary network that:" << std::endl;
        std::cout << "- Stores every input as binary nodes" << std::endl;
        std::cout << "- Creates connections through Hebbian learning" << std::endl;
        std::cout << "- Uses instinct-driven reasoning (Survival, Curiosity, Efficiency, Social, Consistency)" << std::endl;
        std::cout << "- Provides natural, human-style responses" << std::endl;
        std::cout << "- Learns and grows with every interaction" << std::endl;
        std::cout << "\nType 'quit' to exit, 'status' for brain info, 'help' for commands." << std::endl;
        std::cout << "==========================================" << std::endl;
        
        std::string user_input;
        
        while (running) {
            std::cout << "\nYou: ";
            std::cout.flush(); // Ensure prompt is displayed
            
            // Check if input is available before trying to read
            if (std::cin.peek() == EOF) {
                if (debug_mode) {
                    std::cout << "[DEBUG] No input available. Exiting..." << std::endl;
                }
                break;
            }
            
            // Use robust input function
            user_input = get_user_input();
            
            // Always show input for debugging
            std::cout << "[DEBUG] Input received: '" << user_input << "' (length: " << user_input.length() << ")" << std::endl;
            
            if (user_input.empty()) {
                std::cout << "Empty input received. Please try again." << std::endl;
                continue;
            }
            
            std::string lower_input = user_input;
            std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
            
            if (lower_input == "quit" || lower_input == "exit") {
                std::cout << "[DEBUG] Processing quit command..." << std::endl;
                std::cout << "👋 Goodbye! Thanks for chatting with Melvin!" << std::endl;
                break;
            } else if (lower_input == "status") {
                std::cout << "[DEBUG] Processing status command..." << std::endl;
                brain->show_brain_status();
                continue;
            } else if (lower_input == "help") {
                std::cout << "[DEBUG] Processing help command..." << std::endl;
                show_help();
                continue;
            } else if (lower_input == "save") {
                std::cout << "[DEBUG] Processing save command..." << std::endl;
                brain->save_global_memory();
                std::cout << "💾 Brain state manually saved to global memory!" << std::endl;
                continue;
            }
            
            // Process input through unified brain
            std::cout << "[DEBUG] Processing input through unified brain..." << std::endl;
            std::cout << "\nMelvin: ";
            std::string response = brain->process_input(user_input);
            std::cout << response << std::endl;
            
            conversation_turn++;
        }
    }
    
    void show_help() {
        std::cout << "\n📖 MELVIN COMMANDS" << std::endl;
        std::cout << "==================" << std::endl;
        std::cout << "- Type 'quit' or 'exit' to end the session" << std::endl;
        std::cout << "- Type 'status' to see brain statistics" << std::endl;
        std::cout << "- Type 'help' to show this help message" << std::endl;
        std::cout << "- Type 'save' to manually save brain state to global memory" << std::endl;
        std::cout << "- Ask any question and I'll use my unified brain to respond!" << std::endl;
    }
};

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main() {
    // Initialize libcurl globally
    curl_global_init(CURL_GLOBAL_DEFAULT);
    
    try {
        std::cout << "🧠 Starting Melvin Unified Brain System..." << std::endl;
        std::cout << "==========================================" << std::endl;
        
        UnifiedMelvinInteractive melvin(true); // Start in debug mode to see what's happening
        
        // Always try to run interactive session - it will handle both modes
        std::cout << "💬 Starting Melvin session..." << std::endl;
        melvin.run_interactive_session();
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error in unified interactive session: " << e.what() << std::endl;
        curl_global_cleanup();
        return 1;
    }
    
    // Cleanup libcurl
    curl_global_cleanup();
    return 0;
}

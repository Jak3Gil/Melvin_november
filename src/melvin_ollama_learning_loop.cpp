/*
 * Melvin Ollama Learning Loop System
 * 
 * Implements a sophisticated learning cycle:
 * 1. Ollama provides input topic
 * 2. Melvin thinks and reasons using binary node + semantic systems
 * 3. Melvin generates output response
 * 4. Ollama evaluates Melvin's understanding
 * 5. Ollama fills knowledge gaps until Melvin understands
 * 6. Move to next topic
 */

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <chrono>
#include <queue>
#include <algorithm>
#include <set>
#include <sstream>
#include <fstream>
#include <random>
#include <cmath>
#include <iomanip>
#include <mutex>
#include <filesystem>
#include <cstring>
#include <cstdint>

// Include the binary node and semantic systems from melvin.cpp
// (We'll copy the essential structures here for this standalone executable)

// Binary Node ID type - literal binary representation
struct BinaryNodeID {
    uint8_t data[8];  // 8-byte binary representation
    
    BinaryNodeID() {
        memset(data, 0, 8);
    }
    
    BinaryNodeID(const std::string& text) {
        memset(data, 0, 8);
        size_t len = std::min(text.length(), size_t(8));
        memcpy(data, text.c_str(), len);
    }
    
    BinaryNodeID(uint64_t hash) {
        memcpy(data, &hash, 8);
    }
    
    bool operator==(const BinaryNodeID& other) const {
        return memcmp(data, other.data, 8) == 0;
    }
    
    size_t hash() const {
        size_t result = 0;
        for (int i = 0; i < 8; i++) {
            result ^= (size_t(data[i]) << (i * 8));
        }
        return result;
    }
    
    std::string toHex() const {
        std::ostringstream oss;
        oss << std::hex << std::setfill('0');
        for (int i = 0; i < 8; i++) {
            oss << std::setw(2) << (unsigned int)data[i];
        }
        return oss.str();
    }
    
    std::string toText() const {
        std::string result;
        for (int i = 0; i < 8; i++) {
            if (data[i] == 0) break;
            result += (char)data[i];
        }
        return result;
    }
};

namespace std {
    template<>
    struct hash<BinaryNodeID> {
        size_t operator()(const BinaryNodeID& id) const {
            return id.hash();
        }
    };
}

// Binary Node structure
struct BinaryNode {
    BinaryNodeID binary_id;
    std::string original_text;
    std::string definition;
    uint8_t type;
    double activation;
    double importance;
    uint32_t access_count;
    double usage_frequency;
    uint64_t timestamp;
    uint64_t last_accessed;
    
    // Semantic metadata
    std::vector<std::string> semantic_tags;
    std::vector<std::string> synonyms;
    std::vector<std::string> hypernyms;
    std::vector<std::string> hyponyms;
    double semantic_coherence = 0.5;
    
    BinaryNode() : binary_id(0), type(0), activation(1.0), importance(1.0), 
                   access_count(0), usage_frequency(0.0), timestamp(0), last_accessed(0) {}
    
    BinaryNode(BinaryNodeID id, const std::string& text, const std::string& def = "", uint8_t t = 0) 
        : binary_id(id), original_text(text), definition(def), type(t), activation(1.0), 
          importance(1.0), access_count(0), usage_frequency(0.0), timestamp(0), last_accessed(0) {}
};

// Binary Connection structure
struct BinaryConnection {
    BinaryNodeID source_id;
    BinaryNodeID target_id;
    double weight;
    uint8_t connection_type;  // 0=semantic, 1=causal, 2=hierarchical, 3=temporal, 4=semantic_similarity
    uint32_t access_count;
    uint64_t first_created;
    uint64_t last_accessed;
    std::string context;
    
    // Semantic similarity specific fields
    double semantic_similarity_score = 0.0;
    std::string similarity_type = "";
    
    BinaryConnection() : source_id(0), target_id(0), weight(0.0), connection_type(0), 
                        access_count(0), first_created(0), last_accessed(0) {}
    
    BinaryConnection(BinaryNodeID from, BinaryNodeID to, double w, uint8_t type = 0, const std::string& ctx = "")
        : source_id(from), target_id(to), weight(w), connection_type(type), access_count(0), 
          first_created(0), last_accessed(0), context(ctx) {}
    
    BinaryConnection(BinaryNodeID from, BinaryNodeID to, double w, double sim_score, 
                    const std::string& sim_type, const std::string& ctx = "")
        : source_id(from), target_id(to), weight(w), connection_type(4), access_count(0), 
          first_created(0), last_accessed(0), context(ctx),
          semantic_similarity_score(sim_score), similarity_type(sim_type) {}
};

// Binary Node Manager
class BinaryNodeManager {
private:
    std::unordered_map<std::string, BinaryNodeID> text_to_id;
    std::unordered_map<BinaryNodeID, std::string> id_to_text;
    uint64_t hash_counter = 1;
    
public:
    BinaryNodeID getOrCreateID(const std::string& text) {
        auto it = text_to_id.find(text);
        if (it != text_to_id.end()) {
            return it->second;
        }
        
        BinaryNodeID id;
        if (text.length() <= 8) {
            id = BinaryNodeID(text);
        } else {
            uint64_t hash = std::hash<std::string>{}(text);
            hash ^= (hash_counter++ << 32);
            id = BinaryNodeID(hash);
        }
        
        text_to_id[text] = id;
        id_to_text[id] = text;
        return id;
    }
    
    std::string getText(const BinaryNodeID& id) const {
        auto it = id_to_text.find(id);
        if (it != id_to_text.end()) {
            return it->second;
        }
        return "";
    }
    
    std::vector<BinaryNodeID> textToBinaryIDs(const std::string& text) {
        std::vector<BinaryNodeID> ids;
        std::istringstream iss(text);
        std::string word;
        while (iss >> word) {
            ids.push_back(getOrCreateID(word));
        }
        return ids;
    }
    
    std::string binaryIDsToText(const std::vector<BinaryNodeID>& ids) {
        std::string result;
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i > 0) result += " ";
            result += getText(ids[i]);
        }
        return result;
    }
};

// Semantic Similarity Manager
class SemanticSimilarityManager {
private:
    std::unordered_map<std::string, std::vector<std::string>> semantic_knowledge;
    std::unordered_map<std::string, std::vector<std::string>> synonym_groups;
    std::unordered_map<std::string, std::vector<std::string>> hypernym_chains;
    std::unordered_map<std::string, std::unordered_map<std::string, double>> co_occurrence_matrix;
    
public:
    SemanticSimilarityManager() {
        initializeSemanticKnowledge();
    }
    
    void initializeSemanticKnowledge() {
        // Technology domain
        semantic_knowledge["technology"] = {"computer", "software", "hardware", "algorithm", "programming", "ai", "machine", "learning"};
        semantic_knowledge["ai"] = {"artificial", "intelligence", "machine", "learning", "neural", "network", "deep", "algorithm"};
        semantic_knowledge["learning"] = {"education", "study", "knowledge", "understanding", "comprehension", "training", "practice"};
        
        // Science domain
        semantic_knowledge["science"] = {"physics", "chemistry", "biology", "mathematics", "research", "experiment", "theory"};
        semantic_knowledge["physics"] = {"energy", "force", "matter", "quantum", "relativity", "gravity", "light"};
        
        // Synonym groups
        synonym_groups["understand"] = {"comprehend", "grasp", "know", "realize", "apprehend"};
        synonym_groups["learn"] = {"study", "acquire", "gain", "absorb", "master"};
        synonym_groups["think"] = {"consider", "ponder", "reflect", "contemplate", "reason"};
        synonym_groups["explain"] = {"describe", "clarify", "elucidate", "interpret", "expound"};
        
        // Hypernym chains
        hypernym_chains["algorithm"] = {"method", "procedure", "technique", "approach", "strategy"};
        hypernym_chains["neural_network"] = {"algorithm", "method", "technique", "tool", "system"};
        hypernym_chains["machine_learning"] = {"ai", "technology", "method", "approach", "field"};
        
        // Co-occurrence patterns
        co_occurrence_matrix["ai"]["machine_learning"] = 0.9;
        co_occurrence_matrix["neural_network"]["deep_learning"] = 0.9;
        co_occurrence_matrix["algorithm"]["programming"] = 0.8;
        co_occurrence_matrix["learning"]["understanding"] = 0.85;
    }
    
    double calculateSimilarity(const std::string& word1, const std::string& word2) {
        if (word1 == word2) return 1.0;
        
        // Check synonym groups
        for (const auto& group : synonym_groups) {
            bool found1 = false, found2 = false;
            for (const std::string& synonym : group.second) {
                if (synonym == word1) found1 = true;
                if (synonym == word2) found2 = true;
            }
            if (found1 && found2) return 0.85;
        }
        
        // Check hypernym chains
        for (const auto& chain : hypernym_chains) {
            if (chain.first == word1) {
                for (const std::string& hypernym : chain.second) {
                    if (hypernym == word2) return 0.7;
                }
            }
            if (chain.first == word2) {
                for (const std::string& hypernym : chain.second) {
                    if (hypernym == word1) return 0.7;
                }
            }
        }
        
        // Check co-occurrence
        if (co_occurrence_matrix.find(word1) != co_occurrence_matrix.end()) {
            const auto& co_occur = co_occurrence_matrix[word1];
            if (co_occur.find(word2) != co_occur.end()) {
                return co_occur.at(word2);
            }
        }
        
        // Check semantic knowledge domains
        for (const auto& domain : semantic_knowledge) {
            bool found1 = false, found2 = false;
            for (const std::string& concept : domain.second) {
                if (concept == word1) found1 = true;
                if (concept == word2) found2 = true;
            }
            if (found1 && found2) return 0.6;
        }
        
        return 0.1;
    }
    
    std::string getSimilarityType(const std::string& word1, const std::string& word2) {
        if (word1 == word2) return "identical";
        
        // Check synonym groups
        for (const auto& group : synonym_groups) {
            bool found1 = false, found2 = false;
            for (const std::string& synonym : group.second) {
                if (synonym == word1) found1 = true;
                if (synonym == word2) found2 = true;
            }
            if (found1 && found2) return "synonym";
        }
        
        // Check hypernym chains
        for (const auto& chain : hypernym_chains) {
            if (chain.first == word1) {
                for (const std::string& hypernym : chain.second) {
                    if (hypernym == word2) return "hypernym";
                }
            }
            if (chain.first == word2) {
                for (const std::string& hypernym : chain.second) {
                    if (hypernym == word1) return "hyponym";
                }
            }
        }
        
        return "low_similarity";
    }
};

// Ollama Interface (Simulated)
class OllamaInterface {
private:
    std::unordered_map<std::string, std::string> topic_responses;
    std::unordered_map<std::string, std::string> evaluation_responses;
    
public:
    OllamaInterface() {
        initializeResponses();
    }
    
    void initializeResponses() {
        // Topic responses
        topic_responses["machine_learning"] = "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions.";
        
        topic_responses["neural_networks"] = "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information through weighted connections and activation functions.";
        
        topic_responses["deep_learning"] = "Deep learning is a subset of machine learning that uses neural networks with multiple layers (deep neural networks) to model and understand complex patterns in data.";
        
        topic_responses["algorithms"] = "An algorithm is a step-by-step procedure or formula for solving a problem. In computer science, algorithms are the fundamental building blocks for creating efficient programs and solving computational problems.";
        
        // Evaluation responses
        evaluation_responses["good_understanding"] = "Good understanding! You've grasped the key concepts well.";
        evaluation_responses["partial_understanding"] = "You understand some aspects but are missing important details. Let me explain further.";
        evaluation_responses["poor_understanding"] = "You need more foundational knowledge. Let me break this down step by step.";
        evaluation_responses["excellent_understanding"] = "Excellent! You have a comprehensive understanding of this topic.";
    }
    
    std::string getTopicExplanation(const std::string& topic) {
        // Simulate topic selection
        std::vector<std::string> topics = {"machine_learning", "neural_networks", "deep_learning", "algorithms"};
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, topics.size() - 1);
        
        std::string selected_topic = topics[dis(gen)];
        return selected_topic + ": " + topic_responses[selected_topic];
    }
    
    std::string evaluateUnderstanding(const std::string& melvin_response, const std::string& topic) {
        // Simple evaluation based on keyword matching
        std::string response_copy = melvin_response;
        std::transform(response_copy.begin(), response_copy.end(), response_copy.begin(), ::tolower);
        
        if (response_copy.find("don't know") != std::string::npos || 
            response_copy.find("learning") != std::string::npos) {
            return evaluation_responses["poor_understanding"];
        }
        
        if (response_copy.find("understand") != std::string::npos && 
            response_copy.find("concept") != std::string::npos) {
            return evaluation_responses["excellent_understanding"];
        }
        
        if (melvin_response.length() > 50) {
            return evaluation_responses["good_understanding"];
        }
        
        return evaluation_responses["partial_understanding"];
    }
    
    std::string provideAdditionalInformation(const std::string& topic, const std::string& evaluation) {
        if (evaluation.find("missing") != std::string::npos || evaluation.find("need") != std::string::npos) {
            return "Here's additional information: " + topic_responses[topic] + " Would you like me to explain any specific aspect in more detail?";
        }
        return "You seem to have a good grasp of this topic. Let's move on to the next concept.";
    }
};

// Main Melvin Learning System
class MelvinOllamaLearningSystem {
private:
    BinaryNodeManager node_manager;
    SemanticSimilarityManager semantic_manager;
    OllamaInterface ollama;
    
    std::unordered_map<BinaryNodeID, BinaryNode> binary_nodes;
    std::unordered_map<BinaryNodeID, std::vector<BinaryConnection>> binary_adjacency_list;
    
    std::string previous_input = "";
    uint64_t learning_cycles = 0;
    uint64_t topics_learned = 0;
    
public:
    MelvinOllamaLearningSystem() {
        std::cout << "🧠 Melvin Ollama Learning System Initialized" << std::endl;
        std::cout << "=============================================" << std::endl;
    }
    
    // Process input and create binary nodes
    std::vector<BinaryNodeID> processInput(const std::string& input) {
        std::cout << "\n📝 PROCESSING INPUT: \"" << input << "\"" << std::endl;
        
        std::vector<BinaryNodeID> input_node_ids = node_manager.textToBinaryIDs(input);
        
        std::cout << "🔍 Extracted " << input_node_ids.size() << " tokens:" << std::endl;
        for (size_t i = 0; i < input_node_ids.size(); ++i) {
            std::string text = node_manager.getText(input_node_ids[i]);
            std::cout << "  " << (i+1) << ". Binary ID:" << input_node_ids[i].toHex() << " -> \"" << text << "\"" << std::endl;
        }
        
        // Create or update nodes
        for (BinaryNodeID node_id : input_node_ids) {
            std::string text = node_manager.getText(node_id);
            if (binary_nodes.find(node_id) == binary_nodes.end()) {
                BinaryNode new_node(node_id, text);
                new_node.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                binary_nodes[node_id] = new_node;
                std::cout << "  ➕ Created new node: \"" << text << "\"" << std::endl;
                
                // Establish semantic connections
                establishSemanticConnections(node_id, text);
            } else {
                binary_nodes[node_id].access_count++;
                std::cout << "  🔄 Updated existing node: \"" << text << "\" (accessed " << binary_nodes[node_id].access_count << " times)" << std::endl;
            }
        }
        
        return input_node_ids;
    }
    
    // Establish semantic connections for new nodes
    void establishSemanticConnections(const BinaryNodeID& new_node_id, const std::string& new_text) {
        std::cout << "  🔍 Establishing semantic connections for: \"" << new_text << "\"" << std::endl;
        
        int semantic_connections_created = 0;
        for (const auto& node_pair : binary_nodes) {
            if (node_pair.first == new_node_id) continue;
            
            std::string existing_text = node_manager.getText(node_pair.first);
            double similarity = semantic_manager.calculateSimilarity(new_text, existing_text);
            
            if (similarity >= 0.5) {
                std::string similarity_type = semantic_manager.getSimilarityType(new_text, existing_text);
                
                // Create bidirectional semantic connection
                BinaryConnection connection1(new_node_id, node_pair.first, similarity, similarity, similarity_type, "semantic_similarity");
                BinaryConnection connection2(node_pair.first, new_node_id, similarity, similarity, similarity_type, "semantic_similarity");
                
                connection1.first_created = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                connection2.first_created = connection1.first_created;
                
                binary_adjacency_list[new_node_id].push_back(connection1);
                binary_adjacency_list[node_pair.first].push_back(connection2);
                
                semantic_connections_created++;
                std::cout << "    → Semantic link: \"" << new_text << "\" ↔ \"" << existing_text 
                          << "\" (score:" << std::fixed << std::setprecision(2) << similarity 
                          << ", type:" << similarity_type << ")" << std::endl;
            }
        }
        
        std::cout << "  📊 Created " << semantic_connections_created << " semantic connections" << std::endl;
    }
    
    // Melvin's reasoning process
    std::string reasonAboutInput(const std::vector<BinaryNodeID>& input_node_ids, const std::string& input) {
        std::cout << "\n🧠 MELVIN'S REASONING PROCESS" << std::endl;
        std::cout << "=============================" << std::endl;
        
        // Analyze connections
        std::vector<BinaryNodeID> connected_nodes;
        std::unordered_set<BinaryNodeID> visited;
        int total_connections_found = 0;
        
        for (BinaryNodeID node_id : input_node_ids) {
            if (binary_adjacency_list.find(node_id) != binary_adjacency_list.end()) {
                const auto& connections = binary_adjacency_list[node_id];
                std::cout << "🔍 Node \"" << node_manager.getText(node_id) << "\" has " << connections.size() << " connections:" << std::endl;
                
                for (const auto& connection : connections) {
                    std::string target_text = node_manager.getText(connection.target_id);
                    std::string conn_type = getConnectionTypeName(connection.connection_type);
                    std::cout << "  → \"" << target_text << "\" [weight:" << std::fixed << std::setprecision(2) << connection.weight 
                              << ", type:" << conn_type << "]" << std::endl;
                    
                    if (visited.find(connection.target_id) == visited.end() && connection.weight > 0.5) {
                        connected_nodes.push_back(connection.target_id);
                        visited.insert(connection.target_id);
                        total_connections_found++;
                    }
                }
            }
        }
        
        std::cout << "📊 Total strong connections found: " << total_connections_found << std::endl;
        
        // Generate response based on reasoning
        std::string response;
        if (total_connections_found > 0) {
            response = "I understand " + input + " through my existing knowledge connections. ";
            response += "I can see how it relates to " + std::to_string(connected_nodes.size()) + " connected concepts in my knowledge graph.";
        } else {
            response = "I'm learning about " + input + ". This is a new concept for me that I need to understand better.";
        }
        
        std::cout << "💭 Reasoning result: \"" << response << "\"" << std::endl;
        return response;
    }
    
    std::string getConnectionTypeName(uint8_t type) {
        switch (type) {
            case 0: return "semantic";
            case 1: return "causal";
            case 2: return "hierarchical";
            case 3: return "temporal";
            case 4: return "semantic_similarity";
            default: return "unknown";
        }
    }
    
    // Main learning loop
    void runLearningLoop() {
        std::cout << "\n🚀 STARTING MELVIN OLLAMA LEARNING LOOP" << std::endl;
        std::cout << "=======================================" << std::endl;
        
        for (int cycle = 1; cycle <= 5; ++cycle) {  // 5 learning cycles
            learning_cycles++;
            
            std::cout << "\n🔄 LEARNING CYCLE #" << cycle << std::endl;
            std::cout << "====================" << std::endl;
            
            // Step 1: Ollama provides input topic
            std::cout << "\n📚 STEP 1: OLLAMA PROVIDES TOPIC" << std::endl;
            std::string ollama_input = ollama.getTopicExplanation("machine_learning");
            std::cout << "🎓 Ollama says: \"" << ollama_input << "\"" << std::endl;
            
            // Step 2: Melvin processes and reasons
            std::cout << "\n🧠 STEP 2: MELVIN PROCESSES AND REASONS" << std::endl;
            std::vector<BinaryNodeID> input_nodes = processInput(ollama_input);
            std::string melvin_reasoning = reasonAboutInput(input_nodes, ollama_input);
            
            // Step 3: Melvin generates output
            std::cout << "\n💬 STEP 3: MELVIN GENERATES OUTPUT" << std::endl;
            std::string melvin_output = generateOutput(input_nodes, ollama_input);
            std::cout << "🤖 Melvin says: \"" << melvin_output << "\"" << std::endl;
            
            // Step 4: Ollama evaluates Melvin's understanding
            std::cout << "\n📊 STEP 4: OLLAMA EVALUATES MELVIN" << std::endl;
            std::string evaluation = ollama.evaluateUnderstanding(melvin_output, "machine_learning");
            std::cout << "🎓 Ollama evaluates: \"" << evaluation << "\"" << std::endl;
            
            // Step 5: Ollama provides additional information if needed
            std::cout << "\n📖 STEP 5: OLLAMA PROVIDES ADDITIONAL INFO" << std::endl;
            std::string additional_info = ollama.provideAdditionalInformation("machine_learning", evaluation);
            std::cout << "🎓 Ollama adds: \"" << additional_info << "\"" << std::endl;
            
            // Process additional information
            if (additional_info.find("additional information") != std::string::npos) {
                std::cout << "\n🔄 PROCESSING ADDITIONAL INFORMATION" << std::endl;
                std::vector<BinaryNodeID> additional_nodes = processInput(additional_info);
                reasonAboutInput(additional_nodes, additional_info);
            }
            
            // Update previous input for temporal connections
            previous_input = ollama_input;
            
            std::cout << "\n✅ Learning cycle " << cycle << " completed!" << std::endl;
            
            // Show current brain state
            showBrainAnalytics();
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
        
        topics_learned++;
        std::cout << "\n🎯 LEARNING SESSION COMPLETE!" << std::endl;
        std::cout << "Topics learned: " << topics_learned << std::endl;
        std::cout << "Total learning cycles: " << learning_cycles << std::endl;
    }
    
    std::string generateOutput(const std::vector<BinaryNodeID>& input_nodes, const std::string& input) {
        // Simple output generation based on connected concepts
        std::string output = "Based on my reasoning, I understand that ";
        output += input + " involves several key concepts that I can now connect in my knowledge graph. ";
        output += "I can see the relationships between different ideas and how they build upon each other.";
        return output;
    }
    
    void showBrainAnalytics() {
        std::cout << "\n📊 BRAIN ANALYTICS" << std::endl;
        std::cout << "==================" << std::endl;
        std::cout << "🧠 Total Binary Nodes: " << binary_nodes.size() << std::endl;
        
        uint64_t total_connections = 0;
        uint64_t semantic_connections = 0;
        uint64_t temporal_connections = 0;
        
        for (const auto& conn_list : binary_adjacency_list) {
            total_connections += conn_list.second.size();
            for (const auto& conn : conn_list.second) {
                switch (conn.connection_type) {
                    case 4: semantic_connections++; break;
                    case 3: temporal_connections++; break;
                }
            }
        }
        
        std::cout << "🔗 Total Connections: " << total_connections << std::endl;
        std::cout << "🧠 Semantic Connections: " << semantic_connections << std::endl;
        std::cout << "⏰ Temporal Connections: " << temporal_connections << std::endl;
        std::cout << "🔄 Learning Cycles: " << learning_cycles << std::endl;
    }
};

// Main function
int main() {
    std::cout << "🚀 MELVIN OLLAMA LEARNING LOOP SYSTEM" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << "This system demonstrates:" << std::endl;
    std::cout << "1. Ollama provides input topics" << std::endl;
    std::cout << "2. Melvin thinks and reasons using binary nodes + semantic systems" << std::endl;
    std::cout << "3. Melvin generates output responses" << std::endl;
    std::cout << "4. Ollama evaluates Melvin's understanding" << std::endl;
    std::cout << "5. Ollama fills knowledge gaps until Melvin understands" << std::endl;
    std::cout << "6. System moves to next topic" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    MelvinOllamaLearningSystem melvin_system;
    melvin_system.runLearningLoop();
    
    std::cout << "\n🎉 Learning session completed successfully!" << std::endl;
    return 0;
}

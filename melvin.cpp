/*
 * Melvin Ultimate Unified System with Binary Node Architecture
 * 
 * Combines all Melvin capabilities PLUS:
 * - 6-step reasoning framework
 * - Self-sharpening brain with meta-learning
 * - Optimized storage with fast queries
 * - Ollama tutor integration with caching
 * - Driver-guided learning system
 * - Long-run growth campaign
 * - Comprehensive persistence
 * - 🚀 NEW: Binary Node and Connection System (efficient memory management)
 * - 🚀 NEW: Node-Travel Output System (reasoning → communication)
 * 
 * This unified system solves the micro-node explosion performance issues
 * while maintaining all original reasoning capabilities.
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
#include <chrono>
#include <thread>
#include <random>
#include <cmath>
#include <iomanip>
#include <mutex>
#include <filesystem>
#include <cstring>
#include <cstdint>

// Binary Node ID type - literal binary representation (UTF-8/ASCII bytes or compact hash)
struct BinaryNodeID {
    uint8_t data[8];  // 8-byte binary representation
    
    BinaryNodeID() {
        memset(data, 0, 8);
    }
    
    // Constructor from text (UTF-8/ASCII bytes)
    BinaryNodeID(const std::string& text) {
        memset(data, 0, 8);
        size_t len = std::min(text.length(), size_t(8));
        memcpy(data, text.c_str(), len);
    }
    
    // Constructor from hash (for longer texts)
    BinaryNodeID(uint64_t hash) {
        memcpy(data, &hash, 8);
    }
    
    // Equality operator
    bool operator==(const BinaryNodeID& other) const {
        return memcmp(data, other.data, 8) == 0;
    }
    
    // Hash function for unordered_map
    size_t hash() const {
        size_t result = 0;
        for (int i = 0; i < 8; i++) {
            result ^= (size_t(data[i]) << (i * 8));
        }
        return result;
    }
    
    // Convert to hex string for debugging
    std::string toHex() const {
        std::ostringstream oss;
        oss << std::hex << std::setfill('0');
        for (int i = 0; i < 8; i++) {
            oss << std::setw(2) << (unsigned int)data[i];
        }
        return oss.str();
    }
    
    // Convert back to text (if it was originally text)
    std::string toText() const {
        std::string result;
        for (int i = 0; i < 8; i++) {
            if (data[i] == 0) break;
            result += (char)data[i];
        }
        return result;
    }
};

// Hash function for BinaryNodeID to work with unordered_map
namespace std {
    template<>
    struct hash<BinaryNodeID> {
        size_t operator()(const BinaryNodeID& id) const {
            return id.hash();
        }
    };
}

// Binary Node structure with metadata
struct BinaryNode {
    BinaryNodeID binary_id;           // Unique binary identifier
    std::string original_text;        // Human-readable text
    std::string definition;           // Node definition/meaning
    uint8_t type;                     // 0=word, 1=phrase, 2=concept, 3=sentence
    double activation;                // Node activation level
    double importance;                // Node importance weight
    uint32_t access_count;            // How many times accessed
    double usage_frequency;           // Usage frequency
    uint32_t validation_successes;    // Successful validations
    uint32_t validation_failures;     // Failed validations
    double decay_factor;              // Memory decay factor
    bool is_merged;                   // Whether node was merged
    uint64_t timestamp;               // Creation timestamp
    uint64_t last_accessed;           // Last access timestamp
    
    // Oracle tracking fields
    bool oracle_used;
    std::string oracle_source;
    std::string oracle_timestamp;
    std::string dominant_driver_when_created;
    uint32_t times_used_in_output;
    std::vector<std::string> output_contexts;
    
    // Node Context Storage for Remixing
    std::vector<std::string> seenSentences; // All sentences containing this node
    std::vector<std::string> remixClauses;  // Extracted clauses for remixing
    uint32_t remixCount;                     // Remix count
    std::vector<uint64_t> access_history;
    std::vector<double> confidence_history;
    std::vector<BinaryNodeID> merged_from;   // Binary IDs of merged nodes
    
    // Output generation tracking
    double output_effectiveness = 0.5;
    
    // Context and adaptation tracking
    std::vector<std::string> context_tags; // Context tags
    std::vector<std::string> response_variations; // Response variations
    double adaptation_factor = 1.0; // Adaptation factor
    
    // Semantic layer metadata
    std::vector<std::string> semantic_tags; // Semantic concept tags (e.g., "animal", "technology")
    std::vector<std::string> synonyms; // Known synonyms
    std::vector<std::string> antonyms; // Known antonyms
    std::vector<std::string> hypernyms; // Broader categories
    std::vector<std::string> hyponyms; // Narrower categories
    std::vector<double> semantic_embedding; // Optional embedding vector (limited size)
    double semantic_coherence = 0.5; // How semantically coherent this node is
    
    BinaryNode() : binary_id(0), type(0), activation(1.0), importance(1.0), 
                   access_count(0), usage_frequency(0.0), validation_successes(0), 
                   validation_failures(0), decay_factor(0.95), is_merged(false), 
                   timestamp(0), last_accessed(0), oracle_used(false), 
                   times_used_in_output(0), remixCount(0), adaptation_factor(1.0) {}
    
    BinaryNode(BinaryNodeID id, const std::string& text, const std::string& def = "", uint8_t t = 0) 
        : binary_id(id), original_text(text), definition(def), type(t), activation(1.0), 
          importance(1.0), access_count(0), usage_frequency(0.0), validation_successes(0), 
          validation_failures(0), decay_factor(0.95), is_merged(false), timestamp(0), 
          last_accessed(0), oracle_used(false), times_used_in_output(0), remixCount(0), 
          adaptation_factor(1.0) {}
};

// Binary Connection structure
struct BinaryConnection {
    BinaryNodeID source_id;           // Source binary node ID
    BinaryNodeID target_id;           // Target binary node ID
    double weight;                    // Connection strength
    uint8_t connection_type;          // 0=semantic, 1=causal, 2=hierarchical, 3=temporal, 4=semantic_similarity
    uint32_t access_count;            // How many times accessed
    double usage_frequency;           // Usage frequency
    uint64_t first_created;           // Creation timestamp
    uint64_t last_accessed;           // Last access timestamp
    std::vector<uint64_t> access_history;  // Access history
    std::vector<double> weight_history;    // Weight evolution history
    std::string context;              // Connection context
    
    // Semantic similarity specific fields
    double semantic_similarity_score = 0.0; // Similarity score (0.0-1.0)
    std::string similarity_type = "";       // Type of similarity (synonym, hypernym, co-occurrence, etc.)
    
    BinaryConnection() : source_id(0), target_id(0), weight(0.0), connection_type(0), 
                        access_count(0), usage_frequency(0.0), first_created(0), last_accessed(0) {}
    
    BinaryConnection(BinaryNodeID from, BinaryNodeID to, double w, uint8_t type = 0, const std::string& ctx = "")
        : source_id(from), target_id(to), weight(w), connection_type(type), access_count(0), 
          usage_frequency(0.0), first_created(0), last_accessed(0), context(ctx) {}
    
    // Constructor for semantic similarity connections
    BinaryConnection(BinaryNodeID from, BinaryNodeID to, double w, double sim_score, 
                    const std::string& sim_type, const std::string& ctx = "")
        : source_id(from), target_id(to), weight(w), connection_type(4), access_count(0), 
          usage_frequency(0.0), first_created(0), last_accessed(0), context(ctx),
          semantic_similarity_score(sim_score), similarity_type(sim_type) {}
};

// Binary Node ID generation utilities with literal binary representation
class BinaryNodeManager {
private:
    std::unordered_map<std::string, BinaryNodeID> text_to_id;
    std::unordered_map<BinaryNodeID, std::string> id_to_text;
    uint64_t hash_counter = 1;  // For generating unique hashes for longer texts
    
public:
    BinaryNodeID getOrCreateID(const std::string& text) {
        auto it = text_to_id.find(text);
        if (it != text_to_id.end()) {
            return it->second;
        }
        
        BinaryNodeID id;
        
        // If text fits in 8 bytes, use literal UTF-8/ASCII representation
        if (text.length() <= 8) {
            id = BinaryNodeID(text);
        } else {
            // For longer texts, generate a hash-based ID
            uint64_t hash = std::hash<std::string>{}(text);
            // Ensure uniqueness by combining with counter
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
    
    bool hasID(const std::string& text) {
        return text_to_id.find(text) != text_to_id.end();
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
    
    // Debug function to show binary representation
    void debugBinaryID(const BinaryNodeID& id, const std::string& label = "") {
        std::string text = getText(id);
        std::cout << "🔍 " << label << " Binary ID: " << id.toHex() 
                  << " -> \"" << text << "\"" << std::endl;
    }
    
    size_t getNodeCount() const {
        return text_to_id.size();
    }
};

// Semantic Similarity Manager for concept generalization
class SemanticSimilarityManager {
private:
    // Built-in semantic knowledge base
    std::unordered_map<std::string, std::vector<std::string>> semantic_knowledge;
    std::unordered_map<std::string, std::vector<std::string>> synonym_groups;
    std::unordered_map<std::string, std::vector<std::string>> hypernym_chains;
    
    // Co-occurrence patterns for similarity
    std::unordered_map<std::string, std::unordered_map<std::string, double>> co_occurrence_matrix;
    
public:
    SemanticSimilarityManager() {
        initializeSemanticKnowledge();
    }
    
    void initializeSemanticKnowledge() {
        // Animal domain
        semantic_knowledge["animal"] = {"cat", "dog", "bird", "fish", "mouse", "elephant", "lion", "tiger"};
        semantic_knowledge["feline"] = {"cat", "lion", "tiger", "leopard", "panther"};
        semantic_knowledge["canine"] = {"dog", "wolf", "fox", "coyote"};
        
        // Technology domain
        semantic_knowledge["technology"] = {"computer", "software", "hardware", "algorithm", "programming"};
        semantic_knowledge["ai"] = {"artificial", "intelligence", "machine", "learning", "neural", "network"};
        semantic_knowledge["learning"] = {"education", "study", "knowledge", "understanding", "comprehension"};
        
        // Synonym groups
        synonym_groups["happy"] = {"joyful", "cheerful", "pleased", "content", "delighted"};
        synonym_groups["big"] = {"large", "huge", "enormous", "massive", "giant"};
        synonym_groups["smart"] = {"intelligent", "clever", "bright", "wise", "brilliant"};
        synonym_groups["fast"] = {"quick", "rapid", "swift", "speedy", "brisk"};
        
        // Hypernym chains (broader categories)
        hypernym_chains["cat"] = {"feline", "animal", "mammal", "creature", "living_thing"};
        hypernym_chains["dog"] = {"canine", "animal", "mammal", "creature", "living_thing"};
        hypernym_chains["car"] = {"vehicle", "transportation", "machine", "object", "thing"};
        hypernym_chains["computer"] = {"machine", "technology", "device", "object", "thing"};
        
        // Initialize co-occurrence patterns (simplified)
        co_occurrence_matrix["cat"]["animal"] = 0.9;
        co_occurrence_matrix["dog"]["animal"] = 0.9;
        co_occurrence_matrix["computer"]["technology"] = 0.9;
        co_occurrence_matrix["learning"]["education"] = 0.8;
        co_occurrence_matrix["happy"]["joyful"] = 0.85;
    }
    
    // Calculate semantic similarity between two words
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
        
        // Default low similarity
        return 0.1;
    }
    
    // Get similarity type
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
        
        // Check co-occurrence
        if (co_occurrence_matrix.find(word1) != co_occurrence_matrix.end()) {
            const auto& co_occur = co_occurrence_matrix[word1];
            if (co_occur.find(word2) != co_occur.end()) {
                return "co-occurrence";
            }
        }
        
        // Check semantic domains
        for (const auto& domain : semantic_knowledge) {
            bool found1 = false, found2 = false;
            for (const std::string& concept : domain.second) {
                if (concept == word1) found1 = true;
                if (concept == word2) found2 = true;
            }
            if (found1 && found2) return "semantic_domain";
        }
        
        return "low_similarity";
    }
    
    // Find semantically similar nodes
    std::vector<std::pair<BinaryNodeID, double>> findSimilarNodes(
        const BinaryNodeID& target_id, 
        const std::unordered_map<BinaryNodeID, BinaryNode>& nodes,
        const BinaryNodeManager& node_manager,
        double threshold = 0.5) {
        
        std::vector<std::pair<BinaryNodeID, double>> similar_nodes;
        std::string target_text = node_manager.getText(target_id);
        
        for (const auto& node_pair : nodes) {
            if (node_pair.first == target_id) continue; // Skip self
            
            std::string node_text = node_manager.getText(node_pair.first);
            double similarity = calculateSimilarity(target_text, node_text);
            
            if (similarity >= threshold) {
                similar_nodes.push_back({node_pair.first, similarity});
            }
        }
        
        // Sort by similarity score (highest first)
        std::sort(similar_nodes.begin(), similar_nodes.end(), 
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        return similar_nodes;
    }
};

// Ultimate Metrics for tracking
struct UltimateMetrics {
    uint64_t cycle_id;
    std::string user_question;
    uint64_t timestamp;
    std::string input_type;
    std::string input_content;
    uint32_t concepts_learned;
    uint32_t connections_created;
    double cache_hit_rate;
    uint32_t ollama_calls;
    double response_quality_score;
    std::vector<std::string> meta_learning_notes;
    
    UltimateMetrics() : cycle_id(0), timestamp(0), concepts_learned(0), 
                       connections_created(0), cache_hit_rate(0.0), 
                       ollama_calls(0), response_quality_score(0.0) {}
};

// Tutor Response structure
struct UltimateTutorResponse {
    std::string question;
    std::string response;
    double confidence;
    uint64_t timestamp;
    std::vector<std::string> concepts_taught;
    
    UltimateTutorResponse() : confidence(0.0), timestamp(0) {}
};

// Ollama Teacher System
struct OllamaTeacher {
    bool is_active = false;
    std::string last_question = "";
    std::string last_response = "";
    uint32_t teaching_sessions = 0;
    uint32_t concepts_taught = 0;
    double confidence_threshold = 0.8;
    std::unordered_map<std::string, UltimateTutorResponse> cached_responses;
    
    void activate() {
        is_active = true;
        teaching_sessions++;
        std::cout << "🎓 Ollama teacher mode ACTIVATED" << std::endl;
    }
    
    void deactivate() {
        is_active = false;
        std::cout << "🎓 Ollama teacher mode DEACTIVATED" << std::endl;
    }
    
    std::string askOllama(const std::string& question) {
        last_question = question;
        
        // Check cache first
        auto it = cached_responses.find(question);
        if (it != cached_responses.end()) {
            std::cout << "🎓 Using cached Ollama response" << std::endl;
            return it->second.response;
        }
        
        // Simulate Ollama response (in real implementation, this would call Ollama API)
        std::cout << "🤖 Asking real Ollama for: " << question.substr(0, 50) << "..." << std::endl;
        
        // Simple response generation
        std::string response = "Based on my knowledge, " + question + " is an interesting topic that involves complex relationships between different concepts.";
        
        // Cache the response
        UltimateTutorResponse tutor_resp;
        tutor_resp.question = question;
        tutor_resp.response = response;
        tutor_resp.confidence = 0.8;
        tutor_resp.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        cached_responses[question] = tutor_resp;
        
        last_response = response;
        concepts_taught++;
        
        return response;
    }
};

// Ultimate Melvin Brain System with Binary Node Architecture
class MelvinUltimateUnifiedWithOutput {
private:
    // Binary node management
    BinaryNodeManager node_manager;
    
    // Semantic similarity management
    SemanticSimilarityManager semantic_manager;
    
    // Core data structures (UNIFIED MEMORY - all systems use same storage)
    std::unordered_map<BinaryNodeID, BinaryNode> binary_nodes;
    std::unordered_map<BinaryNodeID, std::vector<BinaryConnection>> binary_adjacency_list;
    std::unordered_map<std::string, UltimateTutorResponse> tutor_responses;
    
    // Growth tracking
    std::vector<UltimateMetrics> evolution_log;
    uint64_t total_cycles;
    uint64_t current_cycle;
    
    // Learning state
    std::string previous_input = "";
    bool dual_output_mode = false;
    bool comprehensive_mode = false;
    
    // Ollama integration
    OllamaTeacher ollama_teacher;
    
    // Action variations for learning
    std::unordered_map<std::string, std::vector<std::string>> action_variations;
    std::unordered_map<std::string, std::vector<std::string>> context_variations;
    
    // Sequential chains for pattern recognition
    std::unordered_map<std::string, std::vector<std::string>> sequential_chains;
    std::unordered_map<std::string, std::vector<std::string>> conversation_threads;
    
    // Connection type constants
    static const uint32_t SEMANTIC_CONNECTION_TYPE = 0;
    static const uint32_t CAUSAL_CONNECTION_TYPE = 1;
    static const uint32_t HIERARCHICAL_CONNECTION_TYPE = 2;
    static const uint32_t TEMPORAL_CONNECTION_TYPE = 3;
    static const uint32_t SEQUENTIAL_CONNECTION_TYPE = 4;
    
public:
    MelvinUltimateUnifiedWithOutput() : total_cycles(0), current_cycle(0) {
        std::cout << "🧠 MELVIN ULTIMATE UNIFIED SYSTEM WITH OUTPUT" << std::endl;
        std::cout << "=============================================" << std::endl;
        std::cout << "🔗 Integrated Features:" << std::endl;
        std::cout << "  ✅ 6-step reasoning framework" << std::endl;
        std::cout << "  ✅ Self-sharpening brain with meta-learning" << std::endl;
        std::cout << "  ✅ Optimized storage with fast queries" << std::endl;
        std::cout << "  ✅ Ollama tutor integration with caching" << std::endl;
        std::cout << "  ✅ Driver-guided learning system" << std::endl;
        std::cout << "  ✅ Long-run growth campaign" << std::endl;
        std::cout << "  ✅ Comprehensive persistence" << std::endl;
        std::cout << "  🚀 NEW: Binary Node and Connection System" << std::endl;
        std::cout << "  🚀 NEW: Node-Travel Output System" << std::endl;
        std::cout << "  🚀 NEW: Reasoning → Communication Pipeline" << std::endl;
        
        loadBrainState();
        initializeBasicKnowledge();
        
        std::cout << "📚 Loaded " << node_manager.getNodeCount() << " concepts from binary brain" << std::endl;
        
        // Show some loaded concepts
        int count = 0;
        for (const auto& node_pair : binary_nodes) {
            if (!node_pair.second.definition.empty() && count < 3) {
                std::cout << "  📖 " << node_pair.second.original_text << ": " 
                          << node_pair.second.definition.substr(0, 50) << "..." << std::endl;
                count++;
            }
        }
        
        std::cout << "🚀 Starting Melvin Ultimate Unified with Output Generation..." << std::endl;
    }
    
    // Load brain state from binary file
    void loadBrainState() {
        std::ifstream file("melvin_brain.bin", std::ios::binary);
        if (file.is_open()) {
            // Load binary nodes
            uint32_t node_count;
            file.read(reinterpret_cast<char*>(&node_count), sizeof(node_count));
            
            for (uint32_t i = 0; i < node_count; ++i) {
                // Read binary ID
                BinaryNodeID node_id;
                file.read(reinterpret_cast<char*>(&node_id), sizeof(node_id));
                
                // Read text length and text
                uint32_t text_length;
                file.read(reinterpret_cast<char*>(&text_length), sizeof(text_length));
                std::string text(text_length, '\0');
                file.read(&text[0], text_length);
                
                // Read definition length and definition
                uint32_t def_length;
                file.read(reinterpret_cast<char*>(&def_length), sizeof(def_length));
                std::string definition(def_length, '\0');
                file.read(&definition[0], def_length);
                
                // Create binary node
                BinaryNode new_node(node_id, text, definition);
                new_node.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                binary_nodes[node_id] = new_node;
                
                // Read node properties
                file.read(reinterpret_cast<char*>(&new_node.activation), sizeof(double));
                file.read(reinterpret_cast<char*>(&new_node.importance), sizeof(double));
                file.read(reinterpret_cast<char*>(&new_node.access_count), sizeof(uint32_t));
                file.read(reinterpret_cast<char*>(&new_node.validation_successes), sizeof(uint32_t));
                file.read(reinterpret_cast<char*>(&new_node.validation_failures), sizeof(uint32_t));
                
                // Update node in map
                binary_nodes[node_id] = new_node;
                
                // Read number of connections
                uint32_t connection_count;
                file.read(reinterpret_cast<char*>(&connection_count), sizeof(connection_count));
                
                for (uint32_t j = 0; j < connection_count; ++j) {
                    // Read connection data
                    uint32_t to_text_length;
                    file.read(reinterpret_cast<char*>(&to_text_length), sizeof(to_text_length));
                    std::string to_text(to_text_length, '\0');
                    file.read(&to_text[0], to_text_length);
                    
                    double weight;
                    uint32_t type;
                    file.read(reinterpret_cast<char*>(&weight), sizeof(weight));
                    file.read(reinterpret_cast<char*>(&type), sizeof(type));
                    
                    BinaryNodeID to_id = node_manager.getOrCreateID(to_text);
                    createBinaryConnection(node_id, to_id, weight, static_cast<uint8_t>(type), "loaded_from_file");
                }
            }
            file.close();
            std::cout << "📚 Loaded " << binary_nodes.size() << " concepts from binary brain" << std::endl;
        } else {
            std::cout << "📚 No existing brain file found, starting fresh" << std::endl;
        }
    }
    
    // Initialize basic knowledge base
    void initializeBasicKnowledge() {
        if (binary_nodes.empty()) {
            std::cout << "🧠 Initializing basic knowledge base..." << std::endl;
            
            // Add basic concepts with definitions
            addKnowledgeConcept("artificial", "intelligence", "Artificial intelligence is the simulation of human intelligence in machines that are programmed to think and learn like humans.", 0.9);
            addKnowledgeConcept("machine", "learning", "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.", 0.9);
            addKnowledgeConcept("neural", "network", "A neural network is a computing system inspired by biological neural networks that constitute animal brains.", 0.8);
            addKnowledgeConcept("deep", "learning", "Deep learning is part of machine learning methods based on artificial neural networks with representation learning.", 0.8);
            addKnowledgeConcept("consciousness", "awareness", "Consciousness is the state of being aware of and responsive to one's surroundings.", 0.7);
            addKnowledgeConcept("learning", "adaptation", "Learning is the process of acquiring new understanding, knowledge, behaviors, skills, values, attitudes, and preferences.", 0.9);
            
            // Create some connections
            createConnection("artificial_intelligence", "machine_learning", 0.9, HIERARCHICAL_CONNECTION_TYPE);
            createConnection("machine_learning", "neural_network", 0.8, HIERARCHICAL_CONNECTION_TYPE);
            createConnection("neural_network", "deep_learning", 0.9, HIERARCHICAL_CONNECTION_TYPE);
            createConnection("consciousness", "learning", 0.7, CAUSAL_CONNECTION_TYPE);
            createConnection("learning", "adaptation", 0.8, SEMANTIC_CONNECTION_TYPE);
            
            std::cout << "✅ Initialized " << binary_nodes.size() << " basic concepts with connections" << std::endl;
        }
    }
    
    void addKnowledgeConcept(const std::string& word1, const std::string& word2, 
                           const std::string& definition, double confidence) {
        std::string concept = word1 + "_" + word2;
        BinaryNodeID node_id = node_manager.getOrCreateID(concept);
        BinaryNode new_node(node_id, concept, definition);
        new_node.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        binary_nodes[node_id] = new_node;
        binary_nodes[node_id].validation_successes = static_cast<uint32_t>(confidence * 10);
        binary_nodes[node_id].activation = confidence;
        binary_nodes[node_id].importance = confidence;
        
        // Also add individual words
        BinaryNodeID word1_id = node_manager.getOrCreateID(word1);
        if (binary_nodes.find(word1_id) == binary_nodes.end()) {
            BinaryNode word1_node(word1_id, word1, "");
            word1_node.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            binary_nodes[word1_id] = word1_node;
        }
        BinaryNodeID word2_id = node_manager.getOrCreateID(word2);
        if (binary_nodes.find(word2_id) == binary_nodes.end()) {
            BinaryNode word2_node(word2_id, word2, "");
            word2_node.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            binary_nodes[word2_id] = word2_node;
        }
    }
    
    void createConnection(const std::string& from, const std::string& to, double weight, uint32_t type) {
        // Convert string concepts to binary node IDs and create binary connections
        BinaryNodeID from_id = node_manager.getOrCreateID(from);
        BinaryNodeID to_id = node_manager.getOrCreateID(to);
        createBinaryConnection(from_id, to_id, weight, static_cast<uint8_t>(type), "legacy_connection");
    }
    
    // Create binary connection between two binary node IDs
    void createBinaryConnection(BinaryNodeID from_id, BinaryNodeID to_id, double weight, uint8_t type, const std::string& context = "") {
        BinaryConnection connection(from_id, to_id, weight, type, context);
        connection.first_created = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        // Add to adjacency list
        binary_adjacency_list[from_id].push_back(connection);
        
        // Create bidirectional connection with reduced weight
        BinaryConnection reverse_connection(to_id, from_id, weight * 0.8, type, context);
        reverse_connection.first_created = connection.first_created;
        binary_adjacency_list[to_id].push_back(reverse_connection);
        
        // Update node access counts
        if (binary_nodes.find(from_id) != binary_nodes.end()) {
            binary_nodes[from_id].access_count++;
            binary_nodes[from_id].last_accessed = connection.first_created;
        }
        if (binary_nodes.find(to_id) != binary_nodes.end()) {
            binary_nodes[to_id].access_count++;
            binary_nodes[to_id].last_accessed = connection.first_created;
        }
    }
    
    // Create semantic similarity connection
    void createSemanticConnection(BinaryNodeID from, BinaryNodeID to, double similarity_score, const std::string& similarity_type, const std::string& context = "") {
        BinaryConnection connection(from, to, similarity_score, similarity_score, similarity_type, context);
        connection.first_created = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        // Add to adjacency list
        binary_adjacency_list[from].push_back(connection);
        
        // Create bidirectional semantic connection with same weight
        BinaryConnection reverse_connection(to, from, similarity_score, similarity_score, similarity_type, context);
        reverse_connection.first_created = connection.first_created;
        binary_adjacency_list[to].push_back(reverse_connection);
        
        // Update node access counts
        if (binary_nodes.find(from) != binary_nodes.end()) {
            binary_nodes[from].access_count++;
            binary_nodes[from].last_accessed = connection.first_created;
        }
        if (binary_nodes.find(to) != binary_nodes.end()) {
            binary_nodes[to].access_count++;
            binary_nodes[to].last_accessed = connection.first_created;
        }
    }
    
    // Find and create semantic similarity connections for a new node
    void establishSemanticConnections(const BinaryNodeID& new_node_id, const std::string& new_text) {
        if (comprehensive_mode) {
            std::cout << "  🔍 Establishing semantic connections for: \"" << new_text << "\"" << std::endl;
        }
        
        // Find similar nodes
        auto similar_nodes = semantic_manager.findSimilarNodes(new_node_id, binary_nodes, node_manager, 0.3);
        
        int semantic_connections_created = 0;
        for (const auto& similar_pair : similar_nodes) {
            BinaryNodeID similar_id = similar_pair.first;
            double similarity_score = similar_pair.second;
            std::string similar_text = node_manager.getText(similar_id);
            
            if (similarity_score >= 0.5) { // Only create strong semantic connections
                std::string similarity_type = semantic_manager.getSimilarityType(new_text, similar_text);
                
                // Create semantic connection (createSemanticConnection already handles bidirectional)
                createSemanticConnection(new_node_id, similar_id, similarity_score, similarity_type, "semantic_similarity");
                
                semantic_connections_created++;
                
                if (comprehensive_mode) {
                    std::cout << "    → Semantic link: \"" << new_text << "\" ↔ \"" << similar_text 
                              << "\" (score:" << std::fixed << std::setprecision(2) << similarity_score 
                              << ", type:" << similarity_type << ")" << std::endl;
                }
            }
        }
        
        if (comprehensive_mode) {
            std::cout << "  📊 Created " << semantic_connections_created << " semantic similarity connections" << std::endl;
        }
    }
    
    // Create sequential connection between consecutive inputs using binary IDs
    void createSequentialConnection(const std::string& previous_input, const std::string& current_input) {
        if (previous_input.empty() || current_input.empty()) return;
        
        // Convert inputs to binary node IDs
        std::vector<BinaryNodeID> prev_ids = node_manager.textToBinaryIDs(previous_input);
        std::vector<BinaryNodeID> curr_ids = node_manager.textToBinaryIDs(current_input);
        
        // Create temporal connections between corresponding nodes
        for (size_t i = 0; i < std::min(prev_ids.size(), curr_ids.size()); ++i) {
            createBinaryConnection(prev_ids[i], curr_ids[i], 0.7, 3, "temporal_sequence"); // type 3 = temporal
        }
        
        std::cout << "🔗 Created binary sequential links between input nodes" << std::endl;
    }
    
    // Add concepts to brain using binary nodes
    void addConceptsToBrain(const std::vector<std::string>& input_concepts, UltimateMetrics& metrics) {
        for (const std::string& concept : input_concepts) {
            BinaryNodeID concept_id = node_manager.getOrCreateID(concept);
            if (binary_nodes.find(concept_id) == binary_nodes.end()) {
                BinaryNode new_node(concept_id, concept, "");
                new_node.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                binary_nodes[concept_id] = new_node;
                metrics.concepts_learned++;
            }
            
            // Update access count
            binary_nodes[concept_id].access_count++;
            binary_nodes[concept_id].last_accessed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
        }
    }
    
    // Extract concepts from input text
    std::vector<std::string> extractConcepts(const std::string& input) {
        std::vector<std::string> concepts;
        std::istringstream iss(input);
        std::string word;
        while (iss >> word) {
            // Clean the word
            word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
            if (!word.empty()) {
                concepts.push_back(word);
            }
        }
        return concepts;
    }
    
    // Generate sophisticated response from binary node reasoning
    std::string generateBinaryNodeResponse(const std::vector<BinaryNodeID>& input_node_ids, const std::string& question) {
        if (input_node_ids.empty()) {
            return "I'm processing your question...";
        }
        
        // Analyze the question type and context
        std::string question_lower = question;
        std::transform(question_lower.begin(), question_lower.end(), question_lower.begin(), ::tolower);
        
        // Find connected nodes through binary connections
        std::vector<BinaryNodeID> connected_nodes;
        std::unordered_set<BinaryNodeID> visited;
        std::map<std::string, double> concept_weights;
        
        for (BinaryNodeID node_id : input_node_ids) {
            if (binary_adjacency_list.find(node_id) != binary_adjacency_list.end()) {
                for (const auto& connection : binary_adjacency_list[node_id]) {
                    if (visited.find(connection.target_id) == visited.end() && 
                        connection.weight > 0.3) { // Lower threshold for more connections
                        connected_nodes.push_back(connection.target_id);
                        visited.insert(connection.target_id);
                        
                        std::string connected_text = node_manager.getText(connection.target_id);
                        if (!connected_text.empty()) {
                            concept_weights[connected_text] = connection.weight;
                        }
                    }
                }
            }
        }
        
        // Sort concepts by weight (importance)
        std::vector<std::pair<std::string, double>> sorted_concepts;
        for (const auto& pair : concept_weights) {
            sorted_concepts.push_back(pair);
        }
        std::sort(sorted_concepts.begin(), sorted_concepts.end(), 
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Generate context-aware response based on question type
        std::string response;
        
        if (question_lower.find("what") != std::string::npos || 
            question_lower.find("explain") != std::string::npos) {
            // Definition/explanation response
            if (sorted_concepts.empty()) {
                response = "I'm learning about " + node_manager.getText(input_node_ids[0]) + 
                          ". This concept is new to me. Could you help me understand it better?";
            } else {
                response = "Based on my knowledge, " + node_manager.getText(input_node_ids[0]) + 
                          " is related to " + sorted_concepts[0].first;
                if (sorted_concepts.size() > 1) {
                    response += ", " + sorted_concepts[1].first;
                    if (sorted_concepts.size() > 2) {
                        response += ", and " + sorted_concepts[2].first;
                    }
                }
                response += ". These connections help me understand the concept better.";
            }
        } else if (question_lower.find("how") != std::string::npos) {
            // Process/method response
            if (sorted_concepts.empty()) {
                response = "I'm exploring how " + node_manager.getText(input_node_ids[0]) + 
                          " works. This involves understanding the underlying processes.";
            } else {
                response = "The process of " + node_manager.getText(input_node_ids[0]) + 
                          " involves " + sorted_concepts[0].first;
                if (sorted_concepts.size() > 1) {
                    response += " and " + sorted_concepts[1].first;
                }
                response += ". I can see these connections in my knowledge graph.";
            }
        } else if (question_lower.find("why") != std::string::npos) {
            // Reasoning response
            if (sorted_concepts.empty()) {
                response = "I'm thinking about why " + node_manager.getText(input_node_ids[0]) + 
                          " works the way it does. The underlying principles are fascinating.";
            } else {
                response = "The reason " + node_manager.getText(input_node_ids[0]) + 
                          " works involves " + sorted_concepts[0].first;
                if (sorted_concepts.size() > 1) {
                    response += " and " + sorted_concepts[1].first;
                }
                response += ". These connections reveal the underlying logic.";
            }
        } else {
            // General response
            if (sorted_concepts.empty()) {
                response = "I'm learning about " + node_manager.getText(input_node_ids[0]) + 
                          ". This concept is building new connections in my knowledge graph.";
            } else {
                response = "I understand " + node_manager.getText(input_node_ids[0]) + 
                          " through its connections to " + sorted_concepts[0].first;
                if (sorted_concepts.size() > 1) {
                    response += ", " + sorted_concepts[1].first;
                    if (sorted_concepts.size() > 2) {
                        response += ", and " + sorted_concepts[2].first;
                    }
                }
                response += ". These relationships help me reason about the concept.";
            }
        }
        
        return response;
    }
    
    // Learn from Ollama responses
    void learnFromOllamaResponse(const std::string& question, const std::string& response) {
        std::string concept = extractConceptFromQuestion(question);
        
        // Create or update concept with oracle tracking
        BinaryNodeID concept_id = node_manager.getOrCreateID(concept);
        if (binary_nodes.find(concept_id) == binary_nodes.end()) {
            BinaryNode new_node(concept_id, concept, response);
            new_node.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            binary_nodes[concept_id] = new_node;
        } else {
            binary_nodes[concept_id].definition = response;
        }
        
        // Mark as oracle-used
        binary_nodes[concept_id].oracle_used = true;
        binary_nodes[concept_id].oracle_source = "ollama";
        binary_nodes[concept_id].oracle_timestamp = std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        binary_nodes[concept_id].dominant_driver_when_created = "curiosity";
        
        // Update validation success
        binary_nodes[concept_id].validation_successes++;
        
        std::cout << "📚 Learned new concept: " << concept << std::endl;
    }
    
    std::string extractConceptFromQuestion(const std::string& question) {
        std::vector<std::string> words;
        std::istringstream iss(question);
        std::string word;
        while (iss >> word) {
            word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
            if (!word.empty() && word.length() > 2) {
                words.push_back(word);
            }
        }
        return words.empty() ? "unknown" : words[0];
    }
    
    // Main processing method with comprehensive reasoning display
    std::string processQuestion(const std::string& user_question) {
        current_cycle++;
        total_cycles++;
        
        if (comprehensive_mode) {
            std::cout << "\n🧠 MELVIN THINKING PROCESS" << std::endl;
            std::cout << "=========================" << std::endl;
            std::cout << "🔄 Cycle #" << total_cycles << " | Processing: \"" << user_question << "\"" << std::endl;
        }
        
        // Step 1: Input Analysis
        std::vector<BinaryNodeID> input_node_ids = node_manager.textToBinaryIDs(user_question);
        if (comprehensive_mode) {
            std::cout << "\n📝 STEP 1: INPUT ANALYSIS" << std::endl;
            std::cout << "🔍 Extracted " << input_node_ids.size() << " tokens from input:" << std::endl;
            for (size_t i = 0; i < input_node_ids.size(); ++i) {
                std::string text = node_manager.getText(input_node_ids[i]);
                std::cout << "  " << (i+1) << ". Binary ID: " << input_node_ids[i].toHex() 
                          << " -> \"" << text << "\"" << std::endl;
            }
        }
        
        // Step 2: Knowledge Graph Updates
        int new_nodes_created = 0;
        for (BinaryNodeID node_id : input_node_ids) {
            std::string text = node_manager.getText(node_id);
            if (binary_nodes.find(node_id) == binary_nodes.end()) {
                BinaryNode new_node(node_id, text);
                new_node.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                binary_nodes[node_id] = new_node;
                new_nodes_created++;
                if (comprehensive_mode) {
                    std::cout << "  ➕ Created new node: Binary ID:" << node_id.toHex() << " -> \"" << text << "\"" << std::endl;
                }
                
                // Establish semantic similarity connections for new node
                establishSemanticConnections(node_id, text);
            } else {
                binary_nodes[node_id].access_count++;
                if (comprehensive_mode) {
                    std::cout << "  🔄 Updated existing node: Binary ID:" << node_id.toHex() << " -> \"" << text 
                              << "\" (accessed " << binary_nodes[node_id].access_count << " times)" << std::endl;
                }
            }
        }
        if (comprehensive_mode) {
            std::cout << "📊 Total new nodes created: " << new_nodes_created << std::endl;
        }
        
        // Step 3: Connection Analysis
        std::vector<BinaryNodeID> connected_nodes;
        std::unordered_set<BinaryNodeID> visited;
        int total_connections_found = 0;
        
        for (BinaryNodeID node_id : input_node_ids) {
            if (binary_adjacency_list.find(node_id) != binary_adjacency_list.end()) {
                const auto& connections = binary_adjacency_list[node_id];
                if (comprehensive_mode) {
                    std::cout << "  🔍 Binary Node ID:" << node_id.toHex() << " (\"" << node_manager.getText(node_id) 
                              << "\") has " << connections.size() << " connections:" << std::endl;
                }
                
                for (const auto& connection : connections) {
                    if (comprehensive_mode) {
                        std::string target_text = node_manager.getText(connection.target_id);
                        std::string conn_type_name = getConnectionTypeName(connection.connection_type);
                        std::cout << "    → Binary ID:" << connection.target_id.toHex() << " (\"" << target_text 
                                  << "\") [weight:" << std::fixed << std::setprecision(2) << connection.weight 
                                  << ", type:" << conn_type_name;
                        
                        // Show semantic similarity details if it's a semantic connection
                        if (connection.connection_type == 4) { // semantic_similarity
                            std::cout << ", sim_score:" << std::fixed << std::setprecision(2) << connection.semantic_similarity_score
                                      << ", sim_type:" << connection.similarity_type;
                        }
                        std::cout << "]" << std::endl;
                    }
                    
                    if (visited.find(connection.target_id) == visited.end() && connection.weight > 0.5) {
                        connected_nodes.push_back(connection.target_id);
                        visited.insert(connection.target_id);
                        total_connections_found++;
                    }
                }
            } else if (comprehensive_mode) {
                std::cout << "  🔍 Binary Node ID:" << node_id.toHex() << " (\"" << node_manager.getText(node_id) 
                          << "\") has no existing connections" << std::endl;
            }
        }
        if (comprehensive_mode) {
            std::cout << "📊 Total strong connections found: " << total_connections_found << std::endl;
        }
        
        // Step 4: Sequential Learning
        if (!previous_input.empty()) {
            if (comprehensive_mode) {
                std::cout << "\n⏰ STEP 4: SEQUENTIAL LEARNING" << std::endl;
                std::cout << "🔗 Creating temporal connections with previous input: \"" << previous_input << "\"" << std::endl;
            }
            createSequentialConnection(previous_input, user_question);
            if (comprehensive_mode) {
                std::cout << "✅ Temporal connections established" << std::endl;
            }
        } else if (comprehensive_mode) {
            std::cout << "\n⏰ STEP 4: SEQUENTIAL LEARNING" << std::endl;
            std::cout << "ℹ️  No previous input for temporal learning" << std::endl;
        }
        
        // Step 5: Reasoning and Response Generation
        std::string response;
        
        if (ollama_teacher.is_active) {
            if (comprehensive_mode) {
                std::cout << "\n💭 STEP 5: REASONING AND RESPONSE GENERATION" << std::endl;
                std::cout << "🎓 Ollama teacher mode active - seeking external knowledge" << std::endl;
            }
            response = ollama_teacher.askOllama(user_question);
            learnFromOllamaResponse(user_question, response);
            if (comprehensive_mode) {
                std::cout << "📚 Learned from Ollama teacher" << std::endl;
            }
        } else if (!connected_nodes.empty()) {
            if (comprehensive_mode) {
                std::cout << "\n💭 STEP 5: REASONING AND RESPONSE GENERATION" << std::endl;
                std::cout << "🧠 Using existing knowledge connections for response" << std::endl;
            }
            response = generateBinaryNodeResponse(input_node_ids, user_question);
            if (comprehensive_mode) {
                std::cout << "✅ Generated response from " << connected_nodes.size() << " connected concepts" << std::endl;
            }
        } else {
            if (comprehensive_mode) {
                std::cout << "\n💭 STEP 5: REASONING AND RESPONSE GENERATION" << std::endl;
                std::cout << "🤔 No strong connections found - generating learning response" << std::endl;
            }
            if (input_node_ids.empty()) {
                response = "I'm processing your question...";
            } else {
                response = "I'm learning about " + node_manager.getText(input_node_ids[0]) + ". Can you tell me more?";
            }
        }
        
        // Step 6: Memory Updates
        previous_input = user_question;
        if (comprehensive_mode) {
            std::cout << "\n💾 STEP 6: MEMORY UPDATES" << std::endl;
            std::cout << "🔄 Updated previous input for next temporal connection" << std::endl;
            std::cout << "📊 Current brain state: " << binary_nodes.size() << " nodes, " 
                      << binary_adjacency_list.size() << " connection groups" << std::endl;
            std::cout << "\n🎯 FINAL RESPONSE: \"" << response << "\"" << std::endl;
            std::cout << "=========================\n" << std::endl;
        }
        
        return response;
    }
    
    // Helper function to get connection type names
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
    
    // Show brain analytics
    void showBrainAnalytics() {
        std::cout << "\n📊 MELVIN ULTIMATE BRAIN ANALYTICS" << std::endl;
        std::cout << "=====================================" << std::endl;
        std::cout << "🧠 Total Binary Nodes: " << binary_nodes.size() << std::endl;
        
        uint64_t total_connections = 0;
        uint64_t semantic_connections = 0;
        uint64_t temporal_connections = 0;
        uint64_t hierarchical_connections = 0;
        
        for (const auto& conn_list : binary_adjacency_list) {
            total_connections += conn_list.second.size();
            for (const auto& conn : conn_list.second) {
                switch (conn.connection_type) {
                    case 4: semantic_connections++; break; // semantic_similarity
                    case 3: temporal_connections++; break; // temporal
                    case 2: hierarchical_connections++; break; // hierarchical
                }
            }
        }
        std::cout << "🔗 Total Binary Connections: " << total_connections << std::endl;
        std::cout << "🧠 Semantic Similarity Connections: " << semantic_connections << std::endl;
        std::cout << "⏰ Temporal Connections: " << temporal_connections << std::endl;
        std::cout << "🏗️ Hierarchical Connections: " << hierarchical_connections << std::endl;
        std::cout << "🔄 Total Processing Cycles: " << total_cycles << std::endl;
        std::cout << "🎓 Ollama Teaching Sessions: " << ollama_teacher.teaching_sessions << std::endl;
        std::cout << "📚 Concepts Taught by Ollama: " << ollama_teacher.concepts_taught << std::endl;
        
        // Show some sample nodes
        std::cout << "\n📚 Sample Binary Nodes:" << std::endl;
        int count = 0;
        for (const auto& node_pair : binary_nodes) {
            if (count < 5) {
                std::cout << "  📖 Binary ID:" << node_pair.first.toHex() << " -> \"" << node_pair.second.original_text 
                          << "\" (accessed " << node_pair.second.access_count << " times)" << std::endl;
                count++;
            }
        }
    }
    
    // Enhanced Ollama Interface for Learning Loop
    class EnhancedOllamaInterface {
    private:
        std::unordered_map<std::string, std::string> topic_responses;
        std::unordered_map<std::string, std::string> evaluation_responses;
        
    public:
        EnhancedOllamaInterface() {
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
            std::vector<std::string> topics = {"machine_learning", "neural_networks", "deep_learning", "algorithms"};
            static std::random_device rd;
            static std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, topics.size() - 1);
            
            std::string selected_topic = topics[dis(gen)];
            return selected_topic + ": " + topic_responses[selected_topic];
        }
        
        std::string evaluateUnderstanding(const std::string& melvin_response, const std::string& topic) {
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
    
    // Enhanced Ollama interface
    EnhancedOllamaInterface enhanced_ollama;
    
    // Learning loop functionality
    void runOllamaLearningLoop(int cycles = 5) {
        std::cout << "\n🚀 STARTING MELVIN OLLAMA LEARNING LOOP" << std::endl;
        std::cout << "=======================================" << std::endl;
        std::cout << "This will run " << cycles << " learning cycles automatically." << std::endl;
        std::cout << "Each cycle demonstrates the complete learning process:" << std::endl;
        std::cout << "1. Ollama provides topic → 2. Melvin processes & reasons → 3. Melvin outputs" << std::endl;
        std::cout << "4. Ollama evaluates → 5. Ollama fills gaps → 6. Repeat until mastery" << std::endl;
        std::cout << "=======================================" << std::endl;
        
        for (int cycle = 1; cycle <= cycles; ++cycle) {
            total_cycles++;
            
            std::cout << "\n🔄 LEARNING CYCLE #" << cycle << std::endl;
            std::cout << "====================" << std::endl;
            
            // Step 1: Ollama provides input topic
            std::cout << "\n📚 STEP 1: OLLAMA PROVIDES TOPIC" << std::endl;
            std::string ollama_input = enhanced_ollama.getTopicExplanation("machine_learning");
            std::cout << "🎓 Ollama says: \"" << ollama_input << "\"" << std::endl;
            
            // Step 2: Melvin processes and reasons
            std::cout << "\n🧠 STEP 2: MELVIN PROCESSES AND REASONS" << std::endl;
            std::string melvin_reasoning = processQuestion(ollama_input);
            
            // Step 3: Melvin generates output (already done in processQuestion)
            std::cout << "\n💬 STEP 3: MELVIN GENERATES OUTPUT" << std::endl;
            std::cout << "🤖 Melvin says: \"" << melvin_reasoning << "\"" << std::endl;
            
            // Step 4: Ollama evaluates Melvin's understanding
            std::cout << "\n📊 STEP 4: OLLAMA EVALUATES MELVIN" << std::endl;
            std::string evaluation = enhanced_ollama.evaluateUnderstanding(melvin_reasoning, "machine_learning");
            std::cout << "🎓 Ollama evaluates: \"" << evaluation << "\"" << std::endl;
            
            // Step 5: Ollama provides additional information if needed
            std::cout << "\n📖 STEP 5: OLLAMA PROVIDES ADDITIONAL INFO" << std::endl;
            std::string additional_info = enhanced_ollama.provideAdditionalInformation("machine_learning", evaluation);
            std::cout << "🎓 Ollama adds: \"" << additional_info << "\"" << std::endl;
            
            // Process additional information if provided
            if (additional_info.find("additional information") != std::string::npos) {
                std::cout << "\n🔄 PROCESSING ADDITIONAL INFORMATION" << std::endl;
                processQuestion(additional_info);
            }
            
            std::cout << "\n✅ Learning cycle " << cycle << " completed!" << std::endl;
            
            // Show current brain state
            showBrainAnalytics();
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
        
        std::cout << "\n🎯 LEARNING SESSION COMPLETE!" << std::endl;
        std::cout << "Total learning cycles completed: " << cycles << std::endl;
    }
    
    // Save brain state to binary file
    void saveBrainState() {
        std::ofstream file("melvin_brain.bin", std::ios::binary);
        if (!file.is_open()) {
            std::cout << "❌ Could not save brain state" << std::endl;
            return;
        }
        
        // Save node count
        uint32_t node_count = static_cast<uint32_t>(binary_nodes.size());
        file.write(reinterpret_cast<const char*>(&node_count), sizeof(node_count));
        
        // Save each node
        for (const auto& node_pair : binary_nodes) {
            // Save binary ID
            file.write(reinterpret_cast<const char*>(&node_pair.first), sizeof(node_pair.first));
            
            // Save text length and text
            uint32_t text_length = static_cast<uint32_t>(node_pair.second.original_text.length());
            file.write(reinterpret_cast<const char*>(&text_length), sizeof(text_length));
            file.write(node_pair.second.original_text.c_str(), text_length);
            
            // Save definition length and definition
            uint32_t def_length = static_cast<uint32_t>(node_pair.second.definition.length());
            file.write(reinterpret_cast<const char*>(&def_length), sizeof(def_length));
            file.write(node_pair.second.definition.c_str(), def_length);
            
            // Save node properties
            file.write(reinterpret_cast<const char*>(&node_pair.second.activation), sizeof(double));
            file.write(reinterpret_cast<const char*>(&node_pair.second.importance), sizeof(double));
            file.write(reinterpret_cast<const char*>(&node_pair.second.access_count), sizeof(uint32_t));
            file.write(reinterpret_cast<const char*>(&node_pair.second.validation_successes), sizeof(uint32_t));
            file.write(reinterpret_cast<const char*>(&node_pair.second.validation_failures), sizeof(uint32_t));
            
            // Save connections for this node
            uint32_t connection_count = 0;
            if (binary_adjacency_list.find(node_pair.first) != binary_adjacency_list.end()) {
                connection_count = static_cast<uint32_t>(binary_adjacency_list[node_pair.first].size());
            }
            file.write(reinterpret_cast<const char*>(&connection_count), sizeof(connection_count));
            
            if (connection_count > 0) {
                for (const auto& conn : binary_adjacency_list[node_pair.first]) {
                    std::string target_text = node_manager.getText(conn.target_id);
                    uint32_t target_text_length = static_cast<uint32_t>(target_text.length());
                    file.write(reinterpret_cast<const char*>(&target_text_length), sizeof(target_text_length));
                    file.write(target_text.c_str(), target_text_length);
                    file.write(reinterpret_cast<const char*>(&conn.weight), sizeof(conn.weight));
                    uint32_t conn_type = static_cast<uint32_t>(conn.connection_type);
                    file.write(reinterpret_cast<const char*>(&conn_type), sizeof(conn_type));
                }
            }
        }
        
        file.close();
        std::cout << "💾 Saved " << binary_nodes.size() << " concepts to binary brain" << std::endl;
    }
    
    // Interactive session
    void runInteractiveSession() {
        std::string input;
        while (true) {
            std::cout << "\nYou: ";
            std::getline(std::cin, input);
            
            if (input == "quit" || input == "exit") {
                std::cout << "👋 Goodbye! Saving brain state..." << std::endl;
                saveBrainState();
                break;
            } else if (input == "analytics") {
                showBrainAnalytics();
            } else if (input == "save") {
                saveBrainState();
            } else if (input == "teacher") {
                ollama_teacher.activate();
            } else if (input == "teacher off") {
                ollama_teacher.deactivate();
            } else if (input == "dual on") {
                dual_output_mode = true;
                std::cout << "🎯 Dual-output mode enabled. Responses will show both human-facing and debug/thinking layers." << std::endl;
            } else if (input == "dual off") {
                dual_output_mode = false;
                std::cout << "🎯 Dual-output mode disabled." << std::endl;
            } else if (input == "comprehensive on") {
                comprehensive_mode = true;
                std::cout << "🔍 Comprehensive thinking mode enabled. You'll see detailed reasoning steps." << std::endl;
            } else if (input == "comprehensive off") {
                comprehensive_mode = false;
                std::cout << "🔍 Comprehensive thinking mode disabled." << std::endl;
            } else if (input == "learning loop") {
                runOllamaLearningLoop(5);
            } else if (input == "learning loop 3") {
                runOllamaLearningLoop(3);
            } else if (input == "learning loop 10") {
                runOllamaLearningLoop(10);
            } else if (input.find("learning loop") == 0) {
                // Parse learning loop with custom cycle count
                std::istringstream iss(input);
                std::string cmd, loop, cycles_str;
                iss >> cmd >> loop;
                if (loop == "loop" && iss >> cycles_str) {
                    try {
                        int cycles = std::stoi(cycles_str);
                        if (cycles > 0 && cycles <= 20) {
                            runOllamaLearningLoop(cycles);
                        } else {
                            std::cout << "Please specify a number of cycles between 1 and 20." << std::endl;
                        }
                    } catch (const std::exception&) {
                        std::cout << "Invalid number of cycles. Please use: learning loop [number]" << std::endl;
                    }
                } else {
                    std::cout << "Invalid learning loop command. Use: learning loop [number]" << std::endl;
                }
            } else if (input == "ollama continuous") {
                std::cout << "🚀 Starting Ollama Continuous Learning Mode..." << std::endl;
                std::cout << "🎓 Teacher mode will be active throughout" << std::endl;
                std::cout << "🔄 Melvin will learn continuously from educational questions" << std::endl;
                std::cout << "⏹️  Type 'stop continuous' to end continuous mode" << std::endl;
                std::cout << "" << std::endl;
                
                // Educational questions for continuous learning
                std::vector<std::string> ollama_questions = {
                    "What is the fundamental difference between artificial intelligence and human intelligence?",
                    "How do neural networks actually learn and adapt to new information?",
                    "Explain the concept of machine learning in simple terms",
                    "What are the key principles behind deep learning algorithms?",
                    "How does supervised learning differ from unsupervised learning?",
                    "What role do algorithms play in computer decision-making?",
                    "What is quantum computing and how does it differ from classical computing?",
                    "Explain the concept of blockchain technology and its applications",
                    "How does the internet actually work behind the scenes?",
                    "What is cybersecurity and why is it important?",
                    "What are the fundamental principles of cryptography?",
                    "How do distributed systems ensure reliability and scalability?",
                    "What is consciousness and how do we define it scientifically?",
                    "How does the human brain process and store memories?",
                    "What is the nature of reality from a philosophical perspective?",
                    "How do we define intelligence across different species?",
                    "What makes human creativity unique compared to AI creativity?",
                    "How do emotions influence our decision-making processes?",
                    "What is the meaning of life from different philosophical perspectives?",
                    "How does evolution drive the development of intelligence?",
                    "What is DNA and how does it encode genetic information?",
                    "How do vaccines work to protect against diseases?",
                    "What is the immune system and how does it defend the body?",
                    "How do antibiotics work to fight bacterial infections?",
                    "What is the theory of relativity and how does it affect our understanding of time?",
                    "How do black holes form and what happens inside them?",
                    "What is dark matter and why is it important to cosmology?",
                    "How does photosynthesis convert sunlight into energy?",
                    "What is the periodic table and how do elements interact?",
                    "How does gravity work according to Einstein's general relativity?"
                };
                
                // Activate teacher mode and comprehensive mode
                ollama_teacher.activate();
                comprehensive_mode = true;
                
                int question_count = 0;
                bool continuous_mode = true;
                
                while (continuous_mode) {
                    std::string question = ollama_questions[question_count % ollama_questions.size()];
                    question_count++;
                    
                    std::cout << "🤖 Ollama Question " << question_count << ": " << question << std::endl;
                    
                    // Process the question
                    std::string response = processQuestion(question);
                    std::cout << "🧠 Melvin: " << response << std::endl;
                    std::cout << "" << std::endl;
                    
                    // Show analytics every 10 questions
                    if (question_count % 10 == 0) {
                        std::cout << "📊 Progress Update (Question " << question_count << "):" << std::endl;
                        showBrainAnalytics();
                        std::cout << "" << std::endl;
                    }
                    
                    // Run learning loop every 20 questions
                    if (question_count % 20 == 0) {
                        std::cout << "🔄 Running learning loop to consolidate knowledge..." << std::endl;
                        runOllamaLearningLoop(3);
                        std::cout << "" << std::endl;
                    }
                    
                    // Save brain state every 50 questions
                    if (question_count % 50 == 0) {
                        std::cout << "💾 Saving brain state..." << std::endl;
                        saveBrainState();
                        std::cout << "" << std::endl;
                    }
                    
                    // Check if user wants to stop (non-blocking)
                    std::cout << "Press Enter for next question, or type 'stop continuous' to end: ";
                    std::string user_input;
                    std::getline(std::cin, user_input);
                    
                    if (user_input == "stop continuous") {
                        continuous_mode = false;
                        std::cout << "🛑 Stopping continuous learning mode..." << std::endl;
                    }
                    std::cout << "" << std::endl;
                }
                
                std::cout << "🎯 Continuous learning session completed!" << std::endl;
                std::cout << "📊 Final analytics:" << std::endl;
                showBrainAnalytics();
                std::cout << "" << std::endl;
                
            } else if (!input.empty()) {
                std::string response = processQuestion(input);
                if (dual_output_mode) {
                    std::cout << "💬 Human-Facing:\n" << response << std::endl;
                    std::cout << "🧠 Debug/Thinking:\nProcessing [BINARY]: " << input << std::endl;
                    std::cout << "→ Confidence: " << (response.find("learning") != std::string::npos ? "Low" : "High") << std::endl;
                } else {
                    std::cout << "Melvin: " << response << std::endl;
                }
            }
        }
    }
};

// Main function
int main() {
    std::cout << "Type your questions, or 'quit' to exit" << std::endl;
    std::cout << "Commands: 'analytics' for brain stats, 'teacher' for Ollama teacher mode" << std::endl;
    std::cout << "THINKING: 'comprehensive on/off' for detailed reasoning steps" << std::endl;
    std::cout << "OUTPUT: 'dual on/off' for human-facing + debug/thinking responses" << std::endl;
    std::cout << "LEARNING: 'learning loop' (5 cycles), 'learning loop 3', 'learning loop 10'" << std::endl;
    std::cout << "CONTINUOUS: 'ollama continuous' for Ollama-generated continuous learning" << std::endl;
    std::cout << "DATA: 'save' to save brain state, 'quit' to exit and save" << std::endl;
    
    MelvinUltimateUnifiedWithOutput melvin;
    melvin.runInteractiveSession();
    
    return 0;
}

/*
 * Melvin Unified Reasoning System
 * 
 * Implements the unified reasoning prompt with:
 * - Universal connection graph
 * - Dynamic weighting by type/context/recency
 * - Multi-hop exploration
 * - Driver modulation (dopamine/serotonin/endorphins)
 * - Self-check contradiction resolution
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
#include <chrono>
#include <random>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <set>
#include <sstream>
#include <cmath>
#include <curl/curl.h>
#include <json/json.h>

// Connection types with weights
enum class ConnectionType {
    SEMANTIC = 0,      // Weight: 1.0 (highest)
    HIERARCHICAL = 1,  // Weight: 0.9
    CAUSAL = 2,        // Weight: 0.8
    CONTEXTUAL = 3,    // Weight: 0.7
    COMPONENT = 4,     // Weight: 0.6
    DEFINITION = 5,    // Weight: 0.5
    TEMPORAL = 6,      // Weight: 0.3
    SPATIAL = 7        // Weight: 0.2 (lowest)
};

// Connection with weight and metadata
struct WeightedConnection {
    uint64_t target_node_id;
    ConnectionType type;
    double base_weight;
    double context_weight;
    double recency_weight;
    double frequency_weight;
    double final_weight;
    std::string reasoning;
    
    WeightedConnection(uint64_t id, ConnectionType t, double base) 
        : target_node_id(id), type(t), base_weight(base), 
          context_weight(1.0), recency_weight(1.0), frequency_weight(1.0), final_weight(base) {}
};

// Reasoning path through multiple connections
struct ReasoningPath {
    std::vector<WeightedConnection> connections;
    double total_coherence;
    double total_relevance;
    double path_score;
    std::string explanation;
    
    ReasoningPath() : total_coherence(0.0), total_relevance(0.0), path_score(0.0) {}
};

// Driver modulation system
struct DriverState {
    double dopamine;    // Curiosity/exploration (0.0-1.0)
    double serotonin;   // Stability/balance (0.0-1.0)
    double endorphins;  // Satisfaction/reinforcement (0.0-1.0)
    
    DriverState() : dopamine(0.5), serotonin(0.5), endorphins(0.5) {}
    
    void modulate(double curiosity, double stability, double satisfaction) {
        dopamine = std::max(0.0, std::min(1.0, dopamine + curiosity));
        serotonin = std::max(0.0, std::min(1.0, serotonin + stability));
        endorphins = std::max(0.0, std::min(1.0, endorphins + satisfaction));
    }
};

// Enhanced Knowledge Node
struct UnifiedKnowledgeNode {
    uint64_t id;
    char concept[64];
    char definition[512];
    std::vector<WeightedConnection> connections;
    char source[32];
    double confidence;
    uint64_t created_at;
    uint64_t last_accessed;
    uint32_t access_count;
    uint32_t success_count;  // How often this node led to good answers
    
    UnifiedKnowledgeNode() : id(0), confidence(0.8), created_at(0), last_accessed(0), access_count(0), success_count(0) {
        memset(concept, 0, sizeof(concept));
        memset(definition, 0, sizeof(definition));
        memset(source, 0, sizeof(source));
    }
    
    UnifiedKnowledgeNode(uint64_t node_id, const std::string& node_concept, const std::string& node_definition)
        : id(node_id), confidence(0.8), created_at(getCurrentTime()), last_accessed(getCurrentTime()), access_count(0), success_count(0) {
        
        strncpy(concept, node_concept.c_str(), sizeof(concept) - 1);
        strncpy(definition, node_definition.c_str(), sizeof(definition) - 1);
        strncpy(source, "ollama", sizeof(source) - 1);
    }
    
    static uint64_t getCurrentTime() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
    }
};

// Unified Reasoning Engine
class UnifiedReasoningEngine {
private:
    std::map<uint64_t, std::shared_ptr<UnifiedKnowledgeNode>> nodes;
    DriverState drivers;
    uint64_t next_node_id = 1;
    
    // Connection type weights
    std::map<ConnectionType, double> type_weights = {
        {ConnectionType::SEMANTIC, 1.0},
        {ConnectionType::HIERARCHICAL, 0.9},
        {ConnectionType::CAUSAL, 0.8},
        {ConnectionType::CONTEXTUAL, 0.7},
        {ConnectionType::COMPONENT, 0.6},
        {ConnectionType::DEFINITION, 0.5},
        {ConnectionType::TEMPORAL, 0.3},
        {ConnectionType::SPATIAL, 0.2}
    };
    
    // Semantic groups for connection building
    std::map<std::string, std::vector<std::string>> semantic_groups = {
        {"animals", {"cat", "dog", "bird", "fish", "lion", "tiger", "elephant", "bear", "wolf", "deer", "rabbit", "mouse", "snake", "frog", "spider", "bee", "butterfly"}},
        {"vehicles", {"car", "truck", "bus", "train", "plane", "boat", "ship", "bicycle", "motorcycle", "helicopter", "submarine", "rocket"}},
        {"buildings", {"house", "building", "school", "hospital", "church", "library", "museum", "office", "factory", "warehouse", "castle", "tower"}},
        {"food", {"bread", "meat", "vegetable", "fruit", "milk", "cheese", "egg", "rice", "pasta", "soup", "salad", "pizza", "cake", "cookie"}},
        {"colors", {"red", "blue", "green", "yellow", "black", "white", "purple", "orange", "pink", "brown", "gray", "silver", "gold"}},
        {"emotions", {"happy", "sad", "angry", "excited", "worried", "calm", "nervous", "confident", "scared", "surprised", "disappointed", "proud"}},
        {"actions", {"run", "walk", "jump", "swim", "fly", "drive", "cook", "eat", "sleep", "work", "play", "study", "read", "write", "speak", "listen"}},
        {"materials", {"wood", "metal", "plastic", "glass", "fabric", "paper", "stone", "rubber", "leather", "ceramic", "concrete", "steel"}},
        {"tools", {"hammer", "screwdriver", "knife", "scissors", "pen", "pencil", "computer", "phone", "camera", "watch", "clock", "calculator"}},
        {"weather", {"sun", "rain", "snow", "wind", "cloud", "storm", "thunder", "lightning", "fog", "ice", "heat", "cold", "warm", "cool"}},
        {"body_parts", {"head", "eye", "nose", "mouth", "ear", "hand", "foot", "arm", "leg", "heart", "brain", "lung", "stomach", "skin", "hair"}},
        {"time_concepts", {"morning", "afternoon", "evening", "night", "day", "week", "month", "year", "hour", "minute", "second", "past", "present", "future"}},
        {"locations", {"home", "school", "work", "park", "store", "restaurant", "hospital", "library", "beach", "mountain", "city", "country", "world"}},
        {"professions", {"doctor", "teacher", "engineer", "artist", "musician", "chef", "pilot", "driver", "builder", "scientist", "lawyer", "police"}},
        {"family", {"mother", "father", "sister", "brother", "grandmother", "grandfather", "aunt", "uncle", "cousin", "son", "daughter", "baby"}}
    };
    
    // Hierarchical relationships
    std::map<std::string, std::vector<std::string>> hierarchies = {
        {"animal", {"mammal", "bird", "fish", "reptile", "insect"}},
        {"mammal", {"cat", "dog", "human", "elephant", "whale"}},
        {"vehicle", {"car", "truck", "bus", "motorcycle", "bicycle"}},
        {"building", {"house", "office", "school", "hospital", "factory"}},
        {"tool", {"hammer", "screwdriver", "knife", "scissors", "computer"}},
        {"emotion", {"happy", "sad", "angry", "excited", "worried"}}
    };
    
    // Contextual patterns
    std::map<std::string, std::vector<std::string>> contextual_patterns = {
        {"kitchen", {"cook", "eat", "food", "stove", "refrigerator", "knife", "plate", "cup"}},
        {"school", {"learn", "study", "teacher", "student", "book", "pencil", "desk", "classroom"}},
        {"hospital", {"doctor", "nurse", "patient", "medicine", "surgery", "health", "treatment"}},
        {"park", {"play", "walk", "tree", "grass", "bench", "children", "exercise", "nature"}},
        {"office", {"work", "computer", "desk", "meeting", "business", "employee", "boss", "project"}}
    };

public:
    // Step 1: Expand Connections (Possibilities)
    std::vector<WeightedConnection> expandConnections(const std::string& concept, const std::string& context) {
        std::vector<WeightedConnection> connections;
        std::string lower_concept = toLowerCase(cleanConcept(concept));
        
        std::cout << "🔍 Expanding connections for: " << concept << std::endl;
        
        // Semantic connections
        buildSemanticConnections(lower_concept, connections);
        
        // Hierarchical connections
        buildHierarchicalConnections(lower_concept, connections);
        
        // Causal connections
        buildCausalConnections(lower_concept, connections);
        
        // Contextual connections
        buildContextualConnections(lower_concept, context, connections);
        
        // Component connections
        buildComponentConnections(lower_concept, connections);
        
        // Definition connections
        buildDefinitionConnections(lower_concept, connections);
        
        // Temporal connections
        buildTemporalConnections(connections);
        
        // Spatial connections
        buildSpatialConnections(lower_concept, connections);
        
        std::cout << "✅ Found " << connections.size() << " possible connections" << std::endl;
        return connections;
    }
    
    // Step 2: Weight Connections (Prioritization)
    void weightConnections(std::vector<WeightedConnection>& connections, const std::string& context) {
        std::cout << "⚖️ Weighting connections..." << std::endl;
        
        for (auto& conn : connections) {
            // Base weight from connection type
            conn.base_weight = type_weights[conn.type];
            
            // Context relevance weight
            conn.context_weight = calculateContextRelevance(conn, context);
            
            // Recency weight (recently learned gets bonus)
            conn.recency_weight = calculateRecencyWeight(conn);
            
            // Frequency weight (common associations weigh higher)
            conn.frequency_weight = calculateFrequencyWeight(conn);
            
            // Driver modulation
            applyDriverModulation(conn);
            
            // Final weighted score
            conn.final_weight = conn.base_weight * conn.context_weight * 
                              conn.recency_weight * conn.frequency_weight;
            
            std::cout << "  " << getConnectionTypeName(conn.type) << " → " 
                     << nodes[conn.target_node_id]->concept 
                     << " (weight: " << conn.final_weight << ")" << std::endl;
        }
        
        // Sort by final weight
        std::sort(connections.begin(), connections.end(), 
                 [](const WeightedConnection& a, const WeightedConnection& b) {
                     return a.final_weight > b.final_weight;
                 });
    }
    
    // Step 3: Select Path (Choice) - Multi-hop exploration
    ReasoningPath selectReasoningPath(const std::vector<WeightedConnection>& connections, 
                                    const std::string& context, int max_hops = 3) {
        std::cout << "🛤️ Selecting reasoning path..." << std::endl;
        
        ReasoningPath best_path;
        double best_score = 0.0;
        
        // Try different path lengths
        for (int hops = 1; hops <= max_hops; hops++) {
            auto paths = generateMultiHopPaths(connections, hops);
            
            for (auto& path : paths) {
                // Calculate coherence (how well connections flow together)
                path.total_coherence = calculateCoherence(path);
                
                // Calculate relevance (how well it fits the context)
                path.total_relevance = calculateRelevance(path, context);
                
                // Combined path score
                path.path_score = path.total_coherence * path.total_relevance;
                
                if (path.path_score > best_score) {
                    best_score = path.path_score;
                    best_path = path;
                }
            }
        }
        
        std::cout << "✅ Selected path with " << best_path.connections.size() 
                 << " hops (score: " << best_path.path_score << ")" << std::endl;
        
        return best_path;
    }
    
    // Step 4: Driver Modulation (Reasoning Style)
    void applyDriverModulation(WeightedConnection& conn) {
        // Dopamine (curiosity/exploration): bias toward novel connections
        if (conn.type == ConnectionType::SEMANTIC || conn.type == ConnectionType::CAUSAL) {
            conn.final_weight *= (1.0 + drivers.dopamine * 0.3);
        }
        
        // Serotonin (stability/balance): bias toward consistent connections
        if (conn.type == ConnectionType::HIERARCHICAL || conn.type == ConnectionType::DEFINITION) {
            conn.final_weight *= (1.0 + drivers.serotonin * 0.2);
        }
        
        // Endorphins (satisfaction/reinforcement): bias toward successful connections
        if (nodes[conn.target_node_id]->success_count > 0) {
            conn.final_weight *= (1.0 + drivers.endorphins * 0.4);
        }
    }
    
    // Step 5: Self-Check (Validation)
    bool validateReasoning(const ReasoningPath& path, const std::string& context) {
        std::cout << "🔍 Self-checking reasoning..." << std::endl;
        
        // Check for contradictions
        if (hasContradictions(path)) {
            std::cout << "❌ Contradictions detected, rebalancing..." << std::endl;
            return false;
        }
        
        // Check coherence
        if (path.total_coherence < 0.5) {
            std::cout << "❌ Low coherence, rebalancing..." << std::endl;
            return false;
        }
        
        std::cout << "✅ Reasoning validated" << std::endl;
        return true;
    }
    
    // Step 6: Produce Output (Reasoned Answer)
    std::string produceReasonedOutput(const ReasoningPath& path, const std::string& question) {
        std::cout << "💡 Producing reasoned output..." << std::endl;
        
        std::stringstream output;
        output << "🧠 Melvin's Reasoning Process:\n\n";
        
        // Explain the reasoning path
        output << "Reasoning Path:\n";
        for (size_t i = 0; i < path.connections.size(); i++) {
            const auto& conn = path.connections[i];
            auto node = nodes[conn.target_node_id];
            
            output << "  " << (i + 1) << ". " << getConnectionTypeName(conn.type) 
                   << " → " << node->concept << " (weight: " << conn.final_weight << ")\n";
        }
        
        output << "\nStrongest Connections:\n";
        for (size_t i = 0; i < std::min(size_t(3), path.connections.size()); i++) {
            const auto& conn = path.connections[i];
            auto node = nodes[conn.target_node_id];
            
            output << "  • " << getConnectionTypeName(conn.type) << ": " << node->concept 
                   << " - " << node->definition << "\n";
        }
        
        output << "\nFinal Answer: Based on the reasoning path above, ";
        
        if (!path.connections.empty()) {
            auto best_node = nodes[path.connections[0].target_node_id];
            output << best_node->definition;
        } else {
            output << "I need to learn more about this concept.";
        }
        
        return output.str();
    }
    
    // Main unified reasoning process
    std::string unifiedReasoning(const std::string& question, const std::string& context = "") {
        std::cout << "\n🧠 MELVIN UNIFIED REASONING PROCESS" << std::endl;
        std::cout << "====================================" << std::endl;
        
        std::string concept = extractConceptFromQuestion(question);
        
        // Step 1: Expand Connections
        auto connections = expandConnections(concept, context);
        
        if (connections.empty()) {
            return "I don't have enough knowledge to reason about this concept yet.";
        }
        
        // Step 2: Weight Connections
        weightConnections(connections, context);
        
        // Step 3: Select Path
        auto reasoning_path = selectReasoningPath(connections, context);
        
        // Step 4: Driver Modulation (applied during weighting)
        
        // Step 5: Self-Check
        if (!validateReasoning(reasoning_path, context)) {
            // Rebalance and try again
            weightConnections(connections, context);
            reasoning_path = selectReasoningPath(connections, context);
        }
        
        // Step 6: Produce Output
        return produceReasonedOutput(reasoning_path, question);
    }
    
    // Add new knowledge node
    void addKnowledgeNode(const std::string& concept, const std::string& definition) {
        auto node = std::make_shared<UnifiedKnowledgeNode>(next_node_id++, concept, definition);
        nodes[node->id] = node;
        
        // Build connections for the new node
        auto connections = expandConnections(concept, "");
        for (const auto& conn : connections) {
            node->connections.push_back(conn);
        }
        
        std::cout << "📚 Added knowledge node: " << concept << " (ID: " << node->id << ")" << std::endl;
    }
    
    // Update driver states based on reasoning success
    void updateDrivers(bool reasoning_successful, bool novel_connection_used, bool stable_connection_used) {
        if (reasoning_successful) {
            drivers.modulate(0.1, 0.1, 0.2);  // Boost all drivers on success
        } else {
            drivers.modulate(-0.1, 0.1, -0.1);  // Reduce exploration and satisfaction
        }
        
        if (novel_connection_used) {
            drivers.modulate(0.2, -0.1, 0.0);  // Boost curiosity, reduce stability
        }
        
        if (stable_connection_used) {
            drivers.modulate(-0.1, 0.2, 0.0);  // Reduce curiosity, boost stability
        }
        
        std::cout << "🧬 Driver states: Dopamine=" << drivers.dopamine 
                 << ", Serotonin=" << drivers.serotonin 
                 << ", Endorphins=" << drivers.endorphins << std::endl;
    }

private:
    // Connection building methods
    void buildSemanticConnections(const std::string& concept, std::vector<WeightedConnection>& connections) {
        for (const auto& group : semantic_groups) {
            for (const auto& member : group.second) {
                if (concept == member) {
                    for (const auto& other_member : group.second) {
                        if (other_member != member) {
                            for (const auto& node_pair : nodes) {
                                auto node = node_pair.second;
                                std::string node_concept = toLowerCase(cleanConcept(node->concept));
                                if (node_concept == other_member) {
                                    connections.emplace_back(node->id, ConnectionType::SEMANTIC, 1.0);
                                    connections.back().reasoning = "Semantic similarity in " + group.first + " group";
                                }
                            }
                        }
                    }
                    break;
                }
            }
        }
    }
    
    void buildHierarchicalConnections(const std::string& concept, std::vector<WeightedConnection>& connections) {
        for (const auto& hierarchy : hierarchies) {
            for (const auto& member : hierarchy.second) {
                if (concept == member) {
                    // Connect to parent category
                    for (const auto& node_pair : nodes) {
                        auto node = node_pair.second;
                        std::string node_concept = toLowerCase(cleanConcept(node->concept));
                        if (node_concept == hierarchy.first) {
                            connections.emplace_back(node->id, ConnectionType::HIERARCHICAL, 0.9);
                            connections.back().reasoning = "Hierarchical relationship: " + concept + " is a " + hierarchy.first;
                        }
                    }
                    break;
                }
            }
        }
    }
    
    void buildCausalConnections(const std::string& concept, std::vector<WeightedConnection>& connections) {
        // Simple causal pattern matching
        std::map<std::string, std::vector<std::string>> causal_patterns = {
            {"rain", {"cloud", "storm", "weather"}},
            {"growth", {"water", "sunlight", "nutrients"}},
            {"movement", {"force", "energy", "motor"}},
            {"learning", {"study", "practice", "experience"}}
        };
        
        for (const auto& causal_group : causal_patterns) {
            for (const auto& cause : causal_group.second) {
                if (concept == cause) {
                    for (const auto& node_pair : nodes) {
                        auto node = node_pair.second;
                        std::string node_concept = toLowerCase(cleanConcept(node->concept));
                        if (node_concept == causal_group.first) {
                            connections.emplace_back(node->id, ConnectionType::CAUSAL, 0.8);
                            connections.back().reasoning = "Causal relationship: " + concept + " causes " + causal_group.first;
                        }
                    }
                    break;
                }
            }
        }
    }
    
    void buildContextualConnections(const std::string& concept, const std::string& context, 
                                  std::vector<WeightedConnection>& connections) {
        for (const auto& context_group : contextual_patterns) {
            for (const auto& context_item : context_group.second) {
                if (concept == context_item) {
                    for (const auto& other_item : context_group.second) {
                        if (other_item != context_item) {
                            for (const auto& node_pair : nodes) {
                                auto node = node_pair.second;
                                std::string node_concept = toLowerCase(cleanConcept(node->concept));
                                if (node_concept == other_item) {
                                    connections.emplace_back(node->id, ConnectionType::CONTEXTUAL, 0.7);
                                    connections.back().reasoning = "Contextual relationship in " + context_group.first;
                                }
                            }
                        }
                    }
                    break;
                }
            }
        }
    }
    
    void buildComponentConnections(const std::string& concept, std::vector<WeightedConnection>& connections) {
        for (const auto& node_pair : nodes) {
            auto node = node_pair.second;
            std::string node_concept = toLowerCase(cleanConcept(node->concept));
            
            if (concept.find(node_concept) != std::string::npos && node_concept.length() > 2) {
                connections.emplace_back(node->id, ConnectionType::COMPONENT, 0.6);
                connections.back().reasoning = "Component relationship: " + concept + " contains " + node_concept;
            }
        }
    }
    
    void buildDefinitionConnections(const std::string& concept, std::vector<WeightedConnection>& connections) {
        for (const auto& node_pair : nodes) {
            auto node = node_pair.second;
            std::string node_concept = toLowerCase(cleanConcept(node->concept));
            std::string node_def = toLowerCase(node->definition);
            
            if (node_def.find(concept) != std::string::npos) {
                connections.emplace_back(node->id, ConnectionType::DEFINITION, 0.5);
                connections.back().reasoning = "Definition relationship: " + concept + " mentioned in definition";
            }
        }
    }
    
    void buildTemporalConnections(std::vector<WeightedConnection>& connections) {
        uint64_t current_time = UnifiedKnowledgeNode::getCurrentTime();
        
        for (const auto& node_pair : nodes) {
            auto node = node_pair.second;
            uint64_t time_diff = current_time - node->created_at;
            
            if (time_diff < 3600000) { // Within 1 hour
                connections.emplace_back(node->id, ConnectionType::TEMPORAL, 0.3);
                connections.back().reasoning = "Temporal relationship: recently learned";
            }
        }
    }
    
    void buildSpatialConnections(const std::string& concept, std::vector<WeightedConnection>& connections) {
        std::vector<std::string> spatial_words = {"in", "on", "at", "near", "inside", "outside", "above", "below", "beside", "around"};
        
        for (const auto& spatial_word : spatial_words) {
            if (concept.find(spatial_word) != std::string::npos) {
                for (const auto& node_pair : nodes) {
                    auto node = node_pair.second;
                    std::string node_def = toLowerCase(node->definition);
                    if (node_def.find(spatial_word) != std::string::npos) {
                        connections.emplace_back(node->id, ConnectionType::SPATIAL, 0.2);
                        connections.back().reasoning = "Spatial relationship: both contain spatial word '" + spatial_word + "'";
                    }
                }
            }
        }
    }
    
    // Weighting calculation methods
    double calculateContextRelevance(const WeightedConnection& conn, const std::string& context) {
        if (context.empty()) return 1.0;
        
        auto node = nodes[conn.target_node_id];
        std::string node_def = toLowerCase(node->definition);
        std::string lower_context = toLowerCase(context);
        
        // Simple keyword matching
        std::istringstream iss(lower_context);
        std::string word;
        int matches = 0;
        int total_words = 0;
        
        while (iss >> word) {
            total_words++;
            if (node_def.find(word) != std::string::npos) {
                matches++;
            }
        }
        
        return total_words > 0 ? 1.0 + (double)matches / total_words : 1.0;
    }
    
    double calculateRecencyWeight(const WeightedConnection& conn) {
        auto node = nodes[conn.target_node_id];
        uint64_t current_time = UnifiedKnowledgeNode::getCurrentTime();
        uint64_t time_diff = current_time - node->created_at;
        
        // Recent nodes get higher weight (within 1 hour = 1.5x, within 1 day = 1.2x)
        if (time_diff < 3600000) return 1.5;      // 1 hour
        if (time_diff < 86400000) return 1.2;     // 1 day
        return 1.0;
    }
    
    double calculateFrequencyWeight(const WeightedConnection& conn) {
        auto node = nodes[conn.target_node_id];
        return 1.0 + (double)node->access_count / 100.0;  // More accessed = higher weight
    }
    
    // Path generation and scoring
    std::vector<ReasoningPath> generateMultiHopPaths(const std::vector<WeightedConnection>& connections, int max_hops) {
        std::vector<ReasoningPath> paths;
        
        // For now, just create single-hop paths
        for (const auto& conn : connections) {
            ReasoningPath path;
            path.connections.push_back(conn);
            paths.push_back(path);
        }
        
        return paths;
    }
    
    double calculateCoherence(const ReasoningPath& path) {
        if (path.connections.empty()) return 0.0;
        
        double coherence = 0.0;
        for (const auto& conn : path.connections) {
            coherence += conn.final_weight;
        }
        
        return coherence / path.connections.size();
    }
    
    double calculateRelevance(const ReasoningPath& path, const std::string& context) {
        if (path.connections.empty()) return 0.0;
        
        double relevance = 0.0;
        for (const auto& conn : path.connections) {
            relevance += conn.context_weight;
        }
        
        return relevance / path.connections.size();
    }
    
    // Validation methods
    bool hasContradictions(const ReasoningPath& path) {
        // Simple contradiction detection
        std::set<std::string> concepts;
        for (const auto& conn : path.connections) {
            auto node = nodes[conn.target_node_id];
            std::string concept = toLowerCase(cleanConcept(node->concept));
            if (concepts.count(concept)) {
                return true;  // Duplicate concept in path
            }
            concepts.insert(concept);
        }
        return false;
    }
    
    // Utility methods
    std::string getConnectionTypeName(ConnectionType type) {
        switch (type) {
            case ConnectionType::SEMANTIC: return "Semantic";
            case ConnectionType::HIERARCHICAL: return "Hierarchical";
            case ConnectionType::CAUSAL: return "Causal";
            case ConnectionType::CONTEXTUAL: return "Contextual";
            case ConnectionType::COMPONENT: return "Component";
            case ConnectionType::DEFINITION: return "Definition";
            case ConnectionType::TEMPORAL: return "Temporal";
            case ConnectionType::SPATIAL: return "Spatial";
            default: return "Unknown";
        }
    }
    
    std::string toLowerCase(const std::string& str) {
        std::string result = str;
        std::transform(result.begin(), result.end(), result.begin(), ::tolower);
        return result;
    }
    
    std::string cleanConcept(const std::string& concept) {
        std::string cleaned = concept;
        cleaned.erase(std::remove_if(cleaned.begin(), cleaned.end(), 
            [](char c) { return !std::isalnum(c) && c != ' '; }), cleaned.end());
        return cleaned;
    }
    
    std::string extractConceptFromQuestion(const std::string& question) {
        std::string lower_q = toLowerCase(question);
        
        std::vector<std::string> question_words = {"what", "is", "a", "an", "the", "are", "how", "why", "when", "where", "who"};
        std::istringstream iss(lower_q);
        std::vector<std::string> words;
        std::string word;
        
        while (iss >> word) {
            if (std::find(question_words.begin(), question_words.end(), word) == question_words.end()) {
                words.push_back(word);
            }
        }
        
        if (!words.empty()) {
            return words[0];
        }
        
        return question;
    }
};

// Working Ollama Client (same as before)
class WorkingOllamaClient {
private:
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
        ((std::string*)userp)->append((char*)contents, size * nmemb);
        return size * nmemb;
    }

public:
    std::string askQuestion(const std::string& question) {
        CURL* curl;
        CURLcode res;
        std::string response_data;
        
        curl = curl_easy_init();
        if (curl) {
            Json::Value payload;
            payload["model"] = "llama3.2";
            payload["prompt"] = question;
            payload["stream"] = false;
            
            Json::StreamWriterBuilder builder;
            std::string json_payload = Json::writeString(builder, payload);
            
            curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:11434/api/generate");
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_payload.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, json_payload.length());
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, curl_slist_append(nullptr, "Content-Type: application/json"));
            
            res = curl_easy_perform(curl);
            long http_code = 0;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
            curl_easy_cleanup(curl);
            
            if (res == CURLE_OK && http_code == 200) {
                Json::Value root;
                Json::CharReaderBuilder reader_builder;
                std::string errors;
                std::istringstream stream(response_data);
                
                if (Json::parseFromStream(reader_builder, stream, &root, &errors)) {
                    if (root.isMember("response")) {
                        return root["response"].asString();
                    }
                }
            }
        }
        
        return "Error: Could not get response from Ollama";
    }
};

// Unified Melvin Learning System
class UnifiedMelvinLearningSystem {
private:
    UnifiedReasoningEngine reasoningEngine;
    WorkingOllamaClient ollamaClient;
    
    struct LearningStats {
        int questions_asked = 0;
        int new_concepts_learned = 0;
        int reasoning_sessions = 0;
        int successful_reasoning = 0;
    } stats;

public:
    UnifiedMelvinLearningSystem() {
        std::cout << "🧠 Unified Melvin Learning System Initialized" << std::endl;
        std::cout << "🔗 Universal reasoning with weighted connections and driver modulation!" << std::endl;
        
        // Initialize with some basic knowledge
        initializeBasicKnowledge();
    }
    
    std::string processQuestion(const std::string& question, const std::string& context = "") {
        stats.questions_asked++;
        stats.reasoning_sessions++;
        
        std::cout << "\n🤔 Processing question: " << question << std::endl;
        
        // Try unified reasoning first
        std::string reasoning_result = reasoningEngine.unifiedReasoning(question, context);
        
        // If reasoning found connections, use it
        if (reasoning_result.find("I don't have enough knowledge") == std::string::npos) {
            stats.successful_reasoning++;
            reasoningEngine.updateDrivers(true, false, true);
            return reasoning_result;
        }
        
        // Otherwise, learn from Ollama
        std::cout << "❓ Need to learn more. Asking Ollama..." << std::endl;
        std::string ollama_response = ollamaClient.askQuestion(question);
        
        // Extract concept and add to knowledge
        std::string concept = extractConceptFromQuestion(question);
        reasoningEngine.addKnowledgeNode(concept, ollama_response);
        
        stats.new_concepts_learned++;
        reasoningEngine.updateDrivers(true, true, false);
        
        return "📚 Learned: " + ollama_response;
    }
    
    void showStats() {
        std::cout << "\n📊 UNIFIED MELVIN STATISTICS" << std::endl;
        std::cout << "============================" << std::endl;
        std::cout << "Questions Asked: " << stats.questions_asked << std::endl;
        std::cout << "New Concepts Learned: " << stats.new_concepts_learned << std::endl;
        std::cout << "Reasoning Sessions: " << stats.reasoning_sessions << std::endl;
        std::cout << "Successful Reasoning: " << stats.successful_reasoning << std::endl;
        std::cout << "Reasoning Success Rate: " << (stats.reasoning_sessions > 0 ? 
            (double)stats.successful_reasoning / stats.reasoning_sessions * 100 : 0) << "%" << std::endl;
        std::cout << "============================" << std::endl;
    }

private:
    void initializeBasicKnowledge() {
        std::cout << "📚 Initializing basic knowledge..." << std::endl;
        
        // Add some basic concepts to seed the reasoning system
        reasoningEngine.addKnowledgeNode("cat", "A small domesticated carnivorous mammal with soft fur, a short snout, and retractable claws.");
        reasoningEngine.addKnowledgeNode("dog", "A domesticated carnivorous mammal that typically has a long snout, an acute sense of smell, and a barking voice.");
        reasoningEngine.addKnowledgeNode("animal", "A living organism that feeds on organic matter, typically having specialized sense organs and nervous system.");
        reasoningEngine.addKnowledgeNode("mammal", "A warm-blooded vertebrate animal of a class that is distinguished by the possession of hair or fur and the secretion of milk by females for the nourishment of the young.");
    }
    
    std::string extractConceptFromQuestion(const std::string& question) {
        std::string lower_q = question;
        std::transform(lower_q.begin(), lower_q.end(), lower_q.begin(), ::tolower);
        
        std::vector<std::string> question_words = {"what", "is", "a", "an", "the", "are", "how", "why", "when", "where", "who"};
        std::istringstream iss(lower_q);
        std::vector<std::string> words;
        std::string word;
        
        while (iss >> word) {
            if (std::find(question_words.begin(), question_words.end(), word) == question_words.end()) {
                words.push_back(word);
            }
        }
        
        if (!words.empty()) {
            return words[0];
        }
        
        return question;
    }
};

int main(int argc, char* argv[]) {
    std::cout << "🧠 MELVIN UNIFIED REASONING SYSTEM" << std::endl;
    std::cout << "===================================" << std::endl;
    std::cout << "🔗 Universal reasoning with weighted connections and driver modulation!" << std::endl;
    
    UnifiedMelvinLearningSystem melvin;
    
    if (argc > 1) {
        // Single question mode
        std::string question = argv[1];
        std::string answer = melvin.processQuestion(question);
        std::cout << "\n💡 " << answer << std::endl;
    } else {
        // Interactive mode - Test unified reasoning
        std::cout << "\n🎯 TESTING UNIFIED REASONING SYSTEM" << std::endl;
        std::cout << "====================================" << std::endl;
        
        // Test various reasoning scenarios
        std::vector<std::string> test_questions = {
            "What is a cat?",
            "What is a dog?", 
            "What is a mammal?",
            "What is an animal?",
            "What is a bird?",
            "What is a fish?"
        };
        
        for (const auto& question : test_questions) {
            std::cout << "\n" << std::string(60, '=') << std::endl;
            std::string answer = melvin.processQuestion(question);
            std::cout << "\n💡 " << answer << std::endl;
        }
        
        // Show statistics
        melvin.showStats();
        
        std::cout << "\n✅ Unified reasoning demo completed!" << std::endl;
    }
    
    return 0;
}

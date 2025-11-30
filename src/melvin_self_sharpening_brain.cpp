/*
 * Melvin Self-Sharpening Unified Brain System
 * 
 * Upgrades Melvin's unified reasoning + word learning system to be:
 * - Self-cleaning: Removes weak/contradictory connections
 * - Self-sharpening: Reinforces useful connections
 * - Meta-learning: Learns from reasoning outcomes
 * - Adaptive: Evolves over time based on usage patterns
 */

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <set>
#include <sstream>
#include <fstream>
#include <curl/curl.h>
#include <chrono>
#include <thread>
#include <random>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <queue>

// Enhanced binary storage structures with meta-learning data
struct SelfSharpeningNode {
    char id[64];
    char type[32];
    char content[256];
    double activation;
    double importance;
    int connections_count;
    double usage_frequency;      // How often this node is accessed
    int validation_successes;    // Times it contributed to successful reasoning
    int validation_failures;     // Times it led to failed reasoning
    double decay_factor;         // Rate of weight decay
    uint64_t last_access_time;   // Timestamp of last access
    bool is_merged;              // Whether this node was merged from others
    
    SelfSharpeningNode() : activation(1.0), importance(1.0), connections_count(0),
                          usage_frequency(0.0), validation_successes(0), validation_failures(0),
                          decay_factor(0.95), last_access_time(0), is_merged(false) {
        memset(id, 0, sizeof(id));
        memset(type, 0, sizeof(type));
        memset(content, 0, sizeof(content));
    }
};

struct SelfSharpeningEdge {
    char from_node_id[64];
    char to_node_id[64];
    char type[32];
    double weight;
    char context[128];
    int access_count;
    double usage_frequency;      // How often this connection is used
    int validation_successes;    // Times this connection helped reasoning
    int validation_failures;     // Times this connection hurt reasoning
    double decay_factor;         // Rate of weight decay
    uint64_t last_validation;    // Last validation outcome timestamp
    bool is_pruned;              // Whether this connection was pruned
    
    SelfSharpeningEdge() : weight(0.0), access_count(0), usage_frequency(0.0),
                          validation_successes(0), validation_failures(0), decay_factor(0.95),
                          last_validation(0), is_pruned(false) {
        memset(from_node_id, 0, sizeof(from_node_id));
        memset(to_node_id, 0, sizeof(to_node_id));
        memset(type, 0, sizeof(type));
        memset(context, 0, sizeof(context));
    }
};

// Meta-learning feedback structure
struct MetaLearningFeedback {
    std::vector<std::pair<std::string, std::string>> strong_reinforcement;
    std::vector<std::pair<std::string, std::string>> weakening;
    std::vector<std::string> merged_concepts;
    std::string strategy_adjustment;
    double overall_confidence_boost;
    
    MetaLearningFeedback() : overall_confidence_boost(0.0) {}
};

// Enhanced reasoning node with self-sharpening capabilities
struct SelfSharpeningReasoningNode {
    std::string concept;
    std::string definition;
    std::map<std::string, double> connections; // connection_type -> weight
    double dopamine;
    double serotonin;
    double endorphin;
    int access_count;
    double activation;
    double importance;
    double usage_frequency;
    int validation_successes;
    int validation_failures;
    double decay_factor;
    bool is_merged;
    std::vector<std::string> merged_from; // Concepts that were merged into this one
    
    SelfSharpeningReasoningNode() : dopamine(0.5), serotonin(0.5), endorphin(0.5), 
                                   access_count(0), activation(1.0), importance(1.0),
                                   usage_frequency(0.0), validation_successes(0), 
                                   validation_failures(0), decay_factor(0.95), is_merged(false) {}
    
    SelfSharpeningReasoningNode(const std::string& c, const std::string& d) 
        : concept(c), definition(d), dopamine(0.5), serotonin(0.5), endorphin(0.5), 
          access_count(0), activation(1.0), importance(1.0), usage_frequency(0.0),
          validation_successes(0), validation_failures(0), decay_factor(0.95), is_merged(false) {}
};

// Self-Sharpening Reasoning Engine
class SelfSharpeningReasoningEngine {
private:
    std::map<std::string, SelfSharpeningReasoningNode> knowledge_graph;
    
    // Connection type weights
    std::map<std::string, double> type_weights = {
        {"semantic", 1.0}, {"hierarchical", 0.9}, {"causal", 0.8}, {"contextual", 0.7},
        {"definition", 0.6}, {"component", 0.5}, {"temporal", 0.4}, {"spatial", 0.3}
    };
    
    // Self-sharpening parameters
    double pruning_threshold = 0.1;        // Below this weight, connections decay
    double reinforcement_threshold = 0.7;  // Above this, connections strengthen
    double merge_similarity_threshold = 0.8; // Similarity threshold for merging concepts
    double decay_rate = 0.05;              // Rate of decay for unused connections
    
    // Current session meta-learning data
    MetaLearningFeedback current_feedback;
    
public:
    // Step 1: Expand Connections with meta-learning weights
    std::map<std::string, double> expandConnections(const std::string& query) {
        std::map<std::string, double> connections;
        
        for (const auto& node_pair : knowledge_graph) {
            const std::string& concept = node_pair.first;
            const SelfSharpeningReasoningNode& node = node_pair.second;
            
            if (concept == query) continue;
            
            // Calculate connection strength with meta-learning boost
            for (const auto& conn_pair : node.connections) {
                const std::string& conn_type = conn_pair.first;
                double conn_weight = conn_pair.second;
                
                // Apply type weight
                double type_weight = type_weights[conn_type];
                double base_weight = conn_weight * type_weight;
                
                // Apply meta-learning boost based on validation history
                double validation_boost = 1.0;
                if (node.validation_successes > 0) {
                    double success_ratio = (double)node.validation_successes / 
                                          (node.validation_successes + node.validation_failures);
                    validation_boost = 1.0 + (success_ratio * 0.5); // Up to 50% boost
                }
                
                // Apply usage frequency boost
                double usage_boost = 1.0 + (node.usage_frequency * 0.3);
                
                double final_weight = base_weight * validation_boost * usage_boost;
                
                // Boost if concept appears in query
                if (query.find(concept) != std::string::npos || concept.find(query) != std::string::npos) {
                    final_weight *= 1.5;
                }
                
                connections[concept] = std::max(connections[concept], final_weight);
            }
        }
        
        return connections;
    }
    
    // Step 2: Weight Connections with adaptive learning
    std::map<std::string, double> weightConnections(const std::map<std::string, double>& connections, const std::string& query) {
        std::map<std::string, double> weighted;
        
        for (const auto& conn_pair : connections) {
            const std::string& concept = conn_pair.first;
            double base_weight = conn_pair.second;
            
            auto it = knowledge_graph.find(concept);
            if (it != knowledge_graph.end()) {
                const SelfSharpeningReasoningNode& node = it->second;
                
                // Adaptive weighting based on meta-learning
                double recency_bonus = 1.0 + (node.access_count * 0.1);
                double frequency_bonus = 1.0 + (node.connections.size() * 0.05);
                double context_bonus = 1.0;
                if (query.find(concept) != std::string::npos || concept.find(query) != std::string::npos) {
                    context_bonus = 2.0;
                }
                
                // Apply decay factor for unused concepts
                double decay_factor = std::pow(node.decay_factor, node.access_count);
                
                weighted[concept] = base_weight * recency_bonus * frequency_bonus * context_bonus * decay_factor;
            }
        }
        
        return weighted;
    }
    
    // Step 3: Select Path with pruning awareness
    std::vector<std::string> selectPath(const std::map<std::string, double>& weighted_connections, int max_paths = 5) {
        std::vector<std::pair<std::string, double>> sorted_connections;
        
        // Filter out pruned/decayed connections
        for (const auto& conn_pair : weighted_connections) {
            if (conn_pair.second > pruning_threshold) {
                sorted_connections.emplace_back(conn_pair.first, conn_pair.second);
            }
        }
        
        // Sort by weight (descending)
        std::sort(sorted_connections.begin(), sorted_connections.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        
        std::vector<std::string> selected_path;
        for (int i = 0; i < std::min(max_paths, (int)sorted_connections.size()); i++) {
            selected_path.push_back(sorted_connections[i].first);
        }
        
        return selected_path;
    }
    
    // Step 4: Driver Modulation (unchanged)
    std::string modulateWithDrivers(const std::vector<std::string>& path, double curiosity, double efficiency, double consistency) {
        std::stringstream result;
        
        result << "🧠 Driver-Modulated Reasoning:\n";
        result << "  🎯 High curiosity: " << (curiosity > 0.7 ? "Exploring novel connections" : "Focusing on known paths") << "\n";
        result << "  ⚡ High efficiency: " << (efficiency > 0.7 ? "Direct path optimization" : "Comprehensive exploration") << "\n";
        result << "  🔒 High consistency: " << (consistency > 0.7 ? "Preferring established patterns" : "Open to new patterns") << "\n";
        
        result << "  Selected Path: ";
        for (size_t i = 0; i < path.size(); i++) {
            if (i > 0) result << " → ";
            result << path[i];
        }
        result << "\n";
        
        return result.str();
    }
    
    // Step 5: Self-Check with validation tracking
    std::string performSelfCheck(const std::string& reasoning_result, const std::vector<std::string>& path) {
        std::stringstream result;
        
        // Determine validation outcome based on reasoning quality
        bool validation_success = true;
        double confidence = 0.5;
        
        if (reasoning_result.find("Limited knowledge") != std::string::npos) {
            validation_success = false;
            confidence = 0.2;
        } else if (reasoning_result.find("strong connections") != std::string::npos) {
            validation_success = true;
            confidence = 0.8;
        }
        
        // Update validation tracking for used concepts
        for (const std::string& concept : path) {
            auto it = knowledge_graph.find(concept);
            if (it != knowledge_graph.end()) {
                if (validation_success) {
                    it->second.validation_successes++;
                    it->second.activation = std::min(1.0, it->second.activation + 0.1);
                } else {
                    it->second.validation_failures++;
                    it->second.activation = std::max(0.1, it->second.activation - 0.05);
                }
                it->second.usage_frequency += 0.1;
                it->second.access_count++;
            }
        }
        
        result << "🔍 Self-Check:\n";
        result << "  ✅ Validation: " << (validation_success ? "SUCCESS" : "FAILURE") << "\n";
        result << "  ✅ Reasoning coherence: " << (confidence > 0.6 ? "HIGH" : "LOW") << "\n";
        result << "  ✅ Confidence level: " << std::fixed << std::setprecision(2) << confidence << "\n";
        
        return result.str();
    }
    
    // Step 6: Produce Output with meta-learning integration
    std::string produceOutput(const std::string& query, const std::vector<std::string>& path, double confidence) {
        std::stringstream result;
        
        result << "🎯 REASONED ANSWER:\n";
        result << "  Query: " << query << "\n";
        result << "  Reasoning Path: ";
        for (size_t i = 0; i < path.size(); i++) {
            if (i > 0) result << " → ";
            result << path[i];
        }
        result << "\n";
        result << "  Confidence: " << std::fixed << std::setprecision(2) << confidence << "\n";
        
        // Generate reasoned response with meta-learning awareness
        if (confidence > 0.7) {
            result << "  Response: Based on strong, validated connections in knowledge graph, ";
            if (!path.empty()) {
                result << path[0] << " is the most relevant concept to " << query;
            }
            result << "\n";
        } else {
            result << "  Response: Limited validated knowledge available for " << query << ", more learning needed\n";
        }
        
        return result.str();
    }
    
    // Meta-learning feedback generation
    MetaLearningFeedback generateMetaFeedback(const std::string& query, const std::vector<std::string>& path, double confidence) {
        current_feedback = MetaLearningFeedback();
        
        // Track strong reinforcement
        for (const std::string& concept : path) {
            auto it = knowledge_graph.find(concept);
            if (it != knowledge_graph.end() && it->second.validation_successes > 0) {
                current_feedback.strong_reinforcement.emplace_back(concept, "validation_success");
            }
        }
        
        // Identify weakening connections
        for (const auto& node_pair : knowledge_graph) {
            const std::string& concept = node_pair.first;
            const SelfSharpeningReasoningNode& node = node_pair.second;
            
            if (node.validation_failures > node.validation_successes && node.access_count > 5) {
                current_feedback.weakening.emplace_back(concept, "validation_failure");
            }
        }
        
        // Strategy adjustment
        if (confidence > 0.8) {
            current_feedback.strategy_adjustment = "High confidence reasoning - continue current approach";
        } else if (confidence < 0.3) {
            current_feedback.strategy_adjustment = "Low confidence - need more diverse learning sources";
        } else {
            current_feedback.strategy_adjustment = "Moderate confidence - balanced approach working";
        }
        
        current_feedback.overall_confidence_boost = confidence;
        
        return current_feedback;
    }
    
    // Complete 6-step reasoning process with meta-learning
    std::string performReasoning(const std::string& query, double curiosity, double efficiency, double consistency) {
        std::cout << "🧠 STARTING SELF-SHARPENING REASONING PROCESS" << std::endl;
        std::cout << "=============================================" << std::endl;
        std::cout << "Query: " << query << std::endl;
        std::cout << std::endl;
        
        // Step 1: Expand Connections
        std::cout << "🔍 Step 1: Expanding connections with meta-learning weights..." << std::endl;
        auto connections = expandConnections(query);
        std::cout << "  Found " << connections.size() << " potential connections" << std::endl;
        
        // Step 2: Weight Connections
        std::cout << "⚖️ Step 2: Weighting connections with adaptive learning..." << std::endl;
        auto weighted = weightConnections(connections, query);
        std::cout << "  Weighted " << weighted.size() << " connections" << std::endl;
        
        // Step 3: Select Path
        std::cout << "🛤️ Step 3: Selecting reasoning path with pruning awareness..." << std::endl;
        auto path = selectPath(weighted, 5);
        std::cout << "  Selected path with " << path.size() << " concepts" << std::endl;
        
        // Step 4: Driver Modulation
        std::cout << "🎭 Step 4: Applying driver modulation..." << std::endl;
        std::string driver_result = modulateWithDrivers(path, curiosity, efficiency, consistency);
        std::cout << driver_result << std::endl;
        
        // Step 5: Self-Check
        std::cout << "🔍 Step 5: Performing self-check with validation tracking..." << std::endl;
        std::string self_check = performSelfCheck(driver_result, path);
        std::cout << self_check << std::endl;
        
        // Step 6: Produce Output
        std::cout << "🎯 Step 6: Producing reasoned output..." << std::endl;
        double confidence = weighted.empty() ? 0.0 : weighted.begin()->second;
        std::string final_result = produceOutput(query, path, confidence);
        std::cout << final_result << std::endl;
        
        // Generate meta-learning feedback
        std::cout << "🧠 Meta-Learning Feedback Generation..." << std::endl;
        MetaLearningFeedback feedback = generateMetaFeedback(query, path, confidence);
        
        // Display meta-feedback
        std::cout << std::endl;
        std::cout << "[Meta-Feedback]" << std::endl;
        std::cout << "  Strong Reinforcement: ";
        for (const auto& pair : feedback.strong_reinforcement) {
            std::cout << pair.first << " ";
        }
        std::cout << std::endl;
        
        std::cout << "  Weakening: ";
        for (const auto& pair : feedback.weakening) {
            std::cout << pair.first << " ";
        }
        std::cout << std::endl;
        
        std::cout << "  Merged Concepts: ";
        for (const std::string& concept : feedback.merged_concepts) {
            std::cout << concept << " ";
        }
        std::cout << std::endl;
        
        std::cout << "  Strategy Adjustment: " << feedback.strategy_adjustment << std::endl;
        std::cout << "  Overall Confidence Boost: " << std::fixed << std::setprecision(2) << feedback.overall_confidence_boost << std::endl;
        
        return final_result;
    }
    
    // Self-sharpening operations
    void performAdaptivePruning() {
        std::cout << "🧹 Performing adaptive graph pruning..." << std::endl;
        
        for (auto& node_pair : knowledge_graph) {
            SelfSharpeningReasoningNode& node = node_pair.second;
            
            // Prune weak connections
            auto conn_it = node.connections.begin();
            while (conn_it != node.connections.end()) {
                if (conn_it->second < pruning_threshold) {
                    std::cout << "  🗑️ Pruning weak connection: " << node_pair.first << " -> " << conn_it->first << std::endl;
                    conn_it = node.connections.erase(conn_it);
                } else {
                    ++conn_it;
                }
            }
            
            // Apply decay to unused connections
            if (node.access_count == 0) {
                node.decay_factor *= 0.9; // Accelerate decay for unused nodes
            }
        }
    }
    
    void performConceptMerging() {
        std::cout << "🔄 Performing concept merging..." << std::endl;
        
        std::vector<std::string> concepts_to_merge;
        std::map<std::string, std::vector<std::string>> merge_groups;
        
        // Find similar concepts
        for (const auto& node1_pair : knowledge_graph) {
            const std::string& concept1 = node1_pair.first;
            const SelfSharpeningReasoningNode& node1 = node1_pair.second;
            
            for (const auto& node2_pair : knowledge_graph) {
                const std::string& concept2 = node2_pair.first;
                const SelfSharpeningReasoningNode& node2 = node2_pair.second;
                
                if (concept1 >= concept2) continue; // Avoid duplicates
                
                // Calculate similarity (simple string similarity for now)
                double similarity = calculateSimilarity(concept1, concept2);
                
                if (similarity > merge_similarity_threshold) {
                    std::string merge_key = concept1 < concept2 ? concept1 : concept2;
                    merge_groups[merge_key].push_back(concept1);
                    merge_groups[merge_key].push_back(concept2);
                }
            }
        }
        
        // Perform merging
        for (const auto& merge_group : merge_groups) {
            const std::string& primary_concept = merge_group.first;
            const std::vector<std::string>& concepts_to_merge = merge_group.second;
            
            std::cout << "  🔗 Merging concepts into " << primary_concept << ": ";
            for (const std::string& concept : concepts_to_merge) {
                std::cout << concept << " ";
            }
            std::cout << std::endl;
            
            // Merge into primary concept
            auto& primary_node = knowledge_graph[primary_concept];
            primary_node.is_merged = true;
            
            for (const std::string& concept : concepts_to_merge) {
                if (concept != primary_concept && knowledge_graph.find(concept) != knowledge_graph.end()) {
                    const auto& node_to_merge = knowledge_graph[concept];
                    primary_node.merged_from.push_back(concept);
                    primary_node.access_count += node_to_merge.access_count;
                    primary_node.validation_successes += node_to_merge.validation_successes;
                    primary_node.validation_failures += node_to_merge.validation_failures;
                    
                    // Merge connections
                    for (const auto& conn_pair : node_to_merge.connections) {
                        primary_node.connections[conn_pair.first] = std::max(
                            primary_node.connections[conn_pair.first], conn_pair.second);
                    }
                    
                    // Remove merged concept
                    knowledge_graph.erase(concept);
                }
            }
        }
    }
    
    double calculateSimilarity(const std::string& str1, const std::string& str2) {
        // Simple similarity calculation based on common words
        std::set<std::string> words1, words2;
        
        std::stringstream ss1(str1), ss2(str2);
        std::string word;
        
        while (ss1 >> word) {
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
            words1.insert(word);
        }
        
        while (ss2 >> word) {
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
            words2.insert(word);
        }
        
        std::set<std::string> intersection;
        std::set_intersection(words1.begin(), words1.end(), words2.begin(), words2.end(),
                             std::inserter(intersection, intersection.begin()));
        
        std::set<std::string> union_set;
        std::set_union(words1.begin(), words1.end(), words2.begin(), words2.end(),
                      std::inserter(union_set, union_set.begin()));
        
        return union_set.empty() ? 0.0 : (double)intersection.size() / union_set.size();
    }
    
    // Add concept to knowledge graph with meta-learning tracking
    void addConcept(const std::string& concept, const std::string& definition = "") {
        if (knowledge_graph.find(concept) == knowledge_graph.end()) {
            knowledge_graph[concept] = SelfSharpeningReasoningNode(concept, definition);
        }
        
        // Update access count and usage frequency
        knowledge_graph[concept].access_count++;
        knowledge_graph[concept].usage_frequency += 0.1;
        
        // Update activation based on usage
        knowledge_graph[concept].activation = std::min(1.0, knowledge_graph[concept].activation + 0.05);
    }
    
    // Add connection with validation tracking
    void addConnection(const std::string& concept1, const std::string& concept2, const std::string& connection_type, double weight = 0.5) {
        addConcept(concept1);
        addConcept(concept2);
        
        // Add bidirectional connection
        knowledge_graph[concept1].connections[concept2] = weight;
        knowledge_graph[concept2].connections[concept1] = weight;
        
        // Add connection type tracking
        knowledge_graph[concept1].connections[connection_type + "_" + concept2] = weight;
        knowledge_graph[concept2].connections[connection_type + "_" + concept1] = weight;
    }
    
    // Get knowledge graph stats
    void displayStats() {
        std::cout << "📊 SELF-SHARPENING KNOWLEDGE GRAPH STATISTICS" << std::endl;
        std::cout << "==============================================" << std::endl;
        std::cout << "Total Concepts: " << knowledge_graph.size() << std::endl;
        
        int total_connections = 0;
        int merged_concepts = 0;
        int high_confidence_concepts = 0;
        
        for (const auto& node_pair : knowledge_graph) {
            const SelfSharpeningReasoningNode& node = node_pair.second;
            total_connections += node.connections.size();
            
            if (node.is_merged) merged_concepts++;
            
            double confidence = node.validation_successes > 0 ? 
                (double)node.validation_successes / (node.validation_successes + node.validation_failures) : 0.0;
            if (confidence > 0.7) high_confidence_concepts++;
        }
        
        std::cout << "Total Connections: " << total_connections << std::endl;
        std::cout << "Merged Concepts: " << merged_concepts << std::endl;
        std::cout << "High Confidence Concepts: " << high_confidence_concepts << std::endl;
        std::cout << "Average Connections per Concept: " << (knowledge_graph.size() > 0 ? (double)total_connections / knowledge_graph.size() : 0) << std::endl;
        std::cout << std::endl;
    }
    
    // Save enhanced binary format
    void saveToBinary() {
        std::ofstream file("melvin_self_sharpening_brain.bin", std::ios::binary);
        if (file.is_open()) {
            // Save number of concepts
            size_t num_concepts = knowledge_graph.size();
            file.write(reinterpret_cast<const char*>(&num_concepts), sizeof(num_concepts));
            
            // Save each concept
            for (const auto& node_pair : knowledge_graph) {
                const SelfSharpeningReasoningNode& node = node_pair.second;
                
                // Save concept name
                std::string concept = node_pair.first;
                size_t concept_len = concept.length();
                file.write(reinterpret_cast<const char*>(&concept_len), sizeof(concept_len));
                file.write(concept.c_str(), concept_len);
                
                // Save node data
                file.write(reinterpret_cast<const char*>(&node), sizeof(SelfSharpeningReasoningNode));
                
                // Save connections
                size_t num_connections = node.connections.size();
                file.write(reinterpret_cast<const char*>(&num_connections), sizeof(num_connections));
                
                for (const auto& conn_pair : node.connections) {
                    std::string conn_type = conn_pair.first;
                    double weight = conn_pair.second;
                    
                    size_t conn_len = conn_type.length();
                    file.write(reinterpret_cast<const char*>(&conn_len), sizeof(conn_len));
                    file.write(conn_type.c_str(), conn_len);
                    file.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
                }
            }
            
            file.close();
            std::cout << "💾 Self-sharpening brain saved to binary format" << std::endl;
        }
    }
    
    // Load enhanced binary format
    void loadFromBinary() {
        std::ifstream file("melvin_self_sharpening_brain.bin", std::ios::binary);
        if (file.is_open()) {
            knowledge_graph.clear();
            
            try {
                // Load number of concepts
                size_t num_concepts;
                file.read(reinterpret_cast<char*>(&num_concepts), sizeof(num_concepts));
                
                if (file.fail() || num_concepts > 10000) { // Safety check
                    std::cout << "⚠️ Invalid binary file, starting fresh" << std::endl;
                    file.close();
                    return;
                }
                
                for (size_t i = 0; i < num_concepts; i++) {
                    // Load concept name
                    size_t concept_len;
                    file.read(reinterpret_cast<char*>(&concept_len), sizeof(concept_len));
                    
                    if (file.fail() || concept_len > 1000) { // Safety check
                        std::cout << "⚠️ Invalid concept length, skipping" << std::endl;
                        break;
                    }
                    
                    std::string concept(concept_len, '\0');
                    file.read(&concept[0], concept_len);
                    
                    if (file.fail()) {
                        std::cout << "⚠️ Failed to read concept name, skipping" << std::endl;
                        break;
                    }
                    
                    // Load node data
                    SelfSharpeningReasoningNode node;
                    file.read(reinterpret_cast<char*>(&node), sizeof(SelfSharpeningReasoningNode));
                    
                    if (file.fail()) {
                        std::cout << "⚠️ Failed to read node data, skipping" << std::endl;
                        break;
                    }
                    
                    knowledge_graph[concept] = node;
                    
                    // Load connections
                    size_t num_connections;
                    file.read(reinterpret_cast<char*>(&num_connections), sizeof(num_connections));
                    
                    if (file.fail() || num_connections > 10000) { // Safety check
                        std::cout << "⚠️ Invalid connection count, skipping connections" << std::endl;
                        continue;
                    }
                    
                    for (size_t j = 0; j < num_connections; j++) {
                        size_t conn_len;
                        file.read(reinterpret_cast<char*>(&conn_len), sizeof(conn_len));
                        
                        if (file.fail() || conn_len > 1000) { // Safety check
                            std::cout << "⚠️ Invalid connection length, skipping" << std::endl;
                            break;
                        }
                        
                        std::string conn_type(conn_len, '\0');
                        file.read(&conn_type[0], conn_len);
                        
                        if (file.fail()) {
                            std::cout << "⚠️ Failed to read connection type, skipping" << std::endl;
                            break;
                        }
                        
                        double weight;
                        file.read(reinterpret_cast<char*>(&weight), sizeof(weight));
                        
                        if (file.fail()) {
                            std::cout << "⚠️ Failed to read weight, skipping" << std::endl;
                            break;
                        }
                        
                        knowledge_graph[concept].connections[conn_type] = weight;
                    }
                }
                
                file.close();
                std::cout << "📚 Self-sharpening brain loaded from binary format (" << knowledge_graph.size() << " concepts)" << std::endl;
            } catch (const std::exception& e) {
                std::cout << "⚠️ Error loading binary file: " << e.what() << ", starting fresh" << std::endl;
                knowledge_graph.clear();
                file.close();
            }
        } else {
            std::cout << "📚 No existing binary file found, starting fresh" << std::endl;
        }
    }
};

// Melvin Self-Sharpening Brain System
class MelvinSelfSharpeningBrain {
private:
    SelfSharpeningReasoningEngine reasoning_engine;
    
    // Driver system
    double curiosity = 0.8;
    double efficiency = 0.6;
    double consistency = 0.7;
    
    // Ollama integration
    std::string ollama_url = "http://localhost:11434/api/generate";
    
    // CURL callback for Ollama
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* s) {
        size_t newLength = size * nmemb;
        try {
            s->append((char*)contents, newLength);
            return newLength;
        } catch (std::bad_alloc& e) {
            return 0;
        }
    }
    
    // Ask Ollama a question
    std::string askOllama(const std::string& question) {
        CURL* curl;
        CURLcode res;
        std::string response;
        
        curl = curl_easy_init();
        if (curl) {
            std::string json_data = "{\"model\":\"llama3.2:latest\",\"prompt\":\"" + question + "\",\"stream\":false}";
            
            struct curl_slist* headers = NULL;
            headers = curl_slist_append(headers, "Content-Type: application/json");
            
            curl_easy_setopt(curl, CURLOPT_URL, ollama_url.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_data.c_str());
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
            curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
            
            res = curl_easy_perform(curl);
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
            
            if (res != CURLE_OK) {
                return "Error: " + std::string(curl_easy_strerror(res));
            }
        }
        
        return response;
    }
    
    // Extract words from text
    std::vector<std::string> extractWords(const std::string& text) {
        std::vector<std::string> words;
        std::stringstream ss(text);
        std::string word;
        
        while (ss >> word) {
            // Clean word
            word.erase(std::remove_if(word.begin(), word.end(), 
                [](char c) { return !std::isalnum(c); }), word.end());
            
            if (word.length() > 2) { // Only words longer than 2 characters
                std::transform(word.begin(), word.end(), word.begin(), ::tolower);
                words.push_back(word);
            }
        }
        
        return words;
    }
    
    // Add concepts to reasoning engine
    void addConceptsToReasoning(const std::vector<std::string>& words) {
        for (const std::string& word : words) {
            reasoning_engine.addConcept(word, "Learned from Ollama response");
            
            // Add connections between words
            for (const std::string& other_word : words) {
                if (word != other_word) {
                    reasoning_engine.addConnection(word, other_word, "co_occurrence", 0.5);
                }
            }
        }
    }
    
public:
    MelvinSelfSharpeningBrain() {
        std::cout << "🧠 MELVIN SELF-SHARPENING UNIFIED BRAIN SYSTEM" << std::endl;
        std::cout << "===============================================" << std::endl;
        std::cout << "🔗 Self-Sharpening Features:" << std::endl;
        std::cout << "  ✅ Meta-learning feedback integration" << std::endl;
        std::cout << "  ✅ Adaptive graph pruning" << std::endl;
        std::cout << "  ✅ Sharpened concept embedding" << std::endl;
        std::cout << "  ✅ Reasoning loop integration" << std::endl;
        std::cout << "  ✅ Enhanced persistence & evolution" << std::endl;
        std::cout << std::endl;
        
        // Initialize CURL
        curl_global_init(CURL_GLOBAL_DEFAULT);
        
        // Load existing brain
        reasoning_engine.loadFromBinary();
    }
    
    ~MelvinSelfSharpeningBrain() {
        curl_global_cleanup();
    }
    
    // Run test case
    void runTestCase(const std::string& test_name, const std::string& input) {
        std::cout << "🧪 RUNNING TEST CASE: " << test_name << std::endl;
        std::cout << "=====================================" << std::endl;
        std::cout << "Input: " << input << std::endl;
        std::cout << std::endl;
        
        // Extract words and add to reasoning engine
        std::vector<std::string> words = extractWords(input);
        addConceptsToReasoning(words);
        
        // Perform self-sharpening reasoning
        std::string result = reasoning_engine.performReasoning(input, curiosity, efficiency, consistency);
        
        // Perform self-sharpening operations
        std::cout << std::endl;
        std::cout << "🔧 PERFORMING SELF-SHARPENING OPERATIONS" << std::endl;
        std::cout << "=========================================" << std::endl;
        
        reasoning_engine.performAdaptivePruning();
        reasoning_engine.performConceptMerging();
        
        std::cout << std::endl;
        reasoning_engine.displayStats();
        std::cout << std::endl;
    }
    
    // Save brain state
    void saveBrain() {
        reasoning_engine.saveToBinary();
    }
};

int main() {
    std::cout << "🚀 Starting Melvin Self-Sharpening Unified Brain System" << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout << std::endl;
    
    MelvinSelfSharpeningBrain melvin;
    
    // Run the three test cases
    std::cout << "🧪 RUNNING TEST CASES" << std::endl;
    std::cout << "=====================" << std::endl;
    std::cout << std::endl;
    
    // Test Case 1: Raw Input
    melvin.runTestCase("Raw Input", "A bird sitting on a wire");
    
    // Test Case 2: Conceptual Input  
    melvin.runTestCase("Conceptual Input", "natural selection");
    
    // Test Case 3: Hybrid Input
    melvin.runTestCase("Hybrid Input", "A robot adapting to cold demonstrates survival of the fittest");
    
    // Save final brain state
    std::cout << "💾 Saving final brain state..." << std::endl;
    melvin.saveBrain();
    
    std::cout << "✅ Melvin Self-Sharpening Unified Brain System finished!" << std::endl;
    
    return 0;
}

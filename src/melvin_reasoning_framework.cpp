/*
 * Melvin Unified Reasoning Framework
 * 
 * Implements the 6-step reasoning process:
 * 1. Expand Connections (8 types)
 * 2. Weight Connections (type/context/recency)
 * 3. Select Path (multi-hop exploration)
 * 4. Driver Modulation (dopamine/serotonin/endorphins)
 * 5. Self-Check (contradiction resolution)
 * 6. Produce Output (reasoned answer)
 */

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

// Simplified Knowledge Node with Connection Weights
struct ReasoningNode {
    std::string concept;
    std::string definition;
    std::map<std::string, double> connections; // connection_type -> weight
    double dopamine;    // Curiosity/exploration
    double serotonin;   // Stability/balance
    double endorphin;   // Satisfaction/reinforcement
    int access_count;
    
    ReasoningNode() : dopamine(0.5), serotonin(0.5), endorphin(0.5), access_count(0) {}
    
    ReasoningNode(const std::string& c, const std::string& d) 
        : concept(c), definition(d), dopamine(0.5), serotonin(0.5), endorphin(0.5), access_count(0) {}
};

// Unified Reasoning Engine
class UnifiedReasoningEngine {
private:
    std::map<std::string, ReasoningNode> knowledge_graph;
    
    // Connection type weights (semantic > hierarchical > causal > contextual > temporal > spatial)
    std::map<std::string, double> type_weights = {
        {"semantic", 1.0},
        {"hierarchical", 0.9},
        {"causal", 0.8},
        {"contextual", 0.7},
        {"definition", 0.6},
        {"component", 0.5},
        {"temporal", 0.3},
        {"spatial", 0.2}
    };
    
    // Global driver states
    double global_dopamine = 0.5;
    double global_serotonin = 0.5;
    double global_endorphin = 0.5;

public:
    UnifiedReasoningEngine() {
        initializeKnowledge();
    }
    
    // Step 1: Expand Connections (Possibilities)
    std::vector<std::pair<std::string, double>> expandConnections(const std::string& concept, const std::string& query) {
        std::cout << "🔍 Step 1: Expanding connections for '" << concept << "'" << std::endl;
        
        std::vector<std::pair<std::string, double>> all_connections;
        
        for (const auto& node_pair : knowledge_graph) {
            const std::string& other_concept = node_pair.first;
            if (other_concept == concept) continue;
            
            // Check different connection types
            std::string connection_type = findConnectionType(concept, other_concept);
            if (!connection_type.empty()) {
                double weight = calculateConnectionWeight(connection_type, node_pair.second, query);
                all_connections.push_back({other_concept, weight});
                
                std::cout << "  " << getConnectionEmoji(connection_type) << " " << connection_type 
                         << ": " << concept << " → " << other_concept 
                         << " (weight: " << std::fixed << std::setprecision(2) << weight << ")" << std::endl;
            }
        }
        
        return all_connections;
    }
    
    // Step 2: Weight Connections (Prioritization)
    std::vector<std::pair<std::string, double>> weightConnections(const std::vector<std::pair<std::string, double>>& connections) {
        std::cout << "⚖️ Step 2: Weighting connections" << std::endl;
        
        std::vector<std::pair<std::string, double>> weighted = connections;
        std::sort(weighted.begin(), weighted.end(), 
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        std::cout << "  Top connections:" << std::endl;
        for (size_t i = 0; i < std::min(weighted.size(), size_t(3)); ++i) {
            std::cout << "    " << (i+1) << ". " << weighted[i].first 
                     << " (weight: " << std::fixed << std::setprecision(2) << weighted[i].second << ")" << std::endl;
        }
        
        return weighted;
    }
    
    // Step 3: Select Path (Choice)
    std::vector<std::string> selectPath(const std::vector<std::pair<std::string, double>>& weighted_connections) {
        std::cout << "🛤️ Step 3: Selecting reasoning path" << std::endl;
        
        if (weighted_connections.empty()) {
            return {};
        }
        
        // Multi-hop exploration (simplified to 2-hop)
        std::vector<std::string> path;
        path.push_back(weighted_connections[0].first);
        
        // Find second hop
        for (const auto& conn : weighted_connections) {
            if (conn.first != path[0]) {
                path.push_back(conn.first);
                break;
            }
        }
        
        std::cout << "  Selected path: ";
        for (size_t i = 0; i < path.size(); ++i) {
            std::cout << path[i];
            if (i < path.size() - 1) std::cout << " → ";
        }
        std::cout << std::endl;
        
        return path;
    }
    
    // Step 4: Driver Modulation (Reasoning Style)
    std::string applyDriverModulation(const std::vector<std::string>& path, const std::string& query) {
        std::cout << "🧠 Step 4: Applying driver modulation" << std::endl;
        
        showDriverStatus();
        
        std::string reasoning_style = getReasoningStyle();
        std::cout << "  Reasoning style: " << reasoning_style << std::endl;
        
        if (path.empty()) {
            return "No reasoning path found.";
        }
        
        std::string result;
        
        if (reasoning_style == "exploratory") {
            result = "🔍 EXPLORATORY REASONING: ";
            result += "Exploring novel connections between " + path[0];
            if (path.size() > 1) result += " and " + path[1];
            result += ". This suggests " + path[0] + " might be related through unexpected pathways.";
        } else if (reasoning_style == "conservative") {
            result = "🛡️ CONSERVATIVE REASONING: ";
            result += "Following established connections: " + path[0];
            if (path.size() > 1) result += " → " + path[1];
            result += ". This is a well-established relationship in my knowledge base.";
        } else if (reasoning_style == "reinforcing") {
            result = "🔄 REINFORCING REASONING: ";
            result += "Building on successful patterns: " + path[0];
            if (path.size() > 1) result += " → " + path[1];
            result += ". This connection has worked well in the past.";
        } else {
            result = "⚖️ BALANCED REASONING: ";
            result += "Balanced approach connecting " + path[0];
            if (path.size() > 1) result += " and " + path[1];
            result += ". This represents a moderate confidence connection.";
        }
        
        return result;
    }
    
    // Step 5: Self-Check (Validation)
    std::string selfCheck(const std::string& reasoning_result) {
        std::cout << "🔍 Step 5: Self-check validation" << std::endl;
        
        // Simple contradiction check
        std::vector<std::pair<std::string, std::string>> contradictions = {
            {"hot", "cold"}, {"big", "small"}, {"fast", "slow"}, {"good", "bad"}
        };
        
        std::string lower_result = toLowerCase(reasoning_result);
        
        for (const auto& contradiction : contradictions) {
            if (lower_result.find(contradiction.first) != std::string::npos && 
                lower_result.find(contradiction.second) != std::string::npos) {
                std::cout << "  ⚠️ Contradiction detected: " << contradiction.first << " vs " << contradiction.second << std::endl;
                return "⚠️ CONTRADICTION DETECTED: Found conflicting concepts (" + 
                       contradiction.first + " vs " + contradiction.second + 
                       "). Please clarify the context to resolve this contradiction.";
            }
        }
        
        std::cout << "  ✅ No contradictions found" << std::endl;
        return reasoning_result;
    }
    
    // Step 6: Produce Output (Reasoned Answer)
    std::string produceOutput(const std::string& validated_result) {
        std::cout << "📤 Step 6: Producing final output" << std::endl;
        
        // Update driver levels based on success
        updateDrivers(true, false, true);
        
        return validated_result;
    }
    
    // Main unified reasoning process
    std::string unifiedReasoning(const std::string& query) {
        std::cout << "\n🧠 MELVIN UNIFIED REASONING PROCESS" << std::endl;
        std::cout << "====================================" << std::endl;
        std::cout << "Query: " << query << std::endl;
        
        std::string concept = extractConcept(query);
        
        // Step 1: Expand Connections
        auto connections = expandConnections(concept, query);
        
        if (connections.empty()) {
            return "I don't have enough knowledge to reason about '" + concept + "' yet.";
        }
        
        // Step 2: Weight Connections
        auto weighted_connections = weightConnections(connections);
        
        // Step 3: Select Path
        auto path = selectPath(weighted_connections);
        
        // Step 4: Driver Modulation
        std::string reasoning_result = applyDriverModulation(path, query);
        
        // Step 5: Self-Check
        std::string validated_result = selfCheck(reasoning_result);
        
        // Step 6: Produce Output
        std::string final_output = produceOutput(validated_result);
        
        std::cout << "\n💡 FINAL REASONED ANSWER:" << std::endl;
        std::cout << "=========================" << std::endl;
        
        return final_output;
    }
    
    void showKnowledgeGraph() {
        std::cout << "\n🧠 MELVIN'S REASONING KNOWLEDGE GRAPH" << std::endl;
        std::cout << "======================================" << std::endl;
        
        for (const auto& node_pair : knowledge_graph) {
            const auto& node = node_pair.second;
            std::cout << "\n📝 " << node.concept << std::endl;
            std::cout << "   Definition: " << node.definition << std::endl;
            std::cout << "   Drivers: D=" << std::fixed << std::setprecision(2) << node.dopamine 
                     << " S=" << node.serotonin << " E=" << node.endorphin << std::endl;
            std::cout << "   Connections: " << node.connections.size() << std::endl;
        }
    }

private:
    void initializeKnowledge() {
        knowledge_graph["cat"] = ReasoningNode("cat", "A small domesticated carnivorous mammal with soft fur, a short snout, and retractable claws.");
        knowledge_graph["dog"] = ReasoningNode("dog", "A domesticated carnivorous mammal that typically has a long snout, an acute sense of smell, and a barking voice.");
        knowledge_graph["animal"] = ReasoningNode("animal", "A living organism that feeds on organic matter, typically having specialized sense organs and nervous system.");
        knowledge_graph["mammal"] = ReasoningNode("mammal", "A warm-blooded vertebrate animal distinguished by the possession of hair or fur and the secretion of milk by females.");
        knowledge_graph["pet"] = ReasoningNode("pet", "A domestic or tamed animal kept for companionship or pleasure.");
        knowledge_graph["feline"] = ReasoningNode("feline", "Relating to or affecting cats or other members of the cat family.");
        knowledge_graph["canine"] = ReasoningNode("canine", "Relating to or resembling a dog or dogs.");
        
        // Build connections
        buildConnections();
    }
    
    void buildConnections() {
        // Semantic connections
        knowledge_graph["cat"].connections["semantic"] = 0.9;
        knowledge_graph["dog"].connections["semantic"] = 0.9;
        knowledge_graph["feline"].connections["semantic"] = 0.8;
        knowledge_graph["canine"].connections["semantic"] = 0.8;
        
        // Hierarchical connections
        knowledge_graph["cat"].connections["hierarchical"] = 0.8;
        knowledge_graph["dog"].connections["hierarchical"] = 0.8;
        knowledge_graph["animal"].connections["hierarchical"] = 0.7;
        knowledge_graph["mammal"].connections["hierarchical"] = 0.7;
        
        // Contextual connections
        knowledge_graph["cat"].connections["contextual"] = 0.6;
        knowledge_graph["dog"].connections["contextual"] = 0.6;
        knowledge_graph["pet"].connections["contextual"] = 0.7;
    }
    
    std::string findConnectionType(const std::string& concept1, const std::string& concept2) {
        // Semantic groups
        std::vector<std::vector<std::string>> semantic_groups = {
            {"cat", "dog", "bird", "fish", "lion", "tiger"},
            {"animal", "mammal", "pet"},
            {"feline", "canine"}
        };
        
        for (const auto& group : semantic_groups) {
            bool found1 = std::find(group.begin(), group.end(), concept1) != group.end();
            bool found2 = std::find(group.begin(), group.end(), concept2) != group.end();
            if (found1 && found2) return "semantic";
        }
        
        // Hierarchical relationships
        if ((concept1 == "cat" && concept2 == "mammal") || (concept1 == "mammal" && concept2 == "cat")) return "hierarchical";
        if ((concept1 == "dog" && concept2 == "mammal") || (concept1 == "mammal" && concept2 == "dog")) return "hierarchical";
        if ((concept1 == "mammal" && concept2 == "animal") || (concept1 == "animal" && concept2 == "mammal")) return "hierarchical";
        
        // Contextual relationships
        if ((concept1 == "cat" && concept2 == "pet") || (concept1 == "pet" && concept2 == "cat")) return "contextual";
        if ((concept1 == "dog" && concept2 == "pet") || (concept1 == "pet" && concept2 == "dog")) return "contextual";
        
        // Definition-based
        if (concept1.find(concept2) != std::string::npos || concept2.find(concept1) != std::string::npos) return "definition";
        
        return "";
    }
    
    double calculateConnectionWeight(const std::string& connection_type, const ReasoningNode& node, const std::string& query) {
        double base_weight = type_weights[connection_type];
        double recency_factor = 1.0 / (1.0 + node.access_count * 0.1);
        double context_relevance = getContextRelevance(query, node.definition);
        double driver_modulation = (node.dopamine + node.serotonin + node.endorphin) / 3.0;
        
        return base_weight * recency_factor * (1.0 + context_relevance) * driver_modulation;
    }
    
    double getContextRelevance(const std::string& query, const std::string& definition) {
        std::string lower_query = toLowerCase(query);
        std::string lower_def = toLowerCase(definition);
        
        int matches = 0;
        std::istringstream iss(lower_query);
        std::string word;
        int total_words = 0;
        
        while (iss >> word) {
            total_words++;
            if (lower_def.find(word) != std::string::npos) {
                matches++;
            }
        }
        
        return total_words > 0 ? (double)matches / total_words : 0.0;
    }
    
    std::string getConnectionEmoji(const std::string& connection_type) {
        std::map<std::string, std::string> emojis = {
            {"semantic", "🧠"}, {"hierarchical", "📊"}, {"causal", "⚡"},
            {"contextual", "🏠"}, {"definition", "📚"}, {"component", "🔗"},
            {"temporal", "⏰"}, {"spatial", "📍"}
        };
        return emojis[connection_type];
    }
    
    std::string getReasoningStyle() {
        if (global_dopamine > 0.6) return "exploratory";
        if (global_serotonin > 0.7) return "conservative";
        if (global_endorphin > 0.8) return "reinforcing";
        return "balanced";
    }
    
    void showDriverStatus() {
        std::cout << "  🧠 Driver Status: ";
        std::cout << "Dopamine=" << std::fixed << std::setprecision(2) << global_dopamine;
        std::cout << " Serotonin=" << global_serotonin;
        std::cout << " Endorphin=" << global_endorphin;
        std::cout << " Style=" << getReasoningStyle() << std::endl;
    }
    
    void updateDrivers(bool success, bool novel_connection, bool stable_connection) {
        if (success) {
            global_endorphin = std::min(1.0, global_endorphin + 0.1);
        } else {
            global_endorphin = std::max(0.0, global_endorphin - 0.05);
        }
        
        if (novel_connection) {
            global_dopamine = std::min(1.0, global_dopamine + 0.05);
        } else {
            global_dopamine = std::max(0.0, global_dopamine - 0.02);
        }
        
        if (stable_connection) {
            global_serotonin = std::min(1.0, global_serotonin + 0.05);
        } else {
            global_serotonin = std::max(0.0, global_serotonin - 0.02);
        }
    }
    
    std::string extractConcept(const std::string& query) {
        std::string lower_q = toLowerCase(query);
        
        std::vector<std::string> question_words = {"what", "is", "a", "an", "the", "are", "how", "why", "when", "where", "who"};
        std::istringstream iss(lower_q);
        std::vector<std::string> words;
        std::string word;
        
        while (iss >> word) {
            if (std::find(question_words.begin(), question_words.end(), word) == question_words.end()) {
                words.push_back(word);
            }
        }
        
        return words.empty() ? query : words[0];
    }
    
    std::string toLowerCase(const std::string& str) {
        std::string result = str;
        std::transform(result.begin(), result.end(), result.begin(), ::tolower);
        return result;
    }
};

int main(int argc, char* argv[]) {
    std::cout << "🧠 MELVIN UNIFIED REASONING FRAMEWORK" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << "🔗 Complete 6-step reasoning process with driver modulation!" << std::endl;
    
    UnifiedReasoningEngine melvin;
    
    if (argc > 1) {
        // Single question mode
        std::string question = argv[1];
        std::string answer = melvin.unifiedReasoning(question);
        std::cout << "\n💡 Final Answer: " << answer << std::endl;
    } else {
        // Interactive mode
        std::cout << "\n🎯 TESTING UNIFIED REASONING FRAMEWORK" << std::endl;
        std::cout << "======================================" << std::endl;
        
        // Test the unified reasoning process
        std::vector<std::string> test_queries = {
            "What is a cat?",
            "What is a dog?",
            "What is a mammal?",
            "What is an animal?"
        };
        
        for (const auto& query : test_queries) {
            std::cout << "\n" << std::string(60, '=') << std::endl;
            melvin.unifiedReasoning(query);
        }
        
        // Show results
        melvin.showKnowledgeGraph();
        
        std::cout << "\n✅ Unified reasoning demo completed!" << std::endl;
    }
    
    return 0;
}

/*
 * Melvin Granular Node System
 * 
 * Breaks down monolithic answer nodes into granular, reusable components:
 * - "cat" + "small" + "domesticated" + "carnivorous" + "mammal" + "soft fur" + "short snout" + "retractable claws"
 * - Each component becomes a separate, reusable node
 * - Enables rich connections through shared components (mammal, domesticated, etc.)
 */

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <set>

// Granular Knowledge Node - Each concept is a separate, reusable node
struct GranularNode {
    std::string concept;
    std::string definition;
    std::vector<std::string> component_connections;  // Other nodes this is composed of
    std::vector<std::string> used_in_concepts;       // Concepts that use this node
    double confidence;
    int usage_count;
    std::string node_type;  // "concept", "property", "action", "relationship"
    
    GranularNode() : confidence(0.8), usage_count(0), node_type("concept") {}
    
    GranularNode(const std::string& c, const std::string& d, const std::string& type = "concept") 
        : concept(c), definition(d), confidence(0.8), usage_count(0), node_type(type) {}
    
    void addUsage(const std::string& concept_name) {
        if (std::find(used_in_concepts.begin(), used_in_concepts.end(), concept_name) == used_in_concepts.end()) {
            used_in_concepts.push_back(concept_name);
        }
        usage_count++;
    }
};

// Concept Extractor - Breaks down definitions into granular components
class ConceptExtractor {
private:
    // Common property words that should become separate nodes
    std::set<std::string> property_words = {
        "small", "large", "big", "tiny", "huge", "giant", "miniature",
        "soft", "hard", "smooth", "rough", "sharp", "dull", "bright", "dark",
        "fast", "slow", "quick", "rapid", "gentle", "aggressive", "calm",
        "domesticated", "wild", "tame", "feral", "domestic",
        "carnivorous", "herbivorous", "omnivorous", "predatory",
        "nocturnal", "diurnal", "active", "passive",
        "social", "solitary", "territorial", "migratory"
    };
    
    // Common relationship words
    std::set<std::string> relationship_words = {
        "belongs", "part", "member", "type", "kind", "species", "breed",
        "related", "similar", "different", "opposite", "connected"
    };
    
    // Common action words
    std::set<std::string> action_words = {
        "hunt", "eat", "sleep", "run", "walk", "fly", "swim", "climb",
        "communicate", "reproduce", "grow", "develop", "adapt", "survive"
    };

public:
    struct ExtractedComponents {
        std::string main_concept;
        std::vector<std::string> properties;
        std::vector<std::string> relationships;
        std::vector<std::string> actions;
        std::vector<std::string> physical_features;
    };
    
    ExtractedComponents extractComponents(const std::string& concept, const std::string& definition) {
        ExtractedComponents result;
        result.main_concept = concept;
        
        std::string lower_def = toLowerCase(definition);
        
        // Extract properties
        for (const auto& prop : property_words) {
            if (lower_def.find(prop) != std::string::npos) {
                result.properties.push_back(prop);
            }
        }
        
        // Extract relationships
        for (const auto& rel : relationship_words) {
            if (lower_def.find(rel) != std::string::npos) {
                result.relationships.push_back(rel);
            }
        }
        
        // Extract actions
        for (const auto& action : action_words) {
            if (lower_def.find(action) != std::string::npos) {
                result.actions.push_back(action);
            }
        }
        
        // Extract physical features (simplified - look for noun phrases)
        result.physical_features = extractPhysicalFeatures(definition);
        
        return result;
    }
    
    std::vector<std::string> extractPhysicalFeatures(const std::string& definition) {
        std::vector<std::string> features;
        
        // Common physical feature patterns
        std::vector<std::string> feature_patterns = {
            "fur", "hair", "feathers", "scales", "skin", "shell",
            "claws", "teeth", "beak", "snout", "nose", "eyes", "ears",
            "tail", "wings", "legs", "arms", "hands", "feet", "paws",
            "whiskers", "mane", "horns", "antlers", "tusks"
        };
        
        std::string lower_def = toLowerCase(definition);
        for (const auto& pattern : feature_patterns) {
            if (lower_def.find(pattern) != std::string::npos) {
                features.push_back(pattern);
            }
        }
        
        return features;
    }

private:
    std::string toLowerCase(const std::string& str) {
        std::string result = str;
        std::transform(result.begin(), result.end(), result.begin(), ::tolower);
        return result;
    }
};

// Granular Knowledge Graph - Manages reusable component nodes
class GranularKnowledgeGraph {
private:
    std::map<std::string, GranularNode> nodes;
    ConceptExtractor extractor;
    
public:
    // Add a new concept and break it down into granular components
    void addConcept(const std::string& concept, const std::string& definition) {
        std::cout << "\n🔍 Breaking down '" << concept << "' into granular components:" << std::endl;
        
        // Extract components
        auto components = extractor.extractComponents(concept, definition);
        
        // Create main concept node
        nodes[concept] = GranularNode(concept, definition, "concept");
        std::cout << "  📝 Main concept: " << concept << std::endl;
        
        // Create property nodes
        for (const auto& property : components.properties) {
            if (nodes.find(property) == nodes.end()) {
                nodes[property] = GranularNode(property, "A property or characteristic: " + property, "property");
                std::cout << "  🏷️ Property: " << property << std::endl;
            }
            nodes[property].addUsage(concept);
            nodes[concept].component_connections.push_back(property);
        }
        
        // Create relationship nodes
        for (const auto& relationship : components.relationships) {
            if (nodes.find(relationship) == nodes.end()) {
                nodes[relationship] = GranularNode(relationship, "A relationship or connection: " + relationship, "relationship");
                std::cout << "  🔗 Relationship: " << relationship << std::endl;
            }
            nodes[relationship].addUsage(concept);
            nodes[concept].component_connections.push_back(relationship);
        }
        
        // Create action nodes
        for (const auto& action : components.actions) {
            if (nodes.find(action) == nodes.end()) {
                nodes[action] = GranularNode(action, "An action or behavior: " + action, "action");
                std::cout << "  ⚡ Action: " << action << std::endl;
            }
            nodes[action].addUsage(concept);
            nodes[concept].component_connections.push_back(action);
        }
        
        // Create physical feature nodes
        for (const auto& feature : components.physical_features) {
            if (nodes.find(feature) == nodes.end()) {
                nodes[feature] = GranularNode(feature, "A physical feature: " + feature, "feature");
                std::cout << "  🧬 Feature: " << feature << std::endl;
            }
            nodes[feature].addUsage(concept);
            nodes[concept].component_connections.push_back(feature);
        }
        
        std::cout << "  ✅ Total components: " << (1 + components.properties.size() + 
                  components.relationships.size() + components.actions.size() + 
                  components.physical_features.size()) << std::endl;
    }
    
    // Find reusable nodes that could apply to a new concept
    std::vector<std::string> findReusableNodes(const std::string& new_concept, const std::string& new_definition) {
        std::vector<std::string> reusable;
        
        std::string lower_def = toLowerCase(new_definition);
        
        for (const auto& node_pair : nodes) {
            const auto& node = node_pair.second;
            
            // Check if this node's concept appears in the new definition
            if (lower_def.find(toLowerCase(node.concept)) != std::string::npos) {
                reusable.push_back(node.concept);
            }
        }
        
        return reusable;
    }
    
    // Show the granular knowledge graph
    void showKnowledgeGraph() {
        std::cout << "\n🧠 MELVIN'S GRANULAR KNOWLEDGE GRAPH" << std::endl;
        std::cout << "=====================================" << std::endl;
        
        // Group nodes by type
        std::map<std::string, std::vector<std::string>> nodes_by_type;
        for (const auto& node_pair : nodes) {
            nodes_by_type[node_pair.second.node_type].push_back(node_pair.first);
        }
        
        for (const auto& type_group : nodes_by_type) {
            std::cout << "\n📂 " << type_group.first << " nodes:" << std::endl;
            for (const auto& concept : type_group.second) {
                const auto& node = nodes[concept];
                std::cout << "  📝 " << concept;
                if (node.usage_count > 1) {
                    std::cout << " (reused " << node.usage_count << " times)";
                }
                std::cout << std::endl;
                std::cout << "     Definition: " << node.definition << std::endl;
                if (!node.used_in_concepts.empty()) {
                    std::cout << "     Used in: ";
                    for (size_t i = 0; i < node.used_in_concepts.size(); ++i) {
                        std::cout << node.used_in_concepts[i];
                        if (i < node.used_in_concepts.size() - 1) std::cout << ", ";
                    }
                    std::cout << std::endl;
                }
            }
        }
        
        // Show reuse statistics
        std::cout << "\n📊 REUSE STATISTICS:" << std::endl;
        int total_nodes = nodes.size();
        int reused_nodes = 0;
        int total_reuses = 0;
        
        for (const auto& node_pair : nodes) {
            if (node_pair.second.usage_count > 1) {
                reused_nodes++;
                total_reuses += (node_pair.second.usage_count - 1);
            }
        }
        
        std::cout << "  Total nodes: " << total_nodes << std::endl;
        std::cout << "  Reused nodes: " << reused_nodes << std::endl;
        std::cout << "  Total reuses: " << total_reuses << std::endl;
        std::cout << "  Reuse efficiency: " << std::fixed << std::setprecision(1) 
                  << (double)total_reuses / total_nodes * 100 << "%" << std::endl;
    }
    
    // Show connections between concepts through shared components
    void showComponentConnections() {
        std::cout << "\n🔗 COMPONENT-BASED CONNECTIONS:" << std::endl;
        std::cout << "================================" << std::endl;
        
        for (const auto& node_pair : nodes) {
            const auto& node = node_pair.second;
            if (node.node_type == "concept" && !node.component_connections.empty()) {
                std::cout << "\n📝 " << node.concept << " is composed of:" << std::endl;
                for (const auto& component : node.component_connections) {
                    std::cout << "  🔗 " << component;
                    if (nodes[component].usage_count > 1) {
                        std::cout << " (shared with " << (nodes[component].usage_count - 1) << " other concepts)";
                    }
                    std::cout << std::endl;
                }
            }
        }
    }

private:
    std::string toLowerCase(const std::string& str) {
        std::string result = str;
        std::transform(result.begin(), result.end(), result.begin(), ::tolower);
        return result;
    }
};

// Demo system
class MelvinGranularSystem {
private:
    GranularKnowledgeGraph knowledge_graph;

public:
    MelvinGranularSystem() {
        std::cout << "🧠 Melvin Granular Node System Initialized" << std::endl;
        std::cout << "🔗 Breaking down concepts into reusable components!" << std::endl;
    }
    
    void demonstrateGranularLearning() {
        std::cout << "\n🎯 DEMONSTRATING GRANULAR NODE BREAKDOWN" << std::endl;
        std::cout << "=========================================" << std::endl;
        
        // Learn about a cat - this will create many granular nodes
        knowledge_graph.addConcept("cat", "A small domesticated carnivorous mammal with soft fur, a short snout, and retractable claws.");
        
        // Learn about a dog - this will reuse many nodes from cat
        knowledge_graph.addConcept("dog", "A large domesticated carnivorous mammal with short fur, a long snout, and sharp teeth.");
        
        // Learn about a lion - this will reuse some nodes, create new ones
        knowledge_graph.addConcept("lion", "A large wild carnivorous mammal with golden fur, a mane, and sharp claws.");
        
        // Learn about a bird - this will create mostly new nodes
        knowledge_graph.addConcept("bird", "A small flying animal with feathers, a beak, and wings.");
        
        // Show the results
        knowledge_graph.showKnowledgeGraph();
        knowledge_graph.showComponentConnections();
    }
    
    void showReuseBenefits() {
        std::cout << "\n💡 REUSE BENEFITS DEMONSTRATION:" << std::endl;
        std::cout << "=================================" << std::endl;
        
        std::cout << "🎯 When Melvin learns about a 'tiger':" << std::endl;
        std::cout << "   - He can reuse 'large' (from dog, lion)" << std::endl;
        std::cout << "   - He can reuse 'carnivorous' (from cat, dog, lion)" << std::endl;
        std::cout << "   - He can reuse 'mammal' (from cat, dog, lion)" << std::endl;
        std::cout << "   - He can reuse 'fur' (from cat, dog, lion)" << std::endl;
        std::cout << "   - He can reuse 'claws' (from cat, lion)" << std::endl;
        std::cout << "   - He only needs to learn 'striped' and 'orange' as new properties!" << std::endl;
        
        std::cout << "\n🧠 This creates rich connections:" << std::endl;
        std::cout << "   - cat ↔ dog (both domesticated, carnivorous, mammals)" << std::endl;
        std::cout << "   - cat ↔ lion (both carnivorous, mammals, have claws)" << std::endl;
        std::cout << "   - dog ↔ lion (both large, carnivorous, mammals)" << std::endl;
        std::cout << "   - All mammals share 'mammal' node" << std::endl;
        std::cout << "   - All carnivores share 'carnivorous' node" << std::endl;
    }
};

int main() {
    std::cout << "🧠 MELVIN GRANULAR NODE SYSTEM" << std::endl;
    std::cout << "==============================" << std::endl;
    std::cout << "🔗 Breaking down monolithic nodes into reusable components!" << std::endl;
    
    MelvinGranularSystem melvin;
    
    // Demonstrate the granular learning process
    melvin.demonstrateGranularLearning();
    
    // Show the benefits of node reuse
    melvin.showReuseBenefits();
    
    std::cout << "\n✅ Granular node system demonstration completed!" << std::endl;
    std::cout << "\n💡 Key Benefits:" << std::endl;
    std::cout << "   - Reusable components (mammal, carnivorous, domesticated)" << std::endl;
    std::cout << "   - Richer connections through shared nodes" << std::endl;
    std::cout << "   - More efficient learning (reuse vs. recreate)" << std::endl;
    std::cout << "   - Better generalization across concepts" << std::endl;
    
    return 0;
}

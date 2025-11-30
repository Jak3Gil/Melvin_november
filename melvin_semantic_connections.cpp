#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <memory>
#include <chrono>
#include <cstring>
#include <cctype>
#include <unordered_map>
#include <unordered_set>

// Enhanced Knowledge Node with semantic connections
struct KnowledgeNode {
    uint64_t id;
    char concept[64];
    char definition[512];
    std::vector<uint64_t> connections;
    std::vector<uint64_t> semantic_connections;  // New: semantic relationships
    std::vector<uint64_t> component_connections; // New: word component connections
    char source[32];
    double confidence;
    uint64_t created_at;
    uint64_t last_accessed;
    int access_count;
    
    KnowledgeNode(uint64_t node_id, const std::string& concept_name, const std::string& def) 
        : id(node_id), confidence(0.8), created_at(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()), 
          last_accessed(created_at), access_count(0) {
        strncpy(concept, concept_name.c_str(), sizeof(concept) - 1);
        concept[sizeof(concept) - 1] = '\0';
        strncpy(definition, def.c_str(), sizeof(definition) - 1);
        definition[sizeof(definition) - 1] = '\0';
        strncpy(source, "ollama", sizeof(source) - 1);
        source[sizeof(source) - 1] = '\0';
    }
};

// Semantic Connection Engine
class SemanticConnectionEngine {
private:
    // Common word components for decomposition
    std::set<std::string> common_components = {
        "note", "book", "pad", "paper", "pen", "pencil", "write", "read",
        "computer", "laptop", "desktop", "tablet", "phone", "screen",
        "car", "truck", "bus", "train", "plane", "boat", "ship",
        "house", "home", "building", "room", "door", "window",
        "food", "eat", "drink", "water", "coffee", "tea", "milk",
        "cat", "dog", "bird", "fish", "animal", "pet", "wild",
        "tree", "flower", "plant", "garden", "forest", "park",
        "red", "blue", "green", "yellow", "black", "white", "color",
        "big", "small", "large", "tiny", "huge", "size",
        "fast", "slow", "quick", "rapid", "speed",
        "hot", "cold", "warm", "cool", "temperature",
        "good", "bad", "great", "terrible", "quality"
    };
    
    // Semantic relationship patterns
    std::map<std::string, std::vector<std::string>> semantic_groups = {
        {"writing_tools", {"pen", "pencil", "marker", "crayon", "chalk"}},
        {"containers", {"box", "bag", "bottle", "cup", "bowl", "container"}},
        {"vehicles", {"car", "truck", "bus", "train", "plane", "boat", "ship"}},
        {"animals", {"cat", "dog", "bird", "fish", "lion", "tiger", "elephant"}},
        {"colors", {"red", "blue", "green", "yellow", "black", "white", "purple"}},
        {"sizes", {"big", "small", "large", "tiny", "huge", "miniature"}},
        {"speeds", {"fast", "slow", "quick", "rapid", "sluggish"}},
        {"temperatures", {"hot", "cold", "warm", "cool", "freezing", "boiling"}},
        {"qualities", {"good", "bad", "excellent", "terrible", "great", "awful"}}
    };
    
    // Synonym mappings
    std::map<std::string, std::vector<std::string>> synonyms = {
        {"notebook", {"note book", "writing pad", "journal", "diary"}},
        {"laptop", {"computer", "notebook computer", "portable computer"}},
        {"automobile", {"car", "vehicle", "auto"}},
        {"feline", {"cat", "kitty", "kitten"}},
        {"canine", {"dog", "puppy", "hound"}},
        {"residence", {"house", "home", "dwelling"}},
        {"beverage", {"drink", "liquid"}},
        {"food", {"meal", "cuisine", "nourishment"}}
    };

public:
    // Decompose compound words into components
    std::vector<std::string> decomposeWord(const std::string& word) {
        std::vector<std::string> components;
        std::string lower_word = toLowerCase(word);
        
        // Check if it's a compound word with common separators
        std::vector<std::string> separators = {" ", "-", "_"};
        for (const auto& sep : separators) {
            if (lower_word.find(sep) != std::string::npos) {
                std::istringstream iss(lower_word);
                std::string component;
                while (std::getline(iss, component, sep[0])) {
                    if (!component.empty() && common_components.count(component)) {
                        components.push_back(component);
                    }
                }
                if (!components.empty()) return components;
            }
        }
        
        // Try to find embedded components
        for (const auto& component : common_components) {
            if (lower_word.find(component) != std::string::npos) {
                components.push_back(component);
            }
        }
        
        return components;
    }
    
    // Find semantic relationships between concepts
    std::vector<std::string> findSemanticRelations(const std::string& concept) {
        std::vector<std::string> relations;
        std::string lower_concept = toLowerCase(concept);
        
        // Check semantic groups
        for (const auto& group : semantic_groups) {
            for (const auto& member : group.second) {
                if (lower_concept == member) {
                    // Add all other members of the group
                    for (const auto& other_member : group.second) {
                        if (other_member != member) {
                            relations.push_back(other_member);
                        }
                    }
                    break;
                }
            }
        }
        
        // Check synonyms
        for (const auto& syn_group : synonyms) {
            if (lower_concept == syn_group.first) {
                relations.insert(relations.end(), syn_group.second.begin(), syn_group.second.end());
            } else {
                for (const auto& syn : syn_group.second) {
                    if (lower_concept == syn) {
                        relations.push_back(syn_group.first);
                        for (const auto& other_syn : syn_group.second) {
                            if (other_syn != syn) {
                                relations.push_back(other_syn);
                            }
                        }
                        break;
                    }
                }
            }
        }
        
        return relations;
    }
    
    // Enhanced connection building
    void buildIntelligentConnections(std::shared_ptr<KnowledgeNode> newNode, 
                                   const std::map<uint64_t, std::shared_ptr<KnowledgeNode>>& allNodes) {
        std::string concept = toLowerCase(newNode->concept);
        
        // 1. Component-based connections
        auto components = decomposeWord(concept);
        for (const auto& component : components) {
            for (const auto& node_pair : allNodes) {
                auto existingNode = node_pair.second;
                if (existingNode->id == newNode->id) continue;
                
                std::string existing_concept = toLowerCase(existingNode->concept);
                if (existing_concept == component) {
                    // Create component connection
                    if (std::find(newNode->component_connections.begin(), 
                                newNode->component_connections.end(), 
                                existingNode->id) == newNode->component_connections.end()) {
                        newNode->component_connections.push_back(existingNode->id);
                        std::cout << "🔗 Component connection: " << newNode->concept 
                                 << " → " << existingNode->concept << std::endl;
                    }
                }
            }
        }
        
        // 2. Semantic relationship connections
        auto semantic_relations = findSemanticRelations(concept);
        for (const auto& relation : semantic_relations) {
            for (const auto& node_pair : allNodes) {
                auto existingNode = node_pair.second;
                if (existingNode->id == newNode->id) continue;
                
                std::string existing_concept = toLowerCase(existingNode->concept);
                if (existing_concept == relation) {
                    // Create semantic connection
                    if (std::find(newNode->semantic_connections.begin(), 
                                newNode->semantic_connections.end(), 
                                existingNode->id) == newNode->semantic_connections.end()) {
                        newNode->semantic_connections.push_back(existingNode->id);
                        std::cout << "🧠 Semantic connection: " << newNode->concept 
                                 << " → " << existingNode->concept << std::endl;
                    }
                }
            }
        }
        
        // 3. Definition-based connections (enhanced)
        for (const auto& node_pair : allNodes) {
            auto existingNode = node_pair.second;
            if (existingNode->id == newNode->id) continue;
            
            std::string newDef = toLowerCase(newNode->definition);
            std::string existingDef = toLowerCase(existingNode->definition);
            
            // Check if concepts appear in each other's definitions
            std::string existing_concept = toLowerCase(existingNode->concept);
            if (existingDef.find(concept) != std::string::npos || 
                newDef.find(existing_concept) != std::string::npos) {
                
                if (std::find(newNode->connections.begin(), 
                            newNode->connections.end(), 
                            existingNode->id) == newNode->connections.end()) {
                    newNode->connections.push_back(existingNode->id);
                    std::cout << "📚 Definition connection: " << newNode->concept 
                             << " → " << existingNode->concept << std::endl;
                }
            }
        }
    }
    
    // Enhanced knowledge retrieval with semantic reasoning
    std::string findRelatedKnowledge(const std::string& question, 
                                   const std::map<uint64_t, std::shared_ptr<KnowledgeNode>>& allNodes) {
        std::string lower_question = toLowerCase(question);
        std::string concept = extractConceptFromQuestion(question);
        std::string lower_concept = toLowerCase(concept);
        
        // Direct match
        for (const auto& node_pair : allNodes) {
            auto node = node_pair.second;
            if (toLowerCase(node->concept) == lower_concept) {
                return node->definition;
            }
        }
        
        // Component-based reasoning
        auto components = decomposeWord(lower_concept);
        if (!components.empty()) {
            std::cout << "🔍 Decomposing '" << concept << "' into components: ";
            for (const auto& comp : components) {
                std::cout << comp << " ";
            }
            std::cout << std::endl;
            
            // Try to find knowledge about components
            for (const auto& component : components) {
                for (const auto& node_pair : allNodes) {
                    auto node = node_pair.second;
                    if (toLowerCase(node->concept) == component) {
                        std::cout << "💡 Found component knowledge: " << node->concept << std::endl;
                        return "Based on components: " + std::string(node->definition) + 
                               " (This might relate to " + concept + ")";
                    }
                }
            }
        }
        
        // Semantic relationship reasoning
        auto semantic_relations = findSemanticRelations(lower_concept);
        if (!semantic_relations.empty()) {
            std::cout << "🔍 Found semantic relations for '" << concept << "': ";
            for (const auto& rel : semantic_relations) {
                std::cout << rel << " ";
            }
            std::cout << std::endl;
            
            for (const auto& relation : semantic_relations) {
                for (const auto& node_pair : allNodes) {
                    auto node = node_pair.second;
                    if (toLowerCase(node->concept) == relation) {
                        std::cout << "💡 Found related knowledge: " << node->concept << std::endl;
                        return "Related concept: " + std::string(node->definition) + 
                               " (This is similar to " + concept + ")";
                    }
                }
            }
        }
        
        return "";
    }

private:
    std::string toLowerCase(const std::string& str) {
        std::string result = str;
        std::transform(result.begin(), result.end(), result.begin(), ::tolower);
        return result;
    }
    
    std::string extractConceptFromQuestion(const std::string& question) {
        std::string lower_q = toLowerCase(question);
        
        // Remove common question words
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
            return words[0]; // Return first non-question word
        }
        
        return question;
    }
};

// Enhanced Melvin Learning System with Semantic Connections
class MelvinSemanticLearningSystem {
private:
    std::map<uint64_t, std::shared_ptr<KnowledgeNode>> nodes;
    SemanticConnectionEngine connectionEngine;
    uint64_t next_node_id = 1;
    
    struct LearningStats {
        int questions_asked = 0;
        int new_concepts_learned = 0;
        int concepts_retrieved = 0;
        int component_connections_made = 0;
        int semantic_connections_made = 0;
    } stats;

public:
    MelvinSemanticLearningSystem() {
        std::cout << "🧠 Melvin Semantic Learning System Initialized" << std::endl;
        std::cout << "🔗 Enhanced with intelligent connection building!" << std::endl;
    }
    
    bool melvinKnows(const std::string& question) {
        std::string result = connectionEngine.findRelatedKnowledge(question, nodes);
        return !result.empty();
    }
    
    std::string melvinAnswer(const std::string& question) {
        std::string result = connectionEngine.findRelatedKnowledge(question, nodes);
        if (!result.empty()) {
            stats.concepts_retrieved++;
            return result;
        }
        return "I don't know about that yet.";
    }
    
    std::shared_ptr<KnowledgeNode> createNode(const std::string& concept, const std::string& definition) {
        auto node = std::make_shared<KnowledgeNode>(next_node_id++, concept, definition);
        return node;
    }
    
    void connectToGraph(std::shared_ptr<KnowledgeNode> node) {
        nodes[node->id] = node;
        
        std::cout << "🔗 Building intelligent connections for: " << node->concept << std::endl;
        connectionEngine.buildIntelligentConnections(node, nodes);
        
        stats.new_concepts_learned++;
    }
    
    std::string askOllama(const std::string& question) {
        // Simulated Ollama response
        std::string concept = extractConceptFromQuestion(question);
        return "A " + concept + " is a useful tool or concept that helps people in various ways.";
    }
    
    std::string curiosityLoop(const std::string& question) {
        stats.questions_asked++;
        
        std::cout << "\n🤔 Melvin is thinking about: " << question << std::endl;
        
        if (melvinKnows(question)) {
            std::cout << "🧠 Melvin knows this! Retrieving from memory..." << std::endl;
            return melvinAnswer(question);
        }
        
        std::cout << "❓ Melvin doesn't know this. Asking Ollama tutor..." << std::endl;
        std::string ollamaResponse = askOllama(question);
        
        std::string concept = extractConceptFromQuestion(question);
        std::string definition = ollamaResponse;
        
        std::cout << "📚 Creating new knowledge node for: " << concept << std::endl;
        auto node = createNode(concept, definition);
        
        std::cout << "🔗 Building intelligent connections..." << std::endl;
        connectToGraph(node);
        
        std::cout << "✅ Melvin learned something new!" << std::endl;
        return definition;
    }
    
    void showLearningStats() {
        std::cout << "\n📊 MELVIN'S SEMANTIC LEARNING STATISTICS" << std::endl;
        std::cout << "==========================================" << std::endl;
        std::cout << "Total Concepts: " << nodes.size() << std::endl;
        std::cout << "Questions Asked: " << stats.questions_asked << std::endl;
        std::cout << "New Concepts Learned: " << stats.new_concepts_learned << std::endl;
        std::cout << "Concepts Retrieved: " << stats.concepts_retrieved << std::endl;
        std::cout << "Component Connections: " << stats.component_connections_made << std::endl;
        std::cout << "Semantic Connections: " << stats.semantic_connections_made << std::endl;
        std::cout << "==========================================" << std::endl;
    }
    
    void showKnowledgeGraph() {
        std::cout << "\n🧠 MELVIN'S KNOWLEDGE GRAPH" << std::endl;
        std::cout << "============================" << std::endl;
        
        for (const auto& node_pair : nodes) {
            auto node = node_pair.second;
            std::cout << "\n📝 " << node->concept << " (ID: " << node->id << ")" << std::endl;
            std::cout << "   Definition: " << node->definition << std::endl;
            
            if (!node->component_connections.empty()) {
                std::cout << "   🔗 Component connections: ";
                for (auto conn_id : node->component_connections) {
                    if (nodes.count(conn_id)) {
                        std::cout << nodes[conn_id]->concept << " ";
                    }
                }
                std::cout << std::endl;
            }
            
            if (!node->semantic_connections.empty()) {
                std::cout << "   🧠 Semantic connections: ";
                for (auto conn_id : node->semantic_connections) {
                    if (nodes.count(conn_id)) {
                        std::cout << nodes[conn_id]->concept << " ";
                    }
                }
                std::cout << std::endl;
            }
            
            if (!node->connections.empty()) {
                std::cout << "   📚 Definition connections: ";
                for (auto conn_id : node->connections) {
                    if (nodes.count(conn_id)) {
                        std::cout << nodes[conn_id]->concept << " ";
                    }
                }
                std::cout << std::endl;
            }
        }
    }

private:
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

int main() {
    std::cout << "🧠 MELVIN SEMANTIC CONNECTION DEMO" << std::endl;
    std::cout << "===================================" << std::endl;
    
    MelvinSemanticLearningSystem melvin;
    
    // Test the notebook/note book problem
    std::cout << "\n🎯 TESTING NOTEBOOK/NOTE BOOK CONNECTION" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    // First, teach Melvin about "note" and "book"
    std::cout << "\n📚 Teaching Melvin about basic concepts..." << std::endl;
    melvin.curiosityLoop("What is a note?");
    melvin.curiosityLoop("What is a book?");
    
    // Now ask about "notebook" - should connect to note + book
    std::cout << "\n🔍 Now asking about 'notebook'..." << std::endl;
    std::string result1 = melvin.curiosityLoop("What is a notebook?");
    std::cout << "Result: " << result1 << std::endl;
    
    // Ask about "note book" - should find the connection
    std::cout << "\n🔍 Now asking about 'note book'..." << std::endl;
    std::string result2 = melvin.curiosityLoop("What is a note book?");
    std::cout << "Result: " << result2 << std::endl;
    
    // Show the knowledge graph
    melvin.showKnowledgeGraph();
    
    // Show learning statistics
    melvin.showLearningStats();
    
    std::cout << "\n✅ Demo completed! Melvin now understands connections!" << std::endl;
    
    return 0;
}

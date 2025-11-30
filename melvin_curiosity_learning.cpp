/*
 * Melvin Curiosity Learning System - C++ Implementation
 * 
 * A curiosity-driven learning module for Melvin that learns by asking questions
 * when he doesn't know something, and uses Ollama as his tutor.
 * 
 * Features:
 * - Binary storage for nodes (no JSON)
 * - Curiosity-tutor loop
 * - Knowledge graph with connections
 * - Persistent learning across sessions
 * - Pure C++ implementation
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

// Node structure for knowledge graph
struct KnowledgeNode {
    uint64_t id;
    char concept[64];
    char definition[512];
    std::vector<uint64_t> connections;
    char source[32];
    double confidence;
    uint64_t created_at;
    uint64_t last_accessed;
    uint32_t access_count;
    
    KnowledgeNode() : id(0), confidence(0.8), created_at(0), last_accessed(0), access_count(0) {
        memset(concept, 0, sizeof(concept));
        memset(definition, 0, sizeof(definition));
        memset(source, 0, sizeof(source));
    }
    
    KnowledgeNode(uint64_t node_id, const std::string& node_concept, const std::string& node_definition)
        : id(node_id), confidence(0.8), created_at(getCurrentTime()), last_accessed(getCurrentTime()), access_count(0) {
        
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

// Binary storage for knowledge graph
class BinaryKnowledgeStorage {
private:
    std::string storage_file;
    std::unordered_map<uint64_t, std::shared_ptr<KnowledgeNode>> nodes;
    std::unordered_map<std::string, uint64_t> concept_index;
    uint64_t next_node_id;
    
public:
    BinaryKnowledgeStorage(const std::string& filename = "melvin_knowledge.bin") 
        : storage_file(filename), next_node_id(1) {
        loadKnowledge();
    }
    
    ~BinaryKnowledgeStorage() {
        saveKnowledge();
    }
    
    void loadKnowledge() {
        std::ifstream file(storage_file, std::ios::binary);
        if (!file.is_open()) {
            std::cout << "🧠 No existing knowledge found. Starting fresh." << std::endl;
            return;
        }
        
        // Read header
        uint32_t node_count;
        file.read(reinterpret_cast<char*>(&node_count), sizeof(node_count));
        
        std::cout << "🧠 Loading " << node_count << " knowledge nodes from " << storage_file << std::endl;
        
        for (uint32_t i = 0; i < node_count; i++) {
            auto node = std::make_shared<KnowledgeNode>();
            
            // Read node data
            file.read(reinterpret_cast<char*>(&node->id), sizeof(node->id));
            file.read(node->concept, sizeof(node->concept));
            file.read(node->definition, sizeof(node->definition));
            file.read(node->source, sizeof(node->source));
            file.read(reinterpret_cast<char*>(&node->confidence), sizeof(node->confidence));
            file.read(reinterpret_cast<char*>(&node->created_at), sizeof(node->created_at));
            file.read(reinterpret_cast<char*>(&node->last_accessed), sizeof(node->last_accessed));
            file.read(reinterpret_cast<char*>(&node->access_count), sizeof(node->access_count));
            
            // Read connections
            uint32_t connection_count;
            file.read(reinterpret_cast<char*>(&connection_count), sizeof(connection_count));
            node->connections.resize(connection_count);
            if (connection_count > 0) {
                file.read(reinterpret_cast<char*>(node->connections.data()), 
                         connection_count * sizeof(uint64_t));
            }
            
            nodes[node->id] = node;
            concept_index[node->concept] = node->id;
            next_node_id = std::max(next_node_id, node->id + 1);
        }
        
        file.close();
        std::cout << "✅ Knowledge loaded successfully!" << std::endl;
    }
    
    void saveKnowledge() {
        std::ofstream file(storage_file, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "❌ Error: Cannot save knowledge to " << storage_file << std::endl;
            return;
        }
        
        // Write header
        uint32_t node_count = static_cast<uint32_t>(nodes.size());
        file.write(reinterpret_cast<const char*>(&node_count), sizeof(node_count));
        
        // Write nodes
        for (const auto& pair : nodes) {
            const auto& node = pair.second;
            
            file.write(reinterpret_cast<const char*>(&node->id), sizeof(node->id));
            file.write(node->concept, sizeof(node->concept));
            file.write(node->definition, sizeof(node->definition));
            file.write(node->source, sizeof(node->source));
            file.write(reinterpret_cast<const char*>(&node->confidence), sizeof(node->confidence));
            file.write(reinterpret_cast<const char*>(&node->created_at), sizeof(node->created_at));
            file.write(reinterpret_cast<const char*>(&node->last_accessed), sizeof(node->last_accessed));
            file.write(reinterpret_cast<const char*>(&node->access_count), sizeof(node->access_count));
            
            // Write connections
            uint32_t connection_count = static_cast<uint32_t>(node->connections.size());
            file.write(reinterpret_cast<const char*>(&connection_count), sizeof(connection_count));
            if (connection_count > 0) {
                file.write(reinterpret_cast<const char*>(node->connections.data()), 
                          connection_count * sizeof(uint64_t));
            }
        }
        
        file.close();
        std::cout << "💾 Saved " << node_count << " knowledge nodes to " << storage_file << std::endl;
    }
    
    std::shared_ptr<KnowledgeNode> findConcept(const std::string& concept) {
        auto it = concept_index.find(concept);
        if (it != concept_index.end()) {
            auto node = nodes[it->second];
            node->last_accessed = KnowledgeNode::getCurrentTime();
            node->access_count++;
            return node;
        }
        return nullptr;
    }
    
    void addNode(std::shared_ptr<KnowledgeNode> node) {
        nodes[node->id] = node;
        concept_index[node->concept] = node->id;
        std::cout << "➕ Added concept: " << node->concept << std::endl;
    }
    
    uint64_t getNextNodeId() {
        return next_node_id++;
    }
    
    size_t getNodeCount() const {
        return nodes.size();
    }
    
    std::vector<std::shared_ptr<KnowledgeNode>> getAllNodes() const {
        std::vector<std::shared_ptr<KnowledgeNode>> result;
        for (const auto& pair : nodes) {
            result.push_back(pair.second);
        }
        return result;
    }
};

// Ollama tutor simulation
class OllamaTutor {
private:
    std::map<std::string, std::string> knowledge_base;
    
public:
    OllamaTutor() {
        // Initialize with some basic knowledge
        knowledge_base["cat"] = "A cat is a small domesticated carnivorous mammal with soft fur, a short snout, and retractable claws. Cats are popular pets and are known for their independence and hunting abilities.";
        knowledge_base["dog"] = "A dog is a domesticated carnivorous mammal that is commonly kept as a pet. Dogs are known for their loyalty, intelligence, and ability to be trained for various tasks.";
        knowledge_base["bird"] = "A bird is a warm-blooded vertebrate animal with feathers, wings, and a beak. Most birds can fly, and they lay hard-shelled eggs.";
        knowledge_base["fish"] = "A fish is a cold-blooded aquatic vertebrate animal with gills and fins. Fish live in water and breathe through their gills.";
        knowledge_base["tree"] = "A tree is a perennial plant with an elongated stem, or trunk, supporting branches and leaves. Trees are important for producing oxygen and providing habitat for wildlife.";
        knowledge_base["car"] = "A car is a wheeled motor vehicle used for transportation. Cars typically have four wheels and are powered by an internal combustion engine or electric motor.";
        knowledge_base["computer"] = "A computer is an electronic device that processes data according to instructions. Computers can perform calculations, store information, and communicate with other devices.";
        knowledge_base["book"] = "A book is a written or printed work consisting of pages bound together. Books contain information, stories, or other content and are used for education and entertainment.";
        knowledge_base["house"] = "A house is a building designed for people to live in. Houses provide shelter and typically contain rooms for sleeping, cooking, and other activities.";
        knowledge_base["water"] = "Water is a transparent, odorless, tasteless liquid that is essential for life. Water covers about 71% of Earth's surface and is vital for all living organisms.";
    }
    
    std::string askOllama(const std::string& question) {
        std::string concept = extractConceptFromQuestion(question);
        std::transform(concept.begin(), concept.end(), concept.begin(), ::tolower);
        
        auto it = knowledge_base.find(concept);
        if (it != knowledge_base.end()) {
            return it->second;
        }
        
        return "I don't have specific information about that topic, but I can help you learn more. Could you provide more context about what you'd like to know?";
    }
    
private:
    std::string extractConceptFromQuestion(const std::string& question) {
        std::vector<std::string> words;
        std::string word;
        
        // Simple word extraction
        for (char c : question) {
            if (std::isalpha(c)) {
                word += std::tolower(c);
            } else if (!word.empty()) {
                words.push_back(word);
                word.clear();
            }
        }
        if (!word.empty()) {
            words.push_back(word);
        }
        
        // Remove common question words
        std::vector<std::string> question_words = {"what", "is", "a", "an", "the", "are", "do", "does", "how", "why", "when", "where", "tell", "me", "about"};
        
        // Look for patterns like "what is X"
        if (words.size() >= 3) {
            if (words[0] == "what" && words[1] == "is") {
                std::string concept = words[2];
                if (concept == "a" || concept == "an" || concept == "the") {
                    if (words.size() > 3) {
                        concept = words[3];
                    }
                }
                return concept;
            }
        }
        
        // Look for "what's X" pattern
        if (words.size() >= 2) {
            if (words[0] == "whats") {
                std::string concept = words[1];
                if (concept == "a" || concept == "an" || concept == "the") {
                    if (words.size() > 2) {
                        concept = words[2];
                    }
                }
                return concept;
            }
        }
        
        // Find first non-question word
        for (const std::string& w : words) {
            if (std::find(question_words.begin(), question_words.end(), w) == question_words.end() && w.length() > 2) {
                return w;
            }
        }
        
        // Fallback: last word
        return words.empty() ? "unknown" : words.back();
    }
};

// Main Melvin learning system
class MelvinLearningSystem {
private:
    BinaryKnowledgeStorage storage;
    OllamaTutor tutor;
    
    struct LearningStats {
        uint32_t questions_asked;
        uint32_t new_concepts_learned;
        uint32_t concepts_retrieved;
        
        LearningStats() : questions_asked(0), new_concepts_learned(0), concepts_retrieved(0) {}
    } stats;
    
public:
    MelvinLearningSystem() : storage("melvin_knowledge.bin") {
        std::cout << "🤖 Melvin Learning System Initialized" << std::endl;
        std::cout << "==================================================" << std::endl;
    }
    
    bool melvinKnows(const std::string& question) {
        std::string concept = extractConceptFromQuestion(question);
        return storage.findConcept(concept) != nullptr;
    }
    
    std::string melvinAnswer(const std::string& question) {
        std::string concept = extractConceptFromQuestion(question);
        auto node = storage.findConcept(concept);
        
        if (node) {
            stats.concepts_retrieved++;
            return node->definition;
        }
        
        return "I don't know the answer to that question.";
    }
    
    std::string askOllama(const std::string& question) {
        return tutor.askOllama(question);
    }
    
    std::shared_ptr<KnowledgeNode> createNode(const std::string& concept, const std::string& definition) {
        uint64_t id = storage.getNextNodeId();
        auto node = std::make_shared<KnowledgeNode>(id, concept, definition);
        return node;
    }
    
    void connectToGraph(std::shared_ptr<KnowledgeNode> node) {
        // Add the node
        storage.addNode(node);
        
        // Find related concepts and create connections
        auto allNodes = storage.getAllNodes();
        for (auto& existingNode : allNodes) {
            if (existingNode->id != node->id) {
                // Simple connection logic - if concepts appear in each other's definitions
                std::string nodeDef = node->definition;
                std::string existingDef = existingNode->definition;
                std::transform(nodeDef.begin(), nodeDef.end(), nodeDef.begin(), ::tolower);
                std::transform(existingDef.begin(), existingDef.end(), existingDef.begin(), ::tolower);
                
                if (existingDef.find(node->concept) != std::string::npos || 
                    nodeDef.find(existingNode->concept) != std::string::npos) {
                    
                    // Add bidirectional connections
                    if (std::find(node->connections.begin(), node->connections.end(), existingNode->id) == node->connections.end()) {
                        node->connections.push_back(existingNode->id);
                    }
                    if (std::find(existingNode->connections.begin(), existingNode->connections.end(), node->id) == existingNode->connections.end()) {
                        existingNode->connections.push_back(node->id);
                    }
                }
            }
        }
        
        stats.new_concepts_learned++;
    }
    
    std::string curiosityLoop(const std::string& question) {
        stats.questions_asked++;
        
        std::cout << "🤔 Melvin is thinking about: " << question << std::endl;
        
        // Check if Melvin already knows
        if (melvinKnows(question)) {
            std::cout << "🧠 Melvin knows this! Retrieving from memory..." << std::endl;
            return melvinAnswer(question);
        }
        
        // Melvin doesn't know - ask Ollama
        std::cout << "❓ Melvin doesn't know this. Asking Ollama tutor..." << std::endl;
        std::string ollamaResponse = askOllama(question);
        
        // Extract concept and definition
        std::string concept = extractConceptFromQuestion(question);
        std::string definition = ollamaResponse;
        
        // Create new knowledge node
        std::cout << "📚 Creating new knowledge node for: " << concept << std::endl;
        auto node = createNode(concept, definition);
        
        // Connect to existing knowledge
        std::cout << "🔗 Connecting to existing knowledge..." << std::endl;
        connectToGraph(node);
        
        // Save knowledge
        storage.saveKnowledge();
        
        std::cout << "✅ Melvin learned something new!" << std::endl;
        return definition;
    }
    
    void showLearningStats() {
        std::cout << "\n📊 MELVIN'S LEARNING STATISTICS" << std::endl;
        std::cout << "==========================================" << std::endl;
        std::cout << "Total Concepts: " << storage.getNodeCount() << std::endl;
        std::cout << "Questions Asked: " << stats.questions_asked << std::endl;
        std::cout << "New Concepts Learned: " << stats.new_concepts_learned << std::endl;
        std::cout << "Concepts Retrieved: " << stats.concepts_retrieved << std::endl;
        
        auto allNodes = storage.getAllNodes();
        if (!allNodes.empty()) {
            std::cout << "\n🧠 Recent Knowledge:" << std::endl;
            
            // Sort by creation time (most recent first)
            std::sort(allNodes.begin(), allNodes.end(), 
                     [](const std::shared_ptr<KnowledgeNode>& a, const std::shared_ptr<KnowledgeNode>& b) {
                         return a->created_at > b->created_at;
                     });
            
            size_t displayCount = std::min(allNodes.size(), size_t(5));
            for (size_t i = 0; i < displayCount; i++) {
                const auto& node = allNodes[i];
                std::string shortDef = node->definition;
                if (shortDef.length() > 50) {
                    shortDef = shortDef.substr(0, 47) + "...";
                }
                std::cout << "  • " << node->concept << ": " << shortDef << std::endl;
            }
        }
        std::cout << "==========================================" << std::endl;
    }
    
private:
    std::string extractConceptFromQuestion(const std::string& question) {
        std::vector<std::string> words;
        std::string word;
        
        // Simple word extraction
        for (char c : question) {
            if (std::isalpha(c)) {
                word += std::tolower(c);
            } else if (!word.empty()) {
                words.push_back(word);
                word.clear();
            }
        }
        if (!word.empty()) {
            words.push_back(word);
        }
        
        // Remove common question words
        std::vector<std::string> question_words = {"what", "is", "a", "an", "the", "are", "do", "does", "how", "why", "when", "where", "tell", "me", "about"};
        
        // Look for patterns like "what is X"
        if (words.size() >= 3) {
            if (words[0] == "what" && words[1] == "is") {
                std::string concept = words[2];
                if (concept == "a" || concept == "an" || concept == "the") {
                    if (words.size() > 3) {
                        concept = words[3];
                    }
                }
                return concept;
            }
        }
        
        // Look for "what's X" pattern
        if (words.size() >= 2) {
            if (words[0] == "whats") {
                std::string concept = words[1];
                if (concept == "a" || concept == "an" || concept == "the") {
                    if (words.size() > 2) {
                        concept = words[2];
                    }
                }
                return concept;
            }
        }
        
        // Find first non-question word
        for (const std::string& w : words) {
            if (std::find(question_words.begin(), question_words.end(), w) == question_words.end() && w.length() > 2) {
                return w;
            }
        }
        
        // Fallback: last word
        return words.empty() ? "unknown" : words.back();
    }
};

// Main function
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " \"What is a cat?\"" << std::endl;
        std::cout << "Example: " << argv[0] << " \"What is a dog?\"" << std::endl;
        return 1;
    }
    
    std::string question = argv[1];
    
    // Initialize Melvin's learning system
    MelvinLearningSystem melvin;
    
    // Run the curiosity loop
    std::string answer = melvin.curiosityLoop(question);
    
    std::cout << "\n🎯 Answer: " << answer << std::endl;
    
    // Show learning stats
    melvin.showLearningStats();
    
    // Interactive mode
    std::cout << "\n🔄 Interactive Mode (type 'quit' to exit, 'stats' for summary)" << std::endl;
    std::string userInput;
    
    while (true) {
        std::cout << "\nAsk Melvin: ";
        std::getline(std::cin, userInput);
        
        if (userInput == "quit" || userInput == "exit" || userInput == "q") {
            break;
        } else if (userInput == "stats") {
            melvin.showLearningStats();
        } else if (!userInput.empty()) {
            std::string answer = melvin.curiosityLoop(userInput);
            std::cout << "🎯 Answer: " << answer << std::endl;
        }
    }
    
    std::cout << "\n👋 Goodbye!" << std::endl;
    return 0;
}

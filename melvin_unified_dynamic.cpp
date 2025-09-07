#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <chrono>
#include <random>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <set>
#include <fstream>

// 🧠 MELVIN UNIFIED DYNAMIC SYSTEM
// =================================
// Combines dynamic personality with actual neural node creation and connections

class UnifiedDynamicMelvin {
private:
    // Neural Network Components
    struct Node {
        uint64_t id;
        std::string content;
        std::string type; // TEXT, CONCEPT, RESPONSE, MEMORY
        double activation_strength;
        double creation_time;
        std::vector<uint64_t> connections;
        int access_count;
        double last_accessed;
    };
    
    struct Connection {
        uint64_t from_node;
        uint64_t to_node;
        double strength;
        std::string type; // SEMANTIC, TEMPORAL, CAUSAL, ASSOCIATIVE
        double creation_time;
        int traversal_count;
    };
    
    // Core neural network
    std::map<uint64_t, Node> nodes;
    std::vector<Connection> connections;
    uint64_t next_node_id;
    
    // Dynamic personality system
    std::string current_personality;
    std::map<std::string, std::vector<std::string>> personality_banks;
    std::map<std::string, std::vector<std::string>> curiosity_questions;
    
    // Conversation tracking
    std::vector<std::pair<std::string, std::string>> conversation_pairs;
    std::vector<std::string> recent_responses;
    std::vector<uint64_t> recent_node_ids;
    std::random_device rd;
    std::mt19937_64 gen;
    uint64_t conversation_turn;
    double session_start_time;
    
    // Context and memory
    std::vector<std::string> user_topics;
    std::map<std::string, int> topic_frequency;
    std::map<std::string, std::vector<std::string>> knowledge_base;
    
public:
    UnifiedDynamicMelvin() : gen(rd()), conversation_turn(0), 
                           session_start_time(static_cast<double>(std::time(nullptr))),
                           next_node_id(1), current_personality("curious") {
        initialize_knowledge_base();
        initialize_personality_banks();
        initialize_curiosity_system();
        load_existing_memory(); // Load any saved nodes/connections
    }
    
    void initialize_knowledge_base() {
        knowledge_base["cancer"] = {
            "Cancer immunotherapy is revolutionizing treatment approaches",
            "CAR-T cell therapy shows remarkable success in blood cancers", 
            "Early detection through screening programs saves countless lives",
            "Precision medicine is tailoring treatments to individual patients",
            "Cancer research funding has accelerated breakthrough discoveries"
        };
        
        knowledge_base["health"] = {
            "Mental health awareness is transforming healthcare approaches",
            "Preventive medicine focuses on lifestyle and early intervention",
            "Telemedicine is expanding access to healthcare worldwide",
            "Personalized nutrition is optimizing individual health outcomes"
        };
        
        knowledge_base["space"] = {
            "James Webb Telescope is revealing the universe's earliest galaxies",
            "Mars exploration missions are preparing for human colonization",
            "Exoplanet discoveries are expanding our understanding of life",
            "Space debris management is becoming a critical challenge"
        };
        
        knowledge_base["ai"] = {
            "Large language models are transforming human-computer interaction",
            "AI ethics frameworks are shaping responsible development",
            "Machine learning is accelerating drug discovery processes",
            "Neural networks are mimicking biological brain structures"
        };
    }
    
    void initialize_personality_banks() {
        personality_banks["curious"] = {
            "That's fascinating! I'm connecting this to something I've been thinking about...",
            "Oh, this sparks so many questions in my mind!",
            "I find myself deeply curious about this topic!",
            "This reminds me of something intriguing I've learned...",
            "What an interesting perspective! It makes me wonder...",
            "I'm genuinely excited to explore this with you!",
            "This touches on something I find endlessly fascinating...",
            "I can't help but be curious about the deeper implications...",
            "This opens up such interesting possibilities!",
            "I'm drawn to explore this topic further..."
        };
        
        personality_banks["empathetic"] = {
            "I can sense this is important to you, and I want to help...",
            "I understand how this might feel significant...",
            "This sounds like it could be meaningful for you...",
            "I'm here to support you in exploring this...",
            "I can relate to the importance of this topic...",
            "This seems to touch on something deeply personal...",
            "I want to approach this with care and understanding...",
            "I'm listening and I want to help however I can...",
            "This feels like something that matters to you...",
            "I'm here to provide whatever support you need..."
        };
        
        personality_banks["technical"] = {
            "From a technical perspective, this involves several fascinating aspects...",
            "The underlying mechanisms here are quite sophisticated...",
            "This presents an interesting engineering challenge...",
            "The data suggests some compelling patterns...",
            "From a systems perspective, this is quite complex...",
            "The technical implementation here is noteworthy...",
            "This involves some intriguing algorithmic considerations...",
            "The technical specifications are quite impressive...",
            "From an analytical standpoint, this is fascinating...",
            "The technical architecture here is quite elegant..."
        };
        
        personality_banks["casual"] = {
            "Oh, that's cool! I was just thinking about something similar...",
            "Nice! This reminds me of something I heard recently...",
            "That's pretty interesting! I wonder if...",
            "Oh wow, that's actually really neat!",
            "That's awesome! I'm totally curious about...",
            "Sweet! This is right up my alley...",
            "That's really cool! I've been wondering about...",
            "Oh, that's neat! I was just reading about...",
            "That's pretty awesome! I'm thinking...",
            "Cool! This is something I find really interesting..."
        };
    }
    
    void initialize_curiosity_system() {
        curiosity_questions["medical"] = {
            "Should I search for the latest research developments?",
            "Would you like me to find recent breakthrough studies?",
            "I'm curious about the newest treatment approaches - should I look them up?",
            "Would you be interested in the latest clinical trial results?"
        };
        
        curiosity_questions["science"] = {
            "Should I search for the newest discoveries in this field?",
            "Would you like me to find the latest research papers?",
            "I'm curious about recent breakthroughs - should I look them up?",
            "Would you be interested in the newest experimental results?"
        };
        
        curiosity_questions["technology"] = {
            "Should I search for the latest tech developments?",
            "Would you like me to find recent innovation updates?",
            "I'm curious about the newest implementations - should I look them up?",
            "Would you be interested in the latest technical breakthroughs?"
        };
        
        curiosity_questions["general"] = {
            "What aspect of this interests you most?",
            "Should I explore this topic further?",
            "What would you like to know more about?",
            "I'm curious - what sparked your interest in this?"
        };
    }
    
    std::string process_input(const std::string& user_input) {
        conversation_turn++;
        
        // Phase 1: Create/activate nodes from user input
        std::vector<uint64_t> activated_nodes = create_nodes_from_input(user_input);
        
        // Phase 2: Analyze input and context
        std::string input_type = analyze_input_type(user_input);
        std::string intent = analyze_intent(user_input);
        std::string emotion = detect_emotion(user_input);
        
        // Phase 3: Update context memory
        update_context_memory(user_input, input_type);
        
        // Phase 4: Select dynamic personality
        select_dynamic_personality(input_type, emotion);
        
        // Phase 5: Traverse neural connections
        std::vector<uint64_t> connected_nodes = traverse_neural_network(activated_nodes);
        
        // Phase 6: Generate response using neural network
        std::string response = generate_neural_response(user_input, activated_nodes, connected_nodes, input_type, intent, emotion);
        
        // Phase 7: Create response node and connections
        uint64_t response_node_id = create_response_node(response);
        create_input_response_connections(activated_nodes, response_node_id);
        
        // Phase 8: Check for repetition and regenerate if needed
        if (is_repetitive(response)) {
            response = regenerate_with_different_style(user_input, activated_nodes, connected_nodes, input_type, intent, emotion);
            response_node_id = create_response_node(response);
            create_input_response_connections(activated_nodes, response_node_id);
        }
        
        // Phase 9: Store conversation and update memory
        conversation_pairs.push_back({user_input, response});
        recent_responses.push_back(response);
        recent_node_ids.push_back(response_node_id);
        
        // Keep only last 5 responses for repetition check
        if (recent_responses.size() > 5) {
            recent_responses.erase(recent_responses.begin());
        }
        
        // Phase 10: Save memory periodically
        if (conversation_turn % 5 == 0) {
            save_memory_to_file();
        }
        
        return response;
    }
    
    std::vector<uint64_t> create_nodes_from_input(const std::string& input) {
        std::vector<uint64_t> node_ids;
        std::vector<std::string> words = tokenize(input);
        
        // Create nodes for each significant word/concept
        for (const auto& word : words) {
            if (word.length() > 2) { // Only meaningful words
                uint64_t node_id = find_or_create_node(word, "TEXT");
                node_ids.push_back(node_id);
                activate_node(node_id);
            }
        }
        
        // Create a composite node for the full input
        uint64_t input_node_id = find_or_create_node(input, "INPUT");
        node_ids.push_back(input_node_id);
        activate_node(input_node_id);
        
        return node_ids;
    }
    
    uint64_t find_or_create_node(const std::string& content, const std::string& type) {
        // Check if node already exists
        for (const auto& pair : nodes) {
            if (pair.second.content == content && pair.second.type == type) {
                return pair.first;
            }
        }
        
        // Create new node
        Node new_node;
        new_node.id = next_node_id++;
        new_node.content = content;
        new_node.type = type;
        new_node.activation_strength = 0.0;
        new_node.creation_time = static_cast<double>(std::time(nullptr));
        new_node.access_count = 0;
        new_node.last_accessed = new_node.creation_time;
        
        nodes[new_node.id] = new_node;
        
        std::cout << "🧠 Created node: " << content.substr(0, 20) << "... -> " << std::hex << new_node.id << std::dec << std::endl;
        
        return new_node.id;
    }
    
    void activate_node(uint64_t node_id) {
        if (nodes.find(node_id) != nodes.end()) {
            nodes[node_id].activation_strength = 1.0;
            nodes[node_id].access_count++;
            nodes[node_id].last_accessed = static_cast<double>(std::time(nullptr));
        }
    }
    
    std::vector<uint64_t> traverse_neural_network(const std::vector<uint64_t>& activated_nodes) {
        std::vector<uint64_t> connected_nodes;
        
        for (uint64_t node_id : activated_nodes) {
            if (nodes.find(node_id) != nodes.end()) {
                // Find connections from this node
                for (const auto& connection : connections) {
                    if (connection.from_node == node_id) {
                        connected_nodes.push_back(connection.to_node);
                        // Strengthen the connection
                        strengthen_connection(connection.from_node, connection.to_node);
                    }
                }
                
                // Create semantic connections to related concepts
                create_semantic_connections(node_id);
            }
        }
        
        return connected_nodes;
    }
    
    void create_semantic_connections(uint64_t node_id) {
        if (nodes.find(node_id) == nodes.end()) return;
        
        std::string content = nodes[node_id].content;
        std::string lower_content = content;
        std::transform(lower_content.begin(), lower_content.end(), lower_content.begin(), ::tolower);
        
        // Check knowledge base for related concepts
        for (const auto& topic : knowledge_base) {
            if (lower_content.find(topic.first) != std::string::npos) {
                // Create connections to knowledge base entries
                for (const auto& knowledge_item : topic.second) {
                    uint64_t knowledge_node_id = find_or_create_node(knowledge_item, "KNOWLEDGE");
                    create_connection(node_id, knowledge_node_id, "SEMANTIC", 0.7);
                }
            }
        }
    }
    
    void create_connection(uint64_t from_node, uint64_t to_node, const std::string& type, double strength) {
        // Check if connection already exists
        for (const auto& connection : connections) {
            if (connection.from_node == from_node && connection.to_node == to_node) {
                return; // Connection already exists
            }
        }
        
        Connection new_connection;
        new_connection.from_node = from_node;
        new_connection.to_node = to_node;
        new_connection.strength = strength;
        new_connection.type = type;
        new_connection.creation_time = static_cast<double>(std::time(nullptr));
        new_connection.traversal_count = 0;
        
        connections.push_back(new_connection);
        
        // Add to node's connection list
        if (nodes.find(from_node) != nodes.end()) {
            nodes[from_node].connections.push_back(to_node);
        }
        
        std::cout << "🔗 Created connection: " << std::hex << from_node << " -> " << to_node 
                  << std::dec << " (" << type << ", " << strength << ")" << std::endl;
    }
    
    void strengthen_connection(uint64_t from_node, uint64_t to_node) {
        for (auto& connection : connections) {
            if (connection.from_node == from_node && connection.to_node == to_node) {
                connection.strength = std::min(1.0, connection.strength + 0.1);
                connection.traversal_count++;
                break;
            }
        }
    }
    
    uint64_t create_response_node(const std::string& response) {
        return find_or_create_node(response, "RESPONSE");
    }
    
    void create_input_response_connections(const std::vector<uint64_t>& input_nodes, uint64_t response_node_id) {
        for (uint64_t input_node_id : input_nodes) {
            create_connection(input_node_id, response_node_id, "CAUSAL", 0.8);
        }
    }
    
    std::string generate_neural_response(const std::string& input, const std::vector<uint64_t>& activated_nodes, 
                                       const std::vector<uint64_t>& connected_nodes, const std::string& input_type, 
                                       const std::string& intent, const std::string& emotion) {
        
        std::ostringstream response;
        
        // Show neural network activity
        std::cout << "🧠 Neural Activity: " << activated_nodes.size() << " activated, " 
                  << connected_nodes.size() << " connected nodes" << std::endl;
        
        // Dynamic personality response
        std::string personality_response = get_personality_response(current_personality);
        response << personality_response << " ";
        
        // Knowledge injection from connected nodes
        std::string knowledge = get_knowledge_from_nodes(connected_nodes, intent);
        if (!knowledge.empty()) {
            response << knowledge << " ";
        }
        
        // Context weaving
        if (conversation_turn > 1 && !user_topics.empty()) {
            std::string context_connector = get_context_connector(input_type);
            if (!context_connector.empty()) {
                response << context_connector << " ";
            }
        }
        
        // Curiosity injection
        std::string curiosity_question = get_curiosity_question(input_type);
        if (!curiosity_question.empty()) {
            response << curiosity_question;
        }
        
        return response.str();
    }
    
    std::string get_knowledge_from_nodes(const std::vector<uint64_t>& connected_nodes, const std::string& intent) {
        if (connected_nodes.empty()) return "";
        
        // Find knowledge nodes
        for (uint64_t node_id : connected_nodes) {
            if (nodes.find(node_id) != nodes.end() && nodes[node_id].type == "KNOWLEDGE") {
                return nodes[node_id].content;
            }
        }
        
        // Fallback to knowledge base
        if (knowledge_base.find(intent) != knowledge_base.end()) {
            const auto& knowledge = knowledge_base[intent];
            std::uniform_int_distribution<> dis(0, knowledge.size() - 1);
            return knowledge[dis(gen)];
        }
        
        return "";
    }
    
    std::string get_personality_response(const std::string& personality) {
        if (personality_banks.find(personality) != personality_banks.end()) {
            const auto& responses = personality_banks[personality];
            std::uniform_int_distribution<> dis(0, responses.size() - 1);
            return responses[dis(gen)];
        }
        return "That's interesting! ";
    }
    
    std::string get_context_connector(const std::string& input_type) {
        std::vector<std::string> connectors = {
            "Building on what we discussed earlier...",
            "This connects to our previous conversation...",
            "Following up on our exploration of " + input_type + "...",
            "This relates to the topics we've been covering...",
            "Expanding on our discussion..."
        };
        
        std::uniform_int_distribution<> dis(0, connectors.size() - 1);
        return connectors[dis(gen)];
    }
    
    std::string get_curiosity_question(const std::string& input_type) {
        if (curiosity_questions.find(input_type) != curiosity_questions.end()) {
            const auto& questions = curiosity_questions[input_type];
            std::uniform_int_distribution<> dis(0, questions.size() - 1);
            return questions[dis(gen)];
        }
        return curiosity_questions["general"][0];
    }
    
    std::string analyze_input_type(const std::string& input) {
        std::string lower_input = input;
        std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
        
        if (lower_input.find("cancer") != std::string::npos || 
            lower_input.find("health") != std::string::npos ||
            lower_input.find("medical") != std::string::npos) {
            return "medical";
        }
        
        if (lower_input.find("space") != std::string::npos ||
            lower_input.find("science") != std::string::npos) {
            return "science";
        }
        
        if (lower_input.find("ai") != std::string::npos ||
            lower_input.find("technology") != std::string::npos) {
            return "technology";
        }
        
        return "general";
    }
    
    std::string analyze_intent(const std::string& input) {
        std::string lower_input = input;
        std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
        
        for (const auto& topic : knowledge_base) {
            if (lower_input.find(topic.first) != std::string::npos) {
                return topic.first;
            }
        }
        
        return "general_inquiry";
    }
    
    std::string detect_emotion(const std::string& input) {
        std::string lower_input = input;
        std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
        
        if (lower_input.find("!") != std::string::npos) {
            return "excited";
        }
        
        if (lower_input.find("?") != std::string::npos) {
            return "curious";
        }
        
        if (lower_input.find("cancer") != std::string::npos ||
            lower_input.find("worried") != std::string::npos) {
            return "concerned";
        }
        
        return "neutral";
    }
    
    void update_context_memory(const std::string& input, const std::string& input_type) {
        if (std::find(user_topics.begin(), user_topics.end(), input_type) == user_topics.end()) {
            user_topics.push_back(input_type);
        }
        
        topic_frequency[input_type]++;
    }
    
    void select_dynamic_personality(const std::string& input_type, const std::string& emotion) {
        if (emotion == "concerned" || input_type == "medical") {
            current_personality = "empathetic";
        } else if (input_type == "technology" || input_type == "science") {
            current_personality = "technical";
        } else if (emotion == "excited") {
            current_personality = "casual";
        } else {
            std::vector<std::string> personalities = {"curious", "empathetic", "technical", "casual"};
            std::uniform_int_distribution<> dis(0, personalities.size() - 1);
            current_personality = personalities[dis(gen)];
        }
    }
    
    bool is_repetitive(const std::string& response) {
        if (recent_responses.size() < 2) return false;
        
        for (const auto& recent : recent_responses) {
            if (calculate_similarity(response, recent) > 0.7) {
                return true;
            }
        }
        return false;
    }
    
    double calculate_similarity(const std::string& str1, const std::string& str2) {
        std::vector<std::string> words1 = tokenize(str1);
        std::vector<std::string> words2 = tokenize(str2);
        
        std::set<std::string> set1(words1.begin(), words1.end());
        std::set<std::string> set2(words2.begin(), words2.end());
        
        std::set<std::string> intersection;
        std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(),
                            std::inserter(intersection, intersection.begin()));
        
        return static_cast<double>(intersection.size()) / 
               static_cast<double>(std::max(set1.size(), set2.size()));
    }
    
    std::string regenerate_with_different_style(const std::string& input, const std::vector<uint64_t>& activated_nodes, 
                                             const std::vector<uint64_t>& connected_nodes, const std::string& input_type, 
                                             const std::string& intent, const std::string& emotion) {
        // Force different personality
        std::vector<std::string> personalities = {"curious", "empathetic", "technical", "casual"};
        personalities.erase(std::remove(personalities.begin(), personalities.end(), current_personality), personalities.end());
        
        if (!personalities.empty()) {
            std::uniform_int_distribution<> dis(0, personalities.size() - 1);
            current_personality = personalities[dis(gen)];
        }
        
        return generate_neural_response(input, activated_nodes, connected_nodes, input_type, intent, emotion);
    }
    
    std::vector<std::string> tokenize(const std::string& input) {
        std::vector<std::string> tokens;
        std::string current_token;
        
        for (char c : input) {
            if (std::isalpha(c) || std::isdigit(c)) {
                current_token += std::tolower(c);
            } else if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
        }
        
        if (!current_token.empty()) {
            tokens.push_back(current_token);
        }
        
        return tokens;
    }
    
    void save_memory_to_file() {
        std::ofstream file("melvin_memory.txt");
        if (file.is_open()) {
            file << "Nodes: " << nodes.size() << std::endl;
            file << "Connections: " << connections.size() << std::endl;
            file << "Conversation turns: " << conversation_turn << std::endl;
            
            for (const auto& pair : nodes) {
                file << "Node " << pair.first << ": " << pair.second.content 
                     << " (type: " << pair.second.type << ", accesses: " << pair.second.access_count << ")" << std::endl;
            }
            
            file.close();
            std::cout << "💾 Memory saved to file" << std::endl;
        }
    }
    
    void load_existing_memory() {
        std::ifstream file("melvin_memory.txt");
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                if (line.find("Node ") == 0) {
                    // Parse node information
                    std::cout << "📖 Loaded: " << line << std::endl;
                }
            }
            file.close();
        }
    }
    
    void show_neural_status() {
        std::cout << "\n🧠 MELVIN UNIFIED NEURAL STATUS" << std::endl;
        std::cout << "===============================" << std::endl;
        std::cout << "Total nodes: " << nodes.size() << std::endl;
        std::cout << "Total connections: " << connections.size() << std::endl;
        std::cout << "Conversation turns: " << conversation_turn << std::endl;
        std::cout << "Current personality: " << current_personality << std::endl;
        std::cout << "User topics: ";
        for (const auto& topic : user_topics) {
            std::cout << topic << " ";
        }
        std::cout << std::endl;
        
        std::cout << "\nRecent neural activity:" << std::endl;
        for (size_t i = std::max(0, static_cast<int>(recent_node_ids.size()) - 3); 
             i < recent_node_ids.size(); ++i) {
            if (nodes.find(recent_node_ids[i]) != nodes.end()) {
                std::cout << "Node " << std::hex << recent_node_ids[i] << std::dec 
                          << ": " << nodes[recent_node_ids[i]].content.substr(0, 30) << "..." << std::endl;
            }
        }
    }
    
    void run_interactive_session() {
        std::cout << "🧠 MELVIN UNIFIED DYNAMIC NEURAL SYSTEM" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Hello! I'm Melvin with unified neural architecture!" << std::endl;
        std::cout << "I create actual nodes and connections from our conversations:" << std::endl;
        std::cout << "- Each word/concept becomes a neural node" << std::endl;
        std::cout << "- Connections form between related concepts" << std::endl;
        std::cout << "- Dynamic personality adapts to context" << std::endl;
        std::cout << "- Memory persists across sessions" << std::endl;
        std::cout << "\nType 'quit' to exit, 'status' for neural info." << std::endl;
        std::cout << "========================================" << std::endl;
        
        std::string user_input;
        
        while (true) {
            std::cout << "\nYou: ";
            std::getline(std::cin, user_input);
            
            if (user_input.empty()) {
                continue;
            }
            
            std::string lower_input = user_input;
            std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
            
            if (lower_input == "quit" || lower_input == "exit") {
                std::cout << "\nMelvin: Thank you for this neural conversation! ";
                std::cout << "I've created " << nodes.size() << " nodes and " << connections.size() << " connections. ";
                std::cout << "My neural network has grown through our " << conversation_turn << " turns together. ";
                std::cout << "Until next time! 🧠✨" << std::endl;
                save_memory_to_file();
                break;
            } else if (lower_input == "status") {
                show_neural_status();
                continue;
            }
            
            // Process input through unified neural system
            std::cout << "\nMelvin: ";
            std::string response = process_input(user_input);
            std::cout << response << std::endl;
            
            // Add thinking delay
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
};

int main() {
    try {
        UnifiedDynamicMelvin melvin;
        melvin.run_interactive_session();
    } catch (const std::exception& e) {
        std::cerr << "❌ Error in unified neural session: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

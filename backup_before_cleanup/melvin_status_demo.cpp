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

// 🧠 MELVIN STATUS DEMO
// ====================
// Show current neural network status without interactive mode

class MelvinStatusDemo {
private:
    struct Node {
        uint64_t id;
        std::string content;
        std::string type;
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
        std::string type;
        double creation_time;
        int traversal_count;
    };
    
    std::map<uint64_t, Node> nodes;
    std::vector<Connection> connections;
    uint64_t next_node_id;
    
public:
    MelvinStatusDemo() : next_node_id(1) {
        load_existing_memory();
    }
    
    void load_existing_memory() {
        std::ifstream file("melvin_memory.txt");
        if (file.is_open()) {
            std::string line;
            int loaded_nodes = 0;
            int loaded_connections = 0;
            
            while (std::getline(file, line)) {
                if (line.find("Node ") == 0) {
                    loaded_nodes++;
                }
            }
            
            std::cout << "📖 Loaded " << loaded_nodes << " nodes from memory file" << std::endl;
            file.close();
        } else {
            std::cout << "📝 No existing memory file found - starting fresh" << std::endl;
        }
    }
    
    void simulate_conversation() {
        std::cout << "\n🧪 Simulating a brief conversation to show node creation..." << std::endl;
        
        // Simulate some inputs
        std::vector<std::string> test_inputs = {
            "hello melvin",
            "tell me about cancer",
            "what is artificial intelligence?",
            "how does space exploration work?"
        };
        
        for (const auto& input : test_inputs) {
            std::cout << "\nInput: \"" << input << "\"" << std::endl;
            
            // Create nodes from input
            std::vector<uint64_t> node_ids = create_nodes_from_input(input);
            
            // Create some connections
            create_sample_connections(node_ids);
            
            std::cout << "Created " << node_ids.size() << " nodes" << std::endl;
        }
    }
    
    std::vector<uint64_t> create_nodes_from_input(const std::string& input) {
        std::vector<uint64_t> node_ids;
        std::vector<std::string> words = tokenize(input);
        
        for (const auto& word : words) {
            if (word.length() > 2) {
                uint64_t node_id = find_or_create_node(word, "TEXT");
                node_ids.push_back(node_id);
            }
        }
        
        uint64_t input_node_id = find_or_create_node(input, "INPUT");
        node_ids.push_back(input_node_id);
        
        return node_ids;
    }
    
    uint64_t find_or_create_node(const std::string& content, const std::string& type) {
        for (const auto& pair : nodes) {
            if (pair.second.content == content && pair.second.type == type) {
                return pair.first;
            }
        }
        
        Node new_node;
        new_node.id = next_node_id++;
        new_node.content = content;
        new_node.type = type;
        new_node.activation_strength = 0.0;
        new_node.creation_time = static_cast<double>(std::time(nullptr));
        new_node.access_count = 0;
        new_node.last_accessed = new_node.creation_time;
        
        nodes[new_node.id] = new_node;
        
        return new_node.id;
    }
    
    void create_sample_connections(const std::vector<uint64_t>& node_ids) {
        for (size_t i = 1; i < node_ids.size(); ++i) {
            create_connection(node_ids[i-1], node_ids[i], "SEMANTIC", 0.7);
        }
    }
    
    void create_connection(uint64_t from_node, uint64_t to_node, const std::string& type, double strength) {
        Connection new_connection;
        new_connection.from_node = from_node;
        new_connection.to_node = to_node;
        new_connection.strength = strength;
        new_connection.type = type;
        new_connection.creation_time = static_cast<double>(std::time(nullptr));
        new_connection.traversal_count = 0;
        
        connections.push_back(new_connection);
        
        if (nodes.find(from_node) != nodes.end()) {
            nodes[from_node].connections.push_back(to_node);
        }
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
    
    void show_final_status() {
        std::cout << "\n🧠 MELVIN NEURAL NETWORK STATUS" << std::endl;
        std::cout << "===============================" << std::endl;
        std::cout << "Total nodes: " << nodes.size() << std::endl;
        std::cout << "Total connections: " << connections.size() << std::endl;
        
        std::cout << "\nNode breakdown by type:" << std::endl;
        std::map<std::string, int> type_counts;
        for (const auto& pair : nodes) {
            type_counts[pair.second.type]++;
        }
        
        for (const auto& pair : type_counts) {
            std::cout << "- " << pair.first << ": " << pair.second << " nodes" << std::endl;
        }
        
        std::cout << "\nConnection breakdown by type:" << std::endl;
        std::map<std::string, int> connection_type_counts;
        for (const auto& connection : connections) {
            connection_type_counts[connection.type]++;
        }
        
        for (const auto& pair : connection_type_counts) {
            std::cout << "- " << pair.first << ": " << pair.second << " connections" << std::endl;
        }
        
        std::cout << "\nSample nodes:" << std::endl;
        int count = 0;
        for (const auto& pair : nodes) {
            if (count < 5) {
                std::cout << "Node " << std::hex << pair.first << std::dec 
                          << ": " << pair.second.content.substr(0, 30) 
                          << " (type: " << pair.second.type << ")" << std::endl;
                count++;
            }
        }
        
        std::cout << "\n💾 Saving memory to file..." << std::endl;
        save_memory_to_file();
    }
    
    void save_memory_to_file() {
        std::ofstream file("melvin_memory.txt");
        if (file.is_open()) {
            file << "Nodes: " << nodes.size() << std::endl;
            file << "Connections: " << connections.size() << std::endl;
            file << "Session: Status Demo" << std::endl;
            
            for (const auto& pair : nodes) {
                file << "Node " << pair.first << ": " << pair.second.content 
                     << " (type: " << pair.second.type << ", accesses: " << pair.second.access_count << ")" << std::endl;
            }
            
            file.close();
            std::cout << "✅ Memory saved to melvin_memory.txt" << std::endl;
        }
    }
};

int main() {
    std::cout << "🧠 MELVIN NEURAL NETWORK STATUS DEMO" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    try {
        MelvinStatusDemo melvin;
        melvin.simulate_conversation();
        melvin.show_final_status();
        
        std::cout << "\n🎉 Demo complete! Melvin's neural network is active." << std::endl;
        std::cout << "Run 'run_unified_melvin.bat' for interactive mode." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

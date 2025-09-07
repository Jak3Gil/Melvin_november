#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cstdint>

// 🧠 MELVIN BINARY MEMORY READER
// ==============================
// Reads and counts nodes and connections from binary files

class MelvinBinaryReader {
private:
    struct NodeHeader {
        uint64_t id;
        uint32_t content_length;
        uint32_t type_length;
        uint32_t connections_count;
        double creation_time;
        double last_accessed;
        int access_count;
    };
    
    struct ConnectionHeader {
        uint64_t from_node;
        uint64_t to_node;
        double strength;
        uint32_t type_length;
        double creation_time;
        int traversal_count;
    };
    
public:
    MelvinBinaryReader() {}
    
    void read_and_count() {
        std::cout << "🧠 MELVIN BINARY MEMORY READER" << std::endl;
        std::cout << "==============================" << std::endl;
        
        // Read nodes
        int node_count = read_nodes();
        
        // Read connections
        int connection_count = read_connections();
        
        // Display results
        std::cout << "\n📊 FINAL COUNTS:" << std::endl;
        std::cout << "=================" << std::endl;
        std::cout << "Total Nodes: " << node_count << std::endl;
        std::cout << "Total Connections: " << connection_count << std::endl;
        std::cout << "Network Density: " << (connection_count > 0 ? (double)connection_count / node_count : 0.0) << " connections per node" << std::endl;
        
        // File size analysis
        analyze_file_sizes();
    }
    
    int read_nodes() {
        std::ifstream file("melvin_binary_memory/nodes.bin", std::ios::binary);
        if (!file.is_open()) {
            std::cout << "❌ Cannot open nodes.bin file" << std::endl;
            return 0;
        }
        
        std::cout << "\n📖 Reading nodes.bin..." << std::endl;
        
        int node_count = 0;
        std::map<std::string, int> type_counts;
        
        while (file.good() && !file.eof()) {
            NodeHeader header;
            file.read(reinterpret_cast<char*>(&header), sizeof(NodeHeader));
            
            if (file.gcount() < sizeof(NodeHeader)) {
                break; // End of file or incomplete header
            }
            
            // Read content
            std::string content(header.content_length, '\0');
            file.read(&content[0], header.content_length);
            
            // Read type
            std::string type(header.type_length, '\0');
            file.read(&type[0], header.type_length);
            
            // Skip connections data (we'll count them separately)
            file.seekg(header.connections_count * sizeof(uint64_t), std::ios::cur);
            
            node_count++;
            type_counts[type]++;
            
            if (node_count <= 5) {
                std::cout << "Node " << std::hex << header.id << std::dec 
                          << ": " << content.substr(0, 30) << "... (type: " << type << ")" << std::endl;
            }
        }
        
        file.close();
        
        std::cout << "✅ Read " << node_count << " nodes" << std::endl;
        std::cout << "Node types:" << std::endl;
        for (const auto& pair : type_counts) {
            std::cout << "  - " << pair.first << ": " << pair.second << std::endl;
        }
        
        return node_count;
    }
    
    int read_connections() {
        std::ifstream file("melvin_binary_memory/connections.bin", std::ios::binary);
        if (!file.is_open()) {
            std::cout << "❌ Cannot open connections.bin file" << std::endl;
            return 0;
        }
        
        std::cout << "\n🔗 Reading connections.bin..." << std::endl;
        
        int connection_count = 0;
        std::map<std::string, int> type_counts;
        
        while (file.good() && !file.eof()) {
            ConnectionHeader header;
            file.read(reinterpret_cast<char*>(&header), sizeof(ConnectionHeader));
            
            if (file.gcount() < sizeof(ConnectionHeader)) {
                break; // End of file or incomplete header
            }
            
            // Read connection type
            std::string type(header.type_length, '\0');
            file.read(&type[0], header.type_length);
            
            connection_count++;
            type_counts[type]++;
            
            if (connection_count <= 5) {
                std::cout << "Connection " << std::hex << header.from_node 
                          << " -> " << header.to_node << std::dec 
                          << " (type: " << type << ", strength: " << header.strength << ")" << std::endl;
            }
        }
        
        file.close();
        
        std::cout << "✅ Read " << connection_count << " connections" << std::endl;
        std::cout << "Connection types:" << std::endl;
        for (const auto& pair : type_counts) {
            std::cout << "  - " << pair.first << ": " << pair.second << std::endl;
        }
        
        return connection_count;
    }
    
    void analyze_file_sizes() {
        std::cout << "\n📊 FILE SIZE ANALYSIS:" << std::endl;
        std::cout << "======================" << std::endl;
        
        std::ifstream nodes_file("melvin_binary_memory/nodes.bin", std::ios::binary);
        std::ifstream connections_file("melvin_binary_memory/connections.bin", std::ios::binary);
        
        if (nodes_file.is_open()) {
            nodes_file.seekg(0, std::ios::end);
            size_t nodes_size = nodes_file.tellg();
            nodes_file.close();
            std::cout << "nodes.bin: " << nodes_size << " bytes" << std::endl;
        }
        
        if (connections_file.is_open()) {
            connections_file.seekg(0, std::ios::end);
            size_t connections_size = connections_file.tellg();
            connections_file.close();
            std::cout << "connections.bin: " << connections_size << " bytes" << std::endl;
        }
    }
    
    // Alternative method: Try to parse as simple format
    void try_simple_parsing() {
        std::cout << "\n🔍 TRYING SIMPLE PARSING:" << std::endl;
        std::cout << "=========================" << std::endl;
        
        // Try reading as simple binary format
        std::ifstream file("melvin_binary_memory/nodes.bin", std::ios::binary);
        if (!file.is_open()) {
            std::cout << "❌ Cannot open nodes.bin file" << std::endl;
            return;
        }
        
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::cout << "File size: " << file_size << " bytes" << std::endl;
        
        // Try different node sizes
        std::vector<int> possible_sizes = {32, 64, 128, 256, 512};
        
        for (int node_size : possible_sizes) {
            int estimated_nodes = file_size / node_size;
            std::cout << "If nodes are " << node_size << " bytes each: ~" << estimated_nodes << " nodes" << std::endl;
        }
        
        file.close();
    }
};

int main() {
    try {
        MelvinBinaryReader reader;
        reader.read_and_count();
        reader.try_simple_parsing();
        
        std::cout << "\n🎯 SUMMARY:" << std::endl;
        std::cout << "===========" << std::endl;
        std::cout << "Melvin's neural network contains nodes and connections stored in binary format." << std::endl;
        std::cout << "The exact count depends on the internal data structure format." << std::endl;
        std::cout << "Run the unified Melvin system to see live node creation!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error reading binary files: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

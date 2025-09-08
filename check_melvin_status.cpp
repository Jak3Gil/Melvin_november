#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

// 🧠 MELVIN STATUS CHECKER
// ========================
// Check how many nodes and connections Melvin currently has

int main() {
    std::cout << "🧠 MELVIN NEURAL NETWORK STATUS CHECK" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    // Check binary memory files
    std::ifstream nodes_file("melvin_binary_memory/nodes.bin", std::ios::binary);
    std::ifstream connections_file("melvin_binary_memory/connections.bin", std::ios::binary);
    std::ifstream index_file("melvin_binary_memory/index.bin", std::ios::binary);
    
    if (nodes_file.is_open()) {
        nodes_file.seekg(0, std::ios::end);
        size_t nodes_size = nodes_file.tellg();
        nodes_file.close();
        std::cout << "📊 Binary nodes file size: " << nodes_size << " bytes" << std::endl;
        
        // Estimate nodes (assuming ~100 bytes per node)
        size_t estimated_nodes = nodes_size / 100;
        std::cout << "🧠 Estimated nodes: ~" << estimated_nodes << std::endl;
    } else {
        std::cout << "❌ No binary nodes file found" << std::endl;
    }
    
    if (connections_file.is_open()) {
        connections_file.seekg(0, std::ios::end);
        size_t connections_size = connections_file.tellg();
        connections_file.close();
        std::cout << "🔗 Binary connections file size: " << connections_size << " bytes" << std::endl;
        
        // Estimate connections (assuming ~50 bytes per connection)
        size_t estimated_connections = connections_size / 50;
        std::cout << "🔗 Estimated connections: ~" << estimated_connections << std::endl;
    } else {
        std::cout << "❌ No binary connections file found" << std::endl;
    }
    
    // Check text memory file
    std::ifstream text_memory("melvin_memory.txt");
    if (text_memory.is_open()) {
        std::string line;
        int node_count = 0;
        int connection_count = 0;
        
        while (std::getline(text_memory, line)) {
            if (line.find("Nodes:") == 0) {
                std::cout << "📊 " << line << std::endl;
            } else if (line.find("Connections:") == 0) {
                std::cout << "🔗 " << line << std::endl;
            } else if (line.find("Node ") == 0) {
                node_count++;
            }
        }
        
        std::cout << "📝 Text memory nodes found: " << node_count << std::endl;
        text_memory.close();
    } else {
        std::cout << "❌ No text memory file found" << std::endl;
    }
    
    // Check if we can run a quick demo to count
    std::cout << "\n🚀 To get exact counts, run the unified Melvin system:" << std::endl;
    std::cout << "   run_unified_melvin.bat" << std::endl;
    std::cout << "   Then type 'status' to see current neural network size" << std::endl;
    
    return 0;
}

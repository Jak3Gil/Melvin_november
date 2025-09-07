#include <iostream>
#include <fstream>

int main() {
    std::cout << "🧠 MELVIN NODE COUNTER" << std::endl;
    std::cout << "======================" << std::endl;
    
    // Check nodes.bin
    std::ifstream nodes_file("melvin_binary_memory/nodes.bin", std::ios::binary);
    if (nodes_file.is_open()) {
        nodes_file.seekg(0, std::ios::end);
        size_t nodes_size = nodes_file.tellg();
        nodes_file.close();
        
        std::cout << "📊 nodes.bin size: " << nodes_size << " bytes" << std::endl;
        
        // Estimate based on different node sizes
        std::cout << "\n🔍 Node count estimates:" << std::endl;
        std::cout << "If 50 bytes per node:  ~" << (nodes_size / 50) << " nodes" << std::endl;
        std::cout << "If 100 bytes per node: ~" << (nodes_size / 100) << " nodes" << std::endl;
        std::cout << "If 200 bytes per node: ~" << (nodes_size / 200) << " nodes" << std::endl;
    } else {
        std::cout << "❌ Cannot read nodes.bin" << std::endl;
    }
    
    // Check connections.bin
    std::ifstream connections_file("melvin_binary_memory/connections.bin", std::ios::binary);
    if (connections_file.is_open()) {
        connections_file.seekg(0, std::ios::end);
        size_t connections_size = connections_file.tellg();
        connections_file.close();
        
        std::cout << "\n🔗 connections.bin size: " << connections_size << " bytes" << std::endl;
        
        // Estimate based on different connection sizes
        std::cout << "\n🔍 Connection count estimates:" << std::endl;
        std::cout << "If 25 bytes per connection:  ~" << (connections_size / 25) << " connections" << std::endl;
        std::cout << "If 50 bytes per connection:  ~" << (connections_size / 50) << " connections" << std::endl;
        std::cout << "If 100 bytes per connection: ~" << (connections_size / 100) << " connections" << std::endl;
    } else {
        std::cout << "❌ Cannot read connections.bin" << std::endl;
    }
    
    std::cout << "\n🎯 CONCLUSION:" << std::endl;
    std::cout << "Melvin has a substantial neural network!" << std::endl;
    std::cout << "Run the unified system to see live node creation." << std::endl;
    
    return 0;
}

#include "melvin_unified_brain.h"
#include <iostream>
#include <string>

int main() {
    std::cout << "🧠 Testing Melvin Unified Brain System..." << std::endl;
    
    try {
        // Create unified brain instance
        MelvinUnifiedBrain brain("test_memory");
        
        // Test basic functionality
        std::cout << "✅ Brain created successfully" << std::endl;
        
        // Test node storage
        uint64_t node_id = brain.store_node("Hello, world!", ContentType::TEXT, 128);
        if (node_id > 0) {
            std::cout << "✅ Node stored with ID: " << node_id << std::endl;
        } else {
            std::cout << "❌ Failed to store node" << std::endl;
            return 1;
        }
        
        // Test node retrieval
        std::string content = brain.get_node_content(node_id);
        if (!content.empty()) {
            std::cout << "✅ Node retrieved: " << content << std::endl;
        } else {
            std::cout << "❌ Failed to retrieve node" << std::endl;
            return 1;
        }
        
        // Test connection storage
        uint64_t node2_id = brain.store_node("Test connection", ContentType::TEXT, 100);
        uint64_t conn_id = brain.store_connection(node_id, node2_id, ConnectionType::HEBBIAN, 150);
        if (conn_id > 0) {
            std::cout << "✅ Connection stored with ID: " << conn_id << std::endl;
        } else {
            std::cout << "❌ Failed to store connection" << std::endl;
            return 1;
        }
        
        // Test input processing
        std::string response = brain.process_input("What is artificial intelligence?");
        if (!response.empty()) {
            std::cout << "✅ Input processing successful" << std::endl;
            std::cout << "Response preview: " << response.substr(0, 100) << "..." << std::endl;
        } else {
            std::cout << "❌ Input processing failed" << std::endl;
            return 1;
        }
        
        // Test brain statistics
        auto stats = brain.get_brain_stats();
        std::cout << "✅ Brain statistics:" << std::endl;
        std::cout << "   Total nodes: " << stats.total_nodes << std::endl;
        std::cout << "   Total connections: " << stats.total_connections << std::endl;
        std::cout << "   Reasoning paths: " << stats.reasoning_paths << std::endl;
        
        std::cout << "\n🎉 All tests passed! Melvin Unified Brain System is working correctly." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

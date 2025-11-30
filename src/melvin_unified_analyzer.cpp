/*
 * Melvin Unified System Analyzer
 * 
 * Analyzes the contents of the unified brain files
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <cstring>

// Binary structures (must match the unified system)
struct BinaryWordConnection {
    char word1[64];
    char word2[64];
    int count;
    char context[512];
    double weight;
    char connection_type[32];
};

struct BinaryNode {
    char id[64];
    char type[32];
    char content[256];
    double activation;
    double importance;
    int connections_count;
};

struct BinaryEdge {
    char from_node_id[64];
    char to_node_id[64];
    char type[32];
    double weight;
    char context[128];
    int access_count;
};

int main() {
    std::cout << "🔍 MELVIN UNIFIED SYSTEM ANALYSIS" << std::endl;
    std::cout << "=================================" << std::endl;
    std::cout << std::endl;
    
    // Analyze nodes
    std::cout << "📊 NODE ANALYSIS" << std::endl;
    std::cout << "================" << std::endl;
    
    std::ifstream nodes_file("melvin_unified_nodes.bin", std::ios::binary);
    if (nodes_file.is_open()) {
        BinaryNode node;
        int node_count = 0;
        std::map<std::string, int> node_types;
        
        while (nodes_file.read(reinterpret_cast<char*>(&node), sizeof(BinaryNode))) {
            node_count++;
            node_types[node.type]++;
            
            std::cout << "Node " << node_count << ":" << std::endl;
            std::cout << "  ID: '" << node.id << "'" << std::endl;
            std::cout << "  Type: '" << node.type << "'" << std::endl;
            std::cout << "  Content: '" << node.content << "'" << std::endl;
            std::cout << "  Activation: " << node.activation << std::endl;
            std::cout << "  Importance: " << node.importance << std::endl;
            std::cout << "  Connections: " << node.connections_count << std::endl;
            std::cout << std::endl;
        }
        
        std::cout << "Total Nodes: " << node_count << std::endl;
        std::cout << "Node Types:" << std::endl;
        for (const auto& type_pair : node_types) {
            std::cout << "  " << type_pair.first << ": " << type_pair.second << std::endl;
        }
        nodes_file.close();
    } else {
        std::cout << "❌ Could not open nodes file" << std::endl;
    }
    
    std::cout << std::endl;
    
    // Analyze edges
    std::cout << "🔗 EDGE ANALYSIS" << std::endl;
    std::cout << "================" << std::endl;
    
    std::ifstream edges_file("melvin_unified_edges.bin", std::ios::binary);
    if (edges_file.is_open()) {
        BinaryEdge edge;
        int edge_count = 0;
        std::map<std::string, int> edge_types;
        
        while (edges_file.read(reinterpret_cast<char*>(&edge), sizeof(BinaryEdge))) {
            edge_count++;
            edge_types[edge.type]++;
            
            std::cout << "Edge " << edge_count << ":" << std::endl;
            std::cout << "  From: '" << edge.from_node_id << "'" << std::endl;
            std::cout << "  To: '" << edge.to_node_id << "'" << std::endl;
            std::cout << "  Type: '" << edge.type << "'" << std::endl;
            std::cout << "  Weight: " << edge.weight << std::endl;
            std::cout << "  Access Count: " << edge.access_count << std::endl;
            std::cout << std::endl;
        }
        
        std::cout << "Total Edges: " << edge_count << std::endl;
        std::cout << "Edge Types:" << std::endl;
        for (const auto& type_pair : edge_types) {
            std::cout << "  " << type_pair.first << ": " << type_pair.second << std::endl;
        }
        edges_file.close();
    } else {
        std::cout << "❌ Could not open edges file" << std::endl;
    }
    
    std::cout << std::endl;
    
    // Analyze word connections
    std::cout << "🔤 WORD CONNECTION ANALYSIS" << std::endl;
    std::cout << "===========================" << std::endl;
    
    std::ifstream connections_file("melvin_unified_connections.bin", std::ios::binary);
    if (connections_file.is_open()) {
        BinaryWordConnection conn;
        int conn_count = 0;
        std::map<std::string, int> connection_types;
        
        while (connections_file.read(reinterpret_cast<char*>(&conn), sizeof(BinaryWordConnection))) {
            conn_count++;
            connection_types[conn.connection_type]++;
            
            std::cout << "Connection " << conn_count << ":" << std::endl;
            std::cout << "  Word1: '" << conn.word1 << "'" << std::endl;
            std::cout << "  Word2: '" << conn.word2 << "'" << std::endl;
            std::cout << "  Count: " << conn.count << std::endl;
            std::cout << "  Weight: " << conn.weight << std::endl;
            std::cout << "  Type: '" << conn.connection_type << "'" << std::endl;
            std::cout << "  Context: '" << std::string(conn.context, 0, 100) << "..." << std::endl;
            std::cout << std::endl;
        }
        
        std::cout << "Total Word Connections: " << conn_count << std::endl;
        std::cout << "Connection Types:" << std::endl;
        for (const auto& type_pair : connection_types) {
            std::cout << "  " << type_pair.first << ": " << type_pair.second << std::endl;
        }
        connections_file.close();
    } else {
        std::cout << "❌ Could not open connections file" << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "✅ Analysis complete!" << std::endl;
    
    return 0;
}

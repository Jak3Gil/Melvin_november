#pragma once
#include <string>
#include <vector>
#include <map>
#include <set>
#include <chrono>
#include <fstream>
#include <mutex>

namespace melvin {

// Node types
enum class NodeType {
    TEXT = 0,
    CONCEPT = 1,
    QUESTION = 2,
    ANSWER = 3,
    SEARCH_RESULT = 4,
    CONNECTION = 5
};

// Connection types
enum class ConnectionType {
    SEMANTIC = 0,
    TEMPORAL = 1,
    ASSOCIATIVE = 2,
    CAUSAL = 3,
    HEBBIAN = 4
};

// Memory Node structure
struct MemoryNode {
    uint64_t id;
    std::string content;
    NodeType type;
    uint64_t creation_time;
    uint8_t importance; // 0-255
    std::set<uint64_t> connections; // Connected node IDs
    std::map<std::string, float> attributes; // Additional attributes
    
    MemoryNode() : id(0), type(NodeType::TEXT), creation_time(0), importance(128) {}
    
    MemoryNode(uint64_t node_id, const std::string& node_content, NodeType node_type, uint8_t imp = 128)
        : id(node_id), content(node_content), type(node_type), importance(imp) {
        creation_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
};

// Connection structure
struct MemoryConnection {
    uint64_t id;
    uint64_t source_id;
    uint64_t target_id;
    ConnectionType type;
    float strength; // 0.0 - 1.0
    uint64_t creation_time;
    uint64_t last_activation;
    
    MemoryConnection() : id(0), source_id(0), target_id(0), type(ConnectionType::SEMANTIC), 
                        strength(0.5f), creation_time(0), last_activation(0) {}
    
    MemoryConnection(uint64_t conn_id, uint64_t src, uint64_t tgt, ConnectionType conn_type, float str = 0.5f)
        : id(conn_id), source_id(src), target_id(tgt), type(conn_type), strength(str) {
        creation_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        last_activation = creation_time;
    }
};

// Global Memory Manager
class GlobalMemoryManager {
private:
    std::map<uint64_t, MemoryNode> nodes;
    std::map<uint64_t, MemoryConnection> connections;
    std::map<std::string, uint64_t> content_to_node; // For deduplication
    uint64_t next_node_id;
    uint64_t next_connection_id;
    std::mutex memory_mutex;
    std::string storage_path;
    
    // File paths
    std::string nodes_file;
    std::string connections_file;
    
public:
    GlobalMemoryManager(const std::string& path = "melvin_global_memory");
    ~GlobalMemoryManager();
    
    // Node operations
    uint64_t create_node(const std::string& content, NodeType type, uint8_t importance = 128);
    MemoryNode* get_node(uint64_t id);
    std::vector<MemoryNode*> find_nodes_by_content(const std::string& content);
    std::vector<MemoryNode*> find_nodes_by_type(NodeType type);
    
    // Connection operations
    uint64_t create_connection(uint64_t source_id, uint64_t target_id, ConnectionType type, float strength = 0.5f);
    MemoryConnection* get_connection(uint64_t id);
    std::vector<MemoryConnection*> get_connections_from(uint64_t node_id);
    std::vector<MemoryConnection*> get_connections_to(uint64_t node_id);
    
    // Memory operations
    void strengthen_connection(uint64_t connection_id, float delta = 0.1f);
    void activate_node(uint64_t node_id);
    std::vector<uint64_t> get_related_nodes(uint64_t node_id, int max_depth = 2);
    
    // Processing functions
    std::vector<uint64_t> process_input_to_nodes(const std::string& input);
    std::string generate_response_from_nodes(const std::vector<uint64_t>& activated_nodes);
    
    // Storage operations
    void save_to_disk();
    void load_from_disk();
    
    // Statistics
    size_t get_node_count() const { return nodes.size(); }
    size_t get_connection_count() const { return connections.size(); }
    std::string get_memory_stats() const;
    
private:
    void update_node_index();
    void update_connection_index();
    std::string serialize_node(const MemoryNode& node);
    std::string serialize_connection(const MemoryConnection& conn);
    MemoryNode deserialize_node(const std::string& data);
    MemoryConnection deserialize_connection(const std::string& data);
};

} // namespace melvin

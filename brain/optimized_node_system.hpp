#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdint>
#include <algorithm>
#include <functional>

namespace melvin {

// Forward declarations
class OptimizedNode;
class NodeConnection;
class NodeStorage;

// Efficient byte-aligned enums
enum class NodeSize : uint8_t {
    TINY = 0,      // 1-10 bytes
    SMALL = 1,     // 11-50 bytes  
    MEDIUM = 2,    // 51-200 bytes
    LARGE = 3,     // 201-1000 bytes
    EXTRA_LARGE = 4 // 1001+ bytes
};

enum class ConnectionType : uint8_t {
    SIMILARITY = 0,
    HIERARCHICAL = 1,
    TEMPORAL = 2,
    ALL = 3
};

enum class NodeType : uint8_t {
    WORD = 0,
    PHRASE = 1,
    CONCEPT = 2,
    SECTION = 3,
    DOCUMENT = 4
};

// Compact node configuration (16 bytes total)
struct alignas(8) NodeConfig {
    NodeSize size_category;
    NodeType node_type;
    ConnectionType connection_strategy;
    uint8_t max_connections;
    uint8_t similarity_threshold; // Scaled 0-255
    uint16_t min_size;
    uint16_t max_size;
    uint32_t flags; // Bit flags for additional options
};

// Optimized node structure (variable size, byte-aligned)
struct alignas(8) OptimizedNode {
    uint64_t id;                    // 8 bytes - unique identifier
    uint32_t content_length;        // 4 bytes - length of content
    uint32_t content_offset;        // 4 bytes - offset in content pool
    NodeConfig config;              // 16 bytes - configuration
    float complexity_score;         // 4 bytes - complexity metric
    uint64_t parent_id;             // 8 bytes - parent node ID
    uint64_t creation_time;         // 8 bytes - timestamp
    uint32_t connection_count;      // 4 bytes - number of connections
    uint32_t connection_offset;     // 4 bytes - offset in connection pool
    
    // Total: 60 bytes per node (aligned to 64 bytes)
    // Content and connections stored separately in pools
};

// Compact connection structure (16 bytes)
struct alignas(8) NodeConnection {
    uint64_t source_id;     // 8 bytes
    uint64_t target_id;     // 8 bytes
    float weight;           // 4 bytes
    ConnectionType type;    // 1 byte
    uint8_t padding[3];     // 3 bytes padding for alignment
};

// Efficient storage pools
class NodeStorage {
private:
    // Content pool - stores all node content contiguously
    std::vector<char> content_pool_;
    std::vector<uint32_t> content_free_list_;
    
    // Connection pool - stores all connections contiguously  
    std::vector<NodeConnection> connection_pool_;
    std::vector<uint32_t> connection_free_list_;
    
    // Node pool - stores all nodes contiguously
    std::vector<OptimizedNode> node_pool_;
    std::vector<uint32_t> node_free_list_;
    
    // Indexes for fast lookup
    std::unordered_map<uint64_t, uint32_t> id_to_index_;
    std::unordered_map<std::string, uint64_t> content_to_id_;
    
    // Statistics
    uint64_t total_nodes_;
    uint64_t total_connections_;
    uint64_t total_content_bytes_;
    
    // Configuration
    static constexpr size_t INITIAL_POOL_SIZE = 1024;
    static constexpr size_t GROWTH_FACTOR = 2;

public:
    NodeStorage();
    ~NodeStorage() = default;
    
    // Node management
    uint64_t create_node(const std::string& content, const NodeConfig& config);
    void delete_node(uint64_t node_id);
    OptimizedNode* get_node(uint64_t node_id);
    const OptimizedNode* get_node(uint64_t node_id) const;
    
    // Connection management
    uint64_t create_connection(uint64_t source_id, uint64_t target_id, 
                              float weight, ConnectionType type);
    void delete_connection(uint64_t connection_id);
    std::vector<NodeConnection> get_node_connections(uint64_t node_id) const;
    
    // Content management
    std::string get_node_content(uint64_t node_id) const;
    void update_node_content(uint64_t node_id, const std::string& new_content);
    
    // Memory optimization
    void compact_storage();
    void optimize_layout();
    size_t get_memory_usage() const;
    
    // Statistics
    uint64_t get_total_nodes() const { return total_nodes_; }
    uint64_t get_total_connections() const { return total_connections_; }
    uint64_t get_total_content_bytes() const { return total_content_bytes_; }
    
    // Batch operations
    void batch_create_nodes(const std::vector<std::pair<std::string, NodeConfig>>& nodes);
    void batch_create_connections(const std::vector<std::tuple<uint64_t, uint64_t, float, ConnectionType>>& connections);

private:
    // Memory management helpers
    uint32_t allocate_content_space(size_t size);
    void free_content_space(uint32_t offset, size_t size);
    uint32_t allocate_connection_space();
    void free_connection_space(uint32_t index);
    uint32_t allocate_node_space();
    void free_node_space(uint32_t index);
    
    // Content deduplication
    uint64_t find_existing_content(const std::string& content) const;
    void deduplicate_content();
};

// Optimized dynamic node sizer
class OptimizedDynamicNodeSizer {
private:
    std::unique_ptr<NodeStorage> storage_;
    
    // Size configurations (stored as static data)
    static const std::unordered_map<NodeSize, NodeConfig> SIZE_CONFIGS_;
    
    // Complexity calculation cache
    mutable std::unordered_map<std::string, float> complexity_cache_;
    
    // Statistics
    struct Stats {
        uint64_t tiny_nodes = 0;
        uint64_t small_nodes = 0;
        uint64_t medium_nodes = 0;
        uint64_t large_nodes = 0;
        uint64_t extra_large_nodes = 0;
        uint64_t total_connections = 0;
    } stats_;

public:
    OptimizedDynamicNodeSizer();
    ~OptimizedDynamicNodeSizer() = default;
    
    // Main interface
    std::vector<uint64_t> create_dynamic_nodes(const std::string& text, 
                                              NodeSize preferred_size = NodeSize::MEDIUM,
                                              float complexity_threshold = 0.5f);
    
    // Size-specific creation
    std::vector<uint64_t> create_tiny_nodes(const std::string& text);
    std::vector<uint64_t> create_small_nodes(const std::string& text);
    std::vector<uint64_t> create_medium_nodes(const std::string& text);
    std::vector<uint64_t> create_large_nodes(const std::string& text);
    std::vector<uint64_t> create_extra_large_nodes(const std::string& text);
    
    // Auto-sizing logic
    std::vector<uint64_t> create_auto_sized_nodes(const std::string& text, float complexity_threshold);
    
    // Connection creation
    void create_similarity_connections(uint64_t node_id, const NodeConfig& config);
    void create_hierarchical_connections(uint64_t node_id, const NodeConfig& config);
    void create_temporal_connections(uint64_t node_id, const NodeConfig& config);
    void create_all_connections(uint64_t node_id, const NodeConfig& config);
    
    // Utility functions
    float calculate_complexity(const std::string& text) const;
    NodeSize determine_optimal_size(const std::string& text, float complexity) const;
    std::vector<std::string> extract_phrases(const std::string& text) const;
    std::vector<std::string> split_into_chunks(const std::string& text, size_t target_size) const;
    
    // Statistics
    Stats get_statistics() const { return stats_; }
    size_t get_memory_usage() const;
    
    // Optimization
    void optimize_storage();
    void clear_cache();

private:
    // Helper functions
    uint64_t create_single_node(const std::string& content, NodeSize size_category, 
                               NodeType node_type, float complexity_score);
    void update_stats(NodeSize size_category);
    bool is_meaningful_phrase(const std::string& phrase) const;
};

// Utility functions for byte-level operations
namespace utils {
    
    // Efficient string hashing
    uint64_t hash_string(const std::string& str);
    
    // Byte-level memory operations
    void copy_bytes(void* dest, const void* src, size_t size);
    void zero_bytes(void* ptr, size_t size);
    
    // Alignment utilities
    size_t align_to(size_t size, size_t alignment);
    bool is_aligned(const void* ptr, size_t alignment);
    
    // Compression helpers
    size_t compress_string(const std::string& input, std::vector<uint8_t>& output);
    std::string decompress_string(const std::vector<uint8_t>& input);
    
    // Memory pool management
    class MemoryPool {
    private:
        std::vector<uint8_t> pool_;
        std::vector<size_t> free_list_;
        size_t next_free_;
        
    public:
        explicit MemoryPool(size_t initial_size = 1024);
        void* allocate(size_t size);
        void deallocate(void* ptr, size_t size);
        void compact();
        size_t get_usage() const;
    };
}

// Inline implementations for performance-critical functions
inline uint64_t OptimizedNode::get_id() const { return id; }
inline uint32_t OptimizedNode::get_content_length() const { return content_length; }
inline NodeConfig OptimizedNode::get_config() const { return config; }
inline float OptimizedNode::get_complexity_score() const { return complexity_score; }

// Memory-efficient iterators
class NodeIterator {
private:
    const NodeStorage* storage_;
    uint32_t current_index_;
    
public:
    NodeIterator(const NodeStorage* storage, uint32_t start_index = 0);
    bool has_next() const;
    const OptimizedNode* next();
    void reset();
};

// Connection iterator
class ConnectionIterator {
private:
    const NodeStorage* storage_;
    uint64_t node_id_;
    uint32_t current_index_;
    std::vector<NodeConnection> connections_;
    
public:
    ConnectionIterator(const NodeStorage* storage, uint64_t node_id);
    bool has_next() const;
    const NodeConnection* next();
    void reset();
};

} // namespace melvin

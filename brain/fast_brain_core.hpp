/**
 * 🧠 MELVIN FAST BRAIN CORE
 * High-performance C++ implementation of node and connection operations
 * Optimized for speed and memory efficiency
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <chrono>
#include <thread>
#include <queue>
#include <functional>
#include <algorithm>
#include <random>
#include <immintrin.h>  // For SIMD operations

namespace melvin {
namespace brain {

// Forward declarations
class FastNode;
class FastConnection;
class FastBrainCore;

// Type aliases for performance
using NodeID = uint64_t;
using ConnectionID = uint64_t;
using Weight = float;
using Timestamp = uint64_t;
using EmbeddingVector = std::vector<float>;

// Node types
enum class NodeType : uint8_t {
    LANGUAGE = 0,
    CODE = 1,
    VISUAL = 2,
    AUDIO = 3,
    CONCEPT = 4,
    EMOTION = 5,
    ATOMIC_FACT = 6,
    CONSOLIDATED = 7,
    SPECIALIZED = 8
};

// Connection types
enum class ConnectionType : uint8_t {
    SIMILARITY = 0,
    TEMPORAL = 1,
    HEBBIAN = 2,
    MULTIMODAL = 3,
    ATOMIC_RELATION = 4,
    CONSOLIDATION = 5
};

/**
 * High-performance node representation
 * Memory-aligned and cache-friendly
 */
class alignas(64) FastNode {
public:
    NodeID id;
    NodeType type;
    std::string content;
    EmbeddingVector embedding;
    
    // Performance metrics
    std::atomic<float> activation_strength{0.5f};
    std::atomic<float> firing_rate{0.0f};
    std::atomic<uint32_t> activation_count{0};
    std::atomic<Timestamp> last_activation{0};
    
    // Connection tracking
    std::unordered_set<NodeID> connections;
    std::atomic<uint32_t> connection_count{0};
    
    // Dynamic sizing metrics
    std::atomic<float> success_rate{0.0f};
    std::atomic<uint32_t> chain_success_count{0};
    std::atomic<uint32_t> access_count{0};
    
    // Thread-safe access
    mutable std::shared_mutex mutex;
    
    FastNode(NodeID node_id, NodeType node_type, const std::string& node_content);
    
    // High-performance operations
    void activate(float strength = 1.0f) noexcept;
    void update_success_rate(bool success) noexcept;
    void add_connection(NodeID other_id) noexcept;
    void remove_connection(NodeID other_id) noexcept;
    
    // Getters with thread safety
    float get_activation_strength() const noexcept { return activation_strength.load(); }
    uint32_t get_activation_count() const noexcept { return activation_count.load(); }
    uint32_t get_connection_count() const noexcept { return connection_count.load(); }
    float get_success_rate() const noexcept { return success_rate.load(); }
    
    // Size optimization helpers
    size_t get_content_size() const noexcept { return content.size(); }
    bool should_fragment() const noexcept;
    bool should_consolidate() const noexcept;
    bool should_specialize() const noexcept;
};

/**
 * High-performance connection representation
 */
class alignas(32) FastConnection {
public:
    ConnectionID id;
    NodeID source_id;
    NodeID target_id;
    ConnectionType type;
    
    std::atomic<Weight> weight{1.0f};
    std::atomic<uint32_t> coactivation_count{0};
    std::atomic<Timestamp> last_coactivation{0};
    std::atomic<Timestamp> creation_time;
    
    FastConnection(ConnectionID conn_id, NodeID src, NodeID tgt, ConnectionType conn_type);
    
    // Hebbian learning
    void strengthen(float amount = 0.01f) noexcept;
    void weaken(float amount = 0.005f) noexcept;
    void coactivate() noexcept;
    
    // Getters
    Weight get_weight() const noexcept { return weight.load(); }
    uint32_t get_coactivation_count() const noexcept { return coactivation_count.load(); }
};

/**
 * High-performance brain operations
 * Optimized for concurrent access and SIMD operations
 */
class FastBrainCore {
private:
    // Memory-efficient storage
    std::unordered_map<NodeID, std::unique_ptr<FastNode>> nodes_;
    std::unordered_map<ConnectionID, std::unique_ptr<FastConnection>> connections_;
    
    // Index structures for fast lookup
    std::unordered_map<NodeType, std::vector<NodeID>> nodes_by_type_;
    std::unordered_map<NodeID, std::vector<ConnectionID>> connections_by_node_;
    
    // Thread safety
    mutable std::shared_mutex global_mutex_;
    
    // Performance counters
    std::atomic<uint64_t> total_activations_{0};
    std::atomic<uint64_t> total_searches_{0};
    std::atomic<uint64_t> total_connections_created_{0};
    
    // Random number generation for performance
    thread_local static std::mt19937 rng_;
    
    // Memory pools for allocation efficiency
    std::vector<std::unique_ptr<FastNode>> node_pool_;
    std::vector<std::unique_ptr<FastConnection>> connection_pool_;
    
    // ID generation
    std::atomic<NodeID> next_node_id_{1};
    std::atomic<ConnectionID> next_connection_id_{1};
    
public:
    FastBrainCore();
    ~FastBrainCore();
    
    // Core operations - optimized for speed
    NodeID create_node(NodeType type, const std::string& content);
    ConnectionID create_connection(NodeID source, NodeID target, ConnectionType type, Weight initial_weight = 1.0f);
    
    // High-speed node operations
    bool activate_node(NodeID id, float strength = 1.0f) noexcept;
    bool update_node_content(NodeID id, const std::string& new_content);
    bool remove_node(NodeID id);
    
    // High-speed connection operations
    bool strengthen_connection(ConnectionID id, float amount = 0.01f) noexcept;
    bool weaken_connection(ConnectionID id, float amount = 0.005f) noexcept;
    bool remove_connection(ConnectionID id);
    
    // Ultra-fast search operations using SIMD
    std::vector<NodeID> search_nodes_simd(const std::string& query, size_t max_results = 10) const;
    std::vector<NodeID> search_by_content_hash(uint64_t content_hash, size_t max_results = 10) const;
    std::vector<NodeID> get_connected_nodes(NodeID id, size_t max_results = 100) const;
    
    // Semantic similarity using vectorized operations
    std::vector<std::pair<NodeID, float>> find_similar_nodes(NodeID reference_id, size_t max_results = 10) const;
    float calculate_similarity_simd(const EmbeddingVector& a, const EmbeddingVector& b) const noexcept;
    
    // Hebbian learning operations
    void hebbian_update_batch(const std::vector<NodeID>& active_nodes) noexcept;
    void strengthen_coactivated_connections(NodeID node1, NodeID node2) noexcept;
    
    // Dynamic node sizing operations
    std::vector<NodeID> find_fragmentation_candidates() const;
    std::vector<std::vector<NodeID>> find_consolidation_candidates() const;
    std::vector<NodeID> find_specialization_candidates() const;
    
    // Node optimization operations
    bool fragment_node(NodeID id, const std::vector<std::string>& fragments);
    NodeID consolidate_nodes(const std::vector<NodeID>& node_ids, const std::string& consolidated_content);
    NodeID specialize_node(NodeID base_id, const std::string& specialization_context);
    
    // Bulk operations for performance
    void batch_activate_nodes(const std::vector<std::pair<NodeID, float>>& activations) noexcept;
    void batch_create_connections(const std::vector<std::tuple<NodeID, NodeID, ConnectionType, Weight>>& connections);
    void batch_update_weights(const std::vector<std::pair<ConnectionID, Weight>>& updates) noexcept;
    
    // Memory management
    void compact_memory();
    void defragment_indices();
    size_t get_memory_usage() const noexcept;
    
    // Performance metrics
    struct PerformanceStats {
        uint64_t total_nodes;
        uint64_t total_connections;
        uint64_t total_activations;
        uint64_t total_searches;
        double avg_search_time_ms;
        double avg_activation_time_ms;
        size_t memory_usage_bytes;
        double cache_hit_rate;
    };
    
    PerformanceStats get_performance_stats() const noexcept;
    void reset_performance_counters() noexcept;
    
    // Thread-safe getters
    size_t get_node_count() const noexcept;
    size_t get_connection_count() const noexcept;
    size_t get_node_count_by_type(NodeType type) const noexcept;
    
    // Export/import for Python integration
    std::string export_to_json() const;
    bool import_from_json(const std::string& json_data);
    
    // Database integration
    bool load_from_sqlite(const std::string& db_path);
    bool save_to_sqlite(const std::string& db_path) const;
    
private:
    // Internal optimization helpers
    void update_indices_after_node_creation(NodeID id) noexcept;
    void update_indices_after_connection_creation(ConnectionID id) noexcept;
    void cleanup_indices_after_deletion(NodeID id) noexcept;
    
    // SIMD-optimized string matching
    bool string_contains_simd(const std::string& haystack, const std::string& needle) const noexcept;
    uint64_t hash_string_fast(const std::string& str) const noexcept;
    
    // Memory pool management
    FastNode* allocate_node();
    FastConnection* allocate_connection();
    void deallocate_node(FastNode* node);
    void deallocate_connection(FastConnection* connection);
};

// Utility functions for high-performance operations
namespace utils {
    
    // Fast string operations
    std::vector<std::string> split_string_fast(const std::string& str, char delimiter);
    std::string normalize_content_fast(const std::string& content);
    uint64_t hash_content_fast(const std::string& content) noexcept;
    
    // SIMD vector operations
    float dot_product_simd(const float* a, const float* b, size_t size) noexcept;
    void normalize_vector_simd(float* vec, size_t size) noexcept;
    float cosine_similarity_simd(const float* a, const float* b, size_t size) noexcept;
    
    // Memory-aligned allocations
    void* aligned_malloc(size_t size, size_t alignment);
    void aligned_free(void* ptr);
    
    // High-resolution timing
    class HighResTimer {
    private:
        std::chrono::high_resolution_clock::time_point start_time_;
        
    public:
        HighResTimer() : start_time_(std::chrono::high_resolution_clock::now()) {}
        
        double elapsed_ms() const noexcept {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time_);
            return duration.count() / 1000000.0;
        }
        
        void reset() noexcept {
            start_time_ = std::chrono::high_resolution_clock::now();
        }
    };
}

} // namespace brain
} // namespace melvin

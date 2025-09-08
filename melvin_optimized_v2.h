#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <chrono>
#include <thread>
#include <mutex>
#include <memory>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <set>
#include <optional>
#include <filesystem>
// Compression libraries - optional for testing
#ifdef HAVE_ZLIB
#include <zlib.h>
#endif
#ifdef HAVE_LZMA
#include <lzma.h>
#endif
#ifdef HAVE_ZSTD
#include <zstd.h>
#endif

// ============================================================================
// CORE TYPES AND STRUCTURES
// ============================================================================

enum class ContentType : uint8_t {
    TEXT = 0,
    IMAGE = 1,
    AUDIO = 2,
    CODE = 3,
    EMBEDDING = 4,
    METADATA = 5,
    CONCEPT = 6,
    SEQUENCE = 7,
    VISUAL = 8,
    SENSOR = 9
};

enum class CompressionType : uint8_t {
    NONE = 0,
    GZIP = 1,
    LZMA = 2,
    ZSTD = 3
};

enum class ConnectionType : uint8_t {
    HEBBIAN = 0,
    SIMILARITY = 1,
    TEMPORAL = 2,
    HIERARCHICAL = 3,
    MULTIMODAL = 4,
    CAUSAL = 5,
    ASSOCIATIVE = 6
};

// Binary Node Structure - 28 bytes header + content
struct BinaryNode {
    uint64_t id;                    // 8 bytes - unique identifier
    uint64_t creation_time;         // 8 bytes - timestamp
    ContentType content_type;       // 1 byte
    CompressionType compression;    // 1 byte
    uint8_t importance;             // 1 byte - 0-255 importance score
    uint8_t activation_strength;    // 1 byte - 0-255 activation strength
    uint32_t content_length;       // 4 bytes - length of content
    uint32_t connection_count;     // 4 bytes - number of connections
    
    std::vector<uint8_t> content;  // Raw binary content
    
    BinaryNode() : id(0), creation_time(0), content_type(ContentType::TEXT),
                   compression(CompressionType::NONE), importance(0),
                   activation_strength(0), content_length(0), connection_count(0) {}
    
    // Convert to binary format
    std::vector<uint8_t> to_bytes() const;
    
    // Create from binary data
    static BinaryNode from_bytes(const std::vector<uint8_t>& data);
};

// Binary Connection Structure - 18 bytes
struct BinaryConnection {
    uint64_t id;                    // 8 bytes
    uint64_t source_id;            // 8 bytes
    uint64_t target_id;            // 8 bytes
    ConnectionType connection_type; // 1 byte
    uint8_t weight;                // 1 byte
    
    BinaryConnection() : id(0), source_id(0), target_id(0),
                        connection_type(ConnectionType::HEBBIAN), weight(0) {}
    
    // Convert to binary format
    std::vector<uint8_t> to_bytes() const;
    
    // Create from binary data
    static BinaryConnection from_bytes(const std::vector<uint8_t>& data);
};

// ============================================================================
// COMPRESSION UTILITIES
// ============================================================================

class CompressionUtils {
public:
    static std::vector<uint8_t> compress_gzip(const std::vector<uint8_t>& data);
    static std::vector<uint8_t> decompress_gzip(const std::vector<uint8_t>& data);
    static std::vector<uint8_t> compress_lzma(const std::vector<uint8_t>& data);
    static std::vector<uint8_t> decompress_lzma(const std::vector<uint8_t>& data);
    static std::vector<uint8_t> compress_zstd(const std::vector<uint8_t>& data);
    static std::vector<uint8_t> decompress_zstd(const std::vector<uint8_t>& data);
    static CompressionType determine_best_compression(const std::vector<uint8_t>& data);
    static std::vector<uint8_t> compress_content(const std::vector<uint8_t>& data, 
                                                CompressionType compression_type);
    static std::vector<uint8_t> decompress_content(const std::vector<uint8_t>& data,
                                                  CompressionType compression_type);
};

// ============================================================================
// INTELLIGENT PRUNING SYSTEM
// ============================================================================

struct PruningDecision {
    uint64_t node_id;
    bool keep;
    float confidence;
    std::string reason;
    float importance_score;
    uint64_t timestamp;
    
    PruningDecision() : node_id(0), keep(false), confidence(0.0f),
                        importance_score(0.0f), timestamp(0) {}
};

class IntelligentPruningSystem {
private:
    std::map<ContentType, float> content_type_weights;
    float temporal_half_life_days;
    uint8_t eternal_threshold;
    
public:
    IntelligentPruningSystem();
    float calculate_activation_importance(const BinaryNode& node);
    float calculate_connection_importance(const BinaryNode& node, uint32_t connection_count);
    float calculate_semantic_importance(const std::vector<uint8_t>& content, 
                                      ContentType content_type);
    float calculate_temporal_importance(const BinaryNode& node);
    float calculate_combined_importance(const BinaryNode& node, uint32_t connection_count);
    PruningDecision should_keep_node(const BinaryNode& node, uint32_t connection_count, 
                                    float threshold = 0.3f);
};

// ============================================================================
// PURE BINARY STORAGE SYSTEM
// ============================================================================

class PureBinaryStorage {
private:
    std::string storage_path;
    std::string nodes_file;
    std::string connections_file;
    std::string index_file;
    
    std::mutex storage_mutex;
    std::unordered_map<uint64_t, size_t> node_index; // id -> file position
    
    uint64_t total_nodes;
    uint64_t total_connections;
    uint64_t total_bytes;
    
    IntelligentPruningSystem pruning_system;
    
public:
    PureBinaryStorage(const std::string& path = "melvin_binary_memory");
    ~PureBinaryStorage();
    
    void load_index();
    void save_index();
    uint8_t calculate_importance(const std::vector<uint8_t>& content, ContentType content_type);
    uint64_t store_node(const std::vector<uint8_t>& content, ContentType content_type, 
                        uint64_t node_id = 0);
    uint64_t store_connection(uint64_t source_id, uint64_t target_id, 
                             ConnectionType connection_type, uint8_t weight = 128);
    std::optional<BinaryNode> get_node(uint64_t node_id);
    std::string get_node_as_text(uint64_t node_id);
    std::vector<uint64_t> prune_nodes(uint32_t max_nodes_to_prune = 1000);
    
    struct StorageStats {
        uint64_t total_nodes;
        uint64_t total_connections;
        uint64_t total_bytes;
        double total_mb;
        uint64_t nodes_file_size;
        uint64_t connections_file_size;
        uint64_t index_file_size;
    };
    
    StorageStats get_storage_stats();
};

// ============================================================================
// INTELLIGENT CONNECTION TRAVERSAL SYSTEM
// ============================================================================

// Forward declaration
class MelvinOptimizedV2;

struct ConnectionPath {
    std::vector<uint64_t> node_ids;
    float relevance_score;
    std::string path_description;
};

struct NodeSimilarity {
    uint64_t node_id;
    float similarity_score;
    std::string content;
    std::vector<std::string> keywords;
};

struct SynthesizedAnswer {
    std::string answer;
    float confidence;
    std::vector<uint64_t> source_nodes;
    std::string reasoning;
};

class IntelligentConnectionTraversal {
private:
    MelvinOptimizedV2* brain_ref;
    
public:
    IntelligentConnectionTraversal(MelvinOptimizedV2* brain);
    
    // Core intelligent processing functions
    std::vector<std::string> extract_keywords(const std::string& text);
    std::vector<NodeSimilarity> find_relevant_nodes(const std::vector<std::string>& keywords);
    std::vector<ConnectionPath> analyze_connection_paths(const std::vector<NodeSimilarity>& relevant_nodes);
    SynthesizedAnswer synthesize_answer(const std::string& question, 
                                       const std::vector<NodeSimilarity>& relevant_nodes,
                                       const std::vector<ConnectionPath>& connection_paths);
    void create_dynamic_nodes(const std::string& question, const SynthesizedAnswer& answer);
    
    // Enhanced synthesis functions
    std::string analyze_question_type(const std::string& question);
    std::vector<std::string> extract_knowledge_from_nodes(const std::vector<NodeSimilarity>& nodes);
    SynthesizedAnswer synthesize_opinion_response(const std::string& question, 
                                                  const std::vector<std::string>& knowledge,
                                                  const std::vector<NodeSimilarity>& nodes);
    SynthesizedAnswer synthesize_explanation_response(const std::string& question, 
                                                       const std::vector<std::string>& knowledge,
                                                       const std::vector<NodeSimilarity>& nodes);
    SynthesizedAnswer synthesize_connection_response(const std::string& question, 
                                                      const std::vector<std::string>& knowledge,
                                                      const std::vector<NodeSimilarity>& nodes);
    SynthesizedAnswer synthesize_comparison_response(const std::string& question, 
                                                      const std::vector<std::string>& knowledge,
                                                      const std::vector<NodeSimilarity>& nodes);
    SynthesizedAnswer synthesize_solution_response(const std::string& question, 
                                                   const std::vector<std::string>& knowledge,
                                                   const std::vector<NodeSimilarity>& nodes);
    SynthesizedAnswer synthesize_analysis_response(const std::string& question, 
                                                   const std::vector<std::string>& knowledge,
                                                   const std::vector<NodeSimilarity>& nodes);
    SynthesizedAnswer synthesize_general_response(const std::string& question, 
                                                   const std::vector<std::string>& knowledge,
                                                   const std::vector<NodeSimilarity>& nodes);
    
    // Intelligent answering
    SynthesizedAnswer answer_question_intelligently(const std::string& question);
};

// ============================================================================
// OPTIMIZED MELVIN GLOBAL BRAIN
// ============================================================================

class MelvinOptimizedV2 {
private:
    std::unique_ptr<PureBinaryStorage> binary_storage;
    std::mutex brain_mutex;
    
    // Hebbian learning
    struct Activation {
        uint64_t node_id;
        uint64_t timestamp;
        float strength;
    };
    
    std::vector<Activation> recent_activations;
    std::mutex activation_mutex;
    static constexpr size_t MAX_ACTIVATIONS = 1000;
    static constexpr double COACTIVATION_WINDOW = 2.0; // seconds
    
    // Intelligent connection traversal system
    std::unique_ptr<IntelligentConnectionTraversal> intelligent_traversal;
    
    // Statistics structure
    struct BrainStats {
        uint64_t total_nodes;
        uint64_t total_connections;
        uint64_t hebbian_updates;
        uint64_t similarity_connections;
        uint64_t temporal_connections;
        uint64_t cross_modal_connections;
        uint64_t start_time;
        uint64_t intelligent_answers_generated;
        uint64_t dynamic_nodes_created;
    };
    
    // Statistics instance
    BrainStats stats;
    
public:
    MelvinOptimizedV2(const std::string& storage_path = "melvin_binary_memory");
    
    // Core brain processing with intelligent capabilities
    uint64_t process_text_input(const std::string& text, const std::string& source = "user");
    uint64_t process_code_input(const std::string& code, const std::string& source = "python");
    void update_hebbian_learning(uint64_t node_id);
    std::string get_node_content(uint64_t node_id);
    
    // Intelligent connection traversal and answering
    SynthesizedAnswer answer_question_intelligently(const std::string& question);
    std::vector<std::string> extract_keywords(const std::string& text);
    std::vector<NodeSimilarity> find_relevant_nodes(const std::vector<std::string>& keywords);
    std::vector<ConnectionPath> analyze_connection_paths(const std::vector<NodeSimilarity>& relevant_nodes);
    void create_dynamic_nodes(const std::string& question, const SynthesizedAnswer& answer);
    
    // Statistics update methods
    void increment_intelligent_answers() { stats.intelligent_answers_generated++; }
    void increment_dynamic_nodes(uint64_t count = 1) { stats.dynamic_nodes_created += count; }
    
    // Enhanced brain state with intelligent capabilities
    struct BrainState {
        struct GlobalMemory {
            uint64_t total_nodes;
            uint64_t total_edges;
            double storage_used_mb;
            BrainStats stats;
        } global_memory;
        
        struct System {
            bool running;
            uint64_t uptime_seconds;
        } system;
        
        struct IntelligentCapabilities {
            uint64_t intelligent_answers_generated;
            uint64_t dynamic_nodes_created;
            bool connection_traversal_enabled;
            bool dynamic_node_creation_enabled;
        } intelligent_capabilities;
    };
    
    BrainState get_unified_state();
    std::vector<uint64_t> prune_old_nodes(uint32_t max_nodes_to_prune = 1000);
    void save_complete_state();
};

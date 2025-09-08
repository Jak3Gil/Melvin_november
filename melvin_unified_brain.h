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
#include <queue>
#include <functional>
#include <atomic>

// ============================================================================
// UNIFIED MELVIN BRAIN - CONTINUOUS THOUGHT CYCLE ARCHITECTURE
// ============================================================================
// This is the single, unified Melvin brain that incorporates:
// 1. Continuous self-looping thought cycle
// 2. Binary storage with intelligent traversal
// 3. Dynamic learning and node creation
// 4. Meta-cognitive self-evaluation
// 5. External interrupt handling

// ============================================================================
// UNIFIED BRAIN CORE TYPES
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
    SENSOR = 9,
    THOUGHT = 10,        // Meta-cognitive thoughts
    EVALUATION = 11,     // Self-evaluations
    LESSON = 12,         // Learned lessons
    INTERRUPT = 13       // External interrupts
};

enum class ConnectionType : uint8_t {
    HEBBIAN = 0,
    SIMILARITY = 1,
    TEMPORAL = 2,
    HIERARCHICAL = 3,
    MULTIMODAL = 4,
    CAUSAL = 5,
    ASSOCIATIVE = 6,
    META_COGNITIVE = 7,  // Self-reflection connections
    LEARNING = 8,        // Lesson connections
    INTERRUPT = 9        // External input connections
};

enum class CompressionType : uint8_t {
    NONE = 0,
    GZIP = 1,
    LZMA = 2,
    ZSTD = 3
};

// ============================================================================
// UNIFIED BRAIN NODE
// ============================================================================

struct UnifiedNode {
    uint64_t id;                    // 8 bytes - unique identifier
    uint64_t creation_time;         // 8 bytes - timestamp
    ContentType content_type;       // 1 byte
    CompressionType compression;    // 1 byte
    uint8_t importance;             // 1 byte - 0-255 importance score
    uint8_t activation_strength;     // 1 byte - 0-255 activation strength
    uint32_t content_length;        // 4 bytes - length of content
    uint32_t connection_count;      // 4 bytes - number of connections
    uint64_t last_access_time;      // 8 bytes - last access timestamp
    uint32_t access_count;          // 4 bytes - access frequency
    float confidence_score;         // 4 bytes - confidence in content
    std::vector<uint8_t> content;   // Raw binary content
    
    UnifiedNode() : id(0), creation_time(0), content_type(ContentType::TEXT),
                   compression(CompressionType::NONE), importance(0),
                   activation_strength(0), content_length(0), connection_count(0),
                   last_access_time(0), access_count(0), confidence_score(0.0f) {}
    
    // Convert to binary format
    std::vector<uint8_t> to_bytes() const;
    
    // Create from binary data
    static UnifiedNode from_bytes(const std::vector<uint8_t>& data);
    
    // Activation methods
    void activate(float strength = 0.1f);
    void decay(float rate = 0.01f);
    bool is_active() const;
    
    // Content methods
    std::string get_text_content() const;
    void set_text_content(const std::string& text);
};

// ============================================================================
// UNIFIED BRAIN CONNECTION
// ============================================================================

struct UnifiedConnection {
    uint64_t id;                    // 8 bytes
    uint64_t source_id;            // 8 bytes
    uint64_t target_id;            // 8 bytes
    ConnectionType connection_type; // 1 byte
    uint8_t weight;                // 1 byte
    uint64_t creation_time;         // 8 bytes
    uint32_t usage_count;          // 4 bytes
    float strength;                 // 4 bytes
    
    UnifiedConnection() : id(0), source_id(0), target_id(0),
                         connection_type(ConnectionType::HEBBIAN), weight(0),
                         creation_time(0), usage_count(0), strength(0.0f) {}
    
    // Convert to binary format
    std::vector<uint8_t> to_bytes() const;
    
    // Create from binary data
    static UnifiedConnection from_bytes(const std::vector<uint8_t>& data);
    
    // Connection strengthening
    void strengthen(float amount = 0.1f);
    void weaken(float amount = 0.05f);
};

// ============================================================================
// CONTINUOUS THOUGHT CYCLE STRUCTURES
// ============================================================================

struct ThoughtCycle {
    uint64_t cycle_id;
    std::string input;              // What triggered this cycle
    std::vector<uint64_t> activated_nodes;
    std::string analysis;           // Analysis/hypotheses generated
    std::string output;             // Response/action produced
    float self_evaluation_rating;   // 1-10 rating
    std::string evaluation_reason;  // Why this rating
    std::vector<std::string> lessons_learned;
    uint64_t timestamp;
    bool is_external_interrupt;
    
    ThoughtCycle() : cycle_id(0), self_evaluation_rating(0.0f), timestamp(0),
                     is_external_interrupt(false) {}
};

struct MetaCognitiveState {
    uint64_t current_cycle_id;
    std::string last_output;
    std::vector<std::string> recent_lessons;
    float overall_confidence;
    uint64_t total_cycles;
    uint64_t successful_cycles;
    uint64_t interrupted_cycles;
    std::atomic<bool> cycle_active;
    
    MetaCognitiveState() : current_cycle_id(0), overall_confidence(0.0f),
                          total_cycles(0), successful_cycles(0), interrupted_cycles(0),
                          cycle_active(false) {}
};

// ============================================================================
// INTELLIGENT TRAVERSAL STRUCTURES
// ============================================================================

struct ConnectionPath {
    std::vector<uint64_t> node_ids;
    float relevance_score;
    std::string path_description;
    ConnectionType primary_connection_type;
};

struct NodeSimilarity {
    uint64_t node_id;
    float similarity_score;
    std::string content;
    std::vector<std::string> keywords;
    ContentType content_type;
};

struct SynthesizedAnswer {
    std::string answer;
    float confidence;
    std::vector<uint64_t> source_nodes;
    std::string reasoning;
    std::vector<std::string> keywords_used;
};

// ============================================================================
// UNIFIED MELVIN BRAIN CLASS
// ============================================================================

class MelvinUnifiedBrain {
private:
    // Core storage
    std::unordered_map<uint64_t, UnifiedNode> nodes;
    std::unordered_map<uint64_t, UnifiedConnection> connections;
    std::unordered_map<uint64_t, size_t> node_index; // id -> file position
    
    // Continuous thought cycle
    MetaCognitiveState meta_state;
    std::vector<ThoughtCycle> thought_history;
    std::queue<std::string> input_queue;
    std::mutex cycle_mutex;
    std::thread continuous_cycle_thread;
    std::atomic<bool> should_run_cycle;
    
    // Storage management
    std::string storage_path;
    std::string nodes_file;
    std::string connections_file;
    std::string index_file;
    std::mutex storage_mutex;
    
    // Statistics
    struct BrainStats {
        uint64_t total_nodes;
        uint64_t total_connections;
        uint64_t total_thought_cycles;
        uint64_t hebbian_updates;
        uint64_t similarity_connections;
        uint64_t temporal_connections;
        uint64_t meta_cognitive_connections;
        uint64_t dynamic_nodes_created;
        uint64_t intelligent_answers_generated;
        uint64_t start_time;
        double storage_used_mb;
    } stats;
    
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
    
public:
    MelvinUnifiedBrain(const std::string& path = "melvin_unified_memory");
    ~MelvinUnifiedBrain();
    
    // ============================================================================
    // CONTINUOUS THOUGHT CYCLE METHODS
    // ============================================================================
    
    void start_continuous_cycle();
    void stop_continuous_cycle();
    void process_external_input(const std::string& input);
    ThoughtCycle execute_thought_cycle(const std::string& input, bool is_external = false);
    
    // ============================================================================
    // CORE BRAIN OPERATIONS
    // ============================================================================
    
    uint64_t process_text_input(const std::string& text, const std::string& source = "user");
    uint64_t process_code_input(const std::string& code, const std::string& source = "python");
    uint64_t create_meta_cognitive_node(const std::string& content, ContentType type);
    uint64_t create_connection(uint64_t source_id, uint64_t target_id, 
                              ConnectionType connection_type, uint8_t weight = 128);
    
    // ============================================================================
    // INTELLIGENT TRAVERSAL AND ANSWERING
    // ============================================================================
    
    std::vector<std::string> extract_keywords(const std::string& text);
    std::vector<NodeSimilarity> find_relevant_nodes(const std::vector<std::string>& keywords);
    std::vector<ConnectionPath> analyze_connection_paths(const std::vector<NodeSimilarity>& relevant_nodes);
    SynthesizedAnswer synthesize_answer(const std::string& question, 
                                       const std::vector<NodeSimilarity>& relevant_nodes,
                                       const std::vector<ConnectionPath>& connection_paths);
    SynthesizedAnswer answer_question_intelligently(const std::string& question);
    
    // ============================================================================
    // LEARNING AND MEMORY MANAGEMENT
    // ============================================================================
    
    void update_hebbian_learning(uint64_t node_id);
    void create_dynamic_nodes(const std::string& question, const SynthesizedAnswer& answer);
    void consolidate_memories();
    void prune_old_nodes(uint32_t max_nodes_to_prune = 1000);
    void update_importance_scores();
    
    // ============================================================================
    // META-COGNITIVE METHODS
    // ============================================================================
    
    float evaluate_output_quality(const std::string& output, const std::string& input);
    std::vector<std::string> extract_lessons_from_cycle(const ThoughtCycle& cycle);
    void store_lessons(const std::vector<std::string>& lessons);
    void mutate_lessons_based_on_feedback();
    
    // ============================================================================
    // STORAGE AND PERSISTENCE
    // ============================================================================
    
    void save_complete_state();
    void load_complete_state();
    void save_thought_cycle(const ThoughtCycle& cycle);
    void load_thought_history();
    
    // ============================================================================
    // BRAIN STATE AND INTROSPECTION
    // ============================================================================
    
    struct UnifiedBrainState {
        struct GlobalMemory {
            uint64_t total_nodes;
            uint64_t total_connections;
            double storage_used_mb;
            BrainStats stats;
        } global_memory;
        
        struct ContinuousCycle {
            bool cycle_active;
            uint64_t current_cycle_id;
            uint64_t total_cycles;
            uint64_t successful_cycles;
            float overall_confidence;
            std::string last_output;
        } continuous_cycle;
        
        struct IntelligentCapabilities {
            uint64_t intelligent_answers_generated;
            uint64_t dynamic_nodes_created;
            uint64_t meta_cognitive_connections;
            bool connection_traversal_enabled;
            bool dynamic_node_creation_enabled;
            bool continuous_cycle_enabled;
        } intelligent_capabilities;
        
        struct System {
            bool running;
            uint64_t uptime_seconds;
            std::string storage_path;
        } system;
    };
    
    UnifiedBrainState get_unified_state();
    void print_brain_status();
    void print_recent_thought_cycles(int count = 5);
    void print_meta_cognitive_state();
    
    // ============================================================================
    // UTILITY METHODS
    // ============================================================================
    
    std::string get_node_content(uint64_t node_id);
    std::optional<UnifiedNode> get_node(uint64_t node_id);
    std::vector<uint64_t> get_node_connections(uint64_t node_id);
    void increment_intelligent_answers() { stats.intelligent_answers_generated++; }
    void increment_dynamic_nodes(uint64_t count = 1) { stats.dynamic_nodes_created += count; }
    
private:
    // ============================================================================
    // INTERNAL PROCESSING METHODS
    // ============================================================================
    
    void continuous_cycle_loop();
    std::string generate_self_input_from_last_output();
    std::string analyze_input(const std::string& input);
    std::string generate_hypotheses(const std::string& input);
    std::string produce_output(const std::string& input, const std::string& analysis);
    void self_evaluate_output(const std::string& output, const std::string& input, ThoughtCycle& cycle);
    void create_meta_cognitive_connections(const ThoughtCycle& cycle);
    
    // Storage operations
    void load_index();
    void save_index();
    uint8_t calculate_importance(const std::vector<uint8_t>& content, ContentType content_type);
    std::vector<uint8_t> compress_content(const std::vector<uint8_t>& data);
    std::vector<uint8_t> decompress_content(const std::vector<uint8_t>& data, CompressionType compression_type);
    
    // Learning operations
    void hebbian_learning(uint64_t node1_id, uint64_t node2_id);
    void temporal_learning(const std::vector<uint64_t>& sequence);
    void semantic_learning(const std::string& concept, const std::vector<std::string>& related_concepts);
    
    // Utility functions
    std::vector<std::string> tokenize_text(const std::string& text);
    float calculate_text_similarity(const std::string& text1, const std::string& text2);
    uint64_t generate_unique_id();
    std::string format_timestamp(uint64_t timestamp);
};

// ============================================================================
// UNIFIED BRAIN INTERFACE
// ============================================================================

class MelvinUnifiedInterface {
private:
    std::unique_ptr<MelvinUnifiedBrain> brain;
    
public:
    MelvinUnifiedInterface(const std::string& storage_path = "melvin_unified_memory");
    ~MelvinUnifiedInterface();
    
    // Simple interface for interaction
    std::string ask(const std::string& question);
    std::string tell(const std::string& information);
    std::string think(const std::string& topic);
    std::string remember(const std::string& experience);
    
    // Continuous cycle control
    void start_thinking();
    void stop_thinking();
    void interrupt_with(const std::string& input);
    
    // Brain introspection
    void show_brain_status();
    void show_recent_thoughts();
    void show_meta_cognitive_state();
    
    // Learning and memory
    void consolidate_knowledge();
    void optimize_brain();
    void save_brain_state();
    void load_brain_state();
};

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

// ============================================================================
// FULLY UNIFIED BRAIN ARCHITECTURE
// ============================================================================
// This is a completely unified system where thinking and memory are
// integrated into one cohesive brain - no separate files or systems

// ============================================================================
// UNIFIED BRAIN CORE TYPES
// ============================================================================

enum class NodeType : uint8_t {
    KNOWLEDGE = 0,
    CONCEPT = 1,
    RELATIONSHIP = 2,
    EXPERIENCE = 3,
    REASONING = 4,
    MEMORY = 5,
    THOUGHT = 6,
    INTENTION = 7,
    EMOTION = 8
};

enum class ConnectionStrength : uint8_t {
    WEAK = 1,
    MODERATE = 2,
    STRONG = 3,
    VERY_STRONG = 4,
    CRITICAL = 5
};

// ============================================================================
// UNIFIED BRAIN NODE
// ============================================================================

struct UnifiedNode {
    uint64_t id;
    std::string content;
    NodeType type;
    uint64_t creation_time;
    uint64_t last_access_time;
    uint32_t access_count;
    float activation_level;
    float importance_score;
    std::vector<uint64_t> connections;
    
    UnifiedNode() : id(0), type(NodeType::KNOWLEDGE), creation_time(0), 
                   last_access_time(0), access_count(0), activation_level(0.0f),
                   importance_score(0.0f) {}
    
    UnifiedNode(uint64_t node_id, const std::string& node_content, NodeType node_type)
        : id(node_id), content(node_content), type(node_type),
          creation_time(std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::system_clock::now().time_since_epoch()).count()),
          last_access_time(creation_time), access_count(1), activation_level(1.0f),
          importance_score(calculate_initial_importance()) {}
    
    void activate() {
        activation_level = std::min(1.0f, activation_level + 0.1f);
        last_access_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        access_count++;
    }
    
    void decay() {
        activation_level = std::max(0.0f, activation_level - 0.01f);
    }
    
    float calculate_initial_importance() {
        // Calculate importance based on content length and type
        float base_importance = 0.5f;
        float length_factor = std::min(1.0f, content.length() / 100.0f);
        float type_factor = static_cast<float>(type) / 8.0f;
        return base_importance + (length_factor * 0.3f) + (type_factor * 0.2f);
    }
    
    bool is_active() const {
        return activation_level > 0.1f;
    }
};

// ============================================================================
// UNIFIED BRAIN CONNECTION
// ============================================================================

struct UnifiedConnection {
    uint64_t id;
    uint64_t source_id;
    uint64_t target_id;
    ConnectionStrength strength;
    uint64_t creation_time;
    uint32_t usage_count;
    float weight;
    
    UnifiedConnection() : id(0), source_id(0), target_id(0), 
                         strength(ConnectionStrength::MODERATE), creation_time(0),
                         usage_count(0), weight(0.5f) {}
    
    UnifiedConnection(uint64_t conn_id, uint64_t src_id, uint64_t tgt_id, 
                     ConnectionStrength str)
        : id(conn_id), source_id(src_id), target_id(tgt_id), strength(str),
          creation_time(std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::system_clock::now().time_since_epoch()).count()),
          usage_count(1), weight(static_cast<float>(str) / 5.0f) {}
    
    void strengthen() {
        weight = std::min(1.0f, weight + 0.1f);
        usage_count++;
    }
    
    void weaken() {
        weight = std::max(0.1f, weight - 0.05f);
    }
};

// ============================================================================
// UNIFIED BRAIN THOUGHT PROCESS
// ============================================================================

struct ThoughtProcess {
    uint64_t id;
    std::string input;
    std::vector<uint64_t> activated_nodes;
    std::vector<uint64_t> reasoning_path;
    std::string conclusion;
    float confidence;
    uint64_t timestamp;
    
    ThoughtProcess() : id(0), confidence(0.0f), timestamp(0) {}
    
    ThoughtProcess(uint64_t thought_id, const std::string& input_text)
        : id(thought_id), input(input_text), confidence(0.0f),
          timestamp(std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::system_clock::now().time_since_epoch()).count()) {}
};

// ============================================================================
// FULLY UNIFIED BRAIN CLASS
// ============================================================================

class MelvinFullyUnifiedBrain {
private:
    // Core brain data - everything in memory
    std::unordered_map<uint64_t, UnifiedNode> nodes;
    std::unordered_map<uint64_t, UnifiedConnection> connections;
    std::vector<ThoughtProcess> thought_history;
    
    // Brain state
    std::mutex brain_mutex;
    uint64_t next_node_id;
    uint64_t next_connection_id;
    uint64_t next_thought_id;
    
    // Active processing
    std::vector<uint64_t> active_nodes;
    std::queue<std::function<void()>> thought_queue;
    
    // Statistics
    struct BrainStats {
        uint64_t total_nodes;
        uint64_t total_connections;
        uint64_t total_thoughts;
        uint64_t active_thoughts;
        uint64_t memory_usage_bytes;
        uint64_t start_time;
    } stats;
    
public:
    MelvinFullyUnifiedBrain();
    ~MelvinFullyUnifiedBrain();
    
    // Core unified operations
    uint64_t process_input(const std::string& input);
    std::string think_about(const std::string& question);
    std::string reason_through(const std::string& problem);
    
    // Memory operations (integrated with thinking)
    uint64_t create_node(const std::string& content, NodeType type = NodeType::KNOWLEDGE);
    uint64_t create_connection(uint64_t source_id, uint64_t target_id, ConnectionStrength strength);
    void strengthen_connection(uint64_t connection_id);
    void activate_node(uint64_t node_id);
    
    // Thinking operations (integrated with memory)
    std::vector<uint64_t> find_relevant_nodes(const std::string& query);
    std::vector<uint64_t> traverse_connections(uint64_t start_node, int max_depth = 3);
    std::string synthesize_response(const std::vector<uint64_t>& relevant_nodes, const std::string& question);
    
    // Unified learning
    void learn_from_interaction(const std::string& input, const std::string& response);
    void update_importance_scores();
    void decay_inactive_nodes();
    
    // Brain state and introspection
    struct UnifiedBrainState {
        uint64_t total_nodes;
        uint64_t total_connections;
        uint64_t total_thoughts;
        uint64_t active_nodes;
        uint64_t memory_usage_bytes;
        double memory_usage_mb;
        uint64_t uptime_seconds;
        std::vector<std::string> recent_thoughts;
        std::vector<std::string> active_concepts;
    };
    
    UnifiedBrainState get_brain_state();
    void print_brain_status();
    void print_recent_thoughts(int count = 5);
    
    // Advanced unified operations
    std::string answer_question(const std::string& question);
    std::string solve_problem(const std::string& problem);
    std::string generate_idea(const std::string& topic);
    std::string reflect_on(const std::string& experience);
    
private:
    // Internal unified processing
    std::vector<uint64_t> extract_concepts(const std::string& text);
    std::vector<uint64_t> find_similar_nodes(const std::string& content);
    float calculate_node_similarity(const UnifiedNode& node1, const UnifiedNode& node2);
    void propagate_activation(uint64_t node_id, float strength, int depth = 0);
    float calculate_confidence(const std::vector<uint64_t>& nodes, const std::string& response);
    void consolidate_memories();
    void optimize_connections();
    
    // Unified learning mechanisms
    void hebbian_learning(uint64_t node1_id, uint64_t node2_id);
    void temporal_learning(const std::vector<uint64_t>& sequence);
    void semantic_learning(const std::string& concept, const std::vector<std::string>& related_concepts);
    
    // Memory management
    void cleanup_old_memories();
    void compress_redundant_connections();
    void prioritize_important_nodes();
    
public:
    // Public optimization methods
    void consolidate_knowledge() { consolidate_memories(); optimize_connections(); }
    void optimize_brain() { update_importance_scores(); decay_inactive_nodes(); cleanup_old_memories(); }
};

// ============================================================================
// UNIFIED BRAIN UTILITIES
// ============================================================================

class UnifiedBrainUtils {
public:
    static std::vector<std::string> tokenize_text(const std::string& text);
    static std::vector<std::string> extract_keywords(const std::string& text);
    static float calculate_text_similarity(const std::string& text1, const std::string& text2);
    static std::string clean_text(const std::string& text);
    static uint64_t generate_id();
    static std::string format_timestamp(uint64_t timestamp);
};

// ============================================================================
// UNIFIED BRAIN INTERFACE
// ============================================================================

class MelvinUnifiedInterface {
private:
    std::unique_ptr<MelvinFullyUnifiedBrain> brain;
    
public:
    MelvinUnifiedInterface();
    
    // Simple interface for interaction
    std::string ask(const std::string& question);
    std::string tell(const std::string& information);
    std::string think(const std::string& topic);
    std::string remember(const std::string& experience);
    
    // Brain introspection
    void show_brain_status();
    void show_recent_thoughts();
    void show_active_concepts();
    
    // Learning and memory
    void learn_from(const std::string& input, const std::string& output);
    void consolidate_knowledge();
    void optimize_brain();
};

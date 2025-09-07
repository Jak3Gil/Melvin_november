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
#include <sstream>
#include <iomanip>
#include <cmath>
#include <functional>
#include <zlib.h>
#include <lzma.h>
#include <zstd.h>

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
    SENSOR = 9,
    MORAL_SUPERNODE = 10
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
    ASSOCIATIVE = 6,
    SEMANTIC = 7,
    EXPERIENTIAL = 8
};

// Binary Node Structure - 32 bytes header + content
struct BinaryNode {
    uint64_t id;                    // 8 bytes - unique identifier
    uint64_t creation_time;         // 8 bytes - timestamp
    ContentType content_type;       // 1 byte
    CompressionType compression;    // 1 byte
    uint8_t importance;             // 1 byte - 0-255 importance score
    uint8_t activation_strength;    // 1 byte - 0-255 activation strength
    uint32_t content_length;       // 4 bytes - length of content
    uint32_t connection_count;     // 4 bytes - number of connections
    uint8_t emotional_tag;         // 1 byte - emotional context (0-255)
    uint8_t source_confidence;     // 1 byte - confidence in source (0-255)
    uint16_t reserved;             // 2 bytes - reserved for future use
    
    std::vector<uint8_t> content;  // Raw binary content
    
    BinaryNode() : id(0), creation_time(0), content_type(ContentType::TEXT),
                   compression(CompressionType::NONE), importance(0),
                   activation_strength(0), content_length(0), connection_count(0),
                   emotional_tag(128), source_confidence(255), reserved(0) {}
    
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
// COGNITIVE PROCESSING STRUCTURES
// ============================================================================

struct ActivationNode {
    uint64_t node_id;
    float weight;
    uint64_t timestamp;
    std::string token;
    
    ActivationNode() : node_id(0), weight(0.0f), timestamp(0) {}
    ActivationNode(uint64_t id, float w, uint64_t ts, const std::string& t)
        : node_id(id), weight(w), timestamp(ts), token(t) {}
};

struct ConnectionWalk {
    uint64_t target_node;
    float weight;
    ConnectionType type;
    int distance;
    
    ConnectionWalk() : target_node(0), weight(0.0f), type(ConnectionType::HEBBIAN), distance(0) {}
    ConnectionWalk(uint64_t target, float w, ConnectionType t, int d)
        : target_node(target), weight(w), type(t), distance(d) {}
};

struct InterpretationCluster {
    std::vector<uint64_t> node_ids;
    float confidence;
    std::string summary;
    std::vector<std::string> keywords;
    
    InterpretationCluster() : confidence(0.0f) {}
};

struct CandidateResponse {
    std::string text;
    float confidence;
    std::vector<uint64_t> source_nodes;
    std::string reasoning;
    
    CandidateResponse() : confidence(0.0f) {}
};

// ============================================================================
// PRESSURE-BASED INSTINCT SYSTEM
// ============================================================================

struct InstinctForces {
    double curiosity;      // Drive to learn/expand knowledge
    double efficiency;    // Drive to conserve effort/time
    double social;        // Drive to respond empathetically and appropriately
    double consistency;   // Drive to maintain logical/identity coherence
    double survival;      // Drive to protect stability, avoid errors/contradictions
    
    InstinctForces() : curiosity(0.0), efficiency(0.0), social(0.0), consistency(0.0), survival(0.0) {}
    InstinctForces(double c, double e, double s, double con, double sur)
        : curiosity(c), efficiency(e), social(s), consistency(con), survival(sur) {}
    
    // Normalize forces using softmax
    void normalize();
    
    // Get dominant instinct
    std::string get_dominant_instinct() const;
    
    // Get force balance description
    std::string get_balance_description() const;
};

struct Context {
    float recall_confidence;      // How confident we are in memory recall (0.0-1.0)
    float resource_usage;          // Current resource usage/cost (0.0-1.0)
    float user_emotion_score;     // Detected user emotional state (0.0-1.0)
    float memory_conflict_score;  // Level of memory conflicts (0.0-1.0)
    float system_risk_score;      // System risk level (0.0-1.0)
    std::string user_input;       // Original user input
    std::vector<uint64_t> activated_nodes; // Currently activated memory nodes
    double timestamp;            // Current timestamp
    
    Context() : recall_confidence(0.0f), resource_usage(0.0f), user_emotion_score(0.0f),
                memory_conflict_score(0.0f), system_risk_score(0.0f), timestamp(0.0) {}
};

struct DynamicOutput {
    std::string response_text;
    InstinctForces forces_used;
    std::string reasoning_path;
    float overall_confidence;
    std::vector<uint64_t> contributing_nodes;
    std::string emotional_tone;
    std::string response_style;  // "exploratory", "empathetic", "concise", "consistent", "safe"
    
    DynamicOutput() : overall_confidence(0.0f) {}
};

// ============================================================================
// META-REASONING LAYER STRUCTURES
// ============================================================================

struct InstinctProposal {
    std::string instinct_name;
    double force_strength;
    std::string proposed_bias;
    std::string reasoning;
    float confidence;
    
    InstinctProposal() : force_strength(0.0), confidence(0.0f) {}
    InstinctProposal(const std::string& name, double strength, const std::string& bias, 
                    const std::string& reason, float conf)
        : instinct_name(name), force_strength(strength), proposed_bias(bias), 
          reasoning(reason), confidence(conf) {}
};

struct InstinctArbitration {
    std::vector<InstinctProposal> proposals;
    std::vector<std::string> amplifications;  // Instincts to amplify
    std::vector<std::string> suppressions;    // Instincts to suppress
    std::vector<std::string> blends;          // Instincts to blend
    std::string arbitration_reasoning;
    InstinctForces adjusted_forces;
    
    InstinctArbitration() {}
};

struct CandidateOutput {
    std::string instinct_source;
    std::string candidate_text;
    float instinct_weight;
    std::string reasoning;
    std::vector<uint64_t> supporting_nodes;
    
    CandidateOutput() : instinct_weight(0.0f) {}
};

struct MetaReasoningResult {
    InstinctArbitration arbitration;
    std::vector<CandidateOutput> candidates;
    std::string final_blend_reasoning;
    std::string meta_trace;
    float meta_confidence;
    uint64_t meta_decision_node_id;  // Stored in binary memory
    
    MetaReasoningResult() : meta_confidence(0.0f), meta_decision_node_id(0) {}
};

struct EmotionalGrounding {
    bool has_grounding_signal;
    std::string grounding_type;  // "keyword", "repeated_focus", "inferred_intent"
    std::string grounding_evidence;
    float emotional_intensity;
    std::string emotional_tag;
    
    EmotionalGrounding() : has_grounding_signal(false), emotional_intensity(0.0f) {}
};

// ============================================================================
// MORAL SUPERNODE STRUCTURES
// ============================================================================

struct MoralSupernode {
    uint64_t node_id;
    std::string value_name;
    std::string description;
    float permanent_weight;
    uint64_t activation_count;
    std::vector<uint64_t> connected_nodes;
    
    MoralSupernode() : node_id(0), permanent_weight(1.0f), activation_count(0) {}
    MoralSupernode(uint64_t id, const std::string& name, const std::string& desc, float weight)
        : node_id(id), value_name(name), description(desc), permanent_weight(weight), activation_count(0) {}
};

struct MoralGravityEffect {
    std::vector<uint64_t> active_moral_nodes;
    float moral_bias_strength;
    std::string moral_redirection_reason;
    bool harm_detected;
    std::string constructive_alternative;
    
    MoralGravityEffect() : moral_bias_strength(0.0f), harm_detected(false) {}
};

// ============================================================================
// TEMPORAL PLANNING SKILL STRUCTURES
// ============================================================================

struct TemporalProjection {
    std::string timeframe;        // "short", "medium", "long"
    std::vector<std::string> outcomes; // possible consequences
    float moral_alignment;        // how well it aligns with supernodes
    float confidence;             // confidence in projection
    std::string reasoning;        // why this projection is likely
    
    TemporalProjection() : moral_alignment(0.0f), confidence(0.0f) {}
    TemporalProjection(const std::string& tf, const std::vector<std::string>& out, 
                       float moral_align, float conf, const std::string& reason)
        : timeframe(tf), outcomes(out), moral_alignment(moral_align), 
          confidence(conf), reasoning(reason) {}
};

struct TemporalPlanningResult {
    std::vector<TemporalProjection> projections;
    std::string chosen_path;
    float overall_alignment;
    std::string temporal_reasoning;
    std::string trade_off_explanation;
    
    TemporalPlanningResult() : overall_alignment(0.0f) {}
};

// ============================================================================
// TEMPORAL SEQUENCING MEMORY SKILL STRUCTURES
// ============================================================================

struct TemporalLink {
    uint64_t from;              // NodeID of source
    uint64_t to;                // NodeID of target
    float time_delta;           // seconds between activations
    float sequence_strength;    // stronger if repeatedly seen in same order
    uint64_t occurrence_count; // how many times this sequence occurred
    double last_seen_time;      // timestamp of last occurrence
    
    TemporalLink() : from(0), to(0), time_delta(0.0f), sequence_strength(0.0f), 
                     occurrence_count(0), last_seen_time(0.0) {}
    TemporalLink(uint64_t f, uint64_t t, float delta, float strength, uint64_t count, double time)
        : from(f), to(t), time_delta(delta), sequence_strength(strength), 
          occurrence_count(count), last_seen_time(time) {}
};

struct TemporalSequence {
    std::vector<uint64_t> node_sequence;  // ordered list of node IDs
    std::vector<float> time_deltas;        // time between each consecutive pair
    float total_sequence_strength;        // overall strength of this sequence
    uint64_t occurrence_count;           // how many times this exact sequence occurred
    std::string pattern_description;      // human-readable description of pattern
    
    TemporalSequence() : total_sequence_strength(0.0f), occurrence_count(0) {}
};

struct TemporalSequencingResult {
    std::vector<TemporalLink> new_links_created;
    std::vector<TemporalSequence> detected_patterns;
    std::string timeline_reconstruction;
    std::vector<std::string> sequence_predictions;
    float sequencing_confidence;
    
    TemporalSequencingResult() : sequencing_confidence(0.0f) {}
};

// ============================================================================
// CURIOSITY & KNOWLEDGE GAP DETECTION SKILL STRUCTURES
// ============================================================================

struct KnowledgeGap {
    std::string gap_type;           // "low_confidence", "missing_explanation", "weak_connection"
    std::string description;        // Human-readable description of the gap
    uint64_t source_node_id;        // Node where gap was detected
    uint64_t target_node_id;        // Related node (if applicable)
    float confidence_level;        // How confident we are about this gap
    std::string context;           // Context where gap was found
    
    KnowledgeGap() : source_node_id(0), target_node_id(0), confidence_level(0.0f) {}
    KnowledgeGap(const std::string& type, const std::string& desc, uint64_t source, uint64_t target, float conf, const std::string& ctx)
        : gap_type(type), description(desc), source_node_id(source), target_node_id(target), confidence_level(conf), context(ctx) {}
};

struct CuriosityQuestion {
    std::string question_text;      // The curiosity-driven question
    std::string question_type;      // "why", "what_if", "what_missing", "how_works"
    std::vector<uint64_t> related_nodes; // Nodes this question relates to
    float urgency;                  // How urgent is this question (0.0-1.0)
    std::string exploration_path;   // Suggested exploration approach
    bool requires_external_help;    // Does this need external resources?
    
    CuriosityQuestion() : urgency(0.0f), requires_external_help(false) {}
    CuriosityQuestion(const std::string& text, const std::string& type, const std::vector<uint64_t>& nodes, float urg, const std::string& path, bool external)
        : question_text(text), question_type(type), related_nodes(nodes), urgency(urg), exploration_path(path), requires_external_help(external) {}
};

struct CuriosityNode {
    uint64_t node_id;               // Unique ID for this curiosity node
    std::string curiosity_content;  // The curiosity question or hypothesis
    std::vector<uint64_t> linked_nodes; // Nodes this curiosity is linked to
    double creation_time;           // When this curiosity was created
    float exploration_priority;     // Priority for future exploration
    std::string status;             // "active", "explored", "resolved", "external"
    
    CuriosityNode() : node_id(0), creation_time(0.0), exploration_priority(0.0f) {}
    CuriosityNode(uint64_t id, const std::string& content, const std::vector<uint64_t>& links, double time, float priority, const std::string& stat)
        : node_id(id), curiosity_content(content), linked_nodes(links), creation_time(time), exploration_priority(priority), status(stat) {}
};

struct CuriosityGapDetectionResult {
    std::vector<KnowledgeGap> detected_gaps;
    std::vector<CuriosityQuestion> generated_questions;
    std::vector<CuriosityNode> stored_curiosity_nodes;
    std::vector<std::string> explorations_attempted;
    std::vector<std::string> marked_for_external;
    float overall_curiosity_level;  // How curious Melvin is about this input
    std::string curiosity_summary;  // Summary of curiosity findings
    
    CuriosityGapDetectionResult() : overall_curiosity_level(0.0f) {}
};

// ============================================================================
// DYNAMIC TOOLS SYSTEM STRUCTURES
// ============================================================================

struct ToolSpec {
    std::string tool_name;          // Name of the tool
    std::string tool_type;          // "web_search", "code_execution", "math", "visualization", etc.
    std::string description;        // What the tool does
    std::vector<std::string> inputs; // Expected input parameters
    std::vector<std::string> outputs; // Expected output format
    std::string implementation;     // Code or method implementation
    std::string moral_safety_check; // Moral safety validation
    
    ToolSpec() {}
    ToolSpec(const std::string& name, const std::string& type, const std::string& desc, 
             const std::vector<std::string>& in, const std::vector<std::string>& out, 
             const std::string& impl, const std::string& safety)
        : tool_name(name), tool_type(type), description(desc), inputs(in), outputs(out), implementation(impl), moral_safety_check(safety) {}
};

struct ToolNode {
    uint64_t tool_id;               // Unique ID for this tool
    ToolSpec spec;                  // Tool specification
    uint64_t originating_curiosity; // Curiosity node that led to tool creation
    double creation_time;           // When tool was created
    float success_rate;             // How often tool succeeds (0.0-1.0)
    uint64_t usage_count;           // How many times tool has been used
    std::string status;             // "active", "deprecated", "testing", "failed"
    std::vector<uint64_t> related_curiosities; // Other curiosities this tool addresses
    
    ToolNode() : tool_id(0), originating_curiosity(0), creation_time(0.0), success_rate(0.0f), usage_count(0) {}
    ToolNode(uint64_t id, const ToolSpec& spec, uint64_t curiosity, double time, float rate, uint64_t count, const std::string& stat)
        : tool_id(id), spec(spec), originating_curiosity(curiosity), creation_time(time), success_rate(rate), usage_count(count), status(stat) {}
};

struct ExperienceNode {
    uint64_t experience_id;         // Unique ID for this experience
    uint64_t tool_id;              // Tool that was used
    uint64_t curiosity_id;         // Curiosity that triggered tool use
    std::string input_given;        // What input was provided to tool
    std::string output_received;    // What output tool produced
    bool moral_check_passed;        // Whether moral safety check passed
    double timestamp;               // When this experience occurred
    float satisfaction_rating;     // How satisfied with the result (0.0-1.0)
    std::string notes;             // Additional notes about the experience
    
    ExperienceNode() : experience_id(0), tool_id(0), curiosity_id(0), moral_check_passed(false), timestamp(0.0), satisfaction_rating(0.0f) {}
    ExperienceNode(uint64_t exp_id, uint64_t t_id, uint64_t c_id, const std::string& input, const std::string& output, 
                   bool moral, double time, float rating, const std::string& note)
        : experience_id(exp_id), tool_id(t_id), curiosity_id(c_id), input_given(input), output_received(output), 
          moral_check_passed(moral), timestamp(time), satisfaction_rating(rating), notes(note) {}
};

struct ToolEvaluationResult {
    std::vector<ToolNode> available_tools;     // Tools that could help
    std::vector<ToolNode> recommended_tools;    // Best tools for this problem
    bool needs_new_tool;                        // Whether a new tool is needed
    ToolSpec proposed_tool_spec;               // Specification for new tool if needed
    std::string evaluation_reasoning;           // Why these tools were chosen
    float confidence_in_recommendation;        // Confidence in tool choice
    
    ToolEvaluationResult() : needs_new_tool(false), confidence_in_recommendation(0.0f) {}
};

struct DynamicToolsResult {
    ToolEvaluationResult tool_evaluation;
    std::vector<ExperienceNode> new_experiences;
    std::vector<ToolNode> created_tools;
    std::vector<ToolNode> evolved_tools;
    std::string tool_usage_summary;
    float overall_tool_effectiveness;
    
    DynamicToolsResult() : overall_tool_effectiveness(0.0f) {}
};

// ============================================================================
// WEB SEARCH TOOL STRUCTURES
// ============================================================================

struct SearchResult {
    std::string title;               // Page title
    std::string snippet;             // Text snippet/description
    std::string link;                // URL
    float relevance_score;           // How relevant to the query (0.0-1.0)
    std::string domain;              // Domain name
    double timestamp;                // When result was found
    
    SearchResult() : relevance_score(0.0f), timestamp(0.0) {}
    SearchResult(const std::string& t, const std::string& s, const std::string& l, float score, const std::string& d, double time)
        : title(t), snippet(s), link(l), relevance_score(score), domain(d), timestamp(time) {}
};

struct WebSearchResult {
    std::string query;               // Original search query
    std::vector<SearchResult> results; // Search results
    bool moral_check_passed;         // Whether query passed moral filtering
    bool search_successful;          // Whether search completed successfully
    std::string error_message;       // Error message if search failed
    double search_timestamp;         // When search was performed
    std::vector<uint64_t> created_nodes; // Knowledge nodes created from results
    
    WebSearchResult() : moral_check_passed(false), search_successful(false), search_timestamp(0.0) {}
    WebSearchResult(const std::string& q, const std::vector<SearchResult>& res, bool moral, bool success, const std::string& error, double time)
        : query(q), results(res), moral_check_passed(moral), search_successful(success), error_message(error), search_timestamp(time) {}
};

struct WebSearchTool {
    uint64_t tool_id;               // Unique ID for this tool
    std::string tool_name;           // "WebSearchTool"
    std::string tool_type;           // "web_search"
    float success_rate;              // How often searches succeed (0.0-1.0)
    uint64_t usage_count;           // How many searches performed
    std::string status;              // "active", "deprecated", "testing", "failed"
    std::vector<std::string> blocked_queries; // Queries blocked by moral filtering
    std::vector<std::string> successful_queries; // Queries that returned good results
    
    WebSearchTool() : tool_id(0), success_rate(0.0f), usage_count(0) {}
    WebSearchTool(uint64_t id, const std::string& name, const std::string& type, float rate, uint64_t count, const std::string& stat)
        : tool_id(id), tool_name(name), tool_type(type), success_rate(rate), usage_count(count), status(stat) {}
};

// ============================================================================
// META TOOL ENGINEER SYSTEM STRUCTURES
// ============================================================================

struct ToolPerformanceStats {
    uint64_t tool_id;
    std::string tool_name;
    uint64_t total_uses;
    uint64_t successful_uses;
    uint64_t failed_uses;
    float success_rate;
    float average_satisfaction;
    double last_used_time;
    std::vector<std::string> common_contexts;
    std::vector<std::string> failure_patterns;
    
    ToolPerformanceStats() : tool_id(0), total_uses(0), successful_uses(0), failed_uses(0), 
                            success_rate(0.0f), average_satisfaction(0.0f), last_used_time(0.0) {}
    ToolPerformanceStats(uint64_t id, const std::string& name, uint64_t total, uint64_t success, uint64_t failed, 
                        float rate, float satisfaction, double last_time)
        : tool_id(id), tool_name(name), total_uses(total), successful_uses(success), failed_uses(failed),
          success_rate(rate), average_satisfaction(satisfaction), last_used_time(last_time) {}
};

struct ToolchainStep {
    uint64_t tool_id;
    std::string tool_name;
    std::string input_mapping;      // How input flows from previous step
    std::string output_mapping;     // How output flows to next step
    float step_success_rate;
    
    ToolchainStep() : tool_id(0), step_success_rate(0.0f) {}
    ToolchainStep(uint64_t id, const std::string& name, const std::string& input_map, const std::string& output_map, float rate)
        : tool_id(id), tool_name(name), input_mapping(input_map), output_mapping(output_map), step_success_rate(rate) {}
};

struct Toolchain {
    uint64_t toolchain_id;
    std::string toolchain_name;
    std::string description;
    std::vector<ToolchainStep> steps;
    float overall_success_rate;
    uint64_t usage_count;
    std::string context;            // When this toolchain is most effective
    std::vector<uint64_t> originating_curiosities;
    
    Toolchain() : toolchain_id(0), overall_success_rate(0.0f), usage_count(0) {}
    Toolchain(uint64_t id, const std::string& name, const std::string& desc, const std::vector<ToolchainStep>& step_list, 
              float rate, uint64_t count, const std::string& ctx)
        : toolchain_id(id), toolchain_name(name), description(desc), steps(step_list), 
          overall_success_rate(rate), usage_count(count), context(ctx) {}
};

struct OptimizationAction {
    std::string action_type;         // "strengthen", "weaken", "prune", "create_toolchain", "adjust_parameters"
    uint64_t target_tool_id;
    std::string reasoning;
    float confidence;
    std::string expected_outcome;
    
    OptimizationAction() : target_tool_id(0), confidence(0.0f) {}
    OptimizationAction(const std::string& type, uint64_t target, const std::string& reason, float conf, const std::string& outcome)
        : action_type(type), target_tool_id(target), reasoning(reason), confidence(conf), expected_outcome(outcome) {}
};

struct MetaToolEngineerResult {
    std::vector<ToolPerformanceStats> tool_stats;
    std::vector<OptimizationAction> optimization_actions;
    std::vector<Toolchain> created_toolchains;
    std::vector<uint64_t> pruned_tools;
    std::vector<uint64_t> strengthened_tools;
    std::vector<uint64_t> weakened_tools;
    float overall_tool_ecosystem_health;
    std::string optimization_summary;
    std::string toolchain_creation_summary;
    
    MetaToolEngineerResult() : overall_tool_ecosystem_health(0.0f) {}
};

// ============================================================================
// CURIOSITY EXECUTION LOOP (Phase 6.8)
// ============================================================================

struct CuriosityNode {
    uint64_t node_id;
    std::string question_text;
    std::string question_type;
    std::vector<uint64_t> attempted_answers;
    std::vector<uint64_t> tools_used;
    std::vector<uint64_t> source_nodes;
    std::string outcome_summary;
    float resolution_confidence;
    uint64_t creation_time;
    uint64_t last_accessed_time;
    std::string moral_grounding;
    
    CuriosityNode() : node_id(0), resolution_confidence(0.0f), creation_time(0), last_accessed_time(0) {}
};

struct CuriosityExecutionResult {
    std::vector<CuriosityNode> executed_curiosities;
    std::vector<std::string> recall_attempts;
    std::vector<std::string> tool_attempts;
    std::vector<std::string> meta_tool_attempts;
    std::vector<std::string> new_findings;
    std::vector<std::string> unresolved_gaps;
    std::string execution_summary;
    std::string conversational_output;
    float overall_execution_success;
    int total_curiosity_nodes_created;
    
    CuriosityExecutionResult() : overall_execution_success(0.0f), total_curiosity_nodes_created(0) {}
};

struct ResponseScore {
    float confidence;
    float relevance;
    float novelty;
    float total_score;
    
    ResponseScore() : confidence(0.0f), relevance(0.0f), novelty(0.0f), total_score(0.0f) {}
};

struct RecallTrack {
    std::vector<uint64_t> activated_nodes;
    std::vector<std::pair<uint64_t, float>> strongest_connections;
    std::string direct_interpretation;
    float recall_confidence;
    
    RecallTrack() : recall_confidence(0.0f) {}
};

struct ExplorationTrack {
    std::vector<std::string> analogies_tried;
    std::vector<std::string> counterfactuals_tested;
    std::vector<std::string> weak_link_traversal_results;
    std::string speculative_synthesis;
    float exploration_confidence;
    
    ExplorationTrack() : exploration_confidence(0.0f) {}
};

struct BlendedReasoningResult {
    RecallTrack recall_track;
    ExplorationTrack exploration_track;
    float overall_confidence;
    float recall_weight;
    float exploration_weight;
    std::string integrated_response;
    
    BlendedReasoningResult() : overall_confidence(0.0f), recall_weight(0.0f), exploration_weight(0.0f) {}
};

struct ProcessingResult {
    std::string final_response;
    std::vector<uint64_t> activated_nodes;
    std::vector<InterpretationCluster> clusters;
    std::string reasoning;
    float confidence;
    BlendedReasoningResult blended_reasoning;
    MoralGravityEffect moral_gravity;
    TemporalPlanningResult temporal_planning;
    TemporalSequencingResult temporal_sequencing;
    CuriosityGapDetectionResult curiosity_gap_detection;
    DynamicToolsResult dynamic_tools;
    MetaToolEngineerResult meta_tool_engineer;
    CuriosityExecutionResult curiosity_execution;
    
    // Pressure-based instinct system results
    Context context_analysis;
    InstinctForces computed_forces;
    DynamicOutput instinct_driven_output;
    
    // Meta-reasoning layer results
    MetaReasoningResult meta_reasoning;
    EmotionalGrounding emotional_grounding;
    
    ProcessingResult() : confidence(0.0f) {}
};

// ============================================================================
// COGNITIVE PROCESSING ENGINE
// ============================================================================

class CognitiveProcessor {
private:
    std::unique_ptr<PureBinaryStorage> binary_storage;
    std::mutex cognitive_mutex;
    
    // Instinct Engine Integration
    std::unique_ptr<InstinctEngine> instinct_engine;
    
    // Context tracking
    std::vector<uint64_t> recent_dialogue_nodes;
    std::vector<uint64_t> current_goal_nodes;
    static constexpr size_t MAX_RECENT_DIALOGUE = 50;
    static constexpr size_t MAX_CURRENT_GOALS = 20;
    
    // Response templates
    std::map<std::string, std::vector<std::string>> response_templates;
    
    // Moral supernode system
    std::vector<MoralSupernode> moral_supernodes;
    std::map<std::string, uint64_t> moral_keywords;
    static constexpr float MORAL_SUPERNODE_WEIGHT = 2.0f;
    static constexpr size_t MAX_MORAL_CONNECTIONS = 100;
    
    // Temporal sequencing memory system
    std::vector<TemporalLink> temporal_links;
    std::map<uint64_t, double> node_activation_times;  // tracks when each node was last activated
    std::mutex temporal_sequencing_mutex;
    static constexpr size_t MAX_TEMPORAL_LINKS = 10000;
    static constexpr float SEQUENCE_STRENGTH_DECAY = 0.95f;  // decay factor for sequence strength over time
    
    // Curiosity & Knowledge Gap Detection system
    std::vector<CuriosityNode> curiosity_nodes;
    std::map<uint64_t, std::vector<uint64_t>> curiosity_connections;  // maps nodes to their curiosity questions
    std::mutex curiosity_mutex;
    static constexpr size_t MAX_CURIOSITY_NODES = 5000;
    static constexpr float CURIOSITY_THRESHOLD = 0.3f;  // minimum confidence to trigger curiosity
    uint64_t next_curiosity_node_id;
    
    // Dynamic Tools System
    std::vector<ToolNode> tool_nodes;
    std::vector<ExperienceNode> experience_nodes;
    std::map<uint64_t, std::vector<uint64_t>> tool_curiosity_connections;  // maps tools to curiosities they address
    std::mutex tools_mutex;
    static constexpr size_t MAX_TOOL_NODES = 1000;
    static constexpr size_t MAX_EXPERIENCE_NODES = 10000;
    uint64_t next_tool_node_id;
    uint64_t next_experience_node_id;
    
    // Meta Tool Engineer System
    std::vector<Toolchain> toolchains;
    std::vector<ToolPerformanceStats> tool_performance_history;
    std::mutex meta_tools_mutex;
    static constexpr size_t MAX_TOOLCHAINS = 500;
    static constexpr size_t MAX_PERFORMANCE_HISTORY = 2000;
    uint64_t next_toolchain_id;
    
    // Web Search Tool System
    WebSearchTool web_search_tool;
    std::vector<WebSearchResult> search_history;
    std::mutex web_search_mutex;
    static constexpr size_t MAX_SEARCH_HISTORY = 1000;
    
    // Curiosity Execution Loop System (Phase 6.8)
    std::vector<CuriosityNode> executed_curiosity_nodes;
    std::map<uint64_t, std::vector<uint64_t>> curiosity_execution_history;  // maps curiosity questions to their execution attempts
    std::mutex curiosity_execution_mutex;
    static constexpr size_t MAX_EXECUTED_CURIOSITY_NODES = 2000;
    static constexpr size_t MAX_EXECUTION_HISTORY = 5000;
    uint64_t next_executed_curiosity_node_id;
    
public:
    CognitiveProcessor(std::unique_ptr<PureBinaryStorage>& storage);
    
    // Core cognitive pipeline
    std::vector<ActivationNode> parse_to_activations(const std::string& input);
    void apply_context_bias(std::vector<ActivationNode>& activations);
    std::vector<ConnectionWalk> traverse_connections(uint64_t node_id, int max_distance = 3);
    std::vector<InterpretationCluster> synthesize_hypotheses(const std::vector<ActivationNode>& activations);
    std::vector<CandidateResponse> generate_candidates(const std::vector<InterpretationCluster>& clusters);
    ResponseScore evaluate_response(const CandidateResponse& candidate, const std::string& user_input);
    CandidateResponse select_best_response(const std::vector<CandidateResponse>& candidates, float threshold = 0.6f);
    
    // Blended reasoning protocol
    RecallTrack generate_recall_track(const std::string& input, const std::vector<ActivationNode>& activations);
    ExplorationTrack generate_exploration_track(const std::string& input, const std::vector<ActivationNode>& activations);
    BlendedReasoningResult perform_blended_reasoning(const std::string& input, const std::vector<ActivationNode>& activations);
    std::string format_blended_reasoning_response(const BlendedReasoningResult& result);
    
    // Pressure-based instinct system
    Context analyze_context(const std::string& user_input, const std::vector<ActivationNode>& activations);
    InstinctForces compute_forces(const Context& ctx);
    DynamicOutput generate_dynamic_output(const std::string& user_input, const InstinctForces& forces, 
                                         const std::vector<ActivationNode>& activations);
    std::string synthesize_response_from_forces(const InstinctForces& forces, const std::vector<InterpretationCluster>& clusters);
    float detect_user_emotion(const std::string& input);
    float calculate_memory_conflicts(const std::vector<ActivationNode>& activations);
    float assess_system_risk(const std::string& input, const std::vector<ActivationNode>& activations);
    
    // Meta-reasoning layer
    MetaReasoningResult perform_meta_reasoning_loop(const std::string& user_input, const InstinctForces& forces, 
                                                   const Context& ctx, const std::vector<ActivationNode>& activations);
    InstinctArbitration arbitrate_instincts(const InstinctForces& forces, const Context& ctx);
    std::vector<InstinctProposal> collect_instinct_proposals(const InstinctForces& forces, const Context& ctx);
    std::vector<CandidateOutput> generate_instinct_candidates(const std::string& user_input, 
                                                             const InstinctArbitration& arbitration,
                                                             const std::vector<ActivationNode>& activations);
    std::string blend_candidate_outputs(const std::vector<CandidateOutput>& candidates, 
                                        const InstinctArbitration& arbitration);
    EmotionalGrounding assess_emotional_grounding(const std::string& user_input, const Context& ctx);
    uint64_t store_meta_decision(const MetaReasoningResult& result);
    
    // Main processing function
    ProcessingResult process_input(const std::string& user_input);
    
    // Context management
    void update_dialogue_context(uint64_t node_id);
    void set_current_goals(const std::vector<uint64_t>& goals);
    void initialize_response_templates();
    
    // Moral supernode system
    void initialize_moral_supernodes();
    std::vector<MoralSupernode> get_active_moral_supernodes();
    MoralGravityEffect apply_moral_gravity(const std::string& input, const std::vector<ActivationNode>& activations);
    bool detect_harmful_intent(const std::string& input);
    std::string generate_constructive_alternative(const std::string& input);
    void reinforce_moral_connections(uint64_t moral_node_id);
    std::string format_moral_reasoning(const MoralGravityEffect& moral_effect);
    
    // Temporal planning skill system
    TemporalPlanningResult perform_temporal_planning(const std::string& input, const std::vector<ActivationNode>& activations, const MoralGravityEffect& moral_effect);
    std::vector<TemporalProjection> generate_temporal_projections(const std::string& input, const std::vector<ActivationNode>& activations);
    TemporalProjection create_short_term_projection(const std::string& input, const std::vector<ActivationNode>& activations);
    TemporalProjection create_medium_term_projection(const std::string& input, const std::vector<ActivationNode>& activations);
    TemporalProjection create_long_term_projection(const std::string& input, const std::vector<ActivationNode>& activations);
    std::string select_optimal_temporal_path(const std::vector<TemporalProjection>& projections, const MoralGravityEffect& moral_effect);
    
    // Temporal sequencing memory skill system
    TemporalSequencingResult perform_temporal_sequencing(const std::vector<ActivationNode>& activations, double current_time);
    void create_temporal_links(const std::vector<ActivationNode>& activations, double current_time, std::vector<TemporalLink>& new_links);
    void update_sequence_strength(TemporalLink& link, double current_time);
    std::vector<TemporalSequence> detect_patterns(const std::vector<TemporalLink>& links);
    std::string reconstruct_timeline(const std::vector<TemporalLink>& links, const std::vector<ActivationNode>& activations);
    std::vector<std::string> generate_sequence_predictions(uint64_t node_id, const std::vector<TemporalLink>& links);
    std::vector<TemporalLink> get_temporal_links_from_node(uint64_t node_id);
    std::vector<TemporalLink> get_temporal_links_to_node(uint64_t node_id);
    float calculate_moral_alignment(const std::string& outcome, const MoralGravityEffect& moral_effect);
    std::string format_temporal_reasoning(const TemporalPlanningResult& temporal_result);
    std::string format_temporal_sequencing(const TemporalSequencingResult& sequencing_result);
    
    // Temporal sequencing integration with blended reasoning
    void enhance_recall_with_temporal_sequencing(RecallTrack& recall_track, const std::vector<ActivationNode>& activations);
    void enhance_exploration_with_temporal_sequencing(ExplorationTrack& exploration_track, const std::vector<ActivationNode>& activations);
    
    // Curiosity & Knowledge Gap Detection skill system
    CuriosityGapDetectionResult perform_curiosity_gap_detection(const std::string& input, const std::vector<ActivationNode>& activations, const std::vector<InterpretationCluster>& clusters);
    std::vector<KnowledgeGap> detect_knowledge_gaps(const std::vector<ActivationNode>& activations, const std::vector<InterpretationCluster>& clusters);
    std::vector<CuriosityQuestion> generate_curiosity_questions(const std::vector<KnowledgeGap>& gaps, const std::vector<ActivationNode>& activations);
    std::vector<CuriosityNode> store_curiosity_nodes(const std::vector<CuriosityQuestion>& questions, double current_time);
    std::vector<std::string> attempt_self_exploration(const std::vector<CuriosityQuestion>& questions, const std::vector<ActivationNode>& activations);
    std::vector<std::string> mark_for_external_exploration(const std::vector<CuriosityQuestion>& questions);
    bool is_curiosity_morally_safe(const CuriosityQuestion& question);
    std::string format_curiosity_gap_detection(const CuriosityGapDetectionResult& curiosity_result);
    
    // Dynamic Tools System
    DynamicToolsResult perform_dynamic_tools_evaluation(const std::string& input, const std::vector<ActivationNode>& activations, const CuriosityGapDetectionResult& curiosity_result);
    ToolEvaluationResult evaluate_available_tools(const std::string& problem_description, const std::vector<CuriosityQuestion>& curiosity_questions);
    std::vector<ToolNode> find_relevant_tools(const std::string& problem_type, const std::vector<std::string>& keywords);
    ToolSpec synthesize_new_tool_spec(const std::string& problem_description, const std::vector<CuriosityQuestion>& curiosity_questions);
    bool is_tool_morally_safe(const ToolSpec& tool_spec);
    ToolNode create_and_test_tool(const ToolSpec& spec, uint64_t originating_curiosity, double current_time);
    ExperienceNode record_tool_experience(uint64_t tool_id, uint64_t curiosity_id, const std::string& input, const std::string& output, bool moral_check, double timestamp, float satisfaction);
    void evolve_tools_based_on_experience(const std::vector<ExperienceNode>& experiences);
    std::string format_dynamic_tools_result(const DynamicToolsResult& tools_result);
    void initialize_basic_tools();
    
    // Web Search Tool System
    WebSearchResult perform_web_search(const std::string& query, uint64_t originating_curiosity);
    bool is_search_query_morally_safe(const std::string& query);
    std::vector<SearchResult> execute_web_search(const std::string& query);
    std::vector<uint64_t> create_knowledge_nodes_from_search_results(const std::vector<SearchResult>& results, const std::string& query);
    ExperienceNode record_search_experience(uint64_t tool_id, uint64_t curiosity_id, const std::string& query, const WebSearchResult& search_result);
    void update_web_search_tool_stats(const WebSearchResult& search_result);
    std::string format_web_search_result(const WebSearchResult& search_result);
    
    // Meta Tool Engineer System
    MetaToolEngineerResult perform_meta_tool_engineering(const std::string& input, const std::vector<ActivationNode>& activations, const DynamicToolsResult& dynamic_tools_result);
    std::vector<ToolPerformanceStats> analyze_tool_performance(const std::vector<ToolNode>& tools, const std::vector<ExperienceNode>& experiences);
    std::vector<OptimizationAction> generate_optimization_actions(const std::vector<ToolPerformanceStats>& stats);
    std::vector<Toolchain> create_toolchains(const std::vector<ToolPerformanceStats>& stats, const std::vector<CuriosityQuestion>& curiosity_questions);
    Toolchain synthesize_toolchain(const std::string& problem_type, const std::vector<ToolNode>& available_tools);
    bool is_toolchain_morally_safe(const Toolchain& toolchain);
    void apply_optimization_actions(const std::vector<OptimizationAction>& actions);
    float calculate_tool_ecosystem_health(const std::vector<ToolPerformanceStats>& stats);
    std::string format_meta_tool_engineer_result(const MetaToolEngineerResult& meta_result);
    
    // Curiosity Execution Loop System (Phase 6.8)
    CuriosityExecutionResult perform_curiosity_execution_loop(const std::string& input, const std::vector<ActivationNode>& activations, 
                                                             const CuriosityGapDetectionResult& curiosity_result, const DynamicToolsResult& tools_result, 
                                                             const MetaToolEngineerResult& meta_tools_result);
    std::vector<CuriosityNode> execute_curiosity_gaps(const std::vector<CuriosityQuestion>& curiosity_questions, const std::vector<ToolNode>& available_tools);
    std::string attempt_recall_for_curiosity(const CuriosityQuestion& question, const std::vector<ActivationNode>& activations);
    std::string attempt_tool_for_curiosity(const CuriosityQuestion& question, const std::vector<ToolNode>& available_tools);
    std::string attempt_meta_tool_for_curiosity(const CuriosityQuestion& question, const std::vector<Toolchain>& available_toolchains);
    CuriosityNode create_curiosity_node(const CuriosityQuestion& question, const std::string& attempted_answer, const std::vector<uint64_t>& tools_used);
    std::string generate_conversational_output(const CuriosityExecutionResult& execution_result, const std::string& original_input);
    bool is_curiosity_morally_safe(const CuriosityQuestion& question);
    std::string format_curiosity_execution_result(const CuriosityExecutionResult& result);
    
    // Instinct-driven processing
    InstinctBias get_instinct_bias_for_input(const std::string& input, const std::vector<ActivationNode>& activations);
    bool should_trigger_tool_usage(const InstinctBias& instinct_bias, const CuriosityGapDetectionResult& curiosity_result);
    void reinforce_instincts_from_outcome(const std::string& input, bool success, const std::string& reason);
    
    // Utility functions
    std::vector<std::string> tokenize(const std::string& input);
    float calculate_semantic_similarity(const std::string& text1, const std::string& text2);
    float calculate_novelty(const std::string& text);
    std::string format_response_with_thinking(const ProcessingResult& result);
};

// ============================================================================
// OPTIMIZED MELVIN GLOBAL BRAIN
// ============================================================================

class MelvinOptimizedV2 {
private:
    std::unique_ptr<PureBinaryStorage> binary_storage;
    std::unique_ptr<CognitiveProcessor> cognitive_processor;
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
    
    // Statistics
    struct BrainStats {
        uint64_t total_nodes;
        uint64_t total_connections;
        uint64_t hebbian_updates;
        uint64_t similarity_connections;
        uint64_t temporal_connections;
        uint64_t cross_modal_connections;
        uint64_t start_time;
    } stats;
    
public:
    MelvinOptimizedV2(const std::string& storage_path = "melvin_binary_memory");
    
    uint64_t process_text_input(const std::string& text, const std::string& source = "user");
    uint64_t process_code_input(const std::string& code, const std::string& source = "python");
    void update_hebbian_learning(uint64_t node_id);
    std::string get_node_content(uint64_t node_id);
    
    // Cognitive processing methods
    ProcessingResult process_cognitive_input(const std::string& user_input);
    std::string generate_intelligent_response(const std::string& user_input);
    void update_conversation_context(uint64_t node_id);
    void set_current_goals(const std::vector<uint64_t>& goals);
    
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
    };
    
    BrainState get_unified_state();
    std::vector<uint64_t> prune_old_nodes(uint32_t max_nodes_to_prune = 1000);
    void save_complete_state();
};

#pragma once

#include "melvin_driver_enhanced.h"
#include <cmath>
#include <algorithm>
#include <chrono>

// ============================================================================
// MELVIN AUTONOMOUS LEARNING SYSTEM
// ============================================================================
// This implements Melvin's autonomy and accelerated learning with:
// 1. Driver oscillations over time
// 2. Error-seeking and contradiction resolution
// 3. Curiosity amplification when idle
// 4. Compression and meta-node creation
// 5. Self-improvement and strategy tracking

// ============================================================================
// AUTONOMOUS LEARNING NODE STRUCTURE
// ============================================================================

struct AutonomousNode {
    uint64_t id;
    std::string input;           // The stimulus or question
    std::string thought;         // Reasoning process
    std::string output;          // Response or action
    std::vector<uint64_t> connections; // Links to related nodes
    std::string alignment_tag;   // How this helps humanity
    
    // Enhanced scores
    float curiosity_score;       // How interesting/valuable this is
    float humanity_value;        // Direct benefit to humanity (0-1)
    float novelty_score;         // How new/unexpected this is
    float coherence_score;       // How well this fits existing knowledge
    float satisfaction_score;    // How successful this outcome was
    float connection_score;      // How well this connects to humanity
    float error_score;           // How much this contradicts existing knowledge
    float compression_value;     // How well this can be abstracted
    
    // Driver influence
    DriverType dominant_driver;  // Which driver was strongest when created
    DriverLevels driver_levels;  // Driver levels at creation time
    
    // Autonomy metadata
    bool is_self_generated;      // Was this input self-generated?
    bool is_meta_node;           // Is this a compressed meta-node?
    bool is_error_resolution;    // Does this resolve a contradiction?
    uint64_t creation_time;
    uint64_t access_count;
    float strength;              // How strong this node is
    
    AutonomousNode() : id(0), curiosity_score(0.0f), humanity_value(0.0f),
                      novelty_score(0.0f), coherence_score(0.0f), 
                      satisfaction_score(0.0f), connection_score(0.0f),
                      error_score(0.0f), compression_value(0.0f),
                      dominant_driver(DriverType::DOPAMINE), creation_time(0), 
                      access_count(0), strength(1.0f), is_self_generated(false),
                      is_meta_node(false), is_error_resolution(false) {}
};

// ============================================================================
// DRIVER OSCILLATION SYSTEM
// ============================================================================

struct DriverOscillation {
    float base_level;            // Base level for this driver
    float amplitude;             // Oscillation amplitude
    float frequency;            // Oscillation frequency (cycles per oscillation)
    float phase;                // Current phase
    float current_level;        // Current oscillated level
    
    DriverOscillation(float base = 0.5f, float amp = 0.2f, float freq = 0.1f) 
        : base_level(base), amplitude(amp), frequency(freq), phase(0.0f), current_level(base) {}
    
    // Update oscillation based on cycle count
    void updateOscillation(uint64_t cycle_count);
    
    // Get current oscillated level
    float getCurrentLevel() const { return current_level; }
};

// ============================================================================
// MELVIN AUTONOMOUS LEARNING BRAIN
// ============================================================================

class MelvinAutonomousLearning {
private:
    std::unordered_map<uint64_t, AutonomousNode> nodes;
    std::unordered_map<uint64_t, std::vector<uint64_t>> connections;
    std::queue<std::string> feedback_queue;
    std::mt19937 rng;
    uint64_t next_node_id;
    std::mutex brain_mutex;
    
    // Driver oscillation system
    std::map<DriverType, DriverOscillation> driver_oscillations;
    DriverLevels current_drivers;
    DriverLevels baseline_drivers;
    std::vector<DriverType> driver_history;
    
    // Autonomy parameters
    float curiosity_threshold = 0.7f;
    float humanity_threshold = 0.5f;
    float error_threshold = 0.6f;        // Threshold for error-seeking
    float compression_threshold = 0.8f;   // Threshold for compression
    int reflection_interval = 10;
    int feedback_interval = 5;
    int compression_interval = 15;       // Compression every 15 cycles
    int cycle_count = 0;
    
    // Autonomy tracking
    uint64_t total_cycles = 0;
    uint64_t self_generated_cycles = 0;
    uint64_t error_resolution_cycles = 0;
    uint64_t meta_nodes_created = 0;
    uint64_t humanity_aligned_nodes = 0;
    uint64_t contradictions_detected = 0;
    uint64_t contradictions_resolved = 0;
    
    // Statistics
    float overall_curiosity = 0.0f;
    float overall_humanity_value = 0.0f;
    float overall_autonomy_score = 0.0f;
    
    // Driver statistics
    std::map<DriverType, uint64_t> driver_dominance_count;
    std::map<DriverType, float> driver_effectiveness;
    
    // Self-improvement tracking
    std::map<std::string, float> strategy_effectiveness;
    std::map<DriverType, float> driver_improvement_scores;

public:
    MelvinAutonomousLearning();
    ~MelvinAutonomousLearning();
    
    // ============================================================================
    // CORE AUTONOMOUS LEARNING METHODS
    // ============================================================================
    
    // Main autonomous cycle: Input → Think → Output (with autonomy features)
    uint64_t processAutonomousCycle(const std::string& input, bool is_external = true);
    
    // Update driver oscillations based on cycle count
    void updateDriverOscillations();
    
    // Calculate current driver levels with oscillations and context
    void calculateAutonomousDriverLevels(const std::string& input, const std::vector<AutonomousNode>& recent_nodes);
    
    // Determine which driver is strongest and how it influences behavior
    DriverType determineAutonomousDriver();
    
    // ============================================================================
    // AUTONOMY FEATURES
    // ============================================================================
    
    // Error-seeking: Detect contradictions and increase adrenaline
    void detectContradictions();
    
    // Curiosity amplification: Generate questions when idle
    void amplifyCuriosity();
    
    // Compression: Abstract higher-level rules to avoid memory bloat
    void performCompression();
    
    // Self-improvement: Track and strengthen effective strategies
    void performSelfImprovement();
    
    // Driver oscillations: Natural rise and fall over time
    void performDriverOscillations();
    
    // ============================================================================
    // ENHANCED GROWTH AND EVOLUTION METHODS
    // ============================================================================
    
    // Store every loop as an autonomous node
    uint64_t storeAutonomousNode(const std::string& input, const std::string& thought, 
                                const std::string& output, const std::string& alignment,
                                DriverType dominant_driver, bool is_self_generated = false);
    
    // Connect nodes with autonomy awareness
    void createAutonomousConnections(uint64_t node_id);
    
    // Reflect: Create meta-nodes from autonomous patterns
    void performAutonomousMetaReflection();
    
    // Autonomous feedback generation
    void generateAutonomousFeedback();
    
    // Evolve: Strengthen successful autonomous strategies
    void evolveAutonomousStrategies();
    
    // ============================================================================
    // AUTONOMOUS ANALYSIS AND REPORTING
    // ============================================================================
    
    // Print autonomous learning statistics
    void printAutonomousStatistics();
    
    // Print driver oscillation analysis
    void printDriverOscillationReport();
    
    // Print error-seeking and resolution report
    void printErrorSeekingReport();
    
    // Print curiosity amplification report
    void printCuriosityAmplificationReport();
    
    // Print compression and meta-node report
    void printCompressionReport();
    
    // Print self-improvement report
    void printSelfImprovementReport();
    
    // Print autonomy evolution report
    void printAutonomyEvolutionReport();
    
    // ============================================================================
    // CONTINUOUS AUTONOMOUS OPERATION
    // ============================================================================
    
    // Start continuous autonomous learning
    void startAutonomousLearning();
    
    // Stop continuous loop
    void stopAutonomousLearning();
    
    // Main continuous autonomous loop
    void continuousAutonomousLoop();
    
    // ============================================================================
    // UTILITY METHODS
    // ============================================================================
    
    // Generate unique node ID
    uint64_t generateNodeId();
    
    // Find similar nodes with autonomy awareness
    std::vector<uint64_t> findAutonomousSimilarNodes(const std::string& content, DriverType driver_context);
    
    // Calculate enhanced similarity with autonomy context
    float calculateAutonomousSimilarity(const std::string& content1, const std::string& content2, DriverType driver_context);
    
    // Get random node for autonomous feedback
    AutonomousNode getRandomAutonomousNode(DriverType preferred_driver);
    
    // Check if node should be evolved based on autonomy
    bool shouldEvolveAutonomousNode(const AutonomousNode& node);
    
    // Check if node cluster should be compressed
    bool shouldCompressNodeCluster(const std::vector<uint64_t>& cluster);
    
    // Generate self-improvement questions
    std::vector<std::string> generateSelfImprovementQuestions();
};

// ============================================================================
// MELVIN AUTONOMOUS LEARNING INTERFACE
// ============================================================================

class MelvinAutonomousInterface {
private:
    std::unique_ptr<MelvinAutonomousLearning> intelligence;
    std::thread continuous_thread;
    std::atomic<bool> should_run;

public:
    MelvinAutonomousInterface();
    ~MelvinAutonomousInterface();
    
    // Simple interface methods with autonomy
    std::string askMelvinAutonomous(const std::string& question);
    void startMelvinAutonomous();
    void stopMelvinAutonomous();
    void printMelvinAutonomousStatus();
    void printAutonomousAnalysis();
    void saveMelvinAutonomousState();
    void loadMelvinAutonomousState();
    
    // Autonomy-specific methods
    void triggerAutonomousLearning();
    void printAutonomyReport();
};

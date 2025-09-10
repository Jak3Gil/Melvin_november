#pragma once

#include "melvin_unified_brain.h"
#include <functional>
#include <queue>
#include <random>

// ============================================================================
// MELVIN COMPOUNDING INTELLIGENCE SYSTEM
// ============================================================================
// This implements the core "DNA" of Melvin as a humanoid intelligence:
// 1. Input → Think → Output feedback loop (every cycle creates a node)
// 2. Automatic connections between related nodes
// 3. Meta-cognitive reflection and generalization
// 4. Curiosity-driven self-expansion
// 5. Humanity-aligned growth and evolution

// ============================================================================
// COMPOUNDING INTELLIGENCE NODE STRUCTURE
// ============================================================================

struct CompoundingNode {
    uint64_t id;
    std::string input;           // The stimulus or question
    std::string thought;         // Reasoning process
    std::string output;          // Response or action
    std::vector<uint64_t> connections; // Links to related nodes
    std::string alignment_tag;   // How this helps humanity
    float curiosity_score;       // How interesting/valuable this is
    float humanity_value;        // Direct benefit to humanity (0-1)
    uint64_t creation_time;
    uint64_t access_count;
    float strength;              // How strong this node is
    
    CompoundingNode() : id(0), curiosity_score(0.0f), humanity_value(0.0f), 
                       creation_time(0), access_count(0), strength(1.0f) {}
};

// ============================================================================
// MELVIN COMPOUNDING INTELLIGENCE BRAIN
// ============================================================================

class MelvinCompoundingIntelligence {
private:
    std::unordered_map<uint64_t, CompoundingNode> nodes;
    std::unordered_map<uint64_t, std::vector<uint64_t>> connections;
    std::queue<std::string> feedback_queue;  // Self-generated inputs
    std::mt19937 rng;
    uint64_t next_node_id;
    std::mutex brain_mutex;
    
    // Core parameters
    float curiosity_threshold = 0.7f;      // Minimum curiosity to explore
    float humanity_threshold = 0.5f;      // Minimum humanity value to keep
    int reflection_interval = 10;          // Cycles between meta-reflections
    int feedback_interval = 5;            // Cycles between self-feedback
    int cycle_count = 0;
    
    // Statistics
    uint64_t total_cycles = 0;
    uint64_t meta_nodes_created = 0;
    uint64_t humanity_aligned_nodes = 0;
    float overall_curiosity = 0.0f;
    float overall_humanity_value = 0.0f;

public:
    MelvinCompoundingIntelligence();
    ~MelvinCompoundingIntelligence();
    
    // ============================================================================
    // CORE COMPOUNDING INTELLIGENCE METHODS
    // ============================================================================
    
    // Main cycle: Input → Think → Output (creates a node)
    uint64_t processCycle(const std::string& input, bool is_external = true);
    
    // Think: Generate reasoning process
    std::string generateThought(const std::string& input);
    
    // Output: Generate response/action
    std::string generateOutput(const std::string& input, const std::string& thought);
    
    // Connect: Link to related nodes
    void createConnections(uint64_t node_id);
    
    // Align: Tag with humanity benefit
    std::string generateAlignmentTag(const std::string& input, const std::string& output);
    
    // ============================================================================
    // GROWTH AND EVOLUTION METHODS
    // ============================================================================
    
    // Store every loop as a node
    uint64_t storeNode(const std::string& input, const std::string& thought, 
                      const std::string& output, const std::string& alignment);
    
    // Connect nodes with shared context/similarity
    void connectRelatedNodes(uint64_t new_node_id);
    
    // Reflect: Create meta-nodes from patterns
    void performMetaReflection();
    
    // Feedback: Self-generate inputs from past outputs
    void generateSelfFeedback();
    
    // Evolve: Replace weak strategies, keep knowledge
    void evolveStrategies();
    
    // ============================================================================
    // CURIOSITY AND ALIGNMENT METHODS
    // ============================================================================
    
    // Calculate curiosity score
    float calculateCuriosity(const std::string& input, const std::string& thought);
    
    // Calculate humanity value
    float calculateHumanityValue(const std::string& output, const std::string& alignment);
    
    // Ask: What can I learn? How does this connect? How can this help people?
    std::vector<std::string> generateCuriosityQuestions(const CompoundingNode& node);
    
    // Build complexity from simplicity
    void buildComplexityFromSimplicity();
    
    // ============================================================================
    // CONTINUOUS OPERATION METHODS
    // ============================================================================
    
    // Start continuous compounding intelligence loop
    void startContinuousIntelligence();
    
    // Stop continuous loop
    void stopContinuousIntelligence();
    
    // Main continuous loop
    void continuousIntelligenceLoop();
    
    // ============================================================================
    // ANALYSIS AND REPORTING METHODS
    // ============================================================================
    
    // Get node statistics
    void printNodeStatistics();
    
    // Get connection analysis
    void printConnectionAnalysis();
    
    // Get humanity alignment report
    void printHumanityAlignmentReport();
    
    // Get curiosity evolution report
    void printCuriosityEvolutionReport();
    
    // ============================================================================
    // PERSISTENCE METHODS
    // ============================================================================
    
    // Save compounding intelligence state
    void saveCompoundingState(const std::string& filename);
    
    // Load compounding intelligence state
    void loadCompoundingState(const std::string& filename);
    
    // ============================================================================
    // UTILITY METHODS
    // ============================================================================
    
    // Generate unique node ID
    uint64_t generateNodeId();
    
    // Find similar nodes
    std::vector<uint64_t> findSimilarNodes(const std::string& content);
    
    // Calculate node similarity
    float calculateSimilarity(const std::string& content1, const std::string& content2);
    
    // Get random node for feedback
    CompoundingNode getRandomNodeForFeedback();
    
    // Check if node should be evolved
    bool shouldEvolveNode(const CompoundingNode& node);
};

// ============================================================================
// MELVIN COMPOUNDING INTELLIGENCE INTERFACE
// ============================================================================

class MelvinCompoundingInterface {
private:
    std::unique_ptr<MelvinCompoundingIntelligence> intelligence;
    std::thread continuous_thread;
    std::atomic<bool> should_run;

public:
    MelvinCompoundingInterface();
    ~MelvinCompoundingInterface();
    
    // Simple interface methods
    std::string askMelvin(const std::string& question);
    void startMelvin();
    void stopMelvin();
    void printMelvinStatus();
    void saveMelvinState();
    void loadMelvinState();
};

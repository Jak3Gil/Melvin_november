#pragma once

#include "melvin_compounding_standalone.h"
#include <cmath>
#include <algorithm>

// ============================================================================
// MELVIN DRIVER-ENHANCED COMPOUNDING INTELLIGENCE
// ============================================================================
// This implements Melvin's driver system, modeled after human motivational chemicals.
// These drivers determine where Melvin's attention goes and what he does next.

// ============================================================================
// DRIVER SYSTEM - HUMAN MOTIVATIONAL CHEMICALS
// ============================================================================

enum class DriverType : uint8_t {
    DOPAMINE = 0,      // Curiosity & Novelty
    SEROTONIN = 1,     // Stability & Balance  
    ENDORPHINS = 2,    // Satisfaction & Reinforcement
    OXYTOCIN = 3,      // Connection & Alignment
    ADRENALINE = 4     // Urgency & Tension
};

struct DriverLevels {
    float dopamine = 0.5f;      // Curiosity & Novelty (0.0 - 1.0)
    float serotonin = 0.5f;     // Stability & Balance (0.0 - 1.0)
    float endorphins = 0.5f;    // Satisfaction & Reinforcement (0.0 - 1.0)
    float oxytocin = 0.5f;      // Connection & Alignment (0.0 - 1.0)
    float adrenaline = 0.5f;    // Urgency & Tension (0.0 - 1.0)
    
    // Get the strongest driver
    DriverType getStrongestDriver() const;
    
    // Get driver name
    std::string getDriverName(DriverType driver) const;
    
    // Get driver description
    std::string getDriverDescription(DriverType driver) const;
    
    // Normalize all levels to sum to 1.0
    void normalize();
    
    // Print current driver levels
    void printDriverLevels() const;
};

// ============================================================================
// ENHANCED COMPOUNDING NODE WITH DRIVERS
// ============================================================================

struct DriverEnhancedNode {
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
    
    // Driver influence
    DriverType dominant_driver;  // Which driver was strongest when created
    DriverLevels driver_levels;  // Driver levels at creation time
    
    // Metadata
    uint64_t creation_time;
    uint64_t access_count;
    float strength;              // How strong this node is
    
    DriverEnhancedNode() : id(0), curiosity_score(0.0f), humanity_value(0.0f),
                          novelty_score(0.0f), coherence_score(0.0f), 
                          satisfaction_score(0.0f), connection_score(0.0f),
                          dominant_driver(DriverType::DOPAMINE), creation_time(0), 
                          access_count(0), strength(1.0f) {}
};

// ============================================================================
// MELVIN DRIVER-ENHANCED COMPOUNDING INTELLIGENCE
// ============================================================================

class MelvinDriverEnhancedIntelligence {
private:
    std::unordered_map<uint64_t, DriverEnhancedNode> nodes;
    std::unordered_map<uint64_t, std::vector<uint64_t>> connections;
    std::queue<std::string> feedback_queue;
    std::mt19937 rng;
    uint64_t next_node_id;
    std::mutex brain_mutex;
    
    // Driver system
    DriverLevels current_drivers;
    DriverLevels baseline_drivers;
    std::vector<DriverType> driver_history;  // Track which drivers have been dominant
    
    // Core parameters
    float curiosity_threshold = 0.7f;
    float humanity_threshold = 0.5f;
    int reflection_interval = 10;
    int feedback_interval = 5;
    int cycle_count = 0;
    
    // Statistics
    uint64_t total_cycles = 0;
    uint64_t meta_nodes_created = 0;
    uint64_t humanity_aligned_nodes = 0;
    float overall_curiosity = 0.0f;
    float overall_humanity_value = 0.0f;
    
    // Driver statistics
    std::map<DriverType, uint64_t> driver_dominance_count;
    std::map<DriverType, float> driver_effectiveness;

public:
    MelvinDriverEnhancedIntelligence();
    ~MelvinDriverEnhancedIntelligence();
    
    // ============================================================================
    // CORE DRIVER-ENHANCED METHODS
    // ============================================================================
    
    // Main cycle: Input → Think → Output (creates a node with driver influence)
    uint64_t processDriverCycle(const std::string& input, bool is_external = true);
    
    // Calculate current driver levels based on context and history
    void calculateDriverLevels(const std::string& input, const std::vector<DriverEnhancedNode>& recent_nodes);
    
    // Determine which driver is strongest and how it influences behavior
    DriverType determineDominantDriver();
    
    // Generate input focus based on dominant driver
    std::string generateDriverFocusedInput(const std::string& original_input, DriverType dominant_driver);
    
    // Think: Generate reasoning process influenced by drivers
    std::string generateDriverInfluencedThought(const std::string& input, DriverType dominant_driver);
    
    // Output: Generate response/action influenced by drivers
    std::string generateDriverInfluencedOutput(const std::string& input, const std::string& thought, DriverType dominant_driver);
    
    // ============================================================================
    // DRIVER-SPECIFIC BEHAVIORS
    // ============================================================================
    
    // Dopamine-driven behavior: Curiosity & Novelty
    std::string dopamineBehavior(const std::string& input);
    
    // Serotonin-driven behavior: Stability & Balance
    std::string serotoninBehavior(const std::string& input);
    
    // Endorphins-driven behavior: Satisfaction & Reinforcement
    std::string endorphinsBehavior(const std::string& input);
    
    // Oxytocin-driven behavior: Connection & Alignment
    std::string oxytocinBehavior(const std::string& input);
    
    // Adrenaline-driven behavior: Urgency & Tension
    std::string adrenalineBehavior(const std::string& input);
    
    // ============================================================================
    // ENHANCED GROWTH AND EVOLUTION METHODS
    // ============================================================================
    
    // Store every loop as a node with driver influence
    uint64_t storeDriverNode(const std::string& input, const std::string& thought, 
                            const std::string& output, const std::string& alignment,
                            DriverType dominant_driver);
    
    // Connect nodes with driver-aware similarity
    void createDriverAwareConnections(uint64_t node_id);
    
    // Reflect: Create meta-nodes from driver patterns
    void performDriverMetaReflection();
    
    // Driver-aware feedback generation
    void generateDriverAwareFeedback();
    
    // Evolve: Strengthen successful driver strategies
    void evolveDriverStrategies();
    
    // ============================================================================
    // DRIVER ANALYSIS AND REPORTING
    // ============================================================================
    
    // Print current driver levels
    void printCurrentDrivers();
    
    // Print driver dominance statistics
    void printDriverDominanceReport();
    
    // Print driver effectiveness analysis
    void printDriverEffectivenessReport();
    
    // Print driver evolution over time
    void printDriverEvolutionReport();
    
    // ============================================================================
    // ENHANCED ANALYSIS METHODS
    // ============================================================================
    
    // Get enhanced node statistics
    void printEnhancedNodeStatistics();
    
    // Get driver-aware connection analysis
    void printDriverConnectionAnalysis();
    
    // Get humanity alignment with driver context
    void printDriverHumanityAlignmentReport();
    
    // ============================================================================
    // CONTINUOUS OPERATION WITH DRIVERS
    // ============================================================================
    
    // Start continuous driver-enhanced intelligence
    void startDriverIntelligence();
    
    // Stop continuous loop
    void stopDriverIntelligence();
    
    // Main continuous loop with driver awareness
    void continuousDriverLoop();
    
    // ============================================================================
    // UTILITY METHODS
    // ============================================================================
    
    // Generate unique node ID
    uint64_t generateNodeId();
    
    // Find similar nodes with driver awareness
    std::vector<uint64_t> findDriverSimilarNodes(const std::string& content, DriverType driver_context);
    
    // Calculate enhanced similarity with driver context
    float calculateDriverSimilarity(const std::string& content1, const std::string& content2, DriverType driver_context);
    
    // Get random node for driver-based feedback
    DriverEnhancedNode getRandomDriverNode(DriverType preferred_driver);
    
    // Check if node should be evolved based on driver effectiveness
    bool shouldEvolveDriverNode(const DriverEnhancedNode& node);
};

// ============================================================================
// MELVIN DRIVER-ENHANCED INTERFACE
// ============================================================================

class MelvinDriverInterface {
private:
    std::unique_ptr<MelvinDriverEnhancedIntelligence> intelligence;
    std::thread continuous_thread;
    std::atomic<bool> should_run;

public:
    MelvinDriverInterface();
    ~MelvinDriverInterface();
    
    // Simple interface methods with driver awareness
    std::string askMelvinWithDrivers(const std::string& question);
    void startMelvinDrivers();
    void stopMelvinDrivers();
    void printMelvinDriverStatus();
    void printDriverAnalysis();
    void saveMelvinDriverState();
    void loadMelvinDriverState();
};

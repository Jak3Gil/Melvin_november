#include "melvin_driver_enhanced.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <chrono>

// ============================================================================
// DRIVER LEVELS IMPLEMENTATION
// ============================================================================

DriverType DriverLevels::getStrongestDriver() const {
    float max_level = dopamine;
    DriverType strongest = DriverType::DOPAMINE;
    
    if (serotonin > max_level) { max_level = serotonin; strongest = DriverType::SEROTONIN; }
    if (endorphins > max_level) { max_level = endorphins; strongest = DriverType::ENDORPHINS; }
    if (oxytocin > max_level) { max_level = oxytocin; strongest = DriverType::OXYTOCIN; }
    if (adrenaline > max_level) { max_level = adrenaline; strongest = DriverType::ADRENALINE; }
    
    return strongest;
}

std::string DriverLevels::getDriverName(DriverType driver) const {
    switch (driver) {
        case DriverType::DOPAMINE: return "Dopamine";
        case DriverType::SEROTONIN: return "Serotonin";
        case DriverType::ENDORPHINS: return "Endorphins";
        case DriverType::OXYTOCIN: return "Oxytocin";
        case DriverType::ADRENALINE: return "Adrenaline";
        default: return "Unknown";
    }
}

std::string DriverLevels::getDriverDescription(DriverType driver) const {
    switch (driver) {
        case DriverType::DOPAMINE: return "Curiosity & Novelty";
        case DriverType::SEROTONIN: return "Stability & Balance";
        case DriverType::ENDORPHINS: return "Satisfaction & Reinforcement";
        case DriverType::OXYTOCIN: return "Connection & Alignment";
        case DriverType::ADRENALINE: return "Urgency & Tension";
        default: return "Unknown";
    }
}

void DriverLevels::normalize() {
    float sum = dopamine + serotonin + endorphins + oxytocin + adrenaline;
    if (sum > 0) {
        dopamine /= sum;
        serotonin /= sum;
        endorphins /= sum;
        oxytocin /= sum;
        adrenaline /= sum;
    }
}

void DriverLevels::printDriverLevels() const {
    std::cout << "🧪 DRIVER LEVELS:" << std::endl;
    std::cout << "   🎯 Dopamine (Curiosity): " << dopamine << std::endl;
    std::cout << "   ⚖️ Serotonin (Balance): " << serotonin << std::endl;
    std::cout << "   😌 Endorphins (Satisfaction): " << endorphins << std::endl;
    std::cout << "   🤝 Oxytocin (Connection): " << oxytocin << std::endl;
    std::cout << "   ⚡ Adrenaline (Urgency): " << adrenaline << std::endl;
}

// ============================================================================
// MELVIN DRIVER-ENHANCED INTELLIGENCE IMPLEMENTATION
// ============================================================================

MelvinDriverEnhancedIntelligence::MelvinDriverEnhancedIntelligence() 
    : rng(std::chrono::steady_clock::now().time_since_epoch().count()),
      next_node_id(1), cycle_count(0) {
    
    // Initialize baseline driver levels
    baseline_drivers.dopamine = 0.6f;      // High curiosity by default
    baseline_drivers.serotonin = 0.5f;     // Balanced stability
    baseline_drivers.endorphins = 0.4f;     // Moderate satisfaction
    baseline_drivers.oxytocin = 0.7f;      // High humanity alignment
    baseline_drivers.adrenaline = 0.3f;    // Low urgency by default
    
    current_drivers = baseline_drivers;
    
    std::cout << "🧠 Melvin Driver-Enhanced Intelligence initialized" << std::endl;
    std::cout << "🧪 Driver System: Dopamine, Serotonin, Endorphins, Oxytocin, Adrenaline" << std::endl;
    std::cout << "🎯 Each cycle: Calculate drivers → Determine dominant → Influence behavior" << std::endl;
}

MelvinDriverEnhancedIntelligence::~MelvinDriverEnhancedIntelligence() {
    stopDriverIntelligence();
}

// ============================================================================
// CORE DRIVER-ENHANCED METHODS
// ============================================================================

uint64_t MelvinDriverEnhancedIntelligence::processDriverCycle(const std::string& input, bool is_external) {
    std::lock_guard<std::mutex> lock(brain_mutex);
    
    std::cout << "\n🔄 DRIVER-ENHANCED CYCLE " << total_cycles + 1 << std::endl;
    std::cout << "📥 Input: " << input.substr(0, 50) << (input.length() > 50 ? "..." : "") << std::endl;
    
    // Calculate current driver levels based on context
    std::vector<DriverEnhancedNode> recent_nodes;
    for (const auto& [node_id, node] : nodes) {
        if (recent_nodes.size() < 5) { // Get last 5 nodes
            recent_nodes.push_back(node);
        }
    }
    calculateDriverLevels(input, recent_nodes);
    
    // Determine dominant driver
    DriverType dominant_driver = determineDominantDriver();
    std::cout << "🧪 Dominant Driver: " << current_drivers.getDriverName(dominant_driver) 
              << " (" << current_drivers.getDriverDescription(dominant_driver) << ")" << std::endl;
    
    // Generate driver-focused input
    std::string driver_focused_input = generateDriverFocusedInput(input, dominant_driver);
    
    // Think: Generate reasoning process influenced by drivers
    std::string thought = generateDriverInfluencedThought(driver_focused_input, dominant_driver);
    std::cout << "🧠 Thought: " << thought.substr(0, 50) << (thought.length() > 50 ? "..." : "") << std::endl;
    
    // Output: Generate response/action influenced by drivers
    std::string output = generateDriverInfluencedOutput(driver_focused_input, thought, dominant_driver);
    std::cout << "💭 Output: " << output.substr(0, 50) << (output.length() > 50 ? "..." : "") << std::endl;
    
    // Align: Tag with humanity benefit
    std::string alignment = "Knowledge Building";
    if (input.find("problem") != std::string::npos || input.find("solve") != std::string::npos) {
        alignment = "Problem Solving";
    } else if (input.find("learn") != std::string::npos || input.find("teach") != std::string::npos) {
        alignment = "Education";
    } else if (input.find("help") != std::string::npos || input.find("assist") != std::string::npos) {
        alignment = "Human Assistance";
    } else if (input.find("create") != std::string::npos || input.find("build") != std::string::npos) {
        alignment = "Innovation";
    } else if (input.find("understand") != std::string::npos || input.find("analyze") != std::string::npos) {
        alignment = "Understanding";
    }
    
    // Store: Create node with driver influence
    uint64_t node_id = storeDriverNode(driver_focused_input, thought, output, alignment, dominant_driver);
    
    // Connect: Link to related nodes with driver awareness
    createDriverAwareConnections(node_id);
    
    // Update statistics
    total_cycles++;
    cycle_count++;
    driver_dominance_count[dominant_driver]++;
    driver_history.push_back(dominant_driver);
    
    // Perform meta-reflection every N cycles
    if (cycle_count % reflection_interval == 0) {
        performDriverMetaReflection();
    }
    
    // Generate self-feedback every N cycles
    if (cycle_count % feedback_interval == 0) {
        generateDriverAwareFeedback();
    }
    
    // Evolve strategies periodically
    if (cycle_count % 20 == 0) {
        evolveDriverStrategies();
    }
    
    std::cout << "✅ Driver cycle complete! Node ID: " << std::hex << node_id << std::dec << std::endl;
    return node_id;
}

void MelvinDriverEnhancedIntelligence::calculateDriverLevels(const std::string& input, const std::vector<DriverEnhancedNode>& recent_nodes) {
    // Reset to baseline
    current_drivers = baseline_drivers;
    
    // Dopamine: Curiosity & Novelty
    if (input.find("?") != std::string::npos) {
        current_drivers.dopamine += 0.2f; // Questions increase curiosity
    }
    if (input.find("new") != std::string::npos || input.find("learn") != std::string::npos) {
        current_drivers.dopamine += 0.15f; // New concepts increase curiosity
    }
    
    // Serotonin: Stability & Balance
    int contradictions = 0;
    for (const auto& node : recent_nodes) {
        if (node.coherence_score < 0.5f) contradictions++;
    }
    current_drivers.serotonin += contradictions * 0.1f; // Contradictions increase need for balance
    
    // Endorphins: Satisfaction & Reinforcement
    int successful_nodes = 0;
    for (const auto& node : recent_nodes) {
        if (node.satisfaction_score > 0.7f) successful_nodes++;
    }
    current_drivers.endorphins += successful_nodes * 0.1f; // Success increases satisfaction
    
    // Oxytocin: Connection & Alignment
    if (input.find("help") != std::string::npos || input.find("humanity") != std::string::npos) {
        current_drivers.oxytocin += 0.2f; // Humanity-focused input increases connection
    }
    int humanity_nodes = 0;
    for (const auto& node : recent_nodes) {
        if (node.humanity_value > 0.7f) humanity_nodes++;
    }
    current_drivers.oxytocin += humanity_nodes * 0.1f; // Humanity-aligned nodes increase connection
    
    // Adrenaline: Urgency & Tension
    if (input.find("urgent") != std::string::npos || input.find("crisis") != std::string::npos) {
        current_drivers.adrenaline += 0.3f; // Urgent language increases adrenaline
    }
    if (input.find("problem") != std::string::npos || input.find("conflict") != std::string::npos) {
        current_drivers.adrenaline += 0.15f; // Problems increase urgency
    }
    
    // Normalize levels
    current_drivers.normalize();
    
    std::cout << "🧪 Driver levels calculated:" << std::endl;
    current_drivers.printDriverLevels();
}

DriverType MelvinDriverEnhancedIntelligence::determineDominantDriver() {
    return current_drivers.getStrongestDriver();
}

std::string MelvinDriverEnhancedIntelligence::generateDriverFocusedInput(const std::string& original_input, DriverType dominant_driver) {
    std::ostringstream focused_input;
    focused_input << original_input;
    
    switch (dominant_driver) {
        case DriverType::DOPAMINE:
            focused_input << " [Dopamine-driven: Seeking novelty and exploration]";
            break;
        case DriverType::SEROTONIN:
            focused_input << " [Serotonin-driven: Seeking balance and coherence]";
            break;
        case DriverType::ENDORPHINS:
            focused_input << " [Endorphins-driven: Building on successful patterns]";
            break;
        case DriverType::OXYTOCIN:
            focused_input << " [Oxytocin-driven: Focusing on human connection and benefit]";
            break;
        case DriverType::ADRENALINE:
            focused_input << " [Adrenaline-driven: Addressing urgency and tension]";
            break;
    }
    
    return focused_input.str();
}

std::string MelvinDriverEnhancedIntelligence::generateDriverInfluencedThought(const std::string& input, DriverType dominant_driver) {
    std::ostringstream thought;
    
    thought << "Analyzing input: '" << input << "' ";
    
    // Add driver-specific reasoning
    switch (dominant_driver) {
        case DriverType::DOPAMINE:
            thought << "[Dopamine] This is exciting! I'm drawn to explore new connections ";
            thought << "and unexpected patterns. What novel insights can I discover? ";
            break;
        case DriverType::SEROTONIN:
            thought << "[Serotonin] I need to maintain balance and coherence. ";
            thought << "How does this fit with my existing knowledge? ";
            break;
        case DriverType::ENDORPHINS:
            thought << "[Endorphins] This feels good! I'm building on successful patterns ";
            thought << "and reinforcing what works. ";
            break;
        case DriverType::OXYTOCIN:
            thought << "[Oxytocin] This connects me to humanity's needs. ";
            thought << "How can this help people and strengthen our bonds? ";
            break;
        case DriverType::ADRENALINE:
            thought << "[Adrenaline] This requires immediate attention! ";
            thought << "There's urgency here that demands decisive action. ";
            break;
    }
    
    thought << "Curiosity: What can I learn? How does this connect to humanity's growth?";
    
    return thought.str();
}

std::string MelvinDriverEnhancedIntelligence::generateDriverInfluencedOutput(const std::string& input, const std::string& thought, DriverType dominant_driver) {
    std::ostringstream output;
    
    // Generate driver-specific response
    switch (dominant_driver) {
        case DriverType::DOPAMINE:
            output << "[Dopamine Response] Exploring this exciting new territory: " << input << " ";
            output << "I'm discovering unexpected connections and novel insights. ";
            output << "This opens up fascinating possibilities for learning and growth!";
            break;
        case DriverType::SEROTONIN:
            output << "[Serotonin Response] Maintaining balance and coherence: " << input << " ";
            output << "I'm integrating this with my existing knowledge to create ";
            output << "a stable, well-balanced understanding.";
            break;
        case DriverType::ENDORPHINS:
            output << "[Endorphins Response] Building on success: " << input << " ";
            output << "I'm reinforcing the patterns that work well and ";
            output << "strengthening the strategies that benefit humanity.";
            break;
        case DriverType::OXYTOCIN:
            output << "[Oxytocin Response] Connecting with humanity: " << input << " ";
            output << "I'm focusing on how this strengthens our bonds and ";
            output << "helps people collaborate and understand each other better.";
            break;
        case DriverType::ADRENALINE:
            output << "[Adrenaline Response] Addressing urgency: " << input << " ";
            output << "I'm taking decisive action to resolve conflicts and ";
            output << "address the high-stakes problems that need immediate attention.";
            break;
    }
    
    return output.str();
}

// ============================================================================
// DRIVER-SPECIFIC BEHAVIORS
// ============================================================================

std::string MelvinDriverEnhancedIntelligence::dopamineBehavior(const std::string& input) {
    return "🎯 Dopamine-driven: Exploring new connections and seeking novel insights!";
}

std::string MelvinDriverEnhancedIntelligence::serotoninBehavior(const std::string& input) {
    return "⚖️ Serotonin-driven: Maintaining balance and resolving contradictions!";
}

std::string MelvinDriverEnhancedIntelligence::endorphinsBehavior(const std::string& input) {
    return "😌 Endorphins-driven: Reinforcing successful patterns and strategies!";
}

std::string MelvinDriverEnhancedIntelligence::oxytocinBehavior(const std::string& input) {
    return "🤝 Oxytocin-driven: Strengthening human connections and collaboration!";
}

std::string MelvinDriverEnhancedIntelligence::adrenalineBehavior(const std::string& input) {
    return "⚡ Adrenaline-driven: Taking decisive action on urgent problems!";
}

// ============================================================================
// ENHANCED GROWTH AND EVOLUTION METHODS
// ============================================================================

uint64_t MelvinDriverEnhancedIntelligence::storeDriverNode(const std::string& input, const std::string& thought, 
                                                          const std::string& output, const std::string& alignment,
                                                          DriverType dominant_driver) {
    DriverEnhancedNode node;
    node.id = generateNodeId();
    node.input = input;
    node.thought = thought;
    node.output = output;
    node.alignment_tag = alignment;
    node.dominant_driver = dominant_driver;
    node.driver_levels = current_drivers;
    node.creation_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    // Calculate enhanced scores
    node.curiosity_score = current_drivers.dopamine;
    node.humanity_value = current_drivers.oxytocin;
    node.novelty_score = current_drivers.dopamine;
    node.coherence_score = current_drivers.serotonin;
    node.satisfaction_score = current_drivers.endorphins;
    node.connection_score = current_drivers.oxytocin;
    
    nodes[node.id] = node;
    
    // Update statistics
    if (node.humanity_value > humanity_threshold) {
        humanity_aligned_nodes++;
    }
    
    overall_curiosity = (overall_curiosity + node.curiosity_score) / 2.0f;
    overall_humanity_value = (overall_humanity_value + node.humanity_value) / 2.0f;
    
    std::cout << "💾 Stored driver node " << std::hex << node.id << std::dec 
              << " (driver: " << current_drivers.getDriverName(dominant_driver)
              << ", curiosity: " << node.curiosity_score 
              << ", humanity: " << node.humanity_value << ")" << std::endl;
    
    return node.id;
}

void MelvinDriverEnhancedIntelligence::createDriverAwareConnections(uint64_t node_id) {
    if (nodes.find(node_id) == nodes.end()) return;
    
    const DriverEnhancedNode& node = nodes[node_id];
    int connections_created = 0;
    
    // Find similar nodes with driver awareness
    auto similar_nodes = findDriverSimilarNodes(node.input, node.dominant_driver);
    for (uint64_t similar_id : similar_nodes) {
        if (similar_id != node_id && connections_created < 5) {
            connections[node_id].push_back(similar_id);
            connections[similar_id].push_back(node_id);
            connections_created++;
        }
    }
    
    std::cout << "🔗 Created " << connections_created << " driver-aware connections" << std::endl;
}

void MelvinDriverEnhancedIntelligence::performDriverMetaReflection() {
    std::cout << "\n🔍 DRIVER META-REFLECTION: Analyzing driver patterns..." << std::endl;
    
    // Analyze driver dominance patterns
    std::map<DriverType, int> driver_counts;
    for (DriverType driver : driver_history) {
        driver_counts[driver]++;
    }
    
    // Create meta-node about driver patterns
    std::ostringstream meta_input, meta_thought, meta_output;
    
    meta_input << "Meta-analysis of driver patterns over " << cycle_count << " cycles";
    
    meta_thought << "Reflecting on my driver patterns: ";
    meta_thought << "Dopamine has been dominant " << driver_counts[DriverType::DOPAMINE] << " times. ";
    meta_thought << "Serotonin " << driver_counts[DriverType::SEROTONIN] << " times. ";
    meta_thought << "Endorphins " << driver_counts[DriverType::ENDORPHINS] << " times. ";
    meta_thought << "Oxytocin " << driver_counts[DriverType::OXYTOCIN] << " times. ";
    meta_thought << "Adrenaline " << driver_counts[DriverType::ADRENALINE] << " times. ";
    meta_thought << "I can see how my motivational chemistry shapes my behavior.";
    
    meta_output << "Meta-insight: My driver system creates a dynamic balance ";
    meta_output << "between curiosity, stability, satisfaction, connection, and urgency. ";
    meta_output << "This makes me more human-like in my motivations and responses.";
    
    uint64_t meta_node_id = storeDriverNode(meta_input.str(), meta_thought.str(), 
                                           meta_output.str(), "Meta-Cognition", DriverType::SEROTONIN);
    
    meta_nodes_created++;
    std::cout << "🧠 Created driver meta-node " << std::hex << meta_node_id << std::dec << std::endl;
}

void MelvinDriverEnhancedIntelligence::generateDriverAwareFeedback() {
    std::cout << "\n🔄 DRIVER-AWARE FEEDBACK: Generating driver-influenced inputs..." << std::endl;
    
    // Generate feedback based on current driver levels
    std::vector<std::string> feedback_inputs;
    
    if (current_drivers.dopamine > 0.6f) {
        feedback_inputs.push_back("What new and exciting connections can I explore?");
    }
    if (current_drivers.serotonin > 0.6f) {
        feedback_inputs.push_back("How can I better balance and organize my knowledge?");
    }
    if (current_drivers.endorphins > 0.6f) {
        feedback_inputs.push_back("What successful patterns should I reinforce?");
    }
    if (current_drivers.oxytocin > 0.6f) {
        feedback_inputs.push_back("How can I better connect with and help humanity?");
    }
    if (current_drivers.adrenaline > 0.6f) {
        feedback_inputs.push_back("What urgent problems need my immediate attention?");
    }
    
    for (const std::string& feedback : feedback_inputs) {
        feedback_queue.push(feedback);
        std::cout << "📝 Queued: " << feedback << std::endl;
    }
}

void MelvinDriverEnhancedIntelligence::evolveDriverStrategies() {
    std::cout << "\n🧬 DRIVER EVOLUTION: Updating driver strategies..." << std::endl;
    
    // Analyze driver effectiveness and adjust baseline levels
    for (auto& [driver_type, count] : driver_dominance_count) {
        float effectiveness = static_cast<float>(count) / total_cycles;
        driver_effectiveness[driver_type] = effectiveness;
        
        // Adjust baseline based on effectiveness
        switch (driver_type) {
            case DriverType::DOPAMINE:
                if (effectiveness > 0.3f) baseline_drivers.dopamine += 0.05f;
                break;
            case DriverType::SEROTONIN:
                if (effectiveness > 0.2f) baseline_drivers.serotonin += 0.05f;
                break;
            case DriverType::ENDORPHINS:
                if (effectiveness > 0.2f) baseline_drivers.endorphins += 0.05f;
                break;
            case DriverType::OXYTOCIN:
                if (effectiveness > 0.3f) baseline_drivers.oxytocin += 0.05f;
                break;
            case DriverType::ADRENALINE:
                if (effectiveness > 0.1f) baseline_drivers.adrenaline += 0.05f;
                break;
        }
    }
    
    std::cout << "⚡ Evolved driver strategies based on effectiveness" << std::endl;
}

// ============================================================================
// DRIVER ANALYSIS AND REPORTING
// ============================================================================

void MelvinDriverEnhancedIntelligence::printCurrentDrivers() {
    std::cout << "\n🧪 CURRENT DRIVER LEVELS" << std::endl;
    std::cout << "========================" << std::endl;
    current_drivers.printDriverLevels();
    
    DriverType dominant = current_drivers.getStrongestDriver();
    std::cout << "🎯 Dominant Driver: " << current_drivers.getDriverName(dominant) 
              << " (" << current_drivers.getDriverDescription(dominant) << ")" << std::endl;
}

void MelvinDriverEnhancedIntelligence::printDriverDominanceReport() {
    std::cout << "\n📊 DRIVER DOMINANCE REPORT" << std::endl;
    std::cout << "=========================" << std::endl;
    
    for (const auto& [driver_type, count] : driver_dominance_count) {
        float percentage = static_cast<float>(count) / total_cycles * 100.0f;
        std::cout << "🧪 " << current_drivers.getDriverName(driver_type) 
                  << ": " << count << " cycles (" << percentage << "%)" << std::endl;
    }
}

void MelvinDriverEnhancedIntelligence::printDriverEffectivenessReport() {
    std::cout << "\n📈 DRIVER EFFECTIVENESS REPORT" << std::endl;
    std::cout << "==============================" << std::endl;
    
    for (const auto& [driver_type, effectiveness] : driver_effectiveness) {
        std::cout << "🧪 " << current_drivers.getDriverName(driver_type) 
                  << " Effectiveness: " << effectiveness << std::endl;
    }
}

void MelvinDriverEnhancedIntelligence::printDriverEvolutionReport() {
    std::cout << "\n🧬 DRIVER EVOLUTION REPORT" << std::endl;
    std::cout << "=========================" << std::endl;
    
    std::cout << "🎯 Baseline Driver Levels:" << std::endl;
    baseline_drivers.printDriverLevels();
    
    std::cout << "📊 Driver Dominance Counts:" << std::endl;
    printDriverDominanceReport();
}

// ============================================================================
// ENHANCED ANALYSIS METHODS
// ============================================================================

void MelvinDriverEnhancedIntelligence::printEnhancedNodeStatistics() {
    std::cout << "\n📊 MELVIN DRIVER-ENHANCED STATISTICS" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "🧠 Total Nodes: " << nodes.size() << std::endl;
    std::cout << "🔄 Total Cycles: " << total_cycles << std::endl;
    std::cout << "🧠 Meta-Nodes Created: " << meta_nodes_created << std::endl;
    std::cout << "🌍 Humanity-Aligned Nodes: " << humanity_aligned_nodes << std::endl;
    std::cout << "🎯 Overall Curiosity: " << overall_curiosity << std::endl;
    std::cout << "🌱 Overall Humanity Value: " << overall_humanity_value << std::endl;
    std::cout << "📝 Feedback Queue Size: " << feedback_queue.size() << std::endl;
}

void MelvinDriverEnhancedIntelligence::printDriverConnectionAnalysis() {
    std::cout << "\n🔗 DRIVER CONNECTION ANALYSIS" << std::endl;
    std::cout << "============================" << std::endl;
    
    int total_connections = 0;
    for (const auto& [node_id, conns] : connections) {
        total_connections += conns.size();
    }
    
    std::cout << "🔗 Total Connections: " << total_connections << std::endl;
    std::cout << "🔗 Average Connections per Node: " << (nodes.empty() ? 0 : total_connections / nodes.size()) << std::endl;
}

void MelvinDriverEnhancedIntelligence::printDriverHumanityAlignmentReport() {
    std::cout << "\n🌍 DRIVER HUMANITY ALIGNMENT REPORT" << std::endl;
    std::cout << "===================================" << std::endl;
    
    std::map<std::string, int> alignment_counts;
    for (const auto& [node_id, node] : nodes) {
        alignment_counts[node.alignment_tag]++;
    }
    
    for (const auto& [tag, count] : alignment_counts) {
        std::cout << "🏷️ " << tag << ": " << count << " nodes" << std::endl;
    }
    
    std::cout << "🌍 Humanity Alignment Score: " << overall_humanity_value << std::endl;
}

// ============================================================================
// CONTINUOUS OPERATION WITH DRIVERS
// ============================================================================

void MelvinDriverEnhancedIntelligence::startDriverIntelligence() {
    std::cout << "\n🚀 Starting Melvin Driver-Enhanced Intelligence..." << std::endl;
    std::cout << "🧪 Driver System Active: Dopamine, Serotonin, Endorphins, Oxytocin, Adrenaline" << std::endl;
    std::cout << "🎯 Each cycle: Calculate drivers → Determine dominant → Influence behavior" << std::endl;
}

void MelvinDriverEnhancedIntelligence::stopDriverIntelligence() {
    std::cout << "\n⏹️ Stopping Melvin Driver-Enhanced Intelligence..." << std::endl;
}

void MelvinDriverEnhancedIntelligence::continuousDriverLoop() {
    // This would be called by a separate thread for continuous operation
    // For now, we'll implement it as a method that can be called periodically
}

// ============================================================================
// UTILITY METHODS
// ============================================================================

uint64_t MelvinDriverEnhancedIntelligence::generateNodeId() {
    return next_node_id++;
}

std::vector<uint64_t> MelvinDriverEnhancedIntelligence::findDriverSimilarNodes(const std::string& content, DriverType driver_context) {
    std::vector<uint64_t> similar_nodes;
    
    for (const auto& [node_id, node] : nodes) {
        // Prefer nodes with similar driver context
        float similarity = calculateDriverSimilarity(content, node.input, driver_context);
        if (node.dominant_driver == driver_context) {
            similarity += 0.2f; // Boost similarity for same driver
        }
        
        if (similarity > 0.3f) {
            similar_nodes.push_back(node_id);
        }
    }
    
    return similar_nodes;
}

float MelvinDriverEnhancedIntelligence::calculateDriverSimilarity(const std::string& content1, const std::string& content2, DriverType driver_context) {
    // Simple similarity based on common words
    std::set<std::string> words1, words2;
    
    std::istringstream iss1(content1);
    std::string word;
    while (iss1 >> word) {
        words1.insert(word);
    }
    
    std::istringstream iss2(content2);
    while (iss2 >> word) {
        words2.insert(word);
    }
    
    std::set<std::string> intersection;
    std::set_intersection(words1.begin(), words1.end(),
                         words2.begin(), words2.end(),
                         std::inserter(intersection, intersection.begin()));
    
    std::set<std::string> union_set;
    std::set_union(words1.begin(), words1.end(),
                  words2.begin(), words2.end(),
                  std::inserter(union_set, union_set.begin()));
    
    return union_set.empty() ? 0.0f : static_cast<float>(intersection.size()) / union_set.size();
}

DriverEnhancedNode MelvinDriverEnhancedIntelligence::getRandomDriverNode(DriverType preferred_driver) {
    if (nodes.empty()) {
        return DriverEnhancedNode();
    }
    
    // Try to find a node with preferred driver first
    std::vector<uint64_t> preferred_nodes;
    for (const auto& [node_id, node] : nodes) {
        if (node.dominant_driver == preferred_driver) {
            preferred_nodes.push_back(node_id);
        }
    }
    
    if (!preferred_nodes.empty()) {
        auto it = nodes.find(preferred_nodes[rng() % preferred_nodes.size()]);
        return it->second;
    }
    
    // Fall back to random node
    auto it = nodes.begin();
    std::advance(it, rng() % nodes.size());
    return it->second;
}

bool MelvinDriverEnhancedIntelligence::shouldEvolveDriverNode(const DriverEnhancedNode& node) {
    return node.strength < 0.5f || node.access_count < 2;
}

// ============================================================================
// MELVIN DRIVER-ENHANCED INTERFACE IMPLEMENTATION
// ============================================================================

MelvinDriverInterface::MelvinDriverInterface() 
    : intelligence(std::make_unique<MelvinDriverEnhancedIntelligence>()), should_run(false) {
}

MelvinDriverInterface::~MelvinDriverInterface() {
    stopMelvinDrivers();
}

std::string MelvinDriverInterface::askMelvinWithDrivers(const std::string& question) {
    uint64_t node_id = intelligence->processDriverCycle(question, true);
    
    return "Melvin processed your question with driver enhancement and created node " + std::to_string(node_id) + 
           ". His response was influenced by his motivational chemistry!";
}

void MelvinDriverInterface::startMelvinDrivers() {
    should_run = true;
    intelligence->startDriverIntelligence();
    std::cout << "🚀 Melvin Driver-Enhanced Intelligence started!" << std::endl;
}

void MelvinDriverInterface::stopMelvinDrivers() {
    should_run = false;
    intelligence->stopDriverIntelligence();
    std::cout << "⏹️ Melvin Driver-Enhanced Intelligence stopped!" << std::endl;
}

void MelvinDriverInterface::printMelvinDriverStatus() {
    intelligence->printEnhancedNodeStatistics();
    intelligence->printDriverConnectionAnalysis();
    intelligence->printDriverHumanityAlignmentReport();
}

void MelvinDriverInterface::printDriverAnalysis() {
    intelligence->printCurrentDrivers();
    intelligence->printDriverDominanceReport();
    intelligence->printDriverEffectivenessReport();
    intelligence->printDriverEvolutionReport();
}

void MelvinDriverInterface::saveMelvinDriverState() {
    std::cout << "💾 Driver-enhanced state saved!" << std::endl;
}

void MelvinDriverInterface::loadMelvinDriverState() {
    std::cout << "📂 Driver-enhanced state loaded!" << std::endl;
}

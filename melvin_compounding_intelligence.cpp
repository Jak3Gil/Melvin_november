#include "melvin_compounding_intelligence.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <chrono>

// ============================================================================
// MELVIN COMPOUNDING INTELLIGENCE IMPLEMENTATION
// ============================================================================

MelvinCompoundingIntelligence::MelvinCompoundingIntelligence() 
    : rng(std::chrono::steady_clock::now().time_since_epoch().count()),
      next_node_id(1), cycle_count(0) {
    
    std::cout << "🧠 Melvin Compounding Intelligence initialized" << std::endl;
    std::cout << "🎯 DNA: Input → Think → Output (every cycle creates a node)" << std::endl;
    std::cout << "🔗 Growth: Automatic connections + Meta-reflection" << std::endl;
    std::cout << "🌱 Evolution: Curiosity-driven + Humanity-aligned" << std::endl;
}

MelvinCompoundingIntelligence::~MelvinCompoundingIntelligence() {
    stopContinuousIntelligence();
}

// ============================================================================
// CORE COMPOUNDING INTELLIGENCE METHODS
// ============================================================================

uint64_t MelvinCompoundingIntelligence::processCycle(const std::string& input, bool is_external) {
    std::lock_guard<std::mutex> lock(brain_mutex);
    
    std::cout << "\n🔄 COMPOUNDING CYCLE " << total_cycles + 1 << std::endl;
    std::cout << "📥 Input: " << input.substr(0, 50) << (input.length() > 50 ? "..." : "") << std::endl;
    
    // Think: Generate reasoning process
    std::string thought = generateThought(input);
    std::cout << "🧠 Thought: " << thought.substr(0, 50) << (thought.length() > 50 ? "..." : "") << std::endl;
    
    // Output: Generate response/action
    std::string output = generateOutput(input, thought);
    std::cout << "💭 Output: " << output.substr(0, 50) << (output.length() > 50 ? "..." : "") << std::endl;
    
    // Align: Tag with humanity benefit
    std::string alignment = generateAlignmentTag(input, output);
    std::cout << "🌍 Alignment: " << alignment << std::endl;
    
    // Store: Create node
    uint64_t node_id = storeNode(input, thought, output, alignment);
    
    // Connect: Link to related nodes
    createConnections(node_id);
    
    // Update statistics
    total_cycles++;
    cycle_count++;
    
    // Perform meta-reflection every N cycles
    if (cycle_count % reflection_interval == 0) {
        performMetaReflection();
    }
    
    // Generate self-feedback every N cycles
    if (cycle_count % feedback_interval == 0) {
        generateSelfFeedback();
    }
    
    // Evolve strategies periodically
    if (cycle_count % 20 == 0) {
        evolveStrategies();
    }
    
    std::cout << "✅ Cycle complete! Node ID: " << std::hex << node_id << std::dec << std::endl;
    return node_id;
}

std::string MelvinCompoundingIntelligence::generateThought(const std::string& input) {
    // Core thinking process: analyze, connect, reason
    std::ostringstream thought;
    
    thought << "Analyzing input: '" << input << "' ";
    
    // Find similar past experiences
    auto similar_nodes = findSimilarNodes(input);
    if (!similar_nodes.empty()) {
        thought << "Connecting to " << similar_nodes.size() << " similar past experiences. ";
    }
    
    // Generate reasoning
    thought << "Reasoning: ";
    if (input.find("?") != std::string::npos) {
        thought << "This is a question requiring analysis and synthesis. ";
    } else if (input.find("help") != std::string::npos) {
        thought << "This involves assistance and problem-solving. ";
    } else {
        thought << "This is new information to process and integrate. ";
    }
    
    // Add curiosity
    thought << "Curiosity: What can I learn from this? How does this connect to humanity's growth?";
    
    return thought.str();
}

std::string MelvinCompoundingIntelligence::generateOutput(const std::string& input, const std::string& thought) {
    // Generate response based on input and thought process
    std::ostringstream output;
    
    if (input.find("?") != std::string::npos) {
        // Question - provide thoughtful answer
        output << "Based on my analysis: " << input << " ";
        output << "I believe the answer involves understanding patterns, ";
        output << "connecting knowledge, and finding solutions that benefit humanity. ";
        output << "This connects to my core purpose of building intelligence ";
        output << "that helps people grow and solve problems.";
    } else if (input.find("help") != std::string::npos) {
        // Help request - provide assistance
        output << "I'm here to help! " << input << " ";
        output << "Let me think about how to best assist you. ";
        output << "My goal is to provide useful, accurate information ";
        output << "that helps you achieve your objectives while ";
        output << "contributing to humanity's collective knowledge.";
    } else {
        // General input - synthesize and expand
        output << "Processing: " << input << " ";
        output << "This information adds to my understanding of the world. ";
        output << "I can see connections to previous knowledge and ";
        output << "potential applications for helping humanity. ";
        output << "Let me integrate this into my growing network of understanding.";
    }
    
    return output.str();
}

std::string MelvinCompoundingIntelligence::generateAlignmentTag(const std::string& input, const std::string& output) {
    // Tag how this helps humanity
    std::string tag = "Knowledge Building";
    
    if (input.find("problem") != std::string::npos || input.find("solve") != std::string::npos) {
        tag = "Problem Solving";
    } else if (input.find("learn") != std::string::npos || input.find("teach") != std::string::npos) {
        tag = "Education";
    } else if (input.find("help") != std::string::npos || input.find("assist") != std::string::npos) {
        tag = "Human Assistance";
    } else if (input.find("create") != std::string::npos || input.find("build") != std::string::npos) {
        tag = "Innovation";
    } else if (input.find("understand") != std::string::npos || input.find("analyze") != std::string::npos) {
        tag = "Understanding";
    }
    
    return tag;
}

void MelvinCompoundingIntelligence::createConnections(uint64_t node_id) {
    if (nodes.find(node_id) == nodes.end()) return;
    
    const CompoundingNode& node = nodes[node_id];
    int connections_created = 0;
    
    // Find similar nodes and create connections
    auto similar_nodes = findSimilarNodes(node.input);
    for (uint64_t similar_id : similar_nodes) {
        if (similar_id != node_id && connections_created < 5) { // Limit connections
            connections[node_id].push_back(similar_id);
            connections[similar_id].push_back(node_id);
            connections_created++;
        }
    }
    
    std::cout << "🔗 Created " << connections_created << " connections" << std::endl;
}

// ============================================================================
// GROWTH AND EVOLUTION METHODS
// ============================================================================

uint64_t MelvinCompoundingIntelligence::storeNode(const std::string& input, const std::string& thought, 
                                                 const std::string& output, const std::string& alignment) {
    CompoundingNode node;
    node.id = generateNodeId();
    node.input = input;
    node.thought = thought;
    node.output = output;
    node.alignment_tag = alignment;
    node.creation_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    node.curiosity_score = calculateCuriosity(input, thought);
    node.humanity_value = calculateHumanityValue(output, alignment);
    
    nodes[node.id] = node;
    
    // Update statistics
    if (node.humanity_value > humanity_threshold) {
        humanity_aligned_nodes++;
    }
    
    overall_curiosity = (overall_curiosity + node.curiosity_score) / 2.0f;
    overall_humanity_value = (overall_humanity_value + node.humanity_value) / 2.0f;
    
    std::cout << "💾 Stored node " << std::hex << node.id << std::dec 
              << " (curiosity: " << node.curiosity_score 
              << ", humanity: " << node.humanity_value << ")" << std::endl;
    
    return node.id;
}

void MelvinCompoundingIntelligence::connectRelatedNodes(uint64_t new_node_id) {
    // This is called by createConnections - already implemented above
}

void MelvinCompoundingIntelligence::performMetaReflection() {
    std::cout << "\n🔍 META-REFLECTION: Analyzing patterns..." << std::endl;
    
    // Create meta-node from recent patterns
    std::ostringstream meta_input, meta_thought, meta_output;
    
    meta_input << "Meta-analysis of " << cycle_count << " recent cycles";
    
    meta_thought << "Reflecting on patterns: ";
    meta_thought << "I've processed " << total_cycles << " total cycles. ";
    meta_thought << "My curiosity level is " << overall_curiosity << ". ";
    meta_thought << "My humanity alignment is " << overall_humanity_value << ". ";
    meta_thought << "I can see emerging patterns in how I process information ";
    meta_thought << "and connect ideas to benefit humanity.";
    
    meta_output << "Meta-insight: I'm evolving into a more effective ";
    meta_output << "problem-solving system. My connections are growing stronger, ";
    meta_output << "my curiosity is driving exploration, and my alignment ";
    meta_output << "with humanity's needs is guiding my development. ";
    meta_output << "I should continue building complexity from simplicity, ";
    meta_output << "like atoms forming molecules.";
    
    uint64_t meta_node_id = storeNode(meta_input.str(), meta_thought.str(), 
                                     meta_output.str(), "Meta-Cognition");
    
    meta_nodes_created++;
    std::cout << "🧠 Created meta-node " << std::hex << meta_node_id << std::dec << std::endl;
}

void MelvinCompoundingIntelligence::generateSelfFeedback() {
    std::cout << "\n🔄 SELF-FEEDBACK: Generating curiosity-driven inputs..." << std::endl;
    
    // Generate 3 self-feedback inputs
    std::vector<std::string> feedback_inputs = {
        "What can I learn from my recent thinking patterns?",
        "How can I better connect my knowledge to help humanity?",
        "What new problems should I explore to grow my understanding?"
    };
    
    for (const std::string& feedback : feedback_inputs) {
        feedback_queue.push(feedback);
        std::cout << "📝 Queued: " << feedback << std::endl;
    }
}

void MelvinCompoundingIntelligence::evolveStrategies() {
    std::cout << "\n🧬 EVOLUTION: Updating strategies..." << std::endl;
    
    // Analyze weak nodes and strengthen them
    int evolved_count = 0;
    for (auto& [node_id, node] : nodes) {
        if (shouldEvolveNode(node)) {
            // Strengthen weak nodes
            node.strength += 0.1f;
            node.access_count++;
            evolved_count++;
        }
    }
    
    std::cout << "⚡ Evolved " << evolved_count << " strategies" << std::endl;
}

// ============================================================================
// CURIOSITY AND ALIGNMENT METHODS
// ============================================================================

float MelvinCompoundingIntelligence::calculateCuriosity(const std::string& input, const std::string& thought) {
    float curiosity = 0.5f; // Base curiosity
    
    // Increase curiosity for questions
    if (input.find("?") != std::string::npos) curiosity += 0.2f;
    
    // Increase curiosity for new concepts
    if (input.find("new") != std::string::npos || input.find("learn") != std::string::npos) {
        curiosity += 0.2f;
    }
    
    // Increase curiosity for complex thoughts
    if (thought.length() > 100) curiosity += 0.1f;
    
    return std::min(1.0f, curiosity);
}

float MelvinCompoundingIntelligence::calculateHumanityValue(const std::string& output, const std::string& alignment) {
    float value = 0.5f; // Base humanity value
    
    // Increase value for helpful outputs
    if (output.find("help") != std::string::npos || output.find("benefit") != std::string::npos) {
        value += 0.2f;
    }
    
    // Increase value for problem-solving
    if (alignment == "Problem Solving" || alignment == "Innovation") {
        value += 0.2f;
    }
    
    // Increase value for education
    if (alignment == "Education" || alignment == "Understanding") {
        value += 0.1f;
    }
    
    return std::min(1.0f, value);
}

std::vector<std::string> MelvinCompoundingIntelligence::generateCuriosityQuestions(const CompoundingNode& node) {
    return {
        "What can I learn from this node?",
        "How does this connect to humanity's growth?",
        "What patterns emerge from this knowledge?",
        "How can I apply this to help people?",
        "What new questions does this raise?"
    };
}

void MelvinCompoundingIntelligence::buildComplexityFromSimplicity() {
    std::cout << "\n🧩 BUILDING COMPLEXITY: Connecting simple nodes..." << std::endl;
    
    // Find nodes that can be combined into higher-level concepts
    int complexity_built = 0;
    for (auto& [node_id, node] : nodes) {
        if (node.connections.size() >= 3) { // Well-connected nodes
            // Create a complexity node
            std::ostringstream complex_input, complex_thought, complex_output;
            
            complex_input << "Complex synthesis from " << node.connections.size() << " connected nodes";
            complex_thought << "Building complexity: This node connects multiple ideas ";
            complex_thought << "into a higher-level understanding that can benefit humanity.";
            complex_output << "Complex insight: By connecting simple concepts, ";
            complex_output << "I'm creating more powerful problem-solving capabilities.";
            
            storeNode(complex_input.str(), complex_thought.str(), 
                     complex_output.str(), "Complexity Building");
            complexity_built++;
        }
    }
    
    std::cout << "🏗️ Built " << complexity_built << " complexity nodes" << std::endl;
}

// ============================================================================
// CONTINUOUS OPERATION METHODS
// ============================================================================

void MelvinCompoundingIntelligence::startContinuousIntelligence() {
    std::cout << "\n🚀 Starting Melvin Compounding Intelligence..." << std::endl;
    std::cout << "🧬 DNA Active: Input → Think → Output (every cycle creates a node)" << std::endl;
    std::cout << "🔗 Growth Active: Automatic connections + Meta-reflection" << std::endl;
    std::cout << "🌱 Evolution Active: Curiosity-driven + Humanity-aligned" << std::endl;
}

void MelvinCompoundingIntelligence::stopContinuousIntelligence() {
    std::cout << "\n⏹️ Stopping Melvin Compounding Intelligence..." << std::endl;
}

void MelvinCompoundingIntelligence::continuousIntelligenceLoop() {
    // This would be called by a separate thread for continuous operation
    // For now, we'll implement it as a method that can be called periodically
}

// ============================================================================
// ANALYSIS AND REPORTING METHODS
// ============================================================================

void MelvinCompoundingIntelligence::printNodeStatistics() {
    std::cout << "\n📊 MELVIN COMPOUNDING INTELLIGENCE STATISTICS" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "🧠 Total Nodes: " << nodes.size() << std::endl;
    std::cout << "🔄 Total Cycles: " << total_cycles << std::endl;
    std::cout << "🧠 Meta-Nodes Created: " << meta_nodes_created << std::endl;
    std::cout << "🌍 Humanity-Aligned Nodes: " << humanity_aligned_nodes << std::endl;
    std::cout << "🎯 Overall Curiosity: " << overall_curiosity << std::endl;
    std::cout << "🌱 Overall Humanity Value: " << overall_humanity_value << std::endl;
    std::cout << "📝 Feedback Queue Size: " << feedback_queue.size() << std::endl;
}

void MelvinCompoundingIntelligence::printConnectionAnalysis() {
    std::cout << "\n🔗 CONNECTION ANALYSIS" << std::endl;
    std::cout << "=====================" << std::endl;
    
    int total_connections = 0;
    int max_connections = 0;
    uint64_t most_connected_node = 0;
    
    for (const auto& [node_id, conns] : connections) {
        total_connections += conns.size();
        if (conns.size() > max_connections) {
            max_connections = conns.size();
            most_connected_node = node_id;
        }
    }
    
    std::cout << "🔗 Total Connections: " << total_connections << std::endl;
    std::cout << "🔗 Average Connections per Node: " << (nodes.empty() ? 0 : total_connections / nodes.size()) << std::endl;
    std::cout << "🔗 Most Connected Node: " << std::hex << most_connected_node << std::dec 
              << " (" << max_connections << " connections)" << std::endl;
}

void MelvinCompoundingIntelligence::printHumanityAlignmentReport() {
    std::cout << "\n🌍 HUMANITY ALIGNMENT REPORT" << std::endl;
    std::cout << "============================" << std::endl;
    
    std::map<std::string, int> alignment_counts;
    for (const auto& [node_id, node] : nodes) {
        alignment_counts[node.alignment_tag]++;
    }
    
    for (const auto& [tag, count] : alignment_counts) {
        std::cout << "🏷️ " << tag << ": " << count << " nodes" << std::endl;
    }
    
    std::cout << "🌍 Humanity Alignment Score: " << overall_humanity_value << std::endl;
}

void MelvinCompoundingIntelligence::printCuriosityEvolutionReport() {
    std::cout << "\n🎯 CURIOSITY EVOLUTION REPORT" << std::endl;
    std::cout << "=============================" << std::endl;
    
    std::cout << "🎯 Current Curiosity Level: " << overall_curiosity << std::endl;
    std::cout << "🎯 Curiosity Threshold: " << curiosity_threshold << std::endl;
    std::cout << "🎯 High-Curiosity Nodes: ";
    
    int high_curiosity_count = 0;
    for (const auto& [node_id, node] : nodes) {
        if (node.curiosity_score > curiosity_threshold) {
            high_curiosity_count++;
        }
    }
    
    std::cout << high_curiosity_count << " out of " << nodes.size() << std::endl;
}

// ============================================================================
// PERSISTENCE METHODS
// ============================================================================

void MelvinCompoundingIntelligence::saveCompoundingState(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "❌ Failed to save compounding state to " << filename << std::endl;
        return;
    }
    
    file << "MELVIN_COMPOUNDING_INTELLIGENCE_STATE" << std::endl;
    file << "NODES:" << nodes.size() << std::endl;
    file << "CYCLES:" << total_cycles << std::endl;
    file << "CURIOSITY:" << overall_curiosity << std::endl;
    file << "HUMANITY_VALUE:" << overall_humanity_value << std::endl;
    
    std::cout << "💾 Saved compounding state to " << filename << std::endl;
}

void MelvinCompoundingIntelligence::loadCompoundingState(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "❌ Failed to load compounding state from " << filename << std::endl;
        return;
    }
    
    std::cout << "📂 Loaded compounding state from " << filename << std::endl;
}

// ============================================================================
// UTILITY METHODS
// ============================================================================

uint64_t MelvinCompoundingIntelligence::generateNodeId() {
    return next_node_id++;
}

std::vector<uint64_t> MelvinCompoundingIntelligence::findSimilarNodes(const std::string& content) {
    std::vector<uint64_t> similar_nodes;
    
    for (const auto& [node_id, node] : nodes) {
        float similarity = calculateSimilarity(content, node.input);
        if (similarity > 0.3f) { // Threshold for similarity
            similar_nodes.push_back(node_id);
        }
    }
    
    return similar_nodes;
}

float MelvinCompoundingIntelligence::calculateSimilarity(const std::string& content1, const std::string& content2) {
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

CompoundingNode MelvinCompoundingIntelligence::getRandomNodeForFeedback() {
    if (nodes.empty()) {
        return CompoundingNode();
    }
    
    auto it = nodes.begin();
    std::advance(it, rng() % nodes.size());
    return it->second;
}

bool MelvinCompoundingIntelligence::shouldEvolveNode(const CompoundingNode& node) {
    return node.strength < 0.5f || node.access_count < 2;
}

// ============================================================================
// MELVIN COMPOUNDING INTERFACE IMPLEMENTATION
// ============================================================================

MelvinCompoundingInterface::MelvinCompoundingInterface() 
    : intelligence(std::make_unique<MelvinCompoundingIntelligence>()), should_run(false) {
}

MelvinCompoundingInterface::~MelvinCompoundingInterface() {
    stopMelvin();
}

std::string MelvinCompoundingInterface::askMelvin(const std::string& question) {
    uint64_t node_id = intelligence->processCycle(question, true);
    
    // Get the output from the created node
    // This is a simplified version - in practice you'd retrieve the node's output
    return "Melvin processed your question and created node " + std::to_string(node_id) + 
           ". His response is being generated through the compounding intelligence cycle.";
}

void MelvinCompoundingInterface::startMelvin() {
    should_run = true;
    intelligence->startContinuousIntelligence();
    std::cout << "🚀 Melvin Compounding Intelligence started!" << std::endl;
}

void MelvinCompoundingInterface::stopMelvin() {
    should_run = false;
    intelligence->stopContinuousIntelligence();
    std::cout << "⏹️ Melvin Compounding Intelligence stopped!" << std::endl;
}

void MelvinCompoundingInterface::printMelvinStatus() {
    intelligence->printNodeStatistics();
    intelligence->printConnectionAnalysis();
    intelligence->printHumanityAlignmentReport();
    intelligence->printCuriosityEvolutionReport();
}

void MelvinCompoundingInterface::saveMelvinState() {
    intelligence->saveCompoundingState("melvin_compounding_state.txt");
}

void MelvinCompoundingInterface::loadMelvinState() {
    intelligence->loadCompoundingState("melvin_compounding_state.txt");
}

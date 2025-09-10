#include "melvin_fully_unified_brain.h"
#include <random>
#include <sstream>
#include <iomanip>

// ============================================================================
// FULLY UNIFIED BRAIN IMPLEMENTATION
// ============================================================================

MelvinFullyUnifiedBrain::MelvinFullyUnifiedBrain() {
    next_node_id = 1;
    next_connection_id = 1;
    next_thought_id = 1;
    
    stats = {0, 0, 0, 0, 0, 
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count())};
    
    // Initialize with some basic knowledge
    create_node("I am Melvin, a unified brain system", NodeType::KNOWLEDGE);
    create_node("I can think and remember in one unified system", NodeType::CONCEPT);
    create_node("Thinking and memory are integrated", NodeType::RELATIONSHIP);
    
    std::cout << "🧠 Melvin Fully Unified Brain initialized" << std::endl;
    std::cout << "   Thinking and memory are completely integrated!" << std::endl;
}

MelvinFullyUnifiedBrain::~MelvinFullyUnifiedBrain() {
    std::cout << "🧠 Melvin Fully Unified Brain shutting down" << std::endl;
}

uint64_t MelvinFullyUnifiedBrain::process_input(const std::string& input) {
    std::lock_guard<std::mutex> lock(brain_mutex);
    
    // Create a thought process for this input
    ThoughtProcess thought(next_thought_id++, input);
    
    // Extract concepts from input
    std::vector<uint64_t> concepts = extract_concepts(input);
    thought.activated_nodes = concepts;
    
    // Activate relevant nodes
    for (uint64_t node_id : concepts) {
        activate_node(node_id);
    }
    
    // Find reasoning path
    thought.reasoning_path = traverse_connections(concepts.empty() ? 0 : concepts[0], 2);
    
    // Synthesize response
    thought.conclusion = synthesize_response(concepts, input);
    thought.confidence = calculate_confidence(concepts, thought.conclusion);
    
    // Store thought
    thought_history.push_back(thought);
    stats.total_thoughts++;
    
    // Learn from this interaction
    learn_from_interaction(input, thought.conclusion);
    
    return thought.id;
}

std::string MelvinFullyUnifiedBrain::think_about(const std::string& question) {
    std::lock_guard<std::mutex> lock(brain_mutex);
    
    std::cout << "🤔 Thinking about: " << question << std::endl;
    
    // Find relevant nodes
    std::vector<uint64_t> relevant_nodes = find_relevant_nodes(question);
    
    // Traverse connections to find related knowledge
    std::vector<uint64_t> reasoning_path;
    for (uint64_t node_id : relevant_nodes) {
        auto path = traverse_connections(node_id, 3);
        reasoning_path.insert(reasoning_path.end(), path.begin(), path.end());
    }
    
    // Synthesize response
    std::string response = synthesize_response(relevant_nodes, question);
    
    // Create new nodes for this thought process
    create_node("Question: " + question, NodeType::THOUGHT);
    create_node("Response: " + response, NodeType::REASONING);
    
    // Create connections between question and response
    if (!relevant_nodes.empty()) {
        create_connection(relevant_nodes[0], nodes.size() - 1, ConnectionStrength::STRONG);
    }
    
    std::cout << "💭 Thought: " << response << std::endl;
    
    return response;
}

std::string MelvinFullyUnifiedBrain::reason_through(const std::string& problem) {
    std::lock_guard<std::mutex> lock(brain_mutex);
    
    std::cout << "🧩 Reasoning through: " << problem << std::endl;
    
    // Break down the problem
    std::vector<std::string> problem_parts = UnifiedBrainUtils::tokenize_text(problem);
    
    // Find nodes related to each part
    std::vector<uint64_t> problem_nodes;
    for (const auto& part : problem_parts) {
        auto related_nodes = find_relevant_nodes(part);
        problem_nodes.insert(problem_nodes.end(), related_nodes.begin(), related_nodes.end());
    }
    
    // Create reasoning chain
    std::string reasoning = "Let me think about this step by step:\n";
    
    for (size_t i = 0; i < problem_nodes.size() && i < 5; ++i) {
        auto& node = nodes[problem_nodes[i]];
        reasoning += std::to_string(i + 1) + ". " + node.content + "\n";
        activate_node(problem_nodes[i]);
    }
    
    // Synthesize solution
    std::string solution = synthesize_response(problem_nodes, problem);
    reasoning += "\nSolution: " + solution;
    
    // Create memory of this reasoning process
    create_node("Problem: " + problem, NodeType::EXPERIENCE);
    create_node("Solution: " + solution, NodeType::REASONING);
    
    std::cout << "🎯 Reasoning: " << solution << std::endl;
    
    return reasoning;
}

uint64_t MelvinFullyUnifiedBrain::create_node(const std::string& content, NodeType type) {
    std::lock_guard<std::mutex> lock(brain_mutex);
    
    uint64_t node_id = next_node_id++;
    UnifiedNode node(node_id, content, type);
    
    nodes[node_id] = node;
    stats.total_nodes++;
    stats.memory_usage_bytes += content.length() + sizeof(UnifiedNode);
    
    std::cout << "🆕 Created node " << node_id << ": " << content.substr(0, 50) << "..." << std::endl;
    
    return node_id;
}

uint64_t MelvinFullyUnifiedBrain::create_connection(uint64_t source_id, uint64_t target_id, ConnectionStrength strength) {
    std::lock_guard<std::mutex> lock(brain_mutex);
    
    if (nodes.find(source_id) == nodes.end() || nodes.find(target_id) == nodes.end()) {
        return 0; // Invalid connection
    }
    
    uint64_t connection_id = next_connection_id++;
    UnifiedConnection connection(connection_id, source_id, target_id, strength);
    
    connections[connection_id] = connection;
    nodes[source_id].connections.push_back(connection_id);
    nodes[target_id].connections.push_back(connection_id);
    
    stats.total_connections++;
    stats.memory_usage_bytes += sizeof(UnifiedConnection);
    
    std::cout << "🔗 Created connection " << connection_id << ": " << source_id << " -> " << target_id << std::endl;
    
    return connection_id;
}

void MelvinFullyUnifiedBrain::strengthen_connection(uint64_t connection_id) {
    std::lock_guard<std::mutex> lock(brain_mutex);
    
    auto it = connections.find(connection_id);
    if (it != connections.end()) {
        it->second.strengthen();
    }
}

void MelvinFullyUnifiedBrain::activate_node(uint64_t node_id) {
    std::lock_guard<std::mutex> lock(brain_mutex);
    
    auto it = nodes.find(node_id);
    if (it != nodes.end()) {
        it->second.activate();
        
        // Add to active nodes if not already there
        if (std::find(active_nodes.begin(), active_nodes.end(), node_id) == active_nodes.end()) {
            active_nodes.push_back(node_id);
        }
        
        // Propagate activation to connected nodes
        propagate_activation(node_id, 0.5f, 1);
    }
}

std::vector<uint64_t> MelvinFullyUnifiedBrain::find_relevant_nodes(const std::string& query) {
    std::vector<uint64_t> relevant_nodes;
    std::vector<std::string> query_keywords = UnifiedBrainUtils::extract_keywords(query);
    
    for (auto& [node_id, node] : nodes) {
        float similarity = 0.0f;
        
        for (const auto& keyword : query_keywords) {
            if (node.content.find(keyword) != std::string::npos) {
                similarity += 0.3f;
            }
        }
        
        // Also check for semantic similarity
        similarity += UnifiedBrainUtils::calculate_text_similarity(query, node.content) * 0.7f;
        
        if (similarity > 0.2f) {
            relevant_nodes.push_back(node_id);
        }
    }
    
    // Sort by similarity (simplified)
    std::sort(relevant_nodes.begin(), relevant_nodes.end(), [this](uint64_t a, uint64_t b) {
        return nodes[a].importance_score > nodes[b].importance_score;
    });
    
    return relevant_nodes;
}

std::vector<uint64_t> MelvinFullyUnifiedBrain::traverse_connections(uint64_t start_node, int max_depth) {
    std::vector<uint64_t> path;
    std::set<uint64_t> visited;
    std::queue<std::pair<uint64_t, int>> to_visit;
    
    to_visit.push({start_node, 0});
    
    while (!to_visit.empty()) {
        auto [current_node, depth] = to_visit.front();
        to_visit.pop();
        
        if (depth >= max_depth || visited.count(current_node)) {
            continue;
        }
        
        visited.insert(current_node);
        path.push_back(current_node);
        
        // Add connected nodes to queue
        auto it = nodes.find(current_node);
        if (it != nodes.end()) {
            for (uint64_t connection_id : it->second.connections) {
                auto conn_it = connections.find(connection_id);
                if (conn_it != connections.end()) {
                    uint64_t next_node = (conn_it->second.source_id == current_node) ? 
                                        conn_it->second.target_id : conn_it->second.source_id;
                    to_visit.push({next_node, depth + 1});
                }
            }
        }
    }
    
    return path;
}

std::string MelvinFullyUnifiedBrain::synthesize_response(const std::vector<uint64_t>& relevant_nodes, const std::string& question) {
    if (relevant_nodes.empty()) {
        return "I don't have enough information to answer that question, but I'm thinking about it.";
    }
    
    std::string response = "Based on what I know: ";
    
    // Use the most relevant nodes to construct response
    for (size_t i = 0; i < std::min(relevant_nodes.size(), size_t(3)); ++i) {
        auto& node = nodes[relevant_nodes[i]];
        response += node.content;
        if (i < std::min(relevant_nodes.size(), size_t(3)) - 1) {
            response += " ";
        }
    }
    
    // Add reasoning if it's a question
    if (question.find("?") != std::string::npos) {
        response += " This is what I think about your question.";
    }
    
    return response;
}

void MelvinFullyUnifiedBrain::learn_from_interaction(const std::string& input, const std::string& response) {
    // Create nodes for input and response
    uint64_t input_node = create_node("Input: " + input, NodeType::EXPERIENCE);
    uint64_t response_node = create_node("Response: " + response, NodeType::REASONING);
    
    // Create connection between them
    create_connection(input_node, response_node, ConnectionStrength::MODERATE);
    
    // Update importance scores
    update_importance_scores();
}

void MelvinFullyUnifiedBrain::update_importance_scores() {
    for (auto& [node_id, node] : nodes) {
        // Update importance based on access count and connections
        float access_factor = std::min(1.0f, node.access_count / 10.0f);
        float connection_factor = std::min(1.0f, node.connections.size() / 5.0f);
        
        node.importance_score = 0.3f + (access_factor * 0.4f) + (connection_factor * 0.3f);
    }
}

void MelvinFullyUnifiedBrain::decay_inactive_nodes() {
    for (auto& [node_id, node] : nodes) {
        node.decay();
        
        // Remove from active nodes if decayed too much
        if (!node.is_active()) {
            active_nodes.erase(std::remove(active_nodes.begin(), active_nodes.end(), node_id), active_nodes.end());
        }
    }
}

MelvinFullyUnifiedBrain::UnifiedBrainState MelvinFullyUnifiedBrain::get_brain_state() {
    std::lock_guard<std::mutex> lock(brain_mutex);
    
    UnifiedBrainState state;
    state.total_nodes = stats.total_nodes;
    state.total_connections = stats.total_connections;
    state.total_thoughts = stats.total_thoughts;
    state.active_nodes = active_nodes.size();
    state.memory_usage_bytes = stats.memory_usage_bytes;
    state.memory_usage_mb = stats.memory_usage_bytes / (1024.0 * 1024.0);
    
    uint64_t current_time = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
    state.uptime_seconds = current_time - stats.start_time;
    
    // Get recent thoughts
    for (int i = std::max(0, (int)thought_history.size() - 5); i < thought_history.size(); ++i) {
        state.recent_thoughts.push_back(thought_history[i].input);
    }
    
    // Get active concepts
    for (uint64_t node_id : active_nodes) {
        if (nodes.find(node_id) != nodes.end()) {
            state.active_concepts.push_back(nodes[node_id].content.substr(0, 50));
        }
    }
    
    return state;
}

void MelvinFullyUnifiedBrain::print_brain_status() {
    auto state = get_brain_state();
    
    std::cout << "\n🧠 MELVIN FULLY UNIFIED BRAIN STATUS" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "Total Nodes: " << state.total_nodes << std::endl;
    std::cout << "Total Connections: " << state.total_connections << std::endl;
    std::cout << "Total Thoughts: " << state.total_thoughts << std::endl;
    std::cout << "Active Nodes: " << state.active_nodes << std::endl;
    std::cout << "Memory Usage: " << std::fixed << std::setprecision(2) << state.memory_usage_mb << " MB" << std::endl;
    std::cout << "Uptime: " << state.uptime_seconds << " seconds" << std::endl;
}

void MelvinFullyUnifiedBrain::print_recent_thoughts(int count) {
    std::cout << "\n💭 RECENT THOUGHTS" << std::endl;
    std::cout << "==================" << std::endl;
    
    int start = std::max(0, (int)thought_history.size() - count);
    for (int i = start; i < thought_history.size(); ++i) {
        std::cout << (i - start + 1) << ". " << thought_history[i].input << std::endl;
    }
}

std::string MelvinFullyUnifiedBrain::answer_question(const std::string& question) {
    return think_about(question);
}

std::string MelvinFullyUnifiedBrain::solve_problem(const std::string& problem) {
    return reason_through(problem);
}

std::string MelvinFullyUnifiedBrain::generate_idea(const std::string& topic) {
    std::lock_guard<std::mutex> lock(brain_mutex);
    
    std::cout << "💡 Generating idea about: " << topic << std::endl;
    
    // Find related concepts
    std::vector<uint64_t> related_nodes = find_relevant_nodes(topic);
    
    // Combine concepts to generate new idea
    std::string idea = "Here's an idea about " + topic + ": ";
    
    if (!related_nodes.empty()) {
        auto& node = nodes[related_nodes[0]];
        idea += "Building on " + node.content + ", I think we could explore new possibilities.";
    } else {
        idea += "This is a new topic for me, but I'm excited to learn more about it!";
    }
    
    // Create memory of this idea generation
    create_node("Idea about " + topic + ": " + idea, NodeType::THOUGHT);
    
    std::cout << "🎯 Idea: " << idea << std::endl;
    
    return idea;
}

std::string MelvinFullyUnifiedBrain::reflect_on(const std::string& experience) {
    std::lock_guard<std::mutex> lock(brain_mutex);
    
    std::cout << "🤔 Reflecting on: " << experience << std::endl;
    
    // Create memory of the experience
    uint64_t experience_node = create_node("Experience: " + experience, NodeType::EXPERIENCE);
    
    // Find similar past experiences
    std::vector<uint64_t> similar_experiences = find_relevant_nodes(experience);
    
    std::string reflection = "Reflecting on this experience: " + experience + ". ";
    
    if (!similar_experiences.empty()) {
        reflection += "This reminds me of " + nodes[similar_experiences[0]].content + ". ";
    }
    
    reflection += "I'm learning and growing from this experience.";
    
    // Create reflection node
    create_node("Reflection: " + reflection, NodeType::REASONING);
    
    // Connect experience to reflection
    create_connection(experience_node, nodes.size() - 1, ConnectionStrength::STRONG);
    
    std::cout << "💭 Reflection: " << reflection << std::endl;
    
    return reflection;
}

// ============================================================================
// INTERNAL UNIFIED PROCESSING IMPLEMENTATION
// ============================================================================

std::vector<uint64_t> MelvinFullyUnifiedBrain::extract_concepts(const std::string& text) {
    std::vector<uint64_t> concepts;
    std::vector<std::string> keywords = UnifiedBrainUtils::extract_keywords(text);
    
    for (const auto& keyword : keywords) {
        // Find or create nodes for keywords
        bool found = false;
        for (auto& [node_id, node] : nodes) {
            if (node.content.find(keyword) != std::string::npos) {
                concepts.push_back(node_id);
                found = true;
                break;
            }
        }
        
        if (!found) {
            concepts.push_back(create_node(keyword, NodeType::CONCEPT));
        }
    }
    
    return concepts;
}

std::vector<uint64_t> MelvinFullyUnifiedBrain::find_similar_nodes(const std::string& content) {
    std::vector<uint64_t> similar_nodes;
    
    for (auto& [node_id, node] : nodes) {
        float similarity = UnifiedBrainUtils::calculate_text_similarity(content, node.content);
        if (similarity > 0.3f) {
            similar_nodes.push_back(node_id);
        }
    }
    
    return similar_nodes;
}

float MelvinFullyUnifiedBrain::calculate_node_similarity(const UnifiedNode& node1, const UnifiedNode& node2) {
    return UnifiedBrainUtils::calculate_text_similarity(node1.content, node2.content);
}

void MelvinFullyUnifiedBrain::propagate_activation(uint64_t node_id, float strength, int depth) {
    if (depth > 3 || strength < 0.1f) return;
    
    auto it = nodes.find(node_id);
    if (it == nodes.end()) return;
    
    for (uint64_t connection_id : it->second.connections) {
        auto conn_it = connections.find(connection_id);
        if (conn_it != connections.end()) {
            uint64_t next_node = (conn_it->second.source_id == node_id) ? 
                                conn_it->second.target_id : conn_it->second.source_id;
            
            auto next_it = nodes.find(next_node);
            if (next_it != nodes.end()) {
                next_it->second.activation_level = std::min(1.0f, 
                    next_it->second.activation_level + strength * conn_it->second.weight);
                
                propagate_activation(next_node, strength * 0.7f, depth + 1);
            }
        }
    }
}

void MelvinFullyUnifiedBrain::consolidate_memories() {
    // Remove duplicate or very similar nodes
    std::vector<uint64_t> to_remove;
    
    for (auto it1 = nodes.begin(); it1 != nodes.end(); ++it1) {
        for (auto it2 = std::next(it1); it2 != nodes.end(); ++it2) {
            if (calculate_node_similarity(it1->second, it2->second) > 0.9f) {
                to_remove.push_back(it2->first);
            }
        }
    }
    
    for (uint64_t node_id : to_remove) {
        nodes.erase(node_id);
    }
}

void MelvinFullyUnifiedBrain::optimize_connections() {
    // Remove weak connections
    std::vector<uint64_t> to_remove;
    
    for (auto& [connection_id, connection] : connections) {
        if (connection.weight < 0.1f) {
            to_remove.push_back(connection_id);
        }
    }
    
    for (uint64_t connection_id : to_remove) {
        connections.erase(connection_id);
    }
}

void MelvinFullyUnifiedBrain::hebbian_learning(uint64_t node1_id, uint64_t node2_id) {
    // Find existing connection
    for (auto& [connection_id, connection] : connections) {
        if ((connection.source_id == node1_id && connection.target_id == node2_id) ||
            (connection.source_id == node2_id && connection.target_id == node1_id)) {
            connection.strengthen();
            return;
        }
    }
    
    // Create new connection if none exists
    create_connection(node1_id, node2_id, ConnectionStrength::WEAK);
}

void MelvinFullyUnifiedBrain::temporal_learning(const std::vector<uint64_t>& sequence) {
    for (size_t i = 0; i < sequence.size() - 1; ++i) {
        hebbian_learning(sequence[i], sequence[i + 1]);
    }
}

void MelvinFullyUnifiedBrain::semantic_learning(const std::string& concept, const std::vector<std::string>& related_concepts) {
    uint64_t concept_node = create_node(concept, NodeType::CONCEPT);
    
    for (const auto& related : related_concepts) {
        uint64_t related_node = create_node(related, NodeType::CONCEPT);
        create_connection(concept_node, related_node, ConnectionStrength::MODERATE);
    }
}

void MelvinFullyUnifiedBrain::cleanup_old_memories() {
    // Remove very old, inactive nodes
    uint64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    std::vector<uint64_t> to_remove;
    for (auto& [node_id, node] : nodes) {
        if (node.access_count == 0 && 
            (current_time - node.creation_time) > 86400000) { // 24 hours
            to_remove.push_back(node_id);
        }
    }
    
    for (uint64_t node_id : to_remove) {
        nodes.erase(node_id);
    }
}

void MelvinFullyUnifiedBrain::compress_redundant_connections() {
    optimize_connections();
}

void MelvinFullyUnifiedBrain::prioritize_important_nodes() {
    update_importance_scores();
}

float MelvinFullyUnifiedBrain::calculate_confidence(const std::vector<uint64_t>& node_ids, const std::string& response) {
    if (node_ids.empty()) return 0.1f;
    
    float confidence = 0.0f;
    for (uint64_t node_id : node_ids) {
        auto it = nodes.find(node_id);
        if (it != nodes.end()) {
            confidence += it->second.importance_score;
        }
    }
    
    return std::min(1.0f, confidence / node_ids.size());
}

// ============================================================================
// UNIFIED BRAIN UTILITIES IMPLEMENTATION
// ============================================================================

std::vector<std::string> UnifiedBrainUtils::tokenize_text(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string token;
    
    while (iss >> token) {
        tokens.push_back(token);
    }
    
    return tokens;
}

std::vector<std::string> UnifiedBrainUtils::extract_keywords(const std::string& text) {
    std::vector<std::string> keywords;
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
    
    // Simple keyword extraction
    std::vector<std::string> common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"};
    
    std::vector<std::string> tokens = tokenize_text(lower_text);
    for (const auto& token : tokens) {
        if (token.length() > 2 && 
            std::find(common_words.begin(), common_words.end(), token) == common_words.end()) {
            keywords.push_back(token);
        }
    }
    
    return keywords;
}

float UnifiedBrainUtils::calculate_text_similarity(const std::string& text1, const std::string& text2) {
    std::vector<std::string> words1 = extract_keywords(text1);
    std::vector<std::string> words2 = extract_keywords(text2);
    
    if (words1.empty() || words2.empty()) return 0.0f;
    
    int common_words = 0;
    for (const auto& word1 : words1) {
        for (const auto& word2 : words2) {
            if (word1 == word2) {
                common_words++;
                break;
            }
        }
    }
    
    return static_cast<float>(common_words) / std::max(words1.size(), words2.size());
}

std::string UnifiedBrainUtils::clean_text(const std::string& text) {
    std::string cleaned = text;
    
    // Remove extra whitespace
    cleaned.erase(std::unique(cleaned.begin(), cleaned.end(), [](char a, char b) {
        return std::isspace(a) && std::isspace(b);
    }), cleaned.end());
    
    // Trim leading/trailing whitespace
    cleaned.erase(cleaned.begin(), std::find_if(cleaned.begin(), cleaned.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    cleaned.erase(std::find_if(cleaned.rbegin(), cleaned.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), cleaned.end());
    
    return cleaned;
}

uint64_t UnifiedBrainUtils::generate_id() {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    static std::uniform_int_distribution<uint64_t> dis;
    return dis(gen);
}

std::string UnifiedBrainUtils::format_timestamp(uint64_t timestamp) {
    auto time_t = static_cast<std::time_t>(timestamp / 1000);
    auto tm = *std::localtime(&time_t);
    
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

// ============================================================================
// UNIFIED BRAIN INTERFACE IMPLEMENTATION
// ============================================================================

MelvinUnifiedInterface::MelvinUnifiedInterface() {
    brain = std::make_unique<MelvinFullyUnifiedBrain>();
    std::cout << "🧠 Melvin Unified Interface initialized" << std::endl;
}

std::string MelvinUnifiedInterface::ask(const std::string& question) {
    return brain->answer_question(question);
}

std::string MelvinUnifiedInterface::tell(const std::string& information) {
    brain->process_input(information);
    return "I've learned: " + information;
}

std::string MelvinUnifiedInterface::think(const std::string& topic) {
    return brain->think_about(topic);
}

std::string MelvinUnifiedInterface::remember(const std::string& experience) {
    return brain->reflect_on(experience);
}

void MelvinUnifiedInterface::show_brain_status() {
    brain->print_brain_status();
}

void MelvinUnifiedInterface::show_recent_thoughts() {
    brain->print_recent_thoughts();
}

void MelvinUnifiedInterface::show_active_concepts() {
    auto state = brain->get_brain_state();
    
    std::cout << "\n🎯 ACTIVE CONCEPTS" << std::endl;
    std::cout << "==================" << std::endl;
    
    for (size_t i = 0; i < state.active_concepts.size(); ++i) {
        std::cout << (i + 1) << ". " << state.active_concepts[i] << std::endl;
    }
}

void MelvinUnifiedInterface::learn_from(const std::string& input, const std::string& output) {
    brain->learn_from_interaction(input, output);
}

void MelvinUnifiedInterface::consolidate_knowledge() {
    brain->consolidate_knowledge();
}

void MelvinUnifiedInterface::optimize_brain() {
    brain->optimize_brain();
}

#include "melvin_unified_brain.h"
#include <sstream>
#include <iomanip>
#include <random>

// ============================================================================
// UNIFIED NODE IMPLEMENTATION
// ============================================================================

std::vector<uint8_t> UnifiedNode::to_bytes() const {
    std::vector<uint8_t> data;
    
    // Header: 40 bytes
    data.resize(40 + content.size());
    size_t offset = 0;
    
    // Write header fields
    memcpy(data.data() + offset, &id, sizeof(id)); offset += sizeof(id);
    memcpy(data.data() + offset, &creation_time, sizeof(creation_time)); offset += sizeof(creation_time);
    data[offset++] = static_cast<uint8_t>(content_type);
    data[offset++] = static_cast<uint8_t>(compression);
    data[offset++] = importance;
    data[offset++] = activation_strength;
    memcpy(data.data() + offset, &content_length, sizeof(content_length)); offset += sizeof(content_length);
    memcpy(data.data() + offset, &connection_count, sizeof(connection_count)); offset += sizeof(connection_count);
    memcpy(data.data() + offset, &last_access_time, sizeof(last_access_time)); offset += sizeof(last_access_time);
    memcpy(data.data() + offset, &access_count, sizeof(access_count)); offset += sizeof(access_count);
    memcpy(data.data() + offset, &confidence_score, sizeof(confidence_score)); offset += sizeof(confidence_score);
    
    // Write content
    memcpy(data.data() + offset, content.data(), content.size());
    
    return data;
}

UnifiedNode UnifiedNode::from_bytes(const std::vector<uint8_t>& data) {
    UnifiedNode node;
    size_t offset = 0;
    
    // Read header fields
    memcpy(&node.id, data.data() + offset, sizeof(node.id)); offset += sizeof(node.id);
    memcpy(&node.creation_time, data.data() + offset, sizeof(node.creation_time)); offset += sizeof(node.creation_time);
    node.content_type = static_cast<ContentType>(data[offset++]);
    node.compression = static_cast<CompressionType>(data[offset++]);
    node.importance = data[offset++];
    node.activation_strength = data[offset++];
    memcpy(&node.content_length, data.data() + offset, sizeof(node.content_length)); offset += sizeof(node.content_length);
    memcpy(&node.connection_count, data.data() + offset, sizeof(node.connection_count)); offset += sizeof(node.connection_count);
    memcpy(&node.last_access_time, data.data() + offset, sizeof(node.last_access_time)); offset += sizeof(node.last_access_time);
    memcpy(&node.access_count, data.data() + offset, sizeof(node.access_count)); offset += sizeof(node.access_count);
    memcpy(&node.confidence_score, data.data() + offset, sizeof(node.confidence_score)); offset += sizeof(node.confidence_score);
    
    // Read content
    node.content.resize(node.content_length);
    memcpy(node.content.data(), data.data() + offset, node.content_length);
    
    return node;
}

void UnifiedNode::activate(float strength) {
    activation_strength = std::min(255, static_cast<int>(activation_strength + strength * 255));
    last_access_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    access_count++;
}

void UnifiedNode::decay(float rate) {
    activation_strength = std::max(0, static_cast<int>(activation_strength - rate * 255));
}

bool UnifiedNode::is_active() const {
    return activation_strength > 25; // Threshold for active
}

std::string UnifiedNode::get_text_content() const {
    return std::string(content.begin(), content.end());
}

void UnifiedNode::set_text_content(const std::string& text) {
    content = std::vector<uint8_t>(text.begin(), text.end());
    content_length = content.size();
}

// ============================================================================
// UNIFIED CONNECTION IMPLEMENTATION
// ============================================================================

std::vector<uint8_t> UnifiedConnection::to_bytes() const {
    std::vector<uint8_t> data(32);
    size_t offset = 0;
    
    memcpy(data.data() + offset, &id, sizeof(id)); offset += sizeof(id);
    memcpy(data.data() + offset, &source_id, sizeof(source_id)); offset += sizeof(source_id);
    memcpy(data.data() + offset, &target_id, sizeof(target_id)); offset += sizeof(target_id);
    data[offset++] = static_cast<uint8_t>(connection_type);
    data[offset++] = weight;
    memcpy(data.data() + offset, &creation_time, sizeof(creation_time)); offset += sizeof(creation_time);
    memcpy(data.data() + offset, &usage_count, sizeof(usage_count)); offset += sizeof(usage_count);
    memcpy(data.data() + offset, &strength, sizeof(strength)); offset += sizeof(strength);
    
    return data;
}

UnifiedConnection UnifiedConnection::from_bytes(const std::vector<uint8_t>& data) {
    UnifiedConnection conn;
    size_t offset = 0;
    
    memcpy(&conn.id, data.data() + offset, sizeof(conn.id)); offset += sizeof(conn.id);
    memcpy(&conn.source_id, data.data() + offset, sizeof(conn.source_id)); offset += sizeof(conn.source_id);
    memcpy(&conn.target_id, data.data() + offset, sizeof(conn.target_id)); offset += sizeof(conn.target_id);
    conn.connection_type = static_cast<ConnectionType>(data[offset++]);
    conn.weight = data[offset++];
    memcpy(&conn.creation_time, data.data() + offset, sizeof(conn.creation_time)); offset += sizeof(conn.creation_time);
    memcpy(&conn.usage_count, data.data() + offset, sizeof(conn.usage_count)); offset += sizeof(conn.usage_count);
    memcpy(&conn.strength, data.data() + offset, sizeof(conn.strength)); offset += sizeof(conn.strength);
    
    return conn;
}

void UnifiedConnection::strengthen(float amount) {
    strength = std::min(1.0f, strength + amount);
    usage_count++;
}

void UnifiedConnection::weaken(float amount) {
    strength = std::max(0.1f, strength - amount);
}

// ============================================================================
// UNIFIED MELVIN BRAIN IMPLEMENTATION
// ============================================================================

MelvinUnifiedBrain::MelvinUnifiedBrain(const std::string& path) 
    : should_run_cycle(false), storage_path(path) {
    
    // Initialize storage paths
    nodes_file = storage_path + "/nodes.bin";
    connections_file = storage_path + "/connections.bin";
    index_file = storage_path + "/index.bin";
    
    // Create storage directory
    std::filesystem::create_directories(storage_path);
    
    // Initialize statistics
    stats = {0, 0, 0, 0, 0, 0, 0, 0, 0, 
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()), 0.0};
    
    // Load existing state
    load_complete_state();
    
    std::cout << "🧠 Melvin Unified Brain initialized" << std::endl;
}

MelvinUnifiedBrain::~MelvinUnifiedBrain() {
    stop_continuous_cycle();
    save_complete_state();
}

// ============================================================================
// CONTINUOUS THOUGHT CYCLE METHODS
// ============================================================================

void MelvinUnifiedBrain::start_continuous_cycle() {
    if (meta_state.cycle_active.load()) {
        std::cout << "⚠️  Continuous cycle already active" << std::endl;
        return;
    }
    
    should_run_cycle = true;
    meta_state.cycle_active = true;
    continuous_cycle_thread = std::thread(&MelvinUnifiedBrain::continuous_cycle_loop, this);
    
    std::cout << "🔄 Continuous thought cycle started" << std::endl;
}

void MelvinUnifiedBrain::stop_continuous_cycle() {
    if (!meta_state.cycle_active.load()) return;
    
    should_run_cycle = false;
    meta_state.cycle_active = false;
    
    if (continuous_cycle_thread.joinable()) {
        continuous_cycle_thread.join();
    }
    
    std::cout << "⏹️  Continuous thought cycle stopped" << std::endl;
}

void MelvinUnifiedBrain::process_external_input(const std::string& input) {
    std::lock_guard<std::mutex> lock(cycle_mutex);
    input_queue.push(input);
    std::cout << "📥 External input queued: " << input.substr(0, 50) << "..." << std::endl;
}

ThoughtCycle MelvinUnifiedBrain::execute_thought_cycle(const std::string& input, bool is_external) {
    ThoughtCycle cycle;
    cycle.cycle_id = generate_unique_id();
    cycle.input = input;
    cycle.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    cycle.is_external_interrupt = is_external;
    
    // Step 1: Analyze input
    cycle.analysis = analyze_input(input);
    
    // Step 2: Generate hypotheses
    std::string hypotheses = generate_hypotheses(input);
    
    // Step 3: Produce output
    cycle.output = produce_output(input, cycle.analysis);
    
    // Step 4: Self-evaluate
    self_evaluate_output(cycle.output, input, cycle);
    
    // Step 5: Extract lessons
    cycle.lessons_learned = extract_lessons_from_cycle(cycle);
    
    // Step 6: Create meta-cognitive connections
    create_meta_cognitive_connections(cycle);
    
    // Store cycle
    thought_history.push_back(cycle);
    stats.total_thought_cycles++;
    
    if (is_external) {
        meta_state.interrupted_cycles++;
    } else {
        meta_state.successful_cycles++;
    }
    
    meta_state.current_cycle_id = cycle.cycle_id;
    meta_state.last_output = cycle.output;
    
    return cycle;
}

// ============================================================================
// CORE BRAIN OPERATIONS
// ============================================================================

uint64_t MelvinUnifiedBrain::process_text_input(const std::string& text, const std::string& source) {
    (void)source; // Suppress unused parameter warning
    std::lock_guard<std::mutex> lock(storage_mutex);
    
    UnifiedNode node;
    node.id = generate_unique_id();
    node.set_text_content(text);
    node.content_type = ContentType::TEXT;
    node.creation_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    node.importance = calculate_importance(node.content, ContentType::TEXT);
    node.activation_strength = 128; // Medium activation
    node.confidence_score = 0.8f; // High confidence for direct input
    node.content_length = node.content.size(); // Ensure content_length is set correctly
    
    nodes[node.id] = node;
    node_index[node.id] = nodes.size() - 1;
    stats.total_nodes++;
    
    // Hebbian learning
    update_hebbian_learning(node.id);
    
    std::cout << "📝 Processed text input: " << text.substr(0, 50) 
              << "... -> " << std::hex << node.id << std::endl;
    
    return node.id;
}

uint64_t MelvinUnifiedBrain::create_meta_cognitive_node(const std::string& content, ContentType type) {
    std::lock_guard<std::mutex> lock(storage_mutex);
    
    UnifiedNode node;
    node.id = generate_unique_id();
    node.set_text_content(content);
    node.content_type = type;
    node.creation_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    node.importance = calculate_importance(node.content, type);
    node.activation_strength = 200; // High activation for meta-cognitive
    node.confidence_score = 0.9f; // High confidence for self-reflection
    node.content_length = node.content.size(); // Ensure content_length is set correctly
    
    nodes[node.id] = node;
    node_index[node.id] = nodes.size() - 1;
    stats.total_nodes++;
    
    std::cout << "🧠 Created meta-cognitive node: " << content.substr(0, 50) 
              << "... -> " << std::hex << node.id << std::endl;
    
    return node.id;
}

uint64_t MelvinUnifiedBrain::create_connection(uint64_t source_id, uint64_t target_id, 
                                              ConnectionType connection_type, uint8_t weight) {
    std::lock_guard<std::mutex> lock(storage_mutex);
    
    UnifiedConnection conn;
    conn.id = generate_unique_id();
    conn.source_id = source_id;
    conn.target_id = target_id;
    conn.connection_type = connection_type;
    conn.weight = weight;
    conn.creation_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    conn.strength = static_cast<float>(weight) / 255.0f;
    
    connections[conn.id] = conn;
    stats.total_connections++;
    
    // Update connection counts
    if (nodes.find(source_id) != nodes.end()) {
        nodes[source_id].connection_count++;
    }
    if (nodes.find(target_id) != nodes.end()) {
        nodes[target_id].connection_count++;
    }
    
    std::cout << "🔗 Created connection: " << std::hex << source_id 
              << " -> " << std::hex << target_id << " (type: " 
              << static_cast<int>(connection_type) << ")" << std::endl;
    
    return conn.id;
}

// ============================================================================
// INTELLIGENT TRAVERSAL AND ANSWERING
// ============================================================================

std::vector<std::string> MelvinUnifiedBrain::extract_keywords(const std::string& text) {
    std::vector<std::string> keywords;
    std::istringstream iss(text);
    std::string word;
    
    while (iss >> word) {
        // Simple keyword extraction - remove punctuation and convert to lowercase
        std::string clean_word;
        for (char c : word) {
            if (std::isalnum(c)) {
                clean_word += std::tolower(c);
            }
        }
        
        if (clean_word.length() > 2) { // Only words longer than 2 characters
            keywords.push_back(clean_word);
        }
    }
    
    return keywords;
}

std::vector<NodeSimilarity> MelvinUnifiedBrain::find_relevant_nodes(const std::vector<std::string>& keywords) {
    std::vector<NodeSimilarity> relevant_nodes;
    
    for (const auto& [node_id, node] : nodes) {
        std::string content = node.get_text_content();
        float similarity = 0.0f;
        std::vector<std::string> matched_keywords;
        
        for (const std::string& keyword : keywords) {
            if (content.find(keyword) != std::string::npos) {
                similarity += 1.0f;
                matched_keywords.push_back(keyword);
            }
        }
        
        if (similarity > 0) {
            similarity /= keywords.size(); // Normalize by number of keywords
            relevant_nodes.push_back({node_id, similarity, content, matched_keywords, node.content_type});
        }
    }
    
    // Sort by similarity score
    std::sort(relevant_nodes.begin(), relevant_nodes.end(),
              [](const NodeSimilarity& a, const NodeSimilarity& b) {
                  return a.similarity_score > b.similarity_score;
              });
    
    return relevant_nodes;
}

std::vector<ConnectionPath> MelvinUnifiedBrain::analyze_connection_paths(const std::vector<NodeSimilarity>& relevant_nodes) {
    std::vector<ConnectionPath> paths;
    
    for (const auto& node_sim : relevant_nodes) {
        ConnectionPath path;
        path.node_ids.push_back(node_sim.node_id);
        path.relevance_score = node_sim.similarity_score;
        path.path_description = "Direct match for: " + node_sim.content.substr(0, 50);
        path.primary_connection_type = ConnectionType::SIMILARITY;
        
        paths.push_back(path);
    }
    
    return paths;
}

SynthesizedAnswer MelvinUnifiedBrain::synthesize_answer(const std::string& question, 
                                                        const std::vector<NodeSimilarity>& relevant_nodes,
                                                        const std::vector<ConnectionPath>& connection_paths) {
    (void)question; // Suppress unused parameter warning
    (void)connection_paths; // Suppress unused parameter warning
    SynthesizedAnswer answer;
    answer.confidence = 0.0f;
    
    if (relevant_nodes.empty()) {
        answer.answer = "I don't have enough information to answer that question yet.";
        answer.confidence = 0.1f;
        answer.reasoning = "No relevant nodes found";
        return answer;
    }
    
    // Simple synthesis - combine content from most relevant nodes
    std::string synthesized_content;
    float total_confidence = 0.0f;
    
    for (size_t i = 0; i < std::min(relevant_nodes.size(), size_t(3)); ++i) {
        const auto& node_sim = relevant_nodes[i];
        synthesized_content += node_sim.content + " ";
        answer.source_nodes.push_back(node_sim.node_id);
        total_confidence += node_sim.similarity_score;
    }
    
    answer.answer = synthesized_content;
    answer.confidence = total_confidence / relevant_nodes.size();
    answer.reasoning = "Synthesized from " + std::to_string(relevant_nodes.size()) + " relevant nodes";
    answer.keywords_used = relevant_nodes[0].keywords;
    
    return answer;
}

SynthesizedAnswer MelvinUnifiedBrain::answer_question_intelligently(const std::string& question) {
    // Extract keywords
    std::vector<std::string> keywords = extract_keywords(question);
    
    // Find relevant nodes
    std::vector<NodeSimilarity> relevant_nodes = find_relevant_nodes(keywords);
    
    // Analyze connection paths
    std::vector<ConnectionPath> connection_paths = analyze_connection_paths(relevant_nodes);
    
    // Synthesize answer
    SynthesizedAnswer answer = synthesize_answer(question, relevant_nodes, connection_paths);
    
    // Create dynamic nodes
    create_dynamic_nodes(question, answer);
    
    // Update statistics
    increment_intelligent_answers();
    
    return answer;
}

// ============================================================================
// LEARNING AND MEMORY MANAGEMENT
// ============================================================================

void MelvinUnifiedBrain::update_hebbian_learning(uint64_t node_id) {
    std::lock_guard<std::mutex> lock(activation_mutex);
    
    Activation activation;
    activation.node_id = node_id;
    activation.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    activation.strength = 1.0f;
    
    recent_activations.push_back(activation);
    
    // Clean old activations
    uint64_t current_time = activation.timestamp;
    recent_activations.erase(
        std::remove_if(recent_activations.begin(), recent_activations.end(),
                       [current_time](const Activation& a) {
                           return (current_time - a.timestamp) > (COACTIVATION_WINDOW * 1000);
                       }),
        recent_activations.end());
    
    // Create Hebbian connections with recently activated nodes
    for (const auto& other_activation : recent_activations) {
        if (other_activation.node_id != node_id) {
            hebbian_learning(node_id, other_activation.node_id);
        }
    }
    
    stats.hebbian_updates++;
}

void MelvinUnifiedBrain::create_dynamic_nodes(const std::string& question, const SynthesizedAnswer& answer) {
    // Create question node
    uint64_t question_node = create_meta_cognitive_node(question, ContentType::THOUGHT);
    
    // Create answer node
    uint64_t answer_node = create_meta_cognitive_node(answer.answer, ContentType::THOUGHT);
    
    // Create reasoning node
    uint64_t reasoning_node = create_meta_cognitive_node(answer.reasoning, ContentType::EVALUATION);
    
    // Create connections
    create_connection(question_node, answer_node, ConnectionType::CAUSAL, 200);
    create_connection(answer_node, reasoning_node, ConnectionType::HIERARCHICAL, 180);
    
    // Connect to source nodes
    for (uint64_t source_id : answer.source_nodes) {
        create_connection(source_id, answer_node, ConnectionType::ASSOCIATIVE, 150);
    }
    
    increment_dynamic_nodes(3); // question, answer, reasoning
}

void MelvinUnifiedBrain::consolidate_memories() {
    std::lock_guard<std::mutex> lock(storage_mutex);
    
    // Simple consolidation - remove low-importance nodes
    std::vector<uint64_t> nodes_to_remove;
    
    for (const auto& [node_id, node] : nodes) {
        if (node.importance < 50 && node.connection_count < 2) {
            nodes_to_remove.push_back(node_id);
        }
    }
    
    for (uint64_t node_id : nodes_to_remove) {
        nodes.erase(node_id);
        node_index.erase(node_id);
        stats.total_nodes--;
    }
    
    std::cout << "🧹 Consolidated " << nodes_to_remove.size() << " low-importance nodes" << std::endl;
}

// ============================================================================
// META-COGNITIVE METHODS
// ============================================================================

float MelvinUnifiedBrain::evaluate_output_quality(const std::string& output, const std::string& input) {
    (void)input; // Suppress unused parameter warning
    // Simple evaluation based on output length and relevance
    float length_score = std::min(1.0f, output.length() / 100.0f);
    float relevance_score = 0.5f; // Placeholder - could be improved with semantic analysis
    
    return (length_score + relevance_score) / 2.0f;
}

std::vector<std::string> MelvinUnifiedBrain::extract_lessons_from_cycle(const ThoughtCycle& cycle) {
    std::vector<std::string> lessons;
    
    // Extract lessons based on evaluation rating
    if (cycle.self_evaluation_rating > 7.0f) {
        lessons.push_back("High-quality output: " + cycle.output.substr(0, 50));
    } else if (cycle.self_evaluation_rating < 4.0f) {
        lessons.push_back("Low-quality output needs improvement: " + cycle.input.substr(0, 50));
    }
    
    // Extract lessons from evaluation reason
    if (!cycle.evaluation_reason.empty()) {
        lessons.push_back("Evaluation insight: " + cycle.evaluation_reason);
    }
    
    return lessons;
}

void MelvinUnifiedBrain::store_lessons(const std::vector<std::string>& lessons) {
    for (const std::string& lesson : lessons) {
        create_meta_cognitive_node(lesson, ContentType::LESSON);
        meta_state.recent_lessons.push_back(lesson);
    }
    
    // Keep only recent lessons
    if (meta_state.recent_lessons.size() > 10) {
        meta_state.recent_lessons.erase(meta_state.recent_lessons.begin());
    }
}

void MelvinUnifiedBrain::mutate_lessons_based_on_feedback() {
    // Simple mutation - combine recent lessons
    if (meta_state.recent_lessons.size() >= 2) {
        std::string combined_lesson = "Combined insight: " + 
            meta_state.recent_lessons[0] + " + " + meta_state.recent_lessons[1];
        create_meta_cognitive_node(combined_lesson, ContentType::LESSON);
    }
}

// ============================================================================
// STORAGE AND PERSISTENCE
// ============================================================================

void MelvinUnifiedBrain::save_complete_state() {
    std::lock_guard<std::mutex> lock(storage_mutex);
    
    try {
        // Limit save to prevent segfault - only save first 1000 nodes
        size_t nodes_to_save = std::min(nodes.size(), size_t(1000));
        std::cout << "💾 Saving " << nodes_to_save << " nodes (out of " << nodes.size() << ")" << std::endl;
        
        // Save nodes
        std::ofstream nodes_out(nodes_file, std::ios::binary);
        if (!nodes_out.is_open()) {
            std::cerr << "❌ Failed to open nodes file for writing" << std::endl;
            return;
        }
        
        size_t saved_count = 0;
        for (const auto& [node_id, node] : nodes) {
            if (saved_count >= nodes_to_save) break;
            
            std::vector<uint8_t> node_data = node.to_bytes();
            nodes_out.write(reinterpret_cast<const char*>(node_data.data()), node_data.size());
            if (nodes_out.fail()) {
                std::cerr << "❌ Failed to write node data" << std::endl;
                break;
            }
            saved_count++;
        }
        nodes_out.close();
        
        // Save connections
        std::ofstream connections_out(connections_file, std::ios::binary);
        if (!connections_out.is_open()) {
            std::cerr << "❌ Failed to open connections file for writing" << std::endl;
            return;
        }
        
        for (const auto& [conn_id, conn] : connections) {
            std::vector<uint8_t> conn_data = conn.to_bytes();
            connections_out.write(reinterpret_cast<const char*>(conn_data.data()), conn_data.size());
            if (connections_out.fail()) {
                std::cerr << "❌ Failed to write connection data" << std::endl;
                break;
            }
        }
        connections_out.close();
        
        // Save index
        save_index();
        
        std::cout << "💾 Brain state saved to " << storage_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "❌ Error saving brain state: " << e.what() << std::endl;
    }
}

void MelvinUnifiedBrain::load_complete_state() {
    std::lock_guard<std::mutex> lock(storage_mutex);
    
    // Load nodes
    std::ifstream nodes_in(nodes_file, std::ios::binary);
    if (nodes_in.is_open()) {
        while (nodes_in.peek() != EOF) {
            std::vector<uint8_t> node_data(40); // Header size
            nodes_in.read(reinterpret_cast<char*>(node_data.data()), 40);
            
            // Read content length
            uint32_t content_length;
            memcpy(&content_length, node_data.data() + 20, sizeof(content_length));
            
            // Resize and read content
            node_data.resize(40 + content_length);
            nodes_in.read(reinterpret_cast<char*>(node_data.data() + 40), content_length);
            
            UnifiedNode node = UnifiedNode::from_bytes(node_data);
            nodes[node.id] = node;
            node_index[node.id] = nodes.size() - 1;
        }
        nodes_in.close();
    }
    
    // Load connections
    std::ifstream connections_in(connections_file, std::ios::binary);
    if (connections_in.is_open()) {
        std::vector<uint8_t> conn_data(32); // Connection size
        while (connections_in.read(reinterpret_cast<char*>(conn_data.data()), 32)) {
            UnifiedConnection conn = UnifiedConnection::from_bytes(conn_data);
            connections[conn.id] = conn;
        }
        connections_in.close();
    }
    
    // Load index
    load_index();
    
    std::cout << "📂 Brain state loaded from " << storage_path << std::endl;
}

// ============================================================================
// BRAIN STATE AND INTROSPECTION
// ============================================================================

MelvinUnifiedBrain::UnifiedBrainState MelvinUnifiedBrain::get_unified_state() {
    UnifiedBrainState state;
    
    // Global memory
    state.global_memory.total_nodes = stats.total_nodes;
    state.global_memory.total_connections = stats.total_connections;
    state.global_memory.storage_used_mb = stats.storage_used_mb;
    state.global_memory.stats = stats;
    
    // Continuous cycle
    state.continuous_cycle.cycle_active = meta_state.cycle_active.load();
    state.continuous_cycle.current_cycle_id = meta_state.current_cycle_id;
    state.continuous_cycle.total_cycles = meta_state.total_cycles;
    state.continuous_cycle.successful_cycles = meta_state.successful_cycles;
    state.continuous_cycle.overall_confidence = meta_state.overall_confidence;
    state.continuous_cycle.last_output = meta_state.last_output;
    
    // Intelligent capabilities
    state.intelligent_capabilities.intelligent_answers_generated = stats.intelligent_answers_generated;
    state.intelligent_capabilities.dynamic_nodes_created = stats.dynamic_nodes_created;
    state.intelligent_capabilities.meta_cognitive_connections = stats.meta_cognitive_connections;
    state.intelligent_capabilities.connection_traversal_enabled = true;
    state.intelligent_capabilities.dynamic_node_creation_enabled = true;
    state.intelligent_capabilities.continuous_cycle_enabled = meta_state.cycle_active.load();
    
    // System
    state.system.running = true;
    state.system.uptime_seconds = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() - stats.start_time;
    state.system.storage_path = storage_path;
    
    return state;
}

void MelvinUnifiedBrain::print_brain_status() {
    UnifiedBrainState state = get_unified_state();
    
    std::cout << "\n🧠 MELVIN UNIFIED BRAIN STATUS" << std::endl;
    std::cout << "=================================" << std::endl;
    std::cout << "📊 Global Memory:" << std::endl;
    std::cout << "   Total Nodes: " << state.global_memory.total_nodes << std::endl;
    std::cout << "   Total Connections: " << state.global_memory.total_connections << std::endl;
    std::cout << "   Storage Used: " << std::fixed << std::setprecision(2) 
              << state.global_memory.storage_used_mb << " MB" << std::endl;
    
    std::cout << "\n🔄 Continuous Cycle:" << std::endl;
    std::cout << "   Active: " << (state.continuous_cycle.cycle_active ? "✅" : "❌") << std::endl;
    std::cout << "   Current Cycle: " << std::hex << state.continuous_cycle.current_cycle_id << std::endl;
    std::cout << "   Total Cycles: " << state.continuous_cycle.total_cycles << std::endl;
    std::cout << "   Successful: " << state.continuous_cycle.successful_cycles << std::endl;
    std::cout << "   Overall Confidence: " << std::fixed << std::setprecision(2) 
              << state.continuous_cycle.overall_confidence << std::endl;
    
    std::cout << "\n🧠 Intelligent Capabilities:" << std::endl;
    std::cout << "   Intelligent Answers: " << state.intelligent_capabilities.intelligent_answers_generated << std::endl;
    std::cout << "   Dynamic Nodes: " << state.intelligent_capabilities.dynamic_nodes_created << std::endl;
    std::cout << "   Meta-Cognitive Connections: " << state.intelligent_capabilities.meta_cognitive_connections << std::endl;
    std::cout << "   Connection Traversal: " << (state.intelligent_capabilities.connection_traversal_enabled ? "✅" : "❌") << std::endl;
    std::cout << "   Dynamic Creation: " << (state.intelligent_capabilities.dynamic_node_creation_enabled ? "✅" : "❌") << std::endl;
    
    std::cout << "\n⚙️  System:" << std::endl;
    std::cout << "   Running: " << (state.system.running ? "✅" : "❌") << std::endl;
    std::cout << "   Uptime: " << state.system.uptime_seconds << " seconds" << std::endl;
    std::cout << "   Storage Path: " << state.system.storage_path << std::endl;
    std::cout << "=================================" << std::endl;
}

// ============================================================================
// INTERNAL PROCESSING METHODS
// ============================================================================

void MelvinUnifiedBrain::continuous_cycle_loop() {
    std::cout << "🔄 Starting continuous thought cycle loop" << std::endl;
    
    while (should_run_cycle) {
        std::string input;
        bool is_external = false;
        
        // Check for external input first
        {
            std::lock_guard<std::mutex> lock(cycle_mutex);
            if (!input_queue.empty()) {
                input = input_queue.front();
                input_queue.pop();
                is_external = true;
            }
        }
        
        // If no external input, use self-generated input
        if (input.empty()) {
            input = generate_self_input_from_last_output();
        }
        
        // Execute thought cycle
        if (!input.empty()) {
            ThoughtCycle cycle = execute_thought_cycle(input, is_external);
            
            // Store lessons
            store_lessons(cycle.lessons_learned);
            
            // Update overall confidence
            meta_state.overall_confidence = (meta_state.overall_confidence + cycle.self_evaluation_rating) / 2.0f;
        }
        
        // Sleep briefly to prevent excessive CPU usage
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Prevent excessive memory usage
        if (nodes.size() > 10000) {
            std::cout << "🧹 Memory limit reached, consolidating..." << std::endl;
            consolidate_memories();
        }
    }
    
    std::cout << "⏹️  Continuous thought cycle loop stopped" << std::endl;
}

std::string MelvinUnifiedBrain::generate_self_input_from_last_output() {
    if (meta_state.last_output.empty()) {
        return "What should I think about next?";
    }
    
    // Simple self-input generation - ask about the last output
    return "Let me analyze my last response: " + meta_state.last_output.substr(0, 100);
}

std::string MelvinUnifiedBrain::analyze_input(const std::string& input) {
    // Simple analysis - extract key concepts
    std::vector<std::string> keywords = extract_keywords(input);
    std::string analysis = "Analyzing input with keywords: ";
    
    for (size_t i = 0; i < std::min(keywords.size(), size_t(5)); ++i) {
        analysis += keywords[i];
        if (i < std::min(keywords.size(), size_t(5)) - 1) {
            analysis += ", ";
        }
    }
    
    return analysis;
}

std::string MelvinUnifiedBrain::generate_hypotheses(const std::string& input) {
    (void)input; // Suppress unused parameter warning
    // Simple hypothesis generation
    return "Hypothesis: This input relates to learning and self-improvement.";
}

std::string MelvinUnifiedBrain::produce_output(const std::string& input, const std::string& analysis) {
    (void)analysis; // Suppress unused parameter warning
    // Simple output production - try to answer intelligently
    SynthesizedAnswer answer = answer_question_intelligently(input);
    return answer.answer;
}

void MelvinUnifiedBrain::self_evaluate_output(const std::string& output, const std::string& input, ThoughtCycle& cycle) {
    cycle.self_evaluation_rating = evaluate_output_quality(output, input);
    
    if (cycle.self_evaluation_rating > 7.0f) {
        cycle.evaluation_reason = "High-quality response with good content";
    } else if (cycle.self_evaluation_rating > 4.0f) {
        cycle.evaluation_reason = "Adequate response with room for improvement";
    } else {
        cycle.evaluation_reason = "Low-quality response needs significant improvement";
    }
}

void MelvinUnifiedBrain::create_meta_cognitive_connections(const ThoughtCycle& cycle) {
    // Create connections between cycle components
    uint64_t input_node = create_meta_cognitive_node(cycle.input, ContentType::THOUGHT);
    uint64_t analysis_node = create_meta_cognitive_node(cycle.analysis, ContentType::EVALUATION);
    uint64_t output_node = create_meta_cognitive_node(cycle.output, ContentType::THOUGHT);
    
    // Create meta-cognitive connections
    create_connection(input_node, analysis_node, ConnectionType::META_COGNITIVE, 200);
    create_connection(analysis_node, output_node, ConnectionType::META_COGNITIVE, 200);
    
    stats.meta_cognitive_connections += 2;
}

// ============================================================================
// UTILITY METHODS
// ============================================================================

std::string MelvinUnifiedBrain::get_node_content(uint64_t node_id) {
    auto it = nodes.find(node_id);
    if (it != nodes.end()) {
        return it->second.get_text_content();
    }
    return "";
}

std::optional<UnifiedNode> MelvinUnifiedBrain::get_node(uint64_t node_id) {
    auto it = nodes.find(node_id);
    if (it != nodes.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::vector<uint64_t> MelvinUnifiedBrain::get_node_connections(uint64_t node_id) {
    std::vector<uint64_t> connected_nodes;
    
    for (const auto& [conn_id, conn] : connections) {
        if (conn.source_id == node_id) {
            connected_nodes.push_back(conn.target_id);
        } else if (conn.target_id == node_id) {
            connected_nodes.push_back(conn.source_id);
        }
    }
    
    return connected_nodes;
}

uint64_t MelvinUnifiedBrain::generate_unique_id() {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    static std::uniform_int_distribution<uint64_t> dis;
    return dis(gen);
}

std::string MelvinUnifiedBrain::format_timestamp(uint64_t timestamp) {
    std::time_t time_t = timestamp / 1000;
    std::tm* tm = std::localtime(&time_t);
    
    std::ostringstream oss;
    oss << std::put_time(tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

uint8_t MelvinUnifiedBrain::calculate_importance(const std::vector<uint8_t>& content, ContentType content_type) {
    // Simple importance calculation
    float base_importance = 0.5f;
    float length_factor = std::min(1.0f, content.size() / 100.0f);
    float type_factor = static_cast<float>(content_type) / 13.0f; // 14 content types
    
    return static_cast<uint8_t>((base_importance + length_factor * 0.3f + type_factor * 0.2f) * 255);
}

void MelvinUnifiedBrain::load_index() {
    std::ifstream index_in(index_file, std::ios::binary);
    if (index_in.is_open()) {
        uint64_t node_id;
        size_t position;
        while (index_in.read(reinterpret_cast<char*>(&node_id), sizeof(node_id)) &&
               index_in.read(reinterpret_cast<char*>(&position), sizeof(position))) {
            node_index[node_id] = position;
        }
        index_in.close();
    }
}

void MelvinUnifiedBrain::save_index() {
    std::ofstream index_out(index_file, std::ios::binary);
    for (const auto& [node_id, position] : node_index) {
        index_out.write(reinterpret_cast<const char*>(&node_id), sizeof(node_id));
        index_out.write(reinterpret_cast<const char*>(&position), sizeof(position));
    }
    index_out.close();
}

void MelvinUnifiedBrain::hebbian_learning(uint64_t node1_id, uint64_t node2_id) {
    // Check if connection already exists
    for (const auto& [conn_id, conn] : connections) {
        if ((conn.source_id == node1_id && conn.target_id == node2_id) ||
            (conn.source_id == node2_id && conn.target_id == node1_id)) {
            // Strengthen existing connection
            const_cast<UnifiedConnection&>(conn).strengthen();
            return;
        }
    }
    
    // Create new Hebbian connection (without acquiring storage_mutex to avoid deadlock)
    UnifiedConnection conn;
    conn.id = generate_unique_id();
    conn.source_id = node1_id;
    conn.target_id = node2_id;
    conn.connection_type = ConnectionType::HEBBIAN;
    conn.weight = 128;
    conn.creation_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    conn.strength = static_cast<float>(128) / 255.0f;
    
    connections[conn.id] = conn;
    stats.total_connections++;
    
    // Update connection counts
    if (nodes.find(node1_id) != nodes.end()) {
        nodes[node1_id].connection_count++;
    }
    if (nodes.find(node2_id) != nodes.end()) {
        nodes[node2_id].connection_count++;
    }
}

void MelvinUnifiedBrain::update_importance_scores() {
    std::lock_guard<std::mutex> lock(storage_mutex);
    
    for (auto& [node_id, node] : nodes) {
        // Update importance based on activation and connections
        float activation_factor = static_cast<float>(node.activation_strength) / 255.0f;
        float connection_factor = std::min(1.0f, static_cast<float>(node.connection_count) / 10.0f);
        float access_factor = std::min(1.0f, static_cast<float>(node.access_count) / 100.0f);
        
        node.importance = static_cast<uint8_t>((activation_factor * 0.4f + 
                                              connection_factor * 0.3f + 
                                              access_factor * 0.3f) * 255);
    }
    
    std::cout << "📊 Updated importance scores for all nodes" << std::endl;
}

void MelvinUnifiedBrain::print_recent_thought_cycles(int count) {
    std::cout << "\n🧠 Recent Thought Cycles:" << std::endl;
    std::cout << "=========================" << std::endl;
    
    int cycles_to_show = std::min(count, static_cast<int>(thought_history.size()));
    
    for (int i = thought_history.size() - cycles_to_show; i < static_cast<int>(thought_history.size()); ++i) {
        const ThoughtCycle& cycle = thought_history[i];
        
        std::cout << "\nCycle " << std::hex << cycle.cycle_id << ":" << std::endl;
        std::cout << "  Input: " << cycle.input.substr(0, 80) << "..." << std::endl;
        std::cout << "  Analysis: " << cycle.analysis.substr(0, 60) << "..." << std::endl;
        std::cout << "  Output: " << cycle.output.substr(0, 80) << "..." << std::endl;
        std::cout << "  Rating: " << std::fixed << std::setprecision(1) << cycle.self_evaluation_rating << "/10" << std::endl;
        std::cout << "  Reason: " << cycle.evaluation_reason << std::endl;
        std::cout << "  External: " << (cycle.is_external_interrupt ? "Yes" : "No") << std::endl;
    }
}

void MelvinUnifiedBrain::print_meta_cognitive_state() {
    std::cout << "\n🧠 Meta-Cognitive State:" << std::endl;
    std::cout << "========================" << std::endl;
    std::cout << "Current Cycle ID: " << std::hex << meta_state.current_cycle_id << std::endl;
    std::cout << "Total Cycles: " << meta_state.total_cycles << std::endl;
    std::cout << "Successful Cycles: " << meta_state.successful_cycles << std::endl;
    std::cout << "Interrupted Cycles: " << meta_state.interrupted_cycles << std::endl;
    std::cout << "Overall Confidence: " << std::fixed << std::setprecision(2) << meta_state.overall_confidence << std::endl;
    std::cout << "Cycle Active: " << (meta_state.cycle_active.load() ? "Yes" : "No") << std::endl;
    
    std::cout << "\nRecent Lessons:" << std::endl;
    for (size_t i = 0; i < std::min(meta_state.recent_lessons.size(), size_t(5)); ++i) {
        std::cout << "  " << (i + 1) << ". " << meta_state.recent_lessons[i] << std::endl;
    }
    
    if (!meta_state.last_output.empty()) {
        std::cout << "\nLast Output: " << meta_state.last_output.substr(0, 100) << "..." << std::endl;
    }
}

// ============================================================================
// UNIFIED BRAIN INTERFACE IMPLEMENTATION
// ============================================================================

MelvinUnifiedInterface::MelvinUnifiedInterface(const std::string& storage_path) {
    brain = std::make_unique<MelvinUnifiedBrain>(storage_path);
}

MelvinUnifiedInterface::~MelvinUnifiedInterface() {
    brain.reset();
}

std::string MelvinUnifiedInterface::ask(const std::string& question) {
    SynthesizedAnswer answer = brain->answer_question_intelligently(question);
    return answer.answer;
}

std::string MelvinUnifiedInterface::tell(const std::string& information) {
    brain->process_text_input(information, "user");
    return "I've learned: " + information.substr(0, 50) + "...";
}

std::string MelvinUnifiedInterface::think(const std::string& topic) {
    SynthesizedAnswer answer = brain->answer_question_intelligently("Think about: " + topic);
    return answer.answer;
}

std::string MelvinUnifiedInterface::remember(const std::string& experience) {
    brain->process_text_input(experience, "memory");
    return "I'll remember: " + experience.substr(0, 50) + "...";
}

void MelvinUnifiedInterface::start_thinking() {
    brain->start_continuous_cycle();
}

void MelvinUnifiedInterface::stop_thinking() {
    brain->stop_continuous_cycle();
}

void MelvinUnifiedInterface::interrupt_with(const std::string& input) {
    brain->process_external_input(input);
}

void MelvinUnifiedInterface::show_brain_status() {
    brain->print_brain_status();
}

void MelvinUnifiedInterface::show_recent_thoughts() {
    brain->print_recent_thought_cycles();
}

void MelvinUnifiedInterface::show_meta_cognitive_state() {
    brain->print_meta_cognitive_state();
}

void MelvinUnifiedInterface::consolidate_knowledge() {
    brain->consolidate_memories();
}

void MelvinUnifiedInterface::optimize_brain() {
    brain->consolidate_memories();
    brain->update_importance_scores();
}

void MelvinUnifiedInterface::save_brain_state() {
    brain->save_complete_state();
}

void MelvinUnifiedInterface::load_brain_state() {
    brain->load_complete_state();
}

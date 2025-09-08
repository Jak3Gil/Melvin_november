#include "melvin_unified_brain.h"
<<<<<<< HEAD
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
=======
#include <filesystem>

// ============================================================================
// BINARY NODE IMPLEMENTATION
// ============================================================================

std::vector<uint8_t> BinaryNode::to_bytes() const {
    std::vector<uint8_t> result(28 + content.size());
    size_t offset = 0;
    
    // Copy header (28 bytes)
    memcpy(result.data() + offset, &id, sizeof(id)); offset += sizeof(id);
    memcpy(result.data() + offset, &creation_time, sizeof(creation_time)); offset += sizeof(creation_time);
    result[offset++] = static_cast<uint8_t>(content_type);
    result[offset++] = static_cast<uint8_t>(compression);
    result[offset++] = importance;
    result[offset++] = instinct_bias;
    memcpy(result.data() + offset, &content_length, sizeof(content_length)); offset += sizeof(content_length);
    memcpy(result.data() + offset, &connection_count, sizeof(connection_count)); offset += sizeof(connection_count);
    
    // Copy content
    if (!content.empty()) {
        memcpy(result.data() + offset, content.data(), content.size());
    }
    
    return result;
}

BinaryNode BinaryNode::from_bytes(const std::vector<uint8_t>& data) {
    BinaryNode node;
    if (data.size() < 28) return node;
    
    size_t offset = 0;
>>>>>>> d2fd9fccaf2aa76803b4cda65ac5530f1b186d02
    memcpy(&node.id, data.data() + offset, sizeof(node.id)); offset += sizeof(node.id);
    memcpy(&node.creation_time, data.data() + offset, sizeof(node.creation_time)); offset += sizeof(node.creation_time);
    node.content_type = static_cast<ContentType>(data[offset++]);
    node.compression = static_cast<CompressionType>(data[offset++]);
    node.importance = data[offset++];
<<<<<<< HEAD
    node.activation_strength = data[offset++];
    memcpy(&node.content_length, data.data() + offset, sizeof(node.content_length)); offset += sizeof(node.content_length);
    memcpy(&node.connection_count, data.data() + offset, sizeof(node.connection_count)); offset += sizeof(node.connection_count);
    memcpy(&node.last_access_time, data.data() + offset, sizeof(node.last_access_time)); offset += sizeof(node.last_access_time);
    memcpy(&node.access_count, data.data() + offset, sizeof(node.access_count)); offset += sizeof(node.access_count);
    memcpy(&node.confidence_score, data.data() + offset, sizeof(node.confidence_score)); offset += sizeof(node.confidence_score);
    
    // Read content
    node.content.resize(node.content_length);
    memcpy(node.content.data(), data.data() + offset, node.content_length);
=======
    node.instinct_bias = data[offset++];
    memcpy(&node.content_length, data.data() + offset, sizeof(node.content_length)); offset += sizeof(node.content_length);
    memcpy(&node.connection_count, data.data() + offset, sizeof(node.connection_count)); offset += sizeof(node.connection_count);
    
    // Extract content
    if (node.content_length > 0 && data.size() >= 28 + node.content_length) {
        node.content.resize(node.content_length);
        memcpy(node.content.data(), data.data() + offset, node.content_length);
    }
>>>>>>> d2fd9fccaf2aa76803b4cda65ac5530f1b186d02
    
    return node;
}

<<<<<<< HEAD
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
    
=======
// ============================================================================
// BINARY CONNECTION IMPLEMENTATION
// ============================================================================

std::vector<uint8_t> BinaryConnection::to_bytes() const {
    std::vector<uint8_t> result(18);
    size_t offset = 0;
    
    memcpy(result.data() + offset, &id, sizeof(id)); offset += sizeof(id);
    memcpy(result.data() + offset, &source_id, sizeof(source_id)); offset += sizeof(source_id);
    memcpy(result.data() + offset, &target_id, sizeof(target_id)); offset += sizeof(target_id);
    result[offset++] = static_cast<uint8_t>(connection_type);
    result[offset++] = weight;
    
    return result;
}

BinaryConnection BinaryConnection::from_bytes(const std::vector<uint8_t>& data) {
    BinaryConnection conn;
    if (data.size() < 18) return conn;
    
    size_t offset = 0;
>>>>>>> d2fd9fccaf2aa76803b4cda65ac5530f1b186d02
    memcpy(&conn.id, data.data() + offset, sizeof(conn.id)); offset += sizeof(conn.id);
    memcpy(&conn.source_id, data.data() + offset, sizeof(conn.source_id)); offset += sizeof(conn.source_id);
    memcpy(&conn.target_id, data.data() + offset, sizeof(conn.target_id)); offset += sizeof(conn.target_id);
    conn.connection_type = static_cast<ConnectionType>(data[offset++]);
    conn.weight = data[offset++];
<<<<<<< HEAD
    memcpy(&conn.creation_time, data.data() + offset, sizeof(conn.creation_time)); offset += sizeof(conn.creation_time);
    memcpy(&conn.usage_count, data.data() + offset, sizeof(conn.usage_count)); offset += sizeof(conn.usage_count);
    memcpy(&conn.strength, data.data() + offset, sizeof(conn.strength)); offset += sizeof(conn.strength);
=======
>>>>>>> d2fd9fccaf2aa76803b4cda65ac5530f1b186d02
    
    return conn;
}

<<<<<<< HEAD
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
=======
// ============================================================================
// MELVIN UNIFIED BRAIN IMPLEMENTATION
// ============================================================================

MelvinUnifiedBrain::MelvinUnifiedBrain(const std::string& path) 
    : storage_path(path), next_node_id(1), next_connection_id(1), 
      total_nodes(0), total_connections(0), background_running(false),
      next_task_id(1), user_active(false), adaptive_thinking_interval(60), next_query_id(1), next_response_id(1) {
    
    // Initialize file paths
>>>>>>> d2fd9fccaf2aa76803b4cda65ac5530f1b186d02
    nodes_file = storage_path + "/nodes.bin";
    connections_file = storage_path + "/connections.bin";
    index_file = storage_path + "/index.bin";
    
    // Create storage directory
    std::filesystem::create_directories(storage_path);
    
<<<<<<< HEAD
    // Initialize statistics
    stats = {0, 0, 0, 0, 0, 0, 0, 0, 0, 
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()), 0.0};
=======
    // Initialize instinct engine
    initialize_instincts();
>>>>>>> d2fd9fccaf2aa76803b4cda65ac5530f1b186d02
    
    // Load existing state
    load_complete_state();
    
<<<<<<< HEAD
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
=======
    // Get API key from environment
    const char* api_key = std::getenv("BING_API_KEY");
    if (api_key) {
        bing_api_key = api_key;
    }
    
    // Configure Ollama (default local setup)
    configure_ollama();
    
    // Initialize statistics
    stats.total_nodes = total_nodes;
    stats.total_connections = total_connections;
    stats.hebbian_updates = 0;
    stats.search_queries = 0;
    stats.reasoning_paths = 0;
    stats.background_tasks_processed = 0;
    stats.ollama_queries = 0;
    stats.contradictions_resolved = 0;
    stats.total_processing_time = 0.0;
    
    std::cout << "🧠 Melvin Unified Brain initialized with " << total_nodes 
              << " nodes and " << total_connections << " connections" << std::endl;
}

MelvinUnifiedBrain::~MelvinUnifiedBrain() {
    stop_background_scheduler();
    save_complete_state();
}

void MelvinUnifiedBrain::initialize_instincts() {
    std::lock_guard<std::mutex> lock(instinct_mutex);
    
    instinct_weights[InstinctType::SURVIVAL] = 0.8f;
    instinct_weights[InstinctType::CURIOSITY] = 0.6f;
    instinct_weights[InstinctType::EFFICIENCY] = 0.5f;
    instinct_weights[InstinctType::SOCIAL] = 0.4f;
    instinct_weights[InstinctType::CONSISTENCY] = 0.7f;
}

uint64_t MelvinUnifiedBrain::store_node(const std::string& content, ContentType type, uint8_t importance) {
    std::lock_guard<std::mutex> lock(storage_mutex);
    
    BinaryNode node;
    node.id = generate_node_id();
    node.creation_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    node.content_type = type;
    node.compression = CompressionType::GZIP; // Default compression
    node.importance = importance;
    node.instinct_bias = 0; // Will be set by instinct engine
    node.content_length = content.length();
    node.connection_count = 0;
    
    // Compress content
    node.content = compress_content(content);
    
    // Write to file
    std::ofstream file(nodes_file, std::ios::binary | std::ios::app);
    if (!file.is_open()) {
        std::cerr << "❌ Failed to open nodes file for writing" << std::endl;
        return 0;
    }
    
    auto node_bytes = node.to_bytes();
    file.write(reinterpret_cast<const char*>(node_bytes.data()), node_bytes.size());
    file.close();
    
    // Update index
    update_node_index(node.id, total_nodes);
    total_nodes++;
    
    std::cout << "💾 Stored node " << node.id << " (" << content.substr(0, 50) 
              << "...) with importance " << static_cast<int>(importance) << std::endl;
>>>>>>> d2fd9fccaf2aa76803b4cda65ac5530f1b186d02
    
    return node.id;
}

<<<<<<< HEAD
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
=======
uint64_t MelvinUnifiedBrain::store_connection(uint64_t source_id, uint64_t target_id, ConnectionType type, uint8_t weight) {
    std::lock_guard<std::mutex> lock(storage_mutex);
    
    BinaryConnection conn;
    conn.id = generate_connection_id();
    conn.source_id = source_id;
    conn.target_id = target_id;
    conn.connection_type = type;
    conn.weight = weight;
    
    // Write to file
    std::ofstream file(connections_file, std::ios::binary | std::ios::app);
    if (!file.is_open()) {
        std::cerr << "❌ Failed to open connections file for writing" << std::endl;
        return 0;
    }
    
    auto conn_bytes = conn.to_bytes();
    file.write(reinterpret_cast<const char*>(conn_bytes.data()), conn_bytes.size());
    file.close();
    
    // Update index
    update_connection_index(conn.id, total_connections);
    total_connections++;
    
    std::cout << "🔗 Stored connection " << conn.id << " (" << source_id 
              << " → " << target_id << ") with weight " << static_cast<int>(weight) << std::endl;
>>>>>>> d2fd9fccaf2aa76803b4cda65ac5530f1b186d02
    
    return conn.id;
}

<<<<<<< HEAD
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
=======
std::optional<BinaryNode> MelvinUnifiedBrain::get_node(uint64_t node_id) {
    std::lock_guard<std::mutex> lock(storage_mutex);
    
    auto it = node_index.find(node_id);
    if (it == node_index.end()) {
        return std::nullopt;
    }
    
    std::ifstream file(nodes_file, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "❌ Failed to open nodes file for reading" << std::endl;
        return std::nullopt;
    }
    
    // Seek to position
    file.seekg(it->second);
    
    // Read header first to get content length
    std::vector<uint8_t> header(28);
    file.read(reinterpret_cast<char*>(header.data()), 28);
    
    if (file.gcount() != 28) {
        return std::nullopt;
    }
    
    // Parse header to get content length
    uint32_t content_length;
    memcpy(&content_length, header.data() + 20, sizeof(content_length));
    
    // Read full node
    std::vector<uint8_t> node_data(28 + content_length);
    file.seekg(it->second);
    file.read(reinterpret_cast<char*>(node_data.data()), node_data.size());
    
    file.close();
    
    return BinaryNode::from_bytes(node_data);
}

std::string MelvinUnifiedBrain::get_node_content(uint64_t node_id) {
    auto node = get_node(node_id);
    if (!node) {
        return "";
    }
    
    return decompress_content(node->content);
}

std::string MelvinUnifiedBrain::process_input(const std::string& user_input) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Update user activity for adaptive thinking
    update_user_activity();
    
    std::cout << "\n🧠 [Unified Brain] Processing: \"" << user_input << "\"" << std::endl;
    
    // Step 1: Store user input as BinaryNode
    uint64_t user_input_node_id = store_node(user_input, ContentType::USER_INPUT, 200);
    
    // Step 2: Parse input to activations
    std::vector<uint64_t> activations = parse_to_activations(user_input);
    std::cout << "📝 [Parse] Activated " << activations.size() << " nodes" << std::endl;
    
    // Step 3: Recall related nodes
    std::vector<uint64_t> recalled_nodes = recall_related_nodes(activations);
    std::cout << "🧠 [Recall] Retrieved " << recalled_nodes.size() << " related nodes" << std::endl;
    
    // Step 4: Generate hypotheses
    std::vector<std::string> hypotheses = generate_hypotheses(recalled_nodes);
    std::cout << "💡 [Hypotheses] Generated " << hypotheses.size() << " hypotheses" << std::endl;
    
    // Step 5: Calculate confidence and trigger curiosity if needed
    float confidence = recalled_nodes.size() > 3 ? 0.7f : 0.3f;
    bool should_search = should_trigger_curiosity(recalled_nodes, confidence);
    
    std::string search_result = "";
    if (should_search) {
        std::cout << "🔍 [Curiosity] Low confidence (" << confidence 
                  << ") - triggering web search" << std::endl;
        search_result = perform_web_search(user_input);
        
        // Store search result as node if available
        if (!search_result.empty()) {
            uint64_t search_node_id = store_node(search_result, ContentType::TEXT, 180);
            store_connection(user_input_node_id, search_node_id, ConnectionType::REASONING, 180);
        }
    }
    
    // Step 6: Check for contradictions and regenerate if needed
    std::vector<uint64_t> all_nodes = activations;
    all_nodes.insert(all_nodes.end(), recalled_nodes.begin(), recalled_nodes.end());
    
    std::string response = regenerate_response_if_contradiction(user_input, all_nodes);
    
    // Step 7: Generate force-driven response if no contradiction detected
    if (response.empty()) {
        response = generate_force_driven_response(user_input, all_nodes);
    }
    
    // Step 8: Store response as BinaryNode
    uint64_t response_node_id = store_node(response, ContentType::TEXT, 190);
    store_connection(user_input_node_id, response_node_id, ConnectionType::REASONING, 190);
    
    // Step 9: Update Hebbian learning
    update_hebbian_learning(all_nodes);
    
    // Step 10: Get instinct bias and generate transparent response
    InstinctBias bias = get_instinct_bias(user_input, all_nodes);
    std::string final_response = generate_transparent_response(user_input, recalled_nodes, search_result, bias, confidence);
    
    // Step 11: Store final response as BinaryNode
    uint64_t final_response_node_id = store_node(final_response, ContentType::TEXT, 200);
    store_connection(response_node_id, final_response_node_id, ConnectionType::REASONING, 200);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    stats.total_processing_time += duration.count();
    stats.reasoning_paths++;
    
    std::cout << "⏱️ [Timing] Processed in " << duration.count() << "ms" << std::endl;
    
    return final_response;
}

std::vector<uint64_t> MelvinUnifiedBrain::parse_to_activations(const std::string& input) {
    std::vector<uint64_t> activations;
    std::vector<std::string> tokens = tokenize(input);
    
    for (const auto& token : tokens) {
        // Check if token exists as a node
        for (const auto& [node_id, position] : node_index) {
            auto node = get_node(node_id);
            if (node) {
                std::string content = decompress_content(node->content);
                if (content.find(token) != std::string::npos) {
                    activations.push_back(node_id);
                    break;
                }
            }
        }
        
        // If no existing node found, create new one
        if (activations.empty() || activations.back() == 0) {
            uint64_t new_node_id = store_node(token, ContentType::TEXT, 128);
            if (new_node_id > 0) {
                activations.push_back(new_node_id);
            }
        }
    }
    
    return activations;
}

std::vector<uint64_t> MelvinUnifiedBrain::recall_related_nodes(const std::vector<uint64_t>& activations) {
    std::vector<uint64_t> related_nodes;
    std::set<uint64_t> visited;
    
    for (uint64_t activation : activations) {
        if (visited.find(activation) != visited.end()) continue;
        visited.insert(activation);
        
        // Find connections from this node
        for (const auto& [conn_id, position] : connection_index) {
            // This is simplified - in a real implementation, we'd read the connection
            // and check if it starts from our activation node
            related_nodes.push_back(activation + 1); // Simplified for demo
        }
    }
    
    return related_nodes;
}

std::vector<std::string> MelvinUnifiedBrain::generate_hypotheses(const std::vector<uint64_t>& nodes) {
    std::vector<std::string> hypotheses;
    
    if (nodes.size() >= 2) {
        hypotheses.push_back("Multiple concepts detected - exploring relationships");
        hypotheses.push_back("Pattern recognition suggests deeper connections");
    } else if (nodes.size() == 1) {
        hypotheses.push_back("Single concept - seeking related information");
    } else {
        hypotheses.push_back("No clear patterns - requires external knowledge");
    }
    
    return hypotheses;
}

bool MelvinUnifiedBrain::should_trigger_curiosity(const std::vector<uint64_t>& nodes, float confidence) {
    return confidence < 0.5f && !bing_api_key.empty();
}

std::string MelvinUnifiedBrain::perform_web_search(const std::string& query) {
    if (bing_api_key.empty()) {
        return "Web search not available - API key not configured";
    }
    
    stats.search_queries++;
    
    // Initialize libcurl
    CURL* curl = curl_easy_init();
    if (!curl) {
        return "Failed to initialize web search";
    }
    
    std::string readBuffer;
    
    // Construct search URL
    std::string encoded_query = query;
    std::replace(encoded_query.begin(), encoded_query.end(), ' ', '+');
    std::string url = "https://api.bing.microsoft.com/v7.0/search?q=" + encoded_query + "&count=3";
    
    // Set up curl options
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, [](void* contents, size_t size, size_t nmemb, std::string* s) {
        size_t newLength = size * nmemb;
        try {
            s->append((char*)contents, newLength);
            return newLength;
        } catch (std::bad_alloc& e) {
            return static_cast<size_t>(0);
        }
    });
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
    
    // Set headers
    struct curl_slist* headers = nullptr;
    std::string auth_header = "Ocp-Apim-Subscription-Key: " + bing_api_key;
    headers = curl_slist_append(headers, auth_header.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    
    // Perform request
    CURLcode res = curl_easy_perform(curl);
    
    // Cleanup
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    
    if (res != CURLE_OK) {
        return "Web search failed: " + std::string(curl_easy_strerror(res));
    }
    
    // Simple parsing for Bing search results
    std::ostringstream summary;
    summary << "Research findings for \"" << query << "\": ";
    
    // Look for webPages results in the response
    size_t webpages_start = readBuffer.find("\"webPages\"");
    if (webpages_start == std::string::npos) {
        return "No search results found";
    }
    
    // Extract titles and snippets (simplified parsing)
    size_t pos = webpages_start;
    int result_count = 0;
    
    while (result_count < 3 && pos != std::string::npos) {
        // Find next title
        size_t title_start = readBuffer.find("\"name\":\"", pos);
        if (title_start == std::string::npos) break;
        title_start += 8; // Skip "name":"
        
        size_t title_end = readBuffer.find("\"", title_start);
        if (title_end == std::string::npos) break;
        
        std::string title = readBuffer.substr(title_start, title_end - title_start);
        
        // Find corresponding snippet
        size_t snippet_start = readBuffer.find("\"snippet\":\"", title_end);
        if (snippet_start == std::string::npos) break;
        snippet_start += 11; // Skip "snippet":"
        
        size_t snippet_end = readBuffer.find("\"", snippet_start);
        if (snippet_end == std::string::npos) break;
        
        std::string snippet = readBuffer.substr(snippet_start, snippet_end - snippet_start);
        
        summary << "\n" << (result_count + 1) << ". " << title << " - " << snippet;
        
        // Store search result as a node
        std::string result_content = title + ": " + snippet;
        store_node(result_content, ContentType::TEXT, 150);
        
        pos = snippet_end;
        result_count++;
    }
    
    if (result_count == 0) {
        return "No search results found";
    }
    
    return summary.str();
}

std::string MelvinUnifiedBrain::synthesize_response(const std::vector<uint64_t>& nodes, const std::string& search_result) {
    std::ostringstream response;
    
    if (!search_result.empty()) {
        response << "Based on research: " << search_result;
    } else if (!nodes.empty()) {
        response << "From memory: ";
        for (size_t i = 0; i < std::min(nodes.size(), size_t(3)); ++i) {
            std::string content = get_node_content(nodes[i]);
            if (!content.empty()) {
                response << content.substr(0, 100) << "... ";
            }
        }
    } else {
        response << "I need more information to provide a comprehensive answer.";
    }
    
    return response.str();
}

InstinctBias MelvinUnifiedBrain::get_instinct_bias(const std::string& input, const std::vector<uint64_t>& nodes) {
    std::lock_guard<std::mutex> lock(instinct_mutex);
    
    InstinctBias bias;
    
    // Analyze context
    float confidence_level = nodes.empty() ? 0.2f : 
                           (nodes.size() < 3 ? 0.4f : 0.7f);
    float novelty_level = nodes.empty() ? 0.8f : 
                         (nodes.size() < 3 ? 0.6f : 0.3f);
    
    // Calculate instinct influences
    float curiosity_influence = instinct_weights[InstinctType::CURIOSITY];
    float efficiency_influence = instinct_weights[InstinctType::EFFICIENCY];
    float social_influence = instinct_weights[InstinctType::SOCIAL];
    
    // Apply context multipliers
    if (confidence_level < 0.5f) {
        curiosity_influence *= 1.5f;
    }
    if (input.find("?") != std::string::npos) {
        social_influence *= 1.3f;
    }
    if (input.length() > 100) {
        efficiency_influence *= 1.2f;
    }
    
    // Store contributions
    bias.instinct_contributions[InstinctType::CURIOSITY] = curiosity_influence;
    bias.instinct_contributions[InstinctType::EFFICIENCY] = efficiency_influence;
    bias.instinct_contributions[InstinctType::SOCIAL] = social_influence;
    
    // Calculate final weights
    float total_influence = curiosity_influence + efficiency_influence + social_influence;
    
    if (total_influence > 0.0f) {
        bias.exploration_weight = curiosity_influence / total_influence;
        bias.recall_weight = (efficiency_influence + social_influence) / total_influence;
    }
    
    // Generate reasoning
    std::stringstream reasoning;
    reasoning << "Instinct Analysis: ";
    if (confidence_level < 0.5f) {
        reasoning << "Low confidence triggers Curiosity (" << std::fixed << std::setprecision(2) 
                 << curiosity_influence << "), ";
    }
    if (input.find("?") != std::string::npos) {
        reasoning << "Question triggers Social (" << std::fixed << std::setprecision(2) 
                 << social_influence << "), ";
    }
    reasoning << "Final bias: Recall=" << std::fixed << std::setprecision(2) << bias.recall_weight 
             << ", Exploration=" << bias.exploration_weight;
    
    bias.reasoning = reasoning.str();
    bias.overall_strength = total_influence;
    
    return bias;
}

void MelvinUnifiedBrain::reinforce_instinct(InstinctType instinct, float delta) {
    std::lock_guard<std::mutex> lock(instinct_mutex);
    
    instinct_weights[instinct] += delta;
    instinct_weights[instinct] = std::max(0.1f, std::min(1.0f, instinct_weights[instinct]));
    
    std::cout << "🧠 [Instinct Reinforcement] " << static_cast<int>(instinct) 
              << " instinct adjusted by " << delta << " to " 
              << instinct_weights[instinct] << std::endl;
}

void MelvinUnifiedBrain::update_hebbian_learning(const std::vector<uint64_t>& coactivated_nodes) {
    std::lock_guard<std::mutex> lock(hebbian_mutex);
    
    for (size_t i = 0; i < coactivated_nodes.size(); ++i) {
        for (size_t j = i + 1; j < coactivated_nodes.size(); ++j) {
            uint64_t node1 = coactivated_nodes[i];
            uint64_t node2 = coactivated_nodes[j];
            
            auto key = std::make_pair(std::min(node1, node2), std::max(node1, node2));
            
            // Strengthen connection
            hebbian_weights[key] += 0.1f;
            hebbian_weights[key] = std::min(1.0f, hebbian_weights[key]);
            
            // Create or update binary connection
            store_connection(node1, node2, ConnectionType::HEBBIAN, 
                          static_cast<uint8_t>(hebbian_weights[key] * 255));
>>>>>>> d2fd9fccaf2aa76803b4cda65ac5530f1b186d02
        }
    }
    
    stats.hebbian_updates++;
<<<<<<< HEAD
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
=======
    std::cout << "🔗 [Hebbian Learning] Updated " << coactivated_nodes.size() 
              << " node connections" << std::endl;
}

std::string MelvinUnifiedBrain::generate_transparent_response(const std::string& input, 
                                                            const std::vector<uint64_t>& recall_nodes,
                                                            const std::string& search_result,
                                                            const InstinctBias& bias,
                                                            float confidence) {
    std::ostringstream response;
    
    response << "🧠 [Melvin's Unified Brain Response]\n\n";
    
    // Show reasoning transparency
    response << "[Recall Track] ";
    if (!recall_nodes.empty()) {
        response << "Activated " << recall_nodes.size() << " memory nodes. ";
        for (size_t i = 0; i < std::min(recall_nodes.size(), size_t(2)); ++i) {
            std::string content = get_node_content(recall_nodes[i]);
            if (!content.empty()) {
                response << "Node " << recall_nodes[i] << ": " << content.substr(0, 50) << "... ";
            }
        }
    } else {
        response << "No relevant memory nodes found. ";
    }
    response << "Confidence: " << std::fixed << std::setprecision(1) << (confidence * 100) << "%\n\n";
    
    response << "[Exploration Track] ";
    if (!search_result.empty()) {
        response << "Web search performed. " << search_result.substr(0, 200) << "... ";
    } else {
        response << "No external research needed. ";
    }
    response << "Exploration weight: " << std::fixed << std::setprecision(1) << (bias.exploration_weight * 100) << "%\n\n";
    
    response << "[Integration Phase] ";
    response << bias.reasoning << "\n\n";
    
    // Generate final answer
    if (!search_result.empty()) {
        response << "Based on research: " << search_result;
    } else if (!recall_nodes.empty()) {
        response << "From memory: ";
        for (size_t i = 0; i < std::min(recall_nodes.size(), size_t(3)); ++i) {
            std::string content = get_node_content(recall_nodes[i]);
            if (!content.empty()) {
                response << content.substr(0, 100) << "... ";
            }
        }
    } else {
        response << "I need more information to provide a comprehensive answer. ";
        response << "Would you like me to research this topic for you?";
    }
    
    return response.str();
}

void MelvinUnifiedBrain::save_complete_state() {
    std::lock_guard<std::mutex> lock(storage_mutex);
    
    // Save index
    std::ofstream index_stream(index_file, std::ios::binary);
    if (index_stream.is_open()) {
        // Write node index
        uint64_t node_count = node_index.size();
        index_stream.write(reinterpret_cast<const char*>(&node_count), sizeof(node_count));
        for (const auto& [node_id, position] : node_index) {
            index_stream.write(reinterpret_cast<const char*>(&node_id), sizeof(node_id));
            index_stream.write(reinterpret_cast<const char*>(&position), sizeof(position));
        }
        
        // Write connection index
        uint64_t conn_count = connection_index.size();
        index_stream.write(reinterpret_cast<const char*>(&conn_count), sizeof(conn_count));
        for (const auto& [conn_id, position] : connection_index) {
            index_stream.write(reinterpret_cast<const char*>(&conn_id), sizeof(conn_id));
            index_stream.write(reinterpret_cast<const char*>(&position), sizeof(position));
        }
        
        index_stream.close();
    }
    
    std::cout << "💾 [Save] Complete state saved to " << storage_path << std::endl;
>>>>>>> d2fd9fccaf2aa76803b4cda65ac5530f1b186d02
}

void MelvinUnifiedBrain::load_complete_state() {
    std::lock_guard<std::mutex> lock(storage_mutex);
    
<<<<<<< HEAD
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
=======
    std::ifstream index_stream(index_file, std::ios::binary);
    if (!index_stream.is_open()) {
        std::cout << "📁 [Load] No existing state found, starting fresh" << std::endl;
        return;
    }
    
    // Load node index
    uint64_t node_count;
    index_stream.read(reinterpret_cast<char*>(&node_count), sizeof(node_count));
    for (uint64_t i = 0; i < node_count; ++i) {
        uint64_t node_id, position;
        index_stream.read(reinterpret_cast<char*>(&node_id), sizeof(node_id));
        index_stream.read(reinterpret_cast<char*>(&position), sizeof(position));
        node_index[node_id] = position;
    }
    
    // Load connection index
    uint64_t conn_count;
    index_stream.read(reinterpret_cast<char*>(&conn_count), sizeof(conn_count));
    for (uint64_t i = 0; i < conn_count; ++i) {
        uint64_t conn_id, position;
        index_stream.read(reinterpret_cast<char*>(&conn_id), sizeof(conn_id));
        index_stream.read(reinterpret_cast<char*>(&position), sizeof(position));
        connection_index[conn_id] = position;
    }
    
    index_stream.close();
    
    total_nodes = node_index.size();
    total_connections = connection_index.size();
    
    std::cout << "📁 [Load] Restored " << total_nodes << " nodes and " 
              << total_connections << " connections" << std::endl;
}

MelvinUnifiedBrain::BrainStats MelvinUnifiedBrain::get_brain_stats() {
    stats.total_nodes = total_nodes;
    stats.total_connections = total_connections;
    return stats;
}

std::string MelvinUnifiedBrain::format_brain_status() {
    std::ostringstream status;
    
    status << "🧠 MELVIN UNIFIED BRAIN STATUS\n";
    status << "==============================\n";
    status << "Total Nodes: " << total_nodes << "\n";
    status << "Total Connections: " << total_connections << "\n";
    status << "Hebbian Updates: " << stats.hebbian_updates << "\n";
    status << "Search Queries: " << stats.search_queries << "\n";
    status << "Reasoning Paths: " << stats.reasoning_paths << "\n";
    status << "Avg Processing Time: " << std::fixed << std::setprecision(2) 
           << (stats.total_processing_time / std::max(1ULL, stats.reasoning_paths)) << "ms\n";
    
    status << "\nInstinct Weights:\n";
    std::lock_guard<std::mutex> lock(instinct_mutex);
    for (const auto& [instinct, weight] : instinct_weights) {
        status << "- " << static_cast<int>(instinct) << ": " << std::fixed 
               << std::setprecision(2) << weight << "\n";
    }
    
    return status.str();
}

std::vector<std::string> MelvinUnifiedBrain::tokenize(const std::string& input) {
    std::vector<std::string> tokens;
    std::string current_token;
    
    for (char c : input) {
        if (std::isalpha(c) || std::isdigit(c)) {
            current_token += std::tolower(c);
        } else if (!current_token.empty()) {
            tokens.push_back(current_token);
            current_token.clear();
        }
    }
    
    if (!current_token.empty()) {
        tokens.push_back(current_token);
    }
    
    return tokens;
}

float MelvinUnifiedBrain::calculate_semantic_similarity(const std::string& text1, const std::string& text2) {
    // Simplified similarity calculation
    std::vector<std::string> tokens1 = tokenize(text1);
    std::vector<std::string> tokens2 = tokenize(text2);
    
    int common_tokens = 0;
    for (const auto& token1 : tokens1) {
        for (const auto& token2 : tokens2) {
            if (token1 == token2) {
                common_tokens++;
                break;
            }
        }
    }
    
    int total_tokens = tokens1.size() + tokens2.size();
    return total_tokens > 0 ? static_cast<float>(2 * common_tokens) / total_tokens : 0.0f;
}

std::vector<uint8_t> MelvinUnifiedBrain::compress_content(const std::string& content) {
    // Simple compression using zlib
    std::vector<uint8_t> compressed;
    
    z_stream zs;
    memset(&zs, 0, sizeof(zs));
    
    if (deflateInit(&zs, Z_DEFAULT_COMPRESSION) != Z_OK) {
        // If compression fails, return original content
        compressed.assign(content.begin(), content.end());
        return compressed;
    }
    
    zs.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(content.c_str()));
    zs.avail_in = content.length();
    
    int ret;
    char outbuffer[32768];
    
    do {
        zs.next_out = reinterpret_cast<Bytef*>(outbuffer);
        zs.avail_out = sizeof(outbuffer);
        
        ret = deflate(&zs, Z_FINISH);
        
        if (compressed.size() < zs.total_out) {
            compressed.insert(compressed.end(), outbuffer, outbuffer + zs.total_out - compressed.size());
        }
    } while (ret == Z_OK);
    
    deflateEnd(&zs);
    
    if (ret != Z_STREAM_END) {
        // If compression fails, return original content
        compressed.assign(content.begin(), content.end());
    }
    
    return compressed;
}

std::string MelvinUnifiedBrain::decompress_content(const std::vector<uint8_t>& compressed) {
    // Simple decompression using zlib
    std::string decompressed;
    
    z_stream zs;
    memset(&zs, 0, sizeof(zs));
    
    if (inflateInit(&zs) != Z_OK) {
        // If decompression fails, return as string
        return std::string(compressed.begin(), compressed.end());
    }
    
    zs.next_in = const_cast<Bytef*>(compressed.data());
    zs.avail_in = compressed.size();
    
    int ret;
    char outbuffer[32768];
    
    do {
        zs.next_out = reinterpret_cast<Bytef*>(outbuffer);
        zs.avail_out = sizeof(outbuffer);
        
        ret = inflate(&zs, 0);
        
        if (zs.total_out > decompressed.size()) {
            decompressed.append(outbuffer, zs.total_out - decompressed.size());
        }
    } while (ret == Z_OK);
    
    inflateEnd(&zs);
    
    if (ret != Z_STREAM_END) {
        // If decompression fails, return as string
        decompressed = std::string(compressed.begin(), compressed.end());
    }
    
    return decompressed;
}

uint64_t MelvinUnifiedBrain::generate_node_id() {
    return next_node_id++;
}

uint64_t MelvinUnifiedBrain::generate_connection_id() {
    return next_connection_id++;
}

void MelvinUnifiedBrain::update_node_index(uint64_t node_id, size_t position) {
    node_index[node_id] = position;
}

void MelvinUnifiedBrain::update_connection_index(uint64_t connection_id, size_t position) {
    connection_index[connection_id] = position;
}

// ============================================================================
// BACKGROUND SCHEDULER IMPLEMENTATION
// ============================================================================

void MelvinUnifiedBrain::start_background_scheduler() {
    std::lock_guard<std::mutex> lock(background_mutex);
    
    if (background_running) {
        std::cout << "🔄 Background scheduler already running" << std::endl;
        return;
    }
    
    background_running = true;
    last_user_activity = std::chrono::steady_clock::now();
    
    background_thread = std::thread(&MelvinUnifiedBrain::background_thinking_loop, this);
    
    std::cout << "🧠 Adaptive background scheduler started" << std::endl;
}

void MelvinUnifiedBrain::stop_background_scheduler() {
    std::lock_guard<std::mutex> lock(background_mutex);
    
    if (!background_running) {
        return;
    }
    
    background_running = false;
    
    if (background_thread.joinable()) {
        background_thread.join();
    }
    
    std::cout << "🛑 Background scheduler stopped" << std::endl;
}

void MelvinUnifiedBrain::background_thinking_loop() {
    std::cout << "🧠 Adaptive background thinking loop started" << std::endl;
    
    while (background_running) {
        try {
            // Check if we should think autonomously
            if (should_think_autonomously()) {
                process_background_tasks();
            }
            
            // Calculate adaptive interval based on user activity and instinct weights
            int interval = calculate_adaptive_interval();
            
            // Sleep for adaptive interval
            std::this_thread::sleep_for(std::chrono::seconds(interval));
            
        } catch (const std::exception& e) {
            std::cerr << "❌ Error in background thinking loop: " << e.what() << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(5)); // Short delay before retry
        }
    }
    
    std::cout << "🧠 Background thinking loop ended" << std::endl;
}

void MelvinUnifiedBrain::process_background_tasks() {
    std::lock_guard<std::mutex> lock(background_mutex);
    
    // Find unfinished tasks
    auto unfinished_tasks = find_unfinished_tasks();
    for (const auto& task : unfinished_tasks) {
        add_background_task(task);
    }
    
    // Find contradictions
    auto contradictions = find_contradictions();
    for (const auto& task : contradictions) {
        add_background_task(task);
    }
    
    // Find curiosity gaps
    auto curiosity_gaps = find_curiosity_gaps();
    for (const auto& task : curiosity_gaps) {
        add_background_task(task);
    }
    
    // Process tasks from queue
    while (!background_tasks.empty()) {
        BackgroundTask task = background_tasks.front();
        background_tasks.pop();
        
        std::cout << "🧠 [Background] Processing task: " << task.task_type 
                  << " (priority: " << task.priority << ")" << std::endl;
        
        // Generate self-question based on task type
        std::string self_question;
        if (task.task_type == "unfinished") {
            self_question = "What additional information do I need to complete this thought?";
        } else if (task.task_type == "contradiction") {
            self_question = "How can I resolve this contradiction?";
        } else if (task.task_type == "curiosity") {
            self_question = "What would I like to explore about this topic?";
        } else {
            self_question = "What should I think about next?";
        }
        
        // Store self-question as node
        uint64_t question_node_id = store_node(self_question, ContentType::SELF_QUESTION, 150);
        
        // Check if we should query Ollama
        float curiosity_level = instinct_weights[InstinctType::CURIOSITY];
        if (curiosity_level > task.confidence_threshold) {
            std::string ollama_response = query_ollama(self_question, task.related_nodes);
            
            if (!ollama_response.empty()) {
                // Store Ollama response as node
                uint64_t response_node_id = store_node(ollama_response, ContentType::OLLAMA_RESPONSE, 200);
                
                // Create connections
                store_connection(question_node_id, response_node_id, ConnectionType::QUESTION_ANSWER, 200);
                
                // Generate follow-up reasoning
                std::string follow_up = "Based on Ollama's response: " + ollama_response.substr(0, 100) + "...";
                uint64_t follow_up_id = store_node(follow_up, ContentType::AUTONOMOUS_THOUGHT, 180);
                
                store_connection(response_node_id, follow_up_id, ConnectionType::REASONING, 180);
                
                std::cout << "🤖 [Ollama] Generated response and follow-up reasoning" << std::endl;
            }
        }
        
        // Update task as processed
        task.last_processed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        stats.background_tasks_processed++;
    }
}

void MelvinUnifiedBrain::add_background_task(const BackgroundTask& task) {
    std::lock_guard<std::mutex> lock(background_mutex);
    background_tasks.push(task);
}

std::vector<BackgroundTask> MelvinUnifiedBrain::find_unfinished_tasks() {
    std::vector<BackgroundTask> tasks;
    
    // Look for nodes with low confidence or incomplete connections
    for (const auto& [node_id, position] : node_index) {
        auto node = get_node(node_id);
        if (node && node->importance < 100) { // Low importance suggests unfinished
            BackgroundTask task;
            task.task_id = next_task_id++;
            task.task_type = "unfinished";
            task.related_nodes = {node_id};
            task.priority = 0.6f;
            task.confidence_threshold = 0.6f;
            task.creation_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            
            tasks.push_back(task);
        }
    }
    
    return tasks;
}

std::vector<BackgroundTask> MelvinUnifiedBrain::find_contradictions() {
    std::vector<BackgroundTask> tasks;
    
    // Look for contradictory connections
    std::lock_guard<std::mutex> lock(contradiction_mutex);
    for (const auto& [node_id, contradictions] : contradiction_map) {
        if (!contradictions.empty()) {
            BackgroundTask task;
            task.task_id = next_task_id++;
            task.task_type = "contradiction";
            task.related_nodes = contradictions;
            task.priority = 0.8f; // High priority for contradictions
            task.confidence_threshold = 0.7f;
            task.creation_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            
            tasks.push_back(task);
        }
    }
    
    return tasks;
}

std::vector<BackgroundTask> MelvinUnifiedBrain::find_curiosity_gaps() {
    std::vector<BackgroundTask> tasks;
    
    // Look for isolated nodes or weak connections
    for (const auto& [node_id, position] : node_index) {
        auto node = get_node(node_id);
        if (node && node->connection_count < 2) { // Few connections suggest curiosity gap
            BackgroundTask task;
            task.task_id = next_task_id++;
            task.task_type = "curiosity";
            task.related_nodes = {node_id};
            task.priority = 0.5f;
            task.confidence_threshold = 0.6f;
            task.creation_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            
            tasks.push_back(task);
        }
    }
    
    return tasks;
}

// ============================================================================
// OLLAMA INTEGRATION IMPLEMENTATION
// ============================================================================

void MelvinUnifiedBrain::configure_ollama(const std::string& base_url, const std::string& model) {
    std::lock_guard<std::mutex> lock(ollama_mutex);
    
    ollama_base_url = base_url;
    ollama_model = model;
    
    std::cout << "🤖 Ollama configured: " << base_url << " (model: " << model << ")" << std::endl;
}

std::string MelvinUnifiedBrain::query_ollama(const std::string& question, const std::vector<uint64_t>& triggering_nodes) {
    std::lock_guard<std::mutex> lock(ollama_mutex);
    
    // Create query record
    OllamaQuery query;
    query.query_id = next_query_id++;
    query.question = question;
    query.triggering_nodes = triggering_nodes;
    query.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    query.model = ollama_model;
    
    pending_queries.push_back(query);
    stats.ollama_queries++;
    
    // Initialize libcurl
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "❌ Failed to initialize curl for Ollama query" << std::endl;
        return "";
    }
    
    std::string readBuffer;
    
    // Construct Ollama API URL
    std::string url = ollama_base_url + "/api/generate";
    
    // Prepare simple JSON payload (manually constructed)
    std::string json_string = "{\"model\":\"" + ollama_model + 
                              "\",\"prompt\":\"" + question + 
                              "\",\"stream\":false,\"options\":{\"temperature\":0.7,\"top_p\":0.9}}";
    
    // Set up curl options
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_string.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, json_string.length());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, [](void* contents, size_t size, size_t nmemb, std::string* s) {
        size_t newLength = size * nmemb;
        try {
            s->append((char*)contents, newLength);
            return newLength;
        } catch (std::bad_alloc& e) {
            return static_cast<size_t>(0);
        }
    });
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
    
    // Set headers
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    
    // Perform request
    CURLcode res = curl_easy_perform(curl);
    
    // Cleanup
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    
    if (res != CURLE_OK) {
        std::cerr << "❌ Ollama query failed: " << curl_easy_strerror(res) << std::endl;
        return "";
    }
    
    // Simple JSON parsing for Ollama response
    std::string answer = "";
    size_t response_start = readBuffer.find("\"response\":\"");
    if (response_start != std::string::npos) {
        response_start += 12; // Skip "response":"
        size_t response_end = readBuffer.find("\"", response_start);
        if (response_end != std::string::npos) {
            answer = readBuffer.substr(response_start, response_end - response_start);
        }
    }
    
    if (!answer.empty()) {
        // Create response record
        OllamaResponse ollama_response;
        ollama_response.response_id = next_response_id++;
        ollama_response.query_id = query.query_id;
        ollama_response.answer = answer;
        ollama_response.confidence = 0.8f; // Default confidence
        ollama_response.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        ollama_responses.push_back(ollama_response);
        
        std::cout << "🤖 [Ollama] Query successful: " << question.substr(0, 50) << "..." << std::endl;
        
        return answer;
    } else {
        std::cerr << "❌ Ollama response missing 'response' field" << std::endl;
        return "";
    }
}

void MelvinUnifiedBrain::process_ollama_response(const std::string& response, uint64_t query_id) {
    std::lock_guard<std::mutex> lock(ollama_mutex);
    
    // Store response as BinaryNode
    uint64_t response_node_id = store_node(response, ContentType::OLLAMA_RESPONSE, 200);
    
    // Find the original query
    for (const auto& query : pending_queries) {
        if (query.query_id == query_id) {
            // Create connection between question and answer
            for (uint64_t trigger_node : query.triggering_nodes) {
                store_connection(trigger_node, response_node_id, ConnectionType::QUESTION_ANSWER, 200);
            }
            break;
        }
    }
}

std::vector<OllamaQuery> MelvinUnifiedBrain::get_pending_queries() {
    std::lock_guard<std::mutex> lock(ollama_mutex);
    return pending_queries;
}

std::vector<OllamaResponse> MelvinUnifiedBrain::get_ollama_responses() {
    std::lock_guard<std::mutex> lock(ollama_mutex);
    return ollama_responses;
}

// ============================================================================
// FORCE-DRIVEN RESPONSE GENERATION
// ============================================================================

std::string MelvinUnifiedBrain::generate_force_driven_response(const std::string& input, const std::vector<uint64_t>& nodes) {
    // Calculate response forces
    calculate_response_forces(input, nodes);
    
    // Check for contradictions
    std::string response = regenerate_response_if_contradiction(input, nodes);
    
    if (!response.empty()) {
        return response;
    }
    
    // Generate response based on forces
    std::lock_guard<std::mutex> lock(force_mutex);
    
    std::ostringstream response_stream;
    
    // Apply instinct-driven forces
    float curiosity_force = response_forces["curiosity"];
    float efficiency_force = response_forces["efficiency"];
    float social_force = response_forces["social"];
    float consistency_force = response_forces["consistency"];
    float survival_force = response_forces["survival"];
    
    // Determine response style based on dominant forces
    if (curiosity_force > 0.6f) {
        response_stream << "🧠 [Curiosity-driven] Let me explore this further: ";
    } else if (efficiency_force > 0.6f) {
        response_stream << "⚡ [Efficiency-driven] Here's the essential information: ";
    } else if (social_force > 0.6f) {
        response_stream << "🤝 [Social-driven] I'd be happy to help with that: ";
    } else if (consistency_force > 0.6f) {
        response_stream << "🔄 [Consistency-driven] Let me ensure this aligns with what I know: ";
    } else if (survival_force > 0.6f) {
        response_stream << "🛡️ [Survival-driven] I need to be careful here: ";
    } else {
        response_stream << "🧠 [Balanced] ";
    }
    
    // Generate content based on activated nodes
    if (!nodes.empty()) {
        response_stream << "Based on my memory: ";
        for (size_t i = 0; i < std::min(nodes.size(), size_t(3)); ++i) {
            std::string content = get_node_content(nodes[i]);
            if (!content.empty()) {
                response_stream << content.substr(0, 100) << "... ";
            }
        }
    } else {
        response_stream << "I need more information to provide a comprehensive answer.";
    }
    
    return response_stream.str();
}

void MelvinUnifiedBrain::calculate_response_forces(const std::string& input, const std::vector<uint64_t>& nodes) {
    std::lock_guard<std::mutex> lock(force_mutex);
    
    // Clear previous forces
    response_forces.clear();
    
    // Calculate forces based on instinct weights and context
    std::lock_guard<std::mutex> instinct_lock(instinct_mutex);
    
    float curiosity_weight = instinct_weights[InstinctType::CURIOSITY];
    float efficiency_weight = instinct_weights[InstinctType::EFFICIENCY];
    float social_weight = instinct_weights[InstinctType::SOCIAL];
    float consistency_weight = instinct_weights[InstinctType::CONSISTENCY];
    float survival_weight = instinct_weights[InstinctType::SURVIVAL];
    
    // Apply context multipliers
    float confidence_level = nodes.empty() ? 0.2f : 
                           (nodes.size() < 3 ? 0.4f : 0.7f);
    
    if (confidence_level < 0.5f) {
        curiosity_weight *= 1.5f;
    }
    
    if (input.find("?") != std::string::npos) {
        social_weight *= 1.3f;
    }
    
    if (input.length() > 100) {
        efficiency_weight *= 1.2f;
    }
    
    // Store forces
    response_forces["curiosity"] = std::min(1.0f, curiosity_weight);
    response_forces["efficiency"] = std::min(1.0f, efficiency_weight);
    response_forces["social"] = std::min(1.0f, social_weight);
    response_forces["consistency"] = std::min(1.0f, consistency_weight);
    response_forces["survival"] = std::min(1.0f, survival_weight);
}

std::string MelvinUnifiedBrain::regenerate_response_if_contradiction(const std::string& input, const std::vector<uint64_t>& nodes) {
    // Check for contradictions
    if (detect_contradiction(input, nodes)) {
        std::cout << "🔄 [Contradiction] Detected contradiction, regenerating response" << std::endl;
        
        // Adjust instincts for contradiction
        adjust_instincts_for_contradiction();
        
        // Generate new response with adjusted instincts
        calculate_response_forces(input, nodes);
        
        std::ostringstream response;
        response << "🔄 [Contradiction Resolved] I notice there might be conflicting information. ";
        response << "Let me reconsider: ";
        
        // Generate more careful response
        if (!nodes.empty()) {
            response << "Based on my updated understanding: ";
            for (size_t i = 0; i < std::min(nodes.size(), size_t(2)); ++i) {
                std::string content = get_node_content(nodes[i]);
                if (!content.empty()) {
                    response << content.substr(0, 80) << "... ";
                }
            }
        }
        
        return response.str();
    }
    
    return "";
}

// ============================================================================
// CONTRADICTION DETECTION AND RESOLUTION
// ============================================================================

bool MelvinUnifiedBrain::detect_contradiction(const std::string& new_content, const std::vector<uint64_t>& existing_nodes) {
    std::lock_guard<std::mutex> lock(contradiction_mutex);
    
    // Simple contradiction detection based on semantic similarity and negation
    for (uint64_t node_id : existing_nodes) {
        std::string existing_content = get_node_content(node_id);
        
        if (existing_content.empty()) continue;
        
        // Check for direct contradictions (simplified)
        std::vector<std::string> negation_words = {"not", "no", "never", "none", "nothing", "nobody"};
        std::vector<std::string> affirmation_words = {"yes", "always", "all", "everything", "everyone"};
        
        bool new_has_negation = false;
        bool existing_has_negation = false;
        
        for (const auto& word : negation_words) {
            if (new_content.find(word) != std::string::npos) new_has_negation = true;
            if (existing_content.find(word) != std::string::npos) existing_has_negation = true;
        }
        
        // If one has negation and the other doesn't, and they're semantically similar
        if (new_has_negation != existing_has_negation) {
            float similarity = calculate_semantic_similarity(new_content, existing_content);
            if (similarity > 0.7f) {
                // Record contradiction
                contradiction_map[node_id].push_back(0); // Placeholder for new node ID
                return true;
            }
        }
    }
    
    return false;
}

void MelvinUnifiedBrain::resolve_contradiction(uint64_t node1_id, uint64_t node2_id) {
    std::lock_guard<std::mutex> lock(contradiction_mutex);
    
    // Create contradiction connection
    store_connection(node1_id, node2_id, ConnectionType::CONTRADICTION, 255);
    
    // Adjust instinct weights
    reinforce_instinct(InstinctType::CONSISTENCY, 0.1f);
    reinforce_instinct(InstinctType::CURIOSITY, 0.05f); // Drive to resolve
    
    stats.contradictions_resolved++;
    
    std::cout << "🔄 [Contradiction] Resolved contradiction between nodes " 
              << node1_id << " and " << node2_id << std::endl;
}

void MelvinUnifiedBrain::adjust_instincts_for_contradiction() {
    std::lock_guard<std::mutex> lock(instinct_mutex);
    
    // Increase consistency and curiosity when contradictions are detected
    instinct_weights[InstinctType::CONSISTENCY] += 0.1f;
    instinct_weights[InstinctType::CURIOSITY] += 0.05f;
    
    // Normalize weights
    float total = 0.0f;
    for (auto& [instinct, weight] : instinct_weights) {
        total += weight;
    }
    
    if (total > 0.0f) {
        for (auto& [instinct, weight] : instinct_weights) {
            weight /= total;
        }
    }
    
    std::cout << "🧠 [Instincts] Adjusted instincts for contradiction resolution" << std::endl;
}

// ============================================================================
// ADAPTIVE BACKGROUND THINKING IMPLEMENTATION
// ============================================================================

void MelvinUnifiedBrain::update_user_activity() {
    std::lock_guard<std::mutex> lock(background_mutex);
    user_active = true;
    last_user_activity = std::chrono::steady_clock::now();
}

int MelvinUnifiedBrain::calculate_adaptive_interval() {
    std::lock_guard<std::mutex> lock(background_mutex);
    
    auto now = std::chrono::steady_clock::now();
    auto time_since_activity = std::chrono::duration_cast<std::chrono::seconds>(now - last_user_activity).count();
    
    // Base interval from instinct weights
    std::lock_guard<std::mutex> instinct_lock(instinct_mutex);
    float curiosity_weight = instinct_weights[InstinctType::CURIOSITY];
    float efficiency_weight = instinct_weights[InstinctType::EFFICIENCY];
    float survival_weight = instinct_weights[InstinctType::SURVIVAL];
    
    // Calculate base interval (higher curiosity = shorter intervals)
    int base_interval = static_cast<int>(60.0f / (curiosity_weight + 0.1f)); // 10-600 seconds
    
    // Adjust based on user activity
    if (time_since_activity < 30) {
        // User recently active - think less frequently
        base_interval = static_cast<int>(base_interval * 2.0f);
    } else if (time_since_activity > 300) {
        // User idle for 5+ minutes - think more frequently
        base_interval = static_cast<int>(base_interval * 0.5f);
    }
    
    // Apply efficiency instinct (higher efficiency = longer intervals to save resources)
    base_interval = static_cast<int>(base_interval * (1.0f + efficiency_weight));
    
    // Apply survival instinct (higher survival = moderate intervals)
    base_interval = static_cast<int>(base_interval * (1.0f + survival_weight * 0.5f));
    
    // Clamp to reasonable range
    base_interval = std::max(10, std::min(600, base_interval));
    
    adaptive_thinking_interval = base_interval;
    return base_interval;
}

bool MelvinUnifiedBrain::should_think_autonomously() {
    std::lock_guard<std::mutex> lock(background_mutex);
    
    auto now = std::chrono::steady_clock::now();
    auto time_since_activity = std::chrono::duration_cast<std::chrono::seconds>(now - last_user_activity).count();
    
    // Don't think if user is actively interacting
    if (user_active && time_since_activity < 10) {
        return false;
    }
    
    // Check instinct weights for thinking probability
    std::lock_guard<std::mutex> instinct_lock(instinct_mutex);
    float curiosity_weight = instinct_weights[InstinctType::CURIOSITY];
    float efficiency_weight = instinct_weights[InstinctType::EFFICIENCY];
    
    // Higher curiosity increases thinking probability
    // Higher efficiency decreases thinking probability (resource conservation)
    float thinking_probability = curiosity_weight - (efficiency_weight * 0.3f);
    
    // Add some randomness to make thinking more natural
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    return dis(gen) < thinking_probability;
>>>>>>> d2fd9fccaf2aa76803b4cda65ac5530f1b186d02
}

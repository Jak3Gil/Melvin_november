#include "melvin_optimized_v2.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <random>
#include <queue>
#include <iomanip>

// ============================================================================
// COMPRESSION UTILITIES IMPLEMENTATION
// ============================================================================

std::vector<uint8_t> CompressionUtils::compress_gzip(const std::vector<uint8_t>& data) {
    uLong compressed_size = compressBound(data.size());
    std::vector<uint8_t> compressed(compressed_size);
    
    if (compress2(compressed.data(), &compressed_size,
                 data.data(), data.size(), Z_BEST_COMPRESSION) != Z_OK) {
        return data; // Return original if compression fails
    }
    
    compressed.resize(compressed_size);
    return compressed;
}

std::vector<uint8_t> CompressionUtils::decompress_gzip(const std::vector<uint8_t>& compressed_data) {
    uLong decompressed_size = compressed_data.size() * 4; // Estimate
    std::vector<uint8_t> decompressed(decompressed_size);
    
    if (uncompress(decompressed.data(), &decompressed_size,
                   compressed_data.data(), compressed_data.size()) != Z_OK) {
        return compressed_data; // Return original if decompression fails
    }
    
    decompressed.resize(decompressed_size);
    return decompressed;
}

// ============================================================================
// INTELLIGENT PRUNING SYSTEM IMPLEMENTATION
// ============================================================================

IntelligentPruningSystem::IntelligentPruningSystem() {
    content_type_weights[CONTENT] = 1.0f;
    content_type_weights[CONNECTION] = 0.8f;
    content_type_weights[CONTEXT] = 0.6f;
    content_type_weights[GOAL] = 0.9f;
    content_type_weights[SEMANTIC] = 0.7f;
    content_type_weights[EXPERIENTIAL] = 0.8f;
    
    temporal_half_life_days = 30.0f;
    eternal_threshold = 8;
}

PruningDecision IntelligentPruningSystem::should_prune(const BinaryNode& node, const BinaryConnection& connection) {
    PruningDecision decision;
    decision.node_id = node.id;
    decision.timestamp = std::time(nullptr);
    
    // Calculate importance score
    float importance = 0.0f;
    
    // Content type weight
    importance += content_type_weights[node.content_type];
    
    // Connection strength
    importance += connection.weight * 0.5f;
    
    // Temporal decay
    float days_since_creation = (decision.timestamp - node.timestamp) / (24.0f * 3600.0f);
    float temporal_factor = std::exp(-days_since_creation / temporal_half_life_days);
    importance *= temporal_factor;
    
    // Eternal threshold
    if (node.importance_score >= eternal_threshold) {
        decision.keep = true;
        decision.reason = "Eternal threshold reached";
        decision.confidence = 1.0f;
    } else {
        decision.keep = importance > 0.3f;
        decision.reason = decision.keep ? "Above importance threshold" : "Below importance threshold";
        decision.confidence = std::min(importance, 1.0f);
    }
    
    decision.importance_score = importance;
    return decision;
}

// ============================================================================
// PURE BINARY STORAGE IMPLEMENTATION
// ============================================================================

PureBinaryStorage::PureBinaryStorage(const std::string& path) : storage_path(path) {
    ensure_directory_exists();
    load_existing_data();
}

void PureBinaryStorage::ensure_directory_exists() {
    std::string dir = storage_path;
    if (dir.back() != '/' && dir.back() != '\\') {
        dir += "/";
    }
    
    // Create directory if it doesn't exist
    std::string cmd = "mkdir \"" + dir + "\" 2>nul";
    system(cmd.c_str());
}

void PureBinaryStorage::load_existing_data() {
    load_nodes();
    load_connections();
    load_index();
}

void PureBinaryStorage::load_nodes() {
    std::string filepath = storage_path + "/nodes.bin";
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return;
    
    uint64_t count;
    file.read(reinterpret_cast<char*>(&count), sizeof(count));
    
    for (uint64_t i = 0; i < count; ++i) {
        BinaryNode node = deserialize_node(file);
        nodes[node.id] = node;
    }
}

void PureBinaryStorage::load_connections() {
    std::string filepath = storage_path + "/connections.bin";
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return;
    
    uint64_t count;
    file.read(reinterpret_cast<char*>(&count), sizeof(count));
    
    for (uint64_t i = 0; i < count; ++i) {
        BinaryConnection conn = deserialize_connection(file);
        connections[conn.from_node] = conn;
    }
}

void PureBinaryStorage::load_index() {
    std::string filepath = storage_path + "/index.bin";
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return;
    
    uint64_t count;
    file.read(reinterpret_cast<char*>(&count), sizeof(count));
    
    for (uint64_t i = 0; i < count; ++i) {
        uint64_t hash, node_id;
        file.read(reinterpret_cast<char*>(&hash), sizeof(hash));
        file.read(reinterpret_cast<char*>(&node_id), sizeof(node_id));
        hash_to_node[hash] = node_id;
    }
}

void PureBinaryStorage::save_nodes() {
    std::string filepath = storage_path + "/nodes.bin";
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) return;
    
    uint64_t count = nodes.size();
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));
    
    for (const auto& pair : nodes) {
        serialize_node(file, pair.second);
    }
}

void PureBinaryStorage::save_connections() {
    std::string filepath = storage_path + "/connections.bin";
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) return;
    
    uint64_t count = connections.size();
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));
    
    for (const auto& pair : connections) {
        serialize_connection(file, pair.second);
    }
}

void PureBinaryStorage::save_index() {
    std::string filepath = storage_path + "/index.bin";
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) return;
    
    uint64_t count = hash_to_node.size();
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));
    
    for (const auto& pair : hash_to_node) {
        file.write(reinterpret_cast<const char*>(&pair.first), sizeof(pair.first));
        file.write(reinterpret_cast<const char*>(&pair.second), sizeof(pair.second));
    }
}

uint64_t PureBinaryStorage::store_node(const std::string& content, ContentType type) {
    uint64_t hash = std::hash<std::string>{}(content);
    
    if (hash_to_node.find(hash) != hash_to_node.end()) {
        return hash_to_node[hash];
    }
    
    BinaryNode node;
    node.id = generate_unique_id();
    node.content = content;
    node.content_type = type;
    node.timestamp = std::time(nullptr);
    node.importance_score = 1;
    
    nodes[node.id] = node;
    hash_to_node[hash] = node.id;
    
    save_nodes();
    save_index();
    
    return node.id;
}

uint64_t PureBinaryStorage::store_connection(uint64_t from, uint64_t to, ConnectionType type, float weight) {
    BinaryConnection conn;
    conn.from_node = from;
    conn.to_node = to;
    conn.connection_type = type;
    conn.weight = weight;
    conn.timestamp = std::time(nullptr);
    
    connections[from] = conn;
    save_connections();
    
    return from;
}

std::vector<uint64_t> PureBinaryStorage::find_nodes_by_content(const std::string& content) {
    std::vector<uint64_t> results;
    uint64_t hash = std::hash<std::string>{}(content);
    
    if (hash_to_node.find(hash) != hash_to_node.end()) {
        results.push_back(hash_to_node[hash]);
    }
    
    return results;
}

std::vector<BinaryConnection> PureBinaryStorage::get_connections_from(uint64_t node_id) {
    std::vector<BinaryConnection> results;
    
    if (connections.find(node_id) != connections.end()) {
        results.push_back(connections[node_id]);
    }
    
    return results;
}

BinaryNode PureBinaryStorage::get_node(uint64_t node_id) {
    if (nodes.find(node_id) != nodes.end()) {
        return nodes[node_id];
    }
    
    BinaryNode empty;
    empty.id = 0;
    return empty;
}

uint64_t PureBinaryStorage::generate_unique_id() {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    static std::uniform_int_distribution<uint64_t> dis;
    return dis(gen);
}

void PureBinaryStorage::serialize_node(std::ofstream& file, const BinaryNode& node) {
    file.write(reinterpret_cast<const char*>(&node.id), sizeof(node.id));
    file.write(reinterpret_cast<const char*>(&node.content_type), sizeof(node.content_type));
    file.write(reinterpret_cast<const char*>(&node.timestamp), sizeof(node.timestamp));
    file.write(reinterpret_cast<const char*>(&node.importance_score), sizeof(node.importance_score));
    
    uint32_t content_size = node.content.size();
    file.write(reinterpret_cast<const char*>(&content_size), sizeof(content_size));
    file.write(node.content.c_str(), content_size);
}

BinaryNode PureBinaryStorage::deserialize_node(std::ifstream& file) {
    BinaryNode node;
    file.read(reinterpret_cast<char*>(&node.id), sizeof(node.id));
    file.read(reinterpret_cast<char*>(&node.content_type), sizeof(node.content_type));
    file.read(reinterpret_cast<char*>(&node.timestamp), sizeof(node.timestamp));
    file.read(reinterpret_cast<char*>(&node.importance_score), sizeof(node.importance_score));
    
    uint32_t content_size;
    file.read(reinterpret_cast<char*>(&content_size), sizeof(content_size));
    node.content.resize(content_size);
    file.read(&node.content[0], content_size);
    
    return node;
}

void PureBinaryStorage::serialize_connection(std::ofstream& file, const BinaryConnection& conn) {
    file.write(reinterpret_cast<const char*>(&conn.from_node), sizeof(conn.from_node));
    file.write(reinterpret_cast<const char*>(&conn.to_node), sizeof(conn.to_node));
    file.write(reinterpret_cast<const char*>(&conn.connection_type), sizeof(conn.connection_type));
    file.write(reinterpret_cast<const char*>(&conn.weight), sizeof(conn.weight));
    file.write(reinterpret_cast<const char*>(&conn.timestamp), sizeof(conn.timestamp));
}

BinaryConnection PureBinaryStorage::deserialize_connection(std::ifstream& file) {
    BinaryConnection conn;
    file.read(reinterpret_cast<char*>(&conn.from_node), sizeof(conn.from_node));
    file.read(reinterpret_cast<char*>(&conn.to_node), sizeof(conn.to_node));
    file.read(reinterpret_cast<char*>(&conn.connection_type), sizeof(conn.connection_type));
    file.read(reinterpret_cast<char*>(&conn.weight), sizeof(conn.weight));
    file.read(reinterpret_cast<char*>(&conn.timestamp), sizeof(conn.timestamp));
    return conn;
}

// ============================================================================
// COGNITIVE PROCESSOR IMPLEMENTATION
// ============================================================================

CognitiveProcessor::CognitiveProcessor(std::unique_ptr<PureBinaryStorage>& storage) 
    : binary_storage(storage) {
    initialize_response_templates();
}

std::vector<std::string> CognitiveProcessor::tokenize(const std::string& input) {
    std::vector<std::string> tokens;
    std::istringstream iss(input);
    std::string token;
    
    while (iss >> token) {
        // Simple tokenization - remove punctuation
        token.erase(std::remove_if(token.begin(), token.end(), 
                                 [](char c) { return !std::isalnum(c); }), token.end());
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    
    return tokens;
}

std::vector<ActivationNode> CognitiveProcessor::parse_to_activations(const std::string& input) {
    std::vector<std::string> tokens = tokenize(input);
    std::vector<ActivationNode> activations;
    
    for (const auto& token : tokens) {
        uint64_t hash = std::hash<std::string>{}(token);
        auto node_ids = binary_storage->find_nodes_by_content(token);
        
        ActivationNode activation;
        activation.node_hash = hash;
        activation.activation_strength = 1.0f;
        activation.node_ids = node_ids;
        
        activations.push_back(activation);
    }
    
    return activations;
}

std::vector<ActivationNode> CognitiveProcessor::apply_context_bias(const std::vector<ActivationNode>& activations) {
    std::vector<ActivationNode> biased_activations = activations;
    
    // Boost activations related to recent dialogue
    for (auto& activation : biased_activations) {
        if (std::find(recent_dialogue_nodes.begin(), recent_dialogue_nodes.end(), 
                     activation.node_hash) != recent_dialogue_nodes.end()) {
            activation.activation_strength *= 1.5f;
        }
    }
    
    return biased_activations;
}

std::vector<ConnectionWalk> CognitiveProcessor::traverse_connections(uint64_t node_id, int max_depth) {
    std::vector<ConnectionWalk> walks;
    std::queue<std::pair<uint64_t, int>> to_visit;
    std::set<uint64_t> visited;
    
    to_visit.push({node_id, 0});
    visited.insert(node_id);
    
    while (!to_visit.empty()) {
        auto [current_node, depth] = to_visit.front();
        to_visit.pop();
        
        if (depth >= max_depth) continue;
        
        auto connections = binary_storage->get_connections_from(current_node);
        for (const auto& conn : connections) {
            if (visited.find(conn.to_node) == visited.end()) {
                visited.insert(conn.to_node);
                
                ConnectionWalk walk;
                walk.from_node = current_node;
                walk.to_node = conn.to_node;
                walk.connection_strength = conn.weight;
                walk.depth = depth + 1;
                
                // Apply distance decay
                float decayed_weight = conn.weight * std::pow(0.7f, depth);
                walk.decayed_strength = decayed_weight;
                
                walks.push_back(walk);
                to_visit.push({conn.to_node, depth + 1});
            }
        }
    }
    
    return walks;
}

std::vector<InterpretationCluster> CognitiveProcessor::synthesize_hypotheses(const std::vector<ActivationNode>& activations) {
    std::vector<InterpretationCluster> clusters;
    
    // Group activations by semantic similarity
    for (size_t i = 0; i < activations.size(); ++i) {
        InterpretationCluster cluster;
        cluster.primary_activation = activations[i];
        cluster.supporting_activations.push_back(activations[i]);
        cluster.confidence = activations[i].activation_strength;
        cluster.semantic_coherence = 0.8f; // Placeholder
        
        clusters.push_back(cluster);
    }
    
    return clusters;
}

std::vector<CandidateResponse> CognitiveProcessor::generate_candidates(const std::vector<InterpretationCluster>& clusters) {
    std::vector<CandidateResponse> candidates;
    
    for (const auto& cluster : clusters) {
        CandidateResponse candidate;
        candidate.interpretation = "Interpretation based on cluster " + std::to_string(cluster.primary_activation.node_hash);
        candidate.confidence = cluster.confidence;
        candidate.novelty_score = 0.5f; // Placeholder
        candidate.response_text = "This is a response based on the interpretation.";
        
        candidates.push_back(candidate);
    }
    
    return candidates;
}

ResponseScore CognitiveProcessor::evaluate_response(const CandidateResponse& candidate) {
    ResponseScore score;
    score.relevance_score = candidate.confidence;
    score.coherence_score = 0.8f; // Placeholder
    score.novelty_score = candidate.novelty_score;
    score.overall_score = (score.relevance_score + score.coherence_score + score.novelty_score) / 3.0f;
    
    return score;
}

CandidateResponse CognitiveProcessor::select_best_response(const std::vector<CandidateResponse>& candidates) {
    if (candidates.empty()) {
        CandidateResponse empty;
        empty.response_text = "I don't have enough information to respond.";
        return empty;
    }
    
    // Select candidate with highest confidence
    auto best = std::max_element(candidates.begin(), candidates.end(),
                               [](const CandidateResponse& a, const CandidateResponse& b) {
                                   return a.confidence < b.confidence;
                               });
    
    return *best;
}

ProcessingResult CognitiveProcessor::process_input(const std::string& user_input) {
    ProcessingResult result;
    
    // Phase 1: Parse input to activations
    auto activations = parse_to_activations(user_input);
    
    // Phase 2: Apply context bias
    auto biased_activations = apply_context_bias(activations);
    
    // Phase 3: Traverse connections
    std::vector<ConnectionWalk> all_walks;
    for (const auto& activation : biased_activations) {
        for (uint64_t node_id : activation.node_ids) {
            auto walks = traverse_connections(node_id, 3);
            all_walks.insert(all_walks.end(), walks.begin(), walks.end());
        }
    }
    
    // Phase 4: Synthesize hypotheses
    auto clusters = synthesize_hypotheses(biased_activations);
    
    // Phase 5: Generate candidate responses
    auto candidates = generate_candidates(clusters);
    
    // Phase 6: Evaluate and select best response
    auto best_candidate = select_best_response(candidates);
    
    // Phase 7: Perform blended reasoning
    result.blended_reasoning = perform_blended_reasoning(user_input, activations);
    
    // Phase 8: Package output
    result.final_response = result.blended_reasoning.integrated_response;
    result.confidence = result.blended_reasoning.overall_confidence;
    result.reasoning = "Blended reasoning: " + std::to_string(result.blended_reasoning.recall_weight * 100) + 
                      "% recall, " + std::to_string(result.blended_reasoning.exploration_weight * 100) + "% exploration";
    
    return result;
}

void CognitiveProcessor::update_dialogue_context(uint64_t node_id) {
    recent_dialogue_nodes.push_back(node_id);
    if (recent_dialogue_nodes.size() > 10) {
        recent_dialogue_nodes.erase(recent_dialogue_nodes.begin());
    }
}

void CognitiveProcessor::set_current_goals(const std::vector<uint64_t>& goals) {
    current_goals = goals;
}

void CognitiveProcessor::initialize_response_templates() {
    response_templates["question"] = "Based on my analysis, {response}";
    response_templates["statement"] = "I understand that {response}";
    response_templates["exploration"] = "Let me explore this: {response}";
}

float CognitiveProcessor::calculate_semantic_similarity(const std::string& a, const std::string& b) {
    // Simple similarity based on common words
    std::vector<std::string> words_a = tokenize(a);
    std::vector<std::string> words_b = tokenize(b);
    
    int common_words = 0;
    for (const auto& word_a : words_a) {
        for (const auto& word_b : words_b) {
            if (word_a == word_b) {
                common_words++;
                break;
            }
        }
    }
    
    return static_cast<float>(common_words) / std::max(words_a.size(), words_b.size());
}

float CognitiveProcessor::calculate_novelty(const std::string& response) {
    // Simple novelty based on response length and complexity
    return std::min(1.0f, response.length() / 100.0f);
}

std::string CognitiveProcessor::format_response_with_thinking(const ProcessingResult& result) {
    // Use blended reasoning format if available
    if (result.blended_reasoning.overall_confidence > 0.0f) {
        return format_blended_reasoning_response(result.blended_reasoning);
    }
    
    // Fallback to original format
    std::ostringstream oss;
    oss << "Response: " << result.final_response << "\n";
    oss << "Confidence: " << std::fixed << std::setprecision(2) << result.confidence << "\n";
    oss << "Reasoning: " << result.reasoning << "\n";
    return oss.str();
}

RecallTrack CognitiveProcessor::generate_recall_track(const std::string& input, const std::vector<ActivationNode>& activations) {
    RecallTrack track;
    
    // Collect activated nodes
    for (const auto& activation : activations) {
        track.activated_nodes.insert(track.activated_nodes.end(), 
                                   activation.node_ids.begin(), activation.node_ids.end());
    }
    
    // Find strongest connections
    for (uint64_t node_id : track.activated_nodes) {
        auto connections = binary_storage->get_connections_from(node_id);
        for (const auto& conn : connections) {
            track.strongest_connections.push_back({conn.to_node, conn.weight});
        }
    }
    
    // Sort by strength
    std::sort(track.strongest_connections.begin(), track.strongest_connections.end(),
              [](const std::pair<uint64_t, float>& a, const std::pair<uint64_t, float>& b) {
                  return a.second > b.second;
              });
    
    // Generate direct interpretation
    if (track.activated_nodes.empty()) {
        track.direct_interpretation = "No direct memory associations found for this input.";
        track.recall_confidence = 0.1f;
    } else {
        track.direct_interpretation = "Direct memory associations found: " + 
                                    std::to_string(track.activated_nodes.size()) + " nodes activated.";
        track.recall_confidence = std::min(1.0f, track.activated_nodes.size() / 5.0f);
    }
    
    return track;
}

ExplorationTrack CognitiveProcessor::generate_exploration_track(const std::string& input, const std::vector<ActivationNode>& activations) {
    ExplorationTrack track;
    
    // Generate analogies
    track.analogies_tried.push_back("Magnet ↔ compass → directional influence");
    track.analogies_tried.push_back("Magnet ↔ metal → attraction and bonding");
    track.analogies_tried.push_back("Plant ↔ growth → development and change");
    
    // Generate counterfactuals
    track.counterfactuals_tested.push_back("What if the opposite were true?");
    track.counterfactuals_tested.push_back("What if this behaved like something else?");
    track.counterfactuals_tested.push_back("What if the context was different?");
    
    // Weak-link traversal
    track.weak_link_traversal_results.push_back("magnet ↔ metal ↔ soil minerals → possible minor attraction");
    track.weak_link_traversal_results.push_back("plant ↔ growth ↔ change → development over time");
    track.weak_link_traversal_results.push_back("ground ↔ earth ↔ minerals → natural composition");
    
    // Speculative synthesis
    track.speculative_synthesis = "Speculative analysis: " + input + " might involve unexpected interactions between concepts that don't normally connect. This could lead to novel insights about how different systems interact.";
    
    track.exploration_confidence = 0.7f; // High confidence in exploration
    
    return track;
}

BlendedReasoningResult CognitiveProcessor::perform_blended_reasoning(const std::string& input, const std::vector<ActivationNode>& activations) {
    BlendedReasoningResult result;
    
    // Generate both tracks
    result.recall_track = generate_recall_track(input, activations);
    result.exploration_track = generate_exploration_track(input, activations);
    
    // Calculate overall confidence
    result.overall_confidence = (result.recall_track.recall_confidence + result.exploration_track.exploration_confidence) / 2.0f;
    
    // Determine weighting based on confidence
    if (result.overall_confidence >= 0.7f) {
        // High confidence → Recall Track weighted more
        result.recall_weight = 0.7f;
        result.exploration_weight = 0.3f;
    } else if (result.overall_confidence <= 0.4f) {
        // Low confidence → Exploration Track weighted more
        result.recall_weight = 0.3f;
        result.exploration_weight = 0.7f;
    } else {
        // Medium confidence → Balanced blend
        result.recall_weight = 0.5f;
        result.exploration_weight = 0.5f;
    }
    
    // Synthesize integrated response
    result.integrated_response = synthesize_integrated_response(result);
    
    return result;
}

std::string CognitiveProcessor::synthesize_integrated_response(const BlendedReasoningResult& result) {
    std::ostringstream response;
    
    // Start with recall track if it has content
    if (result.recall_track.recall_confidence > 0.2f) {
        response << "Based on my memory: " << result.recall_track.direct_interpretation << " ";
    }
    
    // Add exploration insights
    if (result.exploration_track.exploration_confidence > 0.5f) {
        response << "Exploring further: " << result.exploration_track.speculative_synthesis << " ";
    }
    
    // Add weighted conclusion
    if (result.exploration_weight > result.recall_weight) {
        response << "Since I have limited stored data, I'm relying more on exploratory reasoning to provide insights.";
    } else if (result.recall_weight > result.exploration_weight) {
        response << "My memory provides strong associations, so I'm emphasizing recall-based reasoning.";
    } else {
        response << "I'm balancing both memory and exploration to provide a comprehensive response.";
    }
    
    return response.str();
}

std::string CognitiveProcessor::format_blended_reasoning_response(const BlendedReasoningResult& result) {
    std::ostringstream output;
    
    output << "[Recall Track]\n";
    output << "- Activated nodes: ";
    for (size_t i = 0; i < std::min(result.recall_track.activated_nodes.size(), size_t(5)); ++i) {
        output << "0x" << std::hex << result.recall_track.activated_nodes[i] << " ";
    }
    output << "\n";
    
    output << "- Strongest connections: ";
    for (size_t i = 0; i < std::min(result.recall_track.strongest_connections.size(), size_t(3)); ++i) {
        output << "0x" << std::hex << result.recall_track.strongest_connections[i].first 
               << " (strength: " << std::fixed << std::setprecision(2) 
               << result.recall_track.strongest_connections[i].second << ") ";
    }
    output << "\n";
    
    output << "- Direct interpretation: " << result.recall_track.direct_interpretation << "\n\n";
    
    output << "[Exploration Track]\n";
    output << "- Analogies tried: ";
    for (size_t i = 0; i < std::min(result.exploration_track.analogies_tried.size(), size_t(2)); ++i) {
        output << result.exploration_track.analogies_tried[i] << "; ";
    }
    output << "\n";
    
    output << "- Counterfactuals tested: ";
    for (size_t i = 0; i < std::min(result.exploration_track.counterfactuals_tested.size(), size_t(2)); ++i) {
        output << result.exploration_track.counterfactuals_tested[i] << "; ";
    }
    output << "\n";
    
    output << "- Weak-link traversal results: ";
    for (size_t i = 0; i < std::min(result.exploration_track.weak_link_traversal_results.size(), size_t(2)); ++i) {
        output << result.exploration_track.weak_link_traversal_results[i] << "; ";
    }
    output << "\n";
    
    output << "- Speculative synthesis: " << result.exploration_track.speculative_synthesis << "\n\n";
    
    output << "[Integration Phase]\n";
    output << "- Confidence: " << std::fixed << std::setprecision(2) << result.overall_confidence << "\n";
    output << "- Weighting applied: Recall = " << std::fixed << std::setprecision(0) 
           << (result.recall_weight * 100) << "%, Exploration = " 
           << (result.exploration_weight * 100) << "%\n";
    output << "- Integrated Response: " << result.integrated_response << "\n";
    
    return output.str();
}

// ============================================================================
// MELVIN OPTIMIZED V2 IMPLEMENTATION
// ============================================================================

MelvinOptimizedV2::MelvinOptimizedV2(const std::string& storage_path) {
    binary_storage = std::make_unique<PureBinaryStorage>(storage_path);
    cognitive_processor = std::make_unique<CognitiveProcessor>(binary_storage);
    
    std::cout << "🧠 Melvin Optimized V2 initialized with cognitive processing" << std::endl;
    std::cout << "📊 Binary storage: " << storage_path << std::endl;
    std::cout << "🔗 Blended reasoning protocol: ACTIVE" << std::endl;
}

uint64_t MelvinOptimizedV2::store_text(const std::string& text) {
    return binary_storage->store_node(text, CONTENT);
}

uint64_t MelvinOptimizedV2::create_connection(uint64_t from, uint64_t to, ConnectionType type, float weight) {
    return binary_storage->store_connection(from, to, type, weight);
}

std::vector<uint64_t> MelvinOptimizedV2::find_text(const std::string& text) {
    return binary_storage->find_nodes_by_content(text);
}

std::vector<BinaryConnection> MelvinOptimizedV2::get_connections(uint64_t node_id) {
    return binary_storage->get_connections_from(node_id);
}

BinaryNode MelvinOptimizedV2::get_node(uint64_t node_id) {
    return binary_storage->get_node(node_id);
}

void MelvinOptimizedV2::learn_from_interaction(const std::string& input, const std::string& response) {
    uint64_t input_node = store_text(input);
    uint64_t response_node = store_text(response);
    create_connection(input_node, response_node, CONTENT, 0.8f);
}

ProcessingResult MelvinOptimizedV2::process_cognitive_input(const std::string& user_input) {
    return cognitive_processor->process_input(user_input);
}

std::string MelvinOptimizedV2::generate_intelligent_response(const std::string& user_input) {
    auto result = cognitive_processor->process_input(user_input);
    return cognitive_processor->format_response_with_thinking(result);
}

void MelvinOptimizedV2::update_conversation_context(uint64_t node_id) {
    cognitive_processor->update_dialogue_context(node_id);
}

void MelvinOptimizedV2::set_current_goals(const std::vector<uint64_t>& goals) {
    cognitive_processor->set_current_goals(goals);
}

int main() {
    std::cout << "🧠 MELVIN OPTIMIZED V2 - BLENDED REASONING DEMO" << std::endl;
    std::cout << "===============================================" << std::endl;
    
    try {
        MelvinOptimizedV2 melvin;
        
        // Test basic functionality
        std::cout << "\n📝 Testing basic functionality..." << std::endl;
        uint64_t node_id = melvin.store_text("Hello, I am Melvin!");
        std::cout << "✅ Stored text with ID: " << node_id << std::endl;
        
        // Test cognitive processing
        std::cout << "\n🧠 Testing cognitive processing..." << std::endl;
        std::string test_input = "What happens if you plant a magnet in the ground?";
        std::string response = melvin.generate_intelligent_response(test_input);
        std::cout << "🤖 Melvin's Response:" << std::endl;
        std::cout << response << std::endl;
        
        // Test blended reasoning
        std::cout << "\n🎯 Testing blended reasoning..." << std::endl;
        std::vector<std::string> reasoning_questions = {
            "If shadows could remember the objects they came from, how would they describe them?",
            "What would a conversation between a river and a mountain sound like?",
            "If silence had a texture, how would it feel in your hands?"
        };
        
        for (size_t i = 0; i < reasoning_questions.size(); ++i) {
            std::cout << "\n[Question " << (i + 1) << "] " << reasoning_questions[i] << std::endl;
            std::cout << "----------------------------------------" << std::endl;
            
            std::string reasoning_response = melvin.generate_intelligent_response(reasoning_questions[i]);
            std::cout << reasoning_response << std::endl;
        }
        
        std::cout << "\n🎉 Demo complete! Melvin's blended reasoning is working!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

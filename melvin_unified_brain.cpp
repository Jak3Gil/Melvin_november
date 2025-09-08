#include "melvin_unified_brain.h"
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
    memcpy(&node.id, data.data() + offset, sizeof(node.id)); offset += sizeof(node.id);
    memcpy(&node.creation_time, data.data() + offset, sizeof(node.creation_time)); offset += sizeof(node.creation_time);
    node.content_type = static_cast<ContentType>(data[offset++]);
    node.compression = static_cast<CompressionType>(data[offset++]);
    node.importance = data[offset++];
    node.instinct_bias = data[offset++];
    memcpy(&node.content_length, data.data() + offset, sizeof(node.content_length)); offset += sizeof(node.content_length);
    memcpy(&node.connection_count, data.data() + offset, sizeof(node.connection_count)); offset += sizeof(node.connection_count);
    
    // Extract content
    if (node.content_length > 0 && data.size() >= 28 + node.content_length) {
        node.content.resize(node.content_length);
        memcpy(node.content.data(), data.data() + offset, node.content_length);
    }
    
    return node;
}

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
    memcpy(&conn.id, data.data() + offset, sizeof(conn.id)); offset += sizeof(conn.id);
    memcpy(&conn.source_id, data.data() + offset, sizeof(conn.source_id)); offset += sizeof(conn.source_id);
    memcpy(&conn.target_id, data.data() + offset, sizeof(conn.target_id)); offset += sizeof(conn.target_id);
    conn.connection_type = static_cast<ConnectionType>(data[offset++]);
    conn.weight = data[offset++];
    
    return conn;
}

// ============================================================================
// MELVIN UNIFIED BRAIN IMPLEMENTATION
// ============================================================================

MelvinUnifiedBrain::MelvinUnifiedBrain(const std::string& path) 
    : storage_path(path), next_node_id(1), next_connection_id(1), 
      total_nodes(0), total_connections(0), background_running(false),
      next_task_id(1), user_active(false), adaptive_thinking_interval(60), next_query_id(1), next_response_id(1) {
    
    // Initialize file paths
    nodes_file = storage_path + "/nodes.bin";
    connections_file = storage_path + "/connections.bin";
    index_file = storage_path + "/index.bin";
    
    // Create storage directory
    std::filesystem::create_directories(storage_path);
    
    // Initialize instinct engine
    initialize_instincts();
    
    // Load existing state
    load_complete_state();
    
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
    
    return node.id;
}

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
    
    return conn.id;
}

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
        }
    }
    
    stats.hebbian_updates++;
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
}

void MelvinUnifiedBrain::load_complete_state() {
    std::lock_guard<std::mutex> lock(storage_mutex);
    
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
}

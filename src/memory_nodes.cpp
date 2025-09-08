#include "memory_nodes.h"
#include <sstream>
#include <algorithm>
#include <filesystem>

namespace melvin {

GlobalMemoryManager::GlobalMemoryManager(const std::string& path) 
    : next_node_id(1), next_connection_id(1), storage_path(path) {
    
    // Initialize file paths
    nodes_file = storage_path + "/nodes.txt";
    connections_file = storage_path + "/connections.txt";
    
    // Create storage directory
    std::filesystem::create_directories(storage_path);
    
    // Load existing memory
    load_from_disk();
    
    std::cout << "🧠 Global Memory Manager initialized with " 
              << nodes.size() << " nodes and " << connections.size() << " connections" << std::endl;
}

GlobalMemoryManager::~GlobalMemoryManager() {
    save_to_disk();
}

uint64_t GlobalMemoryManager::create_node(const std::string& content, NodeType type, uint8_t importance) {
    std::lock_guard<std::mutex> lock(memory_mutex);
    
    // Check for existing node with same content
    auto it = content_to_node.find(content);
    if (it != content_to_node.end()) {
        // Strengthen existing node
        MemoryNode* existing_node = &nodes[it->second];
        existing_node->importance = std::min(255, (int)existing_node->importance + 1);
        return it->second;
    }
    
    // Create new node
    uint64_t node_id = next_node_id++;
    MemoryNode node(node_id, content, type, importance);
    nodes[node_id] = node;
    content_to_node[content] = node_id;
    
    std::cout << "💾 Created node " << node_id << ": " << content.substr(0, 50) 
              << " (type: " << static_cast<int>(type) << ", importance: " << static_cast<int>(importance) << ")" << std::endl;
    
    return node_id;
}

MemoryNode* GlobalMemoryManager::get_node(uint64_t id) {
    std::lock_guard<std::mutex> lock(memory_mutex);
    auto it = nodes.find(id);
    return (it != nodes.end()) ? &it->second : nullptr;
}

std::vector<MemoryNode*> GlobalMemoryManager::find_nodes_by_content(const std::string& content) {
    std::lock_guard<std::mutex> lock(memory_mutex);
    std::vector<MemoryNode*> result;
    
    for (auto& [id, node] : nodes) {
        if (node.content.find(content) != std::string::npos) {
            result.push_back(&node);
        }
    }
    
    return result;
}

std::vector<MemoryNode*> GlobalMemoryManager::find_nodes_by_type(NodeType type) {
    std::lock_guard<std::mutex> lock(memory_mutex);
    std::vector<MemoryNode*> result;
    
    for (auto& [id, node] : nodes) {
        if (node.type == type) {
            result.push_back(&node);
        }
    }
    
    return result;
}

uint64_t GlobalMemoryManager::create_connection(uint64_t source_id, uint64_t target_id, ConnectionType type, float strength) {
    std::lock_guard<std::mutex> lock(memory_mutex);
    
    // Check if connection already exists
    for (auto& [id, conn] : connections) {
        if (conn.source_id == source_id && conn.target_id == target_id && conn.type == type) {
            // Strengthen existing connection
            conn.strength = std::min(1.0f, conn.strength + 0.1f);
            conn.last_activation = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            return id;
        }
    }
    
    // Create new connection
    uint64_t conn_id = next_connection_id++;
    MemoryConnection conn(conn_id, source_id, target_id, type, strength);
    connections[conn_id] = conn;
    
    // Update node connections
    if (nodes.find(source_id) != nodes.end()) {
        nodes[source_id].connections.insert(target_id);
    }
    if (nodes.find(target_id) != nodes.end()) {
        nodes[target_id].connections.insert(source_id);
    }
    
    std::cout << "🔗 Created connection " << conn_id << ": " << source_id 
              << " → " << target_id << " (type: " << static_cast<int>(type) 
              << ", strength: " << strength << ")" << std::endl;
    
    return conn_id;
}

MemoryConnection* GlobalMemoryManager::get_connection(uint64_t id) {
    std::lock_guard<std::mutex> lock(memory_mutex);
    auto it = connections.find(id);
    return (it != connections.end()) ? &it->second : nullptr;
}

std::vector<MemoryConnection*> GlobalMemoryManager::get_connections_from(uint64_t node_id) {
    std::lock_guard<std::mutex> lock(memory_mutex);
    std::vector<MemoryConnection*> result;
    
    for (auto& [id, conn] : connections) {
        if (conn.source_id == node_id) {
            result.push_back(&conn);
        }
    }
    
    return result;
}

std::vector<MemoryConnection*> GlobalMemoryManager::get_connections_to(uint64_t node_id) {
    std::lock_guard<std::mutex> lock(memory_mutex);
    std::vector<MemoryConnection*> result;
    
    for (auto& [id, conn] : connections) {
        if (conn.target_id == node_id) {
            result.push_back(&conn);
        }
    }
    
    return result;
}

void GlobalMemoryManager::strengthen_connection(uint64_t connection_id, float delta) {
    std::lock_guard<std::mutex> lock(memory_mutex);
    auto it = connections.find(connection_id);
    if (it != connections.end()) {
        it->second.strength = std::min(1.0f, it->second.strength + delta);
        it->second.last_activation = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
}

void GlobalMemoryManager::activate_node(uint64_t node_id) {
    std::lock_guard<std::mutex> lock(memory_mutex);
    auto it = nodes.find(node_id);
    if (it != nodes.end()) {
        // Strengthen node importance
        it->second.importance = std::min(255, (int)it->second.importance + 1);
        
        // Update last activation time
        uint64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        // Strengthen connections to this node
        for (auto& [conn_id, conn] : connections) {
            if (conn.source_id == node_id || conn.target_id == node_id) {
                conn.last_activation = current_time;
                conn.strength = std::min(1.0f, conn.strength + 0.05f);
            }
        }
    }
}

std::vector<uint64_t> GlobalMemoryManager::get_related_nodes(uint64_t node_id, int max_depth) {
    std::lock_guard<std::mutex> lock(memory_mutex);
    std::vector<uint64_t> result;
    std::set<uint64_t> visited;
    std::vector<uint64_t> current_level = {node_id};
    
    for (int depth = 0; depth < max_depth && !current_level.empty(); ++depth) {
        std::vector<uint64_t> next_level;
        
        for (uint64_t node_id : current_level) {
            if (visited.find(node_id) != visited.end()) continue;
            visited.insert(node_id);
            
            auto it = nodes.find(node_id);
            if (it != nodes.end()) {
                for (uint64_t connected_id : it->second.connections) {
                    if (visited.find(connected_id) == visited.end()) {
                        next_level.push_back(connected_id);
                        result.push_back(connected_id);
                    }
                }
            }
        }
        
        current_level = next_level;
    }
    
    return result;
}

std::vector<uint64_t> GlobalMemoryManager::process_input_to_nodes(const std::string& input) {
    std::vector<uint64_t> activated_nodes;
    
    // Tokenize input
    std::istringstream iss(input);
    std::string word;
    std::vector<std::string> words;
    
    while (iss >> word) {
        // Clean word (remove punctuation)
        word.erase(std::remove_if(word.begin(), word.end(), 
            [](char c) { return !std::isalnum(c); }), word.end());
        if (!word.empty()) {
            words.push_back(word);
        }
    }
    
    // Create nodes for each word
    for (const std::string& word : words) {
        uint64_t node_id = create_node(word, NodeType::TEXT, 100);
        activated_nodes.push_back(node_id);
    }
    
    // Create a concept node for the full input
    uint64_t concept_id = create_node(input, NodeType::CONCEPT, 150);
    activated_nodes.push_back(concept_id);
    
    // Create connections between words and concept
    for (size_t i = 0; i < words.size(); ++i) {
        uint64_t word_id = activated_nodes[i];
        create_connection(word_id, concept_id, ConnectionType::SEMANTIC, 0.7f);
        
        // Create connections between adjacent words
        if (i > 0) {
            uint64_t prev_word_id = activated_nodes[i-1];
            create_connection(prev_word_id, word_id, ConnectionType::TEMPORAL, 0.6f);
        }
    }
    
    return activated_nodes;
}

std::string GlobalMemoryManager::generate_response_from_nodes(const std::vector<uint64_t>& activated_nodes) {
    if (activated_nodes.empty()) {
        return "I don't have enough information to respond.";
    }
    
    std::ostringstream response;
    response << "Based on my memory: ";
    
    // Get related nodes
    std::set<uint64_t> all_related;
    for (uint64_t node_id : activated_nodes) {
        auto related = get_related_nodes(node_id, 1);
        all_related.insert(related.begin(), related.end());
    }
    
    // Build response from most important nodes
    std::vector<std::pair<uint64_t, uint8_t>> node_importance;
    for (uint64_t node_id : all_related) {
        MemoryNode* node = get_node(node_id);
        if (node) {
            node_importance.push_back({node_id, node->importance});
        }
    }
    
    // Sort by importance
    std::sort(node_importance.begin(), node_importance.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Add top 3 most important concepts to response
    for (size_t i = 0; i < std::min(size_t(3), node_importance.size()); ++i) {
        MemoryNode* node = get_node(node_importance[i].first);
        if (node && node->type == NodeType::CONCEPT) {
            response << node->content.substr(0, 100) << "... ";
        }
    }
    
    return response.str();
}

void GlobalMemoryManager::save_to_disk() {
    std::lock_guard<std::mutex> lock(memory_mutex);
    
    // Save nodes
    std::ofstream nodes_stream(nodes_file);
    if (nodes_stream.is_open()) {
        for (const auto& [id, node] : nodes) {
            nodes_stream << serialize_node(node) << "\n";
        }
        nodes_stream.close();
    }
    
    // Save connections
    std::ofstream conn_stream(connections_file);
    if (conn_stream.is_open()) {
        for (const auto& [id, conn] : connections) {
            conn_stream << serialize_connection(conn) << "\n";
        }
        conn_stream.close();
    }
    
    std::cout << "💾 Saved " << nodes.size() << " nodes and " << connections.size() 
              << " connections to disk" << std::endl;
}

void GlobalMemoryManager::load_from_disk() {
    std::lock_guard<std::mutex> lock(memory_mutex);
    
    // Load nodes
    std::ifstream nodes_stream(nodes_file);
    if (nodes_stream.is_open()) {
        std::string line;
        while (std::getline(nodes_stream, line)) {
            if (!line.empty()) {
                MemoryNode node = deserialize_node(line);
                nodes[node.id] = node;
                content_to_node[node.content] = node.id;
                next_node_id = std::max(next_node_id, node.id + 1);
            }
        }
        nodes_stream.close();
    }
    
    // Load connections
    std::ifstream conn_stream(connections_file);
    if (conn_stream.is_open()) {
        std::string line;
        while (std::getline(conn_stream, line)) {
            if (!line.empty()) {
                MemoryConnection conn = deserialize_connection(line);
                connections[conn.id] = conn;
                next_connection_id = std::max(next_connection_id, conn.id + 1);
            }
        }
        conn_stream.close();
    }
    
    std::cout << "📁 Loaded " << nodes.size() << " nodes and " << connections.size() 
              << " connections from disk" << std::endl;
}

std::string GlobalMemoryManager::get_memory_stats() const {
    std::ostringstream stats;
    stats << "🧠 Memory Statistics:\n";
    stats << "  Nodes: " << nodes.size() << "\n";
    stats << "  Connections: " << connections.size() << "\n";
    stats << "  Storage: " << storage_path << "\n";
    return stats.str();
}

std::string GlobalMemoryManager::serialize_node(const MemoryNode& node) {
    std::ostringstream oss;
    oss << node.id << "|" << node.content << "|" << static_cast<int>(node.type) 
        << "|" << node.creation_time << "|" << static_cast<int>(node.importance);
    return oss.str();
}

std::string GlobalMemoryManager::serialize_connection(const MemoryConnection& conn) {
    std::ostringstream oss;
    oss << conn.id << "|" << conn.source_id << "|" << conn.target_id 
        << "|" << static_cast<int>(conn.type) << "|" << conn.strength 
        << "|" << conn.creation_time << "|" << conn.last_activation;
    return oss.str();
}

MemoryNode GlobalMemoryManager::deserialize_node(const std::string& data) {
    MemoryNode node;
    std::istringstream iss(data);
    std::string token;
    
    if (std::getline(iss, token, '|')) node.id = std::stoull(token);
    if (std::getline(iss, token, '|')) node.content = token;
    if (std::getline(iss, token, '|')) node.type = static_cast<NodeType>(std::stoi(token));
    if (std::getline(iss, token, '|')) node.creation_time = std::stoull(token);
    if (std::getline(iss, token, '|')) node.importance = static_cast<uint8_t>(std::stoi(token));
    
    return node;
}

MemoryConnection GlobalMemoryManager::deserialize_connection(const std::string& data) {
    MemoryConnection conn;
    std::istringstream iss(data);
    std::string token;
    
    if (std::getline(iss, token, '|')) conn.id = std::stoull(token);
    if (std::getline(iss, token, '|')) conn.source_id = std::stoull(token);
    if (std::getline(iss, token, '|')) conn.target_id = std::stoull(token);
    if (std::getline(iss, token, '|')) conn.type = static_cast<ConnectionType>(std::stoi(token));
    if (std::getline(iss, token, '|')) conn.strength = std::stof(token);
    if (std::getline(iss, token, '|')) conn.creation_time = std::stoull(token);
    if (std::getline(iss, token, '|')) conn.last_activation = std::stoull(token);
    
    return conn;
}

} // namespace melvin

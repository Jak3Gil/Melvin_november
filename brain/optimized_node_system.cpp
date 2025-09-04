#include "optimized_node_system.hpp"
#include <chrono>
#include <cstring>
#include <sstream>
#include <regex>
#include <algorithm>
#include <numeric>

namespace melvin {

// Static size configurations
const std::unordered_map<NodeSize, NodeConfig> OptimizedDynamicNodeSizer::SIZE_CONFIGS_ = {
    {NodeSize::TINY, {
        .size_category = NodeSize::TINY,
        .node_type = NodeType::WORD,
        .connection_strategy = ConnectionType::SIMILARITY,
        .max_connections = 5,
        .similarity_threshold = 204, // 0.8 * 255
        .min_size = 1,
        .max_size = 10,
        .flags = 0
    }},
    {NodeSize::SMALL, {
        .size_category = NodeSize::SMALL,
        .node_type = NodeType::PHRASE,
        .connection_strategy = ConnectionType::SIMILARITY,
        .max_connections = 10,
        .similarity_threshold = 153, // 0.6 * 255
        .min_size = 11,
        .max_size = 50,
        .flags = 0
    }},
    {NodeSize::MEDIUM, {
        .size_category = NodeSize::MEDIUM,
        .node_type = NodeType::CONCEPT,
        .connection_strategy = ConnectionType::HIERARCHICAL,
        .max_connections = 20,
        .similarity_threshold = 102, // 0.4 * 255
        .min_size = 51,
        .max_size = 200,
        .flags = 0
    }},
    {NodeSize::LARGE, {
        .size_category = NodeSize::LARGE,
        .node_type = NodeType::SECTION,
        .connection_strategy = ConnectionType::TEMPORAL,
        .max_connections = 50,
        .similarity_threshold = 77, // 0.3 * 255
        .min_size = 201,
        .max_size = 1000,
        .flags = 0
    }},
    {NodeSize::EXTRA_LARGE, {
        .size_category = NodeSize::EXTRA_LARGE,
        .node_type = NodeType::DOCUMENT,
        .connection_strategy = ConnectionType::ALL,
        .max_connections = 100,
        .similarity_threshold = 51, // 0.2 * 255
        .min_size = 1001,
        .max_size = 10000,
        .flags = 0
    }}
};

// NodeStorage implementation
NodeStorage::NodeStorage() 
    : total_nodes_(0), total_connections_(0), total_content_bytes_(0) {
    
    // Pre-allocate pools
    content_pool_.reserve(INITIAL_POOL_SIZE);
    connection_pool_.reserve(INITIAL_POOL_SIZE);
    node_pool_.reserve(INITIAL_POOL_SIZE);
}

uint64_t NodeStorage::create_node(const std::string& content, const NodeConfig& config) {
    // Check for content deduplication
    uint64_t existing_id = find_existing_content(content);
    if (existing_id != 0) {
        return existing_id;
    }
    
    // Allocate space for content
    uint32_t content_offset = allocate_content_space(content.size());
    if (content_offset == UINT32_MAX) {
        return 0; // Allocation failed
    }
    
    // Copy content to pool
    std::copy(content.begin(), content.end(), 
              content_pool_.begin() + content_offset);
    
    // Allocate node space
    uint32_t node_index = allocate_node_space();
    if (node_index == UINT32_MAX) {
        free_content_space(content_offset, content.size());
        return 0; // Allocation failed
    }
    
    // Create node
    OptimizedNode& node = node_pool_[node_index];
    node.id = hash_string(content); // Use content hash as ID
    node.content_length = static_cast<uint32_t>(content.size());
    node.content_offset = content_offset;
    node.config = config;
    node.complexity_score = 0.0f; // Will be calculated later
    node.parent_id = 0;
    node.creation_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    node.connection_count = 0;
    node.connection_offset = 0;
    
    // Update indexes
    id_to_index_[node.id] = node_index;
    content_to_id_[content] = node.id;
    
    // Update statistics
    total_nodes_++;
    total_content_bytes_ += content.size();
    
    return node.id;
}

OptimizedNode* NodeStorage::get_node(uint64_t node_id) {
    auto it = id_to_index_.find(node_id);
    if (it == id_to_index_.end()) {
        return nullptr;
    }
    return &node_pool_[it->second];
}

const OptimizedNode* NodeStorage::get_node(uint64_t node_id) const {
    auto it = id_to_index_.find(node_id);
    if (it == id_to_index_.end()) {
        return nullptr;
    }
    return &node_pool_[it->second];
}

std::string NodeStorage::get_node_content(uint64_t node_id) const {
    const OptimizedNode* node = get_node(node_id);
    if (!node) {
        return "";
    }
    
    return std::string(content_pool_.begin() + node->content_offset,
                      content_pool_.begin() + node->content_offset + node->content_length);
}

uint64_t NodeStorage::create_connection(uint64_t source_id, uint64_t target_id, 
                                       float weight, ConnectionType type) {
    // Allocate connection space
    uint32_t conn_index = allocate_connection_space();
    if (conn_index == UINT32_MAX) {
        return 0; // Allocation failed
    }
    
    // Create connection
    NodeConnection& conn = connection_pool_[conn_index];
    conn.source_id = source_id;
    conn.target_id = target_id;
    conn.weight = weight;
    conn.type = type;
    
    // Update source node's connection info
    OptimizedNode* source_node = get_node(source_id);
    if (source_node) {
        if (source_node->connection_count == 0) {
            source_node->connection_offset = conn_index;
        }
        source_node->connection_count++;
    }
    
    total_connections_++;
    return static_cast<uint64_t>(conn_index);
}

std::vector<NodeConnection> NodeStorage::get_node_connections(uint64_t node_id) const {
    const OptimizedNode* node = get_node(node_id);
    if (!node || node->connection_count == 0) {
        return {};
    }
    
    std::vector<NodeConnection> connections;
    connections.reserve(node->connection_count);
    
    // Collect all connections for this node
    for (size_t i = 0; i < connection_pool_.size(); ++i) {
        const NodeConnection& conn = connection_pool_[i];
        if (conn.source_id == node_id || conn.target_id == node_id) {
            connections.push_back(conn);
        }
    }
    
    return connections;
}

// Memory management helpers
uint32_t NodeStorage::allocate_content_space(size_t size) {
    // Find best fit in free list
    size_t best_fit = SIZE_MAX;
    size_t best_index = SIZE_MAX;
    
    for (size_t i = 0; i < content_free_list_.size(); ++i) {
        if (content_free_list_[i] >= size && content_free_list_[i] < best_fit) {
            best_fit = content_free_list_[i];
            best_index = i;
        }
    }
    
    if (best_index != SIZE_MAX) {
        // Use existing space
        uint32_t offset = static_cast<uint32_t>(best_index);
        content_free_list_.erase(content_free_list_.begin() + best_index);
        return offset;
    }
    
    // Allocate new space at end
    uint32_t offset = static_cast<uint32_t>(content_pool_.size());
    content_pool_.resize(content_pool_.size() + size);
    return offset;
}

uint32_t NodeStorage::allocate_node_space() {
    if (!node_free_list_.empty()) {
        uint32_t index = node_free_list_.back();
        node_free_list_.pop_back();
        return index;
    }
    
    uint32_t index = static_cast<uint32_t>(node_pool_.size());
    node_pool_.resize(node_pool_.size() + 1);
    return index;
}

uint32_t NodeStorage::allocate_connection_space() {
    if (!connection_free_list_.empty()) {
        uint32_t index = connection_free_list_.back();
        connection_free_list_.pop_back();
        return index;
    }
    
    uint32_t index = static_cast<uint32_t>(connection_pool_.size());
    connection_pool_.resize(connection_pool_.size() + 1);
    return index;
}

uint64_t NodeStorage::find_existing_content(const std::string& content) const {
    auto it = content_to_id_.find(content);
    return (it != content_to_id_.end()) ? it->second : 0;
}

size_t NodeStorage::get_memory_usage() const {
    return content_pool_.size() + 
           (connection_pool_.size() * sizeof(NodeConnection)) +
           (node_pool_.size() * sizeof(OptimizedNode)) +
           (id_to_index_.size() * (sizeof(uint64_t) + sizeof(uint32_t))) +
           (content_to_id_.size() * (sizeof(std::string) + sizeof(uint64_t)));
}

// OptimizedDynamicNodeSizer implementation
OptimizedDynamicNodeSizer::OptimizedDynamicNodeSizer() 
    : storage_(std::make_unique<NodeStorage>()) {
}

std::vector<uint64_t> OptimizedDynamicNodeSizer::create_dynamic_nodes(
    const std::string& text, NodeSize preferred_size, float complexity_threshold) {
    
    if (preferred_size == NodeSize::MEDIUM) {
        return create_auto_sized_nodes(text, complexity_threshold);
    }
    
    switch (preferred_size) {
        case NodeSize::TINY:
            return create_tiny_nodes(text);
        case NodeSize::SMALL:
            return create_small_nodes(text);
        case NodeSize::MEDIUM:
            return create_medium_nodes(text);
        case NodeSize::LARGE:
            return create_large_nodes(text);
        case NodeSize::EXTRA_LARGE:
            return create_extra_large_nodes(text);
        default:
            return {};
    }
}

std::vector<uint64_t> OptimizedDynamicNodeSizer::create_auto_sized_nodes(
    const std::string& text, float complexity_threshold) {
    
    std::vector<uint64_t> node_ids;
    float complexity = calculate_complexity(text);
    NodeSize optimal_size = determine_optimal_size(text, complexity);
    
    // Create nodes based on optimal size
    switch (optimal_size) {
        case NodeSize::TINY:
            node_ids = create_tiny_nodes(text);
            break;
        case NodeSize::SMALL:
            node_ids = create_small_nodes(text);
            break;
        case NodeSize::MEDIUM:
            node_ids = create_medium_nodes(text);
            break;
        case NodeSize::LARGE:
            node_ids = create_large_nodes(text);
            break;
        case NodeSize::EXTRA_LARGE:
            node_ids = create_extra_large_nodes(text);
            break;
    }
    
    // If complexity is high, add granular nodes
    if (complexity > complexity_threshold) {
        auto granular_ids = create_tiny_nodes(text);
        node_ids.insert(node_ids.end(), granular_ids.begin(), granular_ids.end());
    }
    
    return node_ids;
}

std::vector<uint64_t> OptimizedDynamicNodeSizer::create_tiny_nodes(const std::string& text) {
    std::vector<uint64_t> node_ids;
    
    // Simple word extraction using regex
    std::regex word_regex(R"(\b[a-zA-Z]+\b)");
    std::sregex_iterator iter(text.begin(), text.end(), word_regex);
    std::sregex_iterator end;
    
    for (; iter != end; ++iter) {
        std::string word = iter->str();
        if (word.length() >= 3) { // Skip very short words
            uint64_t node_id = create_single_node(word, NodeSize::TINY, 
                                                NodeType::WORD, calculate_complexity(word));
            if (node_id != 0) {
                node_ids.push_back(node_id);
                update_stats(NodeSize::TINY);
            }
        }
    }
    
    return node_ids;
}

std::vector<uint64_t> OptimizedDynamicNodeSizer::create_small_nodes(const std::string& text) {
    std::vector<uint64_t> node_ids;
    auto phrases = extract_phrases(text);
    
    for (const auto& phrase : phrases) {
        if (phrase.length() >= 11 && phrase.length() <= 50) {
            uint64_t node_id = create_single_node(phrase, NodeSize::SMALL, 
                                                NodeType::PHRASE, calculate_complexity(phrase));
            if (node_id != 0) {
                node_ids.push_back(node_id);
                update_stats(NodeSize::SMALL);
            }
        }
    }
    
    return node_ids;
}

std::vector<uint64_t> OptimizedDynamicNodeSizer::create_medium_nodes(const std::string& text) {
    std::vector<uint64_t> node_ids;
    auto chunks = split_into_chunks(text, 100);
    
    for (const auto& chunk : chunks) {
        if (chunk.length() >= 51 && chunk.length() <= 200) {
            uint64_t node_id = create_single_node(chunk, NodeSize::MEDIUM, 
                                                NodeType::CONCEPT, calculate_complexity(chunk));
            if (node_id != 0) {
                node_ids.push_back(node_id);
                update_stats(NodeSize::MEDIUM);
            }
        }
    }
    
    return node_ids;
}

std::vector<uint64_t> OptimizedDynamicNodeSizer::create_large_nodes(const std::string& text) {
    std::vector<uint64_t> node_ids;
    auto chunks = split_into_chunks(text, 500);
    
    for (const auto& chunk : chunks) {
        if (chunk.length() >= 201 && chunk.length() <= 1000) {
            uint64_t node_id = create_single_node(chunk, NodeSize::LARGE, 
                                                NodeType::SECTION, calculate_complexity(chunk));
            if (node_id != 0) {
                node_ids.push_back(node_id);
                update_stats(NodeSize::LARGE);
            }
        }
    }
    
    return node_ids;
}

std::vector<uint64_t> OptimizedDynamicNodeSizer::create_extra_large_nodes(const std::string& text) {
    std::vector<uint64_t> node_ids;
    auto chunks = split_into_chunks(text, 2000);
    
    for (const auto& chunk : chunks) {
        if (chunk.length() >= 1001 && chunk.length() <= 10000) {
            uint64_t node_id = create_single_node(chunk, NodeSize::EXTRA_LARGE, 
                                                NodeType::DOCUMENT, calculate_complexity(chunk));
            if (node_id != 0) {
                node_ids.push_back(node_id);
                update_stats(NodeSize::EXTRA_LARGE);
            }
        }
    }
    
    return node_ids;
}

uint64_t OptimizedDynamicNodeSizer::create_single_node(const std::string& content, 
                                                      NodeSize size_category, 
                                                      NodeType node_type, 
                                                      float complexity_score) {
    auto it = SIZE_CONFIGS_.find(size_category);
    if (it == SIZE_CONFIGS_.end()) {
        return 0;
    }
    
    NodeConfig config = it->second;
    config.node_type = node_type;
    
    uint64_t node_id = storage_->create_node(content, config);
    if (node_id != 0) {
        OptimizedNode* node = storage_->get_node(node_id);
        if (node) {
            node->complexity_score = complexity_score;
        }
    }
    
    return node_id;
}

float OptimizedDynamicNodeSizer::calculate_complexity(const std::string& text) const {
    // Check cache first
    auto it = complexity_cache_.find(text);
    if (it != complexity_cache_.end()) {
        return it->second;
    }
    
    // Simple complexity calculation
    std::istringstream iss(text);
    std::vector<std::string> words(std::istream_iterator<std::string>{iss},
                                  std::istream_iterator<std::string>());
    
    if (words.empty()) {
        return 0.0f;
    }
    
    // Calculate average word length
    float avg_word_length = std::accumulate(words.begin(), words.end(), 0.0f,
        [](float acc, const std::string& word) { return acc + word.length(); }) / words.size();
    
    // Calculate vocabulary diversity
    std::unordered_set<std::string> unique_words(words.begin(), words.end());
    float vocabulary_diversity = static_cast<float>(unique_words.size()) / words.size();
    
    // Combine metrics
    float complexity = (avg_word_length * 0.4f) + (vocabulary_diversity * 0.6f);
    complexity = std::min(1.0f, complexity);
    
    // Cache result
    complexity_cache_[text] = complexity;
    
    return complexity;
}

NodeSize OptimizedDynamicNodeSizer::determine_optimal_size(const std::string& text, float complexity) const {
    size_t length = text.length();
    
    if (length <= 10) return NodeSize::TINY;
    if (length <= 50) return NodeSize::SMALL;
    if (length <= 200) return NodeSize::MEDIUM;
    if (length <= 1000) return NodeSize::LARGE;
    return NodeSize::EXTRA_LARGE;
}

std::vector<std::string> OptimizedDynamicNodeSizer::extract_phrases(const std::string& text) const {
    std::vector<std::string> phrases;
    std::istringstream iss(text);
    std::vector<std::string> words(std::istream_iterator<std::string>{iss},
                                  std::istream_iterator<std::string>());
    
    // Extract 2-word phrases
    for (size_t i = 0; i < words.size() - 1; ++i) {
        std::string phrase = words[i] + " " + words[i + 1];
        if (is_meaningful_phrase(phrase)) {
            phrases.push_back(phrase);
        }
    }
    
    // Extract 3-word phrases
    for (size_t i = 0; i < words.size() - 2; ++i) {
        std::string phrase = words[i] + " " + words[i + 1] + " " + words[i + 2];
        if (is_meaningful_phrase(phrase)) {
            phrases.push_back(phrase);
        }
    }
    
    return phrases;
}

std::vector<std::string> OptimizedDynamicNodeSizer::split_into_chunks(const std::string& text, size_t target_size) const {
    std::vector<std::string> chunks;
    std::istringstream iss(text);
    std::vector<std::string> sentences;
    std::string sentence;
    
    // Split into sentences
    while (std::getline(iss, sentence, '.')) {
        if (!sentence.empty()) {
            sentences.push_back(sentence + ".");
        }
    }
    
    // Group sentences into chunks
    std::string current_chunk;
    for (const auto& sent : sentences) {
        if (current_chunk.length() + sent.length() <= target_size) {
            current_chunk += sent + " ";
        } else {
            if (!current_chunk.empty()) {
                chunks.push_back(current_chunk);
            }
            current_chunk = sent + " ";
        }
    }
    
    if (!current_chunk.empty()) {
        chunks.push_back(current_chunk);
    }
    
    return chunks;
}

bool OptimizedDynamicNodeSizer::is_meaningful_phrase(const std::string& phrase) const {
    std::istringstream iss(phrase);
    std::vector<std::string> words(std::istream_iterator<std::string>{iss},
                                  std::istream_iterator<std::string>());
    
    if (words.size() < 2) return false;
    
    // Check if all words are substantial
    return std::all_of(words.begin(), words.end(), 
                      [](const std::string& word) { return word.length() > 2; });
}

void OptimizedDynamicNodeSizer::update_stats(NodeSize size_category) {
    switch (size_category) {
        case NodeSize::TINY: stats_.tiny_nodes++; break;
        case NodeSize::SMALL: stats_.small_nodes++; break;
        case NodeSize::MEDIUM: stats_.medium_nodes++; break;
        case NodeSize::LARGE: stats_.large_nodes++; break;
        case NodeSize::EXTRA_LARGE: stats_.extra_large_nodes++; break;
    }
}

size_t OptimizedDynamicNodeSizer::get_memory_usage() const {
    return storage_->get_memory_usage() + 
           (complexity_cache_.size() * (sizeof(std::string) + sizeof(float)));
}

// Utility functions
namespace utils {

uint64_t hash_string(const std::string& str) {
    // Simple hash function - in production, use a better one
    uint64_t hash = 0;
    for (char c : str) {
        hash = hash * 31 + c;
    }
    return hash;
}

void copy_bytes(void* dest, const void* src, size_t size) {
    std::memcpy(dest, src, size);
}

void zero_bytes(void* ptr, size_t size) {
    std::memset(ptr, 0, size);
}

size_t align_to(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

bool is_aligned(const void* ptr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

// MemoryPool implementation
MemoryPool::MemoryPool(size_t initial_size) 
    : next_free_(0) {
    pool_.resize(initial_size);
}

void* MemoryPool::allocate(size_t size) {
    // Simple first-fit allocation
    for (size_t i = 0; i < pool_.size(); ++i) {
        if (i + size <= pool_.size()) {
            return &pool_[i];
        }
    }
    
    // Need to grow pool
    size_t old_size = pool_.size();
    pool_.resize(old_size + size);
    return &pool_[old_size];
}

void MemoryPool::deallocate(void* ptr, size_t size) {
    // Add to free list for future reuse
    size_t offset = static_cast<char*>(ptr) - pool_.data();
    free_list_.push_back(offset);
}

void MemoryPool::compact() {
    // Simple compaction - move all allocated blocks to front
    std::vector<uint8_t> new_pool;
    new_pool.reserve(pool_.size());
    
    // This is a simplified version - real implementation would be more complex
    pool_ = std::move(new_pool);
    free_list_.clear();
    next_free_ = 0;
}

size_t MemoryPool::get_usage() const {
    return pool_.size();
}

} // namespace utils

} // namespace melvin

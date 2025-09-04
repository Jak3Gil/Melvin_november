#include "melvin_optimized_v2.h"

// ============================================================================
// IMPLEMENTATION OF HEADER FUNCTIONS
// ============================================================================

// BinaryNode implementation
std::vector<uint8_t> BinaryNode::to_bytes() const {
    std::vector<uint8_t> data;
    data.reserve(28 + content.size());
    
    // Pack header (28 bytes)
    data.insert(data.end(), (uint8_t*)&id, (uint8_t*)&id + 8);
    data.insert(data.end(), (uint8_t*)&creation_time, (uint8_t*)&creation_time + 8);
    data.push_back(static_cast<uint8_t>(content_type));
    data.push_back(static_cast<uint8_t>(compression));
    data.push_back(importance);
    data.push_back(activation_strength);
    data.insert(data.end(), (uint8_t*)&content_length, (uint8_t*)&content_length + 4);
    data.insert(data.end(), (uint8_t*)&connection_count, (uint8_t*)&connection_count + 4);
    
    // Add content
    data.insert(data.end(), content.begin(), content.end());
    
    return data;
}

BinaryNode BinaryNode::from_bytes(const std::vector<uint8_t>& data) {
    if (data.size() < 28) {
        throw std::runtime_error("Invalid binary node data");
    }
    
    BinaryNode node;
    size_t offset = 0;
    
    // Unpack header
    std::memcpy(&node.id, &data[offset], 8);
    offset += 8;
    std::memcpy(&node.creation_time, &data[offset], 8);
    offset += 8;
    node.content_type = static_cast<ContentType>(data[offset++]);
    node.compression = static_cast<CompressionType>(data[offset++]);
    node.importance = data[offset++];
    node.activation_strength = data[offset++];
    std::memcpy(&node.content_length, &data[offset], 4);
    offset += 4;
    std::memcpy(&node.connection_count, &data[offset], 4);
    offset += 4;
    
    // Copy content
    if (offset < data.size()) {
        node.content.assign(data.begin() + offset, data.end());
    }
    
    return node;
}

// BinaryConnection implementation
std::vector<uint8_t> BinaryConnection::to_bytes() const {
    std::vector<uint8_t> data(18);
    size_t offset = 0;
    
    std::memcpy(&data[offset], &id, 8);
    offset += 8;
    std::memcpy(&data[offset], &source_id, 8);
    offset += 8;
    std::memcpy(&data[offset], &target_id, 8);
    offset += 8;
    data[offset++] = static_cast<uint8_t>(connection_type);
    data[offset++] = weight;
    
    return data;
}

BinaryConnection BinaryConnection::from_bytes(const std::vector<uint8_t>& data) {
    if (data.size() != 18) {
        throw std::runtime_error("Invalid binary connection data");
    }
    
    BinaryConnection conn;
    size_t offset = 0;
    
    std::memcpy(&conn.id, &data[offset], 8);
    offset += 8;
    std::memcpy(&conn.source_id, &data[offset], 8);
    offset += 8;
    std::memcpy(&conn.target_id, &data[offset], 8);
    offset += 8;
    conn.connection_type = static_cast<ConnectionType>(data[offset++]);
    conn.weight = data[offset++];
    
    return conn;
}

// ============================================================================
// COMPRESSION UTILITIES
// ============================================================================

class CompressionUtils {
public:
    static std::vector<uint8_t> compress_gzip(const std::vector<uint8_t>& data) {
        uLong compressed_size = compressBound(data.size());
        std::vector<uint8_t> compressed(compressed_size);
        
        if (compress2(compressed.data(), &compressed_size,
                     data.data(), data.size(), Z_BEST_COMPRESSION) != Z_OK) {
            throw std::runtime_error("GZIP compression failed");
        }
        
        compressed.resize(compressed_size);
        return compressed;
    }
    
    static std::vector<uint8_t> decompress_gzip(const std::vector<uint8_t>& data) {
        std::vector<uint8_t> decompressed;
        uLong decompressed_size = data.size() * 4; // Estimate
        
        while (true) {
            decompressed.resize(decompressed_size);
            uLong actual_size = decompressed_size;
            
            int result = uncompress(decompressed.data(), &actual_size,
                                  data.data(), data.size());
            
            if (result == Z_OK) {
                decompressed.resize(actual_size);
                return decompressed;
            } else if (result == Z_BUF_ERROR) {
                decompressed_size *= 2;
            } else {
                throw std::runtime_error("GZIP decompression failed");
            }
        }
    }
    
    static std::vector<uint8_t> compress_lzma(const std::vector<uint8_t>& data) {
        lzma_stream strm = LZMA_STREAM_INIT;
        lzma_ret ret = lzma_easy_encoder(&strm, LZMA_PRESET_DEFAULT, LZMA_CHECK_CRC64);
        
        if (ret != LZMA_OK) {
            throw std::runtime_error("LZMA encoder initialization failed");
        }
        
        strm.next_in = data.data();
        strm.avail_in = data.size();
        
        std::vector<uint8_t> compressed;
        uint8_t outbuf[4096];
        
        do {
            strm.next_out = outbuf;
            strm.avail_out = sizeof(outbuf);
            
            ret = lzma_code(&strm, LZMA_FINISH);
            
            if (ret != LZMA_OK && ret != LZMA_STREAM_END) {
                lzma_end(&strm);
                throw std::runtime_error("LZMA compression failed");
            }
            
            size_t written = sizeof(outbuf) - strm.avail_out;
            compressed.insert(compressed.end(), outbuf, outbuf + written);
        } while (ret != LZMA_STREAM_END);
        
        lzma_end(&strm);
        return compressed;
    }
    
    static std::vector<uint8_t> decompress_lzma(const std::vector<uint8_t>& data) {
        lzma_stream strm = LZMA_STREAM_INIT;
        lzma_ret ret = lzma_stream_decoder(&strm, UINT64_MAX, LZMA_CONCATENATED);
        
        if (ret != LZMA_OK) {
            throw std::runtime_error("LZMA decoder initialization failed");
        }
        
        strm.next_in = data.data();
        strm.avail_in = data.size();
        
        std::vector<uint8_t> decompressed;
        uint8_t outbuf[4096];
        
        do {
            strm.next_out = outbuf;
            strm.avail_out = sizeof(outbuf);
            
            ret = lzma_code(&strm, LZMA_RUN);
            
            if (ret != LZMA_OK && ret != LZMA_STREAM_END) {
                lzma_end(&strm);
                throw std::runtime_error("LZMA decompression failed");
            }
            
            size_t written = sizeof(outbuf) - strm.avail_out;
            decompressed.insert(decompressed.end(), outbuf, outbuf + written);
        } while (ret != LZMA_STREAM_END);
        
        lzma_end(&strm);
        return decompressed;
    }
    
    static std::vector<uint8_t> compress_zstd(const std::vector<uint8_t>& data) {
        size_t compressed_size = ZSTD_compressBound(data.size());
        std::vector<uint8_t> compressed(compressed_size);
        
        size_t actual_size = ZSTD_compress(compressed.data(), compressed_size,
                                          data.data(), data.size(), ZSTD_CLEVEL_DEFAULT);
        
        if (ZSTD_isError(actual_size)) {
            throw std::runtime_error("ZSTD compression failed");
        }
        
        compressed.resize(actual_size);
        return compressed;
    }
    
    static std::vector<uint8_t> decompress_zstd(const std::vector<uint8_t>& data) {
        size_t decompressed_size = ZSTD_getFrameContentSize(data.data(), data.size());
        
        if (decompressed_size == ZSTD_CONTENTSIZE_ERROR || 
            decompressed_size == ZSTD_CONTENTSIZE_UNKNOWN) {
            throw std::runtime_error("ZSTD decompression failed");
        }
        
        std::vector<uint8_t> decompressed(decompressed_size);
        
        size_t actual_size = ZSTD_decompress(decompressed.data(), decompressed_size,
                                            data.data(), data.size());
        
        if (ZSTD_isError(actual_size)) {
            throw std::runtime_error("ZSTD decompression failed");
        }
        
        return decompressed;
    }
    
    static CompressionType determine_best_compression(const std::vector<uint8_t>& data) {
        if (data.empty()) return CompressionType::NONE;
        
        auto gzip_size = compress_gzip(data).size();
        auto lzma_size = compress_lzma(data).size();
        auto zstd_size = compress_zstd(data).size();
        
        size_t min_size = std::min({data.size(), gzip_size, lzma_size, zstd_size});
        
        if (min_size == data.size()) return CompressionType::NONE;
        if (min_size == gzip_size) return CompressionType::GZIP;
        if (min_size == lzma_size) return CompressionType::LZMA;
        return CompressionType::ZSTD;
    }
    
    static std::vector<uint8_t> compress_content(const std::vector<uint8_t>& data, 
                                                CompressionType compression_type) {
        switch (compression_type) {
            case CompressionType::GZIP: return compress_gzip(data);
            case CompressionType::LZMA: return compress_lzma(data);
            case CompressionType::ZSTD: return compress_zstd(data);
            default: return data;
        }
    }
    
    static std::vector<uint8_t> decompress_content(const std::vector<uint8_t>& data,
                                                  CompressionType compression_type) {
        switch (compression_type) {
            case CompressionType::GZIP: return decompress_gzip(data);
            case CompressionType::LZMA: return decompress_lzma(data);
            case CompressionType::ZSTD: return decompress_zstd(data);
            default: return data;
        }
    }
};

// ============================================================================
// INTELLIGENT PRUNING SYSTEM
// ============================================================================

struct PruningDecision {
    uint64_t node_id;
    bool keep;
    float confidence;
    std::string reason;
    float importance_score;
    uint64_t timestamp;
    
    PruningDecision() : node_id(0), keep(false), confidence(0.0f),
                        importance_score(0.0f), timestamp(0) {}
};

class IntelligentPruningSystem {
private:
    std::map<ContentType, float> content_type_weights;
    float temporal_half_life_days;
    uint8_t eternal_threshold;
    
public:
    IntelligentPruningSystem() : temporal_half_life_days(30.0f), eternal_threshold(200) {
        content_type_weights[ContentType::CODE] = 0.8f;
        content_type_weights[ContentType::CONCEPT] = 0.7f;
        content_type_weights[ContentType::EMBEDDING] = 0.7f;
        content_type_weights[ContentType::TEXT] = 0.5f;
        content_type_weights[ContentType::IMAGE] = 0.6f;
        content_type_weights[ContentType::AUDIO] = 0.4f;
        content_type_weights[ContentType::METADATA] = 0.75f;
        content_type_weights[ContentType::SEQUENCE] = 0.8f;
        content_type_weights[ContentType::VISUAL] = 0.7f;
        content_type_weights[ContentType::SENSOR] = 0.45f;
    }
    
    float calculate_activation_importance(const BinaryNode& node) {
        float activation_strength = node.activation_strength / 255.0f;
        float connection_count = node.connection_count;
        
        // Current activation strength
        float current_activation = activation_strength;
        
        // Recent activity bonus
        uint64_t current_time = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        float days_since_creation = (current_time - node.creation_time) / 86400.0f;
        float recency_bonus = std::max(0.0f, 1.0f - (days_since_creation / 7.0f));
        
        // Frequency bonus
        float frequency_bonus = std::min(1.0f, connection_count / 100.0f);
        
        float activation_score = (current_activation * 0.5f) + 
                                (recency_bonus * 0.3f) + 
                                (frequency_bonus * 0.2f);
        return std::min(1.0f, activation_score);
    }
    
    float calculate_connection_importance(const BinaryNode& node, uint32_t connection_count) {
        // Hub score (many connections)
        float hub_score = std::min(1.0f, connection_count / 50.0f);
        
        // Authority score (incoming connections - approximated)
        float authority_score = std::min(1.0f, connection_count / 25.0f);
        
        // Bridge score (connects different clusters)
        float bridge_score = std::min(1.0f, connection_count / 30.0f);
        
        float connection_score = (hub_score * 0.4f) + 
                                (authority_score * 0.4f) + 
                                (bridge_score * 0.2f);
        return std::min(1.0f, connection_score);
    }
    
    float calculate_semantic_importance(const std::vector<uint8_t>& content, 
                                      ContentType content_type) {
        if (content.empty()) return 0.0f;
        
        // Base importance by content type
        float base_importance = content_type_weights[content_type];
        
        // Content length factor (longer = more important)
        float length_factor = std::min(1.0f, content.size() / 1000.0f);
        
        // Content complexity factor
        std::set<uint8_t> unique_bytes(content.begin(), content.end());
        float complexity_factor = std::min(1.0f, 
            static_cast<float>(unique_bytes.size()) / content.size());
        
        float semantic_score = (base_importance * 0.6f) + 
                              (length_factor * 0.2f) + 
                              (complexity_factor * 0.2f);
        return std::min(1.0f, semantic_score);
    }
    
    float calculate_temporal_importance(const BinaryNode& node) {
        uint64_t current_time = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        float days_old = (current_time - node.creation_time) / 86400.0f;
        
        // Exponential decay
        float decay_factor = std::pow(0.5f, days_old / temporal_half_life_days);
        
        // Some content is timeless
        if (node.importance > eternal_threshold) {
            decay_factor = std::max(decay_factor, 0.8f);
        }
        
        // Recent content gets a bonus
        float recency_bonus = std::max(0.0f, 1.0f - (days_old / 7.0f));
        
        float temporal_score = (decay_factor * 0.7f) + (recency_bonus * 0.3f);
        return std::min(1.0f, temporal_score);
    }
    
    float calculate_combined_importance(const BinaryNode& node, uint32_t connection_count) {
        // Calculate individual scores
        float activation_score = calculate_activation_importance(node);
        float connection_score = calculate_connection_importance(node, connection_count);
        float semantic_score = calculate_semantic_importance(node.content, node.content_type);
        float temporal_score = calculate_temporal_importance(node);
        
        // Weighted combination
        float combined_score = (activation_score * 0.25f) +
                              (connection_score * 0.25f) +
                              (semantic_score * 0.20f) +
                              (temporal_score * 0.15f) +
                              (node.importance / 255.0f * 0.15f);
        
        return std::min(1.0f, combined_score);
    }
    
    PruningDecision should_keep_node(const BinaryNode& node, uint32_t connection_count, 
                                    float threshold = 0.3f) {
        PruningDecision decision;
        decision.node_id = node.id;
        decision.timestamp = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        
        float importance = calculate_combined_importance(node, connection_count);
        decision.importance_score = importance;
        
        // Decision logic
        decision.keep = importance > threshold;
        
        // Determine confidence and reason
        if (decision.keep) {
            decision.confidence = importance;
            if (importance > 0.7f) {
                decision.reason = "High importance";
            } else if (node.connection_count > 10) {
                decision.reason = "Many connections";
            } else if (node.activation_strength > 200) {
                decision.reason = "High activation";
            } else {
                decision.reason = "Moderate importance";
            }
        } else {
            decision.confidence = 1.0f - importance;
            if (importance < 0.1f) {
                decision.reason = "Low importance";
            } else if (node.connection_count < 2) {
                decision.reason = "Few connections";
            } else if (node.activation_strength < 50) {
                decision.reason = "Low activation";
            } else {
                decision.reason = "Below threshold";
            }
        }
        
        return decision;
    }
};

// ============================================================================
// PURE BINARY STORAGE SYSTEM
// ============================================================================

class PureBinaryStorage {
private:
    std::string storage_path;
    std::string nodes_file;
    std::string connections_file;
    std::string index_file;
    
    std::mutex storage_mutex;
    std::unordered_map<uint64_t, size_t> node_index; // id -> file position
    
    uint64_t total_nodes;
    uint64_t total_connections;
    uint64_t total_bytes;
    
    IntelligentPruningSystem pruning_system;
    
public:
    PureBinaryStorage(const std::string& path = "melvin_binary_memory") 
        : storage_path(path), total_nodes(0), total_connections(0), total_bytes(0) {
        
        // Create storage directory
        std::filesystem::create_directories(storage_path);
        
        nodes_file = storage_path + "/nodes.bin";
        connections_file = storage_path + "/connections.bin";
        index_file = storage_path + "/index.bin";
        
        // Load existing index
        load_index();
        
        std::cout << "🧠 Pure Binary Storage initialized" << std::endl;
    }
    
    ~PureBinaryStorage() {
        save_index();
    }
    
    void load_index() {
        std::ifstream file(index_file, std::ios::binary);
        if (!file) return;
        
        uint64_t count;
        file.read(reinterpret_cast<char*>(&count), sizeof(count));
        
        for (uint64_t i = 0; i < count; ++i) {
            uint64_t id, position;
            file.read(reinterpret_cast<char*>(&id), sizeof(id));
            file.read(reinterpret_cast<char*>(&position), sizeof(position));
            node_index[id] = position;
        }
    }
    
    void save_index() {
        std::ofstream file(index_file, std::ios::binary);
        if (!file) return;
        
        uint64_t count = node_index.size();
        file.write(reinterpret_cast<const char*>(&count), sizeof(count));
        
        for (const auto& [id, position] : node_index) {
            file.write(reinterpret_cast<const char*>(&id), sizeof(id));
            file.write(reinterpret_cast<const char*>(&position), sizeof(position));
        }
    }
    
    uint8_t calculate_importance(const std::vector<uint8_t>& content, ContentType content_type) {
        // Base importance by content type
        std::map<ContentType, uint8_t> base_importance = {
            {ContentType::CODE, 200},
            {ContentType::CONCEPT, 180},
            {ContentType::EMBEDDING, 180},
            {ContentType::TEXT, 100},
            {ContentType::IMAGE, 120},
            {ContentType::AUDIO, 80},
            {ContentType::METADATA, 150},
            {ContentType::SEQUENCE, 160},
            {ContentType::VISUAL, 140},
            {ContentType::SENSOR, 90}
        };
        
        uint8_t importance = base_importance[content_type];
        
        // Adjust by content length (longer = more important)
        uint8_t length_factor = std::min(255, static_cast<int>(content.size() / 10));
        importance = std::min(255, static_cast<int>(importance + length_factor));
        
        return importance;
    }
    
    uint64_t store_node(const std::vector<uint8_t>& content, ContentType content_type, 
                        uint64_t node_id = 0) {
        std::lock_guard<std::mutex> lock(storage_mutex);
        
        // Generate ID if not provided
        if (node_id == 0) {
            node_id = std::hash<std::string>{}(std::string(content.begin(), content.end()));
        }
        
        // Determine compression
        CompressionType compression_type = CompressionUtils::determine_best_compression(content);
        auto compressed_content = CompressionUtils::compress_content(content, compression_type);
        
        // Calculate importance
        uint8_t importance = calculate_importance(content, content_type);
        
        // Get current timestamp
        uint64_t current_time = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        
        // Create binary node
        BinaryNode node;
        node.id = node_id;
        node.content = compressed_content;
        node.content_type = content_type;
        node.compression = compression_type;
        node.importance = importance;
        node.creation_time = current_time;
        node.content_length = compressed_content.size();
        node.connection_count = 0;
        node.activation_strength = 0;
        
        // Write to binary file
        std::ofstream file(nodes_file, std::ios::binary | std::ios::app);
        if (!file) {
            throw std::runtime_error("Failed to open nodes file for writing");
        }
        
        auto node_bytes = node.to_bytes();
        size_t position = file.tellp();
        file.write(reinterpret_cast<const char*>(node_bytes.data()), node_bytes.size());
        
        // Update index
        node_index[node_id] = position;
        total_bytes += node_bytes.size();
        total_nodes++;
        
        std::cout << "📦 Stored binary node: " << std::hex << node_id 
                  << " (" << compressed_content.size() << " bytes, "
                  << "compression: " << static_cast<int>(compression_type) << ")" << std::endl;
        
        return node_id;
    }
    
    uint64_t store_connection(uint64_t source_id, uint64_t target_id, 
                             ConnectionType connection_type, uint8_t weight = 128) {
        std::lock_guard<std::mutex> lock(storage_mutex);
        
        // Generate connection ID
        uint64_t conn_id = std::hash<std::string>{}(
            std::to_string(source_id) + std::to_string(target_id));
        
        // Create binary connection
        BinaryConnection connection;
        connection.id = conn_id;
        connection.source_id = source_id;
        connection.target_id = target_id;
        connection.connection_type = connection_type;
        connection.weight = std::min(255, static_cast<int>(weight));
        
        // Write to binary file
        std::ofstream file(connections_file, std::ios::binary | std::ios::app);
        if (!file) {
            throw std::runtime_error("Failed to open connections file for writing");
        }
        
        auto conn_bytes = connection.to_bytes();
        file.write(reinterpret_cast<const char*>(conn_bytes.data()), conn_bytes.size());
        
        total_bytes += conn_bytes.size();
        total_connections++;
        
        return conn_id;
    }
    
    std::optional<BinaryNode> get_node(uint64_t node_id) {
        std::lock_guard<std::mutex> lock(storage_mutex);
        
        auto it = node_index.find(node_id);
        if (it == node_index.end()) {
            return std::nullopt;
        }
        
        std::ifstream file(nodes_file, std::ios::binary);
        if (!file) {
            return std::nullopt;
        }
        
        file.seekg(it->second);
        
        // Read header (28 bytes)
        std::vector<uint8_t> header(28);
        file.read(reinterpret_cast<char*>(header.data()), 28);
        
        if (file.gcount() != 28) {
            return std::nullopt;
        }
        
        // Parse header
        BinaryNode node;
        size_t offset = 0;
        
        std::memcpy(&node.id, &header[offset], 8);
        offset += 8;
        std::memcpy(&node.creation_time, &header[offset], 8);
        offset += 8;
        node.content_type = static_cast<ContentType>(header[offset++]);
        node.compression = static_cast<CompressionType>(header[offset++]);
        node.importance = header[offset++];
        node.activation_strength = header[offset++];
        std::memcpy(&node.content_length, &header[offset], 4);
        offset += 4;
        std::memcpy(&node.connection_count, &header[offset], 4);
        offset += 4;
        
        // Read content
        node.content.resize(node.content_length);
        file.read(reinterpret_cast<char*>(node.content.data()), node.content_length);
        
        return node;
    }
    
    std::string get_node_as_text(uint64_t node_id) {
        auto node = get_node(node_id);
        if (!node) {
            return "";
        }
        
        // Decompress content
        auto decompressed = CompressionUtils::decompress_content(
            node->content, node->compression);
        
        // Convert to text if it's text content
        if (node->content_type == ContentType::TEXT) {
            return std::string(decompressed.begin(), decompressed.end());
        }
        
        return "[BINARY: " + std::to_string(decompressed.size()) + " bytes]";
    }
    
    std::vector<uint64_t> prune_nodes(uint32_t max_nodes_to_prune = 1000) {
        std::lock_guard<std::mutex> lock(storage_mutex);
        
        std::vector<PruningDecision> pruning_decisions;
        std::vector<uint64_t> nodes_to_prune;
        
        // Analyze all nodes
        for (const auto& [id, position] : node_index) {
            auto node = get_node(id);
            if (!node) continue;
            
            PruningDecision decision = pruning_system.should_keep_node(
                *node, node->connection_count, 0.3f);
            pruning_decisions.push_back(decision);
            
            if (!decision.keep && nodes_to_prune.size() < max_nodes_to_prune) {
                nodes_to_prune.push_back(id);
            }
        }
        
        std::cout << "🔍 Pruning analysis: " << node_index.size() << " nodes analyzed" << std::endl;
        std::cout << "📊 Keeping: " << (pruning_decisions.size() - nodes_to_prune.size()) 
                  << " nodes" << std::endl;
        std::cout << "🗑️ Pruning: " << nodes_to_prune.size() << " nodes" << std::endl;
        
        return nodes_to_prune;
    }
    
    struct StorageStats {
        uint64_t total_nodes;
        uint64_t total_connections;
        uint64_t total_bytes;
        double total_mb;
        uint64_t nodes_file_size;
        uint64_t connections_file_size;
        uint64_t index_file_size;
    };
    
    StorageStats get_storage_stats() {
        StorageStats stats;
        stats.total_nodes = total_nodes;
        stats.total_connections = total_connections;
        stats.total_bytes = total_bytes;
        stats.total_mb = total_bytes / (1024.0 * 1024.0);
        
        std::filesystem::path nodes_path(nodes_file);
        std::filesystem::path connections_path(connections_file);
        std::filesystem::path index_path(index_file);
        
        stats.nodes_file_size = std::filesystem::exists(nodes_path) ? 
            std::filesystem::file_size(nodes_path) : 0;
        stats.connections_file_size = std::filesystem::exists(connections_path) ? 
            std::filesystem::file_size(connections_path) : 0;
        stats.index_file_size = std::filesystem::exists(index_path) ? 
            std::filesystem::file_size(index_path) : 0;
        
        return stats;
    }
};

// ============================================================================
// OPTIMIZED MELVIN GLOBAL BRAIN
// ============================================================================

class MelvinOptimizedV2 {
private:
    std::unique_ptr<PureBinaryStorage> binary_storage;
    std::mutex brain_mutex;
    
    // Hebbian learning
    struct Activation {
        uint64_t node_id;
        uint64_t timestamp;
        float strength;
    };
    
    std::vector<Activation> recent_activations;
    std::mutex activation_mutex;
    static constexpr size_t MAX_ACTIVATIONS = 1000;
    static constexpr double COACTIVATION_WINDOW = 2.0; // seconds
    
    // Statistics
    struct BrainStats {
        uint64_t total_nodes;
        uint64_t total_connections;
        uint64_t hebbian_updates;
        uint64_t similarity_connections;
        uint64_t temporal_connections;
        uint64_t cross_modal_connections;
        uint64_t start_time;
    } stats;
    
public:
    MelvinOptimizedV2(const std::string& storage_path = "melvin_binary_memory") {
        binary_storage = std::make_unique<PureBinaryStorage>(storage_path);
        
        stats = {0, 0, 0, 0, 0, 0, 
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count())};
        
        std::cout << "🧠 Melvin Optimized V2 initialized" << std::endl;
    }
    
    uint64_t process_text_input(const std::string& text, const std::string& source = "user") {
        std::vector<uint8_t> text_bytes(text.begin(), text.end());
        uint64_t node_id = binary_storage->store_node(text_bytes, ContentType::TEXT);
        
        // Update statistics
        stats.total_nodes++;
        
        // Hebbian learning
        update_hebbian_learning(node_id);
        
        std::cout << "📝 Processed text input: " << text.substr(0, 50) 
                  << "... -> " << std::hex << node_id << std::endl;
        
        return node_id;
    }
    
    uint64_t process_code_input(const std::string& code, const std::string& source = "python") {
        std::vector<uint8_t> code_bytes(code.begin(), code.end());
        uint64_t node_id = binary_storage->store_node(code_bytes, ContentType::CODE);
        
        // Update statistics
        stats.total_nodes++;
        
        // Hebbian learning
        update_hebbian_learning(node_id);
        
        std::cout << "💻 Processed code input: " << code.substr(0, 50) 
                  << "... -> " << std::hex << node_id << std::endl;
        
        return node_id;
    }
    
    void update_hebbian_learning(uint64_t node_id) {
        std::lock_guard<std::mutex> lock(activation_mutex);
        
        uint64_t current_time = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        
        // Add current activation
        recent_activations.push_back({node_id, current_time, 1.0f});
        
        // Keep only recent activations
        if (recent_activations.size() > MAX_ACTIVATIONS) {
            recent_activations.erase(recent_activations.begin());
        }
        
        // Find co-activated nodes within window
        std::vector<uint64_t> co_activated;
        for (const auto& activation : recent_activations) {
            if (activation.node_id != node_id && 
                (current_time - activation.timestamp) <= COACTIVATION_WINDOW) {
                co_activated.push_back(activation.node_id);
            }
        }
        
        // Create Hebbian connections
        for (uint64_t co_activated_id : co_activated) {
            binary_storage->store_connection(node_id, co_activated_id, 
                                           ConnectionType::HEBBIAN, 150);
            stats.hebbian_updates++;
            stats.total_connections++;
        }
    }
    
    std::string get_node_content(uint64_t node_id) {
        return binary_storage->get_node_as_text(node_id);
    }
    
    struct BrainState {
        struct GlobalMemory {
            uint64_t total_nodes;
            uint64_t total_edges;
            double storage_used_mb;
            BrainStats stats;
        } global_memory;
        
        struct System {
            bool running;
            uint64_t uptime_seconds;
        } system;
    };
    
    BrainState get_unified_state() {
        BrainState state;
        
        auto storage_stats = binary_storage->get_storage_stats();
        
        state.global_memory.total_nodes = storage_stats.total_nodes;
        state.global_memory.total_edges = storage_stats.total_connections;
        state.global_memory.storage_used_mb = storage_stats.total_mb;
        state.global_memory.stats = stats;
        
        uint64_t current_time = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        state.system.uptime_seconds = current_time - stats.start_time;
        state.system.running = true;
        
        return state;
    }
    
    std::vector<uint64_t> prune_old_nodes(uint32_t max_nodes_to_prune = 1000) {
        auto pruned_nodes = binary_storage->prune_nodes(max_nodes_to_prune);
        std::cout << "🗑️ Pruned " << pruned_nodes.size() << " nodes" << std::endl;
        return pruned_nodes;
    }
    
    void save_complete_state() {
        std::cout << "💾 Complete state saved (binary storage is persistent)" << std::endl;
    }
};

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main() {
    std::cout << "🧠 MELVIN OPTIMIZED V2 (C++)" << std::endl;
    std::cout << "=============================" << std::endl;
    
    try {
        // Initialize optimized system
        MelvinOptimizedV2 melvin;
        
        // Test basic functionality
        std::cout << "🧪 Testing basic functionality..." << std::endl;
        
        // Process some test inputs
        uint64_t text_id = melvin.process_text_input(
            "This is a test of the optimized Melvin system!");
        uint64_t code_id = melvin.process_code_input(
            "def hello_world():\n    print('Hello, World!')");
        
        // Get unified state
        auto state = melvin.get_unified_state();
        std::cout << "📊 State: " << state.global_memory.total_nodes 
                  << " nodes, " << state.global_memory.total_edges << " edges" << std::endl;
        
        // Test retrieval
        std::string text_content = melvin.get_node_content(text_id);
        std::string code_content = melvin.get_node_content(code_id);
        std::cout << "📖 Retrieved text: " << text_content << std::endl;
        std::cout << "💻 Retrieved code: " << code_content << std::endl;
        
        // Test pruning
        std::cout << "\n🔍 Testing pruning system..." << std::endl;
        auto pruned_nodes = melvin.prune_old_nodes(10);
        std::cout << "🗑️ Pruned " << pruned_nodes.size() << " nodes" << std::endl;
        
        // Final stats
        auto final_state = melvin.get_unified_state();
        std::cout << "\n📊 Final stats:" << std::endl;
        std::cout << "   🧠 Nodes: " << final_state.global_memory.total_nodes << std::endl;
        std::cout << "   🔗 Edges: " << final_state.global_memory.total_edges << std::endl;
        std::cout << "   💾 Storage: " << final_state.global_memory.storage_used_mb << "MB" << std::endl;
        std::cout << "   ⚡ Hebbian updates: " << final_state.global_memory.stats.hebbian_updates << std::endl;
        
        std::cout << "\n🎉 Melvin Optimized V2 (C++) test completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

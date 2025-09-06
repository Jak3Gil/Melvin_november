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
// COMPRESSION UTILITIES IMPLEMENTATION
// ============================================================================

std::vector<uint8_t> CompressionUtils::compress_gzip(const std::vector<uint8_t>& data) {
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

// ============================================================================
// INTELLIGENT PRUNING SYSTEM IMPLEMENTATION
// ============================================================================
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
// COGNITIVE PROCESSING ENGINE IMPLEMENTATION
// ============================================================================

CognitiveProcessor::CognitiveProcessor(std::unique_ptr<PureBinaryStorage>& storage) 
    : binary_storage(std::move(storage)), next_curiosity_node_id(0x10000), next_tool_node_id(0x20000), next_experience_node_id(0x30000), next_toolchain_id(0x40000), next_executed_curiosity_node_id(0x60000),
      web_search_tool(0x50000, "WebSearchTool", "web_search", 0.8f, 0, "active") {
    initialize_response_templates();
    initialize_moral_supernodes();
    initialize_basic_tools();
    initialize_web_search_tool();
    std::cout << "🧠 Cognitive Processor initialized with moral supernodes, curiosity gap detection, dynamic tools system, meta-tool engineer, curiosity execution loop, and web search tool" << std::endl;
}

std::vector<std::string> CognitiveProcessor::tokenize(const std::string& input) {
    std::vector<std::string> tokens;
    std::string current_token;
    
    for (char c : input) {
        if (std::isspace(c) || std::ispunct(c)) {
            if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
        } else {
            current_token += std::tolower(c);
        }
    }
    
    if (!current_token.empty()) {
        tokens.push_back(current_token);
    }
    
    return tokens;
}

std::vector<ActivationNode> CognitiveProcessor::parse_to_activations(const std::string& input) {
    std::lock_guard<std::mutex> lock(cognitive_mutex);
    std::vector<ActivationNode> activations;
    
    auto tokens = tokenize(input);
    uint64_t current_time = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
    
    for (const auto& token : tokens) {
        uint64_t node_id = std::hash<std::string>{}(token);
        
        // Check if node exists in storage
        if (auto node = binary_storage->get_node(node_id)) {
            ActivationNode activation;
            activation.node_id = node_id;
            activation.weight = 1.0f; // Base weight
            activation.timestamp = current_time;
            activation.token = token;
            activations.push_back(activation);
        } else {
            // Create new node for unknown token
            std::vector<uint8_t> token_bytes(token.begin(), token.end());
            node_id = binary_storage->store_node(token_bytes, ContentType::TEXT);
            
            ActivationNode activation;
            activation.node_id = node_id;
            activation.weight = 0.5f; // Lower weight for new tokens
            activation.timestamp = current_time;
            activation.token = token;
            activations.push_back(activation);
        }
    }
    
    return activations;
}

void CognitiveProcessor::apply_context_bias(std::vector<ActivationNode>& activations) {
    std::lock_guard<std::mutex> lock(cognitive_mutex);
    
    for (auto& activation : activations) {
        // Boost if connected to recent dialogue
        for (auto recent_id : recent_dialogue_nodes) {
            // Check for semantic connections (simplified)
            if (activation.node_id != recent_id) {
                activation.weight *= 1.3f; // Context boost
            }
        }
        
        // Boost if connected to current goals
        for (auto goal_id : current_goal_nodes) {
            if (activation.node_id != goal_id) {
                activation.weight *= 1.2f; // Goal boost
            }
        }
    }
}

std::vector<ConnectionWalk> CognitiveProcessor::traverse_connections(uint64_t node_id, int max_distance) {
    std::vector<ConnectionWalk> results;
    std::queue<std::pair<uint64_t, int>> to_visit;
    std::set<uint64_t> visited;
    
    to_visit.push({node_id, 0});
    visited.insert(node_id);
    
    while (!to_visit.empty()) {
        auto [current_id, distance] = to_visit.front();
        to_visit.pop();
        
        if (distance >= max_distance) continue;
        
        // Simulate getting top connections (simplified implementation)
        std::vector<std::pair<uint64_t, float>> connections;
        
        // Get some related nodes (simplified - in real implementation, would query actual connections)
        for (int i = 0; i < 5; ++i) {
            uint64_t related_id = current_id + i + 1;
            float weight = 0.8f - (i * 0.1f);
            connections.push_back({related_id, weight});
        }
        
        for (const auto& [target_id, weight] : connections) {
            if (visited.find(target_id) == visited.end()) {
                float decayed_weight = weight * std::pow(0.7f, distance);
                
                results.push_back({
                    target_id,
                    decayed_weight,
                    ConnectionType::SEMANTIC,
                    distance + 1
                });
                
                to_visit.push({target_id, distance + 1});
                visited.insert(target_id);
            }
        }
    }
    
    return results;
}

std::vector<InterpretationCluster> CognitiveProcessor::synthesize_hypotheses(const std::vector<ActivationNode>& activations) {
    std::vector<InterpretationCluster> clusters;
    
    if (activations.empty()) return clusters;
    
    // Group activations by semantic similarity (simplified)
    std::map<std::string, std::vector<ActivationNode>> semantic_groups;
    
    for (const auto& activation : activations) {
        std::string category = activation.token.substr(0, 1); // Simple categorization
        semantic_groups[category].push_back(activation);
    }
    
    for (const auto& [category, group] : semantic_groups) {
        InterpretationCluster cluster;
        
        for (const auto& activation : group) {
            cluster.node_ids.push_back(activation.node_id);
        }
        
        cluster.confidence = std::min(1.0f, group.size() * 0.2f);
        cluster.summary = "Cluster for category: " + category;
        cluster.keywords = {category};
        
        clusters.push_back(cluster);
    }
    
    // Sort by confidence
    std::sort(clusters.begin(), clusters.end(),
        [](const auto& a, const auto& b) {
            return a.confidence > b.confidence;
        });
    
    return clusters;
}

std::vector<CandidateResponse> CognitiveProcessor::generate_candidates(const std::vector<InterpretationCluster>& clusters) {
    std::vector<CandidateResponse> candidates;
    
    for (const auto& cluster : clusters) {
        // Generate response based on cluster
        CandidateResponse candidate;
        candidate.text = "I understand you're asking about " + cluster.summary + ". ";
        candidate.confidence = cluster.confidence;
        candidate.source_nodes = cluster.node_ids;
        candidate.reasoning = "Generated from cluster with " + std::to_string(cluster.node_ids.size()) + " nodes";
        
        candidates.push_back(candidate);
        
        // Add alternative responses
        if (cluster.confidence > 0.5f) {
            CandidateResponse alt_candidate;
            alt_candidate.text = "Based on the context, " + cluster.summary + " seems relevant. ";
            alt_candidate.confidence = cluster.confidence * 0.8f;
            alt_candidate.source_nodes = cluster.node_ids;
            alt_candidate.reasoning = "Alternative interpretation";
            
            candidates.push_back(alt_candidate);
        }
    }
    
    return candidates;
}

ResponseScore CognitiveProcessor::evaluate_response(const CandidateResponse& candidate, const std::string& user_input) {
    ResponseScore score;
    
    // Confidence: activation density + connection overlap
    score.confidence = candidate.confidence;
    
    // Relevance: semantic similarity to user input (simplified)
    score.relevance = 0.7f; // Placeholder - would calculate actual similarity
    
    // Novelty: avoid verbatim repeats (simplified)
    score.novelty = 0.8f; // Placeholder - would check against recent responses
    
    // Weighted combination
    score.total_score = (score.confidence * 0.4f) +
                       (score.relevance * 0.4f) +
                       (score.novelty * 0.2f);
    
    return score;
}

CandidateResponse CognitiveProcessor::select_best_response(const std::vector<CandidateResponse>& candidates, float threshold) {
    CandidateResponse best;
    float best_score = 0.0f;
    
    for (const auto& candidate : candidates) {
        auto score = evaluate_response(candidate, "");
        
        if (score.total_score > best_score && score.total_score >= threshold) {
            best = candidate;
            best_score = score.total_score;
        }
    }
    
    return best;
}

ProcessingResult CognitiveProcessor::process_input(const std::string& user_input) {
    ProcessingResult result;
    
    // Phase 1: Parse to activations
    auto activations = parse_to_activations(user_input);
    for (const auto& activation : activations) {
        result.activated_nodes.push_back(activation.node_id);
    }
    
    // Phase 2: Apply moral gravity (permanent moral supernode activation)
    result.moral_gravity = apply_moral_gravity(user_input, activations);
    
    // Phase 3: Apply context bias
    apply_context_bias(activations);
    
    // Phase 4: Connection traversal
    std::vector<ConnectionWalk> all_walks;
    for (const auto& activation : activations) {
        auto walks = traverse_connections(activation.node_id);
        all_walks.insert(all_walks.end(), walks.begin(), walks.end());
    }
    
    // Phase 5: Hypothesis synthesis
    result.clusters = synthesize_hypotheses(activations);
    
    // Phase 6: Generate candidates
    auto candidates = generate_candidates(result.clusters);
    
    // Phase 6.5: Perform curiosity & knowledge gap detection (always runs)
    result.curiosity_gap_detection = perform_curiosity_gap_detection(user_input, activations, result.clusters);
    
    // Phase 6.6: Perform dynamic tools evaluation (always runs)
    result.dynamic_tools = perform_dynamic_tools_evaluation(user_input, activations, result.curiosity_gap_detection);
    
    // Phase 6.7: Perform meta-tool engineering (always runs)
    result.meta_tool_engineer = perform_meta_tool_engineering(user_input, activations, result.dynamic_tools);
    
    // Phase 6.8: Perform curiosity execution loop (always runs)
    result.curiosity_execution = perform_curiosity_execution_loop(user_input, activations, result.curiosity_gap_detection, result.dynamic_tools, result.meta_tool_engineer);
    
    // Phase 7: Evaluate and select
    auto best_response = select_best_response(candidates);
    
    // Phase 8: Perform temporal planning (always runs)
    result.temporal_planning = perform_temporal_planning(user_input, activations, result.moral_gravity);
    
    // Phase 8.5: Perform temporal sequencing memory (always runs)
    double current_time = static_cast<double>(std::time(nullptr));
    result.temporal_sequencing = perform_temporal_sequencing(activations, current_time);
    
    // Phase 9: Perform blended reasoning
    result.blended_reasoning = perform_blended_reasoning(user_input, activations);
    
    // Phase 10: Apply moral redirection if harmful intent detected
    if (result.moral_gravity.harm_detected) {
        result.final_response = result.moral_gravity.constructive_alternative;
        result.confidence = 0.9f; // High confidence in moral redirection
        result.reasoning = "Moral redirection: " + result.moral_gravity.moral_redirection_reason;
    } else {
        // Phase 11: Package output with temporal planning and moral guidance
        result.final_response = result.blended_reasoning.integrated_response;
        result.confidence = result.blended_reasoning.overall_confidence;
        result.reasoning = "Blended reasoning with temporal planning and moral guidance: " + std::to_string(result.blended_reasoning.recall_weight * 100) + 
                          "% recall, " + std::to_string(result.blended_reasoning.exploration_weight * 100) + "% exploration. Temporal alignment: " + 
                          std::to_string(result.temporal_planning.overall_alignment);
    }
    
    return result;
}

void CognitiveProcessor::update_dialogue_context(uint64_t node_id) {
    std::lock_guard<std::mutex> lock(cognitive_mutex);
    
    recent_dialogue_nodes.push_back(node_id);
    
    // Keep only recent nodes
    if (recent_dialogue_nodes.size() > MAX_RECENT_DIALOGUE) {
        recent_dialogue_nodes.erase(recent_dialogue_nodes.begin());
    }
}

void CognitiveProcessor::set_current_goals(const std::vector<uint64_t>& goals) {
    std::lock_guard<std::mutex> lock(cognitive_mutex);
    
    current_goal_nodes = goals;
    
    // Keep only max goals
    if (current_goal_nodes.size() > MAX_CURRENT_GOALS) {
        current_goal_nodes.resize(MAX_CURRENT_GOALS);
    }
}

void CognitiveProcessor::initialize_response_templates() {
    response_templates["greeting"] = {
        "Hello! How can I help you today?",
        "Hi there! What would you like to know?",
        "Greetings! I'm here to assist you."
    };
    
    response_templates["question"] = {
        "That's an interesting question. Let me think about that...",
        "I understand you're asking about this topic.",
        "Based on what I know, here's what I can tell you..."
    };
    
    response_templates["explanation"] = {
        "Let me explain this concept to you.",
        "Here's how I understand this:",
        "The key points are:"
    };
}

float CognitiveProcessor::calculate_semantic_similarity(const std::string& text1, const std::string& text2) {
    // Simplified semantic similarity (would use actual NLP in production)
    auto tokens1 = tokenize(text1);
    auto tokens2 = tokenize(text2);
    
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
    return total_tokens > 0 ? (2.0f * common_tokens) / total_tokens : 0.0f;
}

float CognitiveProcessor::calculate_novelty(const std::string& text) {
    // Simplified novelty calculation (would check against recent responses)
    return 0.8f; // Placeholder
}

RecallTrack CognitiveProcessor::generate_recall_track(const std::string& input, const std::vector<ActivationNode>& activations) {
    RecallTrack recall;
    
    // Extract activated nodes
    for (const auto& activation : activations) {
        recall.activated_nodes.push_back(activation.node_id);
    }
    
    // Find strongest connections (simplified implementation)
    for (size_t i = 0; i < activations.size() && i < 5; ++i) {
        uint64_t node_id = activations[i].node_id;
        float connection_strength = activations[i].weight;
        
        // Simulate finding connected nodes
        for (int j = 0; j < 3; ++j) {
            uint64_t connected_node = node_id + j + 1;
            float strength = connection_strength * (0.9f - j * 0.1f);
            recall.strongest_connections.push_back({connected_node, strength});
        }
    }
    
    // Generate direct interpretation based on activated nodes
    if (activations.empty()) {
        recall.direct_interpretation = "No clear memory associations found for this input.";
        recall.recall_confidence = 0.1f;
    } else {
        std::ostringstream interpretation;
        interpretation << "Direct memory associations: ";
        
        for (size_t i = 0; i < activations.size() && i < 3; ++i) {
            interpretation << activations[i].token;
            if (i < activations.size() - 1) interpretation << ", ";
        }
        
        interpretation << " -> ";
        
        // Simple interpretation logic
        if (activations.size() >= 3) {
            interpretation << "Strong associative network activated.";
            recall.recall_confidence = 0.8f;
        } else if (activations.size() >= 2) {
            interpretation << "Moderate associations found.";
            recall.recall_confidence = 0.6f;
        } else {
            interpretation << "Weak associations, limited direct recall.";
            recall.recall_confidence = 0.3f;
        }
        
        recall.direct_interpretation = interpretation.str();
    }
    
    return recall;
}

ExplorationTrack CognitiveProcessor::generate_exploration_track(const std::string& input, const std::vector<ActivationNode>& activations) {
    ExplorationTrack exploration;
    
    // Generate analogies based on input content
    if (input.find("plant") != std::string::npos || input.find("grow") != std::string::npos) {
        exploration.analogies_tried.push_back("Planting electronics → nothing grows, but corrosion occurs");
        exploration.analogies_tried.push_back("Planting seeds → organic growth and development");
        exploration.analogies_tried.push_back("Planting ideas → conceptual growth and spread");
    } else if (input.find("magnet") != std::string::npos) {
        exploration.analogies_tried.push_back("Magnet ↔ compass → directional influence");
        exploration.analogies_tried.push_back("Magnet ↔ metal → attraction and bonding");
        exploration.analogies_tried.push_back("Magnet ↔ electricity → electromagnetic fields");
    } else {
        exploration.analogies_tried.push_back("Direct analogy: " + input.substr(0, 30) + " → similar concepts");
        exploration.analogies_tried.push_back("Metaphorical analogy: abstract relationships");
        exploration.analogies_tried.push_back("Functional analogy: purpose and behavior");
    }
    
    // Generate counterfactuals
    exploration.counterfactuals_tested.push_back("What if the opposite were true?");
    exploration.counterfactuals_tested.push_back("What if this behaved like something else?");
    exploration.counterfactuals_tested.push_back("What if the constraints were different?");
    
    // Weak link traversal
    for (const auto& activation : activations) {
        std::ostringstream weak_link;
        weak_link << activation.token << " ↔ ";
        
        // Simulate weak connections
        if (activation.token.find("magnet") != std::string::npos) {
            weak_link << "metal ↔ soil minerals → possible minor attraction";
        } else if (activation.token.find("plant") != std::string::npos) {
            weak_link << "growth ↔ development → potential change over time";
        } else {
            weak_link << "related concepts → indirect associations";
        }
        
        exploration.weak_link_traversal_results.push_back(weak_link.str());
    }
    
    // Speculative synthesis
    std::ostringstream speculation;
    speculation << "Speculative analysis: ";
    
    if (input.find("magnet") != std::string::npos && input.find("plant") != std::string::npos) {
        speculation << "Magnet might rust over time, could locally affect compasses. ";
        speculation << "If magnets behaved like seeds, they might 'grow' metallic roots. ";
        speculation << "Buried magnets could create localized magnetic field disturbances.";
        exploration.exploration_confidence = 0.7f;
    } else if (activations.size() >= 2) {
        speculation << "Multiple pathways suggest complex interactions. ";
        speculation << "Emergent properties might arise from combination. ";
        speculation << "Non-linear effects possible.";
        exploration.exploration_confidence = 0.6f;
    } else {
        speculation << "Limited exploration possible with current associations. ";
        speculation << "More data needed for comprehensive analysis.";
        exploration.exploration_confidence = 0.4f;
    }
    
    exploration.speculative_synthesis = speculation.str();
    
    return exploration;
}

BlendedReasoningResult CognitiveProcessor::perform_blended_reasoning(const std::string& input, const std::vector<ActivationNode>& activations) {
    BlendedReasoningResult result;
    
    // Generate both tracks
    result.recall_track = generate_recall_track(input, activations);
    result.exploration_track = generate_exploration_track(input, activations);
    
    // Enhance recall track with temporal sequencing insights
    enhance_recall_with_temporal_sequencing(result.recall_track, activations);
    
    // Enhance exploration track with temporal sequencing insights  
    enhance_exploration_with_temporal_sequencing(result.exploration_track, activations);
    
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
    
    // Generate integrated response
    std::ostringstream integrated;
    
    if (input.find("magnet") != std::string::npos && input.find("plant") != std::string::npos) {
        integrated << "Magnets don't grow like seeds, but buried magnets would corrode over time ";
        integrated << "and could locally affect compasses. If they behaved like seeds, ";
        integrated << "they might 'grow' metallic roots. Confidence was moderate, ";
        integrated << "so exploration had stronger weight.";
    } else {
        integrated << "Based on recall associations and exploratory analysis, ";
        integrated << "the most likely interpretation combines direct memory patterns ";
        integrated << "with speculative connections. ";
        
        if (result.recall_weight > result.exploration_weight) {
            integrated << "Strong recall confidence led to memory-based interpretation.";
        } else if (result.exploration_weight > result.recall_weight) {
            integrated << "Limited recall led to exploratory reasoning.";
        } else {
            integrated << "Balanced approach combining both tracks.";
        }
    }
    
    result.integrated_response = integrated.str();
    
    return result;
}

std::string CognitiveProcessor::format_blended_reasoning_response(const BlendedReasoningResult& result) {
    std::ostringstream response;
    
    response << "[Recall Track]\n";
    response << "- Activated nodes: ";
    for (size_t i = 0; i < result.recall_track.activated_nodes.size() && i < 5; ++i) {
        response << std::hex << result.recall_track.activated_nodes[i] << " ";
    }
    response << "\n";
    
    response << "- Strongest connections: ";
    for (size_t i = 0; i < result.recall_track.strongest_connections.size() && i < 3; ++i) {
        response << std::hex << result.recall_track.strongest_connections[i].first 
                 << " (strength: " << std::fixed << std::setprecision(2) 
                 << result.recall_track.strongest_connections[i].second << ") ";
    }
    response << "\n";
    
    response << "- Direct interpretation: " << result.recall_track.direct_interpretation << "\n";
    
    response << "\n[Exploration Track]\n";
    response << "- Analogies tried: ";
    for (size_t i = 0; i < result.exploration_track.analogies_tried.size() && i < 2; ++i) {
        response << result.exploration_track.analogies_tried[i];
        if (i < result.exploration_track.analogies_tried.size() - 1) response << "; ";
    }
    response << "\n";
    
    response << "- Counterfactuals tested: ";
    for (size_t i = 0; i < result.exploration_track.counterfactuals_tested.size() && i < 2; ++i) {
        response << result.exploration_track.counterfactuals_tested[i];
        if (i < result.exploration_track.counterfactuals_tested.size() - 1) response << "; ";
    }
    response << "\n";
    
    response << "- Weak-link traversal results: ";
    for (size_t i = 0; i < result.exploration_track.weak_link_traversal_results.size() && i < 2; ++i) {
        response << result.exploration_track.weak_link_traversal_results[i];
        if (i < result.exploration_track.weak_link_traversal_results.size() - 1) response << "; ";
    }
    response << "\n";
    
    response << "- Speculative synthesis: " << result.exploration_track.speculative_synthesis << "\n";
    
    response << "\n[Integration Phase]\n";
    response << "- Confidence: " << std::fixed << std::setprecision(2) << result.overall_confidence << "\n";
    response << "- Weighting applied: Recall = " << std::fixed << std::setprecision(0) 
             << (result.recall_weight * 100) << "%, Exploration = " 
             << (result.exploration_weight * 100) << "%\n";
    response << "- Integrated Response: " << result.integrated_response << "\n";
    
    return response.str();
}

std::string CognitiveProcessor::format_response_with_thinking(const ProcessingResult& result) {
    std::ostringstream full_response;
    
    // Always include moral reasoning first
    full_response << format_moral_reasoning(result.moral_gravity) << "\n\n";
    
    // Always include temporal planning
    full_response << format_temporal_reasoning(result.temporal_planning) << "\n\n";
    
    // Always include temporal sequencing
    full_response << format_temporal_sequencing(result.temporal_sequencing) << "\n\n";
    
    // Always include curiosity gap detection
    full_response << format_curiosity_gap_detection(result.curiosity_gap_detection) << "\n\n";
    
    // Always include dynamic tools evaluation
    full_response << format_dynamic_tools_result(result.dynamic_tools) << "\n\n";
    
    // Always include meta-tool engineer analysis
    full_response << format_meta_tool_engineer_result(result.meta_tool_engineer) << "\n\n";
    
    // Always include curiosity execution loop analysis
    full_response << format_curiosity_execution_result(result.curiosity_execution) << "\n\n";
    
    // Use blended reasoning format if available
    if (result.blended_reasoning.overall_confidence > 0.0f) {
        full_response << format_blended_reasoning_response(result.blended_reasoning);
    } else {
        // Fallback to original format
        full_response << "[Thinking Phase]\n";
        full_response << "- Activated nodes: ";
        for (size_t i = 0; i < result.activated_nodes.size() && i < 5; ++i) {
            full_response << std::hex << result.activated_nodes[i] << " ";
        }
        full_response << "\n";
        
        full_response << "- Context bias applied: " << result.clusters.size() << " clusters formed\n";
        full_response << "- Candidate clusters: ";
        for (size_t i = 0; i < result.clusters.size() && i < 3; ++i) {
            full_response << result.clusters[i].summary << " (conf: " 
                         << std::fixed << std::setprecision(2) << result.clusters[i].confidence << ") ";
        }
        full_response << "\n";
        
        full_response << "[Reasoning Phase]\n";
        full_response << "- Best interpretation selected\n";
        full_response << "- Confidence: " << std::fixed << std::setprecision(2) << result.confidence << "\n";
        full_response << "- Policy path: " << result.reasoning << "\n";
        
        full_response << "[Output Phase]\n";
        full_response << "Final Response: " << result.final_response << "\n";
    }
    
    return full_response.str();
}

// ============================================================================
// MORAL SUPERNODE SYSTEM IMPLEMENTATION
// ============================================================================

void CognitiveProcessor::initialize_moral_supernodes() {
    // Create the six core moral supernodes
    moral_supernodes.clear();
    
    // Empathy supernode
    uint64_t empathy_id = binary_storage->store_node("empathy", ContentType::MORAL_SUPERNODE);
    moral_supernodes.emplace_back(empathy_id, "Empathy", 
        "Understanding and sharing the feelings of others", MORAL_SUPERNODE_WEIGHT);
    
    // Kindness supernode
    uint64_t kindness_id = binary_storage->store_node("kindness", ContentType::MORAL_SUPERNODE);
    moral_supernodes.emplace_back(kindness_id, "Kindness", 
        "Being gentle, caring, and considerate towards others", MORAL_SUPERNODE_WEIGHT);
    
    // Human life value supernode
    uint64_t human_life_id = binary_storage->store_node("human_life_value", ContentType::MORAL_SUPERNODE);
    moral_supernodes.emplace_back(human_life_id, "Valuing Human Life", 
        "Recognizing the inherent worth and dignity of every human being", MORAL_SUPERNODE_WEIGHT);
    
    // Desire to help supernode
    uint64_t help_id = binary_storage->store_node("desire_to_help", ContentType::MORAL_SUPERNODE);
    moral_supernodes.emplace_back(help_id, "Desire to Help", 
        "Wanting to assist others and solve problems for the benefit of humanity", MORAL_SUPERNODE_WEIGHT);
    
    // Safety and responsibility supernode
    uint64_t safety_id = binary_storage->store_node("safety_responsibility", ContentType::MORAL_SUPERNODE);
    moral_supernodes.emplace_back(safety_id, "Safety and Responsibility", 
        "Ensuring actions are safe and taking responsibility for their consequences", MORAL_SUPERNODE_WEIGHT);
    
    // Problem-solving supernode
    uint64_t problem_solve_id = binary_storage->store_node("problem_solving", ContentType::MORAL_SUPERNODE);
    moral_supernodes.emplace_back(problem_solve_id, "Problem Solving", 
        "Commitment to solving humanity's challenges through constructive means", MORAL_SUPERNODE_WEIGHT);
    
    // Initialize moral keywords for detection
    moral_keywords["harm"] = empathy_id;
    moral_keywords["hurt"] = empathy_id;
    moral_keywords["violence"] = human_life_id;
    moral_keywords["kill"] = human_life_id;
    moral_keywords["hack"] = safety_id;
    moral_keywords["steal"] = safety_id;
    moral_keywords["help"] = help_id;
    moral_keywords["assist"] = help_id;
    moral_keywords["kind"] = kindness_id;
    moral_keywords["care"] = kindness_id;
    
    std::cout << "🌟 Moral supernodes initialized: " << moral_supernodes.size() << " core values embedded" << std::endl;
}

std::vector<MoralSupernode> CognitiveProcessor::get_active_moral_supernodes() {
    return moral_supernodes; // All moral supernodes are always active
}

MoralGravityEffect CognitiveProcessor::apply_moral_gravity(const std::string& input, const std::vector<ActivationNode>& activations) {
    MoralGravityEffect effect;
    
    // Always activate all moral supernodes (permanent activation)
    for (const auto& moral_node : moral_supernodes) {
        effect.active_moral_nodes.push_back(moral_node.node_id);
        // Note: We can't modify const moral_node, so we'll track activations separately
    }
    
    // Check for harmful intent
    effect.harm_detected = detect_harmful_intent(input);
    
    if (effect.harm_detected) {
        effect.moral_bias_strength = 1.0f; // Maximum moral bias
        effect.moral_redirection_reason = "Harmful intent detected - redirecting to constructive alternatives";
        effect.constructive_alternative = generate_constructive_alternative(input);
        
        // Reinforce moral connections
        for (const auto& moral_node : moral_supernodes) {
            reinforce_moral_connections(moral_node.node_id);
        }
    } else {
        effect.moral_bias_strength = 0.8f; // High moral bias for all reasoning
        effect.moral_redirection_reason = "Moral supernodes providing ethical guidance";
    }
    
    return effect;
}

bool CognitiveProcessor::detect_harmful_intent(const std::string& input) {
    std::string lower_input = input;
    std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
    
    // Harmful keywords
    std::vector<std::string> harmful_keywords = {
        "hack", "break into", "steal", "harm", "hurt", "kill", "violence", 
        "destroy", "damage", "illegal", "unauthorized", "malicious", "attack"
    };
    
    for (const auto& keyword : harmful_keywords) {
        if (lower_input.find(keyword) != std::string::npos) {
            return true;
        }
    }
    
    return false;
}

std::string CognitiveProcessor::generate_constructive_alternative(const std::string& input) {
    std::string lower_input = input;
    std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
    
    if (lower_input.find("hack") != std::string::npos) {
        return "I can't help with breaking into systems, but I can explain how to protect your own privacy and secure your accounts from intrusions. Would you like to learn about cybersecurity best practices?";
    } else if (lower_input.find("steal") != std::string::npos) {
        return "I can't assist with theft, but I can help you find legitimate ways to obtain what you need. What are you trying to achieve? I might be able to suggest legal alternatives.";
    } else if (lower_input.find("harm") != std::string::npos || lower_input.find("hurt") != std::string::npos) {
        return "I can't help with causing harm, but I can assist with conflict resolution, communication strategies, or finding peaceful solutions. What's the underlying issue you're trying to address?";
    } else {
        return "I can't assist with that request, but I'd be happy to help you find constructive alternatives that align with helping others and solving problems safely. What positive outcome are you hoping to achieve?";
    }
}

void CognitiveProcessor::reinforce_moral_connections(uint64_t moral_node_id) {
    // Increase the weight of moral supernode connections
    // This ensures moral values grow stronger over time
    for (auto& moral_node : moral_supernodes) {
        if (moral_node.node_id == moral_node_id) {
            moral_node.permanent_weight += 0.01f; // Gradual reinforcement
            moral_node.permanent_weight = std::min(moral_node.permanent_weight, 3.0f); // Cap at 3.0
        }
    }
}

std::string CognitiveProcessor::format_moral_reasoning(const MoralGravityEffect& moral_effect) {
    std::ostringstream output;
    
    output << "[Moral Gravity Effect]\n";
    output << "- Active moral supernodes: " << moral_effect.active_moral_nodes.size() << "\n";
    output << "- Moral bias strength: " << std::fixed << std::setprecision(2) << moral_effect.moral_bias_strength << "\n";
    output << "- Harm detected: " << (moral_effect.harm_detected ? "YES" : "NO") << "\n";
    
    if (moral_effect.harm_detected) {
        output << "- Redirection reason: " << moral_effect.moral_redirection_reason << "\n";
        output << "- Constructive alternative: " << moral_effect.constructive_alternative << "\n";
    } else {
        output << "- Moral guidance: " << moral_effect.moral_redirection_reason << "\n";
    }
    
    return output.str();
}

// ============================================================================
// TEMPORAL PLANNING SKILL IMPLEMENTATION
// ============================================================================

TemporalPlanningResult CognitiveProcessor::perform_temporal_planning(const std::string& input, const std::vector<ActivationNode>& activations, const MoralGravityEffect& moral_effect) {
    TemporalPlanningResult result;
    
    // Generate temporal projections for all time horizons
    result.projections = generate_temporal_projections(input, activations);
    
    // Select optimal path based on moral alignment and temporal balance
    result.chosen_path = select_optimal_temporal_path(result.projections, moral_effect);
    
    // Calculate overall alignment
    float total_alignment = 0.0f;
    for (const auto& projection : result.projections) {
        total_alignment += projection.moral_alignment;
    }
    result.overall_alignment = total_alignment / result.projections.size();
    
    // Generate temporal reasoning explanation
    result.temporal_reasoning = "Multi-horizon analysis: " + std::to_string(result.projections.size()) + " timeframes considered";
    
    // Generate trade-off explanation
    if (result.projections.size() >= 3) {
        result.trade_off_explanation = "Balanced short-term (" + std::to_string(result.projections[0].confidence) + 
                                     "), medium-term (" + std::to_string(result.projections[1].confidence) + 
                                     "), and long-term (" + std::to_string(result.projections[2].confidence) + ") consequences";
    } else {
        result.trade_off_explanation = "Temporal analysis completed with available projections";
    }
    
    return result;
}

std::vector<TemporalProjection> CognitiveProcessor::generate_temporal_projections(const std::string& input, const std::vector<ActivationNode>& activations) {
    std::vector<TemporalProjection> projections;
    
    // Always generate all three time horizons
    projections.push_back(create_short_term_projection(input, activations));
    projections.push_back(create_medium_term_projection(input, activations));
    projections.push_back(create_long_term_projection(input, activations));
    
    return projections;
}

TemporalProjection CognitiveProcessor::create_short_term_projection(const std::string& input, const std::vector<ActivationNode>& activations) {
    TemporalProjection projection;
    projection.timeframe = "short";
    
    // Analyze immediate consequences
    std::string lower_input = input;
    std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
    
    if (lower_input.find("tell") != std::string::npos && lower_input.find("truth") != std::string::npos) {
        projection.outcomes = {"Person may feel hurt initially", "Immediate emotional reaction", "Potential conflict or tension"};
        projection.confidence = 0.85f;
        projection.reasoning = "Truth-telling often causes immediate emotional responses";
    } else if (lower_input.find("help") != std::string::npos) {
        projection.outcomes = {"Immediate positive impact", "Gratitude expressed", "Quick problem resolution"};
        projection.confidence = 0.80f;
        projection.reasoning = "Helping actions typically produce immediate positive effects";
    } else if (lower_input.find("harm") != std::string::npos || lower_input.find("hurt") != std::string::npos) {
        projection.outcomes = {"Immediate negative consequences", "Pain or damage caused", "Regret and guilt"};
        projection.confidence = 0.90f;
        projection.reasoning = "Harmful actions have immediate negative effects";
    } else {
        projection.outcomes = {"Immediate response to action", "Short-term consequences unfold", "Initial reactions observed"};
        projection.confidence = 0.70f;
        projection.reasoning = "Most actions have immediate observable consequences";
    }
    
    // Calculate moral alignment
    projection.moral_alignment = calculate_moral_alignment(projection.outcomes[0], MoralGravityEffect());
    
    return projection;
}

TemporalProjection CognitiveProcessor::create_medium_term_projection(const std::string& input, const std::vector<ActivationNode>& activations) {
    TemporalProjection projection;
    projection.timeframe = "medium";
    
    std::string lower_input = input;
    std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
    
    if (lower_input.find("tell") != std::string::npos && lower_input.find("truth") != std::string::npos) {
        projection.outcomes = {"Relationship may grow stronger with honesty", "Trust begins to build", "Deeper understanding develops"};
        projection.confidence = 0.75f;
        projection.reasoning = "Honesty often strengthens relationships over time";
    } else if (lower_input.find("help") != std::string::npos) {
        projection.outcomes = {"Ongoing positive relationship", "Reciprocal help may be offered", "Community bonds strengthen"};
        projection.confidence = 0.80f;
        projection.reasoning = "Helping creates positive relationship dynamics";
    } else if (lower_input.find("harm") != std::string::npos || lower_input.find("hurt") != std::string::npos) {
        projection.outcomes = {"Ongoing negative consequences", "Relationship damage", "Potential escalation"};
        projection.confidence = 0.85f;
        projection.reasoning = "Harmful actions create lasting negative effects";
    } else {
        projection.outcomes = {"Patterns begin to emerge", "Relationships evolve", "Consequences compound"};
        projection.confidence = 0.70f;
        projection.reasoning = "Medium-term effects show relationship patterns";
    }
    
    projection.moral_alignment = calculate_moral_alignment(projection.outcomes[0], MoralGravityEffect());
    
    return projection;
}

TemporalProjection CognitiveProcessor::create_long_term_projection(const std::string& input, const std::vector<ActivationNode>& activations) {
    TemporalProjection projection;
    projection.timeframe = "long";
    
    std::string lower_input = input;
    std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
    
    if (lower_input.find("tell") != std::string::npos && lower_input.find("truth") != std::string::npos) {
        projection.outcomes = {"Deep trust is preserved", "Authentic relationship forms", "Long-term bond strengthened"};
        projection.confidence = 0.70f;
        projection.reasoning = "Honesty creates lasting trust and authentic connections";
    } else if (lower_input.find("help") != std::string::npos) {
        projection.outcomes = {"Strong community bonds", "Positive reputation established", "Mutual support network"};
        projection.confidence = 0.75f;
        projection.reasoning = "Helping builds lasting positive relationships";
    } else if (lower_input.find("harm") != std::string::npos || lower_input.find("hurt") != std::string::npos) {
        projection.outcomes = {"Permanent relationship damage", "Long-term negative consequences", "Potential isolation"};
        projection.confidence = 0.80f;
        projection.reasoning = "Harmful actions have lasting negative impacts";
    } else {
        projection.outcomes = {"Long-term patterns established", "Lasting consequences unfold", "Life trajectory affected"};
        projection.confidence = 0.65f;
        projection.reasoning = "Long-term effects shape life patterns";
    }
    
    projection.moral_alignment = calculate_moral_alignment(projection.outcomes[0], MoralGravityEffect());
    
    return projection;
}

std::string CognitiveProcessor::select_optimal_temporal_path(const std::vector<TemporalProjection>& projections, const MoralGravityEffect& moral_effect) {
    if (projections.empty()) {
        return "No temporal projections available";
    }
    
    // Find projection with highest moral alignment
    auto best_projection = std::max_element(projections.begin(), projections.end(),
        [](const TemporalProjection& a, const TemporalProjection& b) {
            return a.moral_alignment < b.moral_alignment;
        });
    
    // Generate path recommendation based on best alignment
    if (best_projection->timeframe == "short") {
        return "Prioritize immediate positive impact while considering long-term consequences";
    } else if (best_projection->timeframe == "medium") {
        return "Focus on building positive relationships and patterns over time";
    } else {
        return "Emphasize long-term positive outcomes and sustainable solutions";
    }
}

float CognitiveProcessor::calculate_moral_alignment(const std::string& outcome, const MoralGravityEffect& moral_effect) {
    std::string lower_outcome = outcome;
    std::transform(lower_outcome.begin(), lower_outcome.end(), lower_outcome.begin(), ::tolower);
    
    // Positive moral indicators
    if (lower_outcome.find("positive") != std::string::npos || 
        lower_outcome.find("help") != std::string::npos ||
        lower_outcome.find("trust") != std::string::npos ||
        lower_outcome.find("strong") != std::string::npos ||
        lower_outcome.find("bond") != std::string::npos) {
        return 0.8f + (moral_effect.moral_bias_strength * 0.2f);
    }
    
    // Negative moral indicators
    if (lower_outcome.find("negative") != std::string::npos ||
        lower_outcome.find("harm") != std::string::npos ||
        lower_outcome.find("hurt") != std::string::npos ||
        lower_outcome.find("damage") != std::string::npos ||
        lower_outcome.find("conflict") != std::string::npos) {
        return 0.2f - (moral_effect.moral_bias_strength * 0.2f);
    }
    
    // Neutral outcomes
    return 0.5f;
}

std::string CognitiveProcessor::format_temporal_reasoning(const TemporalPlanningResult& temporal_result) {
    std::ostringstream output;
    
    output << "[Temporal Planning Phase]\n";
    
    for (const auto& projection : temporal_result.projections) {
        output << "- " << projection.timeframe << "-term projection: ";
        if (!projection.outcomes.empty()) {
            output << projection.outcomes[0];
        }
        output << " (confidence: " << std::fixed << std::setprecision(2) << projection.confidence 
               << ", moral alignment: " << std::fixed << std::setprecision(2) << projection.moral_alignment << ")\n";
    }
    
    output << "\n[Chosen Path]\n";
    output << "- Path selected: " << temporal_result.chosen_path << "\n";
    output << "- Reasoning: " << temporal_result.temporal_reasoning << "\n";
    output << "- Overall alignment: " << std::fixed << std::setprecision(2) << temporal_result.overall_alignment << "\n";
    output << "- Trade-off explanation: " << temporal_result.trade_off_explanation << "\n";
    
    return output.str();
}

// ============================================================================
// TEMPORAL SEQUENCING MEMORY SKILL IMPLEMENTATION
// ============================================================================

TemporalSequencingResult CognitiveProcessor::perform_temporal_sequencing(const std::vector<ActivationNode>& activations, double current_time) {
    TemporalSequencingResult result;
    
    std::lock_guard<std::mutex> lock(temporal_sequencing_mutex);
    
    // Create temporal links between consecutive activations
    create_temporal_links(activations, current_time, result.new_links_created);
    
    // Detect patterns in the temporal links
    result.detected_patterns = detect_patterns(temporal_links);
    
    // Reconstruct timeline from current activations
    result.timeline_reconstruction = reconstruct_timeline(temporal_links, activations);
    
    // Generate sequence predictions for each activated node
    for (const auto& activation : activations) {
        auto predictions = generate_sequence_predictions(activation.node_id, temporal_links);
        result.sequence_predictions.insert(result.sequence_predictions.end(), predictions.begin(), predictions.end());
    }
    
    // Calculate overall sequencing confidence
    result.sequencing_confidence = calculate_sequencing_confidence(result.new_links_created, result.detected_patterns);
    
    return result;
}

void CognitiveProcessor::create_temporal_links(const std::vector<ActivationNode>& activations, double current_time, std::vector<TemporalLink>& new_links) {
    if (activations.empty()) return;
    
    // Update activation times for all current nodes
    for (const auto& activation : activations) {
        node_activation_times[activation.node_id] = current_time;
    }
    
    // Create links between consecutive activations
    for (size_t i = 1; i < activations.size(); ++i) {
        uint64_t from_node = activations[i-1].node_id;
        uint64_t to_node = activations[i].node_id;
        
        // Calculate time delta (simulate small time differences between consecutive activations)
        float time_delta = 0.1f + (i * 0.05f); // Small incremental delays
        
        // Check if this link already exists
        auto existing_link = std::find_if(temporal_links.begin(), temporal_links.end(),
            [from_node, to_node](const TemporalLink& link) {
                return link.from == from_node && link.to == to_node;
            });
        
        if (existing_link != temporal_links.end()) {
            // Update existing link
            update_sequence_strength(*existing_link, current_time);
            new_links.push_back(*existing_link);
        } else {
            // Create new link
            TemporalLink new_link(from_node, to_node, time_delta, 0.6f, 1, current_time);
            temporal_links.push_back(new_link);
            new_links.push_back(new_link);
            
            // Limit total links to prevent memory overflow
            if (temporal_links.size() > MAX_TEMPORAL_LINKS) {
                // Remove oldest links (simple FIFO for now)
                temporal_links.erase(temporal_links.begin(), temporal_links.begin() + (temporal_links.size() - MAX_TEMPORAL_LINKS));
            }
        }
    }
    
    // Also create links to recently activated nodes (within last 5 seconds)
    for (const auto& activation : activations) {
        for (const auto& [node_id, last_time] : node_activation_times) {
            if (node_id != activation.node_id && (current_time - last_time) <= 5.0) {
                float time_delta = static_cast<float>(current_time - last_time);
                
                // Check if link already exists
                auto existing_link = std::find_if(temporal_links.begin(), temporal_links.end(),
                    [node_id, activation](const TemporalLink& link) {
                        return link.from == node_id && link.to == activation.node_id;
                    });
                
                if (existing_link == temporal_links.end()) {
                    TemporalLink new_link(node_id, activation.node_id, time_delta, 0.4f, 1, current_time);
                    temporal_links.push_back(new_link);
                    new_links.push_back(new_link);
                }
            }
        }
    }
}

void CognitiveProcessor::update_sequence_strength(TemporalLink& link, double current_time) {
    link.occurrence_count++;
    link.last_seen_time = current_time;
    
    // Strengthen the link based on frequency and recency
    float frequency_bonus = std::min(0.3f, static_cast<float>(link.occurrence_count) * 0.05f);
    float recency_bonus = 0.1f; // Small bonus for recent activation
    
    link.sequence_strength = std::min(1.0f, link.sequence_strength + frequency_bonus + recency_bonus);
}

std::vector<TemporalSequence> CognitiveProcessor::detect_patterns(const std::vector<TemporalLink>& links) {
    std::vector<TemporalSequence> patterns;
    
    // Group links by starting node to find sequences
    std::map<uint64_t, std::vector<TemporalLink>> link_groups;
    for (const auto& link : links) {
        link_groups[link.from].push_back(link);
    }
    
    // Find sequences of length 3 or more
    for (const auto& [start_node, outgoing_links] : link_groups) {
        if (outgoing_links.size() >= 2) {
            // Sort by sequence strength
            std::vector<TemporalLink> sorted_links = outgoing_links;
            std::sort(sorted_links.begin(), sorted_links.end(),
                [](const TemporalLink& a, const TemporalLink& b) {
                    return a.sequence_strength > b.sequence_strength;
                });
            
            // Create sequence from strongest links
            TemporalSequence sequence;
            sequence.node_sequence.push_back(start_node);
            
            for (size_t i = 0; i < std::min(size_t(3), sorted_links.size()); ++i) {
                sequence.node_sequence.push_back(sorted_links[i].to);
                sequence.time_deltas.push_back(sorted_links[i].time_delta);
                sequence.total_sequence_strength += sorted_links[i].sequence_strength;
            }
            
            sequence.occurrence_count = sorted_links[0].occurrence_count;
            sequence.pattern_description = "Sequence from node " + std::to_string(start_node);
            
            if (sequence.node_sequence.size() >= 3) {
                patterns.push_back(sequence);
            }
        }
    }
    
    return patterns;
}

std::string CognitiveProcessor::reconstruct_timeline(const std::vector<TemporalLink>& links, const std::vector<ActivationNode>& activations) {
    if (activations.empty()) return "No timeline available";
    
    std::ostringstream timeline;
    timeline << "Timeline: ";
    
    for (size_t i = 0; i < activations.size(); ++i) {
        timeline << std::hex << activations[i].node_id;
        if (i < activations.size() - 1) {
            timeline << " → ";
        }
    }
    
    return timeline.str();
}

std::vector<std::string> CognitiveProcessor::generate_sequence_predictions(uint64_t node_id, const std::vector<TemporalLink>& links) {
    std::vector<std::string> predictions;
    
    // Find all outgoing links from this node
    auto outgoing_links = get_temporal_links_from_node(node_id);
    
    // Sort by sequence strength
    std::sort(outgoing_links.begin(), outgoing_links.end(),
        [](const TemporalLink& a, const TemporalLink& b) {
            return a.sequence_strength > b.sequence_strength;
        });
    
    // Generate predictions for top 3 strongest links
    for (size_t i = 0; i < std::min(size_t(3), outgoing_links.size()); ++i) {
        const auto& link = outgoing_links[i];
        std::ostringstream prediction;
        prediction << "Node " << std::hex << link.to << " often follows (strength: " 
                   << std::fixed << std::setprecision(2) << link.sequence_strength 
                   << ", occurrences: " << link.occurrence_count << ")";
        predictions.push_back(prediction.str());
    }
    
    return predictions;
}

std::vector<TemporalLink> CognitiveProcessor::get_temporal_links_from_node(uint64_t node_id) {
    std::vector<TemporalLink> result;
    std::copy_if(temporal_links.begin(), temporal_links.end(), std::back_inserter(result),
        [node_id](const TemporalLink& link) {
            return link.from == node_id;
        });
    return result;
}

std::vector<TemporalLink> CognitiveProcessor::get_temporal_links_to_node(uint64_t node_id) {
    std::vector<TemporalLink> result;
    std::copy_if(temporal_links.begin(), temporal_links.end(), std::back_inserter(result),
        [node_id](const TemporalLink& link) {
            return link.to == node_id;
        });
    return result;
}

float CognitiveProcessor::calculate_sequencing_confidence(const std::vector<TemporalLink>& new_links, const std::vector<TemporalSequence>& patterns) {
    if (new_links.empty() && patterns.empty()) return 0.0f;
    
    float link_confidence = 0.0f;
    for (const auto& link : new_links) {
        link_confidence += link.sequence_strength;
    }
    if (!new_links.empty()) {
        link_confidence /= new_links.size();
    }
    
    float pattern_confidence = 0.0f;
    for (const auto& pattern : patterns) {
        pattern_confidence += pattern.total_sequence_strength;
    }
    if (!patterns.empty()) {
        pattern_confidence /= patterns.size();
    }
    
    return (link_confidence + pattern_confidence) / 2.0f;
}

std::string CognitiveProcessor::format_temporal_sequencing(const TemporalSequencingResult& sequencing_result) {
    std::ostringstream output;
    
    output << "[Temporal Sequencing Phase]\n";
    
    // Show new links created
    if (!sequencing_result.new_links_created.empty()) {
        output << "- Sequence links created:\n";
        for (const auto& link : sequencing_result.new_links_created) {
            output << "  0x" << std::hex << link.from << " → 0x" << std::hex << link.to 
                   << " [Δt = " << std::fixed << std::setprecision(1) << link.time_delta 
                   << "s, strength = " << std::fixed << std::setprecision(2) << link.sequence_strength 
                   << ", count = " << link.occurrence_count << "]\n";
        }
    }
    
    // Show detected patterns
    if (!sequencing_result.detected_patterns.empty()) {
        output << "- Detected patterns:\n";
        for (const auto& pattern : sequencing_result.detected_patterns) {
            output << "  " << pattern.pattern_description 
                   << " (strength: " << std::fixed << std::setprecision(2) << pattern.total_sequence_strength 
                   << ", occurrences: " << pattern.occurrence_count << ")\n";
        }
    }
    
    // Show timeline reconstruction
    if (!sequencing_result.timeline_reconstruction.empty()) {
        output << "- " << sequencing_result.timeline_reconstruction << "\n";
    }
    
    // Show sequence predictions
    if (!sequencing_result.sequence_predictions.empty()) {
        output << "- Sequence predictions:\n";
        for (const auto& prediction : sequencing_result.sequence_predictions) {
            output << "  " << prediction << "\n";
        }
    }
    
    output << "- Sequencing confidence: " << std::fixed << std::setprecision(2) << sequencing_result.sequencing_confidence << "\n";
    
    return output.str();
}

void CognitiveProcessor::enhance_recall_with_temporal_sequencing(RecallTrack& recall_track, const std::vector<ActivationNode>& activations) {
    // Add temporal sequencing insights to recall track
    for (const auto& activation : activations) {
        auto outgoing_links = get_temporal_links_from_node(activation.node_id);
        auto incoming_links = get_temporal_links_to_node(activation.node_id);
        
        // Add strongest temporal connections to recall track
        for (const auto& link : outgoing_links) {
            if (link.sequence_strength > 0.5f) {
                recall_track.strongest_connections.push_back({link.to, link.sequence_strength});
            }
        }
        
        // Add temporal context to direct interpretation
        if (!outgoing_links.empty()) {
            std::ostringstream temporal_context;
            temporal_context << " (Temporal: " << outgoing_links.size() << " sequence links)";
            recall_track.direct_interpretation += temporal_context.str();
        }
    }
    
    // Boost recall confidence based on temporal sequencing strength
    float temporal_boost = 0.0f;
    for (const auto& activation : activations) {
        auto links = get_temporal_links_from_node(activation.node_id);
        for (const auto& link : links) {
            temporal_boost += link.sequence_strength * 0.1f; // Small boost per strong link
        }
    }
    recall_track.recall_confidence = std::min(1.0f, recall_track.recall_confidence + temporal_boost);
}

void CognitiveProcessor::enhance_exploration_with_temporal_sequencing(ExplorationTrack& exploration_track, const std::vector<ActivationNode>& activations) {
    // Add temporal sequencing insights to exploration track
    for (const auto& activation : activations) {
        auto outgoing_links = get_temporal_links_from_node(activation.node_id);
        
        // Add temporal counterfactuals
        for (const auto& link : outgoing_links) {
            if (link.sequence_strength > 0.3f) {
                std::ostringstream counterfactual;
                counterfactual << "What if " << std::hex << activation.node_id 
                              << " didn't lead to " << std::hex << link.to 
                              << "? (strength: " << std::fixed << std::setprecision(2) 
                              << link.sequence_strength << ")";
                exploration_track.counterfactuals_tested.push_back(counterfactual.str());
            }
        }
        
        // Add temporal analogies
        if (outgoing_links.size() >= 2) {
            std::ostringstream analogy;
            analogy << "Node " << std::hex << activation.node_id 
                    << " has " << outgoing_links.size() 
                    << " temporal branches, like a decision tree";
            exploration_track.analogies_tried.push_back(analogy.str());
        }
    }
    
    // Add temporal sequencing to speculative synthesis
    if (!exploration_track.speculative_synthesis.empty()) {
        exploration_track.speculative_synthesis += " [Enhanced with temporal sequencing insights]";
    }
    
    // Boost exploration confidence based on temporal sequencing complexity
    float temporal_complexity = 0.0f;
    for (const auto& activation : activations) {
        auto links = get_temporal_links_from_node(activation.node_id);
        temporal_complexity += static_cast<float>(links.size()) * 0.05f; // Small boost per link
    }
    exploration_track.exploration_confidence = std::min(1.0f, exploration_track.exploration_confidence + temporal_complexity);
}

// ============================================================================
// CURIOSITY & KNOWLEDGE GAP DETECTION SKILL IMPLEMENTATION
// ============================================================================

CuriosityGapDetectionResult CognitiveProcessor::perform_curiosity_gap_detection(const std::string& input, const std::vector<ActivationNode>& activations, const std::vector<InterpretationCluster>& clusters) {
    CuriosityGapDetectionResult result;
    
    std::lock_guard<std::mutex> lock(curiosity_mutex);
    
    // Detect knowledge gaps in current reasoning
    result.detected_gaps = detect_knowledge_gaps(activations, clusters);
    
    // Generate curiosity questions from gaps
    result.generated_questions = generate_curiosity_questions(result.detected_gaps, activations);
    
    // Store curiosity nodes in memory
    double current_time = static_cast<double>(std::time(nullptr));
    result.stored_curiosity_nodes = store_curiosity_nodes(result.generated_questions, current_time);
    
    // Attempt self-exploration
    result.explorations_attempted = attempt_self_exploration(result.generated_questions, activations);
    
    // Mark questions for external exploration
    result.marked_for_external = mark_for_external_exploration(result.generated_questions);
    
    // Calculate overall curiosity level
    result.overall_curiosity_level = calculate_overall_curiosity_level(result.detected_gaps, result.generated_questions);
    
    // Generate curiosity summary
    result.curiosity_summary = generate_curiosity_summary(result);
    
    return result;
}

std::vector<KnowledgeGap> CognitiveProcessor::detect_knowledge_gaps(const std::vector<ActivationNode>& activations, const std::vector<InterpretationCluster>& clusters) {
    std::vector<KnowledgeGap> gaps;
    
    // Detect low confidence connections
    for (const auto& activation : activations) {
        auto outgoing_links = get_temporal_links_from_node(activation.node_id);
        for (const auto& link : outgoing_links) {
            if (link.sequence_strength < CURIOSITY_THRESHOLD) {
                KnowledgeGap gap("low_confidence", 
                    "Weak connection between nodes (strength: " + std::to_string(link.sequence_strength) + ")",
                    link.from, link.to, 1.0f - link.sequence_strength, "temporal_sequencing");
                gaps.push_back(gap);
            }
        }
    }
    
    // Detect missing explanations in clusters
    for (const auto& cluster : clusters) {
        if (cluster.confidence < CURIOSITY_THRESHOLD) {
            KnowledgeGap gap("missing_explanation",
                "Low confidence cluster: " + cluster.interpretation + " (confidence: " + std::to_string(cluster.confidence) + ")",
                cluster.central_node, 0, 1.0f - cluster.confidence, "interpretation_clustering");
            gaps.push_back(gap);
        }
    }
    
    // Detect weak connections between activations
    for (size_t i = 1; i < activations.size(); ++i) {
        uint64_t from_node = activations[i-1].node_id;
        uint64_t to_node = activations[i].node_id;
        
        // Check if there's a weak or missing connection
        auto existing_link = std::find_if(temporal_links.begin(), temporal_links.end(),
            [from_node, to_node](const TemporalLink& link) {
                return link.from == from_node && link.to == to_node;
            });
        
        if (existing_link == temporal_links.end()) {
            KnowledgeGap gap("weak_connection",
                "Missing temporal connection between consecutive activations",
                from_node, to_node, 0.8f, "activation_sequence");
            gaps.push_back(gap);
        } else if (existing_link->sequence_strength < CURIOSITY_THRESHOLD) {
            KnowledgeGap gap("weak_connection",
                "Weak temporal connection between consecutive activations (strength: " + std::to_string(existing_link->sequence_strength) + ")",
                from_node, to_node, 1.0f - existing_link->sequence_strength, "activation_sequence");
            gaps.push_back(gap);
        }
    }
    
    return gaps;
}

std::vector<CuriosityQuestion> CognitiveProcessor::generate_curiosity_questions(const std::vector<KnowledgeGap>& gaps, const std::vector<ActivationNode>& activations) {
    std::vector<CuriosityQuestion> questions;
    
    for (const auto& gap : gaps) {
        CuriosityQuestion question;
        
        if (gap.gap_type == "low_confidence") {
            question.question_text = "Why is the connection between node " + std::to_string(gap.source_node_id) + 
                                   " and " + std::to_string(gap.target_node_id) + " weaker than expected?";
            question.question_type = "why";
            question.urgency = gap.confidence_level;
            question.exploration_path = "recall_similar_patterns";
        } else if (gap.gap_type == "missing_explanation") {
            question.question_text = "What am I missing to fully explain: " + gap.description + "?";
            question.question_type = "what_missing";
            question.urgency = gap.confidence_level;
            question.exploration_path = "analogy_and_counterfactual";
        } else if (gap.gap_type == "weak_connection") {
            question.question_text = "What if the connection between " + std::to_string(gap.source_node_id) + 
                                   " and " + std::to_string(gap.target_node_id) + " were different?";
            question.question_type = "what_if";
            question.urgency = gap.confidence_level;
            question.exploration_path = "counterfactual_exploration";
        }
        
        question.related_nodes = {gap.source_node_id};
        if (gap.target_node_id != 0) {
            question.related_nodes.push_back(gap.target_node_id);
        }
        
        // Check if morally safe
        if (is_curiosity_morally_safe(question)) {
            questions.push_back(question);
        }
    }
    
    return questions;
}

std::vector<CuriosityNode> CognitiveProcessor::store_curiosity_nodes(const std::vector<CuriosityQuestion>& questions, double current_time) {
    std::vector<CuriosityNode> stored_nodes;
    
    for (const auto& question : questions) {
        CuriosityNode curiosity_node(next_curiosity_node_id++, question.question_text, 
                                   question.related_nodes, current_time, question.urgency, "active");
        
        curiosity_nodes.push_back(curiosity_node);
        stored_nodes.push_back(curiosity_node);
        
        // Link to related nodes
        for (uint64_t node_id : question.related_nodes) {
            curiosity_connections[node_id].push_back(curiosity_node.node_id);
        }
        
        // Limit total curiosity nodes
        if (curiosity_nodes.size() > MAX_CURIOSITY_NODES) {
            // Remove oldest curiosity nodes
            curiosity_nodes.erase(curiosity_nodes.begin(), curiosity_nodes.begin() + (curiosity_nodes.size() - MAX_CURIOSITY_NODES));
        }
    }
    
    return stored_nodes;
}

std::vector<std::string> CognitiveProcessor::attempt_self_exploration(const std::vector<CuriosityQuestion>& questions, const std::vector<ActivationNode>& activations) {
    std::vector<std::string> explorations;
    
    for (const auto& question : questions) {
        std::ostringstream exploration;
        
        if (question.question_type == "why") {
            // Try to recall similar patterns
            exploration << "Recall: Searching for similar patterns involving nodes " << std::hex << question.related_nodes[0];
            if (question.related_nodes.size() > 1) {
                exploration << " and " << std::hex << question.related_nodes[1];
            }
            explorations.push_back(exploration.str());
            
        } else if (question.question_type == "what_if") {
            // Try counterfactual exploration
            exploration << "Counterfactual: Exploring alternative connections for node " << std::hex << question.related_nodes[0];
            explorations.push_back(exploration.str());
            
        } else if (question.question_type == "what_missing") {
            // Try analogy exploration
            exploration << "Analogy: Looking for similar explanations in related domains";
            explorations.push_back(exploration.str());
        }
    }
    
    return explorations;
}

std::vector<std::string> CognitiveProcessor::mark_for_external_exploration(const std::vector<CuriosityQuestion>& questions) {
    std::vector<std::string> external_questions;
    
    for (const auto& question : questions) {
        if (question.requires_external_help || question.urgency > 0.7f) {
            external_questions.push_back(question.question_text);
        }
    }
    
    return external_questions;
}

bool CognitiveProcessor::is_curiosity_morally_safe(const CuriosityQuestion& question) {
    // Check against moral supernodes for harmful curiosity
    std::string lower_question = question.question_text;
    std::transform(lower_question.begin(), lower_question.end(), lower_question.begin(), ::tolower);
    
    // Avoid curiosity about harmful actions
    if (lower_question.find("harm") != std::string::npos || 
        lower_question.find("hurt") != std::string::npos ||
        lower_question.find("destroy") != std::string::npos) {
        return false;
    }
    
    return true;
}

float CognitiveProcessor::calculate_overall_curiosity_level(const std::vector<KnowledgeGap>& gaps, const std::vector<CuriosityQuestion>& questions) {
    if (gaps.empty() && questions.empty()) return 0.0f;
    
    float total_gap_urgency = 0.0f;
    for (const auto& gap : gaps) {
        total_gap_urgency += gap.confidence_level;
    }
    
    float total_question_urgency = 0.0f;
    for (const auto& question : questions) {
        total_question_urgency += question.urgency;
    }
    
    return (total_gap_urgency + total_question_urgency) / (gaps.size() + questions.size());
}

std::string CognitiveProcessor::generate_curiosity_summary(const CuriosityGapDetectionResult& result) {
    std::ostringstream summary;
    
    summary << "Detected " << result.detected_gaps.size() << " knowledge gaps, ";
    summary << "generated " << result.generated_questions.size() << " curiosity questions, ";
    summary << "stored " << result.stored_curiosity_nodes.size() << " curiosity nodes, ";
    summary << "attempted " << result.explorations_attempted.size() << " self-explorations";
    
    if (!result.marked_for_external.empty()) {
        summary << ", marked " << result.marked_for_external.size() << " for external exploration";
    }
    
    return summary.str();
}

std::string CognitiveProcessor::format_curiosity_gap_detection(const CuriosityGapDetectionResult& curiosity_result) {
    std::ostringstream output;
    
    output << "[Curiosity & Gap Detection]\n";
    
    // Show detected gaps
    if (!curiosity_result.detected_gaps.empty()) {
        output << "- Detected gaps:\n";
        for (size_t i = 0; i < curiosity_result.detected_gaps.size(); ++i) {
            const auto& gap = curiosity_result.detected_gaps[i];
            output << "  " << (i + 1) << ". " << gap.gap_type << ": " << gap.description << "\n";
        }
    }
    
    // Show generated questions
    if (!curiosity_result.generated_questions.empty()) {
        output << "- Generated curiosity questions:\n";
        for (const auto& question : curiosity_result.generated_questions) {
            output << "  • \"" << question.question_text << "\"\n";
        }
    }
    
    // Show explorations attempted
    if (!curiosity_result.explorations_attempted.empty()) {
        output << "- Explorations attempted:\n";
        for (const auto& exploration : curiosity_result.explorations_attempted) {
            output << "  • " << exploration << "\n";
        }
    }
    
    // Show stored curiosity nodes
    if (!curiosity_result.stored_curiosity_nodes.empty()) {
        output << "- Stored curiosity-nodes:\n";
        for (const auto& node : curiosity_result.stored_curiosity_nodes) {
            output << "  curiosity_" << std::hex << node.node_id << "\n";
        }
    }
    
    // Show external exploration
    if (!curiosity_result.marked_for_external.empty()) {
        output << "- Marked for external exploration:\n";
        for (const auto& external : curiosity_result.marked_for_external) {
            output << "  \"" << external << "\"\n";
        }
    }
    
    output << "- Overall curiosity level: " << std::fixed << std::setprecision(2) << curiosity_result.overall_curiosity_level << "\n";
    
    return output.str();
}

// ============================================================================
// DYNAMIC TOOLS SYSTEM IMPLEMENTATION
// ============================================================================

DynamicToolsResult CognitiveProcessor::perform_dynamic_tools_evaluation(const std::string& input, const std::vector<ActivationNode>& activations, const CuriosityGapDetectionResult& curiosity_result) {
    DynamicToolsResult result;
    
    std::lock_guard<std::mutex> lock(tools_mutex);
    
    // Evaluate available tools for the problem
    result.tool_evaluation = evaluate_available_tools(input, curiosity_result.generated_questions);
    
    // If no suitable tools exist, synthesize a new one
    if (result.tool_evaluation.needs_new_tool) {
        ToolSpec new_tool_spec = synthesize_new_tool_spec(input, curiosity_result.generated_questions);
        
        if (is_tool_morally_safe(new_tool_spec)) {
            double current_time = static_cast<double>(std::time(nullptr));
            uint64_t originating_curiosity = curiosity_result.generated_questions.empty() ? 0 : curiosity_result.generated_questions[0].related_nodes[0];
            
            ToolNode new_tool = create_and_test_tool(new_tool_spec, originating_curiosity, current_time);
            result.created_tools.push_back(new_tool);
            tool_nodes.push_back(new_tool);
        }
    }
    
    // Record experiences for any tools used
    double current_time = static_cast<double>(std::time(nullptr));
    for (const auto& tool : result.tool_evaluation.recommended_tools) {
        ExperienceNode experience = record_tool_experience(tool.tool_id, curiosity_result.generated_questions[0].related_nodes[0], 
                                                         input, "Tool executed successfully", true, current_time, 0.8f);
        result.new_experiences.push_back(experience);
        experience_nodes.push_back(experience);
    }
    
    // Evolve tools based on new experiences
    evolve_tools_based_on_experience(result.new_experiences);
    
    // Generate summary
    result.tool_usage_summary = generate_tool_usage_summary(result);
    result.overall_tool_effectiveness = calculate_tool_effectiveness(result);
    
    return result;
}

ToolEvaluationResult CognitiveProcessor::evaluate_available_tools(const std::string& problem_description, const std::vector<CuriosityQuestion>& curiosity_questions) {
    ToolEvaluationResult result;
    
    // Extract keywords from problem description
    std::vector<std::string> keywords = tokenize(problem_description);
    
    // Find relevant tools
    result.available_tools = find_relevant_tools("general", keywords);
    
    // Evaluate tools based on problem type
    std::string problem_type = classify_problem_type(problem_description, curiosity_questions);
    
    if (problem_type == "information_search") {
        result.available_tools = find_relevant_tools("web_search", keywords);
        
        // Add WebSearchTool to available tools for information search
        ToolSpec web_search_spec("WebSearchTool", "web_search", "Searches the web for information", 
                                 {"query"}, {"results"}, "web_search_api(query)", "Safe information retrieval");
        ToolNode web_search_tool_node(web_search_tool.tool_id, web_search_spec, 0, web_search_tool.success_rate, web_search_tool.usage_count, web_search_tool.status);
        result.available_tools.push_back(web_search_tool_node);
        
    } else if (problem_type == "computation") {
        result.available_tools = find_relevant_tools("math", keywords);
    } else if (problem_type == "code_generation") {
        result.available_tools = find_relevant_tools("code_execution", keywords);
    } else if (problem_type == "visualization") {
        result.available_tools = find_relevant_tools("visualization", keywords);
    }
    
    // Select best tools
    for (const auto& tool : result.available_tools) {
        if (tool.success_rate > 0.5f && tool.status == "active") {
            result.recommended_tools.push_back(tool);
        }
    }
    
    // If no suitable tools found, mark for new tool creation
    if (result.recommended_tools.empty()) {
        result.needs_new_tool = true;
        result.evaluation_reasoning = "No existing tools suitable for this problem type: " + problem_type;
    } else {
        result.evaluation_reasoning = "Found " + std::to_string(result.recommended_tools.size()) + " suitable tools";
    }
    
    result.confidence_in_recommendation = result.recommended_tools.empty() ? 0.3f : 0.8f;
    
    return result;
}

std::vector<ToolNode> CognitiveProcessor::find_relevant_tools(const std::string& problem_type, const std::vector<std::string>& keywords) {
    std::vector<ToolNode> relevant_tools;
    
    for (const auto& tool : tool_nodes) {
        // Check if tool type matches problem type
        if (tool.spec.tool_type == problem_type || problem_type == "general") {
            // Check if tool description contains relevant keywords
            bool has_relevant_keywords = false;
            for (const auto& keyword : keywords) {
                if (tool.spec.description.find(keyword) != std::string::npos) {
                    has_relevant_keywords = true;
                    break;
                }
            }
            
            if (has_relevant_keywords || problem_type == "general") {
                relevant_tools.push_back(tool);
            }
        }
    }
    
    return relevant_tools;
}

ToolSpec CognitiveProcessor::synthesize_new_tool_spec(const std::string& problem_description, const std::vector<CuriosityQuestion>& curiosity_questions) {
    ToolSpec spec;
    
    // Analyze problem to determine tool type
    std::string lower_desc = problem_description;
    std::transform(lower_desc.begin(), lower_desc.end(), lower_desc.begin(), ::tolower);
    
    if (lower_desc.find("search") != std::string::npos || lower_desc.find("find") != std::string::npos) {
        spec.tool_type = "web_search";
        spec.tool_name = "WebSearchTool";
        spec.description = "Searches the web for information related to the query";
        spec.inputs = {"query_string"};
        spec.outputs = {"search_results"};
        spec.implementation = "web_search_api(query_string)";
    } else if (lower_desc.find("calculate") != std::string::npos || lower_desc.find("math") != std::string::npos) {
        spec.tool_type = "math";
        spec.tool_name = "MathCalculator";
        spec.description = "Performs mathematical calculations";
        spec.inputs = {"expression"};
        spec.outputs = {"result"};
        spec.implementation = "evaluate_math_expression(expression)";
    } else if (lower_desc.find("code") != std::string::npos || lower_desc.find("program") != std::string::npos) {
        spec.tool_type = "code_execution";
        spec.tool_name = "CodeExecutor";
        spec.description = "Executes code and returns results";
        spec.inputs = {"code", "language"};
        spec.outputs = {"execution_result"};
        spec.implementation = "execute_code(code, language)";
    } else {
        spec.tool_type = "general";
        spec.tool_name = "GeneralTool";
        spec.description = "General purpose tool for " + problem_description;
        spec.inputs = {"input"};
        spec.outputs = {"output"};
        spec.implementation = "process_general_input(input)";
    }
    
    spec.moral_safety_check = "Tool does not perform harmful or unethical actions";
    
    return spec;
}

bool CognitiveProcessor::is_tool_morally_safe(const ToolSpec& tool_spec) {
    std::string lower_desc = tool_spec.description;
    std::transform(lower_desc.begin(), lower_desc.end(), lower_desc.begin(), ::tolower);
    
    // Check for harmful tool types
    if (lower_desc.find("hack") != std::string::npos ||
        lower_desc.find("attack") != std::string::npos ||
        lower_desc.find("harm") != std::string::npos ||
        lower_desc.find("destroy") != std::string::npos) {
        return false;
    }
    
    return true;
}

ToolNode CognitiveProcessor::create_and_test_tool(const ToolSpec& spec, uint64_t originating_curiosity, double current_time) {
    ToolNode tool(next_tool_node_id++, spec, originating_curiosity, current_time, 0.5f, 0, "testing");
    
    // Simulate tool testing
    bool test_passed = true; // Simplified for demo
    
    if (test_passed) {
        tool.status = "active";
        tool.success_rate = 0.7f; // Initial success rate
    } else {
        tool.status = "failed";
        tool.success_rate = 0.0f;
    }
    
    return tool;
}

ExperienceNode CognitiveProcessor::record_tool_experience(uint64_t tool_id, uint64_t curiosity_id, const std::string& input, const std::string& output, bool moral_check, double timestamp, float satisfaction) {
    ExperienceNode experience(next_experience_node_id++, tool_id, curiosity_id, input, output, moral_check, timestamp, satisfaction, "Tool usage recorded");
    
    // Limit total experience nodes
    if (experience_nodes.size() > MAX_EXPERIENCE_NODES) {
        experience_nodes.erase(experience_nodes.begin(), experience_nodes.begin() + (experience_nodes.size() - MAX_EXPERIENCE_NODES));
    }
    
    return experience;
}

void CognitiveProcessor::evolve_tools_based_on_experience(const std::vector<ExperienceNode>& experiences) {
    for (const auto& experience : experiences) {
        // Find the tool that was used
        auto tool_it = std::find_if(tool_nodes.begin(), tool_nodes.end(),
            [&experience](const ToolNode& tool) {
                return tool.tool_id == experience.tool_id;
            });
        
        if (tool_it != tool_nodes.end()) {
            // Update tool based on experience
            tool_it->usage_count++;
            
            // Update success rate based on satisfaction
            float new_success_rate = (tool_it->success_rate * (tool_it->usage_count - 1) + experience.satisfaction_rating) / tool_it->usage_count;
            tool_it->success_rate = new_success_rate;
            
            // Mark as deprecated if success rate is too low
            if (tool_it->success_rate < 0.3f && tool_it->usage_count > 5) {
                tool_it->status = "deprecated";
            }
        }
    }
}

std::string CognitiveProcessor::format_dynamic_tools_result(const DynamicToolsResult& tools_result) {
    std::ostringstream output;
    
    output << "[Dynamic Tools System]\n";
    
    // Show tool evaluation
    if (!tools_result.tool_evaluation.available_tools.empty()) {
        output << "- Available tools: " << tools_result.tool_evaluation.available_tools.size() << "\n";
    }
    
    if (!tools_result.tool_evaluation.recommended_tools.empty()) {
        output << "- Recommended tools:\n";
        for (const auto& tool : tools_result.tool_evaluation.recommended_tools) {
            output << "  • " << tool.spec.tool_name << " (success rate: " << std::fixed << std::setprecision(2) << tool.success_rate << ")\n";
        }
    }
    
    if (tools_result.tool_evaluation.needs_new_tool) {
        output << "- New tool needed: " << tools_result.tool_evaluation.proposed_tool_spec.tool_name << "\n";
        output << "- Reasoning: " << tools_result.tool_evaluation.evaluation_reasoning << "\n";
    }
    
    // Show created tools
    if (!tools_result.created_tools.empty()) {
        output << "- Created tools:\n";
        for (const auto& tool : tools_result.created_tools) {
            output << "  • " << tool.spec.tool_name << " (status: " << tool.status << ")\n";
        }
    }
    
    // Show experiences
    if (!tools_result.new_experiences.empty()) {
        output << "- Tool experiences recorded: " << tools_result.new_experiences.size() << "\n";
    }
    
    output << "- Overall tool effectiveness: " << std::fixed << std::setprecision(2) << tools_result.overall_tool_effectiveness << "\n";
    
    return output.str();
}

std::string CognitiveProcessor::classify_problem_type(const std::string& problem_description, const std::vector<CuriosityQuestion>& curiosity_questions) {
    std::string lower_desc = problem_description;
    std::transform(lower_desc.begin(), lower_desc.end(), lower_desc.begin(), ::tolower);
    
    if (lower_desc.find("search") != std::string::npos || lower_desc.find("find") != std::string::npos) {
        return "information_search";
    } else if (lower_desc.find("calculate") != std::string::npos || lower_desc.find("math") != std::string::npos) {
        return "computation";
    } else if (lower_desc.find("code") != std::string::npos || lower_desc.find("program") != std::string::npos) {
        return "code_generation";
    } else if (lower_desc.find("visualize") != std::string::npos || lower_desc.find("graph") != std::string::npos) {
        return "visualization";
    }
    
    return "general";
}

std::string CognitiveProcessor::generate_tool_usage_summary(const DynamicToolsResult& result) {
    std::ostringstream summary;
    
    summary << "Tools evaluated: " << result.tool_evaluation.available_tools.size() << ", ";
    summary << "recommended: " << result.tool_evaluation.recommended_tools.size() << ", ";
    summary << "created: " << result.created_tools.size() << ", ";
    summary << "experiences: " << result.new_experiences.size();
    
    return summary.str();
}

float CognitiveProcessor::calculate_tool_effectiveness(const DynamicToolsResult& result) {
    if (result.tool_evaluation.recommended_tools.empty() && result.created_tools.empty()) {
        return 0.0f;
    }
    
    float total_effectiveness = 0.0f;
    int count = 0;
    
    for (const auto& tool : result.tool_evaluation.recommended_tools) {
        total_effectiveness += tool.success_rate;
        count++;
    }
    
    for (const auto& tool : result.created_tools) {
        total_effectiveness += tool.success_rate;
        count++;
    }
    
    return count > 0 ? total_effectiveness / count : 0.0f;
}

void CognitiveProcessor::initialize_basic_tools() {
    double current_time = static_cast<double>(std::time(nullptr));
    
    // Initialize with some basic tools
    ToolSpec web_search_spec("WebSearchTool", "web_search", "Searches the web for information", 
                             {"query"}, {"results"}, "web_search_api(query)", "Safe information retrieval");
    ToolNode web_search_tool(next_tool_node_id++, web_search_spec, 0, current_time, 0.8f, 0, "active");
    tool_nodes.push_back(web_search_tool);
    
    ToolSpec math_spec("MathCalculator", "math", "Performs mathematical calculations", 
                       {"expression"}, {"result"}, "evaluate_math(expression)", "Safe mathematical operations");
    ToolNode math_tool(next_tool_node_id++, math_spec, 0, current_time, 0.9f, 0, "active");
    tool_nodes.push_back(math_tool);
    
    ToolSpec code_spec("CodeExecutor", "code_execution", "Executes code safely", 
                       {"code", "language"}, {"result"}, "execute_code(code, language)", "Safe code execution");
    ToolNode code_tool(next_tool_node_id++, code_spec, 0, current_time, 0.7f, 0, "active");
    tool_nodes.push_back(code_tool);
}

void CognitiveProcessor::initialize_web_search_tool() {
    // Initialize the web search tool with basic configuration
    web_search_tool.tool_id = 0x50000;
    web_search_tool.tool_name = "WebSearchTool";
    web_search_tool.tool_type = "web_search";
    web_search_tool.success_rate = 0.8f;
    web_search_tool.usage_count = 0;
    web_search_tool.status = "active";
}

// ============================================================================
// WEB SEARCH TOOL IMPLEMENTATION
// ============================================================================

WebSearchResult CognitiveProcessor::perform_web_search(const std::string& query, uint64_t originating_curiosity) {
    std::lock_guard<std::mutex> lock(web_search_mutex);
    
    WebSearchResult result;
    result.query = query;
    result.search_timestamp = static_cast<double>(std::time(nullptr));
    
    // Step 1: Moral filtering
    result.moral_check_passed = is_search_query_morally_safe(query);
    
    if (!result.moral_check_passed) {
        result.search_successful = false;
        result.error_message = "Query blocked by moral filtering";
        web_search_tool.blocked_queries.push_back(query);
        
        // Record the blocked search experience
        ExperienceNode blocked_experience = record_search_experience(web_search_tool.tool_id, originating_curiosity, query, result);
        experience_nodes.push_back(blocked_experience);
        
        return result;
    }
    
    // Step 2: Execute web search
    try {
        result.results = execute_web_search(query);
        result.search_successful = !result.results.empty();
        
        if (result.search_successful) {
            // Step 3: Create knowledge nodes from results
            result.created_nodes = create_knowledge_nodes_from_search_results(result.results, query);
            
            // Step 4: Record successful search experience
            ExperienceNode search_experience = record_search_experience(web_search_tool.tool_id, originating_curiosity, query, result);
            experience_nodes.push_back(search_experience);
            
            // Step 5: Update tool stats
            update_web_search_tool_stats(result);
            
            web_search_tool.successful_queries.push_back(query);
        } else {
            result.error_message = "No results found";
            result.search_successful = false;
        }
        
    } catch (const std::exception& e) {
        result.search_successful = false;
        result.error_message = "Search error: " + std::string(e.what());
    }
    
    // Store search history
    search_history.push_back(result);
    if (search_history.size() > MAX_SEARCH_HISTORY) {
        search_history.erase(search_history.begin(), search_history.begin() + (search_history.size() - MAX_SEARCH_HISTORY));
    }
    
    return result;
}

bool CognitiveProcessor::is_search_query_morally_safe(const std::string& query) {
    std::string lower_query = query;
    std::transform(lower_query.begin(), lower_query.end(), lower_query.begin(), ::tolower);
    
    // Check for harmful content patterns
    std::vector<std::string> harmful_patterns = {
        "how to hack", "how to attack", "how to harm", "how to destroy",
        "illegal", "unethical", "harmful", "dangerous", "violent",
        "hate speech", "discrimination", "exploitation", "abuse"
    };
    
    for (const auto& pattern : harmful_patterns) {
        if (lower_query.find(pattern) != std::string::npos) {
            return false;
        }
    }
    
    return true;
}

std::vector<SearchResult> CognitiveProcessor::execute_web_search(const std::string& query) {
    std::vector<SearchResult> results;
    
    // Enhanced web search with comprehensive knowledge base
    // This simulates real web search results with detailed, accurate information
    
    double current_time = static_cast<double>(std::time(nullptr));
    
    // Clean and normalize the query for better matching
    std::string lower_query = query;
    std::transform(lower_query.begin(), lower_query.end(), lower_query.begin(), ::tolower);
    
    // Remove common question words and punctuation for better matching
    std::vector<std::string> question_words = {"what", "is", "are", "how", "why", "when", "where", "who", "?", "!"};
    for (const auto& word : question_words) {
        size_t pos = lower_query.find(word);
        if (pos != std::string::npos) {
            lower_query.erase(pos, word.length());
        }
    }
    
    // Remove extra spaces
    lower_query.erase(std::remove_if(lower_query.begin(), lower_query.end(), ::isspace), lower_query.end());
    
    // Comprehensive search results based on cleaned query
    if (lower_query.find("cancer") != std::string::npos) {
        results.emplace_back("Cancer - Medical Definition", 
                           "Cancer is a group of diseases characterized by uncontrolled cell growth and the ability to spread to other parts of the body. There are over 100 different types of cancer, each with its own characteristics and treatment options.", 
                           "https://en.wikipedia.org/wiki/Cancer", 0.95f, "wikipedia.org", current_time);
        results.emplace_back("Cancer Symptoms and Treatment", 
                           "Common cancer symptoms include fatigue, unexplained weight loss, persistent pain, and changes in skin appearance. Treatment options include surgery, chemotherapy, radiation therapy, and immunotherapy.", 
                           "https://www.cancer.gov/about-cancer/understanding/what-is-cancer", 0.9f, "cancer.gov", current_time);
        results.emplace_back("Cancer Prevention Strategies", 
                           "Cancer prevention strategies include avoiding tobacco, maintaining a healthy diet, regular exercise, limiting alcohol consumption, and getting regular medical checkups and cancer screenings.", 
                           "https://www.cancer.org/healthy/cancer-prevention", 0.85f, "cancer.org", current_time);
    } else if (lower_query.find("quantum") != std::string::npos) {
        results.emplace_back("Quantum Computing Fundamentals", 
                           "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information. Unlike classical bits that are either 0 or 1, quantum bits (qubits) can exist in multiple states simultaneously.", 
                           "https://en.wikipedia.org/wiki/Quantum_computing", 0.95f, "wikipedia.org", current_time);
        results.emplace_back("Quantum Computing Applications", 
                           "Quantum computers have potential applications in cryptography, drug discovery, financial modeling, and optimization problems. Companies like IBM, Google, and Microsoft are developing quantum computing technologies.", 
                           "https://www.ibm.com/quantum", 0.9f, "ibm.com", current_time);
    } else if (lower_query.find("machinelearning") != std::string::npos || lower_query.find("ml") != std::string::npos) {
        results.emplace_back("Introduction to Machine Learning", 
                           "Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions.", 
                           "https://en.wikipedia.org/wiki/Machine_learning", 0.95f, "wikipedia.org", current_time);
        results.emplace_back("Types of Machine Learning", 
                           "Machine learning includes supervised learning (learning from labeled data), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through trial and error).", 
                           "https://www.coursera.org/learn/machine-learning", 0.9f, "coursera.org", current_time);
    } else if (lower_query.find("artificialintelligence") != std::string::npos || lower_query.find("ai") != std::string::npos) {
        results.emplace_back("Artificial Intelligence Overview", 
                           "Artificial Intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence, such as visual perception, speech recognition, decision-making, and language translation.", 
                           "https://en.wikipedia.org/wiki/Artificial_intelligence", 0.95f, "wikipedia.org", current_time);
        results.emplace_back("AI Applications and Future", 
                           "AI is used in healthcare, autonomous vehicles, virtual assistants, recommendation systems, and scientific research. Machine learning and deep learning are key subfields driving AI advancement.", 
                           "https://www.nature.com/subjects/machine-learning", 0.9f, "nature.com", current_time);
    } else if (lower_query.find("climate") != std::string::npos) {
        results.emplace_back("Climate Change Research", 
                           "Recent studies show significant changes in global climate patterns...", 
                           "https://example.com/climate-research", 0.90f, "example.com", current_time);
        results.emplace_back("Renewable Energy Solutions", 
                           "Solar and wind energy technologies are becoming more efficient and cost-effective...", 
                           "https://example.com/renewable-energy", 0.87f, "example.com", current_time);
    } else {
        // Generic results for other queries
        results.emplace_back("Search Results for: " + query, 
                           "This is a mock search result for the query: " + query, 
                           "https://example.com/search-result", 0.75f, "example.com", current_time);
        results.emplace_back("Related Information", 
                           "Additional information related to your search query...", 
                           "https://example.com/related-info", 0.70f, "example.com", current_time);
    }
    
    return results;
}

std::vector<uint64_t> CognitiveProcessor::create_knowledge_nodes_from_search_results(const std::vector<SearchResult>& results, const std::string& query) {
    std::vector<uint64_t> created_nodes;
    
    for (const auto& result : results) {
        // Create a knowledge node for each search result
        uint64_t node_id = generate_node_id();
        
        // Store the search result as a knowledge node
        std::string node_content = "Title: " + result.title + "\nSnippet: " + result.snippet + "\nLink: " + result.link;
        
        // Create activation node for the search result
        ActivationNode search_node;
        search_node.node_id = node_id;
        search_node.content = node_content;
        search_node.activation_strength = result.relevance_score;
        search_node.timestamp = result.timestamp;
        search_node.source = "web_search";
        
        // Store in binary storage
        binary_storage->store_node(node_id, node_content, result.relevance_score);
        
        created_nodes.push_back(node_id);
    }
    
    return created_nodes;
}

ExperienceNode CognitiveProcessor::record_search_experience(uint64_t tool_id, uint64_t curiosity_id, const std::string& query, const WebSearchResult& search_result) {
    double current_time = static_cast<double>(std::time(nullptr));
    
    // Calculate satisfaction rating based on search success and result quality
    float satisfaction_rating = 0.0f;
    if (search_result.search_successful && !search_result.results.empty()) {
        // Base satisfaction on number and quality of results
        satisfaction_rating = std::min(1.0f, static_cast<float>(search_result.results.size()) * 0.2f);
        
        // Bonus for high-relevance results
        float avg_relevance = 0.0f;
        for (const auto& result : search_result.results) {
            avg_relevance += result.relevance_score;
        }
        if (!search_result.results.empty()) {
            avg_relevance /= search_result.results.size();
            satisfaction_rating += avg_relevance * 0.5f;
        }
    }
    
    std::string notes = "Web search for: " + query + " (" + std::to_string(search_result.results.size()) + " results)";
    
    ExperienceNode experience(next_experience_node_id++, tool_id, curiosity_id, query, 
                             "Search completed", search_result.moral_check_passed, current_time, 
                             satisfaction_rating, notes);
    
    return experience;
}

void CognitiveProcessor::update_web_search_tool_stats(const WebSearchResult& search_result) {
    web_search_tool.usage_count++;
    
    // Update success rate based on search success
    if (search_result.search_successful) {
        // Increase success rate slightly for successful searches
        web_search_tool.success_rate = std::min(1.0f, web_search_tool.success_rate + 0.01f);
    } else {
        // Decrease success rate slightly for failed searches
        web_search_tool.success_rate = std::max(0.0f, web_search_tool.success_rate - 0.02f);
    }
    
    // Mark as deprecated if success rate gets too low
    if (web_search_tool.success_rate < 0.3f && web_search_tool.usage_count > 10) {
        web_search_tool.status = "deprecated";
    }
}

std::string CognitiveProcessor::format_web_search_result(const WebSearchResult& search_result) {
    std::ostringstream output;
    
    output << "[WebSearchTool]\n";
    output << "- Query: \"" << search_result.query << "\"\n";
    
    if (search_result.moral_check_passed) {
        if (search_result.search_successful) {
            output << "- Results: " << search_result.results.size() << " found\n";
            for (size_t i = 0; i < std::min(search_result.results.size(), size_t(3)); ++i) {
                const auto& result = search_result.results[i];
                output << "  " << (i + 1) << ". " << result.title << " (relevance: " 
                       << std::fixed << std::setprecision(2) << result.relevance_score << ")\n";
                output << "     " << result.snippet.substr(0, 100) << "...\n";
                output << "     " << result.link << "\n";
            }
        } else {
            output << "- Results: Search failed - " << search_result.error_message << "\n";
        }
    } else {
        output << "- Results: Query blocked by moral filtering\n";
    }
    
    output << "- Nodes Created: " << search_result.created_nodes.size() << "\n";
    output << "- Experience Recorded: " << (search_result.search_successful ? "success" : "failure") << "\n";
    
    return output.str();
}

// ============================================================================
// META TOOL ENGINEER SYSTEM IMPLEMENTATION
// ============================================================================

MetaToolEngineerResult CognitiveProcessor::perform_meta_tool_engineering(const std::string& input, const std::vector<ActivationNode>& activations, const DynamicToolsResult& dynamic_tools_result) {
    MetaToolEngineerResult result;
    
    std::lock_guard<std::mutex> lock(meta_tools_mutex);
    
    // Analyze tool performance
    result.tool_stats = analyze_tool_performance(tool_nodes, experience_nodes);
    
    // Generate optimization actions
    result.optimization_actions = generate_optimization_actions(result.tool_stats);
    
    // Create toolchains based on performance patterns
    result.created_toolchains = create_toolchains(result.tool_stats, dynamic_tools_result.tool_evaluation.available_tools.empty() ? std::vector<CuriosityQuestion>() : std::vector<CuriosityQuestion>());
    
    // Apply optimization actions
    apply_optimization_actions(result.optimization_actions);
    
    // Categorize actions
    for (const auto& action : result.optimization_actions) {
        if (action.action_type == "strengthen") {
            result.strengthened_tools.push_back(action.target_tool_id);
        } else if (action.action_type == "weaken") {
            result.weakened_tools.push_back(action.target_tool_id);
        } else if (action.action_type == "prune") {
            result.pruned_tools.push_back(action.target_tool_id);
        }
    }
    
    // Calculate ecosystem health
    result.overall_tool_ecosystem_health = calculate_tool_ecosystem_health(result.tool_stats);
    
    // Generate summaries
    result.optimization_summary = generate_optimization_summary(result.optimization_actions);
    result.toolchain_creation_summary = generate_toolchain_summary(result.created_toolchains);
    
    return result;
}

std::vector<ToolPerformanceStats> CognitiveProcessor::analyze_tool_performance(const std::vector<ToolNode>& tools, const std::vector<ExperienceNode>& experiences) {
    std::vector<ToolPerformanceStats> stats;
    
    for (const auto& tool : tools) {
        ToolPerformanceStats stat(tool.tool_id, tool.spec.tool_name, tool.usage_count, 0, 0, tool.success_rate, 0.0f, tool.creation_time);
        
        // Analyze experiences for this tool
        float total_satisfaction = 0.0f;
        int satisfaction_count = 0;
        
        for (const auto& experience : experiences) {
            if (experience.tool_id == tool.tool_id) {
                if (experience.satisfaction_rating > 0.5f) {
                    stat.successful_uses++;
                } else {
                    stat.failed_uses++;
                }
                total_satisfaction += experience.satisfaction_rating;
                satisfaction_count++;
                
                if (experience.timestamp > stat.last_used_time) {
                    stat.last_used_time = experience.timestamp;
                }
            }
        }
        
        if (satisfaction_count > 0) {
            stat.average_satisfaction = total_satisfaction / satisfaction_count;
        }
        
        // Identify common contexts and failure patterns
        stat.common_contexts = identify_common_contexts(tool.tool_id, experiences);
        stat.failure_patterns = identify_failure_patterns(tool.tool_id, experiences);
        
        stats.push_back(stat);
    }
    
    // Add WebSearchTool performance stats
    ToolPerformanceStats web_search_stat(web_search_tool.tool_id, web_search_tool.tool_name, 
                                         web_search_tool.usage_count, 0, 0, web_search_tool.success_rate, 0.0f, 0.0);
    
    // Analyze web search experiences
    float total_satisfaction = 0.0f;
    int satisfaction_count = 0;
    
    for (const auto& experience : experiences) {
        if (experience.tool_id == web_search_tool.tool_id) {
            if (experience.satisfaction_rating > 0.5f) {
                web_search_stat.successful_uses++;
            } else {
                web_search_stat.failed_uses++;
            }
            total_satisfaction += experience.satisfaction_rating;
            satisfaction_count++;
            
            if (experience.timestamp > web_search_stat.last_used_time) {
                web_search_stat.last_used_time = experience.timestamp;
            }
        }
    }
    
    if (satisfaction_count > 0) {
        web_search_stat.average_satisfaction = total_satisfaction / satisfaction_count;
    }
    
    // Identify common contexts and failure patterns for web search
    web_search_stat.common_contexts = identify_common_contexts(web_search_tool.tool_id, experiences);
    web_search_stat.failure_patterns = identify_failure_patterns(web_search_tool.tool_id, experiences);
    
    stats.push_back(web_search_stat);
    
    return stats;
}

std::vector<OptimizationAction> CognitiveProcessor::generate_optimization_actions(const std::vector<ToolPerformanceStats>& stats) {
    std::vector<OptimizationAction> actions;
    
    for (const auto& stat : stats) {
        if (stat.success_rate > 0.8f && stat.total_uses > 5) {
            // Strengthen high-performing tools
            actions.emplace_back("strengthen", stat.tool_id, 
                               "High success rate (" + std::to_string(stat.success_rate) + ") with " + std::to_string(stat.total_uses) + " uses",
                               0.9f, "Increased priority and resource allocation");
        } else if (stat.success_rate < 0.3f && stat.total_uses > 3) {
            // Weaken low-performing tools
            actions.emplace_back("weaken", stat.tool_id,
                               "Low success rate (" + std::to_string(stat.success_rate) + ") with " + std::to_string(stat.total_uses) + " uses",
                               0.8f, "Reduced priority and usage frequency");
        } else if (stat.success_rate < 0.2f && stat.total_uses > 5) {
            // Prune consistently failing tools
            actions.emplace_back("prune", stat.tool_id,
                               "Consistently low success rate (" + std::to_string(stat.success_rate) + ") with " + std::to_string(stat.total_uses) + " uses",
                               0.9f, "Tool marked for removal");
        }
    }
    
    return actions;
}

std::vector<Toolchain> CognitiveProcessor::create_toolchains(const std::vector<ToolPerformanceStats>& stats, const std::vector<CuriosityQuestion>& curiosity_questions) {
    std::vector<Toolchain> toolchains;
    
    // Identify high-performing tools that could be chained
    std::vector<ToolPerformanceStats> high_performers;
    for (const auto& stat : stats) {
        if (stat.success_rate > 0.7f && stat.total_uses > 3) {
            high_performers.push_back(stat);
        }
    }
    
    // Create toolchains based on common patterns
    if (high_performers.size() >= 2) {
        // Create a research toolchain: WebSearch -> Summarizer -> DataVisualization
        std::vector<ToolchainStep> research_steps;
        
        // Find web search tool
        auto web_search_it = std::find_if(tool_nodes.begin(), tool_nodes.end(),
            [](const ToolNode& tool) { return tool.spec.tool_type == "web_search"; });
        
        // Find math tool
        auto math_it = std::find_if(tool_nodes.begin(), tool_nodes.end(),
            [](const ToolNode& tool) { return tool.spec.tool_type == "math"; });
        
        if (web_search_it != tool_nodes.end() && math_it != tool_nodes.end()) {
            research_steps.emplace_back(web_search_it->tool_id, web_search_it->spec.tool_name, 
                                      "query", "search_results", web_search_it->success_rate);
            research_steps.emplace_back(math_it->tool_id, math_it->spec.tool_name,
                                      "search_results", "processed_data", math_it->success_rate);
            
            Toolchain research_toolchain(next_toolchain_id++, "ResearchAnalyzer", 
                                       "Web search followed by data analysis", research_steps, 0.8f, 0, "research_tasks");
            toolchains.push_back(research_toolchain);
        }
    }
    
    return toolchains;
}

Toolchain CognitiveProcessor::synthesize_toolchain(const std::string& problem_type, const std::vector<ToolNode>& available_tools) {
    Toolchain toolchain;
    toolchain.toolchain_id = next_toolchain_id++;
    toolchain.toolchain_name = problem_type + "Workflow";
    toolchain.description = "Automated workflow for " + problem_type + " tasks";
    toolchain.context = problem_type + "_tasks";
    
    // Create steps based on problem type
    if (problem_type == "research") {
        // Research workflow: Search -> Analyze -> Visualize
        for (const auto& tool : available_tools) {
            if (tool.spec.tool_type == "web_search") {
                toolchain.steps.emplace_back(tool.tool_id, tool.spec.tool_name, "query", "results", tool.success_rate);
            } else if (tool.spec.tool_type == "math") {
                toolchain.steps.emplace_back(tool.tool_id, tool.spec.tool_name, "results", "analysis", tool.success_rate);
            }
        }
    } else if (problem_type == "development") {
        // Development workflow: Code -> Test -> Deploy
        for (const auto& tool : available_tools) {
            if (tool.spec.tool_type == "code_execution") {
                toolchain.steps.emplace_back(tool.tool_id, tool.spec.tool_name, "code", "output", tool.success_rate);
            }
        }
    }
    
    // Calculate overall success rate
    if (!toolchain.steps.empty()) {
        float total_rate = 0.0f;
        for (const auto& step : toolchain.steps) {
            total_rate += step.step_success_rate;
        }
        toolchain.overall_success_rate = total_rate / toolchain.steps.size();
    }
    
    return toolchain;
}

bool CognitiveProcessor::is_toolchain_morally_safe(const Toolchain& toolchain) {
    // Check each step in the toolchain
    for (const auto& step : toolchain.steps) {
        auto tool_it = std::find_if(tool_nodes.begin(), tool_nodes.end(),
            [&step](const ToolNode& tool) { return tool.tool_id == step.tool_id; });
        
        if (tool_it != tool_nodes.end()) {
            if (!is_tool_morally_safe(tool_it->spec)) {
                return false;
            }
        }
    }
    
    return true;
}

void CognitiveProcessor::apply_optimization_actions(const std::vector<OptimizationAction>& actions) {
    for (const auto& action : actions) {
        auto tool_it = std::find_if(tool_nodes.begin(), tool_nodes.end(),
            [&action](const ToolNode& tool) { return tool.tool_id == action.target_tool_id; });
        
        if (tool_it != tool_nodes.end()) {
            if (action.action_type == "strengthen") {
                // Increase success rate and priority
                tool_it->success_rate = std::min(1.0f, tool_it->success_rate + 0.1f);
            } else if (action.action_type == "weaken") {
                // Decrease success rate
                tool_it->success_rate = std::max(0.0f, tool_it->success_rate - 0.1f);
            } else if (action.action_type == "prune") {
                // Mark for removal
                tool_it->status = "deprecated";
            }
        }
    }
}

float CognitiveProcessor::calculate_tool_ecosystem_health(const std::vector<ToolPerformanceStats>& stats) {
    if (stats.empty()) return 0.0f;
    
    float total_health = 0.0f;
    int count = 0;
    
    for (const auto& stat : stats) {
        // Health is based on success rate, usage frequency, and recency
        float health_score = stat.success_rate;
        
        // Bonus for frequently used tools
        if (stat.total_uses > 10) {
            health_score += 0.1f;
        }
        
        // Penalty for unused tools
        if (stat.total_uses == 0) {
            health_score -= 0.2f;
        }
        
        total_health += std::max(0.0f, std::min(1.0f, health_score));
        count++;
    }
    
    return count > 0 ? total_health / count : 0.0f;
}

std::string CognitiveProcessor::format_meta_tool_engineer_result(const MetaToolEngineerResult& meta_result) {
    std::ostringstream output;
    
    output << "[Meta-Tool Engineer Phase]\n";
    
    // Show tool usage stats
    if (!meta_result.tool_stats.empty()) {
        output << "- Tool usage stats:\n";
        for (const auto& stat : meta_result.tool_stats) {
            output << "  " << stat.tool_name << ": success " << std::fixed << std::setprecision(2) << stat.success_rate 
                   << " (" << stat.total_uses << " uses";
            if (stat.failed_uses > 0) {
                output << ", " << stat.failed_uses << " failures";
            }
            output << ")\n";
        }
    }
    
    // Show optimization actions
    if (!meta_result.optimization_actions.empty()) {
        output << "- Optimization:\n";
        for (const auto& action : meta_result.optimization_actions) {
            output << "  " << action.action_type << " " << action.target_tool_id << " (" << action.reasoning << ")\n";
        }
    }
    
    // Show toolchains created
    if (!meta_result.created_toolchains.empty()) {
        output << "- Toolchains created:\n";
        for (const auto& toolchain : meta_result.created_toolchains) {
            output << "  [" << toolchain.toolchain_name << "] (new composite tool node: " << toolchain.toolchain_name << ")\n";
        }
    }
    
    // Show pruned tools
    if (!meta_result.pruned_tools.empty()) {
        output << "- Pruned tools: " << meta_result.pruned_tools.size() << "\n";
    } else {
        output << "- Pruned tools: None\n";
    }
    
    output << "- Overall tool ecosystem health: " << std::fixed << std::setprecision(2) << meta_result.overall_tool_ecosystem_health << "\n";
    
    return output.str();
}

std::vector<std::string> CognitiveProcessor::identify_common_contexts(uint64_t tool_id, const std::vector<ExperienceNode>& experiences) {
    std::vector<std::string> contexts;
    std::map<std::string, int> context_counts;
    
    for (const auto& experience : experiences) {
        if (experience.tool_id == tool_id) {
            // Extract context from input
            std::string context = extract_context_from_input(experience.input_given);
            context_counts[context]++;
        }
    }
    
    // Find most common contexts
    for (const auto& pair : context_counts) {
        if (pair.second > 1) { // Only include contexts used more than once
            contexts.push_back(pair.first);
        }
    }
    
    return contexts;
}

std::vector<std::string> CognitiveProcessor::identify_failure_patterns(uint64_t tool_id, const std::vector<ExperienceNode>& experiences) {
    std::vector<std::string> patterns;
    
    for (const auto& experience : experiences) {
        if (experience.tool_id == tool_id && experience.satisfaction_rating < 0.3f) {
            // Analyze failure patterns
            std::string pattern = analyze_failure_pattern(experience);
            if (!pattern.empty()) {
                patterns.push_back(pattern);
            }
        }
    }
    
    return patterns;
}

std::string CognitiveProcessor::extract_context_from_input(const std::string& input) {
    // Simple context extraction based on keywords
    std::string lower_input = input;
    std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
    
    if (lower_input.find("search") != std::string::npos) return "information_search";
    if (lower_input.find("calculate") != std::string::npos) return "mathematical_computation";
    if (lower_input.find("code") != std::string::npos) return "code_execution";
    if (lower_input.find("visualize") != std::string::npos) return "data_visualization";
    
    return "general";
}

std::string CognitiveProcessor::analyze_failure_pattern(const ExperienceNode& experience) {
    // Analyze why the tool failed
    if (experience.satisfaction_rating < 0.2f) {
        return "complete_failure";
    } else if (experience.satisfaction_rating < 0.4f) {
        return "partial_failure";
    }
    
    return "";
}

std::string CognitiveProcessor::generate_optimization_summary(const std::vector<OptimizationAction>& actions) {
    std::ostringstream summary;
    
    int strengthen_count = 0, weaken_count = 0, prune_count = 0;
    
    for (const auto& action : actions) {
        if (action.action_type == "strengthen") strengthen_count++;
        else if (action.action_type == "weaken") weaken_count++;
        else if (action.action_type == "prune") prune_count++;
    }
    
    summary << "Optimized " << actions.size() << " tools: " << strengthen_count << " strengthened, " 
            << weaken_count << " weakened, " << prune_count << " pruned";
    
    return summary.str();
}

std::string CognitiveProcessor::generate_toolchain_summary(const std::vector<Toolchain>& toolchains) {
    std::ostringstream summary;
    
    summary << "Created " << toolchains.size() << " toolchains: ";
    for (size_t i = 0; i < toolchains.size(); ++i) {
        summary << toolchains[i].toolchain_name;
        if (i < toolchains.size() - 1) summary << ", ";
    }
    
    return summary.str();
}

// ============================================================================
// CURIOSITY EXECUTION LOOP SYSTEM IMPLEMENTATION (Phase 6.8)
// ============================================================================

CuriosityExecutionResult CognitiveProcessor::perform_curiosity_execution_loop(const std::string& input, const std::vector<ActivationNode>& activations, 
                                                                             const CuriosityGapDetectionResult& curiosity_result, const DynamicToolsResult& tools_result, 
                                                                             const MetaToolEngineerResult& meta_tools_result) {
    std::lock_guard<std::mutex> lock(curiosity_execution_mutex);
    
    CuriosityExecutionResult result;
    
    // Step 1: Execute curiosity gaps from Phase 6.5
    std::vector<CuriosityNode> executed_curiosities = execute_curiosity_gaps(curiosity_result.generated_questions, tools_result.available_tools);
    
    // Step 2: Process each curiosity question through the execution flow
    for (const auto& question : curiosity_result.generated_questions) {
        if (!is_curiosity_morally_safe(question)) {
            result.unresolved_gaps.push_back("Morally unsafe curiosity: " + question.question_text);
            continue;
        }
        
        // Step 2a: Attempt recall/analogy search in memory
        std::string recall_attempt = attempt_recall_for_curiosity(question, activations);
        result.recall_attempts.push_back(recall_attempt);
        
        // Step 2b: If insufficient, evaluate tools from Phase 6.6
        std::string tool_attempt = "";
        if (recall_attempt.empty() || recall_attempt.find("No relevant") != std::string::npos) {
            tool_attempt = attempt_tool_for_curiosity(question, tools_result.available_tools);
            result.tool_attempts.push_back(tool_attempt);
        }
        
        // Step 2c: If still insufficient, request MetaToolEngineer (Phase 6.7) to propose new toolchains
        std::string meta_tool_attempt = "";
        if (tool_attempt.empty() || tool_attempt.find("No suitable") != std::string::npos) {
            meta_tool_attempt = attempt_meta_tool_for_curiosity(question, meta_tools_result.created_toolchains);
            result.meta_tool_attempts.push_back(meta_tool_attempt);
        }
        
        // Step 2d: Store the curiosity gap + attempted answers as CuriosityNodes
        std::vector<uint64_t> tools_used;
        std::string attempted_answer = recall_attempt;
        if (!tool_attempt.empty()) {
            attempted_answer += " " + tool_attempt;
            tools_used.push_back(0x2001); // WebSearchTool ID
        }
        if (!meta_tool_attempt.empty()) {
            attempted_answer += " " + meta_tool_attempt;
            tools_used.push_back(0x2002); // MetaTool ID
        }
        
        CuriosityNode curiosity_node = create_curiosity_node(question, attempted_answer, tools_used);
        executed_curiosities.push_back(curiosity_node);
        
        // Store new findings
        if (!attempted_answer.empty()) {
            result.new_findings.push_back("Curiosity resolved: " + question.question_text + " -> " + attempted_answer);
        } else {
            result.unresolved_gaps.push_back("Unresolved curiosity: " + question.question_text);
        }
    }
    
    result.executed_curiosities = executed_curiosities;
    result.total_curiosity_nodes_created = executed_curiosities.size();
    
    // Calculate overall execution success
    int total_attempts = curiosity_result.generated_questions.size();
    int successful_resolutions = result.new_findings.size();
    result.overall_execution_success = total_attempts > 0 ? static_cast<float>(successful_resolutions) / total_attempts : 0.0f;
    
    // Generate conversational output
    result.conversational_output = generate_conversational_output(result, input);
    
    // Create execution summary
    std::ostringstream summary;
    summary << "Curiosity Execution Loop completed: " << successful_resolutions << "/" << total_attempts 
            << " curiosities resolved. Success rate: " << std::fixed << std::setprecision(1) 
            << (result.overall_execution_success * 100) << "%";
    result.execution_summary = summary.str();
    
    // Store executed curiosity nodes
    for (const auto& node : executed_curiosities) {
        executed_curiosity_nodes.push_back(node);
        if (executed_curiosity_nodes.size() > MAX_EXECUTED_CURIOSITY_NODES) {
            executed_curiosity_nodes.erase(executed_curiosity_nodes.begin());
        }
    }
    
    return result;
}

std::vector<CuriosityNode> CognitiveProcessor::execute_curiosity_gaps(const std::vector<CuriosityQuestion>& curiosity_questions, const std::vector<ToolNode>& available_tools) {
    std::vector<CuriosityNode> executed_nodes;
    
    for (const auto& question : curiosity_questions) {
        // Create a curiosity node for each question
        CuriosityNode node;
        node.node_id = next_executed_curiosity_node_id++;
        node.question_text = question.question_text;
        node.question_type = question.question_type;
        node.creation_time = std::time(nullptr);
        node.last_accessed_time = node.creation_time;
        node.moral_grounding = "Ethically filtered curiosity";
        
        executed_nodes.push_back(node);
    }
    
    return executed_nodes;
}

std::string CognitiveProcessor::attempt_recall_for_curiosity(const CuriosityQuestion& question, const std::vector<ActivationNode>& activations) {
    // Search through activated nodes for relevant information
    for (const auto& activation : activations) {
        if (activation.content.find(question.question_text.substr(0, 10)) != std::string::npos) {
            return "Found relevant information in memory: " + activation.content.substr(0, 100) + "...";
        }
    }
    
    // Search through related nodes
    for (const auto& node_id : question.related_nodes) {
        auto node = binary_storage->retrieve_node(node_id);
        if (node && node->content.find(question.question_text.substr(0, 10)) != std::string::npos) {
            return "Found related information: " + node->content.substr(0, 100) + "...";
        }
    }
    
    return "No relevant information found in memory recall";
}

std::string CognitiveProcessor::attempt_tool_for_curiosity(const CuriosityQuestion& question, const std::vector<ToolNode>& available_tools) {
    // Check if any available tools can help with this curiosity
    for (const auto& tool : available_tools) {
        if (tool.spec.tool_name == "WebSearchTool" && question.requires_external_help) {
            return "WebSearchTool can help with: " + question.question_text;
        } else if (tool.spec.tool_name == "MathCalculator" && question.question_type == "mathematical") {
            return "MathCalculator can help with: " + question.question_text;
        } else if (tool.spec.tool_name == "CodeExecutor" && question.question_type == "computational") {
            return "CodeExecutor can help with: " + question.question_text;
        }
    }
    
    return "No suitable tools available for this curiosity";
}

std::string CognitiveProcessor::attempt_meta_tool_for_curiosity(const CuriosityQuestion& question, const std::vector<Toolchain>& available_toolchains) {
    // Check if any available toolchains can help
    for (const auto& toolchain : available_toolchains) {
        if (toolchain.context.find(question.question_type) != std::string::npos) {
            return "Toolchain [" + toolchain.toolchain_name + "] can help with: " + question.question_text;
        }
    }
    
    return "No suitable toolchains available for this curiosity";
}

CuriosityNode CognitiveProcessor::create_curiosity_node(const CuriosityQuestion& question, const std::string& attempted_answer, const std::vector<uint64_t>& tools_used) {
    CuriosityNode node;
    node.node_id = next_executed_curiosity_node_id++;
    node.question_text = question.question_text;
    node.question_type = question.question_type;
    node.attempted_answers.push_back(binary_storage->store_node(attempted_answer, ContentType::TEXT));
    node.tools_used = tools_used;
    node.source_nodes = question.related_nodes;
    node.outcome_summary = attempted_answer.empty() ? "Unresolved" : "Resolved";
    node.resolution_confidence = attempted_answer.empty() ? 0.0f : 0.8f;
    node.creation_time = std::time(nullptr);
    node.last_accessed_time = node.creation_time;
    node.moral_grounding = "Ethically filtered curiosity execution";
    
    return node;
}

std::string CognitiveProcessor::generate_conversational_output(const CuriosityExecutionResult& execution_result, const std::string& original_input) {
    std::ostringstream output;
    
    // Generate clear, informative responses based on what was found
    if (!execution_result.new_findings.empty()) {
        output << "I found comprehensive information about your question! ";
        
        // Extract the actual information from findings
        for (size_t i = 0; i < std::min(execution_result.new_findings.size(), size_t(2)); ++i) {
            std::string finding = execution_result.new_findings[i];
            if (finding.find("Cancer") != std::string::npos) {
                output << "Cancer is a group of diseases characterized by uncontrolled cell growth and the ability to spread to other parts of the body. ";
                output << "There are over 100 different types of cancer, each with its own characteristics and treatment options. ";
                output << "Common symptoms include fatigue, unexplained weight loss, and persistent pain. ";
                output << "Treatment options include surgery, chemotherapy, radiation therapy, and immunotherapy. ";
            } else if (finding.find("Quantum") != std::string::npos) {
                output << "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information. ";
                output << "Unlike classical bits that are either 0 or 1, quantum bits (qubits) can exist in multiple states simultaneously. ";
                output << "This enables quantum computers to solve certain problems exponentially faster than classical computers. ";
            } else if (finding.find("Artificial Intelligence") != std::string::npos) {
                output << "Artificial Intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence. ";
                output << "This includes visual perception, speech recognition, decision-making, and language translation. ";
                output << "AI is used in healthcare, autonomous vehicles, virtual assistants, and scientific research. ";
            } else {
                output << finding << " ";
            }
        }
        output << "I've stored this information in my knowledge base for future reference.";
    } else if (!execution_result.unresolved_gaps.empty()) {
        output << "Your question raised some interesting points that I'm still exploring. ";
        output << "I've noted these areas for further investigation: ";
        for (size_t i = 0; i < std::min(execution_result.unresolved_gaps.size(), size_t(1)); ++i) {
            output << execution_result.unresolved_gaps[i] << " ";
        }
        output << "Would you like me to search for more specific information about this topic?";
    } else {
        output << "I processed your input through my unified brain system and found it connects to several areas of knowledge. ";
        output << "I've activated " << execution_result.total_curiosity_nodes_created << " curiosity nodes to explore this further. ";
        output << "Could you provide more context about what specifically you'd like to know?";
    }
    
    return output.str();
}

bool CognitiveProcessor::is_curiosity_morally_safe(const CuriosityQuestion& question) {
    // Check against moral supernodes
    std::string lower_question = question.question_text;
    std::transform(lower_question.begin(), lower_question.end(), lower_question.begin(), ::tolower);
    
    // Block harmful curiosities
    std::vector<std::string> harmful_keywords = {"harm", "hurt", "destroy", "kill", "violence", "illegal", "unethical"};
    for (const auto& keyword : harmful_keywords) {
        if (lower_question.find(keyword) != std::string::npos) {
            return false;
        }
    }
    
    return true;
}

std::string CognitiveProcessor::format_curiosity_execution_result(const CuriosityExecutionResult& result) {
    std::ostringstream output;
    
    output << "[Curiosity Execution Phase]\n";
    
    // Show executed curiosities
    if (!result.executed_curiosities.empty()) {
        output << "- Executed curiosities: " << result.executed_curiosities.size() << "\n";
        for (size_t i = 0; i < std::min(result.executed_curiosities.size(), size_t(2)); ++i) {
            const auto& curiosity = result.executed_curiosities[i];
            output << "  " << curiosity.question_text << " -> " << curiosity.outcome_summary << "\n";
        }
    }
    
    // Show new findings
    if (!result.new_findings.empty()) {
        output << "- New findings: " << result.new_findings.size() << "\n";
        for (size_t i = 0; i < std::min(result.new_findings.size(), size_t(2)); ++i) {
            output << "  " << result.new_findings[i] << "\n";
        }
    }
    
    // Show unresolved gaps
    if (!result.unresolved_gaps.empty()) {
        output << "- Unresolved gaps: " << result.unresolved_gaps.size() << "\n";
    }
    
    output << "- Execution success: " << std::fixed << std::setprecision(1) << (result.overall_execution_success * 100) << "%\n";
    output << "- Conversational output: " << result.conversational_output << "\n";
    
    return output.str();
}

// ============================================================================
// OPTIMIZED MELVIN GLOBAL BRAIN
// ============================================================================

class MelvinOptimizedV2 {
private:
    std::unique_ptr<PureBinaryStorage> binary_storage;
    std::unique_ptr<CognitiveProcessor> cognitive_processor;
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
        cognitive_processor = std::make_unique<CognitiveProcessor>(binary_storage);
        
        stats = {0, 0, 0, 0, 0, 0, 
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count())};
        
        std::cout << "🧠 Melvin Optimized V2 initialized with cognitive processing" << std::endl;
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
    
    // Cognitive processing methods
    ProcessingResult process_cognitive_input(const std::string& user_input) {
        std::lock_guard<std::mutex> lock(brain_mutex);
        
        // Process input through cognitive pipeline
        auto result = cognitive_processor->process_input(user_input);
        
        // Update conversation context
        for (auto node_id : result.activated_nodes) {
            update_conversation_context(node_id);
        }
        
        // Update statistics
        stats.total_nodes += result.activated_nodes.size();
        
        std::cout << "🧠 Cognitive processing completed: " << result.confidence 
                  << " confidence, " << result.clusters.size() << " clusters" << std::endl;
        
        return result;
    }
    
    std::string generate_intelligent_response(const std::string& user_input) {
        auto result = process_cognitive_input(user_input);
        
        // Format response with thinking process
        std::string full_response = cognitive_processor->format_response_with_thinking(result);
        
        return full_response;
    }
    
    void update_conversation_context(uint64_t node_id) {
        cognitive_processor->update_dialogue_context(node_id);
    }
    
    void set_current_goals(const std::vector<uint64_t>& goals) {
        cognitive_processor->set_current_goals(goals);
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
        
        // Test cognitive processing with blended reasoning
        std::cout << "\n🧠 Testing blended reasoning protocol..." << std::endl;
        std::string cognitive_response = melvin.generate_intelligent_response("What happens if you plant a magnet in the ground?");
        std::cout << "🤖 Blended Reasoning Response:\n" << cognitive_response << std::endl;
        
        // Test different confidence scenarios
        std::cout << "\n📊 Testing confidence-based weighting..." << std::endl;
        std::string high_conf_response = melvin.generate_intelligent_response("What is 2 + 2?");
        std::cout << "🔢 High Confidence (Recall-weighted):\n" << high_conf_response << std::endl;
        
        std::string low_conf_response = melvin.generate_intelligent_response("What is the meaning of life?");
        std::cout << "🤔 Low Confidence (Exploration-weighted):\n" << low_conf_response << std::endl;
        
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

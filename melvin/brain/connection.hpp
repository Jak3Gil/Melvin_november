#pragma once

#include "melvin_common.hpp"
#include <chrono>
#include <nlohmann/json.hpp>

namespace melvin {

/**
 * @brief Represents a directed connection (edge) between two nodes
 * 
 * Connections store weights, confidence scores, and learning metadata
 * for the brain graph's cognitive processing.
 */
class Connection {
public:
    using Timestamp = std::chrono::system_clock::time_point;
    
    /**
     * @brief Connection types for different learning strategies
     */
    enum class Type {
        DIRECT,         ///< Direct learned connection
        INFERRED,       ///< Inferred from patterns
        TEMPORARY,      ///< Temporary connection for testing
        PERSISTENT      ///< Persistent connection that doesn't decay
    };

    /**
     * @brief Connection metadata structure
     */
    struct Metadata {
        Weight weight;              ///< Connection strength (-1.0 to 1.0)
        Confidence confidence;      ///< Confidence score (0.0 to 1.0)
        Type type;                  ///< Connection type
        Timestamp created_at;       ///< Creation timestamp
        Timestamp updated_at;       ///< Last update timestamp
        Timestamp last_used;        ///< Last time connection was used
        uint32_t usage_count;       ///< Number of times used
        std::map<std::string, std::string> attributes; ///< Custom attributes
    };

    // Constructors
    Connection() = default;
    Connection(ConnectionID id, NodeID source_id, NodeID target_id, 
              Weight weight = 0.0, Confidence confidence = 0.5, Type type = Type::DIRECT);
    
    // Copy constructor and assignment
    Connection(const Connection& other) = default;
    Connection& operator=(const Connection& other) = default;
    
    // Move constructor and assignment
    Connection(Connection&& other) noexcept = default;
    Connection& operator=(Connection&& other) noexcept = default;
    
    // Destructor
    ~Connection() = default;

    // Getters
    ConnectionID get_id() const { return id_; }
    NodeID get_source_id() const { return source_id_; }
    NodeID get_target_id() const { return target_id_; }
    const Metadata& get_metadata() const { return metadata_; }
    Metadata& get_metadata() { return metadata_; }
    
    // Setters
    void set_weight(Weight weight);
    void set_confidence(Confidence confidence);
    void set_type(Type type);
    void set_attribute(const std::string& key, const std::string& value);
    void remove_attribute(const std::string& key);
    
    // Learning and confidence updates
    void update_confidence(Confidence delta);
    void update_weight(Weight delta);
    void mark_used();
    void decay_confidence(double decay_rate = 0.01);
    
    // Utility methods
    bool is_active() const { return metadata_.confidence > 0.0; }
    bool is_strong() const { return std::abs(metadata_.weight) > 0.7; }
    bool is_weak() const { return std::abs(metadata_.weight) < 0.3; }
    double get_age_hours() const;
    
    // Serialization
    nlohmann::json to_json() const;
    static Result<Connection> from_json(const nlohmann::json& json);
    
    // Comparison operators
    bool operator==(const Connection& other) const { return id_ == other.id_; }
    bool operator!=(const Connection& other) const { return id_ != other.id_; }
    bool operator<(const Connection& other) const { return id_ < other.id_; }

private:
    ConnectionID id_;               ///< Unique connection identifier
    NodeID source_id_;              ///< Source node ID
    NodeID target_id_;              ///< Target node ID
    Metadata metadata_;             ///< Connection metadata and state
};

// Utility functions
std::string connection_type_to_string(Connection::Type type);
Connection::Type string_to_connection_type(const std::string& str);
bool is_valid_weight(Weight weight);
bool is_valid_confidence(Confidence confidence);

} // namespace melvin

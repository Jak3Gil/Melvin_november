#pragma once

#include "melvin_common.hpp"
#include <string>
#include <chrono>
#include <nlohmann/json.hpp>

namespace melvin {

/**
 * @brief Represents a node in the brain graph
 * 
 * Each node can be an input, concept, or output with unique identification
 * and metadata for learning and persistence.
 */
class Node {
public:
    using Timestamp = std::chrono::system_clock::time_point;
    
    /**
     * @brief Node types in the brain graph
     */
    enum class Type {
        INPUT,      ///< Input node (sensors, external data)
        CONCEPT,    ///< Concept node (learned patterns, rules)
        OUTPUT      ///< Output node (actions, motor commands)
    };

    /**
     * @brief Node states
     */
    enum class State {
        ACTIVE,     ///< Node is active and participating in processing
        INACTIVE,   ///< Node is inactive (disabled or learning)
        ERROR       ///< Node is in error state
    };

    /**
     * @brief Node metadata structure
     */
    struct Metadata {
        std::string name;           ///< Human-readable name
        std::string description;    ///< Detailed description
        Timestamp created_at;       ///< Creation timestamp
        Timestamp updated_at;       ///< Last update timestamp
        State state;                ///< Current node state
        std::map<std::string, std::string> attributes; ///< Custom attributes
    };

    // Constructors
    Node() = default;
    Node(NodeID id, Type type, const std::string& name, const std::string& description = "");
    
    // Copy constructor and assignment
    Node(const Node& other) = default;
    Node& operator=(const Node& other) = default;
    
    // Move constructor and assignment
    Node(Node&& other) noexcept = default;
    Node& operator=(Node&& other) noexcept = default;
    
    // Destructor
    ~Node() = default;

    // Getters
    NodeID get_id() const { return id_; }
    Type get_type() const { return type_; }
    const Metadata& get_metadata() const { return metadata_; }
    Metadata& get_metadata() { return metadata_; }
    
    // Setters
    void set_name(const std::string& name);
    void set_description(const std::string& description);
    void set_state(State state);
    void set_attribute(const std::string& key, const std::string& value);
    void remove_attribute(const std::string& key);
    
    // State management
    bool is_active() const { return metadata_.state == State::ACTIVE; }
    bool is_inactive() const { return metadata_.state == State::INACTIVE; }
    bool is_error() const { return metadata_.state == State::ERROR; }
    
    // Utility methods
    void update_timestamp();
    std::string get_type_string() const;
    std::string get_state_string() const;
    
    // Serialization
    nlohmann::json to_json() const;
    static Result<Node> from_json(const nlohmann::json& json);
    
    // Comparison operators
    bool operator==(const Node& other) const { return id_ == other.id_; }
    bool operator!=(const Node& other) const { return id_ != other.id_; }
    bool operator<(const Node& other) const { return id_ < other.id_; }

private:
    NodeID id_;                     ///< Unique node identifier
    Type type_;                     ///< Node type (input, concept, output)
    Metadata metadata_;             ///< Node metadata and state
};

// Utility functions
std::string node_type_to_string(Node::Type type);
Node::Type string_to_node_type(const std::string& str);
std::string node_state_to_string(Node::State state);
Node::State string_to_node_state(const std::string& str);

} // namespace melvin

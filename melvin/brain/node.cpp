#include "node.hpp"
#include "logger.hpp"
#include <sstream>
#include <iomanip>

namespace melvin {

Node::Node(NodeID id, Type type, const std::string& name, const std::string& description)
    : id_(id), type_(type) {
    auto now = std::chrono::system_clock::now();
    metadata_.name = name;
    metadata_.description = description;
    metadata_.created_at = now;
    metadata_.updated_at = now;
    metadata_.state = State::ACTIVE;
    
    LOG_DEBUG("Created node: " + std::to_string(id) + " (" + name + ")", "Node");
}

void Node::set_name(const std::string& name) {
    metadata_.name = name;
    update_timestamp();
    LOG_DEBUG("Updated node name: " + std::to_string(id_) + " -> " + name, "Node");
}

void Node::set_description(const std::string& description) {
    metadata_.description = description;
    update_timestamp();
    LOG_DEBUG("Updated node description: " + std::to_string(id_), "Node");
}

void Node::set_state(State state) {
    auto old_state = metadata_.state;
    metadata_.state = state;
    update_timestamp();
    LOG_INFO("Node state changed: " + std::to_string(id_) + " " + 
             node_state_to_string(old_state) + " -> " + node_state_to_string(state), "Node");
}

void Node::set_attribute(const std::string& key, const std::string& value) {
    metadata_.attributes[key] = value;
    update_timestamp();
    LOG_DEBUG("Set node attribute: " + std::to_string(id_) + " " + key + " = " + value, "Node");
}

void Node::remove_attribute(const std::string& key) {
    auto it = metadata_.attributes.find(key);
    if (it != metadata_.attributes.end()) {
        metadata_.attributes.erase(it);
        update_timestamp();
        LOG_DEBUG("Removed node attribute: " + std::to_string(id_) + " " + key, "Node");
    }
}

void Node::update_timestamp() {
    metadata_.updated_at = std::chrono::system_clock::now();
}

std::string Node::get_type_string() const {
    return node_type_to_string(type_);
}

std::string Node::get_state_string() const {
    return node_state_to_string(metadata_.state);
}

nlohmann::json Node::to_json() const {
    nlohmann::json j;
    j["id"] = id_;
    j["type"] = get_type_string();
    j["metadata"] = {
        {"name", metadata_.name},
        {"description", metadata_.description},
        {"created_at", std::chrono::duration_cast<std::chrono::milliseconds>(
            metadata_.created_at.time_since_epoch()).count()},
        {"updated_at", std::chrono::duration_cast<std::chrono::milliseconds>(
            metadata_.updated_at.time_since_epoch()).count()},
        {"state", get_state_string()},
        {"attributes", metadata_.attributes}
    };
    return j;
}

Result<Node> Node::from_json(const nlohmann::json& json) {
    try {
        if (!json.contains("id") || !json.contains("type") || !json.contains("metadata")) {
            return Error<Node>(ErrorCode::INVALID_DATA, "Missing required fields in JSON");
        }
        
        NodeID id = json["id"];
        std::string type_str = json["type"];
        auto type = string_to_node_type(type_str);
        
        if (type == Node::Type::INPUT && !json["metadata"].contains("name")) {
            return Error<Node>(ErrorCode::INVALID_DATA, "Missing name in metadata");
        }
        
        Node node(id, type, json["metadata"]["name"], json["metadata"]["description"]);
        
        // Parse timestamps
        if (json["metadata"].contains("created_at")) {
            auto created_ms = json["metadata"]["created_at"].get<int64_t>();
            node.metadata_.created_at = std::chrono::system_clock::time_point(
                std::chrono::milliseconds(created_ms));
        }
        
        if (json["metadata"].contains("updated_at")) {
            auto updated_ms = json["metadata"]["updated_at"].get<int64_t>();
            node.metadata_.updated_at = std::chrono::system_clock::time_point(
                std::chrono::milliseconds(updated_ms));
        }
        
        // Parse state
        if (json["metadata"].contains("state")) {
            node.metadata_.state = string_to_node_state(json["metadata"]["state"]);
        }
        
        // Parse attributes
        if (json["metadata"].contains("attributes")) {
            node.metadata_.attributes = json["metadata"]["attributes"].get<std::map<std::string, std::string>>();
        }
        
        LOG_DEBUG("Deserialized node from JSON: " + std::to_string(id), "Node");
        return Success(node);
        
    } catch (const std::exception& e) {
        return Error<Node>(ErrorCode::INVALID_DATA, "JSON parsing error: " + std::string(e.what()));
    }
}

// Utility functions
std::string node_type_to_string(Node::Type type) {
    switch (type) {
        case Node::Type::INPUT: return "input";
        case Node::Type::CONCEPT: return "concept";
        case Node::Type::OUTPUT: return "output";
        default: return "unknown";
    }
}

Node::Type string_to_node_type(const std::string& str) {
    if (str == "input") return Node::Type::INPUT;
    if (str == "concept") return Node::Type::CONCEPT;
    if (str == "output") return Node::Type::OUTPUT;
    return Node::Type::CONCEPT; // Default fallback
}

std::string node_state_to_string(Node::State state) {
    switch (state) {
        case Node::State::ACTIVE: return "active";
        case Node::State::INACTIVE: return "inactive";
        case Node::State::ERROR: return "error";
        default: return "unknown";
    }
}

Node::State string_to_node_state(const std::string& str) {
    if (str == "active") return Node::State::ACTIVE;
    if (str == "inactive") return Node::State::INACTIVE;
    if (str == "error") return Node::State::ERROR;
    return Node::State::INACTIVE; // Default fallback
}

} // namespace melvin

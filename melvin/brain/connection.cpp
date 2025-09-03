#include "connection.hpp"
#include "logger.hpp"
#include <cmath>
#include <algorithm>

namespace melvin {

Connection::Connection(ConnectionID id, NodeID source_id, NodeID target_id, 
                     Weight weight, Confidence confidence, Type type)
    : id_(id), source_id_(source_id), target_id_(target_id) {
    auto now = std::chrono::system_clock::now();
    metadata_.weight = std::clamp(weight, -1.0, 1.0);
    metadata_.confidence = std::clamp(confidence, 0.0, 1.0);
    metadata_.type = type;
    metadata_.created_at = now;
    metadata_.updated_at = now;
    metadata_.last_used = now;
    metadata_.usage_count = 0;
    
    LOG_DEBUG("Created connection: " + std::to_string(id) + " (" + 
              std::to_string(source_id) + " -> " + std::to_string(target_id) + ")", "Connection");
}

void Connection::set_weight(Weight weight) {
    auto old_weight = metadata_.weight;
    metadata_.weight = std::clamp(weight, -1.0, 1.0);
    update_timestamp();
    LOG_DEBUG("Updated connection weight: " + std::to_string(id_) + " " + 
              std::to_string(old_weight) + " -> " + std::to_string(metadata_.weight), "Connection");
}

void Connection::set_confidence(Confidence confidence) {
    auto old_confidence = metadata_.confidence;
    metadata_.confidence = std::clamp(confidence, 0.0, 1.0);
    update_timestamp();
    LOG_DEBUG("Updated connection confidence: " + std::to_string(id_) + " " + 
              std::to_string(old_confidence) + " -> " + std::to_string(metadata_.confidence), "Connection");
}

void Connection::set_type(Type type) {
    auto old_type = metadata_.type;
    metadata_.type = type;
    update_timestamp();
    LOG_DEBUG("Updated connection type: " + std::to_string(id_) + " " + 
              connection_type_to_string(old_type) + " -> " + connection_type_to_string(type), "Connection");
}

void Connection::set_attribute(const std::string& key, const std::string& value) {
    metadata_.attributes[key] = value;
    update_timestamp();
    LOG_DEBUG("Set connection attribute: " + std::to_string(id_) + " " + key + " = " + value, "Connection");
}

void Connection::remove_attribute(const std::string& key) {
    auto it = metadata_.attributes.find(key);
    if (it != metadata_.attributes.end()) {
        metadata_.attributes.erase(it);
        update_timestamp();
        LOG_DEBUG("Removed connection attribute: " + std::to_string(id_) + " " + key, "Connection");
    }
}

void Connection::update_confidence(Confidence delta) {
    auto old_confidence = metadata_.confidence;
    metadata_.confidence = std::clamp(metadata_.confidence + delta, 0.0, 1.0);
    update_timestamp();
    LOG_DEBUG("Updated connection confidence: " + std::to_string(id_) + " " + 
              std::to_string(old_confidence) + " + " + std::to_string(delta) + " = " + 
              std::to_string(metadata_.confidence), "Connection");
}

void Connection::update_weight(Weight delta) {
    auto old_weight = metadata_.weight;
    metadata_.weight = std::clamp(metadata_.weight + delta, -1.0, 1.0);
    update_timestamp();
    LOG_DEBUG("Updated connection weight: " + std::to_string(id_) + " " + 
              std::to_string(old_weight) + " + " + std::to_string(delta) + " = " + 
              std::to_string(metadata_.weight), "Connection");
}

void Connection::mark_used() {
    metadata_.last_used = std::chrono::system_clock::now();
    metadata_.usage_count++;
    update_timestamp();
    LOG_DEBUG("Marked connection as used: " + std::to_string(id_) + " (count: " + 
              std::to_string(metadata_.usage_count) + ")", "Connection");
}

void Connection::decay_confidence(double decay_rate) {
    if (metadata_.type == Type::PERSISTENT) {
        return; // Persistent connections don't decay
    }
    
    auto old_confidence = metadata_.confidence;
    metadata_.confidence = std::max(0.0, metadata_.confidence - decay_rate);
    update_timestamp();
    
    if (metadata_.confidence != old_confidence) {
        LOG_DEBUG("Decayed connection confidence: " + std::to_string(id_) + " " + 
                  std::to_string(old_confidence) + " -> " + std::to_string(metadata_.confidence), "Connection");
    }
}

double Connection::get_age_hours() const {
    auto now = std::chrono::system_clock::now();
    auto duration = now - metadata_.created_at;
    return std::chrono::duration_cast<std::chrono::hours>(duration).count();
}

void Connection::update_timestamp() {
    metadata_.updated_at = std::chrono::system_clock::now();
}

nlohmann::json Connection::to_json() const {
    nlohmann::json j;
    j["id"] = id_;
    j["source_id"] = source_id_;
    j["target_id"] = target_id_;
    j["metadata"] = {
        {"weight", metadata_.weight},
        {"confidence", metadata_.confidence},
        {"type", connection_type_to_string(metadata_.type)},
        {"created_at", std::chrono::duration_cast<std::chrono::milliseconds>(
            metadata_.created_at.time_since_epoch()).count()},
        {"updated_at", std::chrono::duration_cast<std::chrono::milliseconds>(
            metadata_.updated_at.time_since_epoch()).count()},
        {"last_used", std::chrono::duration_cast<std::chrono::milliseconds>(
            metadata_.last_used.time_since_epoch()).count()},
        {"usage_count", metadata_.usage_count},
        {"attributes", metadata_.attributes}
    };
    return j;
}

Result<Connection> Connection::from_json(const nlohmann::json& json) {
    try {
        if (!json.contains("id") || !json.contains("source_id") || !json.contains("target_id") || 
            !json.contains("metadata")) {
            return Error<Connection>(ErrorCode::INVALID_DATA, "Missing required fields in JSON");
        }
        
        ConnectionID id = json["id"];
        NodeID source_id = json["source_id"];
        NodeID target_id = json["target_id"];
        
        if (!json["metadata"].contains("weight") || !json["metadata"].contains("confidence")) {
            return Error<Connection>(ErrorCode::INVALID_DATA, "Missing weight or confidence in metadata");
        }
        
        Weight weight = json["metadata"]["weight"];
        Confidence confidence = json["metadata"]["confidence"];
        Type type = Type::DIRECT;
        
        if (json["metadata"].contains("type")) {
            type = string_to_connection_type(json["metadata"]["type"]);
        }
        
        Connection conn(id, source_id, target_id, weight, confidence, type);
        
        // Parse timestamps
        if (json["metadata"].contains("created_at")) {
            auto created_ms = json["metadata"]["created_at"].get<int64_t>();
            conn.metadata_.created_at = std::chrono::system_clock::time_point(
                std::chrono::milliseconds(created_ms));
        }
        
        if (json["metadata"].contains("updated_at")) {
            auto updated_ms = json["metadata"]["updated_at"].get<int64_t>();
            conn.metadata_.updated_at = std::chrono::system_clock::time_point(
                std::chrono::milliseconds(updated_ms));
        }
        
        if (json["metadata"].contains("last_used")) {
            auto last_used_ms = json["metadata"]["last_used"].get<int64_t>();
            conn.metadata_.last_used = std::chrono::system_clock::time_point(
                std::chrono::milliseconds(last_used_ms));
        }
        
        // Parse usage count
        if (json["metadata"].contains("usage_count")) {
            conn.metadata_.usage_count = json["metadata"]["usage_count"];
        }
        
        // Parse attributes
        if (json["metadata"].contains("attributes")) {
            conn.metadata_.attributes = json["metadata"]["attributes"].get<std::map<std::string, std::string>>();
        }
        
        LOG_DEBUG("Deserialized connection from JSON: " + std::to_string(id), "Connection");
        return Success(conn);
        
    } catch (const std::exception& e) {
        return Error<Connection>(ErrorCode::INVALID_DATA, "JSON parsing error: " + std::string(e.what()));
    }
}

// Utility functions
std::string connection_type_to_string(Connection::Type type) {
    switch (type) {
        case Connection::Type::DIRECT: return "direct";
        case Connection::Type::INFERRED: return "inferred";
        case Connection::Type::TEMPORARY: return "temporary";
        case Connection::Type::PERSISTENT: return "persistent";
        default: return "unknown";
    }
}

Connection::Type string_to_connection_type(const std::string& str) {
    if (str == "direct") return Connection::Type::DIRECT;
    if (str == "inferred") return Connection::Type::INFERRED;
    if (str == "temporary") return Connection::Type::TEMPORARY;
    if (str == "persistent") return Connection::Type::PERSISTENT;
    return Connection::Type::DIRECT; // Default fallback
}

bool is_valid_weight(Weight weight) {
    return weight >= -1.0 && weight <= 1.0;
}

bool is_valid_confidence(Confidence confidence) {
    return confidence >= 0.0 && confidence <= 1.0;
}

} // namespace melvin

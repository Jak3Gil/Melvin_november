#include <gtest/gtest.h>
#include "node.hpp"
#include "logger.hpp"
#include <chrono>

using namespace melvin;

class NodeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize logger for tests
        Logger::instance().init("", LogLevel::DEBUG);
    }
    
    void TearDown() override {
        Logger::instance().shutdown();
    }
};

TEST_F(NodeTest, Constructor) {
    NodeID id = 123;
    std::string name = "Test Node";
    std::string description = "A test node for unit testing";
    
    Node node(id, Node::Type::CONCEPT, name, description);
    
    EXPECT_EQ(node.get_id(), id);
    EXPECT_EQ(node.get_type(), Node::Type::CONCEPT);
    EXPECT_EQ(node.get_metadata().name, name);
    EXPECT_EQ(node.get_metadata().description, description);
    EXPECT_EQ(node.get_metadata().state, Node::State::ACTIVE);
    EXPECT_TRUE(node.is_active());
}

TEST_F(NodeTest, TypeConversion) {
    // Test type to string conversion
    EXPECT_EQ(node_type_to_string(Node::Type::INPUT), "input");
    EXPECT_EQ(node_type_to_string(Node::Type::CONCEPT), "concept");
    EXPECT_EQ(node_type_to_string(Node::Type::OUTPUT), "output");
    
    // Test string to type conversion
    EXPECT_EQ(string_to_node_type("input"), Node::Type::INPUT);
    EXPECT_EQ(string_to_node_type("concept"), Node::Type::CONCEPT);
    EXPECT_EQ(string_to_node_type("output"), Node::Type::OUTPUT);
    EXPECT_EQ(string_to_node_type("unknown"), Node::Type::CONCEPT); // Default fallback
}

TEST_F(NodeTest, StateConversion) {
    // Test state to string conversion
    EXPECT_EQ(node_state_to_string(Node::State::ACTIVE), "active");
    EXPECT_EQ(node_state_to_string(Node::State::INACTIVE), "inactive");
    EXPECT_EQ(node_state_to_string(Node::State::ERROR), "error");
    
    // Test string to state conversion
    EXPECT_EQ(string_to_node_state("active"), Node::State::ACTIVE);
    EXPECT_EQ(string_to_node_state("inactive"), Node::State::INACTIVE);
    EXPECT_EQ(string_to_node_state("error"), Node::State::ERROR);
    EXPECT_EQ(string_to_node_state("unknown"), Node::State::INACTIVE); // Default fallback
}

TEST_F(NodeTest, StateManagement) {
    Node node(1, Node::Type::INPUT, "Test Node");
    
    // Test initial state
    EXPECT_TRUE(node.is_active());
    EXPECT_FALSE(node.is_inactive());
    EXPECT_FALSE(node.is_error());
    
    // Test state changes
    node.set_state(Node::State::INACTIVE);
    EXPECT_FALSE(node.is_active());
    EXPECT_TRUE(node.is_inactive());
    EXPECT_FALSE(node.is_error());
    
    node.set_state(Node::State::ERROR);
    EXPECT_FALSE(node.is_active());
    EXPECT_FALSE(node.is_inactive());
    EXPECT_TRUE(node.is_error());
}

TEST_F(NodeTest, AttributeManagement) {
    Node node(1, Node::Type::CONCEPT, "Test Node");
    
    // Test setting attributes
    node.set_attribute("key1", "value1");
    node.set_attribute("key2", "value2");
    
    EXPECT_EQ(node.get_metadata().attributes["key1"], "value1");
    EXPECT_EQ(node.get_metadata().attributes["key2"], "value2");
    EXPECT_EQ(node.get_metadata().attributes.size(), 2);
    
    // Test removing attributes
    node.remove_attribute("key1");
    EXPECT_EQ(node.get_metadata().attributes.size(), 1);
    EXPECT_EQ(node.get_metadata().attributes["key2"], "value2");
    
    // Test removing non-existent attribute
    node.remove_attribute("nonexistent");
    EXPECT_EQ(node.get_metadata().attributes.size(), 1); // Should remain unchanged
}

TEST_F(NodeTest, TimestampUpdates) {
    Node node(1, Node::Type::OUTPUT, "Test Node");
    
    auto original_updated = node.get_metadata().updated_at;
    
    // Wait a bit to ensure timestamp difference
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    // Update node
    node.set_name("Updated Name");
    
    auto new_updated = node.get_metadata().updated_at;
    EXPECT_GT(new_updated, original_updated);
}

TEST_F(NodeTest, JSONSerialization) {
    Node original_node(123, Node::Type::INPUT, "Test Node", "Test Description");
    original_node.set_attribute("test_attr", "test_value");
    original_node.set_state(Node::State::ACTIVE);
    
    // Test serialization
    nlohmann::json json = original_node.to_json();
    
    EXPECT_EQ(json["id"], 123);
    EXPECT_EQ(json["type"], "input");
    EXPECT_EQ(json["metadata"]["name"], "Test Node");
    EXPECT_EQ(json["metadata"]["description"], "Test Description");
    EXPECT_EQ(json["metadata"]["state"], "active");
    EXPECT_EQ(json["metadata"]["attributes"]["test_attr"], "test_value");
    
    // Test deserialization
    auto result = Node::from_json(json);
    EXPECT_TRUE(result.is_success());
    
    Node deserialized_node = result.value();
    EXPECT_EQ(deserialized_node.get_id(), original_node.get_id());
    EXPECT_EQ(deserialized_node.get_type(), original_node.get_type());
    EXPECT_EQ(deserialized_node.get_metadata().name, original_node.get_metadata().name);
    EXPECT_EQ(deserialized_node.get_metadata().description, original_node.get_metadata().description);
    EXPECT_EQ(deserialized_node.get_metadata().state, original_node.get_metadata().state);
    EXPECT_EQ(deserialized_node.get_metadata().attributes["test_attr"], "test_value");
}

TEST_F(NodeTest, JSONDeserializationErrors) {
    // Test missing required fields
    nlohmann::json invalid_json = {
        {"id", 123}
        // Missing type and metadata
    };
    
    auto result = Node::from_json(invalid_json);
    EXPECT_TRUE(result.is_error());
    EXPECT_EQ(result.error().code, ErrorCode::INVALID_DATA);
    
    // Test invalid JSON structure
    nlohmann::json malformed_json = {
        {"id", "not_a_number"}, // Invalid ID type
        {"type", "input"},
        {"metadata", {{"name", "Test"}}}
    };
    
    result = Node::from_json(malformed_json);
    EXPECT_TRUE(result.is_error());
}

TEST_F(NodeTest, ComparisonOperators) {
    Node node1(1, Node::Type::INPUT, "Node 1");
    Node node2(2, Node::Type::CONCEPT, "Node 2");
    Node node3(1, Node::Type::OUTPUT, "Node 3"); // Same ID as node1
    
    // Test equality
    EXPECT_TRUE(node1 == node3); // Same ID
    EXPECT_FALSE(node1 == node2); // Different ID
    
    // Test inequality
    EXPECT_TRUE(node1 != node2);
    EXPECT_FALSE(node1 != node3);
    
    // Test ordering
    EXPECT_TRUE(node1 < node2);
    EXPECT_FALSE(node2 < node1);
}

TEST_F(NodeTest, CopyAndMove) {
    Node original(1, Node::Type::CONCEPT, "Original");
    original.set_attribute("key", "value");
    
    // Test copy constructor
    Node copied(original);
    EXPECT_EQ(copied.get_id(), original.get_id());
    EXPECT_EQ(copied.get_metadata().attributes["key"], "value");
    
    // Test copy assignment
    Node assigned(2, Node::Type::INPUT, "Assigned");
    assigned = original;
    EXPECT_EQ(assigned.get_id(), original.get_id());
    EXPECT_EQ(assigned.get_metadata().attributes["key"], "value");
    
    // Test move constructor
    Node moved_from(3, Node::Type::OUTPUT, "Moved From");
    moved_from.set_attribute("move_key", "move_value");
    Node moved_to(std::move(moved_from));
    EXPECT_EQ(moved_to.get_metadata().attributes["move_key"], "move_value");
    
    // Test move assignment
    Node move_assigned(4, Node::Type::INPUT, "Move Assigned");
    move_assigned = std::move(moved_to);
    EXPECT_EQ(move_assigned.get_metadata().attributes["move_key"], "move_value");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

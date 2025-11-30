/*
 * GoogleTest Test Suite for Melvin Curiosity Learning System
 * 
 * Tests:
 * - Concept extraction functionality
 * - Binary persistence and retrieval
 * - Knowledge graph operations
 * - Self-check and contradiction detection
 * - Encrypted storage operations
 * - Ollama API integration
 */

#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>
#include "melvin_curiosity_learning.cpp"
#include "self_check_system.cpp"
#include "encrypted_storage.h"
#include "ollama_client.h"

namespace MelvinTests {

// Test fixture for concept extraction
class ConceptExtractionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize test data
    }
    
    void TearDown() override {
        // Clean up test data
    }
};

// Test concept extraction from various question formats
TEST_F(ConceptExtractionTest, BasicQuestionFormats) {
    MelvinLearningSystem melvin;
    
    // Test "What is X?" format
    std::string concept1 = melvin.extractConceptFromQuestion("What is a cat?");
    EXPECT_EQ(concept1, "cat");
    
    // Test "What's X?" format
    std::string concept2 = melvin.extractConceptFromQuestion("What's a dog?");
    EXPECT_EQ(concept2, "dog");
    
    // Test with articles
    std::string concept3 = melvin.extractConceptFromQuestion("What is an elephant?");
    EXPECT_EQ(concept3, "elephant");
    
    // Test with "the"
    std::string concept4 = melvin.extractConceptFromQuestion("What is the sun?");
    EXPECT_EQ(concept4, "sun");
}

TEST_F(ConceptExtractionTest, ComplexQuestionFormats) {
    MelvinLearningSystem melvin;
    
    // Test longer questions
    std::string concept1 = melvin.extractConceptFromQuestion("What is a computer program?");
    EXPECT_EQ(concept1, "computer");
    
    // Test questions with multiple words
    std::string concept2 = melvin.extractConceptFromQuestion("What is artificial intelligence?");
    EXPECT_EQ(concept2, "artificial");
    
    // Test questions with punctuation
    std::string concept3 = melvin.extractConceptFromQuestion("What is a car, exactly?");
    EXPECT_EQ(concept3, "car");
}

// Test fixture for binary persistence
class BinaryPersistenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_file_ = "test_knowledge.bin";
        // Remove test file if it exists
        std::filesystem::remove(test_file_);
    }
    
    void TearDown() override {
        // Clean up test file
        std::filesystem::remove(test_file_);
    }
    
    std::string test_file_;
};

TEST_F(BinaryPersistenceTest, SaveAndLoadEmptyGraph) {
    BinaryKnowledgeStorage storage(test_file_);
    
    // Save empty graph
    storage.saveKnowledge();
    
    // Verify file exists
    EXPECT_TRUE(std::filesystem::exists(test_file_));
    
    // Load into new storage
    BinaryKnowledgeStorage storage2(test_file_);
    
    // Verify it's empty
    EXPECT_EQ(storage2.getNodeCount(), 0);
}

TEST_F(BinaryPersistenceTest, SaveAndLoadSingleNode) {
    BinaryKnowledgeStorage storage(test_file_);
    
    // Create a test node
    auto node = std::make_shared<KnowledgeNode>(1, "test", "A test concept");
    storage.addNode(node);
    
    // Save and reload
    storage.saveKnowledge();
    BinaryKnowledgeStorage storage2(test_file_);
    
    // Verify node was loaded
    EXPECT_EQ(storage2.getNodeCount(), 1);
    auto loaded_node = storage2.findConcept("test");
    EXPECT_NE(loaded_node, nullptr);
    EXPECT_EQ(std::string(loaded_node->concept), "test");
    EXPECT_EQ(std::string(loaded_node->definition), "A test concept");
}

TEST_F(BinaryPersistenceTest, SaveAndLoadMultipleNodes) {
    BinaryKnowledgeStorage storage(test_file_);
    
    // Create multiple test nodes
    auto node1 = std::make_shared<KnowledgeNode>(1, "cat", "A small mammal");
    auto node2 = std::make_shared<KnowledgeNode>(2, "dog", "A loyal pet");
    auto node3 = std::make_shared<KnowledgeNode>(3, "bird", "A flying animal");
    
    storage.addNode(node1);
    storage.addNode(node2);
    storage.addNode(node3);
    
    // Save and reload
    storage.saveKnowledge();
    BinaryKnowledgeStorage storage2(test_file_);
    
    // Verify all nodes were loaded
    EXPECT_EQ(storage2.getNodeCount(), 3);
    EXPECT_NE(storage2.findConcept("cat"), nullptr);
    EXPECT_NE(storage2.findConcept("dog"), nullptr);
    EXPECT_NE(storage2.findConcept("bird"), nullptr);
}

TEST_F(BinaryPersistenceTest, BinaryFileFormat) {
    BinaryKnowledgeStorage storage(test_file_);
    
    // Create a test node
    auto node = std::make_shared<KnowledgeNode>(1, "test", "A test concept");
    storage.addNode(node);
    storage.saveKnowledge();
    
    // Verify file is binary (not text)
    std::ifstream file(test_file_, std::ios::binary);
    EXPECT_TRUE(file.is_open());
    
    char first_bytes[4];
    file.read(first_bytes, 4);
    
    // First 4 bytes should be the node count (uint32_t)
    uint32_t node_count = *reinterpret_cast<uint32_t*>(first_bytes);
    EXPECT_EQ(node_count, 1);
}

// Test fixture for knowledge graph operations
class KnowledgeGraphTest : public ::testing::Test {
protected:
    void SetUp() override {
        melvin = std::make_unique<MelvinLearningSystem>();
    }
    
    void TearDown() override {
        // Clean up
        std::filesystem::remove("melvin_knowledge.bin");
    }
    
    std::unique_ptr<MelvinLearningSystem> melvin;
};

TEST_F(KnowledgeGraphTest, CreateAndRetrieveNode) {
    // Create a node
    auto node = melvin->createNode("test", "A test concept");
    EXPECT_NE(node, nullptr);
    EXPECT_EQ(std::string(node->concept), "test");
    EXPECT_EQ(std::string(node->definition), "A test concept");
}

TEST_F(KnowledgeGraphTest, KnowledgeRetrieval) {
    // Add knowledge
    melvin->curiosityLoop("What is a cat?");
    
    // Check if Melvin knows about cats
    EXPECT_TRUE(melvin->melvinKnows("What is a cat?"));
    
    // Retrieve answer
    std::string answer = melvin->melvinAnswer("What is a cat?");
    EXPECT_FALSE(answer.empty());
    EXPECT_NE(answer, "I don't know the answer to that question.");
}

TEST_F(KnowledgeGraphTest, LearningStatistics) {
    // Add some knowledge
    melvin->curiosityLoop("What is a cat?");
    melvin->curiosityLoop("What is a dog?");
    melvin->curiosityLoop("What is a cat?"); // Repeat to test retrieval
    
    auto stats = melvin->getLearningStats();
    EXPECT_GE(stats.questions_asked, 3);
    EXPECT_GE(stats.new_concepts_learned, 2);
    EXPECT_GE(stats.concepts_retrieved, 1);
}

// Test fixture for self-check system
class SelfCheckTest : public ::testing::Test {
protected:
    void SetUp() override {
        self_check = std::make_unique<MelvinSelfCheck::SelfCheckSystem>();
    }
    
    std::unique_ptr<MelvinSelfCheck::SelfCheckSystem> self_check;
};

TEST_F(SelfCheckTest, NoContradictionDetection) {
    // Create nodes that don't contradict
    KnowledgeNode node1(1, "cat", "A small domesticated mammal");
    KnowledgeNode node2(2, "dog", "A loyal domesticated mammal");
    
    std::vector<std::shared_ptr<KnowledgeNode>> existing_nodes;
    existing_nodes.push_back(std::make_shared<KnowledgeNode>(node2));
    
    auto result = self_check->performSelfCheck(node1, existing_nodes);
    EXPECT_TRUE(result.should_accept_new_knowledge);
    EXPECT_TRUE(result.contradictions_found.empty());
}

TEST_F(SelfCheckTest, DirectContradictionDetection) {
    // Create contradictory nodes
    KnowledgeNode node1(1, "cat", "A cat is a mammal");
    KnowledgeNode node2(2, "cat", "A cat is not a mammal");
    
    std::vector<std::shared_ptr<KnowledgeNode>> existing_nodes;
    existing_nodes.push_back(std::make_shared<KnowledgeNode>(node2));
    
    auto result = self_check->performSelfCheck(node1, existing_nodes);
    EXPECT_FALSE(result.contradictions_found.empty());
    EXPECT_EQ(result.contradictions_found[0].type, 
              MelvinSelfCheck::ContradictionType::DIRECT_CONTRADICTION);
}

TEST_F(SelfCheckTest, SemanticConflictDetection) {
    // Create semantically conflicting nodes
    KnowledgeNode node1(1, "weather", "It is hot today");
    KnowledgeNode node2(2, "weather", "It is cold today");
    
    std::vector<std::shared_ptr<KnowledgeNode>> existing_nodes;
    existing_nodes.push_back(std::make_shared<KnowledgeNode>(node2));
    
    auto result = self_check->performSelfCheck(node1, existing_nodes);
    EXPECT_FALSE(result.contradictions_found.empty());
    EXPECT_EQ(result.contradictions_found[0].type, 
              MelvinSelfCheck::ContradictionType::SEMANTIC_CONFLICT);
}

// Test fixture for Ollama client
class OllamaClientTest : public ::testing::Test {
protected:
    void SetUp() override {
        MelvinOllama::OllamaConfig config;
        config.base_url = "http://localhost:11434";
        config.model = "llama2";
        config.max_retries = 1; // Reduce retries for testing
        config.request_timeout_seconds = 5;
        
        client = std::make_unique<MelvinOllama::OllamaClient>(config);
    }
    
    std::unique_ptr<MelvinOllama::OllamaClient> client;
};

TEST_F(OllamaClientTest, ClientInitialization) {
    EXPECT_NE(client, nullptr);
    // Note: This test will pass even if Ollama is not running
    // The actual API test would require a running Ollama instance
}

TEST_F(OllamaClientTest, ConfigurationUpdate) {
    MelvinOllama::OllamaConfig new_config;
    new_config.model = "llama3";
    new_config.max_retries = 2;
    
    client->updateConfig(new_config);
    
    auto status = client->getStatus();
    EXPECT_EQ(status["model"], "llama3");
}

// Test fixture for encrypted storage
class EncryptedStorageTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_file_ = "test_encrypted.bin";
        std::filesystem::remove(test_file_);
        
        MelvinCrypto::EncryptionConfig config;
        config.password = "test_password_123";
        config.salt_file = ".test_salt";
        
        storage = std::make_unique<MelvinCrypto::EncryptedBinaryStorage>(test_file_, config);
    }
    
    void TearDown() override {
        std::filesystem::remove(test_file_);
        std::filesystem::remove(".test_salt");
    }
    
    std::string test_file_;
    std::unique_ptr<MelvinCrypto::EncryptedBinaryStorage> storage;
};

TEST_F(EncryptedStorageTest, Initialization) {
    EXPECT_TRUE(storage->initialize("test_password_123"));
    EXPECT_TRUE(storage->isInitialized());
}

TEST_F(EncryptedStorageTest, EncryptAndDecryptData) {
    storage->initialize("test_password_123");
    
    // Test data
    std::vector<uint8_t> test_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // Encrypt and save
    EXPECT_TRUE(storage->saveEncryptedData(test_data));
    
    // Load and decrypt
    auto result = storage->loadEncryptedData();
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.data, test_data);
}

TEST_F(EncryptedStorageTest, IntegrityVerification) {
    storage->initialize("test_password_123");
    
    // Test data
    std::vector<uint8_t> test_data = {1, 2, 3, 4, 5};
    EXPECT_TRUE(storage->saveEncryptedData(test_data));
    
    // Verify integrity
    EXPECT_TRUE(storage->verifyIntegrity());
}

// Integration tests
class IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Clean up any existing files
        std::filesystem::remove("melvin_knowledge.bin");
        std::filesystem::remove("melvin_knowledge_encrypted.bin");
    }
    
    void TearDown() override {
        // Clean up
        std::filesystem::remove("melvin_knowledge.bin");
        std::filesystem::remove("melvin_knowledge_encrypted.bin");
    }
};

TEST_F(IntegrationTest, FullLearningCycle) {
    MelvinLearningSystem melvin;
    
    // Learn something new
    std::string answer1 = melvin.curiosityLoop("What is a cat?");
    EXPECT_FALSE(answer1.empty());
    
    // Retrieve from memory
    std::string answer2 = melvin.curiosityLoop("What is a cat?");
    EXPECT_FALSE(answer2.empty());
    
    // Should be the same answer (from memory)
    EXPECT_EQ(answer1, answer2);
}

TEST_F(IntegrationTest, LearningWithSelfCheck) {
    MelvinSelfCheck::MelvinLearningSystemWithSelfCheck melvin;
    
    // Learn with self-check
    std::string answer = melvin.curiosityLoopWithSelfCheck("What is a cat?");
    EXPECT_FALSE(answer.empty());
    
    // Check self-check statistics
    auto stats = melvin.getSelfCheckStats();
    EXPECT_GE(stats.total_checks, 1);
}

TEST_F(IntegrationTest, EncryptedLearning) {
    MelvinCrypto::MelvinLearningSystemWithEncryption melvin("test_password");
    
    // Learn something
    std::string answer = melvin.curiosityLoopSecure("What is a dog?");
    EXPECT_FALSE(answer.empty());
    
    // Verify encryption is enabled
    EXPECT_TRUE(melvin.isEncryptionEnabled());
}

} // namespace MelvinTests

// Main test runner
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

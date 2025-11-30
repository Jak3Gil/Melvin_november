# Melvin Advanced Curiosity Learning System 🧠⚡🔒

A comprehensive curiosity-driven learning module for Melvin with **real Ollama API integration**, **self-check contradiction detection**, **encrypted binary storage**, and **comprehensive testing**.

## 🎯 Overview

This advanced system extends the basic Melvin curiosity learning with enterprise-grade features:

- **🔗 Real Ollama API Integration**: HTTP client with retries, rate limiting, and async support
- **🔍 Self-Check & Contradiction Detection**: Validates new knowledge against existing graph
- **🔒 Encrypted Binary Storage**: AES-256 encryption with HMAC integrity verification
- **🧪 Comprehensive Testing**: GoogleTest suite with CI/CD integration

## ✨ Advanced Features

### 🔗 Real Ollama API Integration

```cpp
// Configure Ollama client
MelvinOllama::OllamaConfig config;
config.base_url = "http://localhost:11434";
config.model = "llama2";
config.max_retries = 3;
config.rate_limit_requests_per_minute = 60;

// Create client
auto client = std::make_unique<MelvinOllama::OllamaClient>(config);

// Synchronous API call
auto response = client->askQuestion("What is a cat?");

// Asynchronous API call
auto future_response = client->askQuestionAsync("What is a dog?");
auto result = future_response.get();
```

**Features:**
- ✅ HTTP client with connection pooling
- ✅ Exponential backoff retry logic
- ✅ Rate limiting and request queuing
- ✅ Async request handling
- ✅ JSON parsing and error handling
- ✅ Secure API key management

### 🔍 Self-Check & Contradiction Detection

```cpp
// Enhanced learning system with self-check
MelvinSelfCheck::MelvinLearningSystemWithSelfCheck melvin;

// Learn with automatic contradiction detection
std::string answer = melvin.curiosityLoopWithSelfCheck("What is a cat?");

// Get self-reflection results
auto stats = melvin.getSelfCheckStats();
std::cout << "Contradictions found: " << stats.contradictions_found << std::endl;
```

**Contradiction Types Detected:**
- ✅ **Direct Contradictions**: "A cat is a mammal" vs "A cat is not a mammal"
- ✅ **Semantic Conflicts**: "It is hot" vs "It is cold"
- ✅ **Logical Inconsistencies**: Self-contradictory definitions
- ✅ **Confidence Mismatches**: High confidence vs low confidence conflicts

**Self-Reflection Features:**
- ✅ Automatic contradiction analysis
- ✅ Confidence scoring and adjustment
- ✅ Clarification question generation
- ✅ Learning insights and recommendations

### 🔒 Encrypted Binary Storage

```cpp
// Configure encryption
MelvinCrypto::EncryptionConfig config;
config.password = "secure_password_123";
config.enable_compression = true;
config.enable_integrity_check = true;

// Create encrypted storage
auto storage = std::make_unique<MelvinCrypto::EncryptedBinaryStorage>(
    "melvin_knowledge_encrypted.bin", config);

// Initialize with password
storage->initialize("secure_password_123");

// Save encrypted data
std::vector<uint8_t> knowledge_data = getKnowledgeData();
storage->saveKnowledgeGraph(knowledge_data);

// Load and verify integrity
auto result = storage->loadKnowledgeGraph();
if (result.success && storage->verifyIntegrity()) {
    std::cout << "Knowledge loaded and verified!" << std::endl;
}
```

**Security Features:**
- ✅ **AES-256 Encryption**: Military-grade encryption
- ✅ **HMAC-SHA256 Signing**: Tamper detection
- ✅ **PBKDF2 Key Derivation**: Secure password hashing
- ✅ **Random Salt Generation**: Prevents rainbow table attacks
- ✅ **Secure Memory Handling**: Zero-clear sensitive data
- ✅ **Integrity Verification**: Detect tampering attempts

### 🧪 Comprehensive Testing

```bash
# Run all tests
./test_melvin_curiosity

# Run specific test categories
./test_melvin_curiosity --gtest_filter="ConceptExtractionTest.*"
./test_melvin_curiosity --gtest_filter="BinaryPersistenceTest.*"
./test_melvin_curiosity --gtest_filter="SelfCheckTest.*"
./test_melvin_curiosity --gtest_filter="EncryptedStorageTest.*"
```

**Test Coverage:**
- ✅ **Unit Tests**: Individual component testing
- ✅ **Integration Tests**: End-to-end system testing
- ✅ **Security Tests**: Encryption and integrity verification
- ✅ **Performance Tests**: Response time and memory usage
- ✅ **CI/CD Integration**: Automated testing on push

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  libcurl4-openssl-dev \
  libjsoncpp-dev \
  libssl-dev \
  libcrypto++-dev \
  libgtest-dev \
  cmake
```

### Building

```bash
# Build all advanced features
./build_advanced_curiosity.sh

# Or build individual components
./build_curiosity.sh                    # Basic system
g++ -std=c++17 -o ollama_test ollama_client.cpp -lcurl -ljsoncpp
g++ -std=c++17 -o self_check_test self_check_system.cpp
g++ -std=c++17 -o encrypted_test encrypted_storage.cpp -lssl -lcrypto
```

### Running

```bash
# Basic curiosity learning
./melvin_curiosity "What is a cat?"

# Comprehensive demo
./comprehensive_demo.sh

# Run tests
./test_melvin_curiosity

# Interactive learning with self-check
./melvin_curiosity "What is a dog?"
# Then ask: "What is a cat?" to test memory retrieval
```

## 📖 API Reference

### Ollama Client API

```cpp
class OllamaClient {
public:
    // Synchronous API
    OllamaResponse generate(const std::string& prompt);
    OllamaResponse askQuestion(const std::string& question);
    
    // Asynchronous API
    std::future<OllamaResponse> generateAsync(const std::string& prompt);
    std::future<OllamaResponse> askQuestionAsync(const std::string& question);
    
    // Health and status
    bool isHealthy() const;
    std::map<std::string, std::string> getStatus() const;
    Statistics getStatistics() const;
};
```

### Self-Check System API

```cpp
class SelfCheckSystem {
public:
    // Main self-check method
    SelfReflectionResult performSelfCheck(
        const KnowledgeNode& new_node,
        const std::vector<std::shared_ptr<KnowledgeNode>>& existing_nodes);
    
    // Individual contradiction detection
    std::vector<ContradictionAnalysis> detectAllContradictions(
        const KnowledgeNode& new_node,
        const std::vector<std::shared_ptr<KnowledgeNode>>& existing_nodes);
    
    // Configuration
    void setConfidenceThresholds(double high, double medium, double low);
    void addContradictionPattern(const std::string& category, 
                                const std::vector<std::string>& patterns);
};
```

### Encrypted Storage API

```cpp
class EncryptedBinaryStorage {
public:
    // Main storage operations
    bool saveEncryptedData(const std::vector<uint8_t>& data);
    CryptoResult loadEncryptedData();
    
    // Knowledge graph specific
    bool saveKnowledgeGraph(const std::vector<uint8_t>& graph_data);
    CryptoResult loadKnowledgeGraph();
    
    // Security operations
    bool changePassword(const std::string& old_password, 
                       const std::string& new_password);
    bool verifyIntegrity();
    void clearSecureData();
};
```

## 🔧 Configuration

### Environment Variables

```bash
# Ollama API configuration
export OLLAMA_API_KEY="your_api_key_here"
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="llama2"

# Encryption configuration
export MELVIN_PASSWORD="your_secure_password"
export MELVIN_ENABLE_ENCRYPTION="true"

# Self-check configuration
export MELVIN_CONFIDENCE_HIGH="0.8"
export MELVIN_CONFIDENCE_MEDIUM="0.6"
export MELVIN_CONFIDENCE_LOW="0.4"
```

### Configuration Files

```cpp
// Ollama configuration
MelvinOllama::OllamaConfig ollama_config;
ollama_config.base_url = "http://localhost:11434";
ollama_config.model = "llama2";
ollama_config.max_retries = 3;
ollama_config.rate_limit_requests_per_minute = 60;
ollama_config.enable_async = true;

// Encryption configuration
MelvinCrypto::EncryptionConfig crypto_config;
crypto_config.password = "secure_password";
crypto_config.key_derivation_iterations = 100000;
crypto_config.enable_compression = true;
crypto_config.enable_integrity_check = true;
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
./test_melvin_curiosity

# Run with verbose output
./test_melvin_curiosity --gtest_verbose

# Run specific test suite
./test_melvin_curiosity --gtest_filter="ConceptExtractionTest.*"

# Generate XML report
./test_melvin_curiosity --gtest_output=xml:test_results.xml
```

### Test Categories

1. **ConceptExtractionTest**: Tests question parsing and concept extraction
2. **BinaryPersistenceTest**: Tests binary storage and retrieval
3. **KnowledgeGraphTest**: Tests graph operations and memory
4. **SelfCheckTest**: Tests contradiction detection
5. **OllamaClientTest**: Tests API integration
6. **EncryptedStorageTest**: Tests encryption and security
7. **IntegrationTest**: Tests end-to-end workflows

### CI/CD Integration

The system includes GitHub Actions CI/CD pipeline:

```yaml
# .github/workflows/ci.yml
- Builds on multiple compilers (GCC, Clang)
- Runs comprehensive test suite
- Performs security scanning
- Generates coverage reports
- Tests on multiple platforms
```

## 📊 Performance

### Benchmarks

| Operation | Time | Memory |
|-----------|------|--------|
| Concept Extraction | <1ms | <1KB |
| Knowledge Retrieval | <10ms | <1KB |
| New Learning | <500ms | <5KB |
| Self-Check Analysis | <50ms | <2KB |
| Encryption/Decryption | <100ms | <10KB |
| Binary I/O | <20ms | <1KB |

### Optimization Features

- ✅ **Connection Pooling**: Reuse HTTP connections
- ✅ **Async Processing**: Non-blocking API calls
- ✅ **Binary Storage**: Fast I/O operations
- ✅ **Memory Management**: Efficient node storage
- ✅ **Caching**: Reduce redundant operations

## 🔒 Security

### Encryption Details

- **Algorithm**: AES-256-CBC
- **Key Derivation**: PBKDF2 with 100,000 iterations
- **Integrity**: HMAC-SHA256
- **Salt**: 128-bit random salt
- **IV**: 128-bit random initialization vector

### Security Best Practices

- ✅ **Secure Key Storage**: Keys never stored in plaintext
- ✅ **Memory Protection**: Sensitive data zeroed after use
- ✅ **Tamper Detection**: HMAC verification on all data
- ✅ **Password Security**: Strong password requirements
- ✅ **Access Control**: File permissions and ownership

## 🐛 Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # Start Ollama if needed
   ollama serve
   ```

2. **Encryption Errors**
   ```bash
   # Check password strength
   echo $MELVIN_PASSWORD | wc -c  # Should be >= 12
   
   # Verify file permissions
   ls -la melvin_knowledge_encrypted.bin
   ```

3. **Test Failures**
   ```bash
   # Run with verbose output
   ./test_melvin_curiosity --gtest_verbose
   
   # Check dependencies
   pkg-config --exists libcurl && echo "curl OK" || echo "curl missing"
   ```

### Debug Mode

```cpp
// Enable debug logging
#define MELVIN_DEBUG 1

// Run with debug output
./melvin_curiosity "What is a cat?" 2>&1 | tee debug.log
```

## 📈 Monitoring

### Statistics

```cpp
// Get learning statistics
auto stats = melvin.getLearningStats();
std::cout << "Questions asked: " << stats.questions_asked << std::endl;
std::cout << "Concepts learned: " << stats.new_concepts_learned << std::endl;

// Get self-check statistics
auto self_check_stats = melvin.getSelfCheckStats();
std::cout << "Contradictions found: " << self_check_stats.contradictions_found << std::endl;

// Get Ollama statistics
auto ollama_stats = client->getStatistics();
std::cout << "Success rate: " << ollama_stats.success_rate << std::endl;
```

### Health Checks

```cpp
// Check system health
bool is_healthy = melvin.isHealthy();
bool ollama_healthy = client->isHealthy();
bool storage_healthy = storage->verifyIntegrity();

if (is_healthy && ollama_healthy && storage_healthy) {
    std::cout << "All systems operational" << std::endl;
}
```

## 🔮 Future Enhancements

### Planned Features

- **Vector Database Integration**: FAISS/Milvus for semantic search
- **Multi-Model Support**: Support for multiple AI models
- **Distributed Learning**: Multi-node knowledge sharing
- **Advanced Analytics**: Learning pattern analysis
- **Web Interface**: Browser-based interaction
- **Mobile Support**: iOS/Android applications

### Extension Points

```cpp
// Custom contradiction patterns
self_check->addContradictionPattern("custom", {"pattern1", "pattern2"});

// Custom confidence scoring
void customConfidenceScorer(const KnowledgeNode& node) {
    // Implement custom confidence logic
}

// Custom encryption algorithms
class CustomCipher : public AESCipher {
    // Implement custom encryption
};
```

## 📝 License

This project is part of the Melvin Unified Brain system.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## 📞 Support

- **Documentation**: README_CURIOSITY.md
- **Issues**: GitHub Issues
- **Tests**: Run `./test_melvin_curiosity`
- **Demo**: Run `./comprehensive_demo.sh`

---

**Melvin Advanced Curiosity Learning System** - Enterprise-grade AI learning with security, testing, and real-world integration! 🧠⚡🔒

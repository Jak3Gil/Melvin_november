/*
 * Encrypted Binary Storage System for Melvin Knowledge Graph
 * 
 * Features:
 * - AES-256 encryption for knowledge data
 * - HMAC-SHA256 signing for integrity verification
 * - Secure key derivation from password
 * - Tamper detection and prevention
 * - Secure memory handling
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <array>

// Forward declarations for crypto libraries
struct EVP_CIPHER_CTX;
struct EVP_MD_CTX;
struct HMAC_CTX;

namespace MelvinCrypto {

// Configuration for encryption
struct EncryptionConfig {
    std::string password = "";           // Master password for encryption
    std::string salt_file = ".melvin_salt"; // File to store salt
    int key_derivation_iterations = 100000; // PBKDF2 iterations
    bool enable_compression = true;      // Compress before encryption
    bool enable_integrity_check = true;  // Enable HMAC verification
};

// Result of encryption/decryption operations
struct CryptoResult {
    bool success = false;
    std::vector<uint8_t> data;
    std::string error_message;
    size_t original_size = 0;
    size_t compressed_size = 0;
    size_t encrypted_size = 0;
};

// Secure key management
class SecureKeyManager {
private:
    std::array<uint8_t, 32> encryption_key;  // AES-256 key
    std::array<uint8_t, 32> hmac_key;        // HMAC key
    std::array<uint8_t, 16> salt;            // Random salt
    bool keys_initialized = false;
    
    void deriveKeys(const std::string& password);
    void generateSalt();
    bool loadSalt();
    void saveSalt();
    void secureZeroMemory(void* ptr, size_t size);
    
public:
    SecureKeyManager();
    ~SecureKeyManager();
    
    bool initialize(const std::string& password);
    bool isInitialized() const { return keys_initialized; }
    
    const std::array<uint8_t, 32>& getEncryptionKey() const { return encryption_key; }
    const std::array<uint8_t, 32>& getHmacKey() const { return hmac_key; }
    const std::array<uint8_t, 16>& getSalt() const { return salt; }
    
    void clearKeys();
};

// AES encryption/decryption
class AESCipher {
private:
    std::array<uint8_t, 32> key;
    std::array<uint8_t, 16> iv;
    
    bool generateIV();
    void secureZeroMemory(void* ptr, size_t size);
    
public:
    explicit AESCipher(const std::array<uint8_t, 32>& encryption_key);
    ~AESCipher();
    
    CryptoResult encrypt(const std::vector<uint8_t>& plaintext);
    CryptoResult decrypt(const std::vector<uint8_t>& ciphertext);
    
    const std::array<uint8_t, 16>& getIV() const { return iv; }
};

// HMAC for integrity verification
class HMACSigner {
private:
    std::array<uint8_t, 32> key;
    
public:
    explicit HMACSigner(const std::array<uint8_t, 32>& hmac_key);
    
    std::array<uint8_t, 32> sign(const std::vector<uint8_t>& data);
    bool verify(const std::vector<uint8_t>& data, const std::array<uint8_t, 32>& signature);
};

// Data compression
class DataCompressor {
public:
    static CryptoResult compress(const std::vector<uint8_t>& data);
    static CryptoResult decompress(const std::vector<uint8_t>& compressed_data);
};

// Main encrypted storage system
class EncryptedBinaryStorage {
private:
    EncryptionConfig config_;
    std::unique_ptr<SecureKeyManager> key_manager_;
    std::string storage_file_;
    
    // Statistics
    struct StorageStats {
        uint64_t total_encryptions = 0;
        uint64_t total_decryptions = 0;
        uint64_t failed_operations = 0;
        uint64_t integrity_failures = 0;
        size_t total_data_processed = 0;
    } stats_;
    
    bool performIntegrityCheck(const std::vector<uint8_t>& data, 
                              const std::array<uint8_t, 32>& expected_signature);
    
public:
    explicit EncryptedBinaryStorage(const std::string& filename = "melvin_knowledge_encrypted.bin",
                                  const EncryptionConfig& config = EncryptionConfig{});
    ~EncryptedBinaryStorage();
    
    // Main storage operations
    bool saveEncryptedData(const std::vector<uint8_t>& data);
    CryptoResult loadEncryptedData();
    
    // Knowledge graph specific operations
    bool saveKnowledgeGraph(const std::vector<uint8_t>& graph_data);
    CryptoResult loadKnowledgeGraph();
    
    // Configuration and status
    bool initialize(const std::string& password);
    bool isInitialized() const;
    void updateConfig(const EncryptionConfig& new_config);
    
    // Security operations
    bool changePassword(const std::string& old_password, const std::string& new_password);
    bool verifyIntegrity();
    void clearSecureData();
    
    // Statistics
    StorageStats getStatistics() const;
    std::string getSecurityReport() const;
    
    // Utility methods
    static std::string generateSecurePassword(size_t length = 32);
    static bool isFileEncrypted(const std::string& filename);
    static size_t getEncryptedFileSize(const std::string& filename);
};

// Enhanced Melvin Learning System with Encrypted Storage
class MelvinLearningSystemWithEncryption : public MelvinLearningSystem {
private:
    std::unique_ptr<EncryptedBinaryStorage> encrypted_storage_;
    EncryptionConfig encryption_config_;
    bool encryption_enabled_ = false;
    
public:
    MelvinLearningSystemWithEncryption(const std::string& password = "",
                                     const EncryptionConfig& config = EncryptionConfig{});
    
    // Override storage methods to use encryption
    void saveKnowledge() override;
    void loadKnowledge() override;
    
    // Encryption management
    bool enableEncryption(const std::string& password);
    bool disableEncryption();
    bool isEncryptionEnabled() const { return encryption_enabled_; }
    
    // Security operations
    bool changePassword(const std::string& old_password, const std::string& new_password);
    bool verifyDataIntegrity();
    std::string getSecurityReport() const;
    
    // Enhanced learning with security
    std::string curiosityLoopSecure(const std::string& question);
    
private:
    void initializeEncryptedStorage();
};

// Utility functions
namespace CryptoUtils {
    std::string bytesToHex(const std::vector<uint8_t>& bytes);
    std::vector<uint8_t> hexToBytes(const std::string& hex);
    std::string generateRandomString(size_t length);
    bool secureCompare(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b);
    void secureZeroMemory(void* ptr, size_t size);
    std::string hashPassword(const std::string& password, const std::vector<uint8_t>& salt);
}

} // namespace MelvinCrypto

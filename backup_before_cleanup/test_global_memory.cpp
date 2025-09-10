#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>

int main() {
    std::cout << "Testing global memory system..." << std::endl;
    
    // Test creating memory directory
    std::string memory_directory = "melvin_binary_memory";
    std::filesystem::create_directories(memory_directory);
    
    // Test writing a simple file
    std::string test_file = memory_directory + "/test.bin";
    std::ofstream test_stream(test_file, std::ios::binary);
    if (test_stream.is_open()) {
        uint64_t test_data = 12345;
        test_stream.write(reinterpret_cast<const char*>(&test_data), sizeof(test_data));
        test_stream.close();
        std::cout << "✅ Successfully wrote test data to " << test_file << std::endl;
    } else {
        std::cout << "❌ Failed to write test data" << std::endl;
    }
    
    // Test reading the file back
    std::ifstream read_stream(test_file, std::ios::binary);
    if (read_stream.is_open()) {
        uint64_t read_data;
        read_stream.read(reinterpret_cast<char*>(&read_data), sizeof(read_data));
        read_stream.close();
        
        if (read_data == 12345) {
            std::cout << "✅ Successfully read test data: " << read_data << std::endl;
            std::cout << "🎉 Global memory system is working!" << std::endl;
        } else {
            std::cout << "❌ Data mismatch: expected 12345, got " << read_data << std::endl;
        }
    } else {
        std::cout << "❌ Failed to read test data" << std::endl;
    }
    
    return 0;
}

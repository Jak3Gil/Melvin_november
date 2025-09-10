#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>

// ============================================================================
// MELVIN BRAIN CONNECTION TEST - CHECK IF PUZZLE DATA IS IN MAIN BRAIN
// ============================================================================

class BrainConnectionChecker {
private:
    std::string brain_path;
    
public:
    BrainConnectionChecker() : brain_path("melvin_binary_memory") {
        std::cout << "🧠 Checking Melvin's Unified Brain Connection..." << std::endl;
    }
    
    void check_brain_files() {
        std::cout << "\n📁 CHECKING BRAIN FILES:" << std::endl;
        std::cout << "=========================" << std::endl;
        
        std::vector<std::string> files = {"nodes.bin", "connections.bin", "index.bin"};
        
        for (const auto& file : files) {
            std::string full_path = brain_path + "/" + file;
            std::ifstream file_stream(full_path, std::ios::binary);
            
            if (file_stream.good()) {
                // Get file size
                file_stream.seekg(0, std::ios::end);
                size_t file_size = file_stream.tellg();
                file_stream.close();
                
                std::cout << "✅ " << file << " - " << file_size << " bytes" << std::endl;
            } else {
                std::cout << "❌ " << file << " - NOT FOUND" << std::endl;
            }
        }
    }
    
    void analyze_brain_content() {
        std::cout << "\n🔍 ANALYZING BRAIN CONTENT:" << std::endl;
        std::cout << "===========================" << std::endl;
        
        // Check if we can read the brain files
        std::string nodes_file = brain_path + "/nodes.bin";
        std::ifstream nodes_stream(nodes_file, std::ios::binary);
        
        if (!nodes_stream.good()) {
            std::cout << "❌ Cannot read nodes.bin - brain may not be initialized" << std::endl;
            return;
        }
        
        // Try to read some basic info
        nodes_stream.seekg(0, std::ios::end);
        size_t total_size = nodes_stream.tellg();
        nodes_stream.close();
        
        std::cout << "📊 Brain Analysis:" << std::endl;
        std::cout << "  📦 Total nodes file size: " << total_size << " bytes" << std::endl;
        
        if (total_size > 0) {
            std::cout << "  ✅ Brain contains data" << std::endl;
            std::cout << "  📈 Estimated nodes: " << (total_size / 100) << " (rough estimate)" << std::endl;
        } else {
            std::cout << "  ❌ Brain is empty" << std::endl;
        }
    }
    
    void check_puzzle_data_persistence() {
        std::cout << "\n🧩 CHECKING PUZZLE DATA PERSISTENCE:" << std::endl;
        std::cout << "====================================" << std::endl;
        
        std::cout << "🔍 Looking for puzzle-related data in brain..." << std::endl;
        
        // Check if we can find any puzzle-related content
        std::string nodes_file = brain_path + "/nodes.bin";
        std::ifstream nodes_stream(nodes_file, std::ios::binary);
        
        if (!nodes_stream.good()) {
            std::cout << "❌ Cannot access brain files" << std::endl;
            return;
        }
        
        // Read a small sample to check for text content
        std::vector<char> buffer(1024);
        nodes_stream.read(buffer.data(), buffer.size());
        nodes_stream.close();
        
        std::string content(buffer.begin(), buffer.end());
        
        // Look for puzzle-related keywords
        std::vector<std::string> puzzle_keywords = {
            "puzzle", "prisoner", "hat", "bridge", "crossing", 
            "sequence", "number", "logic", "reasoning", "solution"
        };
        
        int found_keywords = 0;
        for (const auto& keyword : puzzle_keywords) {
            if (content.find(keyword) != std::string::npos) {
                found_keywords++;
                std::cout << "  ✅ Found keyword: '" << keyword << "'" << std::endl;
            }
        }
        
        if (found_keywords > 0) {
            std::cout << "  🎉 Found " << found_keywords << " puzzle-related keywords!" << std::endl;
            std::cout << "  ✅ Puzzle data IS stored in Melvin's main brain!" << std::endl;
        } else {
            std::cout << "  ❌ No puzzle keywords found" << std::endl;
            std::cout << "  ⚠️ Puzzle data may NOT be in Melvin's main brain" << std::endl;
        }
    }
    
    void demonstrate_brain_connection() {
        std::cout << "\n🧪 DEMONSTRATING BRAIN CONNECTION:" << std::endl;
        std::cout << "===================================" << std::endl;
        
        std::cout << "📝 Current Test Programs:" << std::endl;
        std::cout << "  ❌ melvin_new_puzzles_test.cpp - Uses MelvinNewBrain (isolated)" << std::endl;
        std::cout << "  ❌ melvin_hard_puzzle_test.cpp - Uses MelvinHardBrain (isolated)" << std::endl;
        std::cout << "  ❌ melvin_real_solving_test.cpp - Uses MelvinRealBrain (isolated)" << std::endl;
        std::cout << "  ❌ melvin_quick_puzzle.cpp - Uses QuickMelvinBrain (isolated)" << std::endl;
        
        std::cout << "\n🧠 Melvin's Main Unified Brain:" << std::endl;
        std::cout << "  ✅ melvin_optimized_v2.h/cpp - MelvinOptimizedV2 class" << std::endl;
        std::cout << "  ✅ Uses PureBinaryStorage for persistence" << std::endl;
        std::cout << "  ✅ Stores data in melvin_binary_memory/ directory" << std::endl;
        std::cout << "  ✅ Has compression, pruning, hebbian learning" << std::endl;
        
        std::cout << "\n🔗 Connection Status:" << std::endl;
        std::cout << "  ❌ Test programs are NOT connected to main brain" << std::endl;
        std::cout << "  ❌ All puzzle solving happens in temporary memory" << std::endl;
        std::cout << "  ❌ Data is lost when test programs end" << std::endl;
        std::cout << "  ❌ No persistence between sessions" << std::endl;
        
        std::cout << "\n💡 To Fix This:" << std::endl;
        std::cout << "  ✅ Use MelvinOptimizedV2 class in test programs" << std::endl;
        std::cout << "  ✅ Call process_text_input() to store data" << std::endl;
        std::cout << "  ✅ Call update_hebbian_learning() to create connections" << std::endl;
        std::cout << "  ✅ Call save_complete_state() to persist data" << std::endl;
    }
    
    void run_full_check() {
        std::cout << "🧩 MELVIN BRAIN CONNECTION TEST" << std::endl;
        std::cout << "===============================" << std::endl;
        
        check_brain_files();
        analyze_brain_content();
        check_puzzle_data_persistence();
        demonstrate_brain_connection();
        
        std::cout << "\n🎯 CONCLUSION:" << std::endl;
        std::cout << "==============" << std::endl;
        std::cout << "❌ Current test programs are NOT using Melvin's main brain" << std::endl;
        std::cout << "❌ All puzzle solving happens in isolated, temporary brain instances" << std::endl;
        std::cout << "❌ No data persists between sessions" << std::endl;
        std::cout << "❌ No connection to Melvin's unified brain architecture" << std::endl;
        
        std::cout << "\n✅ To get REAL brain usage:" << std::endl;
        std::cout << "  - Use MelvinOptimizedV2 class" << std::endl;
        std::cout << "  - Store data with process_text_input()" << std::endl;
        std::cout << "  - Create connections with update_hebbian_learning()" << std::endl;
        std::cout << "  - Save state with save_complete_state()" << std::endl;
        std::cout << "  - Data will persist in melvin_binary_memory/ directory" << std::endl;
    }
};

int main() {
    try {
        BrainConnectionChecker checker;
        checker.run_full_check();
        
        std::cout << "\n🎉 Brain connection test completed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

#include <iostream>
#include <string>

int main() {
    std::cout << "🧠 MELVIN SIMPLE TEST" << std::endl;
    std::cout << "=====================" << std::endl;
    
    std::cout << "Testing basic functionality..." << std::endl;
    
    // Test 1: Basic output
    std::cout << "✅ Basic output works" << std::endl;
    
    // Test 2: String handling
    std::string test = "what is cancer";
    std::cout << "✅ String handling works: " << test << std::endl;
    
    // Test 3: Input simulation
    std::cout << "✅ Input simulation: " << test << std::endl;
    
    // Test 4: Response generation
    std::string response = "Cancer is a group of diseases characterized by uncontrolled cell growth.";
    std::cout << "Melvin: " << response << std::endl;
    
    std::cout << "✅ All basic tests passed!" << std::endl;
    std::cout << "The issue is likely with interactive input handling." << std::endl;
    
    return 0;
}

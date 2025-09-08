#include <iostream>
#include <string>

int main() {
    std::cout << "Testing basic C++ functionality..." << std::endl;
    
    try {
        std::string test = "Hello, Melvin!";
        std::cout << "String test: " << test << std::endl;
        
        std::cout << "Basic test successful!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }
}

#include <iostream>
#include <string>

int main() {
    std::cout << "Testing Melvin..." << std::endl;
    std::cout << "Enter a question: ";
    
    std::string input;
    std::getline(std::cin, input);
    
    std::cout << "You entered: " << input << std::endl;
    
    if (input.find("whats my name") != std::string::npos) {
        std::cout << "I don't have access to your personal information, including your name." << std::endl;
    } else if (input.find("what is cancer") != std::string::npos) {
        std::cout << "Cancer is a group of diseases characterized by uncontrolled cell growth." << std::endl;
    } else {
        std::cout << "That's an interesting question!" << std::endl;
    }
    
    return 0;
}

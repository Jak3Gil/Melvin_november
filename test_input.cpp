#include <iostream>
#include <string>

int main() {
    std::cout << "Testing input handling..." << std::endl;
    std::cout << "Type something and press Enter (or 'quit' to exit):" << std::endl;
    
    std::string input;
    while (true) {
        std::cout << "You: ";
        std::getline(std::cin, input);
        
        if (input.empty()) {
            std::cout << "Empty input received" << std::endl;
            continue;
        }
        
        std::cout << "You typed: '" << input << "'" << std::endl;
        
        if (input == "quit") {
            break;
        }
    }
    
    std::cout << "Goodbye!" << std::endl;
    return 0;
}

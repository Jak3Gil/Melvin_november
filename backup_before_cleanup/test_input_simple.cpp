#include <iostream>
#include <string>
#include <limits>

int main() {
    std::cout << "Testing input handling..." << std::endl;
    
    std::string input;
    int count = 0;
    
    while (count < 3) {
        std::cout << "Enter something (or 'quit' to exit): ";
        std::cout.flush();
        
        if (std::cin.peek() == EOF) {
            std::cout << "No input available. Exiting..." << std::endl;
            break;
        }
        
        if (std::getline(std::cin, input)) {
            std::cout << "You entered: '" << input << "'" << std::endl;
            
            if (input == "quit") {
                break;
            }
        } else {
            std::cout << "Input error detected." << std::endl;
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
        
        count++;
    }
    
    std::cout << "Test complete!" << std::endl;
    return 0;
}

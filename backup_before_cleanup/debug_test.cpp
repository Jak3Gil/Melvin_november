#include <iostream>
#include <string>

int main() {
    std::cout << "Debug Test Starting..." << std::endl;
    
    std::string input;
    int count = 0;
    
    while (count < 3) {
        std::cout << "Enter input " << (count + 1) << ": ";
        std::cout.flush();
        
        if (std::getline(std::cin, input)) {
            std::cout << "Received: '" << input << "'" << std::endl;
            
            if (input == "quit") {
                std::cout << "Exiting..." << std::endl;
                break;
            }
        } else {
            std::cout << "Input failed!" << std::endl;
            break;
        }
        
        count++;
    }
    
    std::cout << "Debug Test Complete!" << std::endl;
    return 0;
}

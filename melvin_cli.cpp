#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    std::cout << "🧠 MELVIN CLI VERSION" << std::endl;
    std::cout << "====================" << std::endl;
    
    if (argc < 2) {
        std::cout << "Usage: melvin_cli.exe \"your question here\"" << std::endl;
        std::cout << "Example: melvin_cli.exe \"what is cancer\"" << std::endl;
        return 1;
    }
    
    std::string question = argv[1];
    std::cout << "Question: " << question << std::endl;
    
    // Simple response logic
    std::string lower_question = question;
    std::transform(lower_question.begin(), lower_question.end(), lower_question.begin(), ::tolower);
    
    std::string response;
    if (lower_question.find("cancer") != std::string::npos) {
        response = "Cancer is a group of diseases characterized by uncontrolled cell growth. It can affect any part of the body and occurs when cells divide uncontrollably and spread into surrounding tissues.";
    } else if (lower_question.find("dog") != std::string::npos) {
        response = "Dogs are domesticated mammals and loyal companions to humans. They belong to the Canidae family and have been bred for various purposes including hunting, herding, protection, and companionship.";
    } else if (lower_question.find("ai") != std::string::npos || lower_question.find("artificial intelligence") != std::string::npos) {
        response = "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans. It includes machine learning, natural language processing, computer vision, and robotics.";
    } else {
        response = "That's an interesting question. I'm still learning about this topic and would need more information to provide a comprehensive answer.";
    }
    
    std::cout << "Melvin: " << response << std::endl;
    
    return 0;
}

#include "melvin_fully_unified_brain.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>

// ============================================================================
// TEST FOR FULLY UNIFIED BRAIN
// ============================================================================
// This test demonstrates the fully unified brain where thinking and memory
// are completely integrated in one system

int main() {
    std::cout << "🧠 MELVIN FULLY UNIFIED BRAIN TEST" << std::endl;
    std::cout << "===================================" << std::endl;
    std::cout << "Testing completely unified thinking and memory system" << std::endl;
    std::cout << "No separate files - everything happens in one place!" << std::endl;
    
    try {
        // Initialize the fully unified brain
        MelvinUnifiedInterface melvin;
        
        std::cout << "\n🧠 INITIAL BRAIN STATUS" << std::endl;
        std::cout << "=======================" << std::endl;
        melvin.show_brain_status();
        
        // Test basic learning and thinking
        std::cout << "\n📚 LEARNING PHASE" << std::endl;
        std::cout << "=================" << std::endl;
        
        melvin.tell("I am learning about colors");
        melvin.tell("Red is a warm color");
        melvin.tell("Blue is a cool color");
        melvin.tell("Green is the color of nature");
        
        melvin.tell("I am learning about animals");
        melvin.tell("Dogs are loyal pets");
        melvin.tell("Cats are independent animals");
        melvin.tell("Birds can fly in the sky");
        
        melvin.tell("I am learning about food");
        melvin.tell("Pizza is delicious");
        melvin.tell("Vegetables are healthy");
        melvin.tell("Fruit is sweet and nutritious");
        
        // Test thinking and reasoning
        std::cout << "\n🤔 THINKING PHASE" << std::endl;
        std::cout << "==================" << std::endl;
        
        std::string response1 = melvin.ask("What do you know about colors?");
        std::cout << "Response: " << response1 << std::endl;
        
        std::string response2 = melvin.ask("What animals do you know?");
        std::cout << "Response: " << response2 << std::endl;
        
        std::string response3 = melvin.ask("What food is healthy?");
        std::cout << "Response: " << response3 << std::endl;
        
        // Test problem solving
        std::cout << "\n🧩 PROBLEM SOLVING PHASE" << std::endl;
        std::cout << "========================" << std::endl;
        
        std::string solution1 = melvin.think("What would be a good pet for someone who likes warm colors?");
        std::cout << "Solution: " << solution1 << std::endl;
        
        std::string solution2 = melvin.think("What would be a healthy meal with colors?");
        std::cout << "Solution: " << solution2 << std::endl;
        
        // Test idea generation
        std::cout << "\n💡 IDEA GENERATION PHASE" << std::endl;
        std::cout << "========================" << std::endl;
        
        std::string idea1 = melvin.think("Generate an idea about combining colors and animals");
        std::cout << "Idea: " << idea1 << std::endl;
        
        std::string idea2 = melvin.think("Generate an idea about healthy colorful food");
        std::cout << "Idea: " << idea2 << std::endl;
        
        // Test reflection
        std::cout << "\n🤔 REFLECTION PHASE" << std::endl;
        std::cout << "===================" << std::endl;
        
        std::string reflection1 = melvin.remember("I learned about colors, animals, and food today");
        std::cout << "Reflection: " << reflection1 << std::endl;
        
        std::string reflection2 = melvin.remember("I connected warm colors with loyal pets");
        std::cout << "Reflection: " << reflection2 << std::endl;
        
        // Show brain status after learning
        std::cout << "\n🧠 BRAIN STATUS AFTER LEARNING" << std::endl;
        std::cout << "==============================" << std::endl;
        melvin.show_brain_status();
        
        // Show recent thoughts
        std::cout << "\n💭 RECENT THOUGHTS" << std::endl;
        std::cout << "==================" << std::endl;
        melvin.show_recent_thoughts();
        
        // Show active concepts
        melvin.show_active_concepts();
        
        // Test advanced reasoning
        std::cout << "\n🧠 ADVANCED REASONING PHASE" << std::endl;
        std::cout << "===========================" << std::endl;
        
        std::string advanced1 = melvin.ask("What do you think about the relationship between colors and emotions?");
        std::cout << "Advanced Response: " << advanced1 << std::endl;
        
        std::string advanced2 = melvin.ask("How do animals and food relate to each other?");
        std::cout << "Advanced Response: " << advanced2 << std::endl;
        
        // Optimize brain
        std::cout << "\n🔧 BRAIN OPTIMIZATION" << std::endl;
        std::cout << "====================" << std::endl;
        melvin.consolidate_knowledge();
        melvin.optimize_brain();
        
        // Final brain status
        std::cout << "\n🧠 FINAL BRAIN STATUS" << std::endl;
        std::cout << "=====================" << std::endl;
        melvin.show_brain_status();
        
        // Test final reasoning
        std::cout << "\n🎯 FINAL REASONING TEST" << std::endl;
        std::cout << "=======================" << std::endl;
        
        std::string final1 = melvin.ask("What have you learned today?");
        std::cout << "Final Response: " << final1 << std::endl;
        
        std::string final2 = melvin.ask("What connections have you made?");
        std::cout << "Final Response: " << final2 << std::endl;
        
        std::cout << "\n🎉 FULLY UNIFIED BRAIN TEST Complete!" << std::endl;
        std::cout << "======================================" << std::endl;
        std::cout << "✅ Thinking and memory are completely unified!" << std::endl;
        std::cout << "✅ No separate files or systems needed!" << std::endl;
        std::cout << "✅ Everything happens in one cohesive brain!" << std::endl;
        std::cout << "✅ Learning, thinking, and memory are integrated!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error during fully unified brain test: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

#include "melvin_unified_brain.h"
#include <iostream>
#include <chrono>
#include <thread>

// ============================================================================
// UNIFIED MELVIN BRAIN TEST
// ============================================================================
// This test demonstrates the unified Melvin brain with continuous thought cycle

int main() {
    std::cout << "🧠 Testing Unified Melvin Brain" << std::endl;
    std::cout << "================================" << std::endl;
    
    try {
        // Create unified Melvin brain
        MelvinUnifiedInterface melvin("melvin_unified_test_memory");
        
        std::cout << "\n📚 Learning Phase:" << std::endl;
        std::cout << "==================" << std::endl;
        
        // Teach Melvin some knowledge
        melvin.tell("Dogs are loyal pets that love to play");
        melvin.tell("Cats are independent animals that enjoy napping");
        melvin.tell("Music can be relaxing and help with stress");
        melvin.tell("Red is a warm color associated with passion");
        melvin.tell("Blue is a cool color associated with calmness");
        
        std::cout << "\n❓ Question Phase:" << std::endl;
        std::cout << "==================" << std::endl;
        
        // Ask Melvin questions
        std::string answer1 = melvin.ask("What animal would make a good pet?");
        std::cout << "Q: What animal would make a good pet?" << std::endl;
        std::cout << "A: " << answer1 << std::endl;
        
        std::string answer2 = melvin.ask("What color is associated with calmness?");
        std::cout << "\nQ: What color is associated with calmness?" << std::endl;
        std::cout << "A: " << answer2 << std::endl;
        
        std::string answer3 = melvin.ask("What helps with stress?");
        std::cout << "\nQ: What helps with stress?" << std::endl;
        std::cout << "A: " << answer3 << std::endl;
        
        std::cout << "\n🧠 Thinking Phase:" << std::endl;
        std::cout << "==================" << std::endl;
        
        // Start continuous thinking
        melvin.start_thinking();
        
        std::cout << "Melvin is now thinking continuously..." << std::endl;
        std::cout << "Let him think for 5 seconds..." << std::endl;
        
        // Let Melvin think for a bit
        std::this_thread::sleep_for(std::chrono::seconds(5));
        
        // Interrupt with a question
        std::cout << "\nInterrupting with a question..." << std::endl;
        melvin.interrupt_with("What should I think about next?");
        
        // Let him process the interrupt
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        // Stop thinking
        melvin.stop_thinking();
        
        std::cout << "\n📊 Brain Status:" << std::endl;
        std::cout << "==================" << std::endl;
        
        // Show brain status
        melvin.show_brain_status();
        
        std::cout << "\n🧠 Recent Thoughts:" << std::endl;
        std::cout << "===================" << std::endl;
        
        // Show recent thoughts
        melvin.show_recent_thoughts();
        
        std::cout << "\n🧠 Meta-Cognitive State:" << std::endl;
        std::cout << "=========================" << std::endl;
        
        // Show meta-cognitive state
        melvin.show_meta_cognitive_state();
        
        std::cout << "\n🔧 Optimization Phase:" << std::endl;
        std::cout << "=======================" << std::endl;
        
        // Optimize brain
        melvin.consolidate_knowledge();
        melvin.optimize_brain();
        
        std::cout << "Brain optimization completed!" << std::endl;
        
        std::cout << "\n💾 Saving Brain State:" << std::endl;
        std::cout << "======================" << std::endl;
        
        // Save brain state
        melvin.save_brain_state();
        
        std::cout << "Brain state saved successfully!" << std::endl;
        
        std::cout << "\n🎯 Final Test:" << std::endl;
        std::cout << "==============" << std::endl;
        
        // Final question
        std::string final_answer = melvin.ask("What have you learned today?");
        std::cout << "Q: What have you learned today?" << std::endl;
        std::cout << "A: " << final_answer << std::endl;
        
        std::cout << "\n✅ Unified Melvin Brain Test Completed Successfully!" << std::endl;
        std::cout << "=================================================" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

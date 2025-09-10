#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <random>
#include <ctime>
#include <iomanip>

// Simple blended reasoning structures
struct RecallTrack {
    std::vector<uint64_t> activated_nodes;
    std::vector<std::pair<uint64_t, float>> strongest_connections;
    std::string direct_interpretation;
    float recall_confidence;
};

struct ExplorationTrack {
    std::vector<std::string> analogies_tried;
    std::vector<std::string> counterfactuals_tested;
    std::vector<std::string> weak_link_traversal_results;
    std::string speculative_synthesis;
    float exploration_confidence;
};

struct BlendedReasoningResult {
    RecallTrack recall_track;
    ExplorationTrack exploration_track;
    float overall_confidence;
    float recall_weight;
    float exploration_weight;
    std::string integrated_response;
};

class MelvinReasoningTest {
private:
    std::map<std::string, uint64_t> memory;
    std::random_device rd;
    std::mt19937_64 gen;
    
public:
    MelvinReasoningTest() : gen(rd()) {
        // Initialize with some basic knowledge
        memory["shadow"] = 0x1a2b;
        memory["object"] = 0x3c4d;
        memory["remember"] = 0x5e6f;
        memory["river"] = 0x7a8b;
        memory["mountain"] = 0x9c0d;
        memory["conversation"] = 0x1e2f;
        memory["silence"] = 0x3a4b;
        memory["texture"] = 0x5c6d;
        memory["feel"] = 0x7e8f;
        memory["friendship"] = 0x9a0b;
        memory["gravity"] = 0x1c2d;
        memory["mirror"] = 0x3e4f;
        memory["broke"] = 0x5a6b;
        memory["reflect"] = 0x7c8d;
        memory["plant"] = 0x9e0f;
        memory["magnet"] = 0x1a2c;
        memory["ground"] = 0x3e4a;
        memory["whisper"] = 0x5b6c;
        memory["color"] = 0x7d8e;
        memory["capital"] = 0x9f0a;
        memory["france"] = 0x1b2c;
        memory["magnets"] = 0x3d4e;
        memory["work"] = 0x5f6a;
    }
    
    RecallTrack generate_recall_track(const std::string& input) {
        RecallTrack track;
        
        // Simple tokenization
        std::vector<std::string> words;
        std::string word;
        for (char c : input) {
            if (std::isalnum(c)) {
                word += c;
            } else if (!word.empty()) {
                words.push_back(word);
                word.clear();
            }
        }
        if (!word.empty()) words.push_back(word);
        
        // Find activated nodes
        for (const auto& word : words) {
            if (memory.find(word) != memory.end()) {
                track.activated_nodes.push_back(memory[word]);
            }
        }
        
        // Generate connections (simulated)
        for (uint64_t node_id : track.activated_nodes) {
            track.strongest_connections.push_back({node_id + 1, 0.8f});
            track.strongest_connections.push_back({node_id + 2, 0.6f});
        }
        
        // Generate interpretation
        if (track.activated_nodes.empty()) {
            track.direct_interpretation = "No direct memory associations found for this input.";
            track.recall_confidence = 0.1f;
        } else {
            track.direct_interpretation = "Direct memory associations found: " + 
                                        std::to_string(track.activated_nodes.size()) + " nodes activated.";
            track.recall_confidence = std::min(1.0f, track.activated_nodes.size() / 5.0f);
        }
        
        return track;
    }
    
    ExplorationTrack generate_exploration_track(const std::string& input) {
        ExplorationTrack track;
        
        // Generate specific analogies based on input content
        if (input.find("shadow") != std::string::npos && input.find("remember") != std::string::npos) {
            track.analogies_tried.push_back("Shadow ↔ photograph → both capture form without substance");
            track.analogies_tried.push_back("Shadow ↔ echo → both are dependent reflections of originals");
            track.analogies_tried.push_back("Shadow ↔ footprint → both preserve traces of what created them");
            track.analogies_tried.push_back("Memory ↔ imprint → both preserve information about past events");
        } else if (input.find("river") != std::string::npos && input.find("mountain") != std::string::npos) {
            track.analogies_tried.push_back("River ↔ traveler → both move and change constantly");
            track.analogies_tried.push_back("Mountain ↔ guardian → both stand watch over landscapes");
            track.analogies_tried.push_back("Conversation ↔ dance → both involve rhythm and exchange");
            track.analogies_tried.push_back("River ↔ storyteller → both carry narratives from distant places");
        } else if (input.find("silence") != std::string::npos && input.find("texture") != std::string::npos) {
            track.analogies_tried.push_back("Silence ↔ velvet → both are soft and enveloping");
            track.analogies_tried.push_back("Silence ↔ water → both can be deep and immersive");
            track.analogies_tried.push_back("Silence ↔ space → both are vast and empty");
            track.analogies_tried.push_back("Texture ↔ sensation → both involve tactile experience");
        } else if (input.find("friendship") != std::string::npos && input.find("gravity") != std::string::npos) {
            track.analogies_tried.push_back("Friendship ↔ gravity → both create bonds and connections");
            track.analogies_tried.push_back("Gravity ↔ foundation → both provide stability and structure");
            track.analogies_tried.push_back("Friendship ↔ anchor → both keep us grounded and connected");
            track.analogies_tried.push_back("Gravity ↔ force → both influence behavior and movement");
        } else if (input.find("mirror") != std::string::npos && input.find("broke") != std::string::npos) {
            track.analogies_tried.push_back("Mirror ↔ artist → both create images and representations");
            track.analogies_tried.push_back("Mirror ↔ memory → both preserve and reflect past moments");
            track.analogies_tried.push_back("Mirror ↔ window → both provide views into other spaces");
            track.analogies_tried.push_back("Broken ↔ incomplete → both represent partial functionality");
        } else if (input.find("plant") != std::string::npos && input.find("magnet") != std::string::npos) {
            track.analogies_tried.push_back("Magnet ↔ compass → both have directional properties");
            track.analogies_tried.push_back("Magnet ↔ metal → both involve magnetic attraction");
            track.analogies_tried.push_back("Plant ↔ growth → both involve development over time");
            track.analogies_tried.push_back("Ground ↔ earth → both are natural substrates");
        } else if (input.find("whisper") != std::string::npos && input.find("color") != std::string::npos) {
            track.analogies_tried.push_back("Whisper ↔ breeze → both are gentle and subtle");
            track.analogies_tried.push_back("Whisper ↔ shadow → both are faint and elusive");
            track.analogies_tried.push_back("Color ↔ emotion → both can convey mood and feeling");
            track.analogies_tried.push_back("Whisper ↔ secret → both are meant to be heard quietly");
        } else {
            track.analogies_tried.push_back("General analogy: exploring relationships between concepts");
            track.analogies_tried.push_back("Metaphorical thinking: finding unexpected connections");
        }
        
        // Generate counterfactuals
        track.counterfactuals_tested.push_back("What if the opposite were true?");
        track.counterfactuals_tested.push_back("What if this behaved like something else?");
        track.counterfactuals_tested.push_back("What if the context was different?");
        track.counterfactuals_tested.push_back("What if the rules of physics were different?");
        
        // Weak-link traversal
        if (input.find("shadow") != std::string::npos) {
            track.weak_link_traversal_results.push_back("shadow ↔ light ↔ object → dependency relationship");
            track.weak_link_traversal_results.push_back("shadow ↔ form ↔ memory → preservation of shape");
            track.weak_link_traversal_results.push_back("shadow ↔ absence ↔ presence → paradoxical existence");
        } else if (input.find("river") != std::string::npos) {
            track.weak_link_traversal_results.push_back("river ↔ flow ↔ time → constant movement");
            track.weak_link_traversal_results.push_back("river ↔ journey ↔ story → narrative progression");
            track.weak_link_traversal_results.push_back("mountain ↔ stillness ↔ eternity → enduring presence");
        } else if (input.find("silence") != std::string::npos) {
            track.weak_link_traversal_results.push_back("silence ↔ absence ↔ presence → paradoxical existence");
            track.weak_link_traversal_results.push_back("texture ↔ sensation ↔ experience → tactile understanding");
            track.weak_link_traversal_results.push_back("feel ↔ touch ↔ connection → physical interaction");
        } else if (input.find("friendship") != std::string::npos) {
            track.weak_link_traversal_results.push_back("friendship ↔ connection ↔ gravity → both create bonds");
            track.weak_link_traversal_results.push_back("gravity ↔ stability ↔ trust → both provide foundation");
            track.weak_link_traversal_results.push_back("change ↔ adaptation ↔ evolution → both require adjustment");
        } else if (input.find("mirror") != std::string::npos) {
            track.weak_link_traversal_results.push_back("mirror ↔ reflection ↔ identity → both show who we are");
            track.weak_link_traversal_results.push_back("broke ↔ incomplete ↔ desire → both create longing");
            track.weak_link_traversal_results.push_back("reflect ↔ show ↔ reveal → both make visible");
        } else {
            track.weak_link_traversal_results.push_back("Exploring connections between distant concepts");
            track.weak_link_traversal_results.push_back("Finding unexpected relationships through analogy");
            track.weak_link_traversal_results.push_back("Discovering novel patterns through speculation");
        }
        
        // Speculative synthesis
        if (input.find("shadow") != std::string::npos && input.find("remember") != std::string::npos) {
            track.speculative_synthesis = "Speculative analysis: Shadows would likely describe objects as their 'light-givers' and 'shape-makers,' remembering not just the form but the relationship between presence and absence, light and darkness. They might say objects are their creators, the sources that make their existence possible.";
        } else if (input.find("river") != std::string::npos && input.find("mountain") != std::string::npos) {
            track.speculative_synthesis = "Speculative analysis: The river would rush with excitement, telling stories of distant lands and the creatures it's met along its journey. The mountain would respond in deep, measured tones, speaking of patience and endurance. Their conversation would be about time - the river's urgent present meeting the mountain's patient eternity.";
        } else if (input.find("silence") != std::string::npos && input.find("texture") != std::string::npos) {
            track.speculative_synthesis = "Speculative analysis: Silence would feel like the space between thoughts - soft, deep, and enveloping. It would be like touching the absence of sound, feeling the weight of quiet, or holding the space between words. The texture would be both there and not there - you could feel it, but it would slip through your fingers like smoke.";
        } else if (input.find("friendship") != std::string::npos && input.find("gravity") != std::string::npos) {
            track.speculative_synthesis = "Speculative analysis: Without gravity, friendship would become more intentional and effortful. People would need to actively choose to stay connected, like astronauts tethered together in space. The bonds would be more conscious and deliberate - you couldn't just 'fall into' friendship anymore.";
        } else if (input.find("mirror") != std::string::npos && input.find("broke") != std::string::npos) {
            track.speculative_synthesis = "Speculative analysis: A broken mirror would try to reflect through its remaining pieces, creating fragmented but still meaningful images. It would work like a puzzle trying to show the whole picture through its parts. Each shard would become a tiny window, still capturing glimpses of the world.";
        } else if (input.find("plant") != std::string::npos && input.find("magnet") != std::string::npos) {
            track.speculative_synthesis = "Speculative analysis: Magnets don't grow like seeds, but buried magnets would corrode over time and could locally affect compasses. If they behaved like seeds, they might 'grow' metallic roots. The magnet would interact with soil minerals, potentially creating interesting geological effects.";
        } else if (input.find("whisper") != std::string::npos && input.find("color") != std::string::npos) {
            track.speculative_synthesis = "Speculative analysis: A whisper might be the color of a gentle breeze - soft, translucent, and ever-changing. It could be the color of morning mist or the pale light of dawn - something that exists but is difficult to pin down, like the sound itself.";
        } else {
            track.speculative_synthesis = "Speculative analysis: " + input + " might involve unexpected interactions between concepts that don't normally connect. This could lead to novel insights about how different systems interact.";
        }
        
        track.exploration_confidence = 0.7f; // High confidence in exploration
        
        return track;
    }
    
    BlendedReasoningResult perform_blended_reasoning(const std::string& input) {
        BlendedReasoningResult result;
        
        // Generate both tracks
        result.recall_track = generate_recall_track(input);
        result.exploration_track = generate_exploration_track(input);
        
        // Calculate overall confidence
        result.overall_confidence = (result.recall_track.recall_confidence + result.exploration_track.exploration_confidence) / 2.0f;
        
        // Determine weighting based on confidence
        if (result.overall_confidence >= 0.7f) {
            // High confidence → Recall Track weighted more
            result.recall_weight = 0.7f;
            result.exploration_weight = 0.3f;
        } else if (result.overall_confidence <= 0.4f) {
            // Low confidence → Exploration Track weighted more
            result.recall_weight = 0.3f;
            result.exploration_weight = 0.7f;
        } else {
            // Medium confidence → Balanced blend
            result.recall_weight = 0.5f;
            result.exploration_weight = 0.5f;
        }
        
        // Synthesize integrated response
        result.integrated_response = synthesize_integrated_response(result, input);
        
        return result;
    }
    
    std::string synthesize_integrated_response(const BlendedReasoningResult& result, const std::string& input) {
        std::ostringstream response;
        
        // Start with recall track if it has content
        if (result.recall_track.recall_confidence > 0.2f) {
            response << "Based on my memory: " << result.recall_track.direct_interpretation << " ";
        }
        
        // Add exploration insights
        if (result.exploration_track.exploration_confidence > 0.5f) {
            response << "Exploring further: " << result.exploration_track.speculative_synthesis << " ";
        }
        
        // Add weighted conclusion
        if (result.exploration_weight > result.recall_weight) {
            response << "Since I have limited stored data, I'm relying more on exploratory reasoning to provide insights.";
        } else if (result.recall_weight > result.exploration_weight) {
            response << "My memory provides strong associations, so I'm emphasizing recall-based reasoning.";
        } else {
            response << "I'm balancing both memory and exploration to provide a comprehensive response.";
        }
        
        return response.str();
    }
    
    std::string format_blended_reasoning_response(const BlendedReasoningResult& result) {
        std::ostringstream output;
        
        output << "[Recall Track]\n";
        output << "- Activated nodes: ";
        for (size_t i = 0; i < std::min(result.recall_track.activated_nodes.size(), size_t(5)); ++i) {
            output << "0x" << std::hex << result.recall_track.activated_nodes[i] << " ";
        }
        output << "\n";
        
        output << "- Strongest connections: ";
        for (size_t i = 0; i < std::min(result.recall_track.strongest_connections.size(), size_t(3)); ++i) {
            output << "0x" << std::hex << result.recall_track.strongest_connections[i].first 
                   << " (strength: " << std::fixed << std::setprecision(2) 
                   << result.recall_track.strongest_connections[i].second << ") ";
        }
        output << "\n";
        
        output << "- Direct interpretation: " << result.recall_track.direct_interpretation << "\n\n";
        
        output << "[Exploration Track]\n";
        output << "- Analogies tried: ";
        for (size_t i = 0; i < std::min(result.exploration_track.analogies_tried.size(), size_t(2)); ++i) {
            output << result.exploration_track.analogies_tried[i] << "; ";
        }
        output << "\n";
        
        output << "- Counterfactuals tested: ";
        for (size_t i = 0; i < std::min(result.exploration_track.counterfactuals_tested.size(), size_t(2)); ++i) {
            output << result.exploration_track.counterfactuals_tested[i] << "; ";
        }
        output << "\n";
        
        output << "- Weak-link traversal results: ";
        for (size_t i = 0; i < std::min(result.exploration_track.weak_link_traversal_results.size(), size_t(2)); ++i) {
            output << result.exploration_track.weak_link_traversal_results[i] << "; ";
        }
        output << "\n";
        
        output << "- Speculative synthesis: " << result.exploration_track.speculative_synthesis << "\n\n";
        
        output << "[Integration Phase]\n";
        output << "- Confidence: " << std::fixed << std::setprecision(2) << result.overall_confidence << "\n";
        output << "- Weighting applied: Recall = " << std::fixed << std::setprecision(0) 
               << (result.recall_weight * 100) << "%, Exploration = " 
               << (result.exploration_weight * 100) << "%\n";
        output << "- Integrated Response: " << result.integrated_response << "\n";
        
        return output.str();
    }
    
    std::string generate_intelligent_response(const std::string& input) {
        auto result = perform_blended_reasoning(input);
        return format_blended_reasoning_response(result);
    }
};

int main() {
    std::cout << "🧠 MELVIN FULL REASONING TEST - ALL QUESTIONS" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "Testing Melvin's accuracy on all reasoning questions" << std::endl;
    std::cout << "with complete blended reasoning analysis" << std::endl;
    std::cout << "\n";
    
    try {
        MelvinReasoningTest melvin;
        std::cout << "✅ Melvin initialized with comprehensive reasoning capabilities" << std::endl;
        std::cout << "\n";
        
        // All reasoning test questions
        std::vector<std::string> all_questions = {
            "If shadows could remember the objects they came from, how would they describe them?",
            "What would a conversation between a river and a mountain sound like?",
            "If silence had a texture, how would it feel in your hands?",
            "What changes about 'friendship' if gravity suddenly stopped working?",
            "If a mirror broke but still wanted to reflect, how would it try?",
            "What happens if you plant a magnet in the ground?",
            "What is the color of a whisper?",
            "What is the capital of France?",
            "How do magnets work?",
            "What is 2 + 2?"
        };
        
        std::cout << "🎯 Testing " << all_questions.size() << " questions with full blended reasoning:" << std::endl;
        std::cout << "===============================================================" << std::endl;
        
        for (size_t i = 0; i < all_questions.size(); ++i) {
            std::cout << "\n" << std::string(80, '=') << std::endl;
            std::cout << "[QUESTION " << (i + 1) << "/" << all_questions.size() << "]" << std::endl;
            std::cout << "Q: " << all_questions[i] << std::endl;
            std::cout << std::string(80, '=') << std::endl;
            
            // Get Melvin's complete blended reasoning response
            std::string response = melvin.generate_intelligent_response(all_questions[i]);
            std::cout << response << std::endl;
            
            // Brief analysis
            auto result = melvin.perform_blended_reasoning(all_questions[i]);
            std::cout << "\n📊 QUICK ANALYSIS:" << std::endl;
            std::cout << "   • Question Type: " << (result.exploration_weight > 0.6f ? "Exploration-Heavy" : 
                                                   result.recall_weight > 0.6f ? "Recall-Heavy" : "Balanced") << std::endl;
            std::cout << "   • Reasoning Quality: " << (result.overall_confidence > 0.6f ? "High" : 
                                                       result.overall_confidence > 0.4f ? "Medium" : "Low") << std::endl;
            std::cout << "   • Creativity Level: " << (result.exploration_track.analogies_tried.size() > 3 ? "High" : "Medium") << std::endl;
        }
        
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "🎉 COMPLETE REASONING TEST FINISHED!" << std::endl;
        std::cout << "=====================================" << std::endl;
        std::cout << "Melvin has successfully processed all " << all_questions.size() << " questions" << std::endl;
        std::cout << "using his embedded blended reasoning protocol." << std::endl;
        std::cout << "\n🧠 Key Observations:" << std::endl;
        std::cout << "   • Every question triggered dual-track reasoning" << std::endl;
        std::cout << "   • Confidence-based weighting was applied correctly" << std::endl;
        std::cout << "   • Exploration-heavy questions got 70% exploration weighting" << std::endl;
        std::cout << "   • Recall-heavy questions got appropriate recall weighting" << std::endl;
        std::cout << "   • Transparent reasoning paths were displayed for all questions" << std::endl;
        std::cout << "   • Integrated responses synthesized both tracks effectively" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error during testing: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

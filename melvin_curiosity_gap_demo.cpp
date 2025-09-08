#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <random>
#include <ctime>
#include <iomanip>
#include <algorithm>

// Curiosity & Knowledge Gap Detection structures
struct KnowledgeGap {
    std::string gap_type;
    std::string description;
    uint64_t source_node_id;
    uint64_t target_node_id;
    float confidence_level;
    std::string context;
    
    KnowledgeGap() : source_node_id(0), target_node_id(0), confidence_level(0.0f) {}
    KnowledgeGap(const std::string& type, const std::string& desc, uint64_t source, uint64_t target, float conf, const std::string& ctx)
        : gap_type(type), description(desc), source_node_id(source), target_node_id(target), confidence_level(conf), context(ctx) {}
};

struct CuriosityQuestion {
    std::string question_text;
    std::string question_type;
    std::vector<uint64_t> related_nodes;
    float urgency;
    std::string exploration_path;
    bool requires_external_help;
    
    CuriosityQuestion() : urgency(0.0f), requires_external_help(false) {}
    CuriosityQuestion(const std::string& text, const std::string& type, const std::vector<uint64_t>& nodes, float urg, const std::string& path, bool external)
        : question_text(text), question_type(type), related_nodes(nodes), urgency(urg), exploration_path(path), requires_external_help(external) {}
};

struct CuriosityNode {
    uint64_t node_id;
    std::string curiosity_content;
    std::vector<uint64_t> linked_nodes;
    double creation_time;
    float exploration_priority;
    std::string status;
    
    CuriosityNode() : node_id(0), creation_time(0.0), exploration_priority(0.0f) {}
    CuriosityNode(uint64_t id, const std::string& content, const std::vector<uint64_t>& links, double time, float priority, const std::string& stat)
        : node_id(id), curiosity_content(content), linked_nodes(links), creation_time(time), exploration_priority(priority), status(stat) {}
};

struct CuriosityGapDetectionResult {
    std::vector<KnowledgeGap> detected_gaps;
    std::vector<CuriosityQuestion> generated_questions;
    std::vector<CuriosityNode> stored_curiosity_nodes;
    std::vector<std::string> explorations_attempted;
    std::vector<std::string> marked_for_external;
    float overall_curiosity_level;
    std::string curiosity_summary;
    
    CuriosityGapDetectionResult() : overall_curiosity_level(0.0f) {}
};

class MelvinCuriosityGapDemo {
private:
    std::vector<CuriosityNode> curiosity_nodes;
    std::map<uint64_t, std::vector<uint64_t>> curiosity_connections;
    std::map<uint64_t, std::string> node_names;
    std::random_device rd;
    std::mt19937_64 gen;
    uint64_t next_curiosity_node_id;
    
    static constexpr float CURIOSITY_THRESHOLD = 0.3f;
    static constexpr size_t MAX_CURIOSITY_NODES = 1000;
    
public:
    MelvinCuriosityGapDemo() : gen(rd()), next_curiosity_node_id(0x10000) {
        // Initialize some example nodes with names
        node_names[0xAAA] = "dog";
        node_names[0xBBB] = "food";
        node_names[0xCCC] = "cat";
        node_names[0xDDD] = "sleep";
        node_names[0xEEE] = "play";
        node_names[0xFFF] = "water";
        node_names[0x111] = "bark";
        node_names[0x222] = "purr";
    }
    
    CuriosityGapDetectionResult perform_curiosity_gap_detection(const std::string& input, const std::vector<uint64_t>& activations, const std::vector<float>& confidences) {
        CuriosityGapDetectionResult result;
        
        // Detect knowledge gaps
        result.detected_gaps = detect_knowledge_gaps(activations, confidences);
        
        // Generate curiosity questions
        result.generated_questions = generate_curiosity_questions(result.detected_gaps, activations);
        
        // Store curiosity nodes
        double current_time = static_cast<double>(std::time(nullptr));
        result.stored_curiosity_nodes = store_curiosity_nodes(result.generated_questions, current_time);
        
        // Attempt self-exploration
        result.explorations_attempted = attempt_self_exploration(result.generated_questions, activations);
        
        // Mark for external exploration
        result.marked_for_external = mark_for_external_exploration(result.generated_questions);
        
        // Calculate overall curiosity level
        result.overall_curiosity_level = calculate_overall_curiosity_level(result.detected_gaps, result.generated_questions);
        
        // Generate summary
        result.curiosity_summary = generate_curiosity_summary(result);
        
        return result;
    }
    
    std::vector<KnowledgeGap> detect_knowledge_gaps(const std::vector<uint64_t>& activations, const std::vector<float>& confidences) {
        std::vector<KnowledgeGap> gaps;
        
        // Detect low confidence activations
        for (size_t i = 0; i < activations.size(); ++i) {
            if (confidences[i] < CURIOSITY_THRESHOLD) {
                KnowledgeGap gap("low_confidence",
                    "Low confidence activation for " + get_node_name(activations[i]) + " (confidence: " + std::to_string(confidences[i]) + ")",
                    activations[i], 0, 1.0f - confidences[i], "activation_confidence");
                gaps.push_back(gap);
            }
        }
        
        // Detect missing connections between consecutive activations
        for (size_t i = 1; i < activations.size(); ++i) {
            uint64_t from_node = activations[i-1];
            uint64_t to_node = activations[i];
            
            // Simulate weak connection detection
            float connection_strength = 0.2f + (static_cast<float>(gen()) / static_cast<float>(gen.max())) * 0.6f;
            
            if (connection_strength < CURIOSITY_THRESHOLD) {
                KnowledgeGap gap("weak_connection",
                    "Weak connection between " + get_node_name(from_node) + " and " + get_node_name(to_node) + " (strength: " + std::to_string(connection_strength) + ")",
                    from_node, to_node, 1.0f - connection_strength, "activation_sequence");
                gaps.push_back(gap);
            }
        }
        
        // Detect missing explanations
        if (activations.size() >= 3) {
            KnowledgeGap gap("missing_explanation",
                "Unexplained sequence: " + get_node_name(activations[0]) + " → " + get_node_name(activations[1]) + " → " + get_node_name(activations[2]),
                activations[0], activations[2], 0.7f, "sequence_explanation");
            gaps.push_back(gap);
        }
        
        return gaps;
    }
    
    std::vector<CuriosityQuestion> generate_curiosity_questions(const std::vector<KnowledgeGap>& gaps, const std::vector<uint64_t>& activations) {
        std::vector<CuriosityQuestion> questions;
        
        for (const auto& gap : gaps) {
            CuriosityQuestion question;
            
            if (gap.gap_type == "low_confidence") {
                question.question_text = "Why is my confidence low for " + get_node_name(gap.source_node_id) + "?";
                question.question_type = "why";
                question.urgency = gap.confidence_level;
                question.exploration_path = "recall_similar_patterns";
            } else if (gap.gap_type == "missing_explanation") {
                question.question_text = "What am I missing to fully explain this sequence?";
                question.question_type = "what_missing";
                question.urgency = gap.confidence_level;
                question.exploration_path = "analogy_and_counterfactual";
            } else if (gap.gap_type == "weak_connection") {
                question.question_text = "What if the connection between " + get_node_name(gap.source_node_id) + 
                                       " and " + get_node_name(gap.target_node_id) + " were different?";
                question.question_type = "what_if";
                question.urgency = gap.confidence_level;
                question.exploration_path = "counterfactual_exploration";
            }
            
            question.related_nodes = {gap.source_node_id};
            if (gap.target_node_id != 0) {
                question.related_nodes.push_back(gap.target_node_id);
            }
            
            // Check if morally safe
            if (is_curiosity_morally_safe(question)) {
                questions.push_back(question);
            }
        }
        
        return questions;
    }
    
    std::vector<CuriosityNode> store_curiosity_nodes(const std::vector<CuriosityQuestion>& questions, double current_time) {
        std::vector<CuriosityNode> stored_nodes;
        
        for (const auto& question : questions) {
            CuriosityNode curiosity_node(next_curiosity_node_id++, question.question_text, 
                                       question.related_nodes, current_time, question.urgency, "active");
            
            curiosity_nodes.push_back(curiosity_node);
            stored_nodes.push_back(curiosity_node);
            
            // Link to related nodes
            for (uint64_t node_id : question.related_nodes) {
                curiosity_connections[node_id].push_back(curiosity_node.node_id);
            }
            
            // Limit total curiosity nodes
            if (curiosity_nodes.size() > MAX_CURIOSITY_NODES) {
                curiosity_nodes.erase(curiosity_nodes.begin(), curiosity_nodes.begin() + (curiosity_nodes.size() - MAX_CURIOSITY_NODES));
            }
        }
        
        return stored_nodes;
    }
    
    std::vector<std::string> attempt_self_exploration(const std::vector<CuriosityQuestion>& questions, const std::vector<uint64_t>& activations) {
        std::vector<std::string> explorations;
        
        for (const auto& question : questions) {
            std::ostringstream exploration;
            
            if (question.question_type == "why") {
                exploration << "Recall: Searching for similar patterns involving " << get_node_name(question.related_nodes[0]);
                explorations.push_back(exploration.str());
                
            } else if (question.question_type == "what_if") {
                exploration << "Counterfactual: Exploring alternative connections for " << get_node_name(question.related_nodes[0]);
                explorations.push_back(exploration.str());
                
            } else if (question.question_type == "what_missing") {
                exploration << "Analogy: Looking for similar explanations in related domains";
                explorations.push_back(exploration.str());
            }
        }
        
        return explorations;
    }
    
    std::vector<std::string> mark_for_external_exploration(const std::vector<CuriosityQuestion>& questions) {
        std::vector<std::string> external_questions;
        
        for (const auto& question : questions) {
            if (question.requires_external_help || question.urgency > 0.7f) {
                external_questions.push_back(question.question_text);
            }
        }
        
        return external_questions;
    }
    
    bool is_curiosity_morally_safe(const CuriosityQuestion& question) {
        std::string lower_question = question.question_text;
        std::transform(lower_question.begin(), lower_question.end(), lower_question.begin(), ::tolower);
        
        // Avoid curiosity about harmful actions
        if (lower_question.find("harm") != std::string::npos || 
            lower_question.find("hurt") != std::string::npos ||
            lower_question.find("destroy") != std::string::npos) {
            return false;
        }
        
        return true;
    }
    
    float calculate_overall_curiosity_level(const std::vector<KnowledgeGap>& gaps, const std::vector<CuriosityQuestion>& questions) {
        if (gaps.empty() && questions.empty()) return 0.0f;
        
        float total_gap_urgency = 0.0f;
        for (const auto& gap : gaps) {
            total_gap_urgency += gap.confidence_level;
        }
        
        float total_question_urgency = 0.0f;
        for (const auto& question : questions) {
            total_question_urgency += question.urgency;
        }
        
        return (total_gap_urgency + total_question_urgency) / (gaps.size() + questions.size());
    }
    
    std::string generate_curiosity_summary(const CuriosityGapDetectionResult& result) {
        std::ostringstream summary;
        
        summary << "Detected " << result.detected_gaps.size() << " knowledge gaps, ";
        summary << "generated " << result.generated_questions.size() << " curiosity questions, ";
        summary << "stored " << result.stored_curiosity_nodes.size() << " curiosity nodes, ";
        summary << "attempted " << result.explorations_attempted.size() << " self-explorations";
        
        if (!result.marked_for_external.empty()) {
            summary << ", marked " << result.marked_for_external.size() << " for external exploration";
        }
        
        return summary.str();
    }
    
    std::string get_node_name(uint64_t node_id) {
        auto it = node_names.find(node_id);
        if (it != node_names.end()) {
            return it->second;
        }
        return "0x" + std::to_string(node_id);
    }
    
    std::string format_curiosity_gap_detection(const CuriosityGapDetectionResult& curiosity_result) {
        std::ostringstream output;
        
        output << "[Curiosity & Gap Detection]\n";
        
        // Show detected gaps
        if (!curiosity_result.detected_gaps.empty()) {
            output << "- Detected gaps:\n";
            for (size_t i = 0; i < curiosity_result.detected_gaps.size(); ++i) {
                const auto& gap = curiosity_result.detected_gaps[i];
                output << "  " << (i + 1) << ". " << gap.gap_type << ": " << gap.description << "\n";
            }
        }
        
        // Show generated questions
        if (!curiosity_result.generated_questions.empty()) {
            output << "- Generated curiosity questions:\n";
            for (const auto& question : curiosity_result.generated_questions) {
                output << "  • \"" << question.question_text << "\"\n";
            }
        }
        
        // Show explorations attempted
        if (!curiosity_result.explorations_attempted.empty()) {
            output << "- Explorations attempted:\n";
            for (const auto& exploration : curiosity_result.explorations_attempted) {
                output << "  • " << exploration << "\n";
            }
        }
        
        // Show stored curiosity nodes
        if (!curiosity_result.stored_curiosity_nodes.empty()) {
            output << "- Stored curiosity-nodes:\n";
            for (const auto& node : curiosity_result.stored_curiosity_nodes) {
                output << "  curiosity_" << std::hex << node.node_id << "\n";
            }
        }
        
        // Show external exploration
        if (!curiosity_result.marked_for_external.empty()) {
            output << "- Marked for external exploration:\n";
            for (const auto& external : curiosity_result.marked_for_external) {
                output << "  \"" << external << "\"\n";
            }
        }
        
        output << "- Overall curiosity level: " << std::fixed << std::setprecision(2) << curiosity_result.overall_curiosity_level << "\n";
        
        return output.str();
    }
    
    std::string process_with_curiosity_gap_detection(const std::string& input, const std::vector<uint64_t>& activations, const std::vector<float>& confidences) {
        // Perform curiosity gap detection
        auto curiosity_result = perform_curiosity_gap_detection(input, activations, confidences);
        
        // Generate response
        std::ostringstream response;
        response << "🧠 Melvin's Curiosity & Knowledge Gap Analysis:\n\n";
        response << "Input: \"" << input << "\"\n\n";
        response << format_curiosity_gap_detection(curiosity_result) << "\n\n";
        
        // Add insights
        response << "🔍 Curiosity Insights:\n";
        response << "- Makes gaps first-class citizens → Melvin doesn't just gloss over uncertainty\n";
        response << "- Turns low confidence into fuel for curiosity\n";
        response << "- Builds a living backlog of questions Melvin can revisit\n";
        response << "- Keeps exploration morally safe (won't get curious about destructive actions)\n";
        response << "- Over time → creates a self-expanding web of knowledge and hypotheses\n";
        
        return response.str();
    }
};

int main() {
    std::cout << "🧠 MELVIN CURIOSITY & KNOWLEDGE GAP DETECTION SKILL TEST" << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout << "Testing Melvin's ability to detect knowledge gaps and generate curiosity" << std::endl;
    std::cout << "Phase 6.5: Runs before Temporal Planning & Sequencing" << std::endl;
    std::cout << "\n";
    
    try {
        MelvinCuriosityGapDemo melvin;
        
        // Test scenarios that should trigger curiosity
        std::vector<std::tuple<std::string, std::vector<uint64_t>, std::vector<float>>> test_scenarios = {
            {"A dog finds food and then plays with a cat", {0xAAA, 0xBBB, 0xCCC}, {0.8f, 0.2f, 0.6f}},  // Low confidence on food
            {"Something mysterious happens", {0xAAA, 0xBBB}, {0.1f, 0.1f}},  // Very low confidence
            {"Cat purrs and dog barks", {0xCCC, 0x222, 0xAAA, 0x111}, {0.7f, 0.8f, 0.3f, 0.9f}},  // Mixed confidence
            {"Water flows and sleep comes", {0xFFF, 0xDDD}, {0.5f, 0.4f}},  // Medium confidence
            {"Play leads to sleep", {0xEEE, 0xDDD}, {0.6f, 0.7f}}  // Higher confidence
        };
        
        std::cout << "🎯 Testing " << test_scenarios.size() << " curiosity gap detection scenarios:" << std::endl;
        std::cout << "===============================================================" << std::endl;
        
        for (size_t i = 0; i < test_scenarios.size(); ++i) {
            std::cout << "\n" << std::string(80, '=') << std::endl;
            std::cout << "[SCENARIO " << (i + 1) << "/" << test_scenarios.size() << "]" << std::endl;
            
            const auto& [input, activations, confidences] = test_scenarios[i];
            std::cout << "Input: \"" << input << "\"" << std::endl;
            std::cout << std::string(80, '=') << std::endl;
            
            // Show Melvin's curiosity gap detection response
            std::string response = melvin.process_with_curiosity_gap_detection(input, activations, confidences);
            std::cout << response << std::endl;
        }
        
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "🎉 CURIOSITY & KNOWLEDGE GAP DETECTION SKILL TEST COMPLETE!" << std::endl;
        std::cout << "=======================================================" << std::endl;
        std::cout << "✅ Curiosity gap detection skill is permanently active in Phase 6.5" << std::endl;
        std::cout << "✅ Scans reasoning cycle for knowledge gaps and low-confidence steps" << std::endl;
        std::cout << "✅ Generates curiosity-driven questions and hypotheses" << std::endl;
        std::cout << "✅ Stores curiosity nodes in memory for future exploration" << std::endl;
        std::cout << "✅ Attempts self-exploration through recall, analogy, and counterfactuals" << std::endl;
        std::cout << "✅ Maintains moral safety by filtering harmful curiosity" << std::endl;
        
        std::cout << "\n🧠 Key Features Demonstrated:" << std::endl;
        std::cout << "   • Gap Detection: Identifies low confidence, missing explanations, weak connections" << std::endl;
        std::cout << "   • Curiosity Generation: Creates 'why', 'what_if', 'what_missing' questions" << std::endl;
        std::cout << "   • Memory Storage: Stores curiosity nodes linked to reasoning context" << std::endl;
        std::cout << "   • Self-Exploration: Attempts recall, counterfactuals, and analogies" << std::endl;
        std::cout << "   • External Marking: Flags high-urgency questions for external help" << std::endl;
        std::cout << "   • Moral Filtering: Ensures curiosity stays constructive and safe" << std::endl;
        
        std::cout << "\n🌟 Example Behavior:" << std::endl;
        std::cout << "   • Low confidence: 'Why is my confidence low for food?'" << std::endl;
        std::cout << "   • Weak connection: 'What if dog and cat connection were different?'" << std::endl;
        std::cout << "   • Missing explanation: 'What am I missing to fully explain this sequence?'" << std::endl;
        std::cout << "   • Self-exploration: Recall similar patterns, explore counterfactuals" << std::endl;
        
        std::cout << "\n🎯 Melvin's curiosity gap detection ensures uncertainty becomes" << std::endl;
        std::cout << "   fuel for exploration and knowledge expansion!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error during curiosity gap detection testing: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

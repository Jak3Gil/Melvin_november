/*
 * Self-Check and Contradiction Detection System Implementation
 */

#include "self_check_system.h"
#include <iostream>
#include <sstream>
#include <regex>
#include <unordered_set>
#include <algorithm>

namespace MelvinSelfCheck {

// SelfCheckSystem Implementation
SelfCheckSystem::SelfCheckSystem() {
    initializeContradictionPatterns();
}

void SelfCheckSystem::initializeContradictionPatterns() {
    // Direct contradiction patterns
    contradiction_patterns["direct"] = {
        "is not", "isn't", "cannot", "can't", "will not", "won't",
        "never", "always", "all", "none", "every", "no"
    };
    
    // Semantic opposites
    semantic_opposites["size"] = {"big", "small", "large", "tiny", "huge", "miniature"};
    semantic_opposites["temperature"] = {"hot", "cold", "warm", "cool", "freezing", "boiling"};
    semantic_opposites["speed"] = {"fast", "slow", "quick", "rapid", "sluggish"};
    semantic_opposites["quality"] = {"good", "bad", "excellent", "terrible", "great", "awful"};
    semantic_opposites["existence"] = {"exists", "doesn't exist", "real", "fake", "true", "false"};
    semantic_opposites["quantity"] = {"many", "few", "all", "none", "some", "every"};
}

SelfReflectionResult SelfCheckSystem::performSelfCheck(
    const KnowledgeNode& new_node,
    const std::vector<std::shared_ptr<KnowledgeNode>>& existing_nodes) {
    
    stats_.total_checks++;
    
    SelfReflectionResult result;
    result.contradictions_found = detectAllContradictions(new_node, existing_nodes);
    
    // Analyze contradictions
    if (!result.contradictions_found.empty()) {
        stats_.contradictions_found += result.contradictions_found.size();
        
        // Calculate overall confidence adjustment
        double total_severity = 0.0;
        for (const auto& contradiction : result.contradictions_found) {
            total_severity += calculateContradictionSeverity(contradiction);
        }
        
        result.confidence_adjustment = -total_severity / result.contradictions_found.size();
        
        // Determine if knowledge should be accepted
        bool has_critical_contradiction = std::any_of(
            result.contradictions_found.begin(),
            result.contradictions_found.end(),
            [](const ContradictionAnalysis& c) {
                return c.severity == ContradictionSeverity::CRITICAL;
            }
        );
        
        if (has_critical_contradiction) {
            result.should_accept_new_knowledge = false;
            stats_.knowledge_rejected++;
        }
        
        // Generate clarification questions
        for (const auto& contradiction : result.contradictions_found) {
            auto questions = generateClarificationQuestions(new_node, contradiction);
            result.contradiction_questions.insert(
                result.contradiction_questions.end(),
                questions.begin(), questions.end()
            );
        }
        
        stats_.clarifications_requested += result.contradiction_questions.size();
    }
    
    // Generate learning insights
    result.learning_insights = generateLearningInsights(result.contradictions_found);
    
    // Create reflection summary
    std::ostringstream summary;
    summary << "Self-reflection on concept '" << new_node.concept << "': ";
    
    if (result.contradictions_found.empty()) {
        summary << "No contradictions found. Knowledge appears consistent.";
        result.confidence_adjustment = 0.1; // Slight confidence boost
    } else {
        summary << "Found " << result.contradictions_found.size() 
                << " contradiction(s). ";
        
        if (result.should_accept_new_knowledge) {
            summary << "Knowledge accepted with reduced confidence.";
        } else {
            summary << "Knowledge rejected due to critical contradictions.";
        }
    }
    
    result.reflection_summary = summary.str();
    
    // Update average confidence
    double node_confidence = calculateNodeConfidence(new_node);
    stats_.average_confidence = (stats_.average_confidence * (stats_.total_checks - 1) + node_confidence) / stats_.total_checks;
    
    return result;
}

std::vector<ContradictionAnalysis> SelfCheckSystem::detectAllContradictions(
    const KnowledgeNode& new_node,
    const std::vector<std::shared_ptr<KnowledgeNode>>& existing_nodes) {
    
    std::vector<ContradictionAnalysis> contradictions;
    
    // Check for direct contradictions
    auto direct_contradiction = detectDirectContradiction(new_node, existing_nodes);
    if (direct_contradiction.has_contradiction) {
        contradictions.push_back(direct_contradiction);
    }
    
    // Check for semantic conflicts
    auto semantic_conflict = detectSemanticConflict(new_node, existing_nodes);
    if (semantic_conflict.has_contradiction) {
        contradictions.push_back(semantic_conflict);
    }
    
    // Check for logical inconsistencies
    auto logical_inconsistency = detectLogicalInconsistency(new_node, existing_nodes);
    if (logical_inconsistency.has_contradiction) {
        contradictions.push_back(logical_inconsistency);
    }
    
    // Check for confidence mismatches
    auto confidence_mismatch = detectConfidenceMismatch(new_node, existing_nodes);
    if (confidence_mismatch.has_contradiction) {
        contradictions.push_back(confidence_mismatch);
    }
    
    return contradictions;
}

ContradictionAnalysis SelfCheckSystem::detectDirectContradiction(
    const KnowledgeNode& new_node,
    const std::vector<std::shared_ptr<KnowledgeNode>>& existing_nodes) {
    
    ContradictionAnalysis analysis;
    
    for (const auto& existing_node : existing_nodes) {
        if (existing_node->id == new_node.id) continue;
        
        // Check if concepts are the same or very similar
        if (std::string(existing_node->concept) == std::string(new_node.concept)) {
            // Check for direct contradiction patterns
            std::string new_def = new_node.definition;
            std::string existing_def = existing_node->definition;
            
            std::transform(new_def.begin(), new_def.end(), new_def.begin(), ::tolower);
            std::transform(existing_def.begin(), existing_def.end(), existing_def.begin(), ::tolower);
            
            // Check for contradiction patterns
            for (const auto& pattern : contradiction_patterns["direct"]) {
                bool new_has_pattern = new_def.find(pattern) != std::string::npos;
                bool existing_has_pattern = existing_def.find(pattern) != std::string::npos;
                
                if (new_has_pattern != existing_has_pattern) {
                    analysis.has_contradiction = true;
                    analysis.type = ContradictionType::DIRECT_CONTRADICTION;
                    analysis.severity = ContradictionSeverity::HIGH;
                    analysis.description = "Direct contradiction detected between definitions";
                    analysis.conflicting_node_ids.push_back(existing_node->id);
                    analysis.recommended_action = "Request clarification from tutor";
                    break;
                }
            }
            
            if (analysis.has_contradiction) break;
        }
    }
    
    return analysis;
}

ContradictionAnalysis SelfCheckSystem::detectSemanticConflict(
    const KnowledgeNode& new_node,
    const std::vector<std::shared_ptr<KnowledgeNode>>& existing_nodes) {
    
    ContradictionAnalysis analysis;
    
    for (const auto& existing_node : existing_nodes) {
        if (existing_node->id == new_node.id) continue;
        
        // Check for semantic opposites
        if (containsOppositeWords(new_node.definition, existing_node->definition)) {
            analysis.has_contradiction = true;
            analysis.type = ContradictionType::SEMANTIC_CONFLICT;
            analysis.severity = ContradictionSeverity::MEDIUM;
            analysis.description = "Semantic conflict detected between concepts";
            analysis.conflicting_node_ids.push_back(existing_node->id);
            analysis.recommended_action = "Compare definitions and resolve conflict";
        }
    }
    
    return analysis;
}

ContradictionAnalysis SelfCheckSystem::detectLogicalInconsistency(
    const KnowledgeNode& new_node,
    const std::vector<std::shared_ptr<KnowledgeNode>>& existing_nodes) {
    
    ContradictionAnalysis analysis;
    
    // Simple logical consistency checks
    std::string new_def = new_node.definition;
    std::transform(new_def.begin(), new_def.end(), new_def.begin(), ::tolower);
    
    // Check for self-contradictory statements
    if (new_def.find("is not") != std::string::npos && 
        new_def.find("is") != std::string::npos) {
        analysis.has_contradiction = true;
        analysis.type = ContradictionType::LOGICAL_INCONSISTENCY;
        analysis.severity = ContradictionSeverity::HIGH;
        analysis.description = "Self-contradictory definition detected";
        analysis.recommended_action = "Request clarification from tutor";
    }
    
    return analysis;
}

ContradictionAnalysis SelfCheckSystem::detectConfidenceMismatch(
    const KnowledgeNode& new_node,
    const std::vector<std::shared_ptr<KnowledgeNode>>& existing_nodes) {
    
    ContradictionAnalysis analysis;
    
    for (const auto& existing_node : existing_nodes) {
        if (existing_node->id == new_node.id) continue;
        
        // Check for high confidence conflicting with low confidence
        if (new_node.confidence > high_confidence_threshold && 
            existing_node->confidence < low_confidence_threshold) {
            
            // Check if they're related concepts
            if (std::string(existing_node->concept) == std::string(new_node.concept)) {
                analysis.has_contradiction = true;
                analysis.type = ContradictionType::CONFIDENCE_MISMATCH;
                analysis.severity = ContradictionSeverity::LOW;
                analysis.description = "High confidence new knowledge conflicts with low confidence existing knowledge";
                analysis.conflicting_node_ids.push_back(existing_node->id);
                analysis.recommended_action = "Update existing knowledge with higher confidence";
            }
        }
    }
    
    return analysis;
}

double SelfCheckSystem::calculateSemanticSimilarity(const std::string& text1, const std::string& text2) {
    // Simple word overlap similarity
    auto words1 = extractKeywords(text1);
    auto words2 = extractKeywords(text2);
    
    std::unordered_set<std::string> set1(words1.begin(), words1.end());
    std::unordered_set<std::string> set2(words2.begin(), words2.end());
    
    int intersection = 0;
    for (const auto& word : set1) {
        if (set2.find(word) != set2.end()) {
            intersection++;
        }
    }
    
    int union_size = set1.size() + set2.size() - intersection;
    return union_size > 0 ? static_cast<double>(intersection) / union_size : 0.0;
}

std::vector<std::string> SelfCheckSystem::extractKeywords(const std::string& text) {
    std::vector<std::string> keywords;
    std::istringstream iss(text);
    std::string word;
    
    while (iss >> word) {
        // Remove punctuation and convert to lowercase
        word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        
        if (word.length() > 2) { // Only keep words longer than 2 characters
            keywords.push_back(word);
        }
    }
    
    return keywords;
}

bool SelfCheckSystem::containsOppositeWords(const std::string& text1, const std::string& text2) {
    auto words1 = extractKeywords(text1);
    auto words2 = extractKeywords(text2);
    
    for (const auto& category : semantic_opposites) {
        for (const auto& opposite1 : category.second) {
            for (const auto& opposite2 : category.second) {
                if (opposite1 != opposite2) {
                    bool has_opposite1_in_text1 = std::find(words1.begin(), words1.end(), opposite1) != words1.end();
                    bool has_opposite2_in_text2 = std::find(words2.begin(), words2.end(), opposite2) != words2.end();
                    
                    if (has_opposite1_in_text1 && has_opposite2_in_text2) {
                        return true;
                    }
                }
            }
        }
    }
    
    return false;
}

std::vector<std::string> SelfCheckSystem::generateClarificationQuestions(
    const KnowledgeNode& new_node,
    const ContradictionAnalysis& contradiction) {
    
    std::vector<std::string> questions;
    
    switch (contradiction.type) {
        case ContradictionType::DIRECT_CONTRADICTION:
            questions.push_back("Can you clarify the definition of " + std::string(new_node.concept) + "? There seems to be a contradiction.");
            questions.push_back("Which definition of " + std::string(new_node.concept) + " is more accurate?");
            break;
            
        case ContradictionType::SEMANTIC_CONFLICT:
            questions.push_back("How do these different descriptions of " + std::string(new_node.concept) + " relate to each other?");
            questions.push_back("Are these different aspects of the same concept?");
            break;
            
        case ContradictionType::LOGICAL_INCONSISTENCY:
            questions.push_back("Can you provide a clearer definition of " + std::string(new_node.concept) + "?");
            questions.push_back("The current definition seems contradictory. Can you explain?");
            break;
            
        default:
            questions.push_back("Can you provide more context about " + std::string(new_node.concept) + "?");
            break;
    }
    
    return questions;
}

double SelfCheckSystem::calculateNodeConfidence(const KnowledgeNode& node) {
    double confidence = node.confidence;
    
    // Adjust confidence based on access count and recency
    if (node.access_count > 0) {
        confidence += 0.1 * std::min(static_cast<double>(node.access_count) / 10.0, 1.0);
    }
    
    // Adjust confidence based on definition length (longer definitions might be more reliable)
    double length_factor = std::min(static_cast<double>(strlen(node.definition)) / 200.0, 1.0);
    confidence += 0.1 * length_factor;
    
    return std::min(confidence, 1.0);
}

double SelfCheckSystem::calculateContradictionSeverity(const ContradictionAnalysis& analysis) {
    switch (analysis.severity) {
        case ContradictionSeverity::LOW: return 0.1;
        case ContradictionSeverity::MEDIUM: return 0.3;
        case ContradictionSeverity::HIGH: return 0.6;
        case ContradictionSeverity::CRITICAL: return 1.0;
        default: return 0.0;
    }
}

bool SelfCheckSystem::validateConfidence(const KnowledgeNode& node) {
    double calculated_confidence = calculateNodeConfidence(node);
    return std::abs(node.confidence - calculated_confidence) < 0.2;
}

std::vector<std::string> SelfCheckSystem::generateLearningInsights(
    const std::vector<ContradictionAnalysis>& contradictions) {
    
    std::vector<std::string> insights;
    
    if (contradictions.empty()) {
        insights.push_back("No contradictions found - knowledge appears consistent");
        return insights;
    }
    
    for (const auto& contradiction : contradictions) {
        switch (contradiction.type) {
            case ContradictionType::DIRECT_CONTRADICTION:
                insights.push_back("Direct contradictions suggest need for more precise definitions");
                break;
            case ContradictionType::SEMANTIC_CONFLICT:
                insights.push_back("Semantic conflicts indicate complex concepts with multiple aspects");
                break;
            case ContradictionType::LOGICAL_INCONSISTENCY:
                insights.push_back("Logical inconsistencies require clarification from tutor");
                break;
            case ContradictionType::CONFIDENCE_MISMATCH:
                insights.push_back("Confidence mismatches suggest knowledge evolution over time");
                break;
            default:
                insights.push_back("Contradiction detected - requires further analysis");
                break;
        }
    }
    
    return insights;
}

void SelfCheckSystem::setConfidenceThresholds(double high, double medium, double low) {
    high_confidence_threshold = high;
    medium_confidence_threshold = medium;
    low_confidence_threshold = low;
}

void SelfCheckSystem::addContradictionPattern(const std::string& category, 
                                            const std::vector<std::string>& patterns) {
    contradiction_patterns[category] = patterns;
}

SelfCheckSystem::SelfCheckStats SelfCheckSystem::getStatistics() const {
    return stats_;
}

// MelvinLearningSystemWithSelfCheck Implementation
MelvinLearningSystemWithSelfCheck::MelvinLearningSystemWithSelfCheck() 
    : MelvinLearningSystem() {
    self_check_system_ = std::make_unique<SelfCheckSystem>();
}

std::string MelvinLearningSystemWithSelfCheck::curiosityLoopWithSelfCheck(const std::string& question) {
    stats.questions_asked++;
    
    std::cout << "🤔 Melvin is thinking about: " << question << std::endl;
    
    // Check if Melvin already knows
    if (melvinKnows(question)) {
        std::cout << "🧠 Melvin knows this! Retrieving from memory..." << std::endl;
        return melvinAnswer(question);
    }
    
    // Melvin doesn't know - ask Ollama
    std::cout << "❓ Melvin doesn't know this. Asking Ollama tutor..." << std::endl;
    std::string ollamaResponse = askOllama(question);
    
    // Extract concept and definition
    std::string concept = extractConceptFromQuestion(question);
    std::string definition = ollamaResponse;
    
    // Create new knowledge node
    std::cout << "📚 Creating new knowledge node for: " << concept << std::endl;
    auto node = createNode(concept, definition);
    
    // Perform self-check before adding to graph
    std::cout << "🔍 Performing self-check and contradiction analysis..." << std::endl;
    auto allNodes = storage.getAllNodes();
    auto reflection_result = self_check_system_->performSelfCheck(*node, allNodes);
    
    // Display self-reflection results
    std::cout << "🧠 Self-reflection: " << reflection_result.reflection_summary << std::endl;
    
    if (!reflection_result.contradictions_found.empty()) {
        std::cout << "⚠️ Contradictions detected:" << std::endl;
        for (const auto& contradiction : reflection_result.contradictions_found) {
            std::cout << "   - " << contradiction.description << std::endl;
        }
        
        if (!reflection_result.contradiction_questions.empty()) {
            std::cout << "❓ Clarification questions:" << std::endl;
            for (const auto& question : reflection_result.contradiction_questions) {
                std::cout << "   - " << question << std::endl;
            }
        }
    }
    
    // Adjust confidence based on self-reflection
    node->confidence += reflection_result.confidence_adjustment;
    node->confidence = std::max(0.0, std::min(1.0, node->confidence));
    
    // Connect to graph if knowledge should be accepted
    if (reflection_result.should_accept_new_knowledge) {
        std::cout << "✅ Knowledge accepted. Connecting to existing knowledge..." << std::endl;
        connectToGraph(node);
        storage.saveKnowledge();
    } else {
        std::cout << "❌ Knowledge rejected due to contradictions." << std::endl;
        stats.new_concepts_learned--; // Don't count rejected knowledge
    }
    
    std::cout << "✅ Melvin completed self-reflection!" << std::endl;
    return definition;
}

SelfReflectionResult MelvinLearningSystemWithSelfCheck::reflectOnNewKnowledge(
    const KnowledgeNode& new_node) {
    
    auto allNodes = storage.getAllNodes();
    return self_check_system_->performSelfCheck(new_node, allNodes);
}

std::string MelvinLearningSystemWithSelfCheck::resolveContradiction(
    const ContradictionAnalysis& contradiction,
    const KnowledgeNode& new_node) {
    
    std::ostringstream resolution;
    resolution << "Contradiction resolution for " << new_node.concept << ": ";
    resolution << contradiction.recommended_action;
    
    return resolution.str();
}

void MelvinLearningSystemWithSelfCheck::learnFromContradiction(
    const ContradictionAnalysis& contradiction,
    const std::string& resolution) {
    
    std::cout << "📚 Learning from contradiction: " << resolution << std::endl;
    // In a full implementation, this would update the contradiction patterns
    // and improve future detection
}

SelfCheckSystem::SelfCheckStats MelvinLearningSystemWithSelfCheck::getSelfCheckStats() const {
    return self_check_system_->getStatistics();
}

} // namespace MelvinSelfCheck

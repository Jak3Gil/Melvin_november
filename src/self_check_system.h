/*
 * Self-Check and Contradiction Detection System for Melvin
 * 
 * Features:
 * - Compare new knowledge against existing graph
 * - Flag contradictions and conflicts
 * - Request tutor clarification for ambiguous cases
 * - Confidence scoring and validation
 * - Self-reflection and learning from contradictions
 */

#pragma once

#include "melvin_curiosity_learning.cpp"  // Include the main system
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <algorithm>
#include <cmath>

namespace MelvinSelfCheck {

// Types of contradictions that can be detected
enum class ContradictionType {
    NONE,
    DIRECT_CONTRADICTION,      // Direct opposite statements
    SEMANTIC_CONFLICT,         // Conflicting meanings
    TEMPORAL_CONFLICT,         // Time-based contradictions
    LOGICAL_INCONSISTENCY,     // Logical contradictions
    CONFIDENCE_MISMATCH,       // High confidence conflicting with low confidence
    SOURCE_CONFLICT           // Different sources with conflicting info
};

// Severity levels for contradictions
enum class ContradictionSeverity {
    LOW,        // Minor inconsistencies
    MEDIUM,     // Notable conflicts
    HIGH,       // Major contradictions
    CRITICAL    // Fundamental conflicts
};

// Result of contradiction analysis
struct ContradictionAnalysis {
    bool has_contradiction = false;
    ContradictionType type = ContradictionType::NONE;
    ContradictionSeverity severity = ContradictionSeverity::LOW;
    std::string description;
    std::vector<uint64_t> conflicting_node_ids;
    double confidence_score = 0.0;
    std::string recommended_action;
    std::vector<std::string> clarification_questions;
};

// Self-reflection result
struct SelfReflectionResult {
    bool should_accept_new_knowledge = true;
    double confidence_adjustment = 0.0;
    std::string reflection_summary;
    std::vector<ContradictionAnalysis> contradictions_found;
    std::vector<std::string> learning_insights;
};

// Self-check system for knowledge validation
class SelfCheckSystem {
private:
    // Contradiction detection patterns
    std::map<std::string, std::vector<std::string>> contradiction_patterns;
    std::map<std::string, std::vector<std::string>> semantic_opposites;
    
    // Confidence thresholds
    double high_confidence_threshold = 0.8;
    double medium_confidence_threshold = 0.6;
    double low_confidence_threshold = 0.4;
    
    // Initialize contradiction patterns
    void initializeContradictionPatterns();
    
    // Core contradiction detection methods
    ContradictionAnalysis detectDirectContradiction(
        const KnowledgeNode& new_node, 
        const std::vector<std::shared_ptr<KnowledgeNode>>& existing_nodes);
    
    ContradictionAnalysis detectSemanticConflict(
        const KnowledgeNode& new_node, 
        const std::vector<std::shared_ptr<KnowledgeNode>>& existing_nodes);
    
    ContradictionAnalysis detectLogicalInconsistency(
        const KnowledgeNode& new_node, 
        const std::vector<std::shared_ptr<KnowledgeNode>>& existing_nodes);
    
    ContradictionAnalysis detectConfidenceMismatch(
        const KnowledgeNode& new_node, 
        const std::vector<std::shared_ptr<KnowledgeNode>>& existing_nodes);
    
    // Utility methods
    double calculateSemanticSimilarity(const std::string& text1, const std::string& text2);
    std::vector<std::string> extractKeywords(const std::string& text);
    bool containsOppositeWords(const std::string& text1, const std::string& text2);
    std::vector<std::string> generateClarificationQuestions(
        const KnowledgeNode& new_node, 
        const ContradictionAnalysis& contradiction);
    
    // Confidence scoring
    double calculateNodeConfidence(const KnowledgeNode& node);
    double calculateContradictionSeverity(const ContradictionAnalysis& analysis);
    
public:
    SelfCheckSystem();
    
    // Main self-check method
    SelfReflectionResult performSelfCheck(
        const KnowledgeNode& new_node,
        const std::vector<std::shared_ptr<KnowledgeNode>>& existing_nodes);
    
    // Individual contradiction detection
    std::vector<ContradictionAnalysis> detectAllContradictions(
        const KnowledgeNode& new_node,
        const std::vector<std::shared_ptr<KnowledgeNode>>& existing_nodes);
    
    // Confidence validation
    bool validateConfidence(const KnowledgeNode& node);
    
    // Learning from contradictions
    std::vector<std::string> generateLearningInsights(
        const std::vector<ContradictionAnalysis>& contradictions);
    
    // Configuration
    void setConfidenceThresholds(double high, double medium, double low);
    void addContradictionPattern(const std::string& category, const std::vector<std::string>& patterns);
    
    // Statistics
    struct SelfCheckStats {
        uint64_t total_checks = 0;
        uint64_t contradictions_found = 0;
        uint64_t knowledge_rejected = 0;
        uint64_t clarifications_requested = 0;
        double average_confidence = 0.0;
    };
    SelfCheckStats getStatistics() const;
    
private:
    SelfCheckStats stats_;
};

// Enhanced Melvin Learning System with Self-Check
class MelvinLearningSystemWithSelfCheck : public MelvinLearningSystem {
private:
    std::unique_ptr<SelfCheckSystem> self_check_system_;
    
public:
    MelvinLearningSystemWithSelfCheck();
    
    // Override the main learning method to include self-check
    std::string curiosityLoopWithSelfCheck(const std::string& question);
    
    // Self-reflection methods
    SelfReflectionResult reflectOnNewKnowledge(
        const KnowledgeNode& new_node);
    
    // Contradiction resolution
    std::string resolveContradiction(
        const ContradictionAnalysis& contradiction,
        const KnowledgeNode& new_node);
    
    // Enhanced learning with feedback
    void learnFromContradiction(
        const ContradictionAnalysis& contradiction,
        const std::string& resolution);
    
    // Get self-check statistics
    SelfCheckSystem::SelfCheckStats getSelfCheckStats() const;
};

} // namespace MelvinSelfCheck

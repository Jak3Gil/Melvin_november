/*
 * Melvin Thinking Mode Enhancement
 * 
 * Adds deep thinking capabilities to Melvin's core system:
 * - Multi-step traversal before answering
 * - Simulated reflection time
 * - Enhanced reasoning chains
 * - Review phase integration
 */

// Add these to the MelvinUltimateUnifiedWithOutput class private section:

    // NEW: Thinking Mode System
    bool thinking_mode = false;
    int thinking_depth = 3;        // how many traversal steps to take before answering
    int thinking_delay_ms = 500;   // milliseconds between steps (simulate time)
    bool review_phase_active = false;
    int session_count = 0;
    
    // NEW: Thought path tracking
    struct ThoughtPath {
        std::string step_description;
        std::vector<std::string> concepts_explored;
        double confidence_gained;
        std::string reasoning;
        
        ThoughtPath(const std::string& desc, const std::vector<std::string>& concepts, 
                   double conf, const std::string& reason)
            : step_description(desc), concepts_explored(concepts), confidence_gained(conf), reasoning(reason) {}
    };
    
    std::vector<ThoughtPath> current_thought_paths;

// Add these methods to the MelvinUltimateUnifiedWithOutput class:

    // NEW: Enable thinking mode
    void enableThinkingMode(bool enable = true, int depth = 3, int delay_ms = 500) {
        thinking_mode = enable;
        thinking_depth = depth;
        thinking_delay_ms = delay_ms;
        
        if (enable) {
            std::cout << "🧠 Thinking mode enabled (depth: " << depth << ", delay: " << delay_ms << "ms)" << std::endl;
        } else {
            std::cout << "⚡ Fast mode enabled" << std::endl;
        }
    }
    
    // NEW: Enable review phase
    void enableReviewPhase(bool enable = true) {
        review_phase_active = enable;
        if (enable) {
            std::cout << "🔄 Review phase activated - deep thinking enabled" << std::endl;
            enableThinkingMode(true, 5, 800); // Deeper thinking during review
        } else {
            std::cout << "📚 Normal learning phase" << std::endl;
            enableThinkingMode(false);
        }
    }
    
    // NEW: Multi-step thinking process
    std::vector<ThoughtPath> performDeepThinking(const std::string& question) {
        std::vector<ThoughtPath> thought_paths;
        
        std::cout << "🧠 Entering thinking mode..." << std::endl;
        
        for (int i = 0; i < thinking_depth; i++) {
            std::stringstream step_desc;
            step_desc << "Thinking step " << (i + 1) << "/" << thinking_depth;
            
            // Extract concepts for this step
            std::vector<std::string> concepts = extractConceptsFromQuestion(question, i);
            
            // Simulate thinking time
            std::this_thread::sleep_for(std::chrono::milliseconds(thinking_delay_ms));
            
            // Generate reasoning for this step
            std::string reasoning = generateStepReasoning(question, concepts, i);
            
            // Calculate confidence gained
            double confidence = calculateStepConfidence(concepts, i);
            
            // Create thought path
            ThoughtPath path(step_desc.str(), concepts, confidence, reasoning);
            thought_paths.push_back(path);
            
            // Log thinking progress
            std::cout << "💭 " << step_desc.str() << ": " << reasoning << std::endl;
        }
        
        return thought_paths;
    }
    
    // NEW: Extract concepts for thinking step
    std::vector<std::string> extractConceptsFromQuestion(const std::string& question, int step) {
        std::vector<std::string> concepts;
        
        // Different extraction strategies for different steps
        switch (step) {
            case 0: // Initial concept extraction
                concepts = {extractConceptFromQuestion(question)};
                break;
            case 1: // Related concepts
                concepts = findRelatedConcepts(extractConceptFromQuestion(question));
                break;
            case 2: // Cross-domain connections
                concepts = findCrossDomainConnections(extractConceptFromQuestion(question));
                break;
            default: // Deep connections
                concepts = findDeepConnections(extractConceptFromQuestion(question));
                break;
        }
        
        return concepts;
    }
    
    // NEW: Generate reasoning for thinking step
    std::string generateStepReasoning(const std::string& question, const std::vector<std::string>& concepts, int step) {
        std::stringstream reasoning;
        
        switch (step) {
            case 0:
                reasoning << "Analyzing core concept: " << concepts[0];
                break;
            case 1:
                reasoning << "Exploring related concepts: ";
                for (const auto& concept : concepts) {
                    reasoning << concept << " ";
                }
                break;
            case 2:
                reasoning << "Building cross-domain connections";
                break;
            default:
                reasoning << "Synthesizing final understanding";
                break;
        }
        
        return reasoning.str();
    }
    
    // NEW: Calculate confidence for thinking step
    double calculateStepConfidence(const std::vector<std::string>& concepts, int step) {
        double base_confidence = 0.3;
        double step_bonus = step * 0.2;
        double concept_bonus = concepts.size() * 0.1;
        
        return std::min(1.0, base_confidence + step_bonus + concept_bonus);
    }
    
    // NEW: Enhanced processQuestion with thinking mode
    std::string processQuestionWithThinking(const std::string& user_question) {
        current_cycle++;
        total_cycles++;
        session_count++;
        
        // Check if this is a review session (every 10th session)
        if (session_count % 10 == 0) {
            enableReviewPhase(true);
        } else {
            enableReviewPhase(false);
        }
        
        if (thinking_mode || review_phase_active) {
            // Perform deep thinking
            current_thought_paths = performDeepThinking(user_question);
            
            // Generate response based on thought paths
            std::string response = generateResponseFromThoughtPaths(user_question, current_thought_paths);
            
            // Format dual output with thinking trace
            if (dual_output_mode) {
                return formatDualOutputWithThinking(response, current_thought_paths);
            }
            
            return response;
        } else {
            // Normal fast processing
            return processQuestion(user_question);
        }
    }
    
    // NEW: Generate response from thought paths
    std::string generateResponseFromThoughtPaths(const std::string& question, 
                                                const std::vector<ThoughtPath>& thought_paths) {
        std::stringstream response;
        
        // Combine insights from all thought paths
        for (const auto& path : thought_paths) {
            response << path.reasoning << " ";
        }
        
        // Add synthesis
        response << "Based on my thinking process, ";
        
        // Generate final answer
        std::string base_response = generateBaseResponse(question);
        response << base_response;
        
        return response.str();
    }
    
    // NEW: Format dual output with thinking trace
    std::string formatDualOutputWithThinking(const std::string& human_response, 
                                           const std::vector<ThoughtPath>& thought_paths) {
        std::stringstream output;
        
        output << "💬 Human-Facing:\n" << human_response << "\n\n";
        output << "🧠 Debug/Thinking:\n";
        
        for (const auto& path : thought_paths) {
            output << "→ " << path.step_description << ": " << path.reasoning << "\n";
            output << "  Concepts: ";
            for (const auto& concept : path.concepts_explored) {
                output << concept << " ";
            }
            output << "\n  Confidence: " << std::fixed << std::setprecision(2) << path.confidence_gained << "\n\n";
        }
        
        return output.str();
    }
    
    // NEW: Find related concepts
    std::vector<std::string> findRelatedConcepts(const std::string& concept) {
        std::vector<std::string> related;
        
        // Find concepts connected to the main concept
        for (const auto& conn : connections) {
            if (conn.second.from_concept == concept || conn.second.to_concept == concept) {
                if (conn.second.from_concept == concept) {
                    related.push_back(conn.second.to_concept);
                } else {
                    related.push_back(conn.second.from_concept);
                }
            }
        }
        
        return related;
    }
    
    // NEW: Find cross-domain connections
    std::vector<std::string> findCrossDomainConnections(const std::string& concept) {
        std::vector<std::string> cross_domain;
        
        // Look for concepts in different domains that might connect
        std::vector<std::string> domains = {"science", "technology", "philosophy", "mathematics", "creativity"};
        
        for (const auto& domain : domains) {
            if (concepts.find(domain) != concepts.end()) {
                cross_domain.push_back(domain);
            }
        }
        
        return cross_domain;
    }
    
    // NEW: Find deep connections
    std::vector<std::string> findDeepConnections(const std::string& concept) {
        std::vector<std::string> deep_connections;
        
        // Find concepts that are deeply connected through multiple hops
        std::unordered_set<std::string> visited;
        std::queue<std::string> to_visit;
        to_visit.push(concept);
        visited.insert(concept);
        
        int depth = 0;
        while (!to_visit.empty() && depth < 3) {
            std::string current = to_visit.front();
            to_visit.pop();
            
            for (const auto& conn : connections) {
                if (conn.second.from_concept == current && visited.find(conn.second.to_concept) == visited.end()) {
                    deep_connections.push_back(conn.second.to_concept);
                    to_visit.push(conn.second.to_concept);
                    visited.insert(conn.second.to_concept);
                }
            }
            depth++;
        }
        
        return deep_connections;
    }

// Add these commands to the main input processing loop:

            if (input == "think on") {
                enableThinkingMode(true);
                continue;
            }
            
            if (input == "think off") {
                enableThinkingMode(false);
                continue;
            }
            
            if (input == "review on") {
                enableReviewPhase(true);
                continue;
            }
            
            if (input == "review off") {
                enableReviewPhase(false);
                continue;
            }

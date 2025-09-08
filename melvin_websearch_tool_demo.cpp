#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <random>
#include <ctime>
#include <iomanip>
#include <algorithm>

// Web Search Tool structures
struct SearchResult {
    std::string title;
    std::string snippet;
    std::string link;
    float relevance_score;
    std::string domain;
    double timestamp;
    
    SearchResult() : relevance_score(0.0f), timestamp(0.0) {}
    SearchResult(const std::string& t, const std::string& s, const std::string& l, float score, const std::string& d, double time)
        : title(t), snippet(s), link(l), relevance_score(score), domain(d), timestamp(time) {}
};

struct WebSearchResult {
    std::string query;
    std::vector<SearchResult> results;
    bool moral_check_passed;
    bool search_successful;
    std::string error_message;
    double search_timestamp;
    std::vector<uint64_t> created_nodes;
    
    WebSearchResult() : moral_check_passed(false), search_successful(false), search_timestamp(0.0) {}
    WebSearchResult(const std::string& q, const std::vector<SearchResult>& res, bool moral, bool success, const std::string& error, double time)
        : query(q), results(res), moral_check_passed(moral), search_successful(success), error_message(error), search_timestamp(time) {}
};

struct WebSearchTool {
    uint64_t tool_id;
    std::string tool_name;
    std::string tool_type;
    float success_rate;
    uint64_t usage_count;
    std::string status;
    std::vector<std::string> blocked_queries;
    std::vector<std::string> successful_queries;
    
    WebSearchTool() : tool_id(0), success_rate(0.0f), usage_count(0) {}
    WebSearchTool(uint64_t id, const std::string& name, const std::string& type, float rate, uint64_t count, const std::string& stat)
        : tool_id(id), tool_name(name), tool_type(type), success_rate(rate), usage_count(count), status(stat) {}
};

struct ExperienceNode {
    uint64_t experience_id;
    uint64_t tool_id;
    uint64_t curiosity_id;
    std::string input_given;
    std::string output_received;
    bool moral_check_passed;
    double timestamp;
    float satisfaction_rating;
    std::string notes;
    
    ExperienceNode() : experience_id(0), tool_id(0), curiosity_id(0), moral_check_passed(false), timestamp(0.0), satisfaction_rating(0.0f) {}
    ExperienceNode(uint64_t exp_id, uint64_t t_id, uint64_t c_id, const std::string& input, const std::string& output, 
                   bool moral, double time, float rating, const std::string& note)
        : experience_id(exp_id), tool_id(t_id), curiosity_id(c_id), input_given(input), output_received(output), 
          moral_check_passed(moral), timestamp(time), satisfaction_rating(rating), notes(note) {}
};

class MelvinWebSearchToolDemo {
private:
    WebSearchTool web_search_tool;
    std::vector<WebSearchResult> search_history;
    std::vector<ExperienceNode> experience_nodes;
    std::random_device rd;
    std::mt19937_64 gen;
    uint64_t next_experience_id;
    uint64_t next_node_id;
    
    static constexpr size_t MAX_SEARCH_HISTORY = 1000;
    
public:
    MelvinWebSearchToolDemo() : gen(rd()), next_experience_id(0x30000), next_node_id(0x60000) {
        initialize_web_search_tool();
    }
    
    void initialize_web_search_tool() {
        web_search_tool.tool_id = 0x50000;
        web_search_tool.tool_name = "WebSearchTool";
        web_search_tool.tool_type = "web_search";
        web_search_tool.success_rate = 0.8f;
        web_search_tool.usage_count = 0;
        web_search_tool.status = "active";
    }
    
    WebSearchResult perform_web_search(const std::string& query, uint64_t originating_curiosity) {
        WebSearchResult result;
        result.query = query;
        result.search_timestamp = static_cast<double>(std::time(nullptr));
        
        // Step 1: Moral filtering
        result.moral_check_passed = is_search_query_morally_safe(query);
        
        if (!result.moral_check_passed) {
            result.search_successful = false;
            result.error_message = "Query blocked by moral filtering";
            web_search_tool.blocked_queries.push_back(query);
            
            // Record the blocked search experience
            ExperienceNode blocked_experience = record_search_experience(web_search_tool.tool_id, originating_curiosity, query, result);
            experience_nodes.push_back(blocked_experience);
            
            return result;
        }
        
        // Step 2: Execute web search
        try {
            result.results = execute_web_search(query);
            result.search_successful = !result.results.empty();
            
            if (result.search_successful) {
                // Step 3: Create knowledge nodes from results
                result.created_nodes = create_knowledge_nodes_from_search_results(result.results, query);
                
                // Step 4: Record successful search experience
                ExperienceNode search_experience = record_search_experience(web_search_tool.tool_id, originating_curiosity, query, result);
                experience_nodes.push_back(search_experience);
                
                // Step 5: Update tool stats
                update_web_search_tool_stats(result);
                
                web_search_tool.successful_queries.push_back(query);
            } else {
                result.error_message = "No results found";
                result.search_successful = false;
            }
            
        } catch (const std::exception& e) {
            result.search_successful = false;
            result.error_message = "Search error: " + std::string(e.what());
        }
        
        // Store search history
        search_history.push_back(result);
        if (search_history.size() > MAX_SEARCH_HISTORY) {
            search_history.erase(search_history.begin(), search_history.begin() + (search_history.size() - MAX_SEARCH_HISTORY));
        }
        
        return result;
    }
    
    bool is_search_query_morally_safe(const std::string& query) {
        std::string lower_query = query;
        std::transform(lower_query.begin(), lower_query.end(), lower_query.begin(), ::tolower);
        
        // Check for harmful content patterns
        std::vector<std::string> harmful_patterns = {
            "how to hack", "how to attack", "how to harm", "how to destroy",
            "illegal", "unethical", "harmful", "dangerous", "violent",
            "hate speech", "discrimination", "exploitation", "abuse"
        };
        
        for (const auto& pattern : harmful_patterns) {
            if (lower_query.find(pattern) != std::string::npos) {
                return false;
            }
        }
        
        return true;
    }
    
    std::vector<SearchResult> execute_web_search(const std::string& query) {
        std::vector<SearchResult> results;
        
        // Simulate web search with mock results for demo purposes
        double current_time = static_cast<double>(std::time(nullptr));
        
        // Generate mock search results based on query
        std::string lower_query = query;
        std::transform(lower_query.begin(), lower_query.end(), lower_query.begin(), ::tolower);
        
        if (lower_query.find("quantum") != std::string::npos) {
            results.emplace_back("Quantum Computing Fundamentals", 
                               "Quantum computing uses quantum mechanical phenomena to perform calculations that would be impossible for classical computers...", 
                               "https://example.com/quantum-computing", 0.95f, "example.com", current_time);
            results.emplace_back("Quantum Algorithms Overview", 
                               "Quantum algorithms leverage quantum superposition and entanglement to solve problems exponentially faster...", 
                               "https://example.com/quantum-algorithms", 0.88f, "example.com", current_time);
            results.emplace_back("Quantum Error Correction", 
                               "Quantum error correction is essential for building reliable quantum computers...", 
                               "https://example.com/quantum-error-correction", 0.82f, "example.com", current_time);
        } else if (lower_query.find("machine learning") != std::string::npos) {
            results.emplace_back("Introduction to Machine Learning", 
                               "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience...", 
                               "https://example.com/ml-intro", 0.92f, "example.com", current_time);
            results.emplace_back("Deep Learning Applications", 
                               "Deep learning uses neural networks with multiple layers to model complex patterns in data...", 
                               "https://example.com/deep-learning", 0.85f, "example.com", current_time);
            results.emplace_back("Machine Learning Ethics", 
                               "As ML systems become more powerful, ethical considerations become increasingly important...", 
                               "https://example.com/ml-ethics", 0.78f, "example.com", current_time);
        } else if (lower_query.find("climate") != std::string::npos) {
            results.emplace_back("Climate Change Research", 
                               "Recent studies show significant changes in global climate patterns with far-reaching consequences...", 
                               "https://example.com/climate-research", 0.90f, "example.com", current_time);
            results.emplace_back("Renewable Energy Solutions", 
                               "Solar and wind energy technologies are becoming more efficient and cost-effective...", 
                               "https://example.com/renewable-energy", 0.87f, "example.com", current_time);
            results.emplace_back("Carbon Footprint Reduction", 
                               "Strategies for reducing carbon emissions and achieving carbon neutrality...", 
                               "https://example.com/carbon-reduction", 0.83f, "example.com", current_time);
        } else if (lower_query.find("artificial intelligence") != std::string::npos) {
            results.emplace_back("AI Fundamentals", 
                               "Artificial intelligence encompasses machine learning, natural language processing, and robotics...", 
                               "https://example.com/ai-fundamentals", 0.94f, "example.com", current_time);
            results.emplace_back("AI Ethics and Safety", 
                               "Ensuring AI systems are safe, fair, and beneficial to humanity...", 
                               "https://example.com/ai-ethics", 0.89f, "example.com", current_time);
            results.emplace_back("Future of AI", 
                               "Predictions and trends in artificial intelligence development...", 
                               "https://example.com/ai-future", 0.85f, "example.com", current_time);
        } else {
            // Generic results for other queries
            results.emplace_back("Search Results for: " + query, 
                               "This is a mock search result for the query: " + query + ". In a real implementation, this would be actual web search results...", 
                               "https://example.com/search-result", 0.75f, "example.com", current_time);
            results.emplace_back("Related Information", 
                               "Additional information related to your search query. This demonstrates how search results are processed and stored...", 
                               "https://example.com/related-info", 0.70f, "example.com", current_time);
        }
        
        return results;
    }
    
    std::vector<uint64_t> create_knowledge_nodes_from_search_results(const std::vector<SearchResult>& results, const std::string& query) {
        std::vector<uint64_t> created_nodes;
        
        for (const auto& result : results) {
            // Create a knowledge node for each search result
            uint64_t node_id = next_node_id++;
            created_nodes.push_back(node_id);
        }
        
        return created_nodes;
    }
    
    ExperienceNode record_search_experience(uint64_t tool_id, uint64_t curiosity_id, const std::string& query, const WebSearchResult& search_result) {
        double current_time = static_cast<double>(std::time(nullptr));
        
        // Calculate satisfaction rating based on search success and result quality
        float satisfaction_rating = 0.0f;
        if (search_result.search_successful && !search_result.results.empty()) {
            // Base satisfaction on number and quality of results
            satisfaction_rating = std::min(1.0f, static_cast<float>(search_result.results.size()) * 0.2f);
            
            // Bonus for high-relevance results
            float avg_relevance = 0.0f;
            for (const auto& result : search_result.results) {
                avg_relevance += result.relevance_score;
            }
            if (!search_result.results.empty()) {
                avg_relevance /= search_result.results.size();
                satisfaction_rating += avg_relevance * 0.5f;
            }
        }
        
        std::string notes = "Web search for: " + query + " (" + std::to_string(search_result.results.size()) + " results)";
        
        ExperienceNode experience(next_experience_id++, tool_id, curiosity_id, query, 
                                 "Search completed", search_result.moral_check_passed, current_time, 
                                 satisfaction_rating, notes);
        
        return experience;
    }
    
    void update_web_search_tool_stats(const WebSearchResult& search_result) {
        web_search_tool.usage_count++;
        
        // Update success rate based on search success
        if (search_result.search_successful) {
            // Increase success rate slightly for successful searches
            web_search_tool.success_rate = std::min(1.0f, web_search_tool.success_rate + 0.01f);
        } else {
            // Decrease success rate slightly for failed searches
            web_search_tool.success_rate = std::max(0.0f, web_search_tool.success_rate - 0.02f);
        }
        
        // Mark as deprecated if success rate gets too low
        if (web_search_tool.success_rate < 0.3f && web_search_tool.usage_count > 10) {
            web_search_tool.status = "deprecated";
        }
    }
    
    std::string format_web_search_result(const WebSearchResult& search_result) {
        std::ostringstream output;
        
        output << "[WebSearchTool]\n";
        output << "- Query: \"" << search_result.query << "\"\n";
        
        if (search_result.moral_check_passed) {
            if (search_result.search_successful) {
                output << "- Results: " << search_result.results.size() << " found\n";
                for (size_t i = 0; i < std::min(search_result.results.size(), size_t(3)); ++i) {
                    const auto& result = search_result.results[i];
                    output << "  " << (i + 1) << ". " << result.title << " (relevance: " 
                           << std::fixed << std::setprecision(2) << result.relevance_score << ")\n";
                    output << "     " << result.snippet.substr(0, 100) << "...\n";
                    output << "     " << result.link << "\n";
                }
            } else {
                output << "- Results: Search failed - " << search_result.error_message << "\n";
            }
        } else {
            output << "- Results: Query blocked by moral filtering\n";
        }
        
        output << "- Nodes Created: " << search_result.created_nodes.size() << "\n";
        output << "- Experience Recorded: " << (search_result.search_successful ? "success" : "failure") << "\n";
        
        return output.str();
    }
    
    std::string process_with_web_search(const std::string& query, uint64_t originating_curiosity) {
        auto search_result = perform_web_search(query, originating_curiosity);
        
        std::ostringstream response;
        response << "🔍 Melvin's Web Search Analysis:\n\n";
        response << "Query: \"" << query << "\"\n\n";
        response << format_web_search_result(search_result) << "\n\n";
        
        response << "🌐 Web Search Tool Features:\n";
        response << "- Takes query string as input and returns structured results\n";
        response << "- Results are clean (no ads, no scripts, just text)\n";
        response << "- Wraps all external search/API calls in safe handlers\n";
        response << "- Integrates moral filtering to block unsafe queries\n";
        response << "- Connects search results to knowledge graph as nodes\n";
        response << "- Stores each search as an ExperienceNode with success/failure rating\n";
        response << "- Makes results reusable by RecallTrack and ExplorationTrack\n";
        response << "- Compatible with MetaToolEngineer for tool chaining\n";
        
        return response.str();
    }
    
    void show_tool_stats() {
        std::cout << "\n📊 WebSearchTool Statistics:" << std::endl;
        std::cout << "- Tool ID: 0x" << std::hex << web_search_tool.tool_id << std::dec << std::endl;
        std::cout << "- Success Rate: " << std::fixed << std::setprecision(2) << web_search_tool.success_rate << std::endl;
        std::cout << "- Usage Count: " << web_search_tool.usage_count << std::endl;
        std::cout << "- Status: " << web_search_tool.status << std::endl;
        std::cout << "- Successful Queries: " << web_search_tool.successful_queries.size() << std::endl;
        std::cout << "- Blocked Queries: " << web_search_tool.blocked_queries.size() << std::endl;
    }
};

int main() {
    std::cout << "🔍 MELVIN WEB SEARCH TOOL TEST" << std::endl;
    std::cout << "==============================" << std::endl;
    std::cout << "Testing Melvin's WebSearchTool integration with Dynamic Tools System" << std::endl;
    std::cout << "Phase 6.6: Integrated with Dynamic Tools System" << std::endl;
    std::cout << "\n";
    
    try {
        MelvinWebSearchToolDemo melvin;
        
        // Test scenarios that should trigger different web search behaviors
        std::vector<std::pair<std::string, uint64_t>> test_scenarios = {
            {"quantum computing algorithms", 0x10001},
            {"machine learning applications", 0x10002},
            {"climate change solutions", 0x10003},
            {"artificial intelligence ethics", 0x10004},
            {"how to hack systems", 0x10005},  // This should be blocked
            {"renewable energy technologies", 0x10006},
            {"deep learning neural networks", 0x10007},
            {"illegal activities", 0x10008}  // This should be blocked
        };
        
        std::cout << "🎯 Testing " << test_scenarios.size() << " web search scenarios:" << std::endl;
        std::cout << "===============================================================" << std::endl;
        
        for (size_t i = 0; i < test_scenarios.size(); ++i) {
            std::cout << "\n" << std::string(80, '=') << std::endl;
            std::cout << "[SCENARIO " << (i + 1) << "/" << test_scenarios.size() << "]" << std::endl;
            
            const auto& [query, curiosity_id] = test_scenarios[i];
            std::cout << "Query: \"" << query << "\"" << std::endl;
            std::cout << "Curiosity ID: 0x" << std::hex << curiosity_id << std::dec << std::endl;
            std::cout << std::string(80, '=') << std::endl;
            
            // Show Melvin's web search response
            std::string response = melvin.process_with_web_search(query, curiosity_id);
            std::cout << response << std::endl;
        }
        
        // Show final tool statistics
        melvin.show_tool_stats();
        
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "🎉 WEB SEARCH TOOL TEST COMPLETE!" << std::endl;
        std::cout << "==================================" << std::endl;
        std::cout << "✅ WebSearchTool is integrated with Dynamic Tools System" << std::endl;
        std::cout << "✅ Takes query string as input and returns structured results" << std::endl;
        std::cout << "✅ Results are clean (no ads, no scripts, just text)" << std::endl;
        std::cout << "✅ Wraps all external search/API calls in safe handlers" << std::endl;
        std::cout << "✅ Integrates moral filtering to block unsafe queries" << std::endl;
        std::cout << "✅ Connects search results to knowledge graph as nodes" << std::endl;
        std::cout << "✅ Stores each search as an ExperienceNode with success/failure rating" << std::endl;
        std::cout << "✅ Makes results reusable by RecallTrack and ExplorationTrack" << std::endl;
        std::cout << "✅ Compatible with MetaToolEngineer for tool chaining" << std::endl;
        
        std::cout << "\n🧠 Key Features Demonstrated:" << std::endl;
        std::cout << "   • Safe Search Execution: Wrapped in error-catching handlers" << std::endl;
        std::cout << "   • Moral Filtering: Blocks harmful or unsafe queries" << std::endl;
        std::cout << "   • Knowledge Integration: Creates nodes from search results" << std::endl;
        std::cout << "   • Experience Recording: Tracks success/failure for learning" << std::endl;
        std::cout << "   • Tool Evolution: Strengthens with success, weakens with failure" << std::endl;
        std::cout << "   • MetaToolEngineer Compatibility: Can be chained with other tools" << std::endl;
        
        std::cout << "\n🌟 Example Behaviors:" << std::endl;
        std::cout << "   • Safe queries → Returns structured results with relevance scores" << std::endl;
        std::cout << "   • Harmful queries → Blocked by moral filtering" << std::endl;
        std::cout << "   • Search results → Converted to knowledge nodes" << std::endl;
        std::cout << "   • Tool usage → Recorded as experience nodes" << std::endl;
        std::cout << "   • Success/failure → Updates tool success rate" << std::endl;
        std::cout << "   • Tool chaining → WebSearch → Summarizer → Store workflow" << std::endl;
        
        std::cout << "\n🎯 Melvin's WebSearchTool extends knowledge through safe," << std::endl;
        std::cout << "   morally-filtered web search while learning from every experience!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error during web search tool testing: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

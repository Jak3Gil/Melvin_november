/*
 * Melvin Test Growth System - Small scale test
 * Runs 100 cycles instead of 10,000 for quick testing
 */

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <set>
#include <sstream>
#include <fstream>
#include <chrono>
#include <thread>
#include <random>
#include <cmath>
#include <iomanip>

// Simplified structures for testing
struct TestConcept {
    std::string concept;
    std::string definition;
    double activation;
    uint32_t access_count;
    uint32_t validation_successes;
    uint32_t validation_failures;
    uint64_t first_seen;
    
    TestConcept() : activation(1.0), access_count(0), validation_successes(0), 
                   validation_failures(0), first_seen(0) {}
    
    TestConcept(const std::string& c, const std::string& d = "") 
        : concept(c), definition(d), activation(1.0), access_count(0),
          validation_successes(0), validation_failures(0), first_seen(0) {}
};

struct TestMetrics {
    uint64_t cycle_id;
    std::string input_type;
    std::string input_content;
    double overall_confidence;
    uint64_t timestamp;
    uint32_t concepts_learned;
    std::string dominant_driver;
    
    TestMetrics() : cycle_id(0), overall_confidence(0.0), timestamp(0), concepts_learned(0) {}
};

// Test Growth System
class TestGrowthSystem {
private:
    std::unordered_map<std::string, TestConcept> concepts;
    std::vector<TestMetrics> metrics_log;
    
    // Driver system
    double dopamine = 0.5;
    double serotonin = 0.5;
    double endorphins = 0.5;
    
    uint64_t current_cycle;
    uint64_t total_cycles_target;
    
    // Random number generation
    std::random_device rd;
    std::mt19937 gen;
    
public:
    TestGrowthSystem(uint64_t target_cycles = 100) 
        : current_cycle(0), total_cycles_target(target_cycles) {
        
        gen.seed(rd());
        
        std::cout << "🧪 MELVIN TEST GROWTH SYSTEM" << std::endl;
        std::cout << "============================" << std::endl;
        std::cout << "🎯 Target Cycles: " << total_cycles_target << std::endl;
        std::cout << std::endl;
    }
    
    void runTestCampaign() {
        std::cout << "🚀 STARTING TEST CAMPAIGN" << std::endl;
        std::cout << "=========================" << std::endl;
        std::cout << std::endl;
        
        for (current_cycle = 1; current_cycle <= total_cycles_target; current_cycle++) {
            if (current_cycle % 20 == 0) {
                std::cout << "🔄 Cycle " << current_cycle << "/" << total_cycles_target 
                          << " (" << (current_cycle * 100 / total_cycles_target) << "%)" << std::endl;
            }
            
            runSingleCycle();
        }
        
        generateTestReport();
        std::cout << "✅ Test campaign completed!" << std::endl;
    }
    
    void runSingleCycle() {
        TestMetrics metrics;
        metrics.cycle_id = current_cycle;
        metrics.timestamp = getCurrentTimestamp();
        
        // Generate input
        auto input = generateInput();
        metrics.input_type = input.first;
        metrics.input_content = input.second;
        
        // Extract concepts
        std::vector<std::string> input_concepts = extractConcepts(input.second);
        
        // Add concepts
        for (const std::string& concept : input_concepts) {
            if (concepts.find(concept) == concepts.end()) {
                concepts[concept] = TestConcept(concept, "Learned from input");
                concepts[concept].first_seen = metrics.timestamp;
                metrics.concepts_learned++;
            }
            concepts[concept].access_count++;
        }
        
        // Simulate validation
        for (const std::string& concept : input_concepts) {
            auto it = concepts.find(concept);
            if (it != concepts.end()) {
                if (it->second.access_count > 2) {
                    it->second.validation_successes++;
                } else {
                    it->second.validation_failures++;
                }
            }
        }
        
        // Determine dominant driver
        metrics.dominant_driver = determineDriver();
        
        // Calculate confidence
        metrics.overall_confidence = calculateConfidence();
        
        // Update drivers
        updateDrivers(metrics);
        
        metrics_log.push_back(metrics);
    }
    
    std::pair<std::string, std::string> generateInput() {
        std::vector<std::string> inputs = {
            "A bird sitting on a wire",
            "natural selection",
            "A robot adapting to cold demonstrates survival of the fittest",
            "The sun shining through clouds",
            "evolution",
            "A cat walking across the street",
            "photosynthesis",
            "Water flowing down a river"
        };
        
        std::uniform_int_distribution<> dist(0, inputs.size() - 1);
        std::string input = inputs[dist(gen)];
        
        std::string type;
        if (input.find("A ") == 0) {
            type = "raw";
        } else if (input.find(" ") == std::string::npos) {
            type = "conceptual";
        } else {
            type = "hybrid";
        }
        
        return {type, input};
    }
    
    std::vector<std::string> extractConcepts(const std::string& text) {
        std::vector<std::string> concepts;
        std::stringstream ss(text);
        std::string word;
        
        while (ss >> word) {
            word.erase(std::remove_if(word.begin(), word.end(), 
                [](char c) { return !std::isalnum(c); }), word.end());
            
            if (word.length() > 2) {
                std::transform(word.begin(), word.end(), word.begin(), ::tolower);
                concepts.push_back(word);
            }
        }
        
        return concepts;
    }
    
    std::string determineDriver() {
        if (dopamine >= serotonin && dopamine >= endorphins) {
            return "dopamine";
        } else if (serotonin >= endorphins) {
            return "serotonin";
        } else {
            return "endorphins";
        }
    }
    
    double calculateConfidence() {
        if (concepts.empty()) return 0.0;
        
        double total_confidence = 0.0;
        for (const auto& concept_pair : concepts) {
            const TestConcept& concept = concept_pair.second;
            double confidence = (double)concept.validation_successes / 
                              (concept.validation_successes + concept.validation_failures + 1);
            total_confidence += confidence;
        }
        
        return total_confidence / concepts.size();
    }
    
    void updateDrivers(const TestMetrics& metrics) {
        if (metrics.concepts_learned > 0) {
            dopamine = std::min(1.0, dopamine + 0.01);
        }
        if (metrics.overall_confidence > 0.5) {
            serotonin = std::min(1.0, serotonin + 0.01);
            endorphins = std::min(1.0, endorphins + 0.01);
        }
        
        // Gradual decay
        dopamine *= 0.999;
        serotonin *= 0.999;
        endorphins *= 0.999;
    }
    
    void generateTestReport() {
        std::ofstream file("melvin_test_report.txt");
        if (file.is_open()) {
            file << "MELVIN TEST GROWTH REPORT" << std::endl;
            file << "=========================" << std::endl;
            file << "Total Cycles: " << total_cycles_target << std::endl;
            file << "Total Concepts: " << concepts.size() << std::endl;
            
            double avg_confidence = 0.0;
            if (!metrics_log.empty()) {
                for (const TestMetrics& metrics : metrics_log) {
                    avg_confidence += metrics.overall_confidence;
                }
                avg_confidence /= metrics_log.size();
            }
            
            file << "Average Confidence: " << std::fixed << std::setprecision(3) << avg_confidence << std::endl;
            file << "Final Driver Levels - D:" << std::fixed << std::setprecision(2) 
                 << dopamine << " S:" << serotonin << " E:" << endorphins << std::endl;
            
            file.close();
            std::cout << "📈 Test report generated: melvin_test_report.txt" << std::endl;
        }
        
        // Display summary
        std::cout << std::endl;
        std::cout << "📊 TEST SUMMARY" << std::endl;
        std::cout << "===============" << std::endl;
        std::cout << "Total Cycles: " << total_cycles_target << std::endl;
        std::cout << "Total Concepts: " << concepts.size() << std::endl;
        std::cout << "Average Confidence: " << std::fixed << std::setprecision(3) 
                  << calculateConfidence() << std::endl;
        std::cout << "Driver Levels - D:" << std::fixed << std::setprecision(2) 
                  << dopamine << " S:" << serotonin << " E:" << endorphins << std::endl;
        std::cout << std::endl;
    }
    
private:
    uint64_t getCurrentTimestamp() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
};

int main() {
    std::cout << "🚀 Starting Melvin Test Growth System" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << std::endl;
    
    TestGrowthSystem melvin(100); // 100 cycles for testing
    melvin.runTestCampaign();
    
    std::cout << "🎯 Melvin Test Growth System finished!" << std::endl;
    
    return 0;
}

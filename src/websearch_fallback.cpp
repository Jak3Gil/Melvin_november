#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <string>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iomanip>

using json = nlohmann::json;

// HTTP response callback for libcurl
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* s) {
    size_t newLength = size * nmemb;
    try {
        s->append((char*)contents, newLength);
        return newLength;
    } catch (std::bad_alloc& e) {
        return 0;
    }
}

// URL encode function
std::string url_encode(const std::string& str) {
    std::ostringstream escaped;
    escaped.fill('0');
    escaped << std::hex;

    for (char c : str) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            escaped << c;
        } else {
            escaped << std::uppercase;
            escaped << '%' << std::setw(2) << int(static_cast<unsigned char>(c));
            escaped << std::nouppercase;
        }
    }

    return escaped.str();
}

std::string perform_web_search(const std::string& query) {
    // Check for Bing API key first
    const char* bing_api_key = std::getenv("BING_API_KEY");
    
    if (bing_api_key && strlen(bing_api_key) > 0) {
        // Use Bing API
        CURL* curl = curl_easy_init();
        if (!curl) {
            return "";
        }

        std::string readBuffer;
        
        // Construct Bing Search API URL
        std::string encoded_query = url_encode(query);
        std::string url = "https://api.bing.microsoft.com/v7.0/search?q=" + encoded_query + "&count=3";

        // Set up curl options
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
        
        // Set headers
        struct curl_slist* headers = nullptr;
        std::string auth_header = "Ocp-Apim-Subscription-Key: " + std::string(bing_api_key);
        headers = curl_slist_append(headers, auth_header.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        // Perform the request
        CURLcode res = curl_easy_perform(curl);
        
        // Clean up
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

        if (res != CURLE_OK) {
            return "";
        }

        try {
            // Parse JSON response
            json response = json::parse(readBuffer);
            
            if (!response.contains("webPages") || !response["webPages"].contains("value")) {
                return "";
            }

            auto results = response["webPages"]["value"];
            std::ostringstream summary;
            
            summary << "Research findings for \"" << query << "\": ";
            
            for (size_t i = 0; i < std::min(results.size(), size_t(3)); ++i) {
                auto result = results[i];
                std::string title = result.value("name", "No title");
                std::string snippet = result.value("snippet", "No description");
                
                summary << "\n" << (i + 1) << ". " << title << " - " << snippet;
            }
            
            return summary.str();
            
        } catch (const json::exception& e) {
            return "";
        }
    } else {
        // Fallback to DuckDuckGo Instant Answer API
        CURL* curl = curl_easy_init();
        if (!curl) {
            return "";
        }

        std::string readBuffer;
        
        // Construct DuckDuckGo API URL
        std::string encoded_query = url_encode(query);
        std::string url = "https://api.duckduckgo.com/?q=" + encoded_query + "&format=json&no_html=1";

        // Set up curl options
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "Melvin-Unified-Brain/1.0");

        // Perform the request
        CURLcode res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);

        if (res != CURLE_OK) {
            return "";
        }

        try {
            // Parse JSON response
            json response = json::parse(readBuffer);
            
            std::ostringstream summary;
            summary << "Research findings for \"" << query << "\": ";
            
            // Check for AbstractText first
            if (response.contains("AbstractText") && !response["AbstractText"].is_null()) {
                std::string abstract = response["AbstractText"];
                if (abstract.length() > 500) {
                    abstract = abstract.substr(0, 500) + "...";
                }
                summary << "\n" << abstract;
                return summary.str();
            }
            
            // Check for RelatedTopics
            if (response.contains("RelatedTopics") && response["RelatedTopics"].is_array()) {
                auto topics = response["RelatedTopics"];
                for (size_t i = 0; i < std::min(topics.size(), size_t(3)); ++i) {
                    if (topics[i].contains("Text")) {
                        std::string text = topics[i]["Text"];
                        if (text.length() > 200) {
                            text = text.substr(0, 200) + "...";
                        }
                        summary << "\n" << (i + 1) << ". " << text;
                    }
                }
                return summary.str();
            }
            
            // Check for Definition
            if (response.contains("Definition") && !response["Definition"].is_null()) {
                std::string definition = response["Definition"];
                if (definition.length() > 500) {
                    definition = definition.substr(0, 500) + "...";
                }
                summary << "\n" << definition;
                return summary.str();
            }
            
            return "";
            
        } catch (const json::exception& e) {
            return "";
        }
    }
}
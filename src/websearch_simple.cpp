#include <curl/curl.h>
#include <string>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iomanip>

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

        // Simple JSON parsing for Bing results
        if (readBuffer.find("\"webPages\"") != std::string::npos) {
            std::ostringstream summary;
            summary << "Research findings for \"" << query << "\": ";
            
            // Extract titles and snippets (simplified parsing)
            size_t pos = 0;
            int count = 0;
            while (pos != std::string::npos && count < 3) {
                size_t name_start = readBuffer.find("\"name\":\"", pos);
                if (name_start == std::string::npos) break;
                
                name_start += 8; // Skip "\"name\":\""
                size_t name_end = readBuffer.find("\"", name_start);
                if (name_end == std::string::npos) break;
                
                std::string title = readBuffer.substr(name_start, name_end - name_start);
                
                size_t snippet_start = readBuffer.find("\"snippet\":\"", name_end);
                if (snippet_start == std::string::npos) break;
                
                snippet_start += 11; // Skip "\"snippet\":\""
                size_t snippet_end = readBuffer.find("\"", snippet_start);
                if (snippet_end == std::string::npos) break;
                
                std::string snippet = readBuffer.substr(snippet_start, snippet_end - snippet_start);
                
                summary << "\n" << (count + 1) << ". " << title << " - " << snippet;
                
                pos = snippet_end;
                count++;
            }
            
            return summary.str();
        }
        
        return "";
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

        // Simple JSON parsing for DuckDuckGo results
        std::ostringstream summary;
        summary << "Research findings for \"" << query << "\": ";
        
        // Check for AbstractText first
        size_t abstract_start = readBuffer.find("\"AbstractText\":\"");
        if (abstract_start != std::string::npos) {
            abstract_start += 16; // Skip "\"AbstractText\":\""
            size_t abstract_end = readBuffer.find("\"", abstract_start);
            if (abstract_end != std::string::npos) {
                std::string abstract = readBuffer.substr(abstract_start, abstract_end - abstract_start);
                if (abstract.length() > 500) {
                    abstract = abstract.substr(0, 500) + "...";
                }
                summary << "\n" << abstract;
                return summary.str();
            }
        }
        
        // Check for Definition
        size_t def_start = readBuffer.find("\"Definition\":\"");
        if (def_start != std::string::npos) {
            def_start += 15; // Skip "\"Definition\":\""
            size_t def_end = readBuffer.find("\"", def_start);
            if (def_end != std::string::npos) {
                std::string definition = readBuffer.substr(def_start, def_end - def_start);
                if (definition.length() > 500) {
                    definition = definition.substr(0, 500) + "...";
                }
                summary << "\n" << definition;
                return summary.str();
            }
        }
        
        return "";
    }
}

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

// Wikipedia search function
std::string search_wikipedia(const std::string& query) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        return "";
    }

    std::string readBuffer;
    
    // Construct Wikipedia API URL
    std::string encoded_query = url_encode(query);
    std::string url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + encoded_query;

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

    // Simple JSON parsing for Wikipedia results
    std::ostringstream summary;
    summary << "Wikipedia: ";
    
    // Extract title
    size_t title_start = readBuffer.find("\"title\":\"");
    if (title_start != std::string::npos) {
        title_start += 9; // Skip "\"title\":\""
        size_t title_end = readBuffer.find("\"", title_start);
        if (title_end != std::string::npos) {
            std::string title = readBuffer.substr(title_start, title_end - title_start);
            summary << title << " - ";
        }
    }
    
    // Extract extract (summary)
    size_t extract_start = readBuffer.find("\"extract\":\"");
    if (extract_start != std::string::npos) {
        extract_start += 11; // Skip "\"extract\":\""
        size_t extract_end = readBuffer.find("\"", extract_start);
        if (extract_end != std::string::npos) {
            std::string extract = readBuffer.substr(extract_start, extract_end - extract_start);
            if (extract.length() > 500) {
                extract = extract.substr(0, 500) + "...";
            }
            summary << extract;
            return summary.str();
        }
    }
    
    return "";
}

// DuckDuckGo search function
std::string search_duckduckgo(const std::string& query) {
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
    summary << "DuckDuckGo: ";
    
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
            summary << abstract;
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
            summary << definition;
            return summary.str();
        }
    }
    
    return "";
}

// Main web search function - tries Wikipedia first, then DuckDuckGo
std::string perform_web_search(const std::string& query) {
    // Try Wikipedia first for factual information
    std::string wiki_result = search_wikipedia(query);
    if (!wiki_result.empty()) {
        return wiki_result;
    }
    
    // Fallback to DuckDuckGo
    std::string ddg_result = search_duckduckgo(query);
    if (!ddg_result.empty()) {
        return ddg_result;
    }
    
    return "";
}

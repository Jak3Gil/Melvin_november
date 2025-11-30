#include <iostream>
#include <curl/curl.h>
#include <json/json.h>

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

int main() {
    std::cout << "🔍 Testing Simple Ollama Connection" << std::endl;
    
    CURL* curl;
    CURLcode res;
    std::string response_data;
    
    curl = curl_easy_init();
    if (curl) {
        // Prepare JSON payload
        Json::Value payload;
        payload["model"] = "llama3.2";
        payload["prompt"] = "What is a cat?";
        payload["stream"] = false;
        
        Json::StreamWriterBuilder builder;
        std::string json_payload = Json::writeString(builder, payload);
        
        std::cout << "📤 Sending JSON: " << json_payload << std::endl;
        
        // Set up CURL
        curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:11434/api/generate");
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_payload.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, json_payload.length());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, curl_slist_append(nullptr, "Content-Type: application/json"));
        
        // Perform request
        res = curl_easy_perform(curl);
        
        // Get response code
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        
        std::cout << "📊 Response Code: " << http_code << std::endl;
        std::cout << "📊 CURL Result: " << curl_easy_strerror(res) << std::endl;
        std::cout << "📊 Response Data: " << response_data << std::endl;
        
        curl_easy_cleanup(curl);
    }
    
    return 0;
}

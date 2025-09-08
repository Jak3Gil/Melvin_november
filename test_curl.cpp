#include <iostream>
#include <string>
#include <curl/curl.h>

size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* s) {
    size_t newLength = size * nmemb;
    try {
        s->append((char*)contents, newLength);
        return newLength;
    } catch (std::bad_alloc& e) {
        return 0;
    }
}

int main() {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Failed to initialize curl" << std::endl;
        return 1;
    }
    
    std::string readBuffer;
    std::string url = "https://api.duckduckgo.com/?q=world+hunger&format=json&no_html=1&skip_disambig=1";
    
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "Melvin-Brain/1.0");
    
    CURLcode res = curl_easy_perform(curl);
    
    if (res != CURLE_OK) {
        std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
    } else {
        std::cout << "Success! Response length: " << readBuffer.length() << " bytes" << std::endl;
        if (readBuffer.length() < 1000) {
            std::cout << "Response: " << readBuffer << std::endl;
        }
    }
    
    curl_easy_cleanup(curl);
    curl_global_cleanup();
    
    return 0;
}

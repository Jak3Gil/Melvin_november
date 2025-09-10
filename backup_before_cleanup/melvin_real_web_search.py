#!/usr/bin/env python3
"""
🌐 MELVIN REAL WEB SEARCH INTEGRATION
=====================================
Enables Melvin to perform actual web searches and learn from real internet data.
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Optional
import logging

class RealWebSearchTool:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.search_history = []
        self.blocked_domains = ['adult', 'malware', 'phishing', 'spam']
        self.max_results = 5
        self.timeout = 10
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def is_query_safe(self, query: str) -> bool:
        """Check if query is morally safe"""
        query_lower = query.lower()
        harmful_keywords = [
            'harm', 'hurt', 'destroy', 'kill', 'violence', 'illegal', 
            'unethical', 'malware', 'hack', 'exploit', 'scam'
        ]
        
        for keyword in harmful_keywords:
            if keyword in query_lower:
                return False
        return True
    
    def search_duckduckgo(self, query: str) -> List[Dict]:
        """Search using DuckDuckGo (no API key required)"""
        try:
            # DuckDuckGo instant answer API
            url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
            
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Extract instant answer
            if data.get('Abstract'):
                results.append({
                    'title': data.get('Heading', 'Instant Answer'),
                    'snippet': data.get('Abstract', ''),
                    'link': data.get('AbstractURL', ''),
                    'source': 'DuckDuckGo Instant Answer',
                    'relevance_score': 0.9
                })
            
            # Extract related topics
            for topic in data.get('RelatedTopics', [])[:3]:
                if isinstance(topic, dict) and 'Text' in topic:
                    results.append({
                        'title': topic.get('FirstURL', '').split('/')[-1].replace('_', ' ').title(),
                        'snippet': topic.get('Text', ''),
                        'link': topic.get('FirstURL', ''),
                        'source': 'DuckDuckGo Related',
                        'relevance_score': 0.7
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"DuckDuckGo search failed: {e}")
            return []
    
    def search_wikipedia(self, query: str) -> List[Dict]:
        """Search Wikipedia for comprehensive information"""
        try:
            # Clean query for Wikipedia URL
            clean_query = query.lower().replace('?', '').replace('what is', '').replace('whats', '').strip()
            clean_query = clean_query.replace(' ', '_')
            
            # Wikipedia API search
            search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{clean_query}"
            
            response = self.session.get(search_url, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            if 'extract' in data:
                return [{
                    'title': data.get('title', query),
                    'snippet': data.get('extract', ''),
                    'link': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                    'source': 'Wikipedia',
                    'relevance_score': 0.95
                }]
            
            return []
            
        except Exception as e:
            self.logger.error(f"Wikipedia search failed: {e}")
            return []
    
    def search_news_api(self, query: str) -> List[Dict]:
        """Search news sources (requires API key)"""
        # This would require a News API key
        # For now, return empty list
        return []
    
    def scrape_webpage(self, url: str) -> Optional[Dict]:
        """Scrape content from a webpage"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No Title"
            
            # Extract main content
            content_selectors = [
                'main', 'article', '.content', '#content', 
                '.post-content', '.entry-content', 'p'
            ]
            
            content_text = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content_text = ' '.join([elem.get_text().strip() for elem in elements])
                    break
            
            # Clean up content
            content_text = re.sub(r'\s+', ' ', content_text)
            content_text = content_text[:1000]  # Limit length
            
            return {
                'title': title_text,
                'content': content_text,
                'url': url,
                'scraped_at': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Web scraping failed for {url}: {e}")
            return None
    
    def perform_search(self, query: str) -> Dict:
        """Perform comprehensive web search"""
        if not self.is_query_safe(query):
            return {
                'query': query,
                'success': False,
                'error': 'Query blocked by moral filtering',
                'results': []
            }
        
        self.logger.info(f"Searching for: {query}")
        
        all_results = []
        
        # Search multiple sources
        sources = [
            ('Wikipedia', self.search_wikipedia),
            ('DuckDuckGo', self.search_duckduckgo),
        ]
        
        for source_name, search_func in sources:
            try:
                results = search_func(query)
                for result in results:
                    result['search_source'] = source_name
                all_results.extend(results)
            except Exception as e:
                self.logger.error(f"{source_name} search failed: {e}")
        
        # Remove duplicates and sort by relevance
        unique_results = []
        seen_urls = set()
        
        for result in all_results:
            url = result.get('link', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        # Sort by relevance score
        unique_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Limit results
        final_results = unique_results[:self.max_results]
        
        # Record search
        search_record = {
            'query': query,
            'timestamp': time.time(),
            'results_count': len(final_results),
            'success': len(final_results) > 0
        }
        self.search_history.append(search_record)
        
        return {
            'query': query,
            'success': len(final_results) > 0,
            'results': final_results,
            'total_found': len(all_results),
            'sources_used': [source[0] for source in sources]
        }
    
    def learn_from_search(self, search_result: Dict) -> List[Dict]:
        """Convert search results into knowledge nodes for Melvin"""
        knowledge_nodes = []
        
        for result in search_result.get('results', []):
            # Create knowledge node
            node = {
                'content': f"{result.get('title', '')}: {result.get('snippet', '')}",
                'source': result.get('source', 'Unknown'),
                'url': result.get('link', ''),
                'relevance_score': result.get('relevance_score', 0.5),
                'query': search_result.get('query', ''),
                'timestamp': time.time(),
                'node_type': 'web_knowledge'
            }
            knowledge_nodes.append(node)
        
        return knowledge_nodes
    
    def get_search_stats(self) -> Dict:
        """Get statistics about search usage"""
        total_searches = len(self.search_history)
        successful_searches = sum(1 for s in self.search_history if s['success'])
        success_rate = (successful_searches / total_searches * 100) if total_searches > 0 else 0
        
        return {
            'total_searches': total_searches,
            'successful_searches': successful_searches,
            'success_rate': success_rate,
            'recent_searches': self.search_history[-5:] if self.search_history else []
        }

def test_real_web_search():
    """Test the real web search functionality"""
    print("🌐 Testing Real Web Search Tool")
    print("=" * 40)
    
    search_tool = RealWebSearchTool()
    
    # Test queries
    test_queries = [
        "what is cancer",
        "quantum computing",
        "artificial intelligence",
        "machine learning"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Searching: '{query}'")
        result = search_tool.perform_search(query)
        
        if result['success']:
            print(f"✅ Found {len(result['results'])} results")
            for i, res in enumerate(result['results'][:2], 1):
                print(f"  {i}. {res['title']}")
                print(f"     {res['snippet'][:100]}...")
                print(f"     Source: {res['source']}")
        else:
            print(f"❌ Search failed: {result.get('error', 'Unknown error')}")
        
        time.sleep(1)  # Be respectful to servers
    
    # Show stats
    stats = search_tool.get_search_stats()
    print(f"\n📊 Search Statistics:")
    print(f"Total searches: {stats['total_searches']}")
    print(f"Success rate: {stats['success_rate']:.1f}%")

if __name__ == "__main__":
    test_real_web_search()

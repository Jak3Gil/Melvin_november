#!/usr/bin/env python3
"""
🎯 MELVIN FIXED CHAT
Chat interface that properly prioritizes educational content
"""

import sqlite3
import re
from typing import List, Tuple

class MelvinFixedChat:
    def __init__(self):
        self.conn = sqlite3.connect("melvin_global_memory/global_memory.db")
        self.cursor = self.conn.cursor()
        
        # Get brain stats
        self.cursor.execute('SELECT COUNT(*) FROM nodes')
        self.total_nodes = self.cursor.fetchone()[0]
        
        self.cursor.execute('SELECT COUNT(*) FROM nodes WHERE modality_source = "curated_educational"')
        self.educational_nodes = self.cursor.fetchone()[0]
        
        print("🎯 Melvin Fixed Chat Interface")
        print(f"📊 Brain: {self.total_nodes:,} total nodes")
        print(f"📚 Educational: {self.educational_nodes:,} high-quality nodes")
        print("-" * 50)
    
    def educational_search(self, query: str, limit: int = 3) -> List[Tuple[str, str]]:
        """Search ONLY educational content first"""
        results = []
        query_words = [word.lower() for word in query.split() if len(word) > 2]
        
        # Search our curated educational content FIRST
        for word in query_words:
            self.cursor.execute("""
                SELECT node_type, content FROM nodes 
                WHERE modality_source = 'curated_educational'
                AND LOWER(content) LIKE ?
                ORDER BY 
                    CASE 
                        WHEN LOWER(content) LIKE ? THEN 1  -- Exact phrase match
                        WHEN node_type = 'atomic_fact' THEN 2  -- Atomic facts
                        WHEN node_type = 'concept' THEN 3  -- Concepts
                        ELSE 4
                    END,
                    LENGTH(content) ASC  -- Shorter content first
                LIMIT ?
            """, (f'%{word}%', f'%{query.lower()}%', limit))
            
            found = self.cursor.fetchall()
            results.extend(found)
            
            if found:  # If we found good educational content, use it
                break
        
        # Remove duplicates
        seen = set()
        unique_results = []
        for node_type, content in results:
            if content not in seen:
                seen.add(content)
                unique_results.append((node_type, content))
        
        return unique_results[:limit]
    
    def fallback_search(self, query: str, limit: int = 2) -> List[Tuple[str, str]]:
        """Fallback search in non-movie content"""
        query_words = [word.lower() for word in query.split() if len(word) > 2]
        
        for word in query_words:
            self.cursor.execute("""
                SELECT node_type, content FROM nodes 
                WHERE LOWER(content) LIKE ?
                AND content NOT LIKE '%movie%'
                AND content NOT LIKE '%film%'
                AND content NOT LIKE '%actor%'
                AND content NOT LIKE '%I Am Curious%'
                AND content NOT LIKE '%flick%'
                AND content NOT LIKE '%cinema%'
                AND (node_type = 'atomic_fact' OR node_type = 'concept' OR node_type = 'code')
                ORDER BY LENGTH(content) ASC
                LIMIT ?
            """, (f'%{word}%', limit))
            
            results = self.cursor.fetchall()
            if results:
                return results
        
        return []
    
    def generate_response(self, user_input: str) -> str:
        """Generate response with proper educational prioritization"""
        
        # First, try educational content
        educational_results = self.educational_search(user_input, limit=3)
        
        if educational_results:
            if len(educational_results) == 1:
                node_type, content = educational_results[0]
                return f"Here's what I know: {content}"
            else:
                response = "I found several relevant facts:\n"
                for i, (node_type, content) in enumerate(educational_results, 1):
                    response += f"   {i}. {content}\n"
                return response.strip()
        
        # Fallback to other non-movie content
        fallback_results = self.fallback_search(user_input, limit=2)
        
        if fallback_results:
            node_type, content = fallback_results[0]
            if len(content) > 150:
                content = content[:150] + "..."
            return f"Based on my knowledge: {content}"
        
        # Educational suggestions if nothing found
        suggestions = {
            'algorithm': 'algorithms and problem-solving',
            'programming': 'programming concepts and code',
            'computer': 'computer science and technology',
            'ai': 'artificial intelligence and machine learning',
            'data': 'data structures and databases',
            'network': 'networking and systems'
        }
        
        for keyword, topic in suggestions.items():
            if keyword in user_input.lower():
                educational_examples = self.educational_search(keyword, limit=1)
                if educational_examples:
                    return f"I have information about {topic}: {educational_examples[0][1]}"
        
        return f"I don't have specific information about '{user_input}'. Try asking about algorithms, programming, AI, computer science, or data structures!"
    
    def interactive_demo(self):
        """Run an interactive demo"""
        print("\n🎯 FIXED CHAT DEMO - Educational Responses Only")
        print("=" * 50)
        
        test_questions = [
            "What is machine learning?",
            "Tell me about algorithms", 
            "How do data structures work?",
            "What is programming?",
            "Explain artificial intelligence",
            "What are neural networks?",
            "How do computers work?",
            "What is recursion?"
        ]
        
        for question in test_questions:
            print(f"\n👤 User: {question}")
            response = self.generate_response(question)
            print(f"🤖 Melvin: {response}")
            print("-" * 40)
    
    def chat(self):
        """Interactive chat loop"""
        print("\n💬 Ask me about programming, AI, algorithms, or computer science!")
        print("Type 'demo' for automated demo, 'quit' to exit\n")
        
        while True:
            try:
                user_input = input("💭 You: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("🤖 Melvin: Goodbye! Keep learning!")
                    break
                
                if user_input.lower() == 'demo':
                    self.interactive_demo()
                    continue
                
                # Generate response
                response = self.generate_response(user_input)
                print(f"🤖 Melvin: {response}")
                
            except KeyboardInterrupt:
                print("\n🤖 Melvin: Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"🤖 Melvin: Sorry, I had an error: {e}")
    
    def close(self):
        if hasattr(self, 'conn'):
            self.conn.close()

def main():
    print("🎯 MELVIN FIXED CHAT - EDUCATIONAL FOCUS")
    print("=" * 50)
    
    chat = MelvinFixedChat()
    try:
        # Run demo first to show it works
        chat.interactive_demo()
        
        # Then offer interactive chat
        print(f"\n🎉 Demo complete! Melvin is now giving relevant educational responses!")
        print(f"Would you like to try interactive chat? (y/n)")
        
        choice = input().strip().lower()
        if choice == 'y':
            chat.chat()
            
    finally:
        chat.close()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
🧠 MELVIN SMART CHAT
Improved chat that prioritizes educational content over movie reviews
"""

import sqlite3
import random
from typing import List, Tuple, Dict

class MelvinSmartChat:
    def __init__(self):
        self.conn = sqlite3.connect("melvin_global_memory/global_memory.db")
        self.cursor = self.conn.cursor()
        
        # Get brain stats
        self.cursor.execute('SELECT COUNT(*) FROM nodes')
        self.total_nodes = self.cursor.fetchone()[0]
        
        print("🧠 Melvin Smart Chat Interface")
        print(f"📊 Brain: {self.total_nodes:,} total nodes")
        print("🎯 Prioritizing educational and technical content")
        print("-" * 50)
    
    def smart_search(self, query: str, limit: int = 5) -> List[Tuple[str, str, str]]:
        """Smart search that prioritizes educational content"""
        query_words = query.lower().split()
        
        # Priority order for node types (educational first)
        priority_types = [
            'atomic_fact',
            'concept', 
            'code',
            'language'
        ]
        
        all_results = []
        
        # Search each type in priority order
        for node_type in priority_types:
            for word in query_words:
                if len(word) > 2:
                    self.cursor.execute("""
                        SELECT node_id, node_type, content FROM nodes 
                        WHERE node_type = ? 
                        AND LOWER(content) LIKE ? 
                        AND content NOT LIKE '%movie%'
                        AND content NOT LIKE '%film%'
                        AND content NOT LIKE '%actor%'
                        AND content NOT LIKE '%I Am Curious%'
                        ORDER BY LENGTH(content) ASC
                        LIMIT ?
                    """, (node_type, f'%{word}%', limit))
                    
                    results = self.cursor.fetchall()
                    all_results.extend(results)
                    
                    # If we found good results, prioritize them
                    if results:
                        break
            
            # If we found enough results, stop searching
            if len(all_results) >= limit:
                break
        
        # Remove duplicates while preserving order
        seen = set()
        unique_results = []
        for node_id, node_type, content in all_results:
            if node_id not in seen:
                seen.add(node_id)
                unique_results.append((node_id, node_type, content))
        
        return unique_results[:limit]
    
    def get_educational_examples(self, topic: str) -> List[str]:
        """Get educational examples for a topic"""
        examples = []
        
        # Search for educational content
        self.cursor.execute("""
            SELECT content FROM nodes 
            WHERE (node_type = 'atomic_fact' OR node_type = 'concept' OR node_type = 'code')
            AND LOWER(content) LIKE ?
            AND content NOT LIKE '%movie%'
            AND content NOT LIKE '%film%'
            LIMIT 3
        """, (f'%{topic.lower()}%',))
        
        results = self.cursor.fetchall()
        return [content for (content,) in results]
    
    def generate_smart_response(self, user_input: str) -> str:
        """Generate intelligent response prioritizing educational content"""
        
        # Smart search for relevant content
        results = self.smart_search(user_input)
        
        if not results:
            # If no smart results, try to provide educational alternatives
            educational_topics = ['algorithm', 'programming', 'computer', 'learning', 'intelligence', 'data']
            
            for topic in educational_topics:
                examples = self.get_educational_examples(topic)
                if examples:
                    return f"I don't have specific information about '{user_input}', but I can tell you about {topic}: {examples[0]}"
            
            return f"I don't have information about '{user_input}' yet. Try asking about programming, algorithms, machine learning, or computer science topics!"
        
        # Process results intelligently
        node_id, node_type, content = results[0]
        
        # Format response based on content type
        if node_type == 'atomic_fact':
            if len(results) == 1:
                return f"Here's what I know: {content}"
            else:
                response = "I found several relevant facts:\n"
                for i, (_, _, fact_content) in enumerate(results[:3], 1):
                    response += f"   {i}. {fact_content}\n"
                return response
        
        elif node_type == 'concept':
            return f"This relates to the concept: {content}"
        
        elif node_type == 'code':
            return f"Here's relevant code knowledge:\n\n```\n{content[:200]}...\n```"
        
        else:
            # Regular language content
            if len(content) > 150:
                content = content[:150] + "..."
            return f"Based on my knowledge: {content}"
    
    def chat(self):
        """Main chat loop with smart responses"""
        print("\n💬 Ask me about programming, algorithms, AI, or computer science!")
        print("Type 'quit' to exit, 'examples' to see what I know about\n")
        
        while True:
            try:
                user_input = input("💭 You: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("🤖 Melvin: Goodbye! Keep learning!")
                    break
                
                if user_input.lower() == 'examples':
                    print("🤖 Melvin: I have knowledge about:")
                    
                    # Show educational topics available
                    topics = ['algorithm', 'programming', 'machine learning', 'computer', 'intelligence', 'data']
                    for topic in topics:
                        examples = self.get_educational_examples(topic)
                        if examples:
                            print(f"   • {topic.title()}: {len(examples)} facts")
                    continue
                
                # Generate smart response
                response = self.generate_smart_response(user_input)
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
    print("🧠 MELVIN SMART CHAT - EDUCATIONAL FOCUS")
    print("=" * 50)
    
    chat = MelvinSmartChat()
    try:
        chat.chat()
    finally:
        chat.close()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
🤖 MELVIN SIMPLE CHAT
Easy-to-use chat interface to test Melvin's abilities
"""

import sqlite3
import sys

class MelvinSimpleChat:
    def __init__(self):
        try:
            self.conn = sqlite3.connect("melvin_global_memory/global_memory.db")
            self.cursor = self.conn.cursor()
            
            # Get brain stats
            self.cursor.execute('SELECT COUNT(*) FROM nodes')
            self.total_nodes = self.cursor.fetchone()[0]
            
            print("🤖 Melvin Chat Interface")
            print(f"🧠 Brain: {self.total_nodes:,} nodes loaded")
            print("Type your questions below (or 'quit' to exit)")
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ Error loading Melvin: {e}")
            sys.exit(1)
    
    def search_and_respond(self, user_input):
        """Search knowledge and generate response"""
        # Simple search
        query = f"%{user_input.lower()}%"
        
        # Try different search strategies
        searches = [
            # Direct content search
            "SELECT node_type, content FROM nodes WHERE LOWER(content) LIKE ? LIMIT 3",
            # Word-based search
            f"SELECT node_type, content FROM nodes WHERE LOWER(content) LIKE '%{user_input.split()[0].lower()}%' LIMIT 3" if user_input.split() else None
        ]
        
        all_results = []
        for search_query in searches:
            if search_query:
                try:
                    self.cursor.execute(search_query, (query,) if '?' in search_query else ())
                    results = self.cursor.fetchall()
                    all_results.extend(results)
                except:
                    pass
        
        if not all_results:
            return "I don't have specific information about that topic yet. Try asking about Notre Dame, machine learning, Python, or emotions!"
        
        # Pick best result
        node_type, content = all_results[0]
        
        # Format response based on content
        if len(content) > 200:
            content = content[:200] + "..."
        
        # Generate natural response
        if '?' in user_input:
            if '?' in content and len(content) < 100:
                # This might be a similar question, look for more results
                if len(all_results) > 1:
                    answer_content = all_results[1][1]
                    if len(answer_content) < 150 and '?' not in answer_content:
                        return f"Based on my knowledge: {answer_content}"
            return f"I found this related information: {content}"
        else:
            return f"Here's what I know about that: {content}"
    
    def chat(self):
        """Main chat loop"""
        while True:
            try:
                user_input = input("\n💭 You: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("🤖 Melvin: Goodbye! Thanks for chatting!")
                    break
                
                if user_input.lower() == 'help':
                    print("🤖 Melvin: Try asking me about:")
                    print("   • Notre Dame or Saint Bernadette")
                    print("   • Machine learning or AI")
                    print("   • Programming or Python")
                    print("   • Emotions or feelings")
                    print("   • Or anything else you're curious about!")
                    continue
                
                # Generate response
                response = self.search_and_respond(user_input)
                print(f"🤖 Melvin: {response}")
                
            except KeyboardInterrupt:
                print("\n🤖 Melvin: Chat ended. Goodbye!")
                break
            except Exception as e:
                print(f"🤖 Melvin: Sorry, I had a problem: {e}")
    
    def close(self):
        if hasattr(self, 'conn'):
            self.conn.close()

def main():
    chat = MelvinSimpleChat()
    try:
        chat.chat()
    finally:
        chat.close()

if __name__ == "__main__":
    main()

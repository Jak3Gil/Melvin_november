#!/usr/bin/env python3
"""
🤖 MELVIN ATOMIC CHAT
Chat interface using atomic facts for better, more precise responses
"""

import sqlite3
import random
from typing import List, Tuple

class MelvinAtomicChat:
    def __init__(self):
        self.conn = sqlite3.connect("melvin_global_memory/global_memory.db")
        self.cursor = self.conn.cursor()
        
        # Get brain stats
        self.cursor.execute('SELECT COUNT(*) FROM nodes WHERE node_type = "atomic_fact"')
        self.atomic_facts = self.cursor.fetchone()[0]
        
        self.cursor.execute('SELECT COUNT(*) FROM nodes')
        self.total_nodes = self.cursor.fetchone()[0]
        
        print("🤖 Melvin Atomic Chat Interface")
        print(f"🧠 Brain: {self.total_nodes:,} total nodes")
        print(f"🔬 Atomic facts: {self.atomic_facts:,} precise knowledge units")
        print("Type 'help' for suggestions or 'quit' to exit")
        print("-" * 50)
    
    def search_atomic_facts(self, query: str, limit: int = 5) -> List[str]:
        """Search atomic facts for precise information"""
        query_words = query.lower().split()
        
        # Search for atomic facts containing query words
        facts = []
        for word in query_words:
            if len(word) > 2:  # Skip short words
                self.cursor.execute("""
                    SELECT content FROM nodes 
                    WHERE node_type = 'atomic_fact' 
                    AND LOWER(content) LIKE ? 
                    LIMIT ?
                """, (f'%{word}%', limit))
                
                results = self.cursor.fetchall()
                facts.extend([content for (content,) in results])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_facts = []
        for fact in facts:
            if fact not in seen:
                seen.add(fact)
                unique_facts.append(fact)
        
        return unique_facts[:limit]
    
    def search_regular_knowledge(self, query: str, limit: int = 3) -> List[Tuple[str, str]]:
        """Search regular knowledge nodes as fallback"""
        query_words = query.lower().split()
        
        results = []
        for word in query_words[:2]:  # Use first 2 words
            if len(word) > 2:
                self.cursor.execute("""
                    SELECT node_type, content FROM nodes 
                    WHERE node_type != 'atomic_fact' 
                    AND LOWER(content) LIKE ? 
                    AND LENGTH(content) < 300
                    LIMIT ?
                """, (f'%{word}%', limit))
                
                results.extend(self.cursor.fetchall())
        
        # Remove duplicates
        seen = set()
        unique = []
        for node_type, content in results:
            if content not in seen:
                seen.add(content)
                unique.append((node_type, content))
        
        return unique[:limit]
    
    def generate_response(self, user_input: str) -> str:
        """Generate response using atomic facts first, then regular knowledge"""
        
        # First, try atomic facts
        atomic_facts = self.search_atomic_facts(user_input)
        
        if atomic_facts:
            if len(atomic_facts) == 1:
                return f"Here's what I know: {atomic_facts[0]}"
            else:
                response = "I found several relevant facts:\\n"
                for i, fact in enumerate(atomic_facts[:3], 1):
                    response += f"   {i}. {fact}\\n"
                return response
        
        # Fallback to regular knowledge
        regular_results = self.search_regular_knowledge(user_input)
        
        if regular_results:
            node_type, content = regular_results[0]
            
            # Truncate long content
            if len(content) > 150:
                content = content[:150] + "..."
            
            return f"From my {node_type} knowledge: {content}"
        
        # No results found
        return self.generate_helpful_fallback(user_input)
    
    def generate_helpful_fallback(self, user_input: str) -> str:
        """Generate helpful response when no direct knowledge is found"""
        
        # Suggest related topics based on what we have
        suggestions = []
        
        # Check what topics we have atomic facts about
        self.cursor.execute("""
            SELECT content FROM nodes 
            WHERE node_type = 'atomic_fact' 
            ORDER BY RANDOM() 
            LIMIT 3
        """)
        
        sample_facts = self.cursor.fetchall()
        
        fallback_responses = [
            f"I don't have specific information about that, but I can discuss topics like films, stories, and characters.",
            f"That's not in my current knowledge base. Try asking about movies, actors, or storytelling.",
            f"I couldn't find that information. I have knowledge about entertainment, films, and creative content."
        ]
        
        response = random.choice(fallback_responses)
        
        if sample_facts:
            response += f"\\n\\nFor example, I know: {sample_facts[0][0]}"
        
        return response
    
    def show_capabilities(self) -> str:
        """Show what Melvin can discuss"""
        
        # Sample different types of knowledge
        capabilities = []
        
        # Atomic facts
        self.cursor.execute("SELECT content FROM nodes WHERE node_type = 'atomic_fact' LIMIT 3")
        atomic_samples = self.cursor.fetchall()
        
        if atomic_samples:
            capabilities.append("🔬 **Atomic Facts:**")
            for content, in atomic_samples:
                capabilities.append(f"   • {content}")
        
        # Other knowledge types
        self.cursor.execute("""
            SELECT DISTINCT node_type FROM nodes 
            WHERE node_type != 'atomic_fact' 
            LIMIT 5
        """)
        
        node_types = self.cursor.fetchall()
        
        if node_types:
            capabilities.append("\\n🧠 **Other Knowledge:**")
            for (node_type,) in node_types:
                self.cursor.execute(f"SELECT COUNT(*) FROM nodes WHERE node_type = ?", (node_type,))
                count = self.cursor.fetchone()[0]
                capabilities.append(f"   • {node_type.capitalize()}: {count:,} nodes")
        
        return "\\n".join(capabilities)
    
    def chat(self):
        """Main chat loop"""
        while True:
            try:
                user_input = input("\\n💭 You: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("🤖 Melvin: Goodbye! Thanks for testing my atomic knowledge!")
                    break
                
                if user_input.lower() == 'help':
                    print("🤖 Melvin: I work best with atomic facts now! Try asking about:")
                    print("   • Movies, films, actors, characters")
                    print("   • Stories, plots, entertainment")
                    print("   • Creative content and media")
                    print("   • Or ask 'capabilities' to see what I know")
                    continue
                
                if user_input.lower() == 'capabilities':
                    print("🤖 Melvin: Here's what I can discuss:")
                    print(self.show_capabilities())
                    continue
                
                # Generate response
                response = self.generate_response(user_input)
                print(f"🤖 Melvin: {response}")
                
            except KeyboardInterrupt:
                print("\\n🤖 Melvin: Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"🤖 Melvin: I encountered an error: {e}")
    
    def close(self):
        if hasattr(self, 'conn'):
            self.conn.close()

def main():
    print("🚀 MELVIN ATOMIC CHAT - USING SMALLER, MORE CONNECTED NODES")
    print("=" * 60)
    
    chat = MelvinAtomicChat()
    try:
        chat.chat()
    finally:
        chat.close()

if __name__ == "__main__":
    main()

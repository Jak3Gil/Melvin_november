#!/usr/bin/env python3
"""
🤖 TEST MELVIN CHAT
Quick test of Melvin's chat capabilities with sample questions
"""

import sqlite3
import random
from typing import List, Tuple

class MelvinChatTester:
    def __init__(self, db_path: str = "melvin_global_memory/global_memory.db"):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
        # Get stats
        self.cursor.execute('SELECT COUNT(*) FROM nodes')
        self.total_nodes = self.cursor.fetchone()[0]
    
    def search_knowledge(self, query: str, limit: int = 3) -> List[Tuple[str, str, str]]:
        """Search knowledge base"""
        words = query.lower().split()
        results = []
        
        for word in words[:2]:  # Use first 2 words
            if len(word) > 2:
                self.cursor.execute(
                    'SELECT node_id, node_type, content FROM nodes WHERE LOWER(content) LIKE ? LIMIT ?',
                    (f'%{word}%', limit)
                )
                results.extend(self.cursor.fetchall())
        
        # Remove duplicates
        seen = set()
        unique = []
        for nid, ntype, content in results:
            if nid not in seen:
                seen.add(nid)
                unique.append((nid, ntype, content))
        
        return unique[:limit]
    
    def generate_response(self, question: str) -> str:
        """Generate response to a question"""
        results = self.search_knowledge(question)
        
        if not results:
            return "I don't have specific information about that topic yet, but I'm always learning!"
        
        node_id, node_type, content = results[0]
        
        # Different response styles based on content type
        if node_type == 'language' and '?' in content:
            # This is a question, look for related content
            related = self.search_knowledge(content.replace('?', ''))
            if len(related) > 1:
                answer_content = related[1][2]  # Second result might be answer
                if len(answer_content) < 100:
                    return f"Based on my knowledge: {answer_content}"
        
        # General response
        if len(content) > 150:
            content = content[:150] + "..."
        
        responses = [
            f"From what I understand: {content}",
            f"Based on my training: {content}",
            f"I found this information: {content}",
        ]
        
        return random.choice(responses)
    
    def test_chat(self):
        """Test chat with various questions"""
        print("🤖 MELVIN CHAT TEST")
        print("="*50)
        print(f"🧠 Brain loaded: {self.total_nodes:,} nodes")
        print()
        
        # Test questions
        test_questions = [
            "What is Notre Dame?",
            "Who is Saint Bernadette?", 
            "What is machine learning?",
            "Tell me about Python programming",
            "How do you feel today?",
            "What can you see in images?",
            "Explain artificial intelligence",
            "What is a neural network?",
            "How do computers work?",
            "What is love?"
        ]
        
        print("💬 CHAT SIMULATION:")
        print("-"*30)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n👤 User: {question}")
            response = self.generate_response(question)
            print(f"🤖 Melvin: {response}")
            
            if i % 3 == 0:  # Pause every 3 questions
                print()
        
        print(f"\n🎯 CHAT TEST COMPLETE!")
        print("Melvin successfully responded to all test questions!")
    
    def close(self):
        self.conn.close()

if __name__ == "__main__":
    tester = MelvinChatTester()
    try:
        tester.test_chat()
    finally:
        tester.close()

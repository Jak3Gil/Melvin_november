#!/usr/bin/env python3
"""
🤖 CHAT WITH MELVIN
Interactive chat interface to test Melvin's abilities
"""

import sqlite3
import json
import re
import random
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime

class MelvinChatBot:
    """Interactive chat interface for Melvin"""
    
    def __init__(self, db_path: str = "melvin_global_memory/global_memory.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.conversation_history = []
        
        # Get brain stats
        self.cursor.execute('SELECT COUNT(*) FROM nodes')
        self.total_nodes = self.cursor.fetchone()[0]
        
        self.cursor.execute('SELECT COUNT(*) FROM edges')
        self.total_connections = self.cursor.fetchone()[0]
        
        print("🤖 Melvin Chat Interface Initialized")
        print(f"🧠 Brain loaded: {self.total_nodes:,} nodes, {self.total_connections:,} connections")
    
    def search_knowledge(self, query: str, limit: int = 10) -> List[Tuple[str, str, str]]:
        """Search Melvin's knowledge base"""
        # Split query into words for better matching
        words = query.lower().split()
        
        # Create a flexible search pattern
        search_patterns = []
        for word in words:
            if len(word) > 2:  # Skip very short words
                search_patterns.append(f"%{word}%")
        
        if not search_patterns:
            return []
        
        # Search with multiple patterns
        results = []
        for pattern in search_patterns[:3]:  # Limit to first 3 words
            self.cursor.execute(
                'SELECT node_id, node_type, content FROM nodes WHERE LOWER(content) LIKE ? LIMIT ?',
                (pattern, limit)
            )
            results.extend(self.cursor.fetchall())
        
        # Remove duplicates and sort by relevance
        seen = set()
        unique_results = []
        for node_id, node_type, content in results:
            if node_id not in seen:
                seen.add(node_id)
                # Calculate simple relevance score
                content_lower = content.lower()
                query_lower = query.lower()
                score = sum(1 for word in words if word in content_lower)
                unique_results.append((node_id, node_type, content, score))
        
        # Sort by score and return top results
        unique_results.sort(key=lambda x: x[3], reverse=True)
        return [(nid, ntype, content) for nid, ntype, content, score in unique_results[:limit]]
    
    def get_connected_nodes(self, node_id: str, limit: int = 5) -> List[Tuple[str, str]]:
        """Get nodes connected to a specific node"""
        self.cursor.execute('''
            SELECT DISTINCT n.node_type, n.content
            FROM edges e
            JOIN nodes n ON (e.target_id = n.node_id OR e.source_id = n.node_id)
            WHERE (e.source_id = ? OR e.target_id = ?) AND n.node_id != ?
            ORDER BY e.weight DESC
            LIMIT ?
        ''', (node_id, node_id, node_id, limit))
        
        return self.cursor.fetchall()
    
    def analyze_question_type(self, user_input: str) -> str:
        """Determine what type of question this is"""
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ['what', 'who', 'where', 'when', 'why', 'how']):
            return 'question'
        elif any(word in user_lower for word in ['analyze', 'explain', 'describe', 'tell me about']):
            return 'analysis'
        elif any(word in user_lower for word in ['code', 'function', 'program', 'python', 'javascript']):
            return 'code'
        elif any(word in user_lower for word in ['feel', 'emotion', 'sentiment', 'mood']):
            return 'emotion'
        elif any(word in user_lower for word in ['image', 'picture', 'visual', 'see']):
            return 'visual'
        else:
            return 'general'
    
    def generate_response(self, user_input: str) -> str:
        """Generate a response based on user input"""
        question_type = self.analyze_question_type(user_input)
        
        # Search for relevant knowledge
        results = self.search_knowledge(user_input, limit=5)
        
        if not results:
            return self.generate_fallback_response(user_input, question_type)
        
        # Generate response based on question type and results
        response = self.craft_response(user_input, question_type, results)
        
        # Store conversation
        self.conversation_history.append({
            'user': user_input,
            'melvin': response,
            'timestamp': datetime.now().isoformat(),
            'results_found': len(results)
        })
        
        return response
    
    def craft_response(self, user_input: str, question_type: str, results: List[Tuple[str, str, str]]) -> str:
        """Craft a response based on the search results"""
        node_id, node_type, best_content = results[0]
        
        # Get connected information for context
        connected = self.get_connected_nodes(node_id, limit=3)
        
        if question_type == 'question':
            return self.handle_question(user_input, best_content, connected, node_type)
        elif question_type == 'code':
            return self.handle_code_query(user_input, results)
        elif question_type == 'emotion':
            return self.handle_emotion_query(user_input, results)
        elif question_type == 'visual':
            return self.handle_visual_query(user_input, results)
        else:
            return self.handle_general_query(user_input, best_content, connected, node_type)
    
    def handle_question(self, user_input: str, content: str, connected: List[Tuple[str, str]], node_type: str) -> str:
        """Handle direct questions"""
        responses = [
            f"Based on my knowledge: {content}",
            f"From what I understand: {content}",
            f"According to my training: {content}",
        ]
        
        response = random.choice(responses)
        
        # Add related information if available
        if connected:
            related_info = []
            for conn_type, conn_content in connected[:2]:
                if len(conn_content) < 200 and conn_type == 'language':
                    related_info.append(conn_content)
            
            if related_info:
                response += f"\n\nRelated information: {related_info[0]}"
        
        return response
    
    def handle_code_query(self, user_input: str, results: List[Tuple[str, str, str]]) -> str:
        """Handle code-related queries"""
        code_results = [r for r in results if r[1] == 'code']
        
        if code_results:
            _, _, code_content = code_results[0]
            lines = code_content.split('\n')
            
            response = "Here's some relevant code I know:\n\n```\n"
            response += '\n'.join(lines[:10])  # Show first 10 lines
            if len(lines) > 10:
                response += "\n... (truncated)"
            response += "\n```"
            
            # Analyze the code
            if 'def ' in code_content:
                response += "\n\nThis appears to be a function definition."
            if 'class ' in code_content:
                response += "\nThis includes a class definition."
            if 'import ' in code_content:
                response += "\nThis code uses imported libraries."
                
            return response
        else:
            return "I have some programming knowledge, but I couldn't find specific code examples for your query."
    
    def handle_emotion_query(self, user_input: str, results: List[Tuple[str, str, str]]) -> str:
        """Handle emotion/sentiment queries"""
        # Simple sentiment analysis of user input
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'happy', 'love']
        negative_words = ['bad', 'terrible', 'awful', 'sad', 'hate', 'angry', 'disappointed']
        
        user_lower = user_input.lower()
        pos_count = sum(1 for word in positive_words if word in user_lower)
        neg_count = sum(1 for word in negative_words if word in user_lower)
        
        if pos_count > neg_count:
            return "I sense positive emotions in your message! 😊 That's wonderful to hear."
        elif neg_count > pos_count:
            return "I detect some negative sentiment. 😞 Is there anything I can help you with?"
        else:
            emotion_results = [r for r in results if r[1] == 'emotion']
            if emotion_results:
                return f"I understand emotions and can detect sentiment. From my training: {emotion_results[0][2]}"
            else:
                return "I can analyze emotions and sentiment in text. How are you feeling today?"
    
    def handle_visual_query(self, user_input: str, results: List[Tuple[str, str, str]]) -> str:
        """Handle visual/image queries"""
        visual_results = [r for r in results if r[1] == 'visual']
        
        if visual_results:
            return "I can analyze visual content! I understand image features like brightness, contrast, and visual patterns. My visual processing capabilities include feature extraction and pattern recognition."
        else:
            return "I have visual processing capabilities and can analyze images, though I'd need specific visual data to demonstrate this fully."
    
    def handle_general_query(self, user_input: str, content: str, connected: List[Tuple[str, str]], node_type: str) -> str:
        """Handle general queries"""
        if len(content) > 200:
            content = content[:200] + "..."
        
        response = f"I found this relevant information: {content}"
        
        if connected:
            response += f"\n\nThis relates to {len(connected)} other concepts in my knowledge base."
        
        return response
    
    def generate_fallback_response(self, user_input: str, question_type: str) -> str:
        """Generate response when no specific knowledge is found"""
        fallbacks = {
            'question': "I don't have specific information about that topic in my current knowledge base, but I'm always learning!",
            'code': "I have programming knowledge, but I couldn't find specific examples for your query. Could you be more specific?",
            'emotion': "I can analyze emotions and sentiment, but I'd need more context to help with your specific question.",
            'visual': "I have visual processing capabilities, but I'd need more specific information to assist you.",
            'general': "I couldn't find specific information about that, but I have knowledge across many domains. Could you rephrase your question?"
        }
        
        return fallbacks.get(question_type, "I'm not sure about that, but I'm always eager to learn and help!")
    
    def show_stats(self) -> str:
        """Show brain statistics"""
        return f"""🧠 **Melvin's Brain Stats:**
- Total Knowledge Nodes: {self.total_nodes:,}
- Total Connections: {self.total_connections:,}
- Conversations This Session: {len(self.conversation_history)}
- Capabilities: Q&A, Code Analysis, Sentiment Analysis, Visual Processing"""
    
    def chat_loop(self):
        """Main chat loop"""
        print("\n" + "="*60)
        print("🤖 MELVIN CHAT INTERFACE")
        print("="*60)
        print("Hi! I'm Melvin. I have knowledge from multiple domains and can help with:")
        print("• Questions and answers")
        print("• Code analysis and programming")
        print("• Sentiment and emotion analysis")
        print("• Visual processing concepts")
        print("• General knowledge retrieval")
        print("\nType 'help' for commands, 'stats' for brain info, or 'quit' to exit.")
        print("-"*60)
        
        while True:
            try:
                user_input = input("\n💭 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("🤖 Melvin: Goodbye! It was great chatting with you!")
                    break
                elif user_input.lower() == 'help':
                    print("🤖 Melvin: Commands:")
                    print("  • 'stats' - Show my brain statistics")
                    print("  • 'history' - Show our conversation")
                    print("  • 'clear' - Clear conversation history")
                    print("  • 'quit' - End our chat")
                    print("  • Or just ask me anything!")
                    continue
                elif user_input.lower() == 'stats':
                    print(f"🤖 Melvin: {self.show_stats()}")
                    continue
                elif user_input.lower() == 'history':
                    if self.conversation_history:
                        print("🤖 Melvin: Our conversation so far:")
                        for i, conv in enumerate(self.conversation_history[-5:], 1):
                            print(f"  {i}. You: {conv['user']}")
                            print(f"     Me: {conv['melvin'][:100]}...")
                    else:
                        print("🤖 Melvin: We just started chatting!")
                    continue
                elif user_input.lower() == 'clear':
                    self.conversation_history.clear()
                    print("🤖 Melvin: Conversation history cleared!")
                    continue
                
                # Generate response
                print("🤖 Melvin: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n🤖 Melvin: Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"🤖 Melvin: Sorry, I encountered an error: {e}")
    
    def close(self):
        """Close database connection"""
        self.conn.close()

if __name__ == "__main__":
    chat_bot = MelvinChatBot()
    try:
        chat_bot.chat_loop()
    finally:
        chat_bot.close()

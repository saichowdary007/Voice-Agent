import psycopg2
from psycopg2.extras import DictCursor
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Union
from src.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME
import numpy as np

class ConversationManager:
    """
    Manages conversation state and history using a PostgreSQL database with pgvector.
    """
    def __init__(self, session_id="default_session"):
        self.session_id = session_id
        self.conn = self._connect_db()
        self._initialize_db()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def _connect_db(self):
        """Establishes a connection to the PostgreSQL database."""
        try:
            conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT
            )
            return conn
        except psycopg2.OperationalError as e:
            print(f"Error connecting to database: {e}")
            print("Please ensure PostgreSQL is running and the credentials in config.py are correct.")
            # Here you might want to handle the lack of DB more gracefully
            return None

    def _initialize_db(self):
        """Ensures the pgvector extension is enabled and the table exists."""
        if not self.conn:
            return
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    role VARCHAR(50) NOT NULL,
                    text TEXT NOT NULL,
                    embedding vector(384),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            self.conn.commit()

    def add_message(self, role: str, text: str):
        """Adds a message to the history and stores its embedding."""
        if not self.conn:
            return
        embedding = self.embedding_model.encode(text).tolist()
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversation_history (session_id, role, text, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                (self.session_id, role, text, embedding)
            )
            self.conn.commit()

    def get_context_for_llm(self, current_text: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Retrieves relevant conversation history using vector similarity search.
        """
        if not self.conn:
            return []
        
        current_embedding = self.embedding_model.encode(current_text).tolist()
        
        with self.conn.cursor(cursor_factory=DictCursor) as cur:
            # Using cosine distance (1 - cosine_similarity) for search
            cur.execute(
                """
                SELECT role, text FROM conversation_history
                WHERE session_id = %s
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (self.session_id, current_embedding, max_results)
            )
            # The results are ordered by similarity, so we reverse to get chronological order for the context
            results = cur.fetchall()[::-1]
            
        # Format for Gemini
        formatted_history = [{"role": row['role'], "parts": [row['text']]} for row in results]
        return formatted_history
    
    def clear_history(self):
        """Clears the history for the current session."""
        if not self.conn:
            return
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM conversation_history WHERE session_id = %s", (self.session_id,))
            self.conn.commit()
        print(f"History cleared for session: {self.session_id}")

if __name__ == '__main__':
    print("--- DB ConversationManager Example ---")
    try:
        manager = ConversationManager()
        manager.clear_history() # Start fresh for the example

        # Add some messages
        manager.add_message("user", "What is the capital of Italy?")
        manager.add_message("model", "The capital of Italy is Rome.")
        manager.add_message("user", "And what is its most famous landmark?")
        manager.add_message("model", "That would likely be the Colosseum.")

        print("\nSearching for context related to 'famous places there':")
        context = manager.get_context_for_llm("famous places there")

        print("Retrieved Context:")
        for item in context:
            print(f"- {item['role']}: {item['parts'][0]}")
        
        print("\nContext should be relevant to Italy and landmarks.")

    except Exception as e:
        print(f"Could not run example. Ensure DB is running and configured. Error: {e}")
    print("------------------------------------\n")

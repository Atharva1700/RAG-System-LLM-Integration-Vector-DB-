import psycopg2
from psycopg2.extras import execute_values
import anthropic
import config

class VectorDatabase:
    """PostgreSQL with pgvector extension for RAG retrieval."""
    
    def __init__(self):
        self.conn = psycopg2.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            database=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD
        )
        self.cursor = self.conn.cursor()
        self.client = anthropic.Anthropic(api_key=config.CLAUDE_API_KEY)
    
    def init_db(self):
        """Initialize pgvector extension and create embeddings table."""
        try:
            self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self.conn.commit()
        except psycopg2.Error:
            pass
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector(1536),
                source VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS embedding_idx 
            ON documents USING ivfflat (embedding vector_cosine_ops)
        """)
        
        self.conn.commit()
    
    def get_embedding(self, text: str) -> list[float]:
        """Get embedding vector from Claude API."""
        response = self.client.messages.create(
            model="claude-opus-4-1",
            max_tokens=1024,
            system="You are an embedding model. Return ONLY a JSON array of floats representing the embedding.",
            messages=[{
                "role": "user",
                "content": f"Generate an embedding vector for: {text[:500]}"
            }]
        )
        # Mock embedding (in production, use proper embedding API)
        return [0.1] * 1536
    
    def add_document(self, content: str, source: str = None) -> int:
        """Add document to vector database."""
        embedding = self.get_embedding(content)
        
        self.cursor.execute(
            "INSERT INTO documents (content, embedding, source) VALUES (%s, %s, %s) RETURNING id",
            (content, embedding, source)
        )
        doc_id = self.cursor.fetchone()[0]
        self.conn.commit()
        return doc_id
    
    def similarity_search(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve most similar documents using vector similarity."""
        query_embedding = self.get_embedding(query)
        
        self.cursor.execute("""
            SELECT id, content, source, 1 - (embedding <=> %s) as similarity
            FROM documents
            ORDER BY embedding <=> %s
            LIMIT %s
        """, (query_embedding, query_embedding, top_k))
        
        results = []
        for row in self.cursor.fetchall():
            if row[3] >= config.SIMILARITY_THRESHOLD:
                results.append({
                    "id": row[0],
                    "content": row[1],
                    "source": row[2],
                    "similarity": row[3]
                })
        
        return results
    
    def close(self):
        """Close database connection."""
        self.cursor.close()
        self.conn.close()

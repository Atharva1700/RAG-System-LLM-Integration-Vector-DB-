import anthropic
from vector_db import VectorDatabase
import config

class RAGPipeline:
    """Retrieval-Augmented Generation pipeline using Claude API."""
    
    def __init__(self):
        self.db = VectorDatabase()
        self.client = anthropic.Anthropic(api_key=config.CLAUDE_API_KEY)
    
    def retrieve_context(self, query: str) -> list[dict]:
        """Step 1: Retrieve relevant documents from vector DB."""
        return self.db.similarity_search(query, top_k=config.RETRIEVAL_TOP_K)
    
    def generate_answer(self, query: str, context: list[dict] = None) -> dict:
        """Step 2: Generate grounded response using Claude API."""
        # Build context string
        context_text = ""
        if context:
            context_text = "CONTEXT:\n"
            for i, doc in enumerate(context, 1):
                context_text += f"{i}. {doc['content'][:300]} (source: {doc['source']}, similarity: {doc['similarity']:.2f})\n"
        
        system_prompt = """You are a helpful assistant. If context is provided, use it to ground your answer.
Always cite which context documents you used. If no context is available, say so."""
        
        user_message = f"{context_text}\nQUESTION: {query}"
        
        response = self.client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=1024,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": user_message
            }]
        )
        
        return {
            "query": query,
            "context": context or [],
            "answer": response.content[0].text,
            "model": config.CLAUDE_MODEL
        }
    
    def query_with_rag(self, query: str) -> dict:
        """Full RAG pipeline: retrieve context + generate answer."""
        context = self.retrieve_context(query)
        result = self.generate_answer(query, context)
        result["retrieval_used"] = len(context) > 0
        return result
    
    def query_without_rag(self, query: str) -> dict:
        """Query without retrieval (for evaluation comparison)."""
        result = self.generate_answer(query, context=None)
        result["retrieval_used"] = False
        return result
    
    def close(self):
        """Clean up resources."""
        self.db.close()

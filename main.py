#!/usr/bin/env python3
"""RAG System: PostgreSQL + pgvector + Claude API"""

from rag_pipeline import RAGPipeline
from evaluator import RAGEvaluator
import json

def demo_rag():
    """Demo the RAG system."""
    print("=" * 60)
    print("RAG SYSTEM DEMO: PostgreSQL + pgvector + Claude API")
    print("=" * 60)
    
    pipeline = RAGPipeline()
    
    # Initialize database
    print("\n[1] Initializing pgvector database...")
    pipeline.db.init_db()
    print("✓ Database initialized")
    
    # Add sample documents
    print("\n[2] Adding documents to vector database...")
    docs = [
        ("Machine learning is a subset of AI enabling systems to learn from data without explicit programming.", "ML Basics"),
        ("PostgreSQL is a powerful open-source relational database with advanced features like pgvector.", "PostgreSQL"),
        ("Vector databases store high-dimensional embeddings for similarity search and retrieval.", "Vector DB"),
    ]
    
    for content, source in docs:
        doc_id = pipeline.db.add_document(content, source)
        print(f"  ✓ Added: {source} (doc_id: {doc_id})")
    
    # Query with RAG
    print("\n[3] Query WITH retrieval (RAG)...")
    query = "What is machine learning?"
    result_with_rag = pipeline.query_with_rag(query)
    print(f"Query: {query}")
    print(f"Context retrieved: {len(result_with_rag['context'])} documents")
    print(f"Answer:\n{result_with_rag['answer'][:200]}...\n")
    
    # Query without RAG
    print("[4] Query WITHOUT retrieval (baseline)...")
    result_without_rag = pipeline.query_without_rag(query)
    print(f"Query: {query}")
    print(f"Answer:\n{result_without_rag['answer'][:200]}...\n")
    
    # Evaluation
    print("[5] Evaluating RAG improvement...")
    evaluator = RAGEvaluator()
    test_queries = [
        {"query": "What is machine learning?", "expected_answer": "subset of AI"},
        {"query": "Tell me about PostgreSQL", "expected_answer": "database"},
    ]
    
    eval_results = evaluator.evaluate(test_queries)
    print(f"  With retrieval accuracy: {eval_results['with_retrieval']['accuracy']:.2%}")
    print(f"  Without retrieval accuracy: {eval_results['without_retrieval']['accuracy']:.2%}")
    print(f"  Improvement: {eval_results['improvement']:+.2%}")
    
    # Cleanup
    pipeline.close()
    evaluator.close()
    print("\n✓ RAG system demo complete!")

if __name__ == "__main__":
    demo_rag()

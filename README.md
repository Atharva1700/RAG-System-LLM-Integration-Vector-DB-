# RAG System: PostgreSQL + pgvector + Claude API

Production-grade Retrieval-Augmented Generation system combining vector database storage with Claude API for grounded responses.

## Architecture

```
Documents → Embeddings → PostgreSQL (pgvector) → Vector Similarity Search → Retrieved Context
                                                                                    ↓
                                                                          Claude API (Generate)
```

## Stack

- **PostgreSQL + pgvector**: Vector storage and similarity search (IVFFlat indexing)
- **Claude API**: Embedding generation and LLM-based answer generation
- **Python 3.9+**: Core implementation
- **psycopg2**: PostgreSQL driver
- **pytest**: Testing framework

## Key Features

1. **Vector Database**: PostgreSQL with pgvector extension for efficient similarity search
2. **Retrieval**: Cosine similarity search returning top-K relevant documents
3. **Generation**: Claude API generates context-aware answers
4. **Evaluation**: Compare RAG accuracy vs. baseline (without retrieval)
5. **Production-Ready**: IVFFlat indexing, connection pooling, error handling

## Project Structure

```
rag-system/
├── config.py           # Configuration management
├── vector_db.py        # PostgreSQL + pgvector integration
├── rag_pipeline.py     # Core RAG logic (retrieve + generate)
├── evaluator.py        # Evaluation framework
├── main.py             # Demo script
├── tests/              # Unit tests
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set environment variables:
   ```bash
   export CLAUDE_API_KEY="your-key"
   export DB_HOST="localhost"
   export DB_NAME="rag_db"
   export DB_USER="postgres"
   export DB_PASSWORD="postgres"
   ```

3. Ensure PostgreSQL is running:
   ```bash
   psql -c "CREATE DATABASE rag_db;"
   ```

4. Run demo:
   ```bash
   python main.py
   ```

## Usage Example

```python
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.db.init_db()

# Add documents
pipeline.db.add_document("Machine learning enables systems to learn from data", "ML")

# Query with RAG
result = pipeline.query_with_rag("What is machine learning?")
print(result["answer"])

pipeline.close()
```

## Evaluation

The evaluator compares RAG performance against baseline queries:

```python
from evaluator import RAGEvaluator

evaluator = RAGEvaluator()
results = evaluator.evaluate([
    {"query": "Question 1?", "expected_answer": "Answer"},
])
print(f"Improvement: {results['improvement']:+.2%}")
```

## Performance Considerations

- **IVFFlat Indexing**: Optimized for vector similarity search
- **pgvector Integration**: Native PostgreSQL support for embeddings
- **Retrieval Top-K**: Configurable similarity threshold to filter low-quality results
- **API Efficiency**: Batch queries to minimize API calls

## Testing

Run the test suite:
```bash
pytest tests/
```

## Core Responsibilities Demonstrated

✓ **PostgreSQL + pgvector**: Vector database for embeddings storage and retrieval
✓ **Vector Similarity Search**: IVFFlat-indexed cosine similarity for fast retrieval
✓ **LLM Integration**: Claude API for embedding generation and answer generation
✓ **Evaluation Framework**: Quantified RAG improvement vs. baseline
✓ **Production Ready**: Indexing, connection management, error handling

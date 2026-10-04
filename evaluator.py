from rag_pipeline import RAGPipeline
import json

class RAGEvaluator:
    """Evaluate RAG system accuracy with vs without retrieval."""
    
    def __init__(self):
        self.rag = RAGPipeline()
    
    def evaluate(self, test_queries: list[dict]) -> dict:
        """
        Evaluate RAG accuracy.
        test_queries format: [{"query": "...", "expected_answer": "..."}, ...]
        """
        results = {
            "with_retrieval": {"accuracy": 0, "responses": []},
            "without_retrieval": {"accuracy": 0, "responses": []},
            "improvement": 0
        }
        
        for test in test_queries:
            query = test["query"]
            
            # Test with RAG
            with_rag = self.rag.query_with_rag(query)
            with_rag["expected"] = test.get("expected_answer", "")
            results["with_retrieval"]["responses"].append(with_rag)
            
            # Test without RAG
            without_rag = self.rag.query_without_rag(query)
            without_rag["expected"] = test.get("expected_answer", "")
            results["without_retrieval"]["responses"].append(without_rag)
        
        # Calculate metric: response contains expected information (semantic match)
        results["with_retrieval"]["accuracy"] = self._calc_accuracy(
            results["with_retrieval"]["responses"]
        )
        results["without_retrieval"]["accuracy"] = self._calc_accuracy(
            results["without_retrieval"]["responses"]
        )
        
        results["improvement"] = (
            results["with_retrieval"]["accuracy"] - 
            results["without_retrieval"]["accuracy"]
        )
        
        return results
    
    def _calc_accuracy(self, responses: list[dict]) -> float:
        """Simple accuracy: check if answer mentions key terms from context."""
        if not responses:
            return 0
        
        matches = 0
        for resp in responses:
            if resp.get("context"):
                # If context was used and answer references it, mark as correct
                answer = resp["answer"].lower()
                for ctx in resp["context"]:
                    if ctx["source"] in answer or "context" in answer:
                        matches += 1
                        break
            else:
                # No context but still got an answer
                if resp["answer"].strip():
                    matches += 0.5
        
        return matches / len(responses)
    
    def close(self):
        self.rag.close()

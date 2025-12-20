#!/usr/bin/env python
"""
RAG Pipeline Validation Script

Validates:
1. ChromaDB schema compatibility
2. Embedding alignment
3. Retrieval quality (recall, ranking)
4. Generator context consumption
5. End-to-end pipeline performance
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class RetrievalTestCase:
    """A test case for retrieval validation."""

    query: str
    expected_sources: list[str]  # Expected source file patterns
    query_type: str  # text, code, hybrid
    min_relevance: float = 0.3
    description: str = ""


@dataclass
class ValidationResult:
    """Result of a validation test."""

    name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class RAGPipelineValidator:
    """Validates the RAG pipeline end-to-end."""

    def __init__(self):
        """Initialize validator with RAG components."""
        from ordinis.rag.config import get_config
        from ordinis.rag.embedders.text_embedder import TextEmbedder
        from ordinis.rag.retrieval.engine import RetrievalEngine
        from ordinis.rag.vectordb.chroma_client import ChromaClient

        self.config = get_config()
        self.chroma_client = ChromaClient()
        self.text_embedder = TextEmbedder()
        self.retrieval_engine = RetrievalEngine(
            chroma_client=self.chroma_client,
            text_embedder=self.text_embedder,
        )

        # Reuse ChromaDB client from ChromaClient for schema inspection
        self.raw_client = self.chroma_client.client

        self.results: list[ValidationResult] = []

    def validate_all(self) -> dict[str, Any]:
        """Run all validation tests."""
        print("=" * 70)
        print("RAG PIPELINE VALIDATION")
        print("=" * 70)

        # 1. Schema compatibility
        print("\n[1/5] Validating ChromaDB schema compatibility...")
        self._validate_schema_compatibility()

        # 2. Embedding alignment
        print("\n[2/5] Validating embedding alignment...")
        self._validate_embedding_alignment()

        # 3. Retrieval quality
        print("\n[3/5] Validating retrieval quality...")
        self._validate_retrieval_quality()

        # 4. Generator context
        print("\n[4/5] Validating generator context consumption...")
        self._validate_generator_context()

        # 5. End-to-end pipeline
        print("\n[5/5] Validating end-to-end pipeline...")
        self._validate_end_to_end()

        # Summary
        return self._generate_summary()

    def _validate_schema_compatibility(self):
        """Validate ChromaDB schema is compatible with retriever."""
        try:
            collections = self.raw_client.list_collections()

            schema_issues = []
            collection_stats = {}

            for col in collections:
                name = col.name
                count = col.count()
                metadata = col.metadata or {}

                collection_stats[name] = {
                    "count": count,
                    "metadata": metadata,
                }

                if count == 0:
                    schema_issues.append(f"{name}: empty collection")
                    continue

                # Sample documents to check schema
                sample = col.get(limit=10, include=["metadatas", "embeddings"])

                if sample["metadatas"]:
                    meta_keys = set()
                    for m in sample["metadatas"]:
                        if m:
                            meta_keys.update(m.keys())
                    collection_stats[name]["metadata_keys"] = list(meta_keys)

                # Check embedding dimensions
                embeddings = sample.get("embeddings")
                if embeddings is not None and len(embeddings) > 0:
                    dim = len(embeddings[0])
                    collection_stats[name]["embedding_dim"] = dim

                    # Check consistency
                    if dim != self.config.text_embedding_dimension:
                        schema_issues.append(
                            f"{name}: embedding dim {dim} != config {self.config.text_embedding_dimension}"
                        )

            passed = len(schema_issues) == 0
            self.results.append(
                ValidationResult(
                    name="schema_compatibility",
                    passed=passed,
                    details={
                        "collections": collection_stats,
                        "issues": schema_issues,
                    },
                )
            )

            status = "PASS" if passed else "WARN"
            print(f"  Schema compatibility: {status}")
            for name, stats in collection_stats.items():
                print(f"    {name}: {stats['count']} docs, dim={stats.get('embedding_dim', 'N/A')}")
            if schema_issues:
                for issue in schema_issues:
                    print(f"    [ISSUE] {issue}")

        except Exception as e:
            self.results.append(
                ValidationResult(
                    name="schema_compatibility",
                    passed=False,
                    error=str(e),
                )
            )
            print(f"  Schema compatibility: FAIL - {e}")

    def _validate_embedding_alignment(self):
        """Validate embeddings are properly aligned with model."""
        try:
            # Test embedding generation
            test_texts = [
                "momentum trading strategy",
                "risk management position sizing",
                "def calculate_sharpe_ratio(returns):",
            ]

            embeddings = self.text_embedder.embed(test_texts)

            issues = []

            # Check dimension
            if embeddings.shape[1] != self.config.text_embedding_dimension:
                issues.append(
                    f"Embedding dim {embeddings.shape[1]} != config {self.config.text_embedding_dimension}"
                )

            # Check normalization (informational - L2 distance doesn't require normalization)
            import numpy as np

            norms = np.linalg.norm(embeddings, axis=1)
            is_normalized = np.allclose(norms, 1.0, atol=0.1)

            # Check embedding model availability
            model_available = self.text_embedder.is_available()

            # Pass if dimension is correct and model is available
            # Normalization is informational only (L2 distance works with unnormalized)
            passed = len(issues) == 0 and model_available
            self.results.append(
                ValidationResult(
                    name="embedding_alignment",
                    passed=passed,
                    details={
                        "embedding_dim": int(embeddings.shape[1]),
                        "model_available": model_available,
                        "sample_norms": norms.tolist(),
                        "issues": issues,
                    },
                )
            )

            status = "PASS" if passed else "FAIL"
            print(f"  Embedding alignment: {status}")
            print(f"    Dimension: {embeddings.shape[1]}")
            print(f"    Model available: {model_available}")
            print(f"    Normalized: {is_normalized}")

        except Exception as e:
            self.results.append(
                ValidationResult(
                    name="embedding_alignment",
                    passed=False,
                    error=str(e),
                )
            )
            print(f"  Embedding alignment: FAIL - {e}")

    def _validate_retrieval_quality(self):
        """Validate retrieval quality with test queries."""
        test_cases = [
            RetrievalTestCase(
                query="momentum trading strategy breakout",
                expected_sources=["momentum", "breakout", "strategy"],
                query_type="text",
                description="KB query for momentum strategies",
            ),
            RetrievalTestCase(
                query="risk management position sizing volatility",
                expected_sources=["risk", "position", "volatility"],
                query_type="text",
                description="KB query for risk management",
            ),
            RetrievalTestCase(
                query="technical analysis RSI MACD indicators",
                expected_sources=["technical", "rsi", "macd", "indicator"],
                query_type="text",
                description="KB query for technical indicators",
            ),
            RetrievalTestCase(
                query="def calculate returns sharpe ratio",
                expected_sources=["sharpe", "returns", "calculate"],
                query_type="code",
                description="Code query for Sharpe ratio",
            ),
            RetrievalTestCase(
                query="backtest simulation walk forward optimization",
                expected_sources=["backtest", "walk", "forward", "simulation"],
                query_type="text",
                description="KB query for backtesting",
            ),
        ]

        results_detail = []
        total_recall = 0
        total_mrr = 0  # Mean Reciprocal Rank

        for tc in test_cases:
            try:
                start = time.perf_counter()
                response = self.retrieval_engine.query(
                    query=tc.query,
                    query_type=tc.query_type,
                    top_k=10,
                )
                latency_ms = (time.perf_counter() - start) * 1000

                # Calculate recall
                retrieved_text = " ".join([r.text.lower() for r in response.results])
                hits = sum(1 for exp in tc.expected_sources if exp.lower() in retrieved_text)
                recall = hits / len(tc.expected_sources) if tc.expected_sources else 0

                # Calculate MRR (position of first relevant result)
                mrr = 0
                for i, r in enumerate(response.results, 1):
                    if any(exp.lower() in r.text.lower() for exp in tc.expected_sources):
                        mrr = 1.0 / i
                        break

                # Check relevance scores
                avg_score = (
                    sum(r.score for r in response.results) / len(response.results)
                    if response.results
                    else 0
                )
                min_score = min(r.score for r in response.results) if response.results else 0

                result = {
                    "query": tc.query,
                    "description": tc.description,
                    "num_results": len(response.results),
                    "recall": recall,
                    "mrr": mrr,
                    "avg_score": avg_score,
                    "min_score": min_score,
                    "latency_ms": latency_ms,
                    "passed": recall >= 0.5 and len(response.results) > 0,
                }
                results_detail.append(result)

                total_recall += recall
                total_mrr += mrr

                status = "PASS" if result["passed"] else "FAIL"
                print(f"    [{status}] {tc.description}")
                print(
                    f"         Recall: {recall:.2f}, MRR: {mrr:.2f}, Results: {len(response.results)}, Latency: {latency_ms:.0f}ms"
                )

            except Exception as e:
                results_detail.append(
                    {
                        "query": tc.query,
                        "error": str(e),
                        "passed": False,
                    }
                )
                print(f"    [FAIL] {tc.description}: {e}")

        avg_recall = total_recall / len(test_cases) if test_cases else 0
        avg_mrr = total_mrr / len(test_cases) if test_cases else 0
        passed_count = sum(1 for r in results_detail if r.get("passed", False))

        overall_passed = passed_count >= len(test_cases) * 0.6  # 60% pass rate

        self.results.append(
            ValidationResult(
                name="retrieval_quality",
                passed=overall_passed,
                details={
                    "test_cases": results_detail,
                    "avg_recall": avg_recall,
                    "avg_mrr": avg_mrr,
                    "passed_count": passed_count,
                    "total_count": len(test_cases),
                },
            )
        )

        print(f"\n  Retrieval quality summary:")
        print(f"    Tests passed: {passed_count}/{len(test_cases)}")
        print(f"    Average recall: {avg_recall:.2f}")
        print(f"    Average MRR: {avg_mrr:.2f}")

    def _validate_generator_context(self):
        """Validate that generator correctly consumes retrieved context."""
        try:
            from ordinis.engines.cortex.rag.integration import CortexRAGHelper

            rag_helper = CortexRAGHelper()

            # Test context retrieval
            test_queries = [
                ("momentum strategy market regime", "text"),
                ("calculate portfolio risk metrics", "code"),
            ]

            context_issues = []
            context_samples = []

            for query, qtype in test_queries:
                if qtype == "text":
                    context = rag_helper.get_kb_context(query, top_k=5)
                else:
                    context = rag_helper.get_code_examples(query, top_k=3)

                sample = {
                    "query": query,
                    "type": qtype,
                    "context_length": len(context),
                    "has_content": len(context) > 0,
                    "preview": context[:200] if context else "EMPTY",
                }
                context_samples.append(sample)

                # Check for issues
                if not context:
                    context_issues.append(f"Empty context for '{query}'")
                elif len(context) < 100:
                    context_issues.append(
                        f"Very short context ({len(context)} chars) for '{query}'"
                    )

            passed = len(context_issues) == 0
            self.results.append(
                ValidationResult(
                    name="generator_context",
                    passed=passed,
                    details={
                        "samples": context_samples,
                        "issues": context_issues,
                    },
                )
            )

            status = "PASS" if passed else "WARN"
            print(f"  Generator context: {status}")
            for sample in context_samples:
                print(f"    {sample['type']} query: {sample['context_length']} chars")
            if context_issues:
                for issue in context_issues:
                    print(f"    [ISSUE] {issue}")

        except ImportError as e:
            self.results.append(
                ValidationResult(
                    name="generator_context",
                    passed=False,
                    error=f"CortexRAGHelper not available: {e}",
                )
            )
            print(f"  Generator context: SKIP - CortexRAGHelper not available")
        except Exception as e:
            self.results.append(
                ValidationResult(
                    name="generator_context",
                    passed=False,
                    error=str(e),
                )
            )
            print(f"  Generator context: FAIL - {e}")

    def _validate_end_to_end(self):
        """Validate end-to-end pipeline performance."""
        try:
            # Run a complex query through the full pipeline
            complex_queries = [
                "How do I implement a momentum breakout strategy with proper risk management?",
                "What are the best practices for backtesting and avoiding overfitting?",
                "Show me code examples for calculating technical indicators",
            ]

            e2e_results = []
            total_latency = 0

            for query in complex_queries:
                start = time.perf_counter()

                # Full pipeline query
                response = self.retrieval_engine.query(
                    query=query,
                    query_type=None,  # Auto-detect
                    top_k=5,
                )

                latency_ms = (time.perf_counter() - start) * 1000
                total_latency += latency_ms

                result = {
                    "query": query[:50] + "...",
                    "results": len(response.results),
                    "query_type": response.query_type,
                    "latency_ms": latency_ms,
                    "top_score": response.results[0].score if response.results else 0,
                    "passed": len(response.results) > 0 and latency_ms < 5000,
                }
                e2e_results.append(result)

                status = "PASS" if result["passed"] else "FAIL"
                print(f"    [{status}] {result['query']}")
                print(
                    f"         Type: {result['query_type']}, Results: {result['results']}, Latency: {latency_ms:.0f}ms"
                )

            avg_latency = total_latency / len(complex_queries) if complex_queries else 0
            passed_count = sum(1 for r in e2e_results if r["passed"])
            overall_passed = passed_count == len(complex_queries)

            self.results.append(
                ValidationResult(
                    name="end_to_end",
                    passed=overall_passed,
                    details={
                        "queries": e2e_results,
                        "avg_latency_ms": avg_latency,
                        "passed_count": passed_count,
                        "total_count": len(complex_queries),
                    },
                )
            )

            print(f"\n  End-to-end summary:")
            print(f"    Tests passed: {passed_count}/{len(complex_queries)}")
            print(f"    Average latency: {avg_latency:.0f}ms")

        except Exception as e:
            self.results.append(
                ValidationResult(
                    name="end_to_end",
                    passed=False,
                    error=str(e),
                )
            )
            print(f"  End-to-end: FAIL - {e}")

    def _generate_summary(self) -> dict[str, Any]:
        """Generate validation summary."""
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)

        passed_tests = sum(1 for r in self.results if r.passed)
        total_tests = len(self.results)

        print(f"\nTests passed: {passed_tests}/{total_tests}")

        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            print(f"  [{status}] {result.name}")
            if result.error:
                print(f"         Error: {result.error}")

        # Overall status
        overall_passed = passed_tests >= total_tests * 0.8  # 80% pass rate
        status = "PRODUCTION READY" if overall_passed else "NEEDS ATTENTION"
        print(f"\nOverall status: {status}")

        return {
            "overall_passed": overall_passed,
            "passed_count": passed_tests,
            "total_count": total_tests,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "details": r.details,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


def main():
    """Run RAG pipeline validation."""
    validator = RAGPipelineValidator()
    summary = validator.validate_all()

    # Save report
    report_path = PROJECT_ROOT / "artifacts" / "reports" / "rag_validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nReport saved to: {report_path}")

    return 0 if summary["overall_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

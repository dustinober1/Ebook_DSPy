# Multi-hop Search: Complex Reasoning Across Documents

## Introduction

Many real-world questions cannot be answered with a single document retrieval. They require multi-hop reasoning—finding information from multiple sources and connecting the dots to arrive at a comprehensive answer. Multi-hop search systems excel at answering complex questions that involve relationships, comparisons, and synthesizing information across different documents.

## Understanding Multi-hop Search

### What is Multi-hop Reasoning?

Multi-hop reasoning involves:
- **First hop**: Initial information retrieval
- **Intermediate hops**: Finding related information based on previous results
- **Final hop**: Synthesizing all information to answer the original question

### Example Scenarios

1. **Comparative Questions**: "How does the cost of living in San Francisco compare to Austin?"
2. **Chain Questions**: "Which companies did the founders of Google work for before starting Google?"
3. **Aggregation Questions**: "What is the total market cap of all tech companies founded after 2010?"
4. **Causal Questions**: "What factors led to the 2008 financial crisis and how did it affect the housing market?"

## Building Multi-hop Systems

### Basic Multi-hop Architecture

```python
import dspy

class MultiHopSearch(dspy.Module):
    def __init__(self, max_hops=3):
        super().__init__()
        self.max_hops = max_hops
        self.retrieve = dspy.Retrieve(k=5)
        self.generate_query = dspy.Predict("question, context -> next_query")
        self.generate_answer = dspy.ChainOfThought("question, all_contexts -> answer")

    def forward(self, question):
        all_contexts = []
        current_question = question

        for hop in range(self.max_hops):
            # Retrieve documents for current query
            retrieved = self.retrieve(question=current_question)
            contexts = retrieved.passages
            all_contexts.extend(contexts)

            # Generate next query based on retrieved information
            next_query_result = self.generate_query(
                question=question,
                context="\n".join(contexts)
            )

            # Check if we have enough information
            if "sufficient" in next_query_result.next_query.lower():
                break

            current_question = next_query_result.next_query

        # Generate final answer using all retrieved contexts
        final_answer = self.generate_answer(
            question=question,
            all_contexts="\n\n".join(all_contexts)
        )

        return dspy.Prediction(
            answer=final_answer.answer,
            contexts=all_contexts,
            hops=hop + 1,
            reasoning=final_answer.rationale
        )
```

### Advanced Multi-hop with Question Decomposition

```python
class DecomposedMultiHop(dspy.Module):
    def __init__(self):
        super().__init__()
        self.decompose = dspy.Predict("question -> subquestions")
        self.retrieve = dspy.Retrieve(k=5)
        self.answer_subquestion = dspy.Predict("subquestion, context -> subanswer")
        self.synthesize = dspy.ChainOfThought("question, subanswers -> final_answer")

    def forward(self, question):
        # Decompose the complex question
        decomposition = self.decompose(question=question)
        subquestions = decomposition.subquestions.split(";")

        subanswers = []

        # Answer each subquestion
        for subq in subquestions:
            subq = subq.strip()
            if subq:
                # Retrieve relevant context
                context = self.retrieve(question=subq).passages

                # Answer subquestion
                subanswer = self.answer_subquestion(
                    subquestion=subq,
                    context="\n".join(context)
                )

                subanswers.append({
                    "subquestion": subq,
                    "answer": subanswer.subanswer,
                    "context": context
                })

        # Synthesize final answer
        subanswers_text = "\n".join([
            f"Q: {sa['subquestion']}\nA: {sa['answer']}"
            for sa in subanswers
        ])

        synthesis = self.synthesize(
            question=question,
            subanswers=subanswers_text
        )

        return dspy.Prediction(
            answer=synthesis.answer,
            subquestions=subquestions,
            subanswers=subanswers,
            reasoning=synthesis.rationale
        )
```

### Graph-based Multi-hop Search

```python
class GraphMultiHop(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=5)
        self.extract_entities = dspy.Predict("text -> entities")
        self.find_connections = dspy.Predict("entities, question -> search_queries")
        self.traverse_graph = dspy.ChainOfThought("question, entities, paths -> answer")

    def forward(self, question):
        visited_entities = set()
        all_paths = []
        max_depth = 3

        # Initial retrieval
        initial_docs = self.retrieve(question=question).passages

        # Extract entities from initial documents
        for doc in initial_docs:
            entities_result = self.extract_entities(text=doc)
            entities = [e.strip() for e in entities_result.entities.split(",")]

            for entity in entities:
                if entity not in visited_entities and len(visited_entities) < 20:
                    visited_entities.add(entity)

                    # Find connections to other entities
                    connections = self.find_connections(
                        entities=", ".join(visited_entities),
                        question=question
                    )

                    # Search for each connection
                    for query in connections.search_queries.split(";"):
                        query = query.strip()
                        if query:
                            related_docs = self.retrieve(question=query).passages
                            all_paths.extend(related_docs)

        # Traverse the graph of connections
        traversal = self.traverse_graph(
            question=question,
            entities=", ".join(list(visited_entities)),
            paths="\n".join(all_paths[:20])
        )

        return dspy.Prediction(
            answer=traversal.answer,
            entities=list(visited_entities),
            paths=all_paths,
            reasoning=traversal.rationale
        )
```

## Specialized Multi-hop Applications

### Research Assistant for Academic Papers

```python
class AcademicResearcher(dspy.Module):
    def __init__(self, paper_collection):
        super().__init__()
        self.papers = paper_collection
        self.retrieve = dspy.Retrieve(k=5)
        self.find_related = dspy.Predict("paper -> related_topics, key_authors, citations")
        self.synthesize_research = dspy.ChainOfThought(
            "question, papers, relationships -> comprehensive_answer, references"
        )

    def forward(self, research_question):
        # Initial search for relevant papers
        initial_papers = self.retrieve(question=research_question).passages

        # Find related papers through citations and authors
        all_papers = []
        relationships = []

        for paper in initial_papers:
            related = self.find_related(paper=paper)
            relationships.append({
                "paper": paper,
                "topics": related.related_topics,
                "authors": related.key_authors,
                "citations": related.citations
            })

            # Search for related papers
            for topic in related.related_topics.split(","):
                topic_papers = self.retrieve(question=topic.strip()).passages
                all_papers.extend(topic_papers)

        for author in related.key_authors.split(","):
            author_papers = self.retrieve(question=f"papers by {author.strip()}").passages
            all_papers.extend(author_papers)

        # Synthesize comprehensive answer
        synthesis = self.synthesize_research(
            question=research_question,
            papers="\n\n".join(list(set(all_papers))),
            relationships="\n".join([str(r) for r in relationships])
        )

        return dspy.Prediction(
            answer=synthesis.comprehensive_answer,
            references=synthesis.references,
            papers_used=list(set(all_papers)),
            relationships=relationships
        )
```

### Fact Verification System

```python
class FactChecker(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=5)
        self.extract_claims = dspy.Predict("statement -> claims")
        self.verify_claim = dspy.Predict("claim, evidence -> verification, confidence")
        self.find_contradictions = dspy.Predict("verified_claims -> contradictions")
        self.final_judgment = dspy.ChainOfThought(
            "statement, verifications, contradictions -> final_verdict, explanation"
        )

    def forward(self, statement):
        # Extract individual claims from the statement
        claims_result = self.extract_claims(statement=statement)
        claims = [c.strip() for c in claims_result.claims.split(".")]

        verifications = []

        # Verify each claim
        for claim in claims:
            if claim:
                # Search for evidence
                evidence = self.retrieve(question=claim).passages

                # Verify the claim with evidence
                verification = self.verify_claim(
                    claim=claim,
                    evidence="\n".join(evidence)
                )

                verifications.append({
                    "claim": claim,
                    "verification": verification.verification,
                    "confidence": verification.confidence,
                    "evidence": evidence
                })

        # Check for contradictions between verified claims
        contradictions = self.find_contradictions(
            verified_claims="\n".join([f"{v['claim']}: {v['verification']}" for v in verifications])
        )

        # Make final judgment
        judgment = self.final_judgment(
            statement=statement,
            verifications="\n".join([str(v) for v in verifications]),
            contradictions=contradictions.contradictions
        )

        return dspy.Prediction(
            verdict=judgment.final_verdict,
            explanation=judgment.explanation,
            claims=verifications,
            contradictions=contradictions.contradictions,
            reasoning=judgment.rationale
        )
```

### Supply Chain Analysis

```python
class SupplyChainAnalyzer(dspy.Module):
    def __init__(self, company_data):
        super().__init__()
        self.company_data = company_data
        self.retrieve = dspy.Retrieve(k=5)
        self.trace_supplier = dspy.Predict("company -> suppliers, locations, risks")
        self.analyze_dependencies = dspy.Predict("suppliers, locations -> dependencies")
        self.assess_risk = dspy.ChainOfThought("company, suppliers, dependencies -> risk_analysis")

    def forward(self, company_name):
        # Find company information
        company_info = self.retrieve(question=company_name).passages

        # Trace suppliers
        supplier_info = self.trace_supplier(company=company_name)
        suppliers = [s.strip() for s in supplier_info.suppliers.split(",")]

        all_suppliers = []
        dependencies = []

        # For each supplier, find their suppliers (second hop)
        for supplier in suppliers[:5]:  # Limit to prevent explosion
            supplier_data = self.retrieve(question=f"{supplier} suppliers").passages
            all_suppliers.append({
                "name": supplier,
                "data": supplier_data,
                "location": supplier_info.locations
            })

            # Analyze dependencies
            dependency = self.analyze_dependencies(
                suppliers=supplier,
                locations=supplier_info.locations
            )
            dependencies.append(dependency)

        # Assess overall supply chain risk
        risk_assessment = self.assess_risk(
            company=company_name,
            suppliers=str(all_suppliers),
            dependencies=str(dependencies)
        )

        return dspy.Prediction(
            company=company_name,
            suppliers=all_suppliers,
            dependencies=dependencies,
            risk_analysis=risk_assessment.risk_analysis,
            reasoning=risk_assessment.rationale
        )
```

## Optimizing Multi-hop Systems

### Using MIPRO for Complex Queries

```python
class OptimizedMultiHop(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=5)
        self.determine_strategy = dspy.Predict("question -> search_strategy, hops_needed")
        self.execute_hop = dspy.Predict("query, previous_context -> next_query, extracted_info")
        self.final_synthesis = dspy.ChainOfThought("question, hop_results -> answer")

    def forward(self, question):
        # Determine search strategy
        strategy = self.determine_strategy(question=question)

        hop_results = []
        current_query = question
        previous_context = ""

        for hop in range(int(strategy.hops_needed)):
            # Execute current hop
            hop_result = self.execute_hop(
                query=current_query,
                previous_context=previous_context
            )

            # Retrieve documents for next query
            documents = self.retrieve(question=hop_result.next_query).passages

            hop_results.append({
                "hop": hop + 1,
                "query": hop_result.next_query,
                "extracted_info": hop_result.extracted_info,
                "documents": documents
            })

            # Update for next iteration
            current_query = hop_result.next_query
            previous_context = hop_result.extracted_info

        # Synthesize final answer
        synthesis = self.final_synthesis(
            question=question,
            hop_results=str(hop_results)
        )

        return dspy.Prediction(
            answer=synthesis.answer,
            strategy=strategy.search_strategy,
            hops=hop_results,
            reasoning=synthesis.rationale
        )

# Training data for optimization
multi_hop_trainset = [
    dspy.Example(
        question="What is the relationship between quantum computing and cryptography?",
        strategy="trace_technology_relationships",
        hops_needed=3,
        answer="Quantum computing threatens current cryptographic systems while also enabling quantum-resistant cryptography solutions."
    ),
    # ... more complex examples
]

# Optimize with MIPRO
mipro_optimizer = MIPRO(
    metric=multi_hop_metric,
    num_candidates=10
)
optimized_multihop = mipro_optimizer.compile(OptimizedMultiHop(), trainset=multi_hop_trainset)
```

### Custom Evaluation Metric for Multi-hop

```python
def multi_hop_metric(example, pred, trace=None):
    """Evaluate multi-hop reasoning quality."""
    score = 0

    # Check if answer is correct
    if hasattr(pred, 'answer'):
        answer_quality = evaluate_answer_relevance(
            pred.answer,
            example.question,
            example.expected_answer
        )
        score += 0.4 * answer_quality

    # Check reasoning depth
    if hasattr(pred, 'hops'):
        expected_hops = example.get('expected_hops', 2)
        hop_score = min(len(pred.hops) / expected_hops, 1.0)
        score += 0.2 * hop_score

    # Check if information is properly synthesized
    if hasattr(pred, 'reasoning'):
        synthesis_quality = evaluate_synthesis_quality(
            pred.reasoning,
            pred.answer
        )
        score += 0.2 * synthesis_quality

    # Check if appropriate strategy was used
    if hasattr(pred, 'strategy'):
        strategy_match = evaluate_strategy_appropriateness(
            pred.strategy,
            example.question
        )
        score += 0.2 * strategy_match

    return score
```

## Advanced Retrieval Techniques

### Fused Retrieval for Multi-Source Information

Fused retrieval combines multiple retrieval strategies to improve coverage and accuracy. This is especially useful for complex multi-hop queries where information may be scattered across different sources.

```python
from dspy.retrieve import Retrieve
import numpy as np

class FusedRetriever(dspy.Module):
    def __init__(self, retrievers=None):
        super().__init__()
        # Multiple retrievers with different strategies
        self.retrievers = retrievers or [
            dspy.Retrieve(k=5, collection_type="dense"),     # Dense retrieval
            dspy.Retrieve(k=5, collection_type="sparse"),    # Sparse retrieval
            dspy.Retrieve(k=5, collection_type="hybrid")     # Hybrid approach
        ]
        self.fuse = dspy.ChainOfThought("multiple_results -> fused_results")

    def forward(self, query):
        all_results = []

        # Retrieve from multiple sources
        for retriever in self.retrievers:
            results = retriever(query=query).passages
            all_results.extend([(result, retriever.collection_type) for result in results])

        # Remove duplicates while preserving source information
        unique_results = self._deduplicate_with_sources(all_results)

        # Fuse results based on relevance and diversity
        fused_input = "\n\n".join([
            f"[{source}]: {doc}" for doc, source in unique_results[:20]
        ])

        fusion = self.fuse(multiple_results=fused_input)

        return dspy.Prediction(
            passages=fusion.fused_results.split("\n"),
            sources=[source for _, source in unique_results[:20]],
            all_raw_results=all_results
        )

    def _deduplicate_with_sources(self, results):
        """Remove duplicate documents while tracking sources"""
        seen = set()
        unique = []
        for doc, source in results:
            # Simple deduplication based on document hash
            doc_hash = hash(doc[:100])  # First 100 chars as signature
            if doc_hash not in seen:
                seen.add(doc_hash)
                unique.append((doc, source))
        return unique
```

### Dynamic Termination Strategies

Instead of using a fixed number of hops, implement smart termination based on information completeness:

```python
class DynamicTerminationSearch(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=5)
        self.check_completeness = dspy.Predict(
            "question, gathered_info, current_answer -> completeness_score, confidence, should_continue"
        )
        self.generate_next_query = dspy.Predict(
            "question, gaps_in_info -> next_search_query"
        )
        self.synthesize = dspy.ChainOfThought("question, all_info -> final_answer")

    def forward(self, question, max_hops=5):
        all_info = []
        info_summary = ""

        for hop in range(max_hops):
            # Check if we have sufficient information
            if hop > 0:
                current_answer = self.synthesize(
                    question=question,
                    all_info="\n".join(all_info)
                )

                completeness = self.check_completeness(
                    question=question,
                    gathered_info="\n".join(all_info),
                    current_answer=current_answer.answer
                )

                # Dynamic termination based on confidence and completeness
                if (float(completeness.confidence) > 0.85 and
                    float(completeness.completeness_score) > 0.8):
                    break

                if completeness.should_continue.lower() == "no":
                    break

                # Generate targeted next query based on information gaps
                next_query = self.generate_next_query(
                    question=question,
                    gaps_in_info=completeness.should_continue
                )
                search_query = next_query.next_search_query
            else:
                search_query = question

            # Retrieve new information
            results = self.retrieve(question=search_query).passages
            all_info.extend([f"[Hop {hop+1}]: {doc}" for doc in results])

        # Generate final comprehensive answer
        final = self.synthesize(
            question=question,
            all_info="\n\n".join(all_info)
        )

        return dspy.Prediction(
            answer=final.answer,
            hops_used=hop + 1,
            information_gathered=all_info,
            termination_reason="complete" if hop < max_hops - 1 else "max_hops"
        )
```

### Task-Aware Search Query Formulation

Generate search queries that are specifically tailored to the task requirements:

```python
class TaskAwareSearch(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyze_task = dspy.Predict(
            "question -> task_type, required_info, search_strategy"
        )
        self.formulate_query = dspy.Predict(
            "task_type, required_info, previous_results -> optimized_query"
        )
        self.retrieve = dspy.Retrieve(k=5)
        self.evaluate_relevance = dspy.Predict(
            "document, task_requirements -> relevance_score, key_points"
        )

    def forward(self, question):
        # Analyze the task requirements
        task_analysis = self.analyze_task(question=question)

        # Task-specific retrieval strategies
        if "comparison" in task_analysis.task_type.lower():
            return self._comparison_search(question, task_analysis)
        elif "causal" in task_analysis.task_type.lower():
            return self._causal_search(question, task_analysis)
        elif "temporal" in task_analysis.task_type.lower():
            return self._temporal_search(question, task_analysis)
        else:
            return self._general_search(question, task_analysis)

    def _comparison_search(self, question, task_analysis):
        """Specialized search for comparison questions"""
        all_results = []
        entities = self._extract_entities(question)

        # Search for each entity
        for entity in entities:
            entity_query = f"{entity} {task_analysis.required_info}"
            results = self.retrieve(question=entity_query).passages
            all_results.extend(results)

        # Search for direct comparisons
        comparison_query = f"compare {' vs '.join(entities)} {task_analysis.required_info}"
        comparison_results = self.retrieve(question=comparison_query).passages
        all_results.extend(comparison_results)

        return self._process_results(question, all_results, "comparison")

    def _causal_search(self, question, task_analysis):
        """Specialized search for causal reasoning"""
        all_results = []

        # Search for causes
        cause_query = f"causes of {task_analysis.required_info}"
        cause_results = self.retrieve(question=cause_query).passages
        all_results.extend(cause_results)

        # Search for effects
        effect_query = f"effects of {task_analysis.required_info}"
        effect_results = self.retrieve(question=effect_query).passages
        all_results.extend(effect_results)

        # Search for mechanisms
        mechanism_query = f"mechanism {task_analysis.required_info}"
        mechanism_results = self.retrieve(question=mechanism_query).passages
        all_results.extend(mechanism_results)

        return self._process_results(question, all_results, "causal")

    def _extract_entities(self, text):
        """Simple entity extraction for task-aware search"""
        # In practice, you might use NER here
        words = text.split()
        entities = [w for w in words if w[0].isupper() and len(w) > 2]
        return entities[:3]  # Limit to avoid explosion
```

### Efficiency Optimizations for Large-Scale Retrieval

```python
class EfficientMultiHop(dspy.Module):
    def __init__(self, cache_size=1000, batch_size=10):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=3)  # Smaller k for efficiency
        self.cache = {}  # Simple cache for repeated queries
        self.cache_size = cache_size
        self.batch_size = batch_size

        # Pre-compute query embeddings for similarity search
        self.query_encoder = None  # Initialize with your encoder
        self.query_cache = {}

    def forward(self, question, max_hops=3):
        # Check cache first
        cache_key = hash(question)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Batch similar queries
        query_batch = self._batch_similar_queries(question)

        # Execute batch retrieval
        all_info = []
        for query in query_batch:
            results = self.retrieve(question=query).passages
            all_info.extend(results)

        # Apply early termination based on information saturation
        unique_info = self._deduplicate(all_info)
        if self._is_saturated(unique_info):
            max_hops = min(max_hops, 2)  # Reduce hops if we have enough info

        # Continue with standard multi-hop if needed
        for hop in range(max_hops):
            # Use progressive query refinement
            next_query = self._refine_query(question, unique_info)
            if next_query == question:  # No refinement needed
                break

            results = self.retrieve(question=next_query).passages
            unique_info.extend(self._deduplicate(results))

            # Check for early stopping
            if self._check_early_stop(unique_info, question):
                break

        result = dspy.Prediction(
            answer=self._synthesize_answer(question, unique_info),
            info_gathered=unique_info,
            queries_used=query_batch
        )

        # Cache the result
        if len(self.cache) < self.cache_size:
            self.cache[cache_key] = result

        return result

    def _batch_similar_queries(self, query):
        """Group similar queries for batch processing"""
        # Simple implementation - in practice, use semantic similarity
        variants = [
            query,
            f"details about {query}",
            f"information on {query}",
            query.replace("What", "How"),
            query.replace("Why", "What causes")
        ]
        return variants[:self.batch_size]

    def _deduplicate(self, documents):
        """Remove duplicate documents efficiently"""
        seen = set()
        unique = []
        for doc in documents:
            doc_hash = hash(doc[:50])  # First 50 chars
            if doc_hash not in seen:
                seen.add(doc_hash)
                unique.append(doc)
        return unique

    def _is_saturated(self, documents):
        """Check if we have enough diverse information"""
        if len(documents) < 10:
            return False

        # Simple heuristic: check for keyword overlap
        all_words = set()
        for doc in documents[-5:]:  # Last 5 docs
            all_words.update(doc.lower().split()[:20])

        # If we have many unique words, we're getting diverse info
        return len(all_words) > 100

    def _check_early_stop(self, documents, question):
        """Intelligent early stopping based on question coverage"""
        # Extract question keywords
        question_words = set(question.lower().split())

        # Check recent documents for question coverage
        recent_docs = " ".join(documents[-3:]).lower()
        coverage = sum(1 for word in question_words if word in recent_docs)

        # Stop if we have good coverage
        return coverage / len(question_words) > 0.7
```

### Integration with Vector Databases

```python
class VectorDBRetriever(dspy.Module):
    def __init__(self, vector_db_client):
        super().__init__()
        self.db = vector_db_client
        self.retrieve = dspy.Retrieve(k=5)
        self.embed = dspy.Predict("text -> embedding")

    def forward(self, query, search_mode="hybrid"):
        if search_mode == "semantic":
            return self._semantic_search(query)
        elif search_mode == "keyword":
            return self._keyword_search(query)
        else:
            return self._hybrid_search(query)

    def _semantic_search(self, query):
        """Pure semantic search using vector embeddings"""
        query_embedding = self.embed(text=query).embedding

        # Search vector database
        results = self.db.search(
            vector=query_embedding,
            top_k=10,
            metric="cosine"
        )

        return dspy.Prediction(
            passages=[doc.text for doc in results],
            scores=[doc.score for doc in results]
        )

    def _hybrid_search(self, query):
        """Combine semantic and keyword search"""
        # Get semantic results
        semantic = self._semantic_search(query)

        # Get keyword results
        keyword = self._keyword_search(query)

        # Fuse results using reciprocal rank fusion
        fused_results = self._reciprocal_rank_fusion(
            semantic.passages,
            keyword.passages
        )

        return dspy.Prediction(passages=fused_results)
```

## Advanced Techniques

### Dynamic Hop Determination

```python
class AdaptiveMultiHop(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=5)
        self.check_completeness = dspy.Predict("question, current_info -> is_complete, next_query")
        self.generate_answer = dspy.Predict("question, all_info -> answer")

    def forward(self, question):
        all_info = []
        current_query = question
        max_hops = 5

        for hop in range(max_hops):
            # Retrieve information
            documents = self.retrieve(question=current_query).passages
            all_info.extend(documents)

            # Check if we have enough information
            completeness = self.check_completeness(
                question=question,
                current_info="\n".join(all_info)
            )

            if completeness.is_complete.lower() == "yes":
                break

            current_query = completeness.next_query

        # Generate final answer
        answer = self.generate_answer(
            question=question,
            all_info="\n\n".join(all_info)
        )

        return dspy.Prediction(
            answer=answer.answer,
            info_gathered=all_info,
            hops_used=hop + 1,
            is_complete=completeness.is_complete
        )
```

### Parallel Multi-hop Search

```python
class ParallelMultiHop(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=5)
        self.branch_queries = dspy.Predict("question -> parallel_queries")
        self.merge_results = dspy.ChainOfThought("question, branch_results -> integrated_answer")

    def forward(self, question):
        # Generate parallel search queries
        branching = self.branch_queries(question=question)
        queries = [q.strip() for q in branching.parallel_queries.split(";")]

        branch_results = []

        # Execute parallel searches
        for query in queries:
            documents = self.retrieve(question=query).passages
            branch_results.append({
                "query": query,
                "documents": documents
            })

        # Integrate results
        integration = self.merge_results(
            question=question,
            branch_results=str(branch_results)
        )

        return dspy.Prediction(
            answer=integration.integrated_answer,
            branches=branch_results,
            reasoning=integration.rationale
        )
```

## Best Practices

### 1. Prevent Information Explosion
```python
class ControlledMultiHop(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=3)  # Limit per hop
        self.score_relevance = dspy.Predict("document, question -> relevance_score")
        self.max_total_docs = 15

    def forward(self, question):
        all_docs = []
        # ... multi-hop logic with document deduplication and scoring
```

### 2. Maintain Query Relevance
```python
def maintain_query_focus(original_question, current_query, hop_count):
    """Ensure subsequent queries remain relevant."""
    if hop_count > 3:
        return original_question  # Return to original
    return current_query
```

### 3. Track Reasoning Paths
```python
class TransparentMultiHop(dspy.Module):
    def forward(self, question):
        reasoning_path = []
        # ... search logic with detailed tracking
        reasoning_path.append({
            "step": step,
            "query": query,
            "documents_found": len(docs),
            "decision": decision
        })
```

## Common Challenges and Solutions

### Challenge: Circular Reasoning
**Problem**: System keeps retrieving the same information.

**Solution**: Track visited documents and entities:
```python
visited_docs = set()
visited_entities = set()
```

### Challenge: Query Drift
**Problem**: Queries become too far removed from original question.

**Solution**: Regularly reconnect to original question.

### Challenge: Computational Cost
**Problem**: Multi-hop search can be expensive.

**Solution**: Use caching and limit search depth.

## Joint Optimization for Maximum Multi-Hop Performance

Research demonstrates that multi-hop QA is one of the domains that benefits most from joint optimization (combining fine-tuning with prompt optimization). Studies show **2-26x improvements** on complex multi-hop reasoning tasks.

### Why Joint Optimization Excels at Multi-Hop

Multi-hop reasoning presents unique challenges that benefit from combined optimization:

1. **Complex instruction following**: Multi-hop requires following multi-step instructions
2. **Information synthesis**: Combining information from multiple sources
3. **Strategic planning**: Deciding when to search vs. when to answer
4. **Reasoning depth**: Maintaining coherent reasoning across hops

Fine-tuned models can follow more complex multi-hop instructions, while prompt optimization discovers the best reasoning strategies.

### Multi-Hop with COPA Optimization

```python
from copa_optimizer import COPAOptimizer
from dspy.teleprompter import MIPRO, BootstrapFewShot

class OptimizedMultiHopQA(dspy.Module):
    """Multi-hop QA system optimized with COPA approach."""

    def __init__(self, max_hops=3):
        super().__init__()
        self.max_hops = max_hops
        self.retrieve = dspy.Retrieve(k=5)
        self.decompose = dspy.ChainOfThought("question -> subquestions, strategy")
        self.answer_sub = dspy.ChainOfThought("subquestion, context -> subanswer, confidence")
        self.synthesize = dspy.ChainOfThought(
            "question, subanswers, contexts -> final_answer, reasoning"
        )

    def forward(self, question):
        # Decompose complex question (fine-tuned models do this better)
        decomposition = self.decompose(question=question)
        subquestions = decomposition.subquestions.split(";")

        subanswers = []
        all_contexts = []

        for subq in subquestions[:self.max_hops]:
            subq = subq.strip()
            if not subq:
                continue

            # Retrieve relevant context
            contexts = self.retrieve(question=subq).passages
            all_contexts.extend(contexts)

            # Answer subquestion with confidence assessment
            result = self.answer_sub(
                subquestion=subq,
                context="\n".join(contexts)
            )

            subanswers.append({
                "question": subq,
                "answer": result.subanswer,
                "confidence": result.confidence
            })

        # Synthesize final answer
        subanswers_text = "\n".join([
            f"Q: {sa['question']}\nA: {sa['answer']} (confidence: {sa['confidence']})"
            for sa in subanswers
        ])

        synthesis = self.synthesize(
            question=question,
            subanswers=subanswers_text,
            contexts="\n\n".join(all_contexts[:10])
        )

        return dspy.Prediction(
            answer=synthesis.final_answer,
            reasoning=synthesis.reasoning,
            subquestions=subquestions,
            subanswers=subanswers
        )


def multi_hop_accuracy(example, pred, trace=None):
    """Comprehensive metric for multi-hop QA evaluation."""
    score = 0

    # Answer correctness (40%)
    if hasattr(pred, 'answer') and hasattr(example, 'answer'):
        if example.answer.lower() in pred.answer.lower():
            score += 0.4
        elif any(word in pred.answer.lower() for word in example.answer.lower().split()):
            score += 0.2

    # Reasoning quality (30%)
    if hasattr(pred, 'reasoning'):
        reasoning_length = len(pred.reasoning.split())
        if reasoning_length > 50:
            score += 0.3
        elif reasoning_length > 20:
            score += 0.15

    # Decomposition quality (30%)
    if hasattr(pred, 'subquestions'):
        expected_subs = example.get('expected_subquestions', 2)
        actual_subs = len([s for s in pred.subquestions if s.strip()])
        sub_score = min(actual_subs / expected_subs, 1.0) if expected_subs > 0 else 0
        score += 0.3 * sub_score

    return score


# Joint optimization with COPA
def optimize_multihop_with_copa(trainset, valset, base_model):
    """
    Apply COPA joint optimization for multi-hop QA.
    Achieves 2-26x improvements over baseline.
    """
    # Initialize COPA optimizer
    copa = COPAOptimizer(
        base_model_name=base_model,
        metric=multi_hop_accuracy,
        finetune_epochs=3,
        prompt_optimizer="mipro"
    )

    # Create program instance
    multi_hop = OptimizedMultiHopQA(max_hops=3)

    # Run joint optimization
    optimized_program, finetuned_model = copa.optimize(
        program=multi_hop,
        trainset=trainset,
        valset=valset
    )

    return optimized_program, finetuned_model


# Benchmark comparison
def compare_optimization_approaches(trainset, testset, base_model):
    """
    Compare different optimization approaches on multi-hop QA.

    Expected results based on research:
    - Baseline: ~12%
    - Fine-tuning only: ~28%
    - Prompt optimization only: ~20%
    - COPA (combined): ~45% (2-3.7x improvement)
    """
    results = {}
    program = OptimizedMultiHopQA()

    # Baseline (no optimization)
    dspy.settings.configure(lm=base_model)
    results["baseline"] = evaluate(program, testset)
    print(f"Baseline: {results['baseline']:.2%}")

    # Fine-tuning only
    finetuned = finetune_model(base_model, trainset, epochs=3)
    dspy.settings.configure(lm=finetuned)
    results["fine_tuning_only"] = evaluate(program, testset)
    print(f"Fine-tuning only: {results['fine_tuning_only']:.2%}")

    # Prompt optimization only (on base model)
    dspy.settings.configure(lm=base_model)
    mipro = MIPRO(metric=multi_hop_accuracy, auto="medium")
    prompt_optimized = mipro.compile(program, trainset=trainset)
    results["prompt_opt_only"] = evaluate(prompt_optimized, testset)
    print(f"Prompt optimization only: {results['prompt_opt_only']:.2%}")

    # COPA (combined)
    dspy.settings.configure(lm=finetuned)
    copa_optimized = mipro.compile(program, trainset=trainset)
    results["copa"] = evaluate(copa_optimized, testset)
    print(f"COPA (combined): {results['copa']:.2%}")

    # Calculate improvement factor
    improvement = results["copa"] / results["baseline"]
    print(f"\nImprovement factor: {improvement:.1f}x")

    # Calculate synergy
    additive = (
        results["baseline"] +
        (results["fine_tuning_only"] - results["baseline"]) +
        (results["prompt_opt_only"] - results["baseline"])
    )
    synergy = results["copa"] - additive
    print(f"Synergistic gain: {synergy:.2%}")

    return results
```

### Multi-Hop Performance Benchmarks

| Dataset | Baseline | FT Only | PO Only | COPA | Improvement |
|---------|----------|---------|---------|------|-------------|
| HotpotQA | 12% | 28% | 20% | 45% | 3.7x |
| 2WikiMultihopQA | 15% | 35% | 25% | 52% | 3.5x |
| MuSiQue | 8% | 22% | 18% | 38% | 4.8x |
| Complex QA | 10% | 30% | 22% | 48% | 4.8x |

### Best Practices for Joint Multi-Hop Optimization

1. **Data requirements**: Aim for 50-100 examples with multi-hop structure
2. **Fine-tune on decomposition**: Include examples of question decomposition
3. **Order matters**: Always fine-tune first, then apply prompt optimization
4. **Evaluate comprehensively**: Measure decomposition, reasoning, and final answer

For complete COPA implementation details, see [COPA: Combined Fine-Tuning and Prompt Optimization](../05-optimizers/09-copa-optimizer.md).

## Key Takeaways

1. **Multi-hop reasoning** enables answering complex, interconnected questions
2. **Different strategies** work for different types of questions
3. **Optimization is crucial** for handling the complexity of multi-hop systems
4. **Evaluation must consider** reasoning quality, not just final answer accuracy
5. **Real-world applications** include research, fact-checking, and analysis
6. **Trade-offs exist** between depth, accuracy, and computational cost
7. **Advanced retrieval techniques** like fused retrieval and dynamic termination dramatically improve performance
8. **Task-aware search** tailors queries to specific question types for better results
9. **Efficiency optimizations** are essential for large-scale deployment
10. **Vector database integration** enables semantic search capabilities for better relevance
11. **Joint optimization (COPA)** achieves 2-26x improvements on multi-hop tasks

## Next Steps

In the next section, we'll explore **Classification Tasks**, showing how to build robust text categorization systems that can handle real-world classification challenges.
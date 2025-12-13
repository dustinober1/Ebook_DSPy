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

## Key Takeaways

1. **Multi-hop reasoning** enables answering complex, interconnected questions
2. **Different strategies** work for different types of questions
3. **Optimization is crucial** for handling the complexity of multi-hop systems
4. **Evaluation must consider** reasoning quality, not just final answer accuracy
5. **Real-world applications** include research, fact-checking, and analysis
6. **Trade-offs exist** between depth, accuracy, and computational cost

## Next Steps

In the next section, we'll explore **Classification Tasks**, showing how to build robust text categorization systems that can handle real-world classification challenges.
# InPars+: Advanced Synthetic Data Generation for Information Retrieval

## Overview

**InPars+** extends the InPars (Instructed Pairs) framework for synthetic query generation in information retrieval systems. This enhancement introduces two major improvements: (1) **Contrastive Preference Optimization (CPO)** to fine-tune generator LLMs for higher quality query generation, and (2) **DSPy-based dynamic prompt optimization** using Chain-of-Thought (CoT) reasoning to adapt queries to specific retrieval contexts.

## Key Innovations

1. **CPO Fine-tuning**: Improves generator LLM's ability to create diverse, relevant queries
2. **Dynamic DSPy Optimization**: Real-time prompt adaptation based on retrieval performance
3. **Reduced Filtering**: 60% reduction in query filtering requirements due to higher initial quality
4. **Neural Information Retrieval (NIR) Integration**: Seamless integration with neural re-rankers
5. **Multi-stage Optimization**: Combines instruction and example optimization for superior performance

## Architecture

### 1. CPO Fine-tuned Query Generator

```python
import dspy
from typing import List, Dict, Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class CPOQueryGenerator(dspy.Module):
    """Query generator fine-tuned with Contrastive Preference Optimization."""

    def __init__(self, model_name: str = "mistralai/Mistral-7B"):
        super().__init__()

        # Load the fine-tuned model
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # DSPy module for query generation
        self.query_generator = dspy.Predict(
            """Generate diverse, relevant search queries based on the document.

            Document: {document}

            Generate {num_queries} unique queries that would retrieve this document.
            Each query should:
            - Be natural language
            - Target different aspects of the document
            - Vary in complexity and specificity
            - Be suitable for web/academic search

            Queries:"""
        )

    def generate_queries(self, document: str, num_queries: int = 5) -> List[str]:
        """Generate high-quality queries for a document."""

        result = self.query_generator(
            document=document,
            num_queries=num_queries
        )

        # Parse and clean generated queries
        queries = self._parse_queries(result.queries)

        # Deduplicate and rank by diversity
        diverse_queries = self._ensure_diversity(queries)

        return diverse_queries

    def _parse_queries(self, raw_output: str) -> List[str]:
        """Parse raw model output into individual queries."""
        # Implementation depends on model output format
        lines = raw_output.strip().split('\n')
        queries = []

        for line in lines:
            # Remove numbering and clean
            clean = line.strip()
            if clean and not clean.startswith(('1.', '2.', '3.', '4.', '5.')):
                queries.append(clean)
            elif clean and any(clean.startswith(str(i) + '.') for i in range(1, 10)):
                queries.append(clean.split('.', 1)[1].strip())

        return queries

    def _ensure_diversity(self, queries: List[str]) -> List[str]:
        """Ensure queries are diverse and not redundant."""
        if len(queries) <= 1:
            return queries

        diverse = [queries[0]]

        for query in queries[1:]:
            # Check similarity with existing queries
            is_similar = False
            for existing in diverse:
                similarity = self._calculate_similarity(query, existing)
                if similarity > 0.7:  # Threshold for similarity
                    is_similar = True
                    break

            if not is_similar:
                diverse.append(query)

        return diverse

    def _calculate_similarity(self, query1: str, query2: str) -> float:
        """Calculate semantic similarity between queries."""
        # Simplified implementation - use embedding similarity in practice
        common_words = set(query1.lower().split()) & set(query2.lower().split())
        total_words = set(query1.lower().split()) | set(query2.lower().split())
        return len(common_words) / len(total_words) if total_words else 0
```

### 2. DSPy Dynamic Prompt Optimizer

```python
class DSPyQueryOptimizer(dspy.Module):
    """Dynamic prompt optimizer for query generation using DSPy."""

    def __init__(self, base_retriever, evaluation_set: List[Dict]):
        super().__init__()
        self.base_retriever = base_retriever
        self.evaluation_set = evaluation_set

        # Initialize with Chain of Thought for prompt adaptation
        self.prompt_adaptator = dspy.ChainOfThought(
            """Analyze retrieval performance and adapt the query generation prompt.

            Current Performance:
            - Precision: {precision}
            - Recall: {recall}
            - MRR: {mrr}
            - Failed queries: {failed_queries}

            Document Type: {doc_type}
            Domain: {domain}

            Identify patterns in failed retrievals and suggest prompt improvements:

            1. What query characteristics led to poor performance?
            2. Which aspects of the document are being missed?
            3. How should the prompt be modified?

            Improved prompt:"""
        )

        # Multi-objective optimizer
        self.optimizer = dspy.MIPROv2(
            num_trials=20,
            num_candidates=10,
            voting_weight=0.3
        )

    def optimize_for_document_type(self, doc_type: str, sample_docs: List[str]):
        """Optimize query generation for specific document types."""

        # Define signature for this document type
        class DocTypeSignature(dspy.Signature):
            """Generate queries optimized for {doc_type} documents."""
            document = dspy.InputField(desc="The source document")
            num_queries = dspy.InputField(desc="Number of queries to generate")
            queries = dspy.OutputField(desc="Generated queries")

        # Create evaluation metric
        def retrieval_metric(example, prediction, trace=None):
            """Evaluate based on retrieval performance."""
            queries = prediction.get('queries', [])

            # Test each query
            total_score = 0
            for query in queries:
                retrieved = self.base_retriever.retrieve(query, k=10)
                # Check if target document is in results
                score = 1.0 if example['doc_id'] in [r['id'] for r in retrieved] else 0.0
                total_score += score

            return total_score / len(queries) if queries else 0.0

        # Create training examples
        trainset = []
        for doc in sample_docs:
            trainset.append(dspy.Example(
                document=doc['text'],
                doc_id=doc['id'],
                num_queries=5
            ).with_inputs('document', 'num_queries'))

        # Optimize the prompt
        optimized_program = self.optimizer.compile(
            program=dspy.Predict(DocTypeSignature),
            trainset=trainset,
            evalset=trainset[:5],  # Small validation set
            metric=retrieval_metric
        )

        return optimized_program

    def adaptive_query_generation(self, document: str, context: Dict) -> List[str]:
        """Generate queries with context-aware adaptation."""

        # Analyze document characteristics
        doc_features = self._analyze_document(document, context)

        # Select or create optimized program
        if doc_features['type'] in self.optimized_programs:
            generator = self.optimized_programs[doc_features['type']]
        else:
            # Fall back to base generator
            generator = self.base_query_generator

        # Generate queries
        result = generator(
            document=document,
            num_queries=context.get('num_queries', 5),
            **doc_features
        )

        # Post-process and validate
        queries = self._validate_queries(result.queries, document)

        return queries

    def _analyze_document(self, document: str, context: Dict) -> Dict:
        """Analyze document to determine optimization strategy."""

        analysis = dspy.Predict(
            """Analyze the document characteristics.

            Document: {document}

            Identify:
            1. Document type (academic, news, product, etc.)
            2. Domain/field
            3. Key topics
            4. Complexity level
            5. Target audience

            Analysis:"""
        )

        result = analysis(document=document)

        # Parse analysis into structured format
        return {
            'type': self._extract_field(result.analysis, "Document type"),
            'domain': self._extract_field(result.analysis, "Domain"),
            'complexity': self._assess_complexity(document),
            'topics': self._extract_topics(result.analysis)
        }
```

### 3. End-to-End InPars+ Pipeline

```python
class InParsPlusPipeline(dspy.Module):
    """Complete InPars+ pipeline for synthetic data generation and retrieval."""

    def __init__(self,
                 generator_model: str,
                 retriever,
                 num_synthetic_queries: int = 5):
        super().__init__()

        # Initialize components
        self.query_generator = CPOQueryGenerator(generator_model)
        self.prompt_optimizer = DSPyQueryOptimizer(retriever, evaluation_set=[])
        self.retriever = retriever
        self.num_queries = num_synthetic_queries

        # Performance tracking
        self.performance_history = []

    def generate_synthetic_training_data(self,
                                       corpus: List[Dict],
                                       target_size: int = 10000) -> List[Dict]:
        """Generate synthetic query-document pairs for training."""

        synthetic_data = []

        # Sample documents for generation
        sampled_docs = random.sample(
            corpus,
            min(target_size // self.num_queries, len(corpus))
        )

        for doc in sampled_docs:
            # Generate queries for each document
            queries = self.query_generator.generate_queries(
                document=doc['text'],
                num_queries=self.num_queries
            )

            # Create synthetic pairs
            for query in queries:
                synthetic_data.append({
                    'query': query,
                    'document_id': doc['id'],
                    'document_text': doc['text'],
                    'relevant': True  # All generated queries are relevant by construction
                })

        # Add negative examples through hard negative mining
        synthetic_data.extend(self._generate_hard_negatives(synthetic_data, corpus))

        return synthetic_data

    def _generate_hard_negatives(self,
                               positive_pairs: List[Dict],
                               corpus: List[Dict]) -> List[Dict]:
        """Generate hard negative examples for training."""

        negatives = []

        for pair in positive_pairs[:len(positive_pairs)//2]:  # Sample half
            # Retrieve documents for the query
            retrieved = self.retriever.retrieve(pair['query'], k=10)

            # Add non-retrieved documents as negatives
            retrieved_ids = {doc['id'] for doc in retrieved}

            for doc in corpus:
                if doc['id'] not in retrieved_ids and doc['id'] != pair['document_id']:
                    negatives.append({
                        'query': pair['query'],
                        'document_id': doc['id'],
                        'document_text': doc['text'],
                        'relevant': False
                    })

                    # Limit negatives per query
                    break

        return negatives

    def train_retriever(self, synthetic_data: List[Dict]):
        """Train a retriever using synthetic data."""

        # Split data
        train_data = synthetic_data[:int(0.8 * len(synthetic_data))]
        val_data = synthetic_data[int(0.8 * len(synthetic_data)):]

        # Train neural retriever
        self.retriever.train(
            train_data=train_data,
            val_data=val_data,
            num_epochs=5,
            learning_rate=1e-5
        )

        # Evaluate performance
        metrics = self.retriever.evaluate(val_data)
        self.performance_history.append(metrics)

        return metrics

    def optimize_continuously(self,
                            feedback_data: List[Dict],
                            optimization_interval: int = 100):
        """Continuously optimize based on user feedback."""

        # Update evaluation set with new feedback
        self.prompt_optimizer.evaluation_set.extend(feedback_data)

        # Periodically re-optimize
        if len(feedback_data) >= optimization_interval:
            # Identify underperforming document types
            performance_by_type = self._analyze_performance_by_type(feedback_data)

            # Re-optimize for problematic document types
            for doc_type, performance in performance_by_type.items():
                if performance['precision'] < 0.7:  # Threshold
                    print(f"Re-optimizing for document type: {doc_type}")

                    # Get samples of this document type
                    type_samples = [
                        doc for doc in self.prompt_optimizer.evaluation_set
                        if doc.get('doc_type') == doc_type
                    ]

                    if len(type_samples) >= 5:
                        optimized = self.prompt_optimizer.optimize_for_document_type(
                            doc_type, type_samples
                        )
                        self.prompt_optimizer.optimized_programs[doc_type] = optimized

            # Clear feedback for next iteration
            self.prompt_optimizer.evaluation_set = []

    def _analyze_performance_by_type(self, feedback_data: List[Dict]) -> Dict:
        """Analyze performance by document type."""

        type_performance = {}

        for item in feedback_data:
            doc_type = item.get('doc_type', 'unknown')
            if doc_type not in type_performance:
                type_performance[doc_type] = {
                    'precision': [],
                    'recall': [],
                    'count': 0
                }

            type_performance[doc_type]['precision'].append(item.get('precision', 0))
            type_performance[doc_type]['recall'].append(item.get('recall', 0))
            type_performance[doc_type]['count'] += 1

        # Calculate averages
        for doc_type in type_performance:
            metrics = type_performance[doc_type]
            if metrics['count'] > 0:
                metrics['precision'] = sum(metrics['precision']) / metrics['count']
                metrics['recall'] = sum(metrics['recall']) / metrics['count']

        return type_performance
```

### 4. Contrastive Preference Optimization

```python
class ContrastivePreferenceOptimizer:
    """Implements CPO for fine-tuning query generators."""

    def __init__(self, model_name: str, preference_data: List[Dict]):
        self.model_name = model_name
        self.preference_data = preference_data

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def fine_tune_with_preferences(self,
                                 num_epochs: int = 3,
                                 learning_rate: float = 1e-5):
        """Fine-tune model using preference pairs."""

        # Prepare preference pairs
        preference_pairs = self._prepare_preference_pairs()

        # Set up optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            total_loss = 0

            for batch in self._get_batches(preference_pairs, batch_size=8):
                # Forward pass for preferred and dispreferred
                preferred_loss = self._compute_loss(batch['preferred'])
                dispreferred_loss = self._compute_loss(batch['dispreferred'])

                # Contrastive loss
                contrastive_loss = -torch.log(
                    torch.exp(-preferred_loss) /
                    (torch.exp(-preferred_loss) + torch.exp(-dispreferred_loss))
                )

                # Backward pass
                optimizer.zero_grad()
                contrastive_loss.backward()
                optimizer.step()

                total_loss += contrastive_loss.item()

            print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(preference_pairs):.4f}")

    def _prepare_preference_pairs(self) -> List[Dict]:
        """Prepare preference pairs from training data."""

        pairs = []

        for item in self.preference_data:
            # Generate multiple query candidates
            candidates = self._generate_candidates(item['document'])

            # Score each candidate (simplified - use actual retriever in practice)
            scored_candidates = []
            for candidate in candidates:
                score = self._score_query(candidate, item['document'])
                scored_candidates.append((candidate, score))

            # Sort by score
            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            # Create preference pairs
            if len(scored_candidates) >= 2:
                pairs.append({
                    'document': item['document'],
                    'preferred': scored_candidates[0][0],
                    'dispreferred': scored_candidates[-1][0]
                })

        return pairs

    def _score_query(self, query: str, document: str) -> float:
        """Score query quality based on retrieval performance."""

        # Simplified scoring - use actual retrieval in practice
        score = 0.0

        # Check if query contains key terms from document
        doc_words = set(document.lower().split())
        query_words = set(query.lower().split())

        # Term overlap
        overlap = len(doc_words & query_words) / len(doc_words | query_words)
        score += 0.3 * overlap

        # Query length preference
        if 3 <= len(query.split()) <= 10:
            score += 0.2

        # Natural language score (simplified)
        if not query.startswith(('AND', 'OR', 'NOT')) and query.count('"') <= 2:
            score += 0.3

        # Diversity bonus
        if len(query_words) > 3:
            score += 0.2

        return score
```

## Implementation Guide

### 1. Setting Up InPars+

```python
# Initialize the pipeline
generator_model = "microsoft/DialoGPT-medium"  # Or other fine-tuned model
retriever = YourNeuralRetriever()  # e.g., ColBERT, DenseRetriever

pipeline = InParsPlusPipeline(
    generator_model=generator_model,
    retriever=retriever,
    num_synthetic_queries=5
)

# Load or create preference data for CPO
preference_data = load_preference_data("training_pairs.json")
cpo_optimizer = ContrastivePreferenceOptimizer(generator_model, preference_data)

# Fine-tune the generator
cpo_optimizer.fine_tune_with_preferences(num_epochs=3)
```

### 2. Generating Training Data

```python
# Generate synthetic training data
corpus = load_document_corpus("documents.jsonl")
synthetic_data = pipeline.generate_synthetic_training_data(
    corpus=corpus,
    target_size=50000  # Generate 50k training pairs
)

print(f"Generated {len(synthetic_data)} synthetic pairs")
print(f"Positive examples: {sum(1 for x in synthetic_data if x['relevant'])}")
print(f"Negative examples: {sum(1 for x in synthetic_data if not x['relevant'])}")

# Train the retriever
metrics = pipeline.train_retriever(synthetic_data)
print(f"Training metrics: {metrics}")
```

### 3. Continuous Optimization

```python
# Collect user feedback
feedback_loop = FeedbackCollection()

# Periodically optimize
while True:
    # Collect feedback for period
    feedback = feedback_loop.collect(period_hours=24)

    if feedback:
        # Update with new preferences
        pipeline.optimize_continuously(feedback)

        # Optionally re-fine-tune with new preferences
        if len(feedback) >= 100:
            cpo_optimizer.preference_data.extend(feedback)
            cpo_optimizer.fine_tune_with_preferences(num_epochs=1)
```

## Key Results from Paper

1. **Query Quality**: 85% of generated queries pass quality filters without human review
2. **Retrieval Performance**: 22% improvement in MRR over baseline InPars
3. **Filtering Reduction**: 60% fewer queries filtered out during generation
4. **Training Efficiency**: 40% less synthetic data needed for same performance
5. **Adaptation Speed**: 3x faster adaptation to new domains with DSPy optimization

## Best Practices

1. **Quality Preference Data**: Use human judgments or strong retrievers for preference pairs
2. **Diverse Document Types**: Ensure training data covers all target document types
3. **Regular Optimization**: Re-optimize prompts as user behavior changes
4. **Balanced Datasets**: Maintain good positive/negative example balance
5. **Monitor Drift**: Track performance degradation and re-train as needed

## Advanced Features

### 1. Multi-lingual Support

```python
class MultilingualInParsPlus(InParsPlusPipeline):
    """InPars+ with multi-lingual capabilities."""

    def __init__(self, languages: List[str], **kwargs):
        super().__init__(**kwargs)
        self.languages = languages
        self.translators = {lang: load_translator(lang) for lang in languages}

    def generate_multilingual_queries(self, document: str, doc_lang: str) -> Dict[str, List[str]]:
        """Generate queries in multiple languages."""

        # Generate in source language
        source_queries = self.query_generator.generate_queries(document)

        # Translate to other languages
        multilingual_queries = {doc_lang: source_queries}

        for target_lang in self.languages:
            if target_lang != doc_lang:
                translated = []
                for query in source_queries:
                    t_query = self.translators[target_lang].translate(query)
                    translated.append(t_query)
                multilingual_queries[target_lang] = translated

        return multilingual_queries
```

### 2. Domain-Specific Adaptation

```python
class DomainAdaptiveInPars(InParsPlusPipeline):
    """Domain-adaptive version of InPars+."""

    def __init__(self, domain_vocabs: Dict[str, List[str]], **kwargs):
        super().__init__(**kwargs)
        self.domain_vocabs = domain_vocabs

    def generate_domain_aware_queries(self,
                                    document: str,
                                    domain: str) -> List[str]:
        """Generate queries with domain-specific terminology."""

        # Get domain vocabulary
        domain_terms = self.domain_vocabs.get(domain, [])

        # Enhance prompt with domain context
        enhanced_prompt = f"""
        Generate queries for {domain} documents.

        Important terminology for this domain:
        {', '.join(domain_terms[:20])}

        Document: {document}

        Generate queries that:
        - Use appropriate domain terminology
        - Target domain-specific information needs
        - Match expert search patterns
        """

        # Generate with enhanced context
        queries = self.query_generator.generate_queries(document)

        # Filter for domain relevance
        domain_queries = [
            q for q in queries
            if any(term.lower() in q.lower() for term in domain_terms)
        ]

        return domain_queries
```

## Limitations and Considerations

1. **Preference Data Quality**: CPO performance depends on preference pair quality
2. **Computational Cost**: CPO fine-tuning requires significant compute resources
3. **Domain Specificity**: Performance may vary across different domains
4. **Query Diversity**: Need to balance relevance with diversity
5. **Evaluation Bias**: Metrics may not capture all aspects of query quality

## Conclusion

InPars+ significantly advances synthetic query generation by combining CPO fine-tuning with DSPy's dynamic optimization capabilities. The framework demonstrates how preference-based learning and adaptive prompting can work together to create high-quality training data for information retrieval systems. The reduction in filtering requirements and improved transfer performance make it a practical solution for real-world retrieval applications.
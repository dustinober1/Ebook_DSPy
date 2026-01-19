# IR Model Training from Scratch: Methodology and Best Practices

## Introduction

Information Retrieval (IR) models are the backbone of search engines, recommendation systems, and question-answering systems. Traditional IR model training requires thousands of relevance judgments and significant computational resources. However, with DSPy's innovative approach, we can train effective IR models from scratch using minimal data—sometimes with as few as 10 relevance judgments.

This section provides a comprehensive methodology for training IR models from scratch, focusing on practical implementations, optimization strategies, and real-world applications.

## Understanding IR Model Components

### Core IR Architecture

An IR model typically consists of three main components:

```
Query → Encoder → Document Encoder → Matching → Ranking
```

1. **Query Encoder**: Transforms user queries into vector representations
2. **Document Encoder**: Converts documents into comparable vector representations
3. **Matching & Ranking**: Determines relevance and produces ranked results

### Types of IR Models

| Model Type | Description | Training Requirements |
|------------|-------------|----------------------|
| **Sparse Retrieval** (BM25, TF-IDF) | Keyword-based matching | Minimal (statistical) |
| **Dense Retrieval** (DPR, ColBERT) | Semantic embedding matching | Moderate (hundreds of pairs) |
| **Hybrid Retrieval** | Combines sparse and dense | Moderate to high |
| **Neural Re-ranking** | Cross-attention models | High (thousands of pairs) |
| **Learned Sparse** (SPLADE) | Learned term weighting | Moderate |

## Training IR Models with Minimal Data

### The Zero-to-IR Framework

```python
import dspy
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class IRTrainingConfig:
    """Configuration for IR model training"""
    model_type: str  # 'sparse', 'dense', 'hybrid', 'reranker'
    training_examples: int  # Number of relevance judgments
    optimization_strategy: str  # 'prompt', 'meta', 'hybrid'
    domain: str  # Domain specialization
    base_model: str = "gpt-3.5-turbo"

class IRModelTrainer:
    """Trainer for IR models from scratch"""

    def __init__(self, config: IRTrainingConfig):
        self.config = config
        self.training_data = []
        self.model_components = {}

    def train_from_scratch(self,
                          documents: List[str],
                          relevance_judgments: List[Dict[str, Any]]) -> dspy.Module:
        """Train complete IR model from scratch"""

        print(f"Training {self.config.model_type} IR model with {len(relevance_judgments)} judgments")

        # Phase 1: Initialize components
        self._initialize_components(documents)

        # Phase 2: Process training data
        processed_data = self._process_relevance_judgments(relevance_judgments)

        # Phase 3: Train based on strategy
        if self.config.optimization_strategy == 'prompt':
            trained_model = self._prompt_optimization_training(processed_data)
        elif self.config.optimization_strategy == 'meta':
            trained_model = self._meta_learning_training(processed_data)
        else:  # hybrid
            trained_model = self._hybrid_training(processed_data)

        # Phase 4: Post-processing and calibration
        final_model = self._calibrate_model(trained_model)

        return final_model

    def _initialize_components(self, documents: List[str]):
        """Initialize IR model components based on type"""

        if self.config.model_type == 'dense':
            # Initialize dual encoder architecture
            self.model_components['query_encoder'] = dspy.Predict(
                "query -> query_embedding"
            )
            self.model_components['document_encoder'] = dspy.Predict(
                "document -> document_embedding"
            )
            self.model_components['similarity_calculator'] = dspy.Predict(
                "query_embedding, document_embedding -> similarity_score"
            )

        elif self.config.model_type == 'sparse':
            # Initialize learned sparse retrieval
            self.model_components['term_expander'] = dspy.ChainOfThought(
                "query -> expanded_terms, weights"
            )
            self.model_components['document_scorer'] = dspy.Predict(
                "query_terms, document -> relevance_score"
            )

        elif self.config.model_type == 'hybrid':
            # Initialize both sparse and dense components
            self.model_components['sparse_retriever'] = self._create_sparse_component()
            self.model_components['dense_retriever'] = self._create_dense_component()
            self.model_components['fusion_module'] = dspy.Predict(
                "sparse_scores, dense_scores -> final_scores"
            )

        elif self.config.model_type == 'reranker':
            # Initialize neural re-ranker
            self.model_components['candidate_scorer'] = dspy.ChainOfThought(
                "query, document -> relevance_score, reasoning"
            )

    def _create_sparse_component(self) -> dspy.Module:
        """Create sparse retrieval component"""

        class SparseRetriever(dspy.Module):
            def __init__(self):
                super().__init__()
                self.query_processor = dspy.ChainOfThought(
                    "query -> processed_terms, weights"
                )
                self.document_matcher = dspy.Predict(
                    "query_terms, document -> match_score"
                )

            def forward(self, query: str, documents: List[str]):
                # Process query
                processed = self.query_processor(query=query)

                # Score documents
                scores = []
                for doc in documents:
                    match = self.document_matcher(
                        query_terms=processed.processed_terms,
                        document=doc
                    )
                    scores.append(float(match.match_score))

                return dspy.Prediction(
                    scores=scores,
                    processed_terms=processed.processed_terms,
                    weights=processed.weights
                )

        return SparseRetriever()

    def _create_dense_component(self) -> dspy.Module:
        """Create dense retrieval component"""

        class DenseRetriever(dspy.Module):
            def __init__(self):
                super().__init__()
                self.query_encoder = dspy.Predict(
                    "query, domain_context -> query_vector"
                )
                self.document_encoder = dspy.Predict(
                    "document, domain_context -> document_vector"
                )

            def forward(self, query: str, documents: List[str], domain_context: str):
                # Encode query
                q_encoding = self.query_encoder(
                    query=query,
                    domain_context=domain_context
                )

                # Encode documents
                doc_encodings = []
                for doc in documents:
                    d_encoding = self.document_encoder(
                        document=doc,
                        domain_context=domain_context
                    )
                    doc_encodings.append(d_encoding.document_vector)

                # Calculate similarities
                similarities = self._calculate_similarities(
                    q_encoding.query_vector,
                    doc_encodings
                )

                return dspy.Prediction(
                    similarities=similarities,
                    query_vector=q_encoding.query_vector,
                    document_vectors=doc_encodings
                )

        return DenseRetriever()

    def _prompt_optimization_training(self, processed_data: Dict[str, Any]) -> dspy.Module:
        """Train IR model using prompt optimization"""

        # Create base IR model
        base_model = self._create_base_ir_model()

        # Generate prompt variations
        prompt_variations = self._generate_ir_prompt_variations()

        # Evaluate each variation
        best_prompt = None
        best_score = 0.0

        for prompt in prompt_variations:
            # Apply prompt to model
            temp_model = self._apply_prompt_to_ir_model(base_model, prompt)

            # Evaluate with limited data
            score = self._evaluate_ir_model(temp_model, processed_data)

            if score > best_score:
                best_score = score
                best_prompt = prompt

        # Train final model with best prompt
        final_model = self._apply_prompt_to_ir_model(base_model, best_prompt)
        optimized_model = self._fine_tune_with_limited_data(
            final_model, processed_data
        )

        return optimized_model

    def _meta_learning_training(self, processed_data: Dict[str, Any]) -> dspy.Module:
        """Train IR model using meta-learning"""

        # Step 1: Identify meta-tasks
        meta_tasks = self._identify_meta_ir_tasks(self.config.domain)

        # Step 2: Learn from meta-tasks
        meta_knowledge = self._learn_from_meta_tasks(meta_tasks)

        # Step 3: Adapt to target task
        adapted_model = self._adapt_to_target_task(
            meta_knowledge, processed_data
        )

        return adapted_model

    def _hybrid_training(self, processed_data: Dict[str, Any]) -> dspy.Module:
        """Combine multiple training strategies"""

        # Train multiple models
        models = {}

        # Prompt-based model
        models['prompt'] = self._prompt_optimization_training(processed_data)

        # Meta-learning model
        models['meta'] = self._meta_learning_training(processed_data)

        # Ensemble the models
        ensemble = self._create_ensemble(models)

        return ensemble
```

## Specialized Training for Different IR Tasks

### Task 1: Document Ranking with 10 Relevance Judgments

```python
def train_document_ranker_with_10_judgments():
    """Train document ranking model with only 10 relevance judgments"""

    # Example: 10 relevance judgments for web search
    judgments = [
        {
            'query': 'machine learning tutorials',
            'documents': [
                {'id': 'doc1', 'content': 'Complete guide to machine learning for beginners', 'relevance': 2},
                {'id': 'doc2', 'content': 'Advanced deep learning techniques', 'relevance': 1},
                {'id': 'doc3', 'content': 'Python machine learning libraries comparison', 'relevance': 2},
                {'id': 'doc4', 'content': 'History of artificial intelligence', 'relevance': 0},
                {'id': 'doc5', 'content': 'Machine learning in healthcare applications', 'relevance': 1}
            ]
        },
        # ... 9 more query-document judgments
    ]

    # Configure for minimal data training
    config = IRTrainingConfig(
        model_type='dense',
        training_examples=10,
        optimization_strategy='hybrid',
        domain='web_search'
    )

    # Create document collection
    all_documents = extract_all_documents(judgments)

    # Initialize trainer
    trainer = IRModelTrainer(config)

    # Train from scratch
    ranking_model = trainer.train_from_scratch(
        documents=all_documents,
        relevance_judgments=judgments
    )

    return ranking_model

def test_ranking_model(model, test_queries, document_collection):
    """Test the trained ranking model"""

    for query in test_queries:
        # Get rankings
        results = model(query=query, documents=document_collection)

        # Sort documents by score
        ranked_docs = sorted(
            zip(document_collection, results.scores),
            key=lambda x: x[1],
            reverse=True
        )

        print(f"\nQuery: {query}")
        print("Ranked Documents:")
        for i, (doc, score) in enumerate(ranked_docs[:5]):
            print(f"{i+1}. Score: {score:.3f}")
            print(f"   {doc[:100]}...")
```

### Task 2: Passage Retrieval for QA Systems

```python
class PassageRetrieverTrainer:
    """Specialized trainer for passage retrieval in QA systems"""

    def __init__(self):
        self.passage_encoder = None
        self.query_encoder = None
        self.reranker = None

    def train_passage_retriever_with_10_examples(self,
                                                passages: List[str],
                                                qa_pairs: List[Dict[str, str]]):
        """Train passage retriever with 10 QA pairs"""

        # Step 1: Create training data from QA pairs
        training_data = []
        for qa in qa_pairs:
            # Find relevant passages (simulated here)
            relevant_passages = find_relevant_passages(
                qa['question'], passages, top_k=3
            )
            training_data.append({
                'query': qa['question'],
                'relevant_passages': relevant_passages,
                'answer': qa['answer']
            })

        # Step 2: Initialize passage retrieval components
        self._initialize_passage_components()

        # Step 3: Train with minimal data
        trained_retriever = self._train_with_minimal_data(training_data)

        # Step 4: Add answer-aware re-ranking
        self.reranker = self._train_answer_aware_reranker(training_data)

        return self._create_complete_retriever(trained_retriever)

    def _train_with_minimal_data(self, training_data):
        """Train using minimal data strategies"""

        # Create synthetic examples through data augmentation
        augmented_data = self._augment_qa_training_data(training_data)

        # Use prompt optimization for passage encoding
        prompt_optimizer = PromptOptimizer()
        best_prompts = prompt_optimizer.optimize_for_passage_retrieval(
            augmented_data
        )

        # Create trained retriever
        class TrainedPassageRetriever(dspy.Module):
            def __init__(self, prompts):
                super().__init__()
                self.query_encoder = dspy.ChainOfThought(prompts['query_encoding'])
                self.passage_scorer = dspy.Predict(prompts['passage_scoring'])

            def forward(self, question, passages):
                # Encode question with context
                encoded_q = self.query_encoder(
                    question=question,
                    context="Find passages that answer this question"
                )

                # Score passages
                scored_passages = []
                for passage in passages:
                    score = self.passage_scorer(
                        question=encoded_q.reasoning,
                        passage=passage
                    )
                    scored_passages.append({
                        'passage': passage,
                        'score': float(score.score),
                        'reasoning': score.get('reasoning', '')
                    })

                # Sort by score
                scored_passages.sort(key=lambda x: x['score'], reverse=True)

                return dspy.Prediction(
                    ranked_passages=[p['passage'] for p in scored_passages],
                    scores=[p['score'] for p in scored_passages],
                    reasoning=[p['reasoning'] for p in scored_passages]
                )

        return TrainedPassageRetriever(best_prompts)
```

### Task 3: Cross-Lingual IR with Minimal Bilingual Data

```python
class CrossLingualIRTrainer:
    """Train cross-lingual IR models with minimal bilingual data"""

    def __init__(self, source_lang: str, target_lang: str):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.translation_cache = {}

    def train_with_10_bilingual_examples(self,
                                       source_docs: List[str],
                                       target_docs: List[str],
                                       parallel_examples: List[Dict[str, Any]]):
        """Train cross-lingual IR with 10 parallel examples"""

        # Step 1: Learn cross-lingual representations
        multilingual_encoder = self._learn_cross_lingual_encoding(parallel_examples)

        # Step 2: Train query translation model
        query_translator = self._train_query_translation(parallel_examples)

        # Step 3: Train cross-lingual similarity
        similarity_model = self._train_cross_lingual_similarity(parallel_examples)

        # Step 4: Create complete cross-lingual IR system
        clir_system = self._create_clir_system(
            multilingual_encoder,
            query_translator,
            similarity_model
        )

        return clir_system

    def _learn_cross_lingual_encoding(self, parallel_examples):
        """Learn shared space for multiple languages"""

        # Use parallel examples to learn mapping between languages
        class CrossLingualEncoder(dspy.Module):
            def __init__(self):
                super().__init__()
                # Learn to map both languages to shared space
                self.source_encoder = dspy.Predict(
                    f"text, language='{self.source_lang}' -> embedding"
                )
                self.target_encoder = dspy.Predict(
                    f"text, language='{self.target_lang}' -> embedding"
                )
                # Alignment layer
                self.align_embeddings = dspy.Predict(
                    "source_emb, target_emb -> aligned_source, aligned_target"
                )

            def forward(self, text, language):
                if language == self.source_lang:
                    encoded = self.source_encoder(text=text, language=language)
                else:
                    encoded = self.target_encoder(text=text, language=language)

                return encoded.embedding

        # Train using 10 parallel examples
        encoder = CrossLingualEncoder()
        trained_encoder = self._train_encoder_with_examples(
            encoder, parallel_examples
        )

        return trained_encoder
```

## Advanced Training Techniques

### Technique 1: Self-Supervised Pre-training for IR

```python
def self_supervised_ir_pretraining(documents: List[str]) -> dspy.Module:
    """Pre-train IR components without any labels"""

    # Step 1: Create synthetic training tasks
    synthetic_tasks = create_ir_pretraining_tasks(documents)

    # Step 2: Pre-train query encoder
    query_encoder = pretrain_query_encoder(synthetic_tasks)

    # Step 3: Pre-train document encoder
    document_encoder = pretrain_document_encoder(synthetic_tasks)

    # Step 4: Pre-train matching component
    matcher = pretrain_matching_component(synthetic_tasks)

    return IRModel(query_encoder, document_encoder, matcher)

def create_ir_pretraining_tasks(documents):
    """Create self-supervised tasks for IR pre-training"""

    tasks = []

    # Task 1: Document reconstruction (like masked language modeling)
    for doc in documents[:1000]:  # Limit for computation
        masked_doc = mask_random_tokens(doc)
        tasks.append({
            'type': 'reconstruction',
            'input': masked_doc,
            'target': doc
        })

    # Task 2: Next sentence prediction for document pairs
    for i in range(len(documents) - 1):
        tasks.append({
            'type': 'next_doc',
            'input': documents[i],
            'target': documents[i + 1]
        })

    # Task 3: Query-document matching (synthetic)
    for doc in documents[:500]:
        # Generate synthetic query from document
        synthetic_query = generate_query_from_document(doc)
        tasks.append({
            'type': 'query_doc_match',
            'query': synthetic_query,
            'document': doc,
            'label': 1  # Positive example
        })

        # Add negative example
        negative_doc = documents[np.random.randint(len(documents))]
        if negative_doc != doc:
            tasks.append({
                'type': 'query_doc_match',
                'query': synthetic_query,
                'document': negative_doc,
                'label': 0  # Negative example
            })

    return tasks
```

### Technique 2: Active Learning for IR

```python
class ActiveIRLearner:
    """Active learning framework for IR with minimal annotations"""

    def __init__(self, unlabeled_documents: List[str]):
        self.documents = unlabeled_documents
        self.labeled_queries = []
        self.annotator_feedback = []

    def active_learning_cycle(self,
                            initial_annotations: List[Dict] = None,
                            budget: int = 50) -> dspy.Module:
        """Perform active learning to minimize annotation requirements"""

        # Start with initial annotations (could be 0)
        if initial_annotations:
            self.labeled_queries = initial_annotations

        # Iterative active learning
        for iteration in range(budget):
            print(f"Active learning iteration {iteration + 1}/{budget}")

            # Step 1: Train current model with available labels
            current_model = self._train_with_current_labels()

            # Step 2: Select most informative queries to label
            candidates = self._generate_candidate_queries()
            selected = self._select_informative_queries(
                current_model, candidates, n=5
            )

            # Step 3: Get human annotations (simulated here)
            new_annotations = self._request_annotations(selected)

            # Step 4: Add to labeled set
            self.labeled_queries.extend(new_annotations)

        # Train final model with all collected labels
        final_model = self._train_with_current_labels()

        return final_model

    def _select_informative_queries(self,
                                   model,
                                   candidates,
                                   n: int = 5):
        """Select queries with highest uncertainty or diversity"""

        uncertainties = []
        for query in candidates:
            # Get model uncertainty for this query
            uncertainty = self._calculate_query_uncertainty(model, query)
            uncertainties.append((query, uncertainty))

        # Sort by uncertainty
        uncertainties.sort(key=lambda x: x[1], reverse=True)

        # Select top uncertain queries
        selected = [q for q, _ in uncertainties[:n]]

        # Ensure diversity
        diverse_selected = self._ensure_diversity(selected)

        return diverse_selected
```

### Technique 3: Multi-Task Learning for IR

```python
def multi_task_ir_training(tasks: List[Dict[str, Any]]) -> dspy.Module:
    """Train IR model on multiple related tasks simultaneously"""

    # Task definitions could include:
    # - Document ranking
    # - Passage retrieval
    # - Answer span prediction
    # - Query classification
    # - Document classification

    class MultiTaskIRModel(dspy.Module):
        def __init__(self):
            super().__init__()
            # Shared encoder
            self.shared_encoder = dspy.Predict(
                "text, task_type -> shared_representation"
            )

            # Task-specific heads
            self.ranking_head = dspy.Predict(
                "query_repr, doc_repr -> relevance_score"
            )
            self.retrieval_head = dspy.Predict(
                "query_repr, passage_repr -> retrieval_score"
            )
            self.classification_head = dspy.Predict(
                "text_repr -> class_label"
            )

        def forward(self, inputs, task_type):
            # Get shared representation
            shared = self.shared_encoder(
                text=inputs['text'],
                task_type=task_type
            )

            # Route to appropriate task head
            if task_type == 'ranking':
                return self.ranking_head(**inputs, query_repr=shared)
            elif task_type == 'retrieval':
                return self.retrieval_head(**inputs, query_repr=shared)
            elif task_type == 'classification':
                return self.classification_head(text_repr=shared)

    # Train on all tasks simultaneously
    model = MultiTaskIRModel()
    trained_model = train_multi_task(model, tasks)

    return trained_model
```

## Evaluation Methodology

### IR Evaluation with Minimal Test Data

```python
def evaluate_ir_model_minimal_data(model,
                                  test_queries: List[str],
                                  test_relevance: Dict[str, List[int]],
                                  confidence_adjusted: bool = True) -> Dict[str, float]:
    """Evaluate IR model with minimal test data"""

    metrics = {}

    # Standard IR metrics
    for query in test_queries:
        # Get rankings
        results = model(query=query, documents=all_documents)
        ranked_docs = parse_rankings(results)

        # Calculate metrics with confidence adjustment
        if confidence_adjusted and 'confidence' in results:
            # Weight metrics by confidence
            weights = results['confidence']
            adjusted_ranking = apply_confidence_weights(
                ranked_docs, weights
            )
        else:
            adjusted_ranking = ranked_docs

        # Calculate per-query metrics
        query_metrics = calculate_ir_metrics(
            adjusted_ranking,
            test_relevance[query]
        )

        # Aggregate
        for metric, value in query_metrics.items():
            if metric not in metrics:
                metrics[metric] = []
            metrics[metric].append(value)

    # Calculate final scores
    final_metrics = {
        metric: np.mean(values) + 1.96 * np.std(values) / np.sqrt(len(values))
        for metric, values in metrics.items()
    }

    return final_metrics
```

## Best Practices and Guidelines

### For Training with Minimal Data

1. **Start Simple**: Begin with simpler models (sparse retrieval) before complex ones
2. **Use Pre-training**: Leverage self-supervised pre-training when possible
3. **Data Quality**: Ensure every relevance judgment is accurate
4. **Active Learning**: Select examples that maximize learning
5. **Regular Evaluation**: Continuously evaluate to prevent overfitting

### For Production Deployment

1. **Confidence Estimation**: Always include confidence scores
2. **Fallback Mechanisms**: Have simpler models as fallbacks
3. **Continuous Learning**: Collect feedback for model improvement
4. **Monitoring**: Track performance drift over time
5. **A/B Testing**: Test new models before full deployment

## Key Takeaways

1. **IR Models Can Be Trained from Scratch**: Even with 10 relevance judgments
2. **Strategy Selection is Crucial**: Different tasks require different approaches
3. **Data Efficiency is Possible**: Through prompt optimization and meta-learning
4. **Quality Trumps Quantity**: High-quality judgments are more valuable than many poor ones
5. **Confidence Estimation is Essential**: When working with minimal training data

## Next Steps

This section covered training IR models from scratch with minimal data. The concepts here build upon the optimization techniques discussed in earlier chapters and demonstrate practical applications in real-world scenarios.

For continued learning, explore:
- [Prompt Hyperparameter Optimization](../05-optimizers/20-prompts-as-hyperparameters.md) for deeper optimization techniques
- [Evaluation Strategies](../04-evaluation/03-defining-metrics.md) for comprehensive model evaluation
- [Real-world Case Studies](../08-case-studies/) for production deployment examples
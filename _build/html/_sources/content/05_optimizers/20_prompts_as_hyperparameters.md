# Prompts as Auto-Optimized Hyperparameters

## Introduction

In traditional machine learning, hyperparameters such as learning rate, batch size, and model architecture are carefully tuned to optimize performance. In the context of language models and DSPy, prompts themselves can be treated as trainable hyperparameters that are automatically optimized to maximize performance on specific tasks.

This revolutionary approach treats prompt engineering not as an art form requiring manual crafting, but as a systematic optimization problem where the optimal prompt is discovered through automated search and refinement.

## The Prompt Hyperparameter Framework

### Conceptual Foundation

When we treat prompts as hyperparameters, we're fundamentally changing how we think about prompt engineering:

```
Traditional Approach:
Manual Prompt Design → Test → Manual Refinement → Repeat

DSPy Hyperparameter Approach:
Prompt Space Definition → Automated Optimization → Optimal Prompt
```

### Types of Prompt Hyperparameters

1. **Instruction Templates**: The structure and wording of task instructions
2. **Few-shot Examples**: Selection and ordering of demonstration examples
3. **Formatting Patterns**: How inputs and outputs are presented
4. **Task Decomposition**: How complex tasks are broken down
5. **Reasoning Steps**: Explicit guidance for thinking processes

## Auto-Optimization Architecture

### The Optimization Loop

```python
import dspy
from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass

@dataclass
class PromptHyperparameters:
    """Container for prompt hyperparameters"""
    instruction_template: str
    example_selection_strategy: str
    formatting_pattern: str
    reasoning_guidance: str
    task_decomposition: List[str]

class PromptHyperparameterOptimizer:
    """Automated prompt hyperparameter optimization"""

    def __init__(self,
                 base_program: dspy.Module,
                 metric_fn: callable,
                 search_space: Dict[str, Any]):
        self.base_program = base_program
        self.metric_fn = metric_fn
        self.search_space = search_space
        self.optimization_history = []

    def optimize(self,
                 trainset: List[dspy.Example],
                 valset: List[dspy.Example],
                 num_iterations: int = 50) -> PromptHyperparameters:
        """Optimize prompt hyperparameters using systematic search"""

        best_params = None
        best_score = 0.0

        for iteration in range(num_iterations):
            # Sample hyperparameters from search space
            current_params = self._sample_hyperparameters()

            # Create program with current hyperparameters
            optimized_program = self._apply_hyperparameters(
                self.base_program, current_params
            )

            # Evaluate on validation set
            score = self._evaluate_program(optimized_program, valset)

            # Track best configuration
            if score > best_score:
                best_score = score
                best_params = current_params

            self.optimization_history.append({
                'iteration': iteration,
                'params': current_params,
                'score': score
            })

        return best_params

    def _sample_hyperparameters(self) -> PromptHyperparameters:
        """Sample from hyperparameter search space"""
        return PromptHyperparameters(
            instruction_template=np.random.choice(
                self.search_space['instruction_templates']
            ),
            example_selection_strategy=np.random.choice(
                self.search_space['example_strategies']
            ),
            formatting_pattern=np.random.choice(
                self.search_space['formatting_patterns']
            ),
            reasoning_guidance=np.random.choice(
                self.search_space['reasoning_guidance']
            ),
            task_decomposition=np.random.choice(
                self.search_space['task_decompositions'],
                size=np.random.randint(1, 4)
            ).tolist()
        )
```

## Practical Implementation: IR Model Training

### Information Retrieval with Optimized Prompts

```python
class OptimizedIRRetriever(dspy.Module):
    """IR model with prompts optimized as hyperparameters"""

    def __init__(self, prompt_hyperparams: PromptHyperparameters):
        super().__init__()
        self.hyperparams = prompt_hyperparams

        # Core retrieval components
        self.query_encoder = dspy.Predict(
            f"{prompt_hyperparams.instruction_template} -> encoded_query"
        )

        self.document_ranker = dspy.ChainOfThought(
            f"{prompt_hyperparams.formatting_pattern} -> ranked_documents"
        )

        self.relevance_scorer = dspy.Predict(
            f"{prompt_hyperparams.reasoning_guidance} -> relevance_score"
        )

    def forward(self, query: str, documents: List[str]) -> dspy.Prediction:
        """Execute optimized retrieval pipeline"""

        # Step 1: Encode query using optimized instruction
        encoded_query = self.query_encoder(query=query)

        # Step 2: Apply task decomposition if specified
        if len(self.hyperparams.task_decomposition) > 1:
            # Break down complex query
            sub_queries = self._decompose_query(query)
            all_results = []

            for sub_query in sub_queries:
                results = self.document_ranker(
                    query=sub_query,
                    documents="\n".join(documents),
                    instruction=self.hyperparams.instruction_template
                )
                all_results.append(results)

            # Merge results from sub-queries
            final_results = self._merge_results(all_results)
        else:
            # Single query processing
            final_results = self.document_ranker(
                query=query,
                documents="\n".join(documents),
                instruction=self.hyperparams.instruction_template
            )

        # Step 3: Score relevance using optimized reasoning
        ranked_docs = final_results.ranked_documents.split("\n")
        scored_results = []

        for doc in ranked_docs[:10]:  # Top 10 documents
            score_result = self.relevance_scorer(
                query=query,
                document=doc,
                reasoning_prompt=self.hyperparams.reasoning_guidance
            )
            scored_results.append({
                'document': doc,
                'score': float(score_result.relevance_score),
                'reasoning': score_result.get('reasoning', '')
            })

        # Sort by relevance score
        scored_results.sort(key=lambda x: x['score'], reverse=True)

        return dspy.Prediction(
            ranked_documents=[r['document'] for r in scored_results],
            relevance_scores=[r['score'] for r in scored_results],
            reasoning_steps=[r['reasoning'] for r in scored_results],
            encoded_query=encoded_query.encoded_query
        )
```

## Extreme Few-Shot Learning with 10 Examples

### The Challenge of Minimal Data

Training effective models with only 10 labeled examples represents the frontier of few-shot learning. Traditional approaches fail dramatically in this regime, but prompt hyperparameter optimization enables remarkable performance.

### Data Efficiency Framework

```python
class ExtremeFewShotOptimizer:
    """Specialized optimizer for extreme few-shot scenarios"""

    def __init__(self, base_model: str = "gpt-3.5-turbo"):
        self.base_model = base_model
        self.meta_learning_cache = {}

    def optimize_with_10_examples(self,
                                 task_signature: dspy.Signature,
                                 examples: List[dspy.Example],
                                 num_prompt_variations: int = 100) -> dspy.Module:
        """Optimize for tasks with only 10 labeled examples"""

        # Step 1: Meta-prompt generation
        meta_prompts = self._generate_meta_prompts(
            task_signature, num_prompt_variations
        )

        # Step 2: Cross-validation with 10 examples
        best_prompt = None
        best_cv_score = 0.0

        for meta_prompt in meta_prompts:
            # Perform 5-fold cross-validation with 10 examples
            cv_scores = self._cross_validate_with_10_examples(
                meta_prompt, examples, task_signature
            )

            avg_score = np.mean(cv_scores)

            if avg_score > best_cv_score:
                best_cv_score = avg_score
                best_prompt = meta_prompt

        # Step 3: Create optimized program with best prompt
        optimized_program = self._create_optimized_program(
            best_prompt, task_signature
        )

        return optimized_program

    def _generate_meta_prompts(self,
                              signature: dspy.Signature,
                              num_variations: int) -> List[str]:
        """Generate diverse meta-prompts for optimization"""

        # Use meta-learning to generate effective prompt variations
        meta_instruction = f"""
        Generate {num_variations} different prompts for the following task:
        Task: {signature}

        Each prompt should:
        1. Use different instruction styles (direct, conversational, formal, creative)
        2. Include different levels of guidance (minimal, moderate, detailed)
        3. Suggest different reasoning approaches
        4. Vary in complexity and abstraction

        Make each prompt unique and optimized for few-shot learning.
        """

        # Generate using a powerful model
        prompt_generator = dspy.Predict("task -> prompt_variations")
        result = prompt_generator(task=meta_instruction)

        # Parse and clean the generated prompts
        prompts = self._parse_prompts(result.prompt_variations)

        # Add domain-specific variations if examples provide clues
        if len(self.meta_learning_cache) > 0:
            domain_prompts = self._generate_domain_prompts(signature)
            prompts.extend(domain_prompts)

        return prompts[:num_variations]

    def _cross_validate_with_10_examples(self,
                                        prompt: str,
                                        examples: List[dspy.Example],
                                        signature: dspy.Signature) -> List[float]:
        """Perform cross-validation with only 10 examples"""

        scores = []

        # Create 5 folds of 8 training, 2 testing examples
        folds = self._create_folds_with_10_examples(examples, k=5)

        for fold_train, fold_test in folds:
            # Create temporary program with current prompt
            temp_program = self._create_program_with_prompt(prompt, signature)

            # Compile with training examples
            optimizer = BootstrapFewShot(
                metric=self._create_metric_for_task(signature),
                max_bootstrapped_demos=3  # Very few due to limited data
            )

            compiled = optimizer.compile(temp_program, trainset=fold_train)

            # Evaluate on test examples
            fold_score = self._evaluate_on_examples(compiled, fold_test)
            scores.append(fold_score)

        return scores

    def _create_folds_with_10_examples(self,
                                     examples: List[dspy.Example],
                                     k: int = 5) -> List[tuple]:
        """Create balanced cross-validation folds from 10 examples"""

        # Ensure balanced representation across classes if possible
        folds = []

        # Use leave-two-out cross-validation for 10 examples
        for i in range(len(examples)):
            for j in range(i+1, len(examples)):
                test_set = [examples[i], examples[j]]
                train_set = [ex for idx, ex in enumerate(examples)
                           if idx not in [i, j]]

                # Use only first 5 folds to limit computation
                if len(folds) < 5:
                    folds.append((train_set, test_set))

        return folds
```

## Training Pipeline for Minimal Data

### The 10-Example Training Pipeline

```python
class TenExampleTrainingPipeline:
    """Complete pipeline for training with minimal data"""

    def __init__(self,
                 task_type: str,
                 base_model: str = "gpt-3.5-turbo"):
        self.task_type = task_type
        self.base_model = base_model
        self.pipeline_components = {}

    def train_with_10_examples(self,
                              examples: List[dspy.Example],
                              task_signature: dspy.Signature) -> Dict[str, Any]:
        """Complete training pipeline using only 10 examples"""

        results = {
            'examples_used': len(examples),
            'optimization_steps': [],
            'final_performance': None,
            'trained_components': {}
        }

        # Step 1: Data Analysis and Augmentation
        print("Step 1: Analyzing and augmenting minimal data...")
        augmented_data = self._augment_minimal_data(examples)
        results['optimization_steps'].append(
            f"Augmented {len(examples)} examples to {len(augmented_data)}"
        )

        # Step 2: Prompt Hyperparameter Optimization
        print("Step 2: Optimizing prompt hyperparameters...")
        prompt_optimizer = ExtremeFewShotOptimizer(self.base_model)
        optimized_program = prompt_optimizer.optimize_with_10_examples(
            task_signature, examples
        )
        results['trained_components']['optimized_program'] = optimized_program

        # Step 3: Meta-Learning Integration
        print("Step 3: Integrating meta-learning...")
        meta_enhanced = self._apply_meta_learning(
            optimized_program, augmented_data
        )
        results['trained_components']['meta_enhanced'] = meta_enhanced

        # Step 4: Few-Shot Fine-Tuning
        print("Step 4: Applying few-shot fine-tuning...")
        fine_tuned = self._few_shot_fine_tune(meta_enhanced, examples)
        results['trained_components']['fine_tuned'] = fine_tuned

        # Step 5: Evaluation and Validation
        print("Step 5: Comprehensive evaluation...")
        evaluation_results = self._comprehensive_evaluation(
            fine_tuned, examples
        )
        results['final_performance'] = evaluation_results

        return results

    def _augment_minimal_data(self,
                             examples: List[dspy.Example]) -> List[dspy.Example]:
        """Strategically augment minimal training data"""

        augmented = examples.copy()

        # Strategy 1: Paraphrase generation
        for example in examples:
            paraphraser = dspy.Predict("text -> paraphrase")
            para_result = paraphraser(text=example.input)

            new_example = example.with_inputs(
                input=para_result.paraphrase
            )
            augmented.append(new_example)

        # Strategy 2: Counterfactual generation
        if self.task_type in ['classification', 'qa']:
            for example in examples:
                counterfactual_gen = dspy.ChainOfThought(
                    "example -> counterfactual_example"
                )
                cf_result = counterfactual_gen(
                    example=str(example)
                )

                # Parse and add counterfactual example
                cf_example = self._parse_counterfactual(
                    cf_result.counterfactual_example, example
                )
                if cf_example:
                    augmented.append(cf_example)

        # Strategy 3: Template-based generation
        templates = self._extract_templates_from_examples(examples)
        for template in templates:
            template_gen = dspy.Predict(
                f"template -> new_example_{self.task_type}"
            )
            new_ex_result = template_gen(template=template)

            new_example = self._parse_template_example(
                new_ex_result[f"new_example_{self.task_type}"], example
            )
            if new_example:
                augmented.append(new_example)

        return augmented

    def _apply_meta_learning(self,
                            program: dspy.Module,
                            augmented_data: List[dspy.Example]) -> dspy.Module:
        """Apply meta-learning to improve generalization"""

        # Create meta-learner that learns how to learn
        meta_learner = MetaLearningWrapper(program)

        # Perform MAML-style adaptation with few examples
        adapted_program = meta_learner.adapt(
            support_set=augmented_data[:8],  # Use 8 for adaptation
            query_set=augmented_data[8:],    # 2 for query
            adaptation_steps=3
        )

        return adapted_program

    def _few_shot_fine_tune(self,
                           program: dspy.Module,
                           examples: List[dspy.Example]) -> dspy.Module:
        """Apply specialized fine-tuning for few-shot scenarios"""

        # Use KNNFewShot for example-based learning
        knn_optimizer = KNNFewShot(
            k=3,  # Use 3 nearest neighbors
            metric=self._create_adaptive_metric(examples)
        )

        # Compile with original 10 examples
        fine_tuned = knn_optimizer.compile(program, trainset=examples)

        # Add self-reflection capability
        reflective_wrapper = ReflectiveWrapper(fine_tuned)
        reflective_program = reflective_wrapper.compile(
            trainset=examples,
            reflection_steps=2
        )

        return reflective_program
```

## Real-World Application: Best-in-Class IR with 10 Labels

### Case Study Implementation

```python
class BestInClassIRWith10Labels:
    """Complete implementation of IR system trained with only 10 labels"""

    def __init__(self, document_collection: List[str]):
        self.documents = document_collection
        self.retriever = None
        self.training_history = []

    def train_and_deploy(self,
                        labeled_examples: List[dspy.Example]) -> Dict[str, Any]:
        """Train and deploy IR system with only 10 labeled examples"""

        if len(labeled_examples) != 10:
            raise ValueError("This system requires exactly 10 labeled examples")

        # Phase 1: Setup
        self._setup_initial_components()

        # Phase 2: Train with extreme few-shot learning
        training_results = self._train_with_10_examples(labeled_examples)

        # Phase 3: Optimize prompts as hyperparameters
        optimized_retriever = self._optimize_prompt_hyperparameters(
            labeled_examples
        )

        # Phase 4: Deploy with confidence estimation
        deployed_system = self._deploy_with_confidence_estimation(
            optimized_retriever
        )

        return {
            'training_results': training_results,
            'optimized_retriever': optimized_retriever,
            'deployed_system': deployed_system,
            'performance_metrics': self._measure_performance(deployed_system)
        }

    def _setup_initial_components(self):
        """Setup base IR components"""

        # Initialize base retriever with semantic search
        from dspy.retrieve import ColBERTv2Retriever

        self.base_retriever = ColBERTv2Retriever(
            k=20,  # Retrieve more candidates for re-ranking
            collection=self.documents
        )

        # Initialize query processor
        self.query_processor = dspy.ChainOfThought(
            "query -> processed_query, search_intent"
        )

        # Initialize document ranker
        self.document_ranker = dspy.Predict(
            "query, documents -> ranked_documents"
        )

    def _train_with_10_examples(self,
                               examples: List[dspy.Example]) -> Dict[str, Any]:
        """Train system using only 10 examples"""

        # Create training pipeline
        pipeline = TenExampleTrainingPipeline(
            task_type="information_retrieval"
        )

        # Define IR-specific signature
        ir_signature = dspy.Signature(
            "query, documents -> relevant_documents, relevance_scores"
        )

        # Train with minimal data
        training_results = pipeline.train_with_10_examples(
            examples, ir_signature
        )

        self.training_history.append(training_results)

        return training_results

    def _optimize_prompt_hyperparameters(self,
                                        examples: List[dspy.Example]) -> dspy.Module:
        """Optimize prompts as hyperparameters for IR task"""

        # Define search space for prompt hyperparameters
        search_space = {
            'instruction_templates': [
                "Rank these documents by relevance to the query",
                "Order documents from most to least relevant",
                "Select and rank the most relevant documents",
                "Identify the top documents that answer this query"
            ],
            'example_strategies': ['random', 'diverse', 'representative'],
            'formatting_patterns': [
                "Query: {query}\nDocuments:\n{documents}\nRanking:",
                "Q: {query}\nDocs: {documents}\nRelevant:"
            ],
            'reasoning_guidance': [
                "Consider semantic similarity and query-document matching",
                "Evaluate based on relevance, completeness, and authority",
                "Assess how well each document addresses the query"
            ],
            'task_decompositions': [
                ['direct_ranking'],
                ['query_understanding', 'document_analysis', 'ranking'],
                ['initial_filter', 'detailed_comparison', 'final_rank']
            ]
        }

        # Create IR-specific program
        base_ir_program = self._create_base_ir_program()

        # Optimize hyperparameters
        optimizer = PromptHyperparameterOptimizer(
            base_program=base_ir_program,
            metric_fn=self._ir_metric_function,
            search_space=search_space
        )

        # Use 8 examples for optimization, 2 for validation
        best_params = optimizer.optimize(
            trainset=examples[:8],
            valset=examples[8:],
            num_iterations=30
        )

        # Apply best parameters
        optimized_program = self._apply_hyperparameters(
            base_ir_program, best_params
        )

        return optimized_program

    def _create_base_ir_program(self) -> dspy.Module:
        """Create base IR program for optimization"""

        class BaseIRProgram(dspy.Module):
            def __init__(self):
                super().__init__()
                self.process_query = dspy.ChainOfThought(
                    "query -> processed_query, key_terms"
                )
                self.rank_documents = dspy.Predict(
                    "query, documents -> ranked_documents"
                )

            def forward(self, query: str, documents: List[str]):
                # Process the query
                processed = self.process_query(query=query)

                # Rank documents
                ranked = self.rank_documents(
                    query=processed.processed_query,
                    documents="\n".join(documents)
                )

                return dspy.Prediction(
                    ranked_documents=ranked.ranked_documents,
                    processed_query=processed.processed_query,
                    key_terms=processed.key_terms
                )

        return BaseIRProgram()
```

## Performance Analysis and Validation

### Measuring Success with Minimal Data

```python
def evaluate_ir_with_10_examples(trained_system,
                                test_queries: List[str],
                                ground_truth: Dict[str, List[int]]) -> Dict[str, float]:
    """Comprehensive evaluation of IR system trained with 10 examples"""

    metrics = {
        'precision_at_k': {},
        'recall_at_k': {},
        'ndcg': {},
        'mrr': 0.0,
        'confidence_calibration': 0.0
    }

    # Standard IR metrics
    for k in [1, 3, 5, 10]:
        precisions = []
        recalls = []

        for query in test_queries:
            # Get rankings from trained system
            result = trained_system(query=query, documents=all_documents)
            ranked_docs = parse_ranked_documents(result.ranked_documents)

            # Calculate precision@k
            relevant_retrieved = len(
                set(ranked_docs[:k]) & set(ground_truth[query])
            )
            precision = relevant_retrieved / k
            precisions.append(precision)

            # Calculate recall@k
            total_relevant = len(ground_truth[query])
            recall = relevant_retrieved / total_relevant
            recalls.append(recall)

        metrics['precision_at_k'][k] = np.mean(precisions)
        metrics['recall_at_k'][k] = np.mean(recalls)

    # NDCG calculation
    ndcg_scores = []
    for query in test_queries:
        result = trained_system(query=query, documents=all_documents)
        ndcg = calculate_ndcg(result, ground_truth[query])
        ndcg_scores.append(ndcg)
    metrics['ndcg']['mean'] = np.mean(ndcg_scores)

    # Mean Reciprocal Rank
    mrr_scores = []
    for query in test_queries:
        result = trained_system(query=query, documents=all_documents)
        mrr = calculate_mrr(result, ground_truth[query])
        mrr_scores.append(mrr)
    metrics['mrr'] = np.mean(mrr_scores)

    # Confidence calibration (how well confidence scores predict accuracy)
    calibration_score = calculate_confidence_calibration(
        trained_system, test_queries, ground_truth
    )
    metrics['confidence_calibration'] = calibration_score

    return metrics
```

## Key Insights and Best Practices

### Principles for Success with 10 Examples

1. **Prompt Quality Over Quantity**: With minimal data, the prompt becomes the primary source of task knowledge
2. **Meta-Learning is Essential**: Leverage knowledge from related tasks to compensate for data scarcity
3. **Strategic Data Augmentation**: Every augmentation must be carefully designed to add meaningful variation
4. **Confidence Estimation**: Critical when working with minimal training data
5. **Cross-Validation**: Essential to prevent overfitting with such small datasets

### Common Pitfalls and Solutions

| Pitfall | Solution |
|---------|----------|
| Overfitting to 10 examples | Use rigorous cross-validation and regularization |
| Poor prompt generalization | Optimize prompts as hyperparameters with diverse search |
| Catastrophic forgetting | Maintain meta-knowledge across updates |
| Evaluation bias | Use held-out data and multiple metrics |

## Next Steps

In this section, we've explored how prompts can be treated as auto-optimized hyperparameters, enabling training of sophisticated models with minimal data. We've seen how this approach makes it possible to train best-in-class IR models with only 10 labeled examples.

Next, we'll explore [Minimal Data Training Pipelines](21-minimal-data-pipelines.md), which extends these concepts to create robust training systems for any task with minimal labeled data.
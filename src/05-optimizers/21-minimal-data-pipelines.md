# Minimal Data Training Pipelines

## Introduction

In many real-world scenarios, we face the challenge of training sophisticated models with severely limited labeled data. Whether it's 10 examples for a new task, a handful of annotations for a specialized domain, or minimal feedback for a new application, we need robust training pipelines that can extract maximum learning signal from minimal data.

DSPy provides a comprehensive framework for building minimal data training pipelines that combine multiple optimization strategies, intelligent data augmentation, and sophisticated validation techniques. This section explores how to design, implement, and deploy these pipelines effectively.

## The Minimal Data Training Architecture

### Core Components

```
Minimal Data Pipeline
├── Data Analysis & Understanding
├── Strategic Data Augmentation
├── Multi-Strategy Optimization
├── Robust Validation
├── Confidence Estimation
└── Continuous Learning
```

### Pipeline Design Principles

1. **Data Efficiency**: Extract maximum value from each example
2. **Strategy Diversity**: Combine multiple complementary approaches
3. **Robust Validation**: Prevent overfitting with limited data
4. **Confidence Awareness**: Know when to trust model predictions
5. **Adaptability**: Learn from new data and feedback

## Building a Comprehensive Minimal Data Pipeline

### Base Pipeline Architecture

```python
import dspy
from typing import List, Dict, Any, Optional, Callable
import numpy as np
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

class DataAugmentationType(Enum):
    """Types of data augmentation strategies"""
    PARAPHRASE = "paraphrase"
    COUNTERFACTUAL = "counterfactual"
    TEMPLATE = "template"
    SEMANTIC = "semantic"
    SYNTACTIC = "syntactic"

class OptimizationStrategy(Enum):
    """Optimization strategies for minimal data"""
    PROMPT_OPTIMIZATION = "prompt_optimization"
    META_LEARNING = "meta_learning"
    ACTIVE_LEARNING = "active_learning"
    SELF_SUPERVISED = "self_supervised"
    HYBRID = "hybrid"

@dataclass
class PipelineConfig:
    """Configuration for minimal data training pipeline"""
    num_examples: int
    task_type: str
    domain: str
    augmentation_strategies: List[DataAugmentationType]
    optimization_strategies: List[OptimizationStrategy]
    validation_method: str
    confidence_threshold: float
    continuous_learning: bool

class MinimalDataTrainingPipeline:
    """Comprehensive pipeline for training with minimal data"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.pipeline_components = {}
        self.training_history = []
        self.performance_tracker = {}

    def execute_pipeline(self,
                        base_program: dspy.Module,
                        examples: List[dspy.Example],
                        evaluation_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute complete minimal data training pipeline"""

        print(f"Executing pipeline for {self.config.task_type} with {len(examples)} examples")

        results = {
            'pipeline_config': self.config,
            'execution_timestamp': datetime.now(),
            'stages_completed': [],
            'final_model': None,
            'performance_metrics': {}
        }

        # Stage 1: Data Analysis and Understanding
        print("\n=== Stage 1: Data Analysis ===")
        data_analysis = self._analyze_training_data(examples)
        results['data_analysis'] = data_analysis
        results['stages_completed'].append('data_analysis')

        # Stage 2: Strategic Data Augmentation
        print("\n=== Stage 2: Data Augmentation ===")
        augmented_data = self._strategic_augmentation(examples, data_analysis)
        results['augmentation_stats'] = {
            'original_count': len(examples),
            'augmented_count': len(augmented_data),
            'augmentation_ratio': len(augmented_data) / len(examples)
        }
        results['stages_completed'].append('data_augmentation')

        # Stage 3: Multi-Strategy Optimization
        print("\n=== Stage 3: Multi-Strategy Optimization ===")
        optimization_results = self._multi_strategy_optimization(
            base_program, examples, augmented_data
        )
        results['optimization_results'] = optimization_results
        results['stages_completed'].append('multi_strategy_optimization')

        # Stage 4: Robust Validation and Selection
        print("\n=== Stage 4: Model Selection ===")
        final_model, validation_results = self._robust_validation_and_selection(
            optimization_results['models'], examples
        )
        results['final_model'] = final_model
        results['validation_results'] = validation_results
        results['stages_completed'].append('model_selection')

        # Stage 5: Confidence Estimation Integration
        print("\n=== Stage 5: Confidence Estimation ===")
        confident_model = self._add_confidence_estimation(final_model)
        results['confident_model'] = confident_model
        results['stages_completed'].append('confidence_estimation')

        # Stage 6: Performance Evaluation
        if evaluation_fn:
            print("\n=== Stage 6: Performance Evaluation ===")
            performance = self._comprehensive_evaluation(
                confident_model, evaluation_fn
            )
            results['performance_metrics'] = performance
            results['stages_completed'].append('performance_evaluation')

        # Record pipeline execution
        self.training_history.append(results)

        return results

    def _analyze_training_data(self, examples: List[dspy.Example]) -> Dict[str, Any]:
        """Comprehensive analysis of minimal training data"""

        analysis = {
            'example_count': len(examples),
            'input_patterns': {},
            'output_patterns': {},
            'complexity_distribution': {},
            'domain_features': set(),
            'data_quality': {},
            'potential_biases': [],
            'augmentation_opportunities': []
        }

        # Analyze each example
        for i, example in enumerate(examples):
            # Input analysis
            input_analysis = self._analyze_input_structure(example)
            for pattern, count in input_analysis.items():
                if pattern not in analysis['input_patterns']:
                    analysis['input_patterns'][pattern] = 0
                analysis['input_patterns'][pattern] += count

            # Output analysis
            output_analysis = self._analyze_output_structure(example)
            for pattern, count in output_analysis.items():
                if pattern not in analysis['output_patterns']:
                    analysis['output_patterns'][pattern] = 0
                analysis['output_patterns'][pattern] += count

            # Complexity assessment
            complexity = self._assess_example_complexity(example)
            if complexity not in analysis['complexity_distribution']:
                analysis['complexity_distribution'][complexity] = 0
            analysis['complexity_distribution'][complexity] += 1

            # Domain feature extraction
            domain_features = self._extract_domain_features(example)
            analysis['domain_features'].update(domain_features)

        # Data quality assessment
        analysis['data_quality'] = self._assess_data_quality(examples)

        # Identify augmentation opportunities
        analysis['augmentation_opportunities'] = self._identify_augmentation_opportunities(
            analysis
        )

        return analysis

    def _strategic_augmentation(self,
                                examples: List[dspy.Example],
                                analysis: Dict[str, Any]) -> List[dspy.Example]:
        """Strategically augment training data based on analysis"""

        augmented = examples.copy()
        augmentation_log = []

        for strategy in self.config.augmentation_strategies:
            print(f"Applying {strategy.value} augmentation...")

            if strategy == DataAugmentationType.PARAPHRASE:
                # Generate paraphrases for each example
                for example in examples:
                    paraphrases = self._generate_paraphrases(example, n=2)
                    for para in paraphrases:
                        augmented.append(para)
                    augmentation_log.append({
                        'strategy': 'paraphrase',
                        'original_example': str(example),
                        'generated_count': len(paraphrases)
                    })

            elif strategy == DataAugmentationType.COUNTERFACTUAL:
                # Generate counterfactual examples
                for example in examples:
                    if self._should_generate_counterfactual(example, analysis):
                        counterfactuals = self._generate_counterfactuals(example, n=1)
                        for cf in counterfactuals:
                            augmented.append(cf)
                        augmentation_log.append({
                            'strategy': 'counterfactual',
                            'original_example': str(example),
                            'generated_count': len(counterfactuals)
                        })

            elif strategy == DataAugmentationType.TEMPLATE:
                # Extract and apply templates
                templates = self._extract_templates_from_examples(examples)
                for template in templates:
                    template_examples = self._apply_template_variations(
                        template, examples, n=2
                    )
                    augmented.extend(template_examples)
                    augmentation_log.append({
                        'strategy': 'template',
                        'template': template,
                        'generated_count': len(template_examples)
                    })

            elif strategy == DataAugmentationType.SEMANTIC:
                # Semantic variations
                for example in examples:
                    semantic_variations = self._generate_semantic_variations(example)
                    augmented.extend(semantic_variations)
                    augmentation_log.append({
                        'strategy': 'semantic',
                        'original_example': str(example),
                        'generated_count': len(semantic_variations)
                    })

        # Quality control of augmented data
        filtered_augmented = self._quality_filter_augmentations(
            augmented, examples
        )

        self.pipeline_components['augmentation_log'] = augmentation_log

        return filtered_augmented

    def _multi_strategy_optimization(self,
                                   base_program: dspy.Module,
                                   original_examples: List[dspy.Example],
                                   augmented_examples: List[dspy.Example]) -> Dict[str, Any]:
        """Apply multiple optimization strategies"""

        optimization_results = {
            'models': {},
            'strategy_performance': {},
            'best_strategy': None,
            'ensemble_candidates': []
        }

        for strategy in self.config.optimization_strategies:
            print(f"\nApplying {strategy.value} optimization...")

            if strategy == OptimizationStrategy.PROMPT_OPTIMIZATION:
                model = self._prompt_optimization_pipeline(
                    base_program, original_examples, augmented_examples
                )

            elif strategy == OptimizationStrategy.META_LEARNING:
                model = self._meta_learning_pipeline(
                    base_program, original_examples
                )

            elif strategy == OptimizationStrategy.ACTIVE_LEARNING:
                model = self._active_learning_pipeline(
                    base_program, original_examples
                )

            elif strategy == OptimizationStrategy.SELF_SUPERVISED:
                model = self._self_supervised_pipeline(
                    base_program, augmented_examples
                )

            elif strategy == OptimizationStrategy.HYBRID:
                model = self._hybrid_optimization_pipeline(
                    base_program, original_examples, augmented_examples
                )

            # Evaluate strategy performance
            performance = self._evaluate_strategy_performance(
                model, original_examples
            )

            optimization_results['models'][strategy.value] = model
            optimization_results['strategy_performance'][strategy.value] = performance

        # Identify best performing strategy
        best_strategy = max(
            optimization_results['strategy_performance'].items(),
            key=lambda x: x[1]['mean_score']
        )
        optimization_results['best_strategy'] = best_strategy[0]

        # Identify ensemble candidates (strategies with complementary strengths)
        optimization_results['ensemble_candidates'] = self._identify_ensemble_candidates(
            optimization_results['strategy_performance']
        )

        return optimization_results

    def _prompt_optimization_pipeline(self,
                                    base_program: dspy.Module,
                                    original_examples: List[dspy.Example],
                                    augmented_examples: List[dspy.Example]) -> dspy.Module:
        """Complete prompt optimization pipeline"""

        # Step 1: Generate diverse prompt candidates
        prompt_candidates = self._generate_diverse_prompts(
            original_examples, self.config.task_type
        )

        # Step 2: Evaluate each prompt
        evaluated_prompts = []
        for prompt in prompt_candidates:
            # Apply prompt to program
            prompt_program = self._apply_prompt_to_program(base_program, prompt)

            # Evaluate using cross-validation
            cv_scores = self._cross_validate_minimal_data(
                prompt_program, original_examples
            )

            evaluated_prompts.append({
                'prompt': prompt,
                'cv_score': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'program': prompt_program
            })

        # Step 3: Select best prompts
        best_prompts = sorted(
            evaluated_prompts, key=lambda x: x['cv_score'], reverse=True
        )[:3]

        # Step 4: Create prompt ensemble
        ensemble_program = self._create_prompt_ensemble(
            [p['program'] for p in best_prompts]
        )

        # Step 5: Fine-tune with augmented data
        final_program = self._fine_tune_with_augmented_data(
            ensemble_program, augmented_examples
        )

        return final_program

    def _meta_learning_pipeline(self,
                              base_program: dspy.Module,
                              examples: List[dspy.Example]) -> dspy.Module:
        """Meta-learning pipeline for minimal data"""

        # Step 1: Identify related tasks
        related_tasks = self._discover_related_tasks(
            self.config.task_type, self.config.domain
        )

        # Step 2: Create meta-learner
        meta_learner = self._create_meta_learner(base_program)

        # Step 3: Meta-training on related tasks
        for task in related_tasks:
            task_examples = self._get_task_examples(task)
            meta_learner.meta_train(task, task_examples)

        # Step 4: Rapid adaptation to target task
        adapted_model = meta_learner.adapt(
            target_task=self.config.task_type,
            support_set=examples,
            adaptation_steps=min(10, len(examples))
        )

        return adapted_model

    def _active_learning_pipeline(self,
                                base_program: dspy.Module,
                                examples: List[dspy.Example]) -> dspy.Module:
        """Active learning pipeline for efficient data usage"""

        # Step 1: Initialize with current examples
        active_learner = ActiveLearningWrapper(base_program)
        active_learner.initialize(examples)

        # Step 2: Generate synthetic queries for active selection
        synthetic_pool = self._generate_synthetic_queries(examples, n=100)

        # Step 3: Iterative active learning
        for iteration in range(5):  # Limited iterations due to minimal data
            # Select most informative examples
            selected = active_learner.select_informative(
                synthetic_pool, n=3
            )

            # Simulate annotations (in practice, this would be human input)
            annotations = self._simulate_annotations(selected)

            # Update model
            active_learner.update(annotations)

        return active_learner.get_current_model()

    def _self_supervised_pipeline(self,
                                base_program: dspy.Module,
                                augmented_examples: List[dspy.Example]) -> dspy.Module:
        """Self-supervised learning pipeline"""

        # Step 1: Create self-supervised tasks
        self_supervised_tasks = self._create_self_supervised_tasks(augmented_examples)

        # Step 2: Pre-train on self-supervised tasks
        pretrained_model = self._pretrain_self_supervised(
            base_program, self_supervised_tasks
        )

        # Step 3: Fine-tune on original examples
        fine_tuned_model = self._fine_tune_with_minimal_data(
            pretrained_model, augmented_examples[:10]  # Use original 10
        )

        return fine_tuned_model

    def _hybrid_optimization_pipeline(self,
                                    base_program: dspy.Module,
                                    original_examples: List[dspy.Example],
                                    augmented_examples: List[dspy.Example]) -> dspy.Module:
        """Combine multiple optimization strategies"""

        hybrid_results = []

        # Apply prompt optimization
        prompt_optimized = self._prompt_optimization_pipeline(
            base_program, original_examples, augmented_examples
        )
        hybrid_results.append(('prompt_optimization', prompt_optimized))

        # Apply meta-learning
        meta_learned = self._meta_learning_pipeline(base_program, original_examples)
        hybrid_results.append(('meta_learning', meta_learned))

        # Create weighted ensemble
        weights = self._calculate_strategy_weights(hybrid_results, original_examples)
        ensemble = self._create_weighted_ensemble(hybrid_results, weights)

        return ensemble
```

## Domain-Specific Pipeline Configurations

### Pipeline for Text Classification

```python
def create_classification_pipeline(num_examples: int, domain: str):
    """Create pipeline optimized for text classification"""

    config = PipelineConfig(
        num_examples=num_examples,
        task_type="classification",
        domain=domain,
        augmentation_strategies=[
            DataAugmentationType.PARAPHRASE,
            DataAugmentationType.TEMPLATE,
            DataAugmentationType.COUNTERFACTUAL
        ],
        optimization_strategies=[
            OptimizationStrategy.PROMPT_OPTIMIZATION,
            OptimizationStrategy.META_LEARNING
        ],
        validation_method="stratified_cv",
        confidence_threshold=0.8,
        continuous_learning=True
    )

    return MinimalDataTrainingPipeline(config)

# Example usage
def train_sentiment_classifier_with_10_examples():
    """Train sentiment classifier with only 10 examples"""

    # 10 labeled sentiment examples
    examples = [
        dspy.Example(text="This product exceeded my expectations!", label="positive"),
        dspy.Example(text="Terrible service, would not recommend.", label="negative"),
        dspy.Example(text="It's okay, nothing special.", label="neutral"),
        # ... 7 more examples
    ]

    # Create pipeline
    pipeline = create_classification_pipeline(num_examples=10, domain="product_reviews")

    # Create base classifier
    base_classifier = dspy.Predict("text -> sentiment")

    # Define evaluation function
    def evaluate_classifier(model):
        test_cases = [
            ("Amazing quality!", "positive"),
            ("Worst purchase ever.", "negative"),
            ("It's fine.", "neutral")
        ]
        correct = 0
        for text, true_label in test_cases:
            pred = model(text=text)
            if pred.sentiment.lower() == true_label:
                correct += 1
        return correct / len(test_cases)

    # Execute pipeline
    results = pipeline.execute_pipeline(
        base_program=base_classifier,
        examples=examples,
        evaluation_fn=evaluate_classifier
    )

    return results['confident_model']
```

### Pipeline for Information Retrieval

```python
def create_ir_pipeline(num_examples: int, domain: str):
    """Create pipeline optimized for information retrieval"""

    config = PipelineConfig(
        num_examples=num_examples,
        task_type="information_retrieval",
        domain=domain,
        augmentation_strategies=[
            DataAugmentationType.PARAPHRASE,
            DataAugmentationType.SEMANTIC
        ],
        optimization_strategies=[
            OptimizationStrategy.SELF_SUPERVISED,
            OptimizationStrategy.ACTIVE_LEARNING,
            OptimizationStrategy.HYBRID
        ],
        validation_method="leave_one_out",
        confidence_threshold=0.7,
        continuous_learning=True
    )

    return MinimalDataTrainingPipeline(config)

# Example usage
def train_ir_system_with_minimal_judgments():
    """Train IR system with minimal relevance judgments"""

    # 10 query-document relevance judgments
    judgments = [
        dspy.Example(
            query="machine learning tutorials",
            document="Complete guide to ML for beginners",
            relevance=2
        ),
        # ... 9 more judgments
    ]

    # Create IR pipeline
    pipeline = create_ir_pipeline(num_examples=10, domain="educational_content")

    # Create base IR model
    base_ir = dspy.Predict("query, document -> relevance_score")

    # Execute pipeline
    results = pipeline.execute_pipeline(
        base_program=base_ir,
        examples=judgments
    )

    return results['confident_model']
```

## Advanced Pipeline Features

### Continuous Learning Integration

```python
class ContinuousLearningWrapper:
    """Wrapper for continuous learning with minimal data"""

    def __init__(self, initial_model, pipeline_config):
        self.model = initial_model
        self.config = pipeline_config
        self.feedback_buffer = []
        self.performance_history = []

    def update_with_feedback(self, feedback_examples: List[dspy.Example]):
        """Update model with new feedback"""

        # Add to feedback buffer
        self.feedback_buffer.extend(feedback_examples)

        # Periodic retraining
        if len(self.feedback_buffer) >= 5:  # Retrain every 5 new examples
            print("Retraining with new feedback...")

            # Combine original and feedback data
            all_examples = self.original_examples + self.feedback_buffer

            # Create temporary pipeline for retraining
            temp_pipeline = MinimalDataTrainingPipeline(self.config)
            retrain_results = temp_pipeline.execute_pipeline(
                base_program=self.model.__class__(),
                examples=all_examples
            )

            # Update model
            self.model = retrain_results['confident_model']

            # Clear feedback buffer
            self.feedback_buffer = []

        return self.model

    def predict_with_confidence(self, **kwargs):
        """Make prediction with confidence estimation"""

        prediction = self.model(**kwargs)

        # Add confidence based on feedback history
        if len(self.performance_history) > 0:
            recent_performance = np.mean(self.performance_history[-10:])
            confidence = min(recent_performance, 0.95)
        else:
            confidence = self.config.confidence_threshold

        # Add confidence to prediction
        prediction.confidence = confidence

        return prediction
```

### Pipeline Monitoring and Analytics

```python
class PipelineMonitor:
    """Monitor and analyze pipeline performance"""

    def __init__(self):
        self.metrics_log = []
        self.alerts = []

    def monitor_pipeline_execution(self, pipeline_results: Dict[str, Any]):
        """Monitor pipeline execution and generate insights"""

        metrics = {
            'timestamp': pipeline_results['execution_timestamp'],
            'pipeline_config': pipeline_results['pipeline_config'],
            'stages_completed': pipeline_results['stages_completed'],
            'data_augmentation_ratio': pipeline_results.get(
                'augmentation_stats', {}
            ).get('augmentation_ratio', 1.0),
            'optimization_strategies_used': list(
                pipeline_results.get('optimization_results', {}).get('models', {}).keys()
            ),
            'performance_metrics': pipeline_results.get('performance_metrics', {})
        }

        self.metrics_log.append(metrics)

        # Generate insights and alerts
        insights = self._generate_insights(metrics)
        if insights:
            print("\n=== Pipeline Insights ===")
            for insight in insights:
                print(f"- {insight}")

        # Check for alerts
        alerts = self._check_alerts(metrics)
        self.alerts.extend(alerts)
        if alerts:
            print("\n⚠️  Alerts:")
            for alert in alerts:
                print(f"- {alert}")

        return insights, alerts

    def _generate_insights(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate insights from pipeline metrics"""

        insights = []

        # Augmentation insights
        if metrics['data_augmentation_ratio'] > 3:
            insights.append(
                f"High augmentation ratio ({metrics['data_augmentation_ratio']:.1f}) "
                "may introduce noise"
            )

        # Strategy insights
        if 'hybrid' in metrics['optimization_strategies_used']:
            insights.append(
                "Hybrid optimization selected - combining multiple strategies "
                "for robust performance"
            )

        # Performance insights
        perf = metrics.get('performance_metrics', {})
        if perf:
            if perf.get('mean_score', 0) > 0.9:
                insights.append("Excellent performance achieved!")
            elif perf.get('mean_score', 0) < 0.6:
                insights.append(
                    "Low performance detected - consider collecting more data "
                    "or trying different strategies"
                )

        return insights

    def _check_alerts(self, metrics: Dict[str, Any]) -> List[str]:
        """Check for issues requiring attention"""

        alerts = []

        # Data quality alerts
        if metrics['data_augmentation_ratio'] > 5:
            alerts.append(
                "Warning: Very high augmentation ratio - verify data quality"
            )

        # Performance alerts
        perf = metrics.get('performance_metrics', {})
        if perf.get('std_score', 0) > 0.2:
            alerts.append(
                "Warning: High performance variance - model may be unstable"
            )

        return alerts
```

## Best Practices for Pipeline Design

### 1. Start Simple, Iterate Complex
- Begin with basic prompt optimization
- Add complexity only if needed
- Always validate improvements

### 2. Understand Your Data
- Analyze patterns in minimal examples
- Identify domain-specific features
- Detect potential biases early

### 3. Choose Strategies Wisely
- Match strategies to task characteristics
- Consider computational constraints
- Prioritize based on expected impact

### 4. Validate Rigorously
- Use appropriate cross-validation for minimal data
- Monitor for overfitting
- Include confidence estimation

### 5. Plan for Continuous Learning
- Design for feedback incorporation
- Monitor performance over time
- Schedule periodic retraining

## Key Takeaways

1. **Holistic Approach**: Minimal data training requires comprehensive pipelines
2. **Strategy Combination**: Multiple strategies outperform single approaches
3. **Data Quality**: Augmentation must maintain high quality standards
4. **Robust Validation**: Essential when working with limited data
5. **Continuous Improvement**: Learning should continue after initial training

## Next Steps

This section covered comprehensive pipelines for minimal data training. These concepts integrate with:

- [Extreme Few-Shot Learning](../06-real-world-applications/11-extreme-few-shot-learning.md) for specific techniques
- [Prompt Hyperparameter Optimization](20-prompts-as-hyperparameters.md) for optimization details
- [BootstrapFewShot](02-bootstrapfewshot.md) for foundational optimization methods
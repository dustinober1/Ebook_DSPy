# Extreme Few-Shot Learning: Training with 10 Gold Labels

## Introduction

Traditional machine learning paradigms require thousands or millions of labeled examples to achieve good performance. However, in many real-world scenarios, we only have access to a handful of labeled examples—sometimes as few as 10. Extreme few-shot learning addresses this challenge by leveraging the power of language models and sophisticated optimization techniques to achieve remarkable performance with minimal data.

This section explores how DSPy enables training best-in-class models using only 10 gold-labeled examples, focusing on practical methodologies, optimization strategies, and real-world applications.

## The Challenge of Extreme Data Scarcity

### Why 10 Examples is Special

Training with exactly 10 examples presents unique challenges:

- **Statistical Significance**: 10 examples are often insufficient for traditional statistical methods
- **Overfitting Risk**: Models can easily memorize all 10 examples without learning generalizable patterns
- **Evaluation Difficulty**: Limited data makes it challenging to have separate train/validation/test splits
- **Pattern Discovery**: Extracting meaningful patterns from such small datasets requires specialized techniques

### The Zero-to-Ten Learning Spectrum

```
Zero-Shot (0 examples) ← One-Shot (1 example) ← Few-Shot (2-100) ← Full-Supervision (1000+)
                                        ↑
                                Extreme Few-Shot (10 examples)
```

Extreme few-shot learning occupies a critical middle ground between zero-shot and traditional few-shot learning, where we have just enough data to provide concrete examples but not enough for traditional training.

## DSPy's Approach to Extreme Few-Shot Learning

### Core Principles

1. **Prompt-First Learning**: Treat the prompt as the primary learning mechanism
2. **Meta-Learning Integration**: Leverage knowledge from related tasks and domains
3. **Active Prompt Optimization**: Systematically search for optimal prompt configurations
4. **Data Amplification**: Strategically expand the effective training set
5. **Confidence-Aware Inference**: Estimate uncertainty when working with minimal supervision

### The 10-Example Training Framework

```python
import dspy
from typing import List, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

class ExtremeFewShotStrategy(Enum):
    """Different strategies for extreme few-shot learning"""
    PROMPT_OPTIMIZATION = "prompt_optimization"
    META_LEARNING = "meta_learning"
    DATA_AMPLIFICATION = "data_amplification"
    HYBRID = "hybrid"

@dataclass
class TenExampleConfig:
    """Configuration for 10-example training"""
    strategy: ExtremeFewShotStrategy
    meta_tasks: List[str] = None
    augmentation_methods: List[str] = None
    confidence_threshold: float = 0.7
    validation_method: str = "cross_validation"

class ExtremeFewShotTrainer:
    """Specialized trainer for extreme few-shot scenarios"""

    def __init__(self,
                 base_model: str = "gpt-3.5-turbo",
                 config: TenExampleConfig = None):
        self.base_model = base_model
        self.config = config or TenExampleConfig(
            strategy=ExtremeFewShotStrategy.HYBRID
        )
        self.training_history = []

    def train_with_10_examples(self,
                              task_signature: dspy.Signature,
                              examples: List[dspy.Example],
                              domain_context: str = "") -> dspy.Module:
        """Train model using exactly 10 labeled examples"""

        if len(examples) != 10:
            raise ValueError("This trainer requires exactly 10 examples")

        print(f"Training {task_signature} with 10 examples using {self.config.strategy.value}")

        # Step 1: Analyze and preprocess the 10 examples
        analyzed_examples = self._analyze_examples(examples)

        # Step 2: Apply selected strategy
        if self.config.strategy == ExtremeFewShotStrategy.PROMPT_OPTIMIZATION:
            trained_model = self._prompt_optimization_training(
                task_signature, analyzed_examples, domain_context
            )
        elif self.config.strategy == ExtremeFewShotStrategy.META_LEARNING:
            trained_model = self._meta_learning_training(
                task_signature, analyzed_examples, domain_context
            )
        elif self.config.strategy == ExtremeFewShotStrategy.DATA_AMPLIFICATION:
            trained_model = self._data_amplification_training(
                task_signature, analyzed_examples, domain_context
            )
        else:  # HYBRID
            trained_model = self._hybrid_training(
                task_signature, analyzed_examples, domain_context
            )

        # Step 3: Validate with cross-validation
        validation_results = self._validate_with_10_examples(
            trained_model, examples
        )

        # Step 4: Add confidence estimation
        final_model = self._add_confidence_estimation(trained_model)

        # Record training history
        self.training_history.append({
            'timestamp': datetime.now(),
            'task': str(task_signature),
            'strategy': self.config.strategy.value,
            'validation_results': validation_results,
            'examples_analyzed': analyzed_examples
        })

        return final_model

    def _analyze_examples(self,
                         examples: List[dspy.Example]) -> Dict[str, Any]:
        """Deep analysis of the 10 examples to extract patterns"""

        analysis = {
            'input_patterns': [],
            'output_patterns': [],
            'complexity_distribution': {},
            'domain_features': set(),
            'example_diversity': 0.0,
            'required_reasoning': []
        }

        # Analyze each example
        for example in examples:
            # Extract input patterns
            input_analysis = self._analyze_input(example)
            analysis['input_patterns'].append(input_analysis)

            # Extract output patterns
            output_analysis = self._analyze_output(example)
            analysis['output_patterns'].append(output_analysis)

            # Analyze complexity
            complexity = self._assess_complexity(example)
            if complexity not in analysis['complexity_distribution']:
                analysis['complexity_distribution'][complexity] = 0
            analysis['complexity_distribution'][complexity] += 1

            # Extract domain features
            domain_feats = self._extract_domain_features(example)
            analysis['domain_features'].update(domain_feats)

            # Assess reasoning requirements
            reasoning = self._analyze_reasoning_requirements(example)
            analysis['required_reasoning'].append(reasoning)

        # Calculate example diversity
        analysis['example_diversity'] = self._calculate_diversity(
            analysis['input_patterns'], analysis['output_patterns']
        )

        return analysis

    def _prompt_optimization_training(self,
                                     task_signature: dspy.Signature,
                                     analyzed_examples: Dict[str, Any],
                                     domain_context: str) -> dspy.Module:
        """Train using systematic prompt optimization"""

        # Create base program
        base_program = self._create_base_program(task_signature)

        # Generate diverse prompt candidates
        prompt_candidates = self._generate_prompt_candidates(
            task_signature, analyzed_examples, domain_context
        )

        # Evaluate each prompt using cross-validation
        best_prompt = None
        best_score = 0.0

        for prompt in prompt_candidates:
            # Create program with current prompt
            temp_program = self._apply_prompt_to_program(
                base_program, prompt
            )

            # Evaluate using leave-two-out cross-validation
            cv_score = self._cross_validate_with_10_examples(
                temp_program, analyzed_examples['examples']
            )

            if cv_score > best_score:
                best_score = cv_score
                best_prompt = prompt

        # Create final program with best prompt
        final_program = self._apply_prompt_to_program(
            base_program, best_prompt
        )

        # Fine-tune with all 10 examples
        optimizer = BootstrapFewShot(
            metric=self._create_metric_from_analysis(analyzed_examples),
            max_bootstrapped_demos=3  # Limited by 10 examples
        )

        final_program = optimizer.compile(
            final_program,
            trainset=analyzed_examples['examples']
        )

        return final_program

    def _meta_learning_training(self,
                               task_signature: dspy.Signature,
                               analyzed_examples: Dict[str, Any],
                               domain_context: str) -> dspy.Module:
        """Train using meta-learning from related tasks"""

        # Step 1: Identify related meta-tasks
        if self.config.meta_tasks:
            meta_tasks = self.config.meta_tasks
        else:
            meta_tasks = self._discover_related_tasks(
                analyzed_examples, domain_context
            )

        # Step 2: Create meta-learner
        meta_learner = self._create_meta_learner(task_signature)

        # Step 3: Learn from meta-tasks
        for meta_task in meta_tasks:
            meta_examples = self._get_meta_examples(meta_task)
            meta_learner.adapt_to_task(meta_task, meta_examples)

        # Step 4: Rapid adaptation to target task
        adapted_program = meta_learner.rapid_adaptation(
            task_signature,
            analyzed_examples['examples'],
            adaptation_steps=5  # Very rapid adaptation
        )

        return adapted_program

    def _data_amplification_training(self,
                                     task_signature: dspy.Signature,
                                     analyzed_examples: Dict[str, Any],
                                     domain_context: str) -> dspy.Module:
        """Train by strategically amplifying the 10 examples"""

        # Step 1: Generate high-quality augmentations
        augmented_examples = []

        for example in analyzed_examples['examples']:
            # Paraphrase augmentation
            paraphrases = self._generate_paraphrases(example, n=3)
            augmented_examples.extend(paraphrases)

            # Counterfactual augmentation
            counterfactuals = self._generate_counterfactuals(example, n=2)
            augmented_examples.extend(counterfactuals)

            # Template-based augmentation
            templates = self._extract_templates(example)
            for template in templates:
                template_example = self._apply_template(template, example)
                augmented_examples.append(template_example)

        # Step 2: Quality control of augmentations
        quality_filtered = self._quality_filter_augmentations(
            augmented_examples,
            analyzed_examples
        )

        # Step 3: Train with amplified dataset
        base_program = self._create_base_program(task_signature)

        # Use BootstrapFewShot with augmented data
        optimizer = BootstrapFewShot(
            metric=self._create_robust_metric(),
            max_bootstrapped_demos=5  # Can use more with augmented data
        )

        trained_program = optimizer.compile(
            base_program,
            trainset=quality_filtered[:20]  # Limit to prevent noise
        )

        return trained_program

    def _hybrid_training(self,
                        task_signature: dspy.Signature,
                        analyzed_examples: Dict[str, Any],
                        domain_context: str) -> dspy.Module:
        """Combine multiple strategies for best performance"""

        results = {}

        # Try prompt optimization
        print("Attempting prompt optimization...")
        try:
            results['prompt_opt'] = self._prompt_optimization_training(
                task_signature, analyzed_examples, domain_context
            )
        except Exception as e:
            print(f"Prompt optimization failed: {e}")

        # Try meta-learning
        print("Attempting meta-learning...")
        try:
            results['meta_learning'] = self._meta_learning_training(
                task_signature, analyzed_examples, domain_context
            )
        except Exception as e:
            print(f"Meta-learning failed: {e}")

        # Try data amplification
        print("Attempting data amplification...")
        try:
            results['data_amp'] = self._data_amplification_training(
                task_signature, analyzed_examples, domain_context
            )
        except Exception as e:
            print(f"Data amplification failed: {e}")

        # Select best performer or create ensemble
        if len(results) == 1:
            return list(results.values())[0]
        elif len(results) > 1:
            # Create ensemble of best performers
            return self._create_ensemble(results, analyzed_examples)
        else:
            raise RuntimeError("All training strategies failed")

    def _create_ensemble(self,
                        trained_models: Dict[str, dspy.Module],
                        analyzed_examples: Dict[str, Any]) -> dspy.Module:
        """Create ensemble from multiple trained models"""

        class TenExampleEnsemble(dspy.Module):
            def __init__(self, models: Dict[str, dspy.Module]):
                super().__init__()
                self.models = models
                self.weights = self._calculate_model_weights(models, analyzed_examples)

            def forward(self, **kwargs):
                predictions = {}

                # Get predictions from each model
                for name, model in self.models.items():
                    pred = model(**kwargs)
                    predictions[name] = pred

                # Weighted combination
                final_prediction = self._combine_predictions(
                    predictions, self.weights
                )

                return final_prediction

        return TenExampleEnsemble(trained_models)
```

## Practical Applications

### Application 1: Text Classification with 10 Labels

```python
def train_classifier_with_10_examples():
    """Example: Train a text classifier with only 10 labeled examples"""

    # Create 10 labeled examples for sentiment analysis
    examples = [
        dspy.Example(
            text="This movie was absolutely fantastic! The acting was superb.",
            sentiment="positive"
        ),
        dspy.Example(
            text="I hated every minute of this film. Complete waste of time.",
            sentiment="negative"
        ),
        dspy.Example(
            text="The product works as described. Nothing special but does the job.",
            sentiment="neutral"
        ),
        dspy.Example(
            text="Outstanding service! They went above and beyond expectations.",
            sentiment="positive"
        ),
        dspy.Example(
            text="Disappointing experience. The quality was much lower than promised.",
            sentiment="negative"
        ),
        dspy.Example(
            text="It's okay. Not great, not terrible. Just average.",
            sentiment="neutral"
        ),
        dspy.Example(
            text="Absolutely love this! Best purchase I've made all year.",
            sentiment="positive"
        ),
        dspy.Example(
            text="Terrible customer service. They don't care about customers at all.",
            sentiment="negative"
        ),
        dspy.Example(
            text="The item arrived on time and matches the description.",
            sentiment="neutral"
        ),
        dspy.Example(
            text="Exceeded all my expectations! Highly recommend to everyone.",
            sentiment="positive"
        )
    ]

    # Configure for extreme few-shot learning
    config = TenExampleConfig(
        strategy=ExtremeFewShotStrategy.HYBRID,
        augmentation_methods=['paraphrase', 'counterfactual'],
        confidence_threshold=0.8
    )

    # Initialize trainer
    trainer = ExtremeFewShotTrainer(config=config)

    # Define task signature
    sentiment_signature = dspy.Signature(
        "text -> sentiment"
    )

    # Train with 10 examples
    classifier = trainer.train_with_10_examples(
        task_signature=sentiment_signature,
        examples=examples,
        domain_context="sentiment analysis of product reviews"
    )

    return classifier

# Test the classifier
def test_classifier(classifier, test_texts):
    """Test the trained classifier on new examples"""

    for text in test_texts:
        result = classifier(text=text)
        print(f"Text: {text}")
        print(f"Predicted Sentiment: {result.sentiment}")
        print(f"Confidence: {result.get('confidence', 'N/A')}")
        print("-" * 50)
```

### Application 2: Question Answering with 10 Examples

```python
def train_qa_with_10_examples():
    """Train a QA system with only 10 question-answer pairs"""

    # 10 example question-answer pairs
    qa_examples = [
        dspy.Example(
            question="What is the capital of France?",
            context="France is a country in Western Europe. Its largest city and capital is Paris.",
            answer="Paris"
        ),
        dspy.Example(
            question="Who wrote Romeo and Juliet?",
            context="Romeo and Juliet is a tragedy written by William Shakespeare early in his career.",
            answer="William Shakespeare"
        ),
        dspy.Example(
            question="What is photosynthesis?",
            context="Photosynthesis is the process used by plants to convert light energy into chemical energy.",
            answer="The process used by plants to convert light energy into chemical energy"
        ),
        dspy.Example(
            question="When was the Declaration of Independence signed?",
            context="The Declaration of Independence was signed on July 4, 1776, by representatives of the 13 colonies.",
            answer="July 4, 1776"
        ),
        dspy.Example(
            question="What is H2O?",
            context="H2O is the chemical formula for water, consisting of two hydrogen atoms and one oxygen atom.",
            answer="Water"
        ),
        dspy.Example(
            question="Who painted the Mona Lisa?",
            context="The Mona Lisa was painted by Leonardo da Vinci between 1503 and 1519.",
            answer="Leonardo da Vinci"
        ),
        dspy.Example(
            question="What is the largest planet in our solar system?",
            context="Jupiter is the largest planet in our solar system, with a mass greater than all other planets combined.",
            answer="Jupiter"
        ),
        dspy.Example(
            question="What year did World War II end?",
            context="World War II ended in 1945 after the surrender of Germany and Japan.",
            answer="1945"
        ),
        dspy.Example(
            question="What is DNA?",
            context="DNA (deoxyribonucleic acid) is the molecule that carries genetic instructions for life.",
            answer="The molecule that carries genetic instructions for life"
        ),
        dspy.Example(
            question="How many continents are there?",
            context="There are seven continents on Earth: Asia, Africa, North America, South America, Antarctica, Europe, and Australia.",
            answer="Seven"
        )
    ]

    # Configure for extreme few-shot learning
    config = TenExampleConfig(
        strategy=ExtremeFewShotStrategy.META_LEARNING,
        meta_tasks=['reading_comprehension', 'fact_extraction'],
        validation_method="leave_one_out"
    )

    # Train QA system
    trainer = ExtremeFewShotTrainer(config=config)

    qa_signature = dspy.Signature(
        "question, context -> answer"
    )

    qa_system = trainer.train_with_10_examples(
        task_signature=qa_signature,
        examples=qa_examples,
        domain_context="factual question answering"
    )

    return qa_system
```

### Application 3: Named Entity Recognition with 10 Examples

```python
def train_ner_with_10_examples():
    """Train NER system with only 10 labeled examples"""

    # 10 examples with entities labeled
    ner_examples = [
        dspy.Example(
            text="Apple Inc. announced their new iPhone at the Cupertino conference yesterday.",
            entities="[{'type': 'ORG', 'text': 'Apple Inc.'}, {'type': 'PRODUCT', 'text': 'iPhone'}, {'type': 'LOC', 'text': 'Cupertino'}, {'type': 'TIME', 'text': 'yesterday'}]"
        ),
        dspy.Example(
            text="Dr. Sarah Johnson from Harvard Medical School published her research in Nature Medicine.",
            entities="[{'type': 'PERSON', 'text': 'Dr. Sarah Johnson'}, {'type': 'ORG', 'text': 'Harvard Medical School'}, {'type': 'JOURNAL', 'text': 'Nature Medicine'}]"
        ),
        dspy.Example(
            text="Microsoft acquired GitHub for $7.5 billion in 2018.",
            entities="[{'type': 'ORG', 'text': 'Microsoft'}, {'type': 'ORG', 'text': 'GitHub'}, {'type': 'MONEY', 'text': '$7.5 billion'}, {'type': 'DATE', 'text': '2018'}]"
        ),
        dspy.Example(
            text="The Eiffel Tower in Paris was built in 1889 and stands 324 meters tall.",
            entities="[{'type': 'LANDMARK', 'text': 'Eiffel Tower'}, {'type': 'LOC', 'text': 'Paris'}, {'type': 'DATE', 'text': '1889'}, {'type': 'MEASURE', 'text': '324 meters'}]"
        ),
        dspy.Example(
            text="Tesla's Model 3 costs $35,000 and has a range of 250 miles.",
            entities="[{'type': 'ORG', 'text': 'Tesla'}, {'type': 'PRODUCT', 'text': 'Model 3'}, {'type': 'MONEY', 'text': '$35,000'}, {'type': 'MEASURE', 'text': '250 miles'}]"
        ),
        dspy.Example(
            text="Barack Obama was the 44th President of the United States from 2009 to 2017.",
            entities="[{'type': 'PERSON', 'text': 'Barack Obama'}, {'type': 'ORDINAL', 'text': '44th'}, {'type': 'TITLE', 'text': 'President'}, {'type': 'GPE', 'text': 'United States'}, {'type': 'DATE', 'text': '2009 to 2017'}]"
        ),
        dspy.Example(
            text="The COVID-19 pandemic began in Wuhan, China in December 2019.",
            entities="[{'type': 'DISEASE', 'text': 'COVID-19'}, {'type': 'EVENT', 'text': 'pandemic'}, {'type': 'GPE', 'text': 'Wuhan'}, {'type': 'GPE', 'text': 'China'}, {'type': 'DATE', 'text': 'December 2019'}]"
        ),
        dspy.Example(
            text="Amazon Web Services launched in 2006 and is now worth over $80 billion.",
            entities="[{'type': 'ORG', 'text': 'Amazon Web Services'}, {'type': 'DATE', 'text': '2006'}, {'type': 'MONEY', 'text': '$80 billion'}]"
        ),
        dspy.Example(
            text="The FIFA World Cup 2022 was held in Qatar and Argentina won the championship.",
            entities="[{'type': 'EVENT', 'text': 'FIFA World Cup 2022'}, {'type': 'DATE', 'text': '2022'}, {'type': 'GPE', 'text': 'Qatar'}, {'type': 'GPE', 'text': 'Argentina'}]"
        ),
        dspy.Example(
            text="Google was founded by Larry Page and Sergey Brin in 1998 while they were PhD students at Stanford.",
            entities="[{'type': 'ORG', 'text': 'Google'}, {'type': 'PERSON', 'text': 'Larry Page'}, {'type': 'PERSON', 'text': 'Sergey Brin'}, {'type': 'DATE', 'text': '1998'}, {'type': 'ORG', 'text': 'Stanford'}]"
        )
    ]

    # Configure for data amplification (NER benefits from augmentation)
    config = TenExampleConfig(
        strategy=ExtremeFewShotStrategy.DATA_AMPLIFICATION,
        augmentation_methods=['entity_replacement', 'template_variation'],
        confidence_threshold=0.75
    )

    # Train NER system
    trainer = ExtremeFewShotTrainer(config=config)

    ner_signature = dspy.Signature(
        "text -> entities"
    )

    ner_system = trainer.train_with_10_examples(
        task_signature=ner_signature,
        examples=ner_examples,
        domain_context="named entity recognition across multiple entity types"
    )

    return ner_system
```

## Evaluation and Validation Strategies

### Cross-Validation with 10 Examples

```python
def cross_validate_with_10_examples(model,
                                   examples: List[dspy.Example],
                                   k: int = 5) -> Dict[str, float]:
    """Perform k-fold cross-validation with only 10 examples"""

    # Use leave-two-out cross-validation for 10 examples
    scores = []

    for i in range(len(examples)):
        for j in range(i+1, len(examples)):
            # Create test set with 2 examples
            test_set = [examples[i], examples[j]]

            # Create train set with remaining 8 examples
            train_set = [ex for idx, ex in enumerate(examples)
                        if idx not in [i, j]]

            # Train on train set
            temp_model = train_temporary_model(train_set)

            # Evaluate on test set
            test_score = evaluate_model(temp_model, test_set)
            scores.append(test_score)

            # Stop after k folds
            if len(scores) >= k:
                break
        if len(scores) >= k:
            break

    return {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'scores': scores,
        'fold_count': len(scores)
    }
```

## Best Practices for 10-Example Training

### DO's

1. **Diverse Example Selection**: Choose 10 examples that cover different aspects of the task
2. **Quality Over Quantity**: Ensure each example is high-quality and correctly labeled
3. **Meta-Leverage**: Use knowledge from related tasks whenever possible
4. **Confidence Estimation**: Always include confidence scores in predictions
5. **Rigorous Validation**: Use cross-validation to prevent overfitting

### DON'Ts

1. **Don't Overfit**: Be cautious of models that perform perfectly on training data
2. **Don't Ignore Domain**: Even minimal domain context can significantly improve performance
3. **Don't Skip Validation**: Always validate using held-out examples
4. **Don't Trust Single Metrics**: Use multiple evaluation metrics
5. **Don't Forget Uncertainty**: Acknowledge and quantify prediction uncertainty

## Key Takeaways

1. **10 Examples Can Be Enough**: With the right techniques, 10 examples can train effective models
2. **Strategy Selection Matters**: Different tasks benefit from different few-shot strategies
3. **Quality Trumps Quantity**: The quality and diversity of examples is more important than the number
4. **Meta-Knowledge is Critical**: Leveraging related tasks and domains is essential
5. **Confidence Estimation is Necessary**: Always know when to trust (or not trust) predictions

## Next Steps

This section demonstrated how to train sophisticated models with only 10 labeled examples. The next section, [IR Model Training from Scratch](12-ir-model-training-scratch.md), explores how these extreme few-shot techniques can be applied to build complete information retrieval systems from minimal supervision.
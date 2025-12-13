# Exercises: Prompts as Auto-Optimized Hyperparameters

These exercises will help you master the concepts of treating prompts as auto-optimized hyperparameters and training models with minimal data.

## Exercise 1: Understanding Prompt Hyperparameters

### Objective
Understand how different prompt components act as hyperparameters that can be optimized.

### Tasks

1. **Identify Hyperparameter Types**
   - Given a prompt, identify which parts are:
     - Instruction templates
     - Formatting patterns
     - Few-shot examples
     - Reasoning guidance

   Example prompt:
   ```
   Task: Classify the sentiment of the following review.

   Instructions:
   - Read the review carefully
   - Consider both positive and negative indicators
   - Output only "positive", "negative", or "neutral"

   Example:
   Review: "This product is amazing!"
   Classification: positive

   Now classify:
   Review: "{review_text}"
   Classification:
   ```

2. **Create Hyperparameter Variations**
   - Create 3 different variations of the above prompt
   - Each variation should modify a different hyperparameter aspect
   - Explain which hyperparameter you're changing and why

### Solution

```python
# Solution for Task 1
def analyze_prompt_hyperparameters(prompt_text):
    """Analyze a prompt to identify hyperparameter components"""

    hyperparameters = {
        'instruction_template': [],
        'formatting_pattern': [],
        'few_shot_examples': [],
        'reasoning_guidance': []
    }

    # Look for instruction patterns
    if "Instructions:" in prompt_text:
        instructions_start = prompt_text.find("Instructions:")
        instructions_end = prompt_text.find("\n\n", instructions_start)
        instructions = prompt_text[instructions_start:instructions_end]
        hyperparameters['instruction_template'].append(instructions)

    # Look for examples
    if "Example:" in prompt_text:
        example_start = prompt_text.find("Example:")
        example_end = prompt_text.find("Now classify:")
        example = prompt_text[example_start:example_end]
        hyperparameters['few_shot_examples'].append(example)

    # Look for formatting patterns
    if 'Review: "' in prompt_text and 'Classification:' in prompt_text:
        pattern = 'Review: "{text}"\nClassification: {label}'
        hyperparameters['formatting_pattern'].append(pattern)

    # Look for reasoning guidance
    reasoning_patterns = ["consider", "think about", "evaluate"]
    for pattern in reasoning_patterns:
        if pattern in prompt_text.lower():
            hyperparameters['reasoning_guidance'].append(pattern)

    return hyperparameters

# Solution for Task 2
def create_prompt_variations(base_prompt):
    """Create different variations of a prompt by changing hyperparameters"""

    variations = {}

    # Variation 1: Change instruction template (more conversational)
    variations['conversational'] = """
    Hey there! I need you to help me figure out the sentiment of a product review.

    Here's what you should do:
    1. Read the review like a helpful friend
    2. Look for emotional words and tone
    3. Tell me if it's positive, negative, or neutral

    Here's an example:
    Review: "This product is amazing!"
    My take: positive

    What about this one?
    Review: "{review_text}"
    My take:
    """

    # Variation 2: Change formatting pattern (JSON output)
    variations['json_format'] = """
    Task: Sentiment Classification

    Instructions:
    Analyze the review and classify sentiment.

    Output format:
    {{
        "sentiment": "positive|negative|neutral",
        "confidence": 0.0-1.0,
        "key_phrases": ["phrase1", "phrase2"]
    }}

    Example:
    Input: "This product is amazing!"
    Output: {{
        "sentiment": "positive",
        "confidence": 0.95,
        "key_phrases": ["amazing"]
    }}

    Input: "{review_text}"
    Output:
    """

    # Variation 3: Change reasoning guidance (step-by-step)
    variations['step_by_step'] = """
    Sentiment Analysis Task

    Step 1: Identify positive words/phrases in the review
    Step 2: Identify negative words/phrases in the review
    Step 3: Compare positive and negative indicators
    Step 4: Determine overall sentiment
    Step 5: Provide classification

    Review: "{review_text}"

    Step 1: Positive indicators:
    Step 2: Negative indicators:
    Step 3: Comparison:
    Step 4: Overall sentiment:
    Classification:
    """

    return variations
```

## Exercise 2: Prompt Hyperparameter Optimization

### Objective
Implement a basic prompt hyperparameter optimizer that searches for the best prompt configuration.

### Tasks

1. **Create a Search Space**
   - Define a search space for prompt hyperparameters
   - Include at least 3 variations for each hyperparameter type

2. **Implement Random Search**
   - Create a function that samples random hyperparameter combinations
   - Generate 10 different prompt configurations

3. **Evaluate Prompt Configurations**
   - Use a small validation set (5 examples)
   - Implement a simple evaluation metric
   - Find the best performing configuration

### Solution

```python
import random
import dspy
from typing import Dict, List, Any

class PromptHyperparameterSearch:
    def __init__(self):
        self.search_space = {
            'instruction_templates': [
                "Classify the sentiment of the following text.",
                "Determine if the sentiment is positive, negative, or neutral.",
                "Analyze the emotional tone of this text and classify it."
            ],
            'formatting_patterns': [
                'Text: "{text}"\nSentiment: {label}',
                'Input: {text}\nOutput: {label}',
                'Review: {text}\nClassification: {label}'
            ],
            'reasoning_guidance': [
                "Consider the overall tone and emotion.",
                "Look for specific sentiment-bearing words.",
                "Evaluate both positive and negative indicators."
            ],
            'include_examples': [True, False]
        }

    def sample_configuration(self):
        """Sample a random configuration from the search space"""
        return {
            'instruction_template': random.choice(self.search_space['instruction_templates']),
            'formatting_pattern': random.choice(self.search_space['formatting_patterns']),
            'reasoning_guidance': random.choice(self.search_space['reasoning_guidance']),
            'include_examples': random.choice(self.search_space['include_examples'])
        }

    def build_prompt_from_config(self, config):
        """Build a complete prompt from hyperparameter configuration"""
        prompt = f"{config['instruction_template']}\n\n"

        if config['reasoning_guidance']:
            prompt += f"Guidance: {config['reasoning_guidance']}\n\n"

        if config['include_examples']:
            prompt += """Example:
Text: "This product is wonderful!"
Sentiment: positive

Example:
Text: "Terrible experience, would not recommend."
Sentiment: negative

"""

        prompt += f'\n{config["formatting_pattern"].format(text="{input_text}", label="{output_label}")}'

        return prompt

    def evaluate_configuration(self, config, validation_set):
        """Evaluate a prompt configuration on validation set"""
        correct = 0
        total = len(validation_set)

        for example in validation_set:
            # Build prompt with this example
            prompt = self.build_prompt_from_config(config)

            # Create temporary model with this prompt
            temp_model = dspy.Predict(prompt)

            # Make prediction
            prediction = temp_model(input_text=example.text)

            # Check correctness
            if prediction.output_label.lower() == example.sentiment.lower():
                correct += 1

        return correct / total

    def search_best_configuration(self, validation_set, num_samples=20):
        """Search for best configuration using random search"""
        best_config = None
        best_score = 0.0
        results = []

        for i in range(num_samples):
            config = self.sample_configuration()
            score = self.evaluate_configuration(config, validation_set)

            results.append({
                'config': config,
                'score': score,
                'iteration': i
            })

            if score > best_score:
                best_score = score
                best_config = config

            print(f"Iteration {i+1}/{num_samples}: Score = {score:.3f}")

        return {
            'best_config': best_config,
            'best_score': best_score,
            'all_results': results
        }

# Example validation set
validation_examples = [
    dspy.Example(text="I love this product!", sentiment="positive"),
    dspy.Example(text="This is terrible.", sentiment="negative"),
    dspy.Example(text="It's okay, nothing special.", sentiment="neutral"),
    dspy.Example(text="Amazing quality and fast shipping!", sentiment="positive"),
    dspy.Example(text="Worst purchase ever made.", sentiment="negative")
]

# Run the search
searcher = PromptHyperparameterSearch()
results = searcher.search_best_configuration(validation_examples)

print(f"\nBest configuration found with score: {results['best_score']:.3f}")
print("Best configuration:")
for key, value in results['best_config'].items():
    print(f"  {key}: {value}")
```

## Exercise 3: Training with 10 Examples

### Objective
Implement an extreme few-shot learning system that trains effectively with only 10 labeled examples.

### Tasks

1. **Create a 10-Example Dataset**
   - Design a task (e.g., intent classification, entity extraction)
   - Create exactly 10 high-quality labeled examples
   - Ensure diversity in the examples

2. **Implement Data Augmentation**
   - Create 3 different augmentation strategies
   - Apply them to expand your 10 examples
   - Maintain quality control

3. **Train and Evaluate**
   - Train a model using the augmented data
   - Evaluate on a separate test set
   - Measure performance improvement over baseline

### Solution

```python
import dspy
from typing import List, Dict, Any

class TenExampleTrainer:
    def __init__(self, task_type: str):
        self.task_type = task_type
        self.augmented_data = []

    def create_intent_classification_examples(self):
        """Create 10 diverse intent classification examples"""
        return [
            dspy.Example(
                text="What's the weather like today?",
                intent="weather_query"
            ),
            dspy.Example(
                text="Set an alarm for 7 AM",
                intent="alarm_setting"
            ),
            dspy.Example(
                text="Play some jazz music",
                intent="music_control"
            ),
            dspy.Example(
                text="Remind me to buy milk tomorrow",
                intent="reminder_creation"
            ),
            dspy.Example(
                text="How do I get to the airport?",
                intent="navigation_request"
            ),
            dspy.Example(
                text="Turn off the lights",
                intent="smart_home_control"
            ),
            dspy.Example(
                text="What time is it in Tokyo?",
                intent="time_query"
            ),
            dspy.Example(
                text="Book a table for two at 8 PM",
                intent="reservation"
            ),
            dspy.Example(
                text="Send a message to John",
                intent="communication"
            ),
            dspy.Example(
                text="Tell me a joke",
                intent="entertainment"
            )
        ]

    def augment_with_paraphrases(self, examples, n_per_example=2):
        """Augment data with paraphrases"""
        augmented = []

        paraphraser = dspy.Predict("text -> paraphrase")

        for example in examples:
            augmented.append(example)

            for _ in range(n_per_example):
                try:
                    para_result = paraphraser(text=example.text)
                    new_example = example.with_inputs(
                        text=para_result.paraphrase
                    )
                    augmented.append(new_example)
                except:
                    # Fallback to simple variation
                    simple_var = self._create_simple_variation(example)
                    augmented.append(simple_var)

        return augmented

    def augment_with_slot_filling(self, examples):
        """Augment by filling slots in templates"""
        augmented = examples.copy()

        # Extract templates
        templates = {
            "weather_query": ["What's the weather like in {location}?",
                              "How's the weather {time}?"],
            "alarm_setting": ["Set an alarm for {time}",
                             "Wake me up at {time}"],
            "music_control": ["Play {genre} music",
                             "Put on some {artist}"],
            # ... more templates
        }

        # Generate variations
        slots = {
            'location': ['New York', 'London', 'Paris', 'Tokyo'],
            'time': ['tomorrow', 'in 5 minutes', 'at noon', 'tonight'],
            'genre': ['rock', 'pop', 'classical', 'jazz', 'electronic'],
            'artist': ['The Beatles', 'Taylor Swift', 'Mozart', 'Drake']
        }

        for example in examples:
            intent = example.intent
            if intent in templates:
                for template in templates[intent]:
                    for slot_name, slot_values in slots.items():
                        if f'{{{slot_name}}}' in template:
                            for value in slot_values[:2]:  # Limit variations
                                filled = template.replace(f'{{{slot_name}}}', value)
                                new_example = dspy.Example(
                                    text=filled,
                                    intent=intent
                                )
                                augmented.append(new_example)

        return augmented

    def augment_with_context_addition(self, examples):
        """Add contextual variations"""
        augmented = examples.copy()

        contexts = [
            "Hey assistant, ",
            "Can you help me? ",
            "I need to ",
            "Please ",
            ""
        ]

        politeness = [
            "",
            " please",
            ", thanks"
        ]

        for example in examples:
            for ctx in contexts[:3]:  # Limit to avoid too much expansion
                for polite in politeness:
                    new_text = ctx + example.text + polite
                    new_example = example.with_inputs(
                        text=new_text
                    )
                    augmented.append(new_example)

        return augmented

    def _create_simple_variation(self, example):
        """Create simple variations when paraphrasing fails"""
        text = example.text

        # Simple variations
        if "?" in text:
            variation = text.replace("?", "!")
        elif "!" in text:
            variation = text.replace("!", "?")
        else:
            variation = text + ", please"

        return example.with_inputs(text=variation)

    def train_with_10_examples(self, examples):
        """Complete training pipeline with 10 examples"""

        print(f"Training {self.task_type} classifier with {len(examples)} examples")

        # Step 1: Apply augmentations
        print("\nStep 1: Data Augmentation")

        # Paraphrase augmentation
        para_augmented = self.augment_with_paraphrases(examples)
        print(f"Paraphrase augmentation: {len(examples)} → {len(para_augmented)} examples")

        # Slot filling augmentation
        slot_augmented = self.augment_with_slot_filling(examples)
        print(f"Slot filling augmentation: {len(examples)} → {len(slot_augmented)} examples")

        # Context augmentation
        context_augmented = self.augment_with_context_addition(examples)
        print(f"Context augmentation: {len(examples)} → {len(context_augmented)} examples")

        # Combine all augmentations
        all_augmented = list(set(
            para_augmented + slot_augmented + context_augmented
        ))
        print(f"Total unique examples after augmentation: {len(all_augmented)}")

        # Step 2: Create and train model
        print("\nStep 2: Model Training")

        # Create intent classifier
        classifier = dspy.Predict("text -> intent")

        # Use BootstrapFewShot with augmented data
        optimizer = dspy.BootstrapFewShot(
            metric=self._intent_accuracy,
            max_bootstrapped_demos=5
        )

        # Compile with augmented data (use subset to avoid noise)
        trained_classifier = optimizer.compile(
            classifier,
            trainset=all_augmented[:30]  # Use 30 best examples
        )

        return trained_classifier

    def _intent_accuracy(self, example, prediction, trace=None):
        """Calculate intent accuracy for evaluation"""
        return prediction.intent.lower() == example.intent.lower()

# Example usage
trainer = TenExampleTrainer(task_type="intent_classification")

# Create 10 examples
examples = trainer.create_intent_classification_examples()

# Train model
trained_model = trainer.train_with_10_examples(examples)

# Test on new examples
test_examples = [
    "What's the forecast like for tomorrow?",
    "Wake me up at 6:30 AM",
    "Play some classical music"
]

print("\nTest Results:")
for test_text in test_examples:
    result = trained_model(text=test_text)
    print(f"Text: {test_text}")
    print(f"Predicted Intent: {result.intent}")
    print("-" * 50)
```

## Exercise 4: IR Model Training from Scratch

### Objective
Build and train an information retrieval model using minimal relevance judgments.

### Tasks

1. **Prepare IR Dataset**
   - Create a small document collection (50-100 documents)
   - Create exactly 10 query-document relevance judgments
   - Ensure queries are diverse and cover different information needs

2. **Implement IR Components**
   - Create a query encoder
   - Create a document encoder
   - Implement a matching/ranking component

3. **Train with Minimal Data**
   - Use the 10 relevance judgments to train the model
   - Apply appropriate optimization strategies
   - Evaluate retrieval performance

### Solution

```python
import dspy
import numpy as np
from typing import List, Dict, Any, Tuple

class MinimalIRTrainer:
    def __init__(self):
        self.documents = []
        self.relevance_judgments = []
        self.trained_model = None

    def create_sample_documents(self):
        """Create a sample document collection"""
        return [
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "Python is a high-level programming language known for its simplicity and readability.",
            "Neural networks are computing systems inspired by biological neural networks in animal brains.",
            "Deep learning uses multiple layers to progressively extract higher-level features from raw input.",
            "Natural language processing enables computers to understand and generate human language.",
            "Computer vision allows machines to interpret and understand visual information from the world.",
            "Reinforcement learning is a type of machine learning where agents learn to make decisions.",
            "Data science combines domain expertise, programming skills, and knowledge of mathematics.",
            "Big data refers to data sets that are too large or complex to be dealt with by traditional data-processing software.",
            "Cloud computing provides on-demand availability of computer system resources without direct active management.",
            "Cybersecurity involves protecting computer systems and networks from information disclosure and theft.",
            "Blockchain is a distributed ledger technology that maintains a secure and decentralized record of transactions.",
            "Internet of Things (IoT) describes physical objects with sensors, software, and other technologies for connecting and exchanging data.",
            "Quantum computing harnesses quantum phenomena to process information in fundamentally new ways.",
            "Augmented reality overlays computer-generated images on a user's view of the real world.",
            # ... more documents
        ]

    def create_relevance_judgments(self):
        """Create exactly 10 relevance judgments"""
        return [
            {
                'query': 'What is machine learning?',
                'relevant_docs': [0, 3],  # Indices of relevant documents
                'non_relevant_docs': [10, 14]  # Some non-relevant examples
            },
            {
                'query': 'Python programming features',
                'relevant_docs': [1],
                'non_relevant_docs': [5, 12]
            },
            {
                'query': 'How do neural networks work?',
                'relevant_docs': [2],
                'non_relevant_docs': [8, 11]
            },
            {
                'query': 'Deep learning applications',
                'relevant_docs': [3, 5],
                'non_relevant_docs': [9, 13]
            },
            {
                'query': 'NLP techniques and tools',
                'relevant_docs': [4],
                'non_relevant_docs': [7, 10]
            },
            {
                'query': 'Computer vision algorithms',
                'relevant_docs': [5],
                'non_relevant_docs': [1, 14]
            },
            {
                'query': 'Reinforcement learning examples',
                'relevant_docs': [6],
                'non_relevant_docs': [3, 11]
            },
            {
                'query': 'Data science methodologies',
                'relevant_docs': [7],
                'non_relevant_docs': [2, 12]
            },
            {
                'query': 'Big data challenges',
                'relevant_docs': [8],
                'non_relevant_docs': [4, 13]
            },
            {
                'query': 'Cloud computing services',
                'relevant_docs': [9],
                'non_relevant_docs': [6, 14]
            }
        ]

    def create_ir_model(self):
        """Create the base IR model components"""

        class MinimalIRModel(dspy.Module):
            def __init__(self, documents):
                super().__init__()
                self.documents = documents

                # Query processing
                self.query_processor = dspy.ChainOfThought(
                    "query -> processed_query, key_terms, search_intent"
                )

                # Document relevance scoring
                self.doc_scorer = dspy.Predict(
                    "query, document, query_intent -> relevance_score, explanation"
                )

                # Ranking component
                self.ranker = dspy.Predict(
                    "query, scored_documents -> ranked_documents"
                )

            def forward(self, query: str, top_k: int = 5):
                # Process the query
                processed = self.query_processor(query=query)

                # Score all documents
                scored_docs = []
                for doc in self.documents:
                    score_result = self.doc_scorer(
                        query=processed.processed_query,
                        document=doc,
                        query_intent=processed.search_intent
                    )
                    scored_docs.append({
                        'document': doc,
                        'score': float(score_result.relevance_score),
                        'explanation': score_result.explanation
                    })

                # Sort by score
                scored_docs.sort(key=lambda x: x['score'], reverse=True)

                # Return top-k
                top_docs = scored_docs[:top_k]

                return dspy.Prediction(
                    ranked_documents=[d['document'] for d in top_docs],
                    scores=[d['score'] for d in top_docs],
                    explanations=[d['explanation'] for d in top_docs],
                    processed_query=processed.processed_query,
                    key_terms=processed.key_terms
                )

        return MinimalIRModel(self.documents)

    def train_with_10_judgments(self):
        """Train IR model with only 10 relevance judgments"""

        print("Training IR model with 10 relevance judgments")

        # Create base model
        model = self.create_ir_model()

        # Create training pairs from judgments
        training_pairs = []
        for judgment in self.relevance_judgments:
            query = judgment['query']

            # Positive examples
            for doc_idx in judgment['relevant_docs']:
                training_pairs.append({
                    'query': query,
                    'document': self.documents[doc_idx],
                    'label': 1
                })

            # Negative examples
            for doc_idx in judgment['non_relevant_docs']:
                training_pairs.append({
                    'query': query,
                    'document': self.documents[doc_idx],
                    'label': 0
                })

        print(f"Created {len(training_pairs)} training pairs from judgments")

        # Create training examples for DSPy
        dspy_examples = []
        for pair in training_pairs:
            example = dspy.Example(
                query=pair['query'],
                document=pair['document'],
                relevance_score=pair['label']
            )
            dspy_examples.append(example)

        # Optimize the model with minimal data
        print("\nOptimizing with BootstrapFewShot...")
        optimizer = dspy.BootstrapFewShot(
            metric=self._relevance_accuracy,
            max_bootstrapped_demos=3  # Very few due to limited data
        )

        # Compile the model
        self.trained_model = optimizer.compile(model, trainset=dspy_examples[:20])

        print("Model training completed!")
        return self.trained_model

    def _relevance_accuracy(self, example, prediction, trace=None):
        """Calculate relevance accuracy for binary relevance"""
        pred_score = float(prediction.relevance_score)
        pred_label = 1 if pred_score > 0.5 else 0
        return pred_label == example.relevance_score

    def evaluate_model(self, test_queries: List[str]):
        """Evaluate the trained IR model"""

        if not self.trained_model:
            print("Model not trained yet!")
            return

        print("\n=== Model Evaluation ===")

        for query in test_queries:
            print(f"\nQuery: {query}")
            results = self.trained_model(query=query, top_k=3)

            print("Top retrieved documents:")
            for i, (doc, score, explanation) in enumerate(
                zip(results.ranked_documents, results.scores, results.explanations)
            ):
                print(f"\n{i+1}. Score: {score:.3f}")
                print(f"   Document: {doc}")
                print(f"   Reason: {explanation}")

# Example usage
trainer = MinimalIRTrainer()

# Setup data
trainer.documents = trainer.create_sample_documents()
trainer.relevance_judgments = trainer.create_relevance_judgments()

# Train model
ir_model = trainer.train_with_10_judgments()

# Evaluate
test_queries = [
    "How does artificial intelligence work?",
    "What are the benefits of cloud computing?",
    "Machine learning vs deep learning"
]

trainer.evaluate_model(test_queries)
```

## Exercise 5: Building a Complete Minimal Data Pipeline

### Objective
Integrate all concepts learned to build a complete training pipeline for a real-world task with minimal data.

### Tasks

1. **Choose a Real-World Task**
   - Select a practical NLP task
   - Create exactly 10 labeled examples
   - Define clear evaluation criteria

2. **Design the Pipeline**
   - Configure appropriate augmentation strategies
   - Select optimization methods
   - Plan validation approach

3. **Implement and Evaluate**
   - Build the complete pipeline
   - Execute training
   - Analyze results and iterate

### Solution Template

```python
class CompleteMinimalDataPipeline:
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.original_examples = []
        self.pipeline_results = None

    def execute_complete_pipeline(self):
        """Execute the full minimal data training pipeline"""

        print(f"=== {self.task_name} Training Pipeline ===")

        # Step 1: Load and analyze data
        print("\n[1/5] Loading and analyzing 10 examples...")
        self._load_examples()
        data_analysis = self._analyze_examples()

        # Step 2: Configure pipeline
        print("\n[2/5] Configuring pipeline...")
        config = self._configure_pipeline(data_analysis)

        # Step 3: Create and execute pipeline
        print("\n[3/5] Executing training pipeline...")
        from minimal_data_pipelines import MinimalDataTrainingPipeline
        pipeline = MinimalDataTrainingPipeline(config)

        base_model = self._create_base_model()
        results = pipeline.execute_pipeline(
            base_program=base_model,
            examples=self.original_examples,
            evaluation_fn=self._evaluation_function
        )

        # Step 4: Analyze results
        print("\n[4/5] Analyzing results...")
        self._analyze_results(results)

        # Step 5: Generate report
        print("\n[5/5] Generating report...")
        report = self._generate_report(results)

        self.pipeline_results = results
        return report

    # Implement abstract methods for specific task
    def _load_examples(self):
        """Load the 10 examples for the task"""
        raise NotImplementedError

    def _analyze_examples(self):
        """Analyze the loaded examples"""
        raise NotImplementedError

    def _configure_pipeline(self, analysis):
        """Configure pipeline based on analysis"""
        raise NotImplementedError

    def _create_base_model(self):
        """Create the base model for optimization"""
        raise NotImplementedError

    def _evaluation_function(self, model):
        """Define evaluation function"""
        raise NotImplementedError

    def _analyze_results(self, results):
        """Analyze pipeline results"""
        raise NotImplementedError

    def _generate_report(self, results):
        """Generate final report"""
        raise NotImplementedError

# Example: Task-specific implementation
class AspectBasedSentimentPipeline(CompleteMinimalDataPipeline):
    """Pipeline for aspect-based sentiment analysis with 10 examples"""

    def _load_examples(self):
        """Load 10 aspect-based sentiment examples"""
        self.original_examples = [
            dspy.Example(
                text="The battery life is amazing but the screen is too dim.",
                aspects={
                    "battery_life": {"sentiment": "positive", "evidence": "amazing"},
                    "screen": {"sentiment": "negative", "evidence": "too dim"}
                }
            ),
            # ... 9 more examples
        ]

    def _configure_pipeline(self, analysis):
        """Configure for aspect-based sentiment task"""
        from minimal_data_pipelines import PipelineConfig, DataAugmentationType, OptimizationStrategy

        return PipelineConfig(
            num_examples=10,
            task_type="aspect_based_sentiment",
            domain="product_reviews",
            augmentation_strategies=[
                DataAugmentationType.PARAPHRASE,
                DataAugmentationType.TEMPLATE
            ],
            optimization_strategies=[
                OptimizationStrategy.PROMPT_OPTIMIZATION,
                OptimizationStrategy.HYBRID
            ],
            validation_method="cross_validation",
            confidence_threshold=0.75,
            continuous_learning=True
        )

    def _create_base_model(self):
        """Create base aspect-based sentiment model"""

        class AspectSentimentModel(dspy.Module):
            def __init__(self):
                super().__init__()
                self.aspect_extractor = dspy.Predict(
                    "text -> aspects, sentiments"
                )
                self.sentiment_analyzer = dspy.Predict(
                    "text, aspect -> sentiment, evidence"
                )

            def forward(self, text):
                # Extract aspects
                aspects = self.aspect_extractor(text=text)

                # Analyze each aspect
                results = {}
                for aspect in aspects.aspects:
                    analysis = self.sentiment_analyzer(
                        text=text,
                        aspect=aspect
                    )
                    results[aspect] = {
                        'sentiment': analysis.sentiment,
                        'evidence': analysis.evidence
                    }

                return dspy.Prediction(aspect_sentiments=results)

        return AspectSentimentModel()

    def _evaluation_function(self, model):
        """Evaluate aspect-based sentiment performance"""
        test_reviews = [
            "Great camera quality but poor battery performance.",
            "The phone is fast and the display is beautiful.",
            "Decent value for money though the build quality could be better."
        ]

        total_correct = 0
        total_aspects = 0

        for review in test_reviews:
            prediction = model(text=review)
            # Manual annotation for evaluation
            # In practice, you'd have labeled test data
            # Compare predictions and calculate accuracy

        return total_correct / max(total_aspects, 1)

# Run the complete pipeline
if __name__ == "__main__":
    pipeline = AspectBasedSentimentPipeline("Aspect-Based Sentiment Analysis")
    report = pipeline.execute_complete_pipeline()
    print("\nFinal Report:")
    print(report)
```

## Summary

These exercises cover the key concepts of:
1. Treating prompts as optimizable hyperparameters
2. Training models with only 10 examples
3. Building IR systems from minimal data
4. Creating comprehensive minimal data pipelines

By completing these exercises, you'll gain hands-on experience with advanced techniques for training sophisticated models with severely limited data.
# Chapter 5 Exercises: Optimizers & Compilation

## Overview

These exercises provide hands-on practice with DSPy's optimization capabilities. You'll work with different optimizers, understand their trade-offs, and learn to choose the right approach for various scenarios.

## Exercise 1: Basic BootstrapFewShot Optimization

### Objective
Learn to use BootstrapFewShot to improve a simple QA system.

### Problem
You have a basic question-answering system that needs improvement. Use BootstrapFewShot to optimize it with provided training data.

### Starter Code
```python
import dspy
from dspy.teleprompter import BootstrapFewShot

class BasicQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.Predict("question -> answer")

    def forward(self, question):
        return self.generate_answer(question=question)

# Training data
trainset = [
    dspy.Example(question="What is 2+2?", answer="4"),
    dspy.Example(question="What is the capital of France?", answer="Paris"),
    dspy.Example(question="Who wrote Romeo and Juliet?", answer="William Shakespeare"),
    dspy.Example(question="What is H2O?", answer="Water"),
    dspy.Example(question="How many continents are there?", answer="7"),
]

# Test data
testset = [
    dspy.Example(question="What is 3+3?", answer="6"),
    dspy.Example(question="What is the capital of Spain?", answer="Madrid"),
    dspy.Example(question="Who wrote Hamlet?", answer="William Shakespeare"),
]

# TODO: Implement this function
def bootstrap_optimize(program, trainset, max_demos=4):
    """Optimize the program using BootstrapFewShot."""
    pass
```

### Tasks
1. Define an exact match metric
2. Create a BootstrapFewShot optimizer
3. Compile the program with training data
4. Evaluate on test data
5. Compare with baseline performance

### Hints
- Use string comparison for exact matching
- Remember to configure `max_bootstrapped_demos`
- Create both baseline and compiled versions for comparison

### Expected Output
```
Baseline accuracy: 0.00%
Optimized accuracy: 33.33%
Improvement: +33.33%
```

---

## Exercise 2: KNNFewShot for Context-Aware Selection

### Objective
Implement KNNFewShot to select relevant examples dynamically based on query similarity.

### Problem
Build a context-aware classifier that selects different examples based on the input text's topic.

### Starter Code
```python
import dspy
from dspy.teleprompter import KNNFewShot
from sentence_transformers import SentenceTransformer

class TopicClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict("text, examples -> topic")

    def forward(self, text):
        return self.classify(text=text)

# Diverse training data
trainset = [
    dspy.Example(
        text="The company's stock price increased by 5% after earnings report",
        topic="finance"
    ),
    dspy.Example(
        text="New study reveals the effectiveness of mRNA vaccines",
        topic="healthcare"
    ),
    dspy.Example(
        text="The court ruled in favor of the plaintiff in the trademark case",
        topic="legal"
    ),
    dspy.Example(
        text="The quarterback threw a touchdown pass in the final minute",
        topic="sports"
    ),
    dspy.Example(
        text="The new iPhone features improved camera technology",
        topic="technology"
    ),
    # ... more examples for each category
]

# TODO: Implement these functions
def create_knn_optimizer(k=3, similarity_fn=None):
    """Create KNNFewShot optimizer with custom similarity."""
    pass

def semantic_similarity(text1, text2):
    """Calculate semantic similarity between texts."""
    pass

def evaluate_classifier(classifier, testset):
    """Evaluate classifier accuracy."""
    pass
```

### Tasks
1. Implement semantic similarity using sentence transformers
2. Create KNNFewShot optimizer with k=3
3. Compile the classifier
4. Test with domain-specific queries
5. Observe how different examples are selected

### Hints
- Use `sentence-transformers` for embeddings
- Cosine similarity works well for text similarity
- Print selected examples to understand the selection process

### Expected Output
```
Query: "The merger was approved by shareholders"
Selected topic: finance
Selected examples:
1. Text about stock prices (Topic: finance)
2. Text about company earnings (Topic: finance)
3. Text about market trends (Topic: finance)
```

---

## Exercise 3: MIPRO for Complex Reasoning Tasks

### Objective
Use MIPRO to optimize a Chain of Thought program for mathematical reasoning.

### Problem
Improve a mathematical problem solver that requires step-by-step reasoning.

### Starter Code
```python
import dspy
from dspy.teleprompter import MIPRO

class MathSolver(dspy.Module):
    def __init__(self):
        super().__init__()
        self.solve = dspy.ChainOfThought("problem -> steps, answer")

    def forward(self, problem):
        result = self.solve(problem=problem)
        return dspy.Prediction(
            steps=result.rationale,
            answer=result.answer
        )

# Math problems with solutions
trainset = [
    dspy.Example(
        problem="A rope is 12 meters long and cut into 3 equal pieces. How long is each piece?",
        steps="1. Divide 12 by 3\n2. 12 ÷ 3 = 4\n3. Each piece is 4 meters",
        answer="4 meters"
    ),
    dspy.Example(
        problem="If a train travels 60 km in 1 hour, how far in 3 hours?",
        steps="1. Calculate speed: 60 km/hour\n2. Multiply by time: 60 × 3\n3. Total distance: 180 km",
        answer="180 km"
    ),
    dspy.Example(
        problem="A box has 8 rows of 5 apples each. How many apples total?",
        steps="1. Multiply rows by apples per row\n2. 8 × 5 = 40\n3. Total apples: 40",
        answer="40 apples"
    ),
]

# TODO: Implement these functions
def create_math_metric():
    """Create a metric for evaluating math solutions."""
    pass

def extract_numbers(text):
    """Extract numerical values from text."""
    pass

def mipro_optimize(program, trainset, num_candidates=10):
    """Optimize the math solver using MIPRO."""
    pass
```

### Tasks
1. Create a comprehensive metric for math problems
2. Configure MIPRO with appropriate parameters
3. Optimize the math solver
4. Test with unseen problems
5. Analyze the improved reasoning

### Hints
- Check both the final answer and reasoning steps
- MIPRO benefits from more candidates for complex tasks
- Consider partial credit for correct steps

### Expected Output
```
Problem: "John saves $200 per month. How much in 6 months?"
Original reasoning: Basic calculation
Optimized reasoning:
1. Identify monthly savings: $200
2. Calculate total period: 6 months
3. Multiply: $200 × 6 = $1,200
4. Total savings: $1,200

Answer: $1,200
```

---

## Exercise 4: Optimizer Comparison

### Objective
Compare different optimizers on the same task to understand their trade-offs.

### Problem
Build a sentiment analyzer and optimize it with different approaches, then compare results.

### Starter Code
```python
import dspy
import time
from dspy.teleprompter import BootstrapFewShot, KNNFewShot, MIPRO

class SentimentAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyze = dspy.Predict("text -> sentiment, confidence")

    def forward(self, text):
        return self.analyze(text=text)

# Sentiment training data
trainset = [
    dspy.Example(text="I love this product!", sentiment="positive", confidence="high"),
    dspy.Example(text="This is terrible quality.", sentiment="negative", confidence="high"),
    dspy.Example(text="It works as expected.", sentiment="neutral", confidence="medium"),
    dspy.Example(text="Outstanding service and support.", sentiment="positive", confidence="high"),
    dspy.Example(text="Would not recommend to anyone.", sentiment="negative", confidence="high"),
    # ... more examples
]

# TODO: Implement these functions
def evaluate_sentiment(analyzer, testset):
    """Evaluate sentiment analyzer performance."""
    pass

def benchmark_optimizers(program, trainset, testset):
    """Compare multiple optimizers."""
    results = {}

    # Test baseline
    baseline_score = evaluate_sentiment(program, testset)
    results["Baseline"] = {"score": baseline_score, "time": 0}

    # Test BootstrapFewShot
    # Your code here

    # Test KNNFewShot
    # Your code here

    # Test MIPRO
    # Your code here

    return results

def compare_results(results):
    """Create a comparison report."""
    pass
```

### Tasks
1. Implement evaluation for sentiment analysis
2. Test all three optimizers
3. Measure compilation time for each
4. Create a comparison report
5. Analyze the trade-offs

### Hints
- Use exact match for sentiment
- Consider confidence scores in evaluation
- Track both accuracy and compilation time

### Expected Output
```
Optimizer Comparison Report:
================================
Baseline:
  Accuracy: 60.0%
  Compile Time: 0s

BootstrapFewShot:
  Accuracy: 75.0%
  Compile Time: 45s
  Improvement: +15.0%

KNNFewShot:
  Accuracy: 72.0%
  Compile Time: 30s
  Improvement: +12.0%

MIPRO:
  Accuracy: 80.0%
  Compile Time: 180s
  Improvement: +20.0%

Best Performance: MIPRO
Fastest Optimization: BootstrapFewShot
Best ROI: BootstrapFewShot
```

---

## Exercise 5: Custom Optimization Strategy

### Objective
Design and implement a custom optimization strategy for a specific use case.

### Problem
Create an optimization strategy for a multi-language chatbot that handles English, Spanish, and French.

### Starter Code
```python
import dspy
from dspy.teleprompter import BootstrapFewShot, KNNFewShot

class MultiLangChatbot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.translate = dspy.Predict("text, source_lang, target_lang -> translation")
        self.respond = dspy.Predict("query, context -> response")

    def forward(self, query, language="english"):
        # If not English, translate first
        if language != "english":
            english_query = self.translate(
                text=query,
                source_lang=language,
                target_lang="english"
            ).translation
        else:
            english_query = query

        # Generate response
        response = self.respond(query=english_query, context="customer_service")

        # If not English, translate back
        if language != "english":
            final_response = self.translate(
                text=response.response,
                source_lang="english",
                target_lang=language
            ).translation
        else:
            final_response = response.response

        return dspy.Prediction(response=final_response)

# Multi-language training data
trainset = {
    "english": [
        dspy.Example(query="Where is my order?", response="Your order will arrive tomorrow"),
        dspy.Example(query="How do I return an item?", response="You can return within 30 days"),
        # ... more examples
    ],
    "spanish": [
        dspy.Example(query="¿Dónde está mi pedido?", response="Tu pedido llegará mañana"),
        dspy.Example(query="¿Cómo devuelvo un artículo?", response="Puedes devolver en 30 días"),
        # ... more examples
    ],
    "french": [
        dspy.Example(query="Où est ma commande?", response="Votre commande arrivera demain"),
        dspy.Example(query="Comment retourner un article?", response="Vous pouvez retourner en 30 jours"),
        # ... more examples
    ]
}

# TODO: Implement these functions
def create_language_specific_optimizer(language):
    """Create optimizer specific to language."""
    pass

def optimize_multilingual_bot(chatbot, trainset):
    """Optimize bot for all languages."""
    optimized_bots = {}

    for language in trainset:
        # Your code here
        pass

    return optimized_bots

def evaluate_multilingual(bots, testset):
    """Evaluate performance across languages."""
    pass
```

### Tasks
1. Design optimization strategy for multi-language support
2. Implement language-specific optimization
3. Optimize for each language separately
4. Create a unified evaluation metric
5. Test performance across languages

### Hints
- Consider different k values for different languages
- Some languages might need more examples than others
- Evaluate language-specific performance

### Expected Output
```
Multi-Language Optimization Results:
====================================
English:
  Examples used: 50
  Accuracy: 85.0%

Spanish:
  Examples used: 40
  Accuracy: 82.0%

French:
  Examples used: 35
  Accuracy: 78.0%

Overall Performance: 81.7%
```

---

## Exercise 6: Optimization Debugging

### Objective
Identify and fix issues in optimization that lead to poor performance.

### Problem
A classifier is performing poorly after optimization. Debug and fix the issues.

### Starter Code
```python
import dspy
from dspy.teleprompter import BootstrapFewShot

class ProblematicClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict("text -> category")

    def forward(self, text):
        # BUG: Returns wrong attribute
        prediction = self.classify(text=text)
        return dspy.Prediction(label=prediction.category)

# Poor quality training data
trainset = [
    dspy.Example(text="Good", category="positive"),
    dspy.Example(text="Bad", category="negative"),
    dspy.Example(text="Not good", category="negative"),
    # Too few examples
]

# TODO: Implement these functions
def debug_classifier(classifier, trainset, testset):
    """Debug the classifier issues."""
    bugs_found = []

    # Check data quality
    # Your code here

    # Check attribute mismatch
    # Your code here

    # Check metric issues
    # Your code here

    return bugs_found

def fix_classifier(classifier, trainset):
    """Apply fixes to the classifier."""
    # Fix attribute issue
    # Your code here

    # Improve training data
    # Your code here

    return classifier

def create_better_metric():
    """Create a more robust evaluation metric."""
    pass
```

### Tasks
1. Identify the bug in the classifier
2. Spot issues with training data
3. Fix the problems
4. Re-optimize with corrections
5. Verify improved performance

### Hints
- Check attribute names carefully
- More diverse training data helps
- Consider case sensitivity in text

### Expected Output
```
Debug Results:
===============
Bugs Found:
1. Attribute mismatch: 'category' vs 'label'
2. Insufficient training data (only 3 examples)
3. No text normalization
4. No case-insensitive matching

Fixes Applied:
1. Fixed attribute mapping
2. Expanded training data to 20 examples
3. Added text preprocessing
4. Implemented case-insensitive metric

Performance Before: 20.0%
Performance After: 85.0%
```

---

## Exercise 7: Real-World Optimization Scenario

### Objective
Apply optimization techniques to a realistic scenario.

### Problem
Optimize a customer support ticket classifier that categorizes and prioritizes support requests.

### Starter Code
```python
import dspy
from dspy.teleprompter import KNNFewShot, MIPRO

class SupportTicketAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.categorize = dspy.Predict("ticket_text -> category, priority")
        self.extract_details = dspy.Predict("ticket_text -> product, issue_type")

    def forward(self, ticket_text):
        # Categorize and prioritize
        cat_result = self.categorize(ticket_text=ticket_text)

        # Extract details
        det_result = self.extract_details(ticket_text=ticket_text)

        return dspy.Prediction(
            category=cat_result.category,
            priority=cat_result.priority,
            product=det_result.product,
            issue_type=det_result.issue_type
        )

# Support ticket training data
trainset = [
    dspy.Example(
        ticket_text="I can't log in to my account. It says invalid password.",
        category="authentication",
        priority="high",
        product="mobile_app",
        issue_type="login_issue"
    ),
    dspy.Example(
        ticket_text="The application crashes when I try to upload photos.",
        category="bug",
        priority="high",
        product="mobile_app",
        issue_type="crash"
    ),
    dspy.Example(
        ticket_text="How do I change my notification settings?",
        category="how_to",
        priority="low",
        product="mobile_app",
        issue_type="settings"
    ),
    # ... more examples
]

# TODO: Implement these functions
def support_ticket_metric(example, pred, trace=None):
    """Multi-faceted metric for support tickets."""
    scores = {}

    # Category accuracy
    # Your code here

    # Priority accuracy
    # Your code here

    # Product accuracy
    # Your code here

    # Issue type accuracy
    # Your code here

    # Return weighted average
    return sum(scores.values()) / len(scores)

def optimize_support_system(analyzer, trainset):
    """Choose and apply the best optimizer."""
    # Analyze data characteristics
    # Your code here

    # Select appropriate optimizer
    # Your code here

    # Optimize the system
    # Your code here

    return optimized_analyzer

def analyze_performance(optimized_system, testset):
    """Analyze performance across different aspects."""
    pass
```

### Tasks
1. Create a comprehensive metric for multi-output prediction
2. Select the best optimizer based on data analysis
3. Optimize the support system
4. Evaluate performance by category and priority
5. Generate insights about optimization choices

### Hints
- Weight different outputs by importance
- Priority accuracy is often most critical
- Consider using different optimizers for different components

### Expected Output
```
Support System Optimization Report:
==================================
Data Analysis:
  Total examples: 100
  Categories: 5
  Priority levels: 3
  Products: 4

Selected Optimizer: KNNFewShot (k=5)
Reasoning: Context-sensitive categorization benefits from similarity

Performance by Category:
- Authentication: 92% accuracy
- Bug Reports: 88% accuracy
- How To: 95% accuracy
- Billing: 90% accuracy
- Feature Request: 85% accuracy

Performance by Priority:
- High: 95% accuracy (critical for SLA)
- Medium: 87% accuracy
- Low: 82% accuracy

Overall Score: 89.5%
```

---

## Exercise 7: Reflective Prompt Evolution for Complex Reasoning

### Objective
Learn to use Reflective Prompt Evolution (RPE) for optimizing complex multi-step reasoning tasks.

### Problem
You have a multi-hop reasoning task that requires understanding relationships between multiple pieces of information. Use RPE to evolve better reasoning prompts.

### Starter Code
```python
import dspy
from dspy.teleprompter import ReflectivePromptEvolution

class MultiHopReasoner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.hop1 = dspy.ChainOfThought("question -> first_answer")
        self.hop2 = dspy.ChainOfThought("question, first_answer -> second_answer")
        self.hop3 = dspy.ChainOfThought("question, first_answer, second_answer -> final_answer")

    def forward(self, question):
        result1 = self.hop1(question=question)
        result2 = self.hop2(question=question, first_answer=result1.first_answer)
        result3 = self.hop3(
            question=question,
            first_answer=result1.first_answer,
            second_answer=result2.second_answer
        )
        return dspy.Prediction(
            answer=result3.final_answer,
            reasoning_chain=[
                result1.reasoning,
                result2.reasoning,
                result3.reasoning
            ]
        )

# Multi-hop reasoning dataset
trainset = [
    dspy.Example(
        question="Who was the US President when the author of 'To Kill a Mockingbird' was born?",
        answer="Herbert Hoover",
        hops=[
            "Author of 'To Kill a Mockingbird' is Harper Lee",
            "Harper Lee was born in 1926",
            "Herbert Hoover was president in 1926"
        ]
    ),
    # Add more multi-hop examples...
]

# TODO: Implement these functions
def rpe_optimize(program, trainset, valset):
    """Optimize using Reflective Prompt Evolution."""
    pass

def analyze_evolution_progress(optimizer, program, trainset, valset):
    """Analyze how RPE evolves the prompts over generations."""
    pass

def custom_mutation_operator(program, domain_knowledge):
    """Apply domain-specific mutations."""
    pass
```

### Tasks

1. **Basic RPE Setup** (15 minutes)
   - Initialize RPE with appropriate parameters
   - Run basic optimization
   - Compare with baseline performance

2. **Custom Reflection** (20 minutes)
   - Implement custom reflection prompt templates
   - Add domain-specific reflection questions
   - Improve reflection quality

3. **Mutation Strategies** (25 minutes)
   - Implement custom mutation operators
   - Add domain-specific mutations
   - Balance exploration vs exploitation

4. **Diversity Analysis** (20 minutes)
   - Track population diversity
   - Implement diversity maintenance
   - Analyze convergence patterns

5. **Comparative Analysis** (20 minutes)
   - Compare RPE with MIPRO on the same task
   - Analyze trade-offs
   - Document findings

### Expected Output
```
RPE Optimization Report:
=======================
Configuration:
  Population size: 12
  Generations: 6
  Mutation rate: 0.3
  Selection pressure: 0.5

Evolution Progress:
  Gen 0: Best accuracy = 45.2%, Diversity = 0.85
  Gen 1: Best accuracy = 52.1%, Diversity = 0.78
  Gen 2: Best accuracy = 61.3%, Diversity = 0.71
  Gen 3: Best accuracy = 68.9%, Diversity = 0.65
  Gen 4: Best accuracy = 73.4%, Diversity = 0.58
  Gen 5: Best accuracy = 76.2%, Diversity = 0.52

Evolved Prompt Features:
  - Added explicit multi-step instructions
  - Improved error checking mechanisms
  - Better context preservation between hops
  - Enhanced reasoning verification

Comparison with MIPRO:
  RPE: 76.2% accuracy (45 min optimization)
  MIPRO: 72.8% accuracy (30 min optimization)
  BootstrapFewShot: 58.3% accuracy (5 min optimization)

RPE Strengths:
  + Discovered novel reasoning patterns
  + Better handling of edge cases
  + More diverse solution approaches
  + Continuous improvement over generations
```

---

## Exercise 8: Joint Optimization with COPA

### Objective
Apply combined fine-tuning and prompt optimization to achieve maximum performance on a mathematical reasoning task, demonstrating 3-8x improvements over baseline.

### Background
Research on joint optimization shows that combining fine-tuning with prompt optimization achieves synergistic effects that exceed additive improvements. This exercise demonstrates this approach on mathematical reasoning.

### Problem
Optimize a mathematical reasoning system using the COPA approach (fine-tuning + prompt optimization).

### Starter Code
```python
import dspy
from dspy.teleprompter import BootstrapFewShot, MIPRO

class MathReasoner(dspy.Module):
    """Mathematical reasoning with Chain of Thought."""

    def __init__(self):
        super().__init__()
        self.reason = dspy.ChainOfThought(
            "problem -> steps, intermediate_results, final_answer"
        )

    def forward(self, problem):
        result = self.reason(problem=problem)
        return dspy.Prediction(
            steps=result.rationale,
            answer=result.final_answer
        )

# GSM8K-style math problems
trainset = [
    dspy.Example(
        problem="Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends with 4 every day. She sells the remainder at the farmers' market daily for $2 per egg. How much does she make daily?",
        answer="$18"
    ),
    dspy.Example(
        problem="A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        answer="3"
    ),
    dspy.Example(
        problem="Josh decides to try flipping a house. He buys a house for $80,000 and puts $50,000 in repairs. This increased the value to 150% of the initial purchase price. How much profit did he make?",
        answer="$-10,000"
    ),
    # ... more examples (aim for 50-100 total)
]

# Test set
testset = [
    dspy.Example(
        problem="A farmer has 100 chickens. 20% are roosters. Half of the hens lay an egg a day. How many eggs per day?",
        answer="40"
    ),
    # ... more test examples
]

# TODO: Implement these functions

def math_accuracy_metric(example, pred, trace=None):
    """
    Evaluate mathematical answer accuracy.
    Handle numeric comparisons and text variations.
    """
    # Extract numerical answer
    # Compare with tolerance for floating point
    # Return 1.0 for correct, 0.0 for incorrect
    pass

def finetune_math_model(base_model_name, training_data, epochs=3):
    """
    Fine-tune a small model for mathematical reasoning.
    Focus on instruction-following for math problems.
    """
    # Load model with QLoRA
    # Prepare math-specific training format
    # Fine-tune with appropriate hyperparameters
    # Return fine-tuned model
    pass

def joint_optimization_pipeline(program, trainset, valset, base_model):
    """
    Execute COPA-style joint optimization:
    1. Fine-tune the base model
    2. Apply prompt optimization to fine-tuned model

    Returns: optimized_program, finetuned_model, results_dict
    """
    results = {}

    # Step 1: Baseline evaluation
    print("Evaluating baseline...")
    # Your code here

    # Step 2: Fine-tune the model
    print("Fine-tuning model...")
    # Your code here

    # Step 3: Evaluate fine-tuned only
    print("Evaluating fine-tuned model...")
    # Your code here

    # Step 4: Apply prompt optimization to base model
    print("Prompt optimization (base model)...")
    # Your code here

    # Step 5: Apply prompt optimization to fine-tuned model (COPA)
    print("COPA optimization (fine-tuned + prompts)...")
    # Your code here

    # Step 6: Calculate synergy
    print("Calculating synergy...")
    # Your code here

    return optimized_program, finetuned_model, results

def calculate_synergy(results):
    """
    Calculate synergistic improvement from joint optimization.

    Synergy = Combined - (Baseline + FT_Improvement + PO_Improvement)

    Returns synergy value and interpretation.
    """
    baseline = results["baseline"]
    ft_only = results["fine_tuning_only"]
    po_only = results["prompt_opt_only"]
    combined = results["copa"]

    # Expected additive improvement
    ft_improvement = ft_only - baseline
    po_improvement = po_only - baseline
    additive_expected = baseline + ft_improvement + po_improvement

    # Actual synergy
    synergy = combined - additive_expected
    improvement_factor = combined / baseline if baseline > 0 else float('inf')

    return {
        "synergy_absolute": synergy,
        "improvement_factor": improvement_factor,
        "additive_expected": additive_expected,
        "actual_combined": combined
    }
```

### Tasks

1. **Implement the Evaluation Metric** (15 minutes)
   - Handle numeric extraction from text
   - Compare with tolerance for floating point
   - Account for different answer formats ($18 vs 18 dollars)

2. **Fine-Tune the Model** (30 minutes)
   - Use QLoRA for memory-efficient fine-tuning
   - Format training data for math instruction following
   - Train for 3 epochs with appropriate learning rate

3. **Execute Joint Optimization** (20 minutes)
   - Follow correct order: fine-tune FIRST, then prompt optimize
   - Use MIPRO for prompt optimization
   - Track results at each stage

4. **Analyze Synergy** (15 minutes)
   - Calculate improvement beyond additive expectations
   - Understand why synergy occurs
   - Document findings

5. **Benchmark Comparison** (20 minutes)
   - Compare all approaches: baseline, FT-only, PO-only, COPA
   - Calculate improvement factors
   - Identify optimal strategy for your compute budget

### Hints
- Extract numbers using regex: `import re; numbers = re.findall(r'-?\d+\.?\d*', text)`
- Order matters: fine-tune first consistently outperforms prompt-first
- Synergy is highest for complex reasoning tasks
- Fine-tuned models need fewer demonstrations (3-shot vs 8-shot)
- Data requirement: aim for 50-100 training examples

### Expected Output
```
COPA Joint Optimization Report
==============================

Data Analysis:
  Training examples: 75
  Validation examples: 25
  Test examples: 50

Optimization Results:
---------------------
Baseline (no optimization):     11.2%
Fine-tuning only:               32.4%  (+21.2%)
Prompt optimization only:       22.8%  (+11.6%)
COPA (combined):                54.6%  (+43.4%)

Synergy Analysis:
----------------
Expected additive:              44.0%
Actual combined:                54.6%
Synergistic gain:               10.6%
Improvement factor:             4.9x

Key Findings:
1. Order matters: FT->PO achieved 54.6%, PO->FT only 38.2%
2. Synergy effect: 10.6% beyond additive expectations
3. Demo efficiency: Fine-tuned model achieves 8-shot baseline with just 3 demos
4. Instruction complexity: Fine-tuned model follows complex multi-step instructions

Recommendations:
- Use joint optimization for complex reasoning tasks
- Minimum 50 training examples for effective fine-tuning
- Always fine-tune first, then apply prompt optimization
- Consider compute budget: COPA requires ~2 GPU hours + API calls
```

### Challenge Extension
For advanced practice:
1. Implement Monte Carlo exploration of prompt configurations
2. Add Bayesian optimization for hyperparameter selection
3. Compare COPA with RPE on the same task
4. Measure demonstration efficiency improvements

---

## Exercise Solutions

After completing these exercises, you'll have:

1. **Hands-on experience** with all major DSPy optimizers
2. **Understanding of trade-offs** between different approaches
3. **Debugging skills** for optimization issues
4. **Real-world application** knowledge
5. **Decision-making framework** for choosing optimizers

Remember to:
- Start simple and iterate
- Monitor both accuracy and computation cost
- Validate on held-out data
- Consider your specific use case requirements
- Document your optimization choices

Good luck with your optimization journey!
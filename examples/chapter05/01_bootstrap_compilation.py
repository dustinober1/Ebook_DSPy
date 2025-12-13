"""
BootstrapFewShot Compilation Examples

This file demonstrates how to use BootstrapFewShot to automatically generate
few-shot examples and optimize DSPy programs.

Examples include:
- Basic BootstrapFewShot usage
- QA optimization with custom metrics
- Classification task optimization
- Performance comparison
- Advanced configuration
"""

import dspy
from dspy.teleprompter import BootstrapFewShot
from typing import List, Dict, Any, Union
import time

# Configure language model (placeholder - would need actual LM in practice)
dspy.settings.configure(lm=dspy.LM(model="gpt-3.5-turbo", api_key="your-key"))

# Example 1: Basic QA Optimization
def basic_qa_optimization():
    """Demonstrate basic BootstrapFewShot for QA tasks."""
    print("=" * 60)
    print("Example 1: Basic QA Optimization")
    print("=" * 60)

    # Define simple QA program
    class SimpleQA(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate = dspy.Predict("question -> answer")

        def forward(self, question):
            return self.generate(question=question)

    # Training data
    trainset = [
        dspy.Example(question="What is 2+2?", answer="4"),
        dspy.Example(question="What is the capital of France?", answer="Paris"),
        dspy.Example(question="Who wrote Romeo and Juliet?", answer="William Shakespeare"),
        dspy.Example(question="What is the largest planet?", answer="Jupiter"),
        dspy.Example(question="How many continents are there?", answer="7"),
        dspy.Example(question="What year did WW2 end?", answer="1945"),
        dspy.Example(question="What is H2O?", answer="Water"),
        dspy.Example(question="Who painted the Mona Lisa?", answer="Leonardo da Vinci"),
    ]

    # Test data
    testset = [
        dspy.Example(question="What is 3+3?", answer="6"),
        dspy.Example(question="What is the capital of Spain?", answer="Madrid"),
        dspy.Example(question="Who wrote Hamlet?", answer="William Shakespeare"),
    ]

    # Define evaluation metric
    def exact_match(example, pred, trace=None):
        return str(example.answer).lower() == str(pred.answer).lower()

    # Create baseline program
    baseline_qa = SimpleQA()

    # Create optimizer
    optimizer = BootstrapFewShot(
        metric=exact_match,
        max_bootstrapped_demos=4,
        max_labeled_demos=4
    )

    # Compile program
    print("\nCompiling with BootstrapFewShot...")
    start_time = time.time()
    compiled_qa = optimizer.compile(baseline_qa, trainset=trainset)
    compile_time = time.time() - start_time

    # Evaluate performance
    print(f"Compilation time: {compile_time:.2f} seconds")

    # Test examples
    print("\nTesting examples:")
    for example in testset:
        # Baseline prediction
        baseline_pred = baseline_qa(question=example.question)
        baseline_correct = exact_match(example, baseline_pred)

        # Compiled prediction
        compiled_pred = compiled_qa(question=example.question)
        compiled_correct = exact_match(example, compiled_pred)

        print(f"\nQuestion: {example.question}")
        print(f"Expected: {example.answer}")
        print(f"Baseline: {baseline_pred.answer} ({'✓' if baseline_correct else '✗'})")
        print(f"Optimized: {compiled_pred.answer} ({'✓' if compiled_correct else '✗'})")

    # Calculate overall accuracy
    baseline_correct = sum(
        exact_match(ex, baseline_qa(question=ex.question))
        for ex in testset
    )
    compiled_correct = sum(
        exact_match(ex, compiled_qa(question=ex.question))
        for ex in testset
    )

    baseline_accuracy = baseline_correct / len(testset)
    compiled_accuracy = compiled_correct / len(testset)

    print(f"\nBaseline accuracy: {baseline_accuracy:.0%}")
    print(f"Optimized accuracy: {compiled_accuracy:.0%}")
    print(f"Improvement: {(compiled_accuracy - baseline_accuracy):.0%}")

# Example 2: Classification with BootstrapFewShot
def classification_optimization():
    """Demonstrate BootstrapFewShot for text classification."""
    print("\n" + "=" * 60)
    print("Example 2: Text Classification Optimization")
    print("=" * 60)

    class TextClassifier(dspy.Module):
        def __init__(self, categories):
            super().__init__()
            self.classify = dspy.Predict(
                f"text, categories[{','.join(categories)}] -> classification"
            )

        def forward(self, text):
            return self.classify(text=text)

    # Categories and training data
    categories = ["positive", "negative", "neutral"]
    trainset = [
        dspy.Example(
            text="I absolutely love this product!",
            category="positive"
        ),
        dspy.Example(
            text="This is the worst experience I've ever had.",
            category="negative"
        ),
        dspy.Example(
            text="The product works as expected.",
            category="neutral"
        ),
        dspy.Example(
            text="Outstanding quality and service!",
            category="positive"
        ),
        dspy.Example(
            text="Would not recommend to anyone.",
            category="negative"
        ),
        dspy.Example(
            text="It's okay, nothing special.",
            category="neutral"
        ),
        dspy.Example(
            text="Exceeded all my expectations!",
            category="positive"
        ),
        dspy.Example(
            text="Complete waste of money.",
            category="negative"
        ),
    ]

    # Test data
    testset = [
        dspy.Example(
            text="Great value for the price!",
            category="positive"
        ),
        dspy.Example(
            text="Terrible customer service.",
            category="negative"
        ),
        dspy.Example(
            text="Average quality overall.",
            category="neutral"
        ),
    ]

    # Define classification metric
    def classification_metric(example, pred, trace=None):
        return str(example.category).lower() == str(pred.classification).lower()

    # Create classifier
    classifier = TextClassifier(categories)

    # Create optimizer with more configuration
    optimizer = BootstrapFewShot(
        metric=classification_metric,
        max_bootstrapped_demos=6,
        max_labeled_demos=4,
        max_rounds=3  # More bootstrap rounds
    )

    # Compile
    print("\nCompiling classifier...")
    compiled_classifier = optimizer.compile(classifier, trainset=trainset)

    # Test
    print("\nClassification Results:")
    for example in testset:
        result = compiled_classifier(text=example.text)
        correct = classification_metric(example, result)

        print(f"\nText: {example.text}")
        print(f"Expected: {example.category}")
        print(f"Predicted: {result.classification} ({'✓' if correct else '✗'})")

# Example 3: Chain of Thought with BootstrapFewShot
def cot_optimization():
    """Demonstrate BootstrapFewShot with Chain of Thought."""
    print("\n" + "=" * 60)
    print("Example 3: Chain of Thought Optimization")
    print("=" * 60)

    class MathSolver(dspy.Module):
        def __init__(self):
            super().__init__()
            self.solve = dspy.ChainOfThought("problem -> answer")

        def forward(self, problem):
            result = self.solve(problem=problem)
            return dspy.Prediction(
                answer=result.answer,
                reasoning=result.rationale
            )

    # Math word problems
    trainset = [
        dspy.Example(
            problem="A baker has 24 cupcakes. If she puts them in boxes of 6, how many boxes?",
            answer="4"
        ),
        dspy.Example(
            problem="Tom reads 15 pages per day. How many pages in 2 weeks?",
            answer="210"
        ),
        dspy.Example(
            problem="A car travels 60 mph. How far in 3.5 hours?",
            answer="210"
        ),
        dspy.Example(
            problem="Sarah bought 3 items at $12 each. What's the total?",
            answer="36"
        ),
    ]

    # Advanced metric that checks reasoning
    def math_metric(example, pred, trace=None):
        # Check answer correctness
        answer_correct = str(example.answer).lower() in str(pred.answer).lower()

        # Check if reasoning is present (should have Chain of Thought)
        has_reasoning = hasattr(pred, 'reasoning') and len(str(pred.reasoning)) > 20

        # Score: 70% for correct answer, 30% for reasoning
        return 0.7 * answer_correct + 0.3 * has_reasoning

    # Create solver
    solver = MathSolver()

    # Optimize with BootstrapFewShot
    optimizer = BootstrapFewShot(
        metric=math_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=2
    )

    # Compile
    print("\nCompiling math solver...")
    compiled_solver = optimizer.compile(solver, trainset=trainset)

    # Test with new problems
    test_problems = [
        "If 8 students share 64 cookies equally, how many per student?",
        "A monthly subscription costs $15. How much per year?",
        "A rectangle has area 45 and length 9. What's its width?"
    ]

    print("\nSolving Math Problems:")
    for problem in test_problems:
        result = compiled_solver(problem=problem)

        print(f"\nProblem: {problem}")
        print(f"Answer: {result.answer}")
        print(f"Reasoning: {result.reasoning}")

# Example 4: Performance Comparison
def performance_comparison():
    """Compare different BootstrapFewShot configurations."""
    print("\n" + "=" * 60)
    print("Example 4: Performance Comparison")
    print("=" * 60)

    class TestProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.process = dspy.Predict("input -> output")

        def forward(self, text):
            return self.process(input=text)

    # Generate diverse training data
    trainset = [
        dspy.Example(input="Hello", output="Hi there!"),
        dspy.Example(input="Goodbye", output="See you later!"),
        dspy.Example(input="Thank you", output="You're welcome!"),
        dspy.Example(input="How are you?", output="I'm doing well, thanks!"),
        dspy.Example(input="What's your name?", output="I'm an AI assistant."),
        dspy.Example(input="Help me", output="I'd be happy to help!"),
        dspy.Example(input="I don't understand", output="Let me explain differently."),
        dspy.Example(input="Great!", output="Glad to hear that!"),
    ]

    # Test configurations
    configs = [
        {"max_bootstrapped_demos": 2, "max_labeled_demos": 2, "name": "Minimal"},
        {"max_bootstrapped_demos": 4, "max_labeled_demos": 4, "name": "Balanced"},
        {"max_bootstrapped_demos": 6, "max_labeled_demos": 4, "name": "More Examples"},
        {"max_bootstrapped_demos": 4, "max_labeled_demos": 6, "name": "More Labeled"},
    ]

    # Simple metric
    def similarity_metric(example, pred, trace=None):
        # Simple word overlap
        pred_words = set(str(pred.output).lower().split())
        true_words = set(str(example.output).lower().split())
        overlap = len(pred_words & true_words)
        return overlap / max(len(pred_words | true_words), 1)

    print("\nComparing Configurations:")
    print("-" * 40)

    for config in configs:
        # Create optimizer
        optimizer = BootstrapFewShot(
            metric=similarity_metric,
            max_bootstrapped_demos=config["max_bootstrapped_demos"],
            max_labeled_demos=config["max_labeled_demos"]
        )

        # Compile and time
        program = TestProgram()
        start_time = time.time()
        compiled = optimizer.compile(program, trainset=trainset)
        compile_time = time.time() - start_time

        # Quick test
        test_input = "Hello there"
        result = compiled(text=test_input)
        score = similarity_metric(
            dspy.Example(input=test_input, output="Hi there!"),
            result
        )

        print(f"{config['name']}:")
        print(f"  Compile Time: {compile_time:.2f}s")
        print(f"  Test Score: {score:.2f}")
        print(f"  Config: {config['max_bootstrapped_demos']} bootstrapped, "
              f"{config['max_labeled_demos']} labeled")
        print()

# Example 5: Advanced BootstrapFewShot with Custom Teacher
def advanced_bootstrap():
    """Demonstrate advanced BootstrapFewShot features."""
    print("\n" + "=" * 60)
    print("Example 5: Advanced BootstrapFewShot")
    print("=" * 60)

    class AdvancedQA(dspy.Module):
        def __init__(self):
            super().__init__()
            self.think = dspy.ChainOfThought("question -> analysis")
            self.answer = dspy.Predict("question, analysis -> answer")

        def forward(self, question):
            # First analyze
            analysis = self.think(question=question)
            # Then answer with analysis
            result = self.answer(question=question, analysis=analysis.rationale)
            return result

    # Complex QA data
    trainset = [
        dspy.Example(
            question="Why is the sky blue?",
            answer="The sky appears blue due to Rayleigh scattering of sunlight by air molecules."
        ),
        dspy.Example(
            question="How do vaccines work?",
            answer="Vaccines stimulate the immune system to produce antibodies without causing the disease."
        ),
        dspy.Example(
            question="What causes seasons?",
            answer="Seasons are caused by Earth's axial tilt as it orbits the Sun."
        ),
        dspy.Example(
            question="Why do we dream?",
            answer="Dreaming may help process emotions, consolidate memories, and solve problems."
        ),
    ]

    # Multi-criteria metric
    def advanced_metric(example, pred, trace=None):
        score = 0

        # Answer quality (40%)
        if hasattr(pred, 'answer'):
            answer_len = len(str(pred.answer))
            if answer_len > 10:  # Not too short
                score += 0.4
            if answer_len < 200:  # Not too long
                score += 0.2

        # Analysis quality (40%)
        if hasattr(pred, 'analysis'):
            analysis_words = str(pred.analysis).split()
            if len(analysis_words) > 10:  # Has reasoning
                score += 0.3
            if any(word in analysis_words for word in ["because", "due", "causes"]):
                score += 0.1  # Has causal reasoning

        # Completeness (20%)
        if hasattr(pred, 'answer') and hasattr(pred, 'analysis'):
            score += 0.2

        return score

    # Advanced optimizer configuration
    optimizer = BootstrapFewShot(
        metric=advanced_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=3,
        max_rounds=4,  # More rounds for better convergence
        teacher_settings=dict(temperature=0.7),  # Teacher LM settings
    )

    # Compile
    print("\nCompiling with advanced configuration...")
    program = AdvancedQA()
    compiled = optimizer.compile(program, trainset=trainset)

    # Test with complex question
    test_questions = [
        "How does photosynthesis work?",
        "Why do people age?",
        "What makes something funny?"
    ]

    print("\nAdvanced QA Results:")
    for question in test_questions:
        result = compiled(question=question)
        print(f"\nQ: {question}")
        if hasattr(result, 'analysis'):
            print(f"Analysis: {result.analysis}")
        if hasattr(result, 'answer'):
            print(f"Answer: {result.answer}")

# Main execution
def run_all_examples():
    """Run all BootstrapFewShot examples."""
    print("DSPy BootstrapFewShot Examples")
    print("Demonstrating automatic few-shot example generation\n")

    try:
        basic_qa_optimization()
        classification_optimization()
        cot_optimization()
        performance_comparison()
        advanced_bootstrap()

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("All BootstrapFewShot examples completed!")
    print("=" * 60)

if __name__ == "__main__":
    run_all_examples()
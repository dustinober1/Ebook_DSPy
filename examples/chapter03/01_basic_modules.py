"""
Basic DSPy Modules Examples

This file demonstrates fundamental DSPy module concepts including:
- Using dspy.Predict for simple tasks
- Module configuration with examples
- Basic error handling
- Performance considerations
"""

import dspy
import time
from typing import List, Dict, Any

# Example 1: Simple Predict Module
def demonstrate_basic_predict():
    """Demonstrate basic dspy.Predict usage."""

    print("=" * 60)
    print("Example 1: Basic Predict Module")
    print("=" * 60)

    # Define a signature
    class TextClassifier(dspy.Signature):
        """Classify text into categories."""
        text = dspy.InputField(desc="Text to classify", type=str)
        category = dspy.OutputField(desc="Text category", type=str)
        confidence = dspy.OutputField(desc="Classification confidence", type=float)

    # Create the module
    classifier = dspy.Predict(TextClassifier)

    # Test with sample texts
    texts = [
        "I love this product! It works perfectly.",
        "This is terrible. Worst purchase ever.",
        "It's okay, nothing special."
        "Amazing quality and great value for money!"
        "Poor customer service and slow delivery."
    ]

    for i, text in enumerate(texts, 1):
        result = classifier(text=text)
        print(f"\nText {i}: {text}")
        print(f"Category: {result.category}")
        print(f"Confidence: {result.confidence:.2f}")

# Example 2: Module Configuration
def demonstrate_module_configuration():
    """Demonstrate different module configurations."""

    print("\n" + "=" * 60)
    print("Example 2: Module Configuration")
    print("=" * 60)

    # Modules with different temperatures
    creative_module = dspy.Predict(
        "prompt -> creative_response",
        temperature=0.8,
        max_tokens=150
    )

    precise_module = dspy.Predict(
        "question -> precise_answer",
        temperature=0.1,
        max_tokens=50
    )

    # Test with the same prompt
    prompt = "Describe a sunset in one sentence"

    print(f"\nPrompt: {prompt}")

    print("\nCreative Module (temperature=0.8):")
    creative_result = creative_module(prompt=prompt)
    print(f"Response: {creative_result.creative_response}")

    print("\nPrecise Module (temperature=0.1):")
    precise_result = precise_module(question=prompt)
    print(f"Answer: {precise_result.precise_answer}")

# Example 3: Few-Shot Examples
def demonstrate_few_shot_examples():
    """Demonstrate using few-shot examples with modules."""

    print("\n" + "=" * 60)
    print("Example 3: Few-Shot Examples")
    print("=" * 60)

    # Create examples for math problems
    math_examples = [
        dspy.Example(
            problem="What is 15 + 27?",
            answer="42"
        ),
        dspy.Example(
            problem="What is 8 × 7?",
            answer="56"
        ),
        dspy.Example(
            problem="What is 144 ÷ 12?",
            answer="12"
        )
    ]

    # Create solver with examples
    class MathProblem(dspy.Signature):
        """Solve math problems."""
        problem = dspy.InputField(desc="Math problem to solve", type=str)
        answer = dspy.OutputField(desc="Numerical answer", type=str)

    math_solver = dspy.Predict(MathProblem, demos=math_examples)

    # Test with new problems
    test_problems = [
        "What is 23 × 19?",
        "What is 99 + 101?",
        "What is 256 ÷ 16?"
    ]

    print("\nTesting Math Solver with Examples:")
    for i, problem in enumerate(test_problems, 1):
        result = math_solver(problem=problem)
        print(f"Problem {i}: {problem}")
        print(f"Answer: {result.answer}")
        # Note: Verify these manually or add verification logic

# Example 4: Multiple Outputs
def demonstrate_multiple_outputs():
    """Demonstrate modules with multiple output fields."""

    print("\n" + "=" * 60)
    print("Example 4: Multiple Outputs")
    print("=" * 60)

    class TextAnalyzer(dspy.Signature):
        """Analyze text for multiple attributes."""
        text = dspy.InputField(desc="Text to analyze", type=str)
        sentiment = dspy.OutputField(desc="Overall sentiment", type=str)
        word_count = dspy.OutputField(desc="Number of words", type=int)
        has_questions = dspy.OutputField(desc="Contains questions", type=bool)
        main_topics = dspy.OutputField(desc="Main topics discussed", type=str)

    analyzer = dspy.Predict(TextAnalyzer)

    texts = [
        "Is this product worth buying? I'm considering it.",
        "The implementation uses machine learning algorithms and neural networks.",
        "Great customer service! They resolved my issue immediately."
    ]

    for text in texts:
        result = analyzer(text=text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result.sentiment}")
        print(f"Words: {result.word_count}")
        print(f"Has Questions: {result.has_questions}")
        print(f"Topics: {result.main_topics}")

# Example 5: Batch Processing
def demonstrate_batch_processing():
    """Demonstrate batch processing of inputs."""

    print("\n" + "=" * 60)
    print("Example 5: Batch Processing")
    print("=" * 60)

    class BatchProcessor:
        """Simple batch processor for DSPy modules."""

        def __init__(self, module):
            self.module = module
            self.results = []

        def process_batch(self, texts: List[str]) -> List[dspy.Prediction]:
            """Process multiple texts through the module."""
            self.results = []

            for text in texts:
                try:
                    result = self.module(text=text)
                    self.results.append(result)
                except Exception as e:
                    print(f"Error processing text: {e}")
                    # Add empty result
                    self.results.append(dspy.Prediction(error=str(e)))

            return self.results

    # Create classifier
    class QuickClassifier(dspy.Signature):
        text = dspy.InputField(desc="Text to classify", type=str)
        label = dspy.OutputField(desc="Text label", type=str)

    classifier = dspy.Predict(QuickClassifier)
    batch_processor = BatchProcessor(classifier)

    # Batch of texts
    batch_texts = [
        "This is amazing!",
        "Not worth the money.",
        "Could be better.",
        "Absolutely love it!"
    ]

    print("\nProcessing batch of texts:")
    results = batch_processor.process_batch(batch_texts)

    for i, (text, result) in enumerate(zip(batch_texts, results)):
        print(f"{i+1}. '{text}' -> '{result.label}'")

# Example 6: Performance Testing
def demonstrate_performance():
    """Demonstrate performance testing of modules."""

    print("\n" + "=" * 60)
    print("Example 6: Performance Testing")
    print("=" * 60)

    # Create a simple module
    class SimpleModule(dspy.Predict):
        def __init__(self):
            signature = dspy.Signature("input -> output")
            super().__init__(signature, cache=False)

    module = SimpleModule()

    # Test different input sizes
    input_sizes = [10, 100, 500, 1000]

    print("\nPerformance with different input sizes:")

    for size in input_sizes:
        test_input = "x" * size

        # Measure time
        start_time = time.time()

        for _ in range(5):  # Average over 5 runs
            result = module(input=test_input)

        end_time = time.time()
        avg_time = (end_time - start_time) / 5 * 1000  # Convert to ms

        print(f"Input size: {size:4d} chars | Avg time: {avg_time:6.2f} ms")

        # Show a sample output
        if size == 10:
            print(f"Sample output: {result.output[:50]}...")

# Example 7: Error Handling
def demonstrate_error_handling():
    """Demonstrate error handling in modules."""

    print("\n" + "=" * 60)
    print("Example 7: Error Handling")
    print("=" * 60)

    # Module with validation
    class SafeModule(dspy.Module):
        def __init__(self):
            super().__init__()
            # Use a flexible signature
            self.signature = dspy.Signature("text -> processed_text")
            # Create internal predict with fallback
            self.predict = dspy.Predict(self.signature)

        def forward(self, text):
            """Safe processing with validation."""
            # Input validation
            if not text or not isinstance(text, str):
                return dspy.Prediction(
                    processed_text="[ERROR: Invalid input]",
                    status="error",
                    original_input=str(text)
                )

            try:
                # Process with internal module
                result = self.predict(text=text)
                return dspy.Prediction(
                    processed_text=result.processed_text,
                    status="success"
                )

            except Exception as e:
                return dspy.Prediction(
                    processed_text=f"[ERROR: {str(e)}]",
                    status="error"
                )

    safe_module = SafeModule()

    # Test with various inputs
    test_inputs = [
        "Normal text input",
        "",  # Empty string
        None,  # None value
        123,  # Number instead of string
        "x" * 10000  # Very long string
    ]

    print("\nTesting with different input types:")
    for test_input in test_inputs:
        result = safe_module(text=test_input)
        print(f"Input: {repr(test_input)[:30]:30}")
        print(f"Status: {result.status}")
        if result.status == "success":
            print(f"Output: {result.processed_text[:50]}...")
        print()

# Example 8: Module Comparison
def demonstrate_module_comparison():
    """Compare different module types on the same task."""

    print("\n" + "=" * 60)
    print("Example 8: Module Comparison")
    print("=" * 60)

    # Same signature for all modules
    signature = dspy.Signature("question -> answer")

    # Different modules
    predict_module = dspy.Predict(signature)
    cot_module = dspy.ChainOfThought(signature)

    # Question that might benefit from reasoning
    question = """
    A bakery sells cakes for $10 each and cookies for $2 each.
    If they sold 50 cakes and 200 cookies in one day,
    what was their total revenue?
    """

    print(f"\nQuestion: {question}")
    print("-" * 40)

    # Test Predict module
    print("\n1. Predict Module:")
    start_time = time.time()
    pred_result = predict_module(question=question)
    pred_time = time.time() - start_time
    print(f"Time: {pred_time:.3f}s")
    print(f"Answer: {pred_result.answer}")

    # Test ChainOfThought module
    print("\n2. ChainOfThought Module:")
    start_time = time.time()
    cot_result = cot_module(question=question)
    cot_time = time.time() - start_time
    print(f"Time: {cot_time:.3f}s")
    print(f"Reasoning: {cot_result.reasoning}")
    print(f"Answer: {cot_result.answer}")

    # Comparison
    print("\n3. Comparison:")
    print(f"Predict took {pred_time:.3f}s, CoT took {cot_time:.3f}s")
    print(f"Speed ratio: {cot_time/pred_time:.1f}x")

    # Verify answers are reasonable
    expected_answer = 500 + 400  # 50*10 + 200*2
    try:
        pred_num = float(pred_result.answer.replace("$", "").replace(",", ""))
        cot_num = float(cot_result.answer.replace("$", "").replace(",", ""))

        print(f"Expected: ${expected_answer}")
        print(f"Predict result: ${pred_num} {'✓' if abs(pred_num - expected_answer) < 1 else '✗'}")
        print(f"CoT result: ${cot_num} {'✓' if abs(cot_num - expected_answer) < 1 else '✗'}")
    except:
        print("Could not parse numerical answers")

# Run all examples
def run_all_examples():
    """Run all module examples."""

    print("DSPy Module Examples - Basic Usage")
    print("=" * 60)

    try:
        demonstrate_basic_predict()
        demonstrate_module_configuration()
        demonstrate_few_shot_examples()
        demonstrate_multiple_outputs()
        demonstrate_batch_processing()
        demonstrate_performance()
        demonstrate_error_handling()
        demonstrate_module_comparison()

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_examples()
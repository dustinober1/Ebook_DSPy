"""
MIPRO Optimization Examples

This file demonstrates MIPRO (Multi-step Instruction and demonstration PRompt Optimization),
DSPy's most advanced optimizer that simultaneously optimizes instructions and examples.

Examples include:
- Basic MIPRO usage
- Instruction evolution
- Multi-objective optimization
- Advanced configuration
- Performance tracking
"""

import dspy
from dspy.teleprompter import MIPRO
from typing import List, Dict, Any, Union
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure language model (placeholder)
dspy.settings.configure(lm=dspy.LM(model="gpt-3.5-turbo", api_key="your-key"))

# Example 1: Basic MIPRO Optimization
def basic_mipro_optimization():
    """Demonstrate basic MIPRO for QA tasks."""
    print("=" * 60)
    print("Example 1: Basic MIPRO Optimization")
    print("=" * 60)

    class AdvancedQA(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate = dspy.ChainOfThought("question -> answer")

        def forward(self, question):
            result = self.generate(question=question)
            return dspy.Prediction(
                answer=result.answer,
                reasoning=result.rationale
            )

    # Complex QA training data
    trainset = [
        dspy.Example(
            question="What happens when you mix vinegar and baking soda?",
            answer="They react to produce carbon dioxide gas, water, and sodium acetate. This creates bubbles and fizzing."
        ),
        dspy.Example(
            question="Why do leaves change color in autumn?",
            answer="Leaves change color due to decreasing chlorophyll, revealing pigments like carotenoids (yellow/orange) and anthocyanins (red/purple)."
        ),
        dspy.Example(
            question="How does GPS determine location?",
            answer="GPS uses trilateration by measuring signal travel time from multiple satellites (at least 4) to calculate precise coordinates."
        ),
        dspy.Example(
            question="What causes ocean tides?",
            answer="Tides are primarily caused by the gravitational pull of the Moon, with the Sun also contributing. This creates bulges in ocean water."
        ),
        dspy.Example(
            question="Why do we yawn?",
            answer="Yawning may help regulate brain temperature, increase alertness, or equalize pressure. It might also be a social signal."
        ),
    ]

    # Test data
    testset = [
        dspy.Example(
            question="How do microwave ovens heat food?",
            answer="Microwaves use electromagnetic waves to excite water molecules in food, generating heat through molecular friction."
        ),
        dspy.Example(
            question="What causes rainbows?",
            answer="Rainbows occur when sunlight is refracted, reflected, and dispersed through water droplets in the atmosphere, creating a spectrum of light."
        ),
    ]

    # Define comprehensive evaluation metric
    def comprehensive_metric(example, pred, trace=None):
        score = 0

        # Answer accuracy (50%)
        if hasattr(pred, 'answer'):
            pred_answer = str(pred.answer).lower()
            true_answer = str(example.answer).lower()

            # Check for key concepts
            key_concepts = extract_concepts(true_answer)
            pred_concepts = extract_concepts(pred_answer)

            concept_overlap = len(key_concepts & pred_concepts) / max(len(key_concepts), 1)
            score += 0.5 * concept_overlap

        # Reasoning quality (30%)
        if hasattr(pred, 'reasoning'):
            reasoning = str(pred.reasoning)
            # Check for scientific language
            scientific_words = ["because", "due to", "causes", "results", "process", "mechanism"]
            scientific_score = sum(1 for word in scientific_words if word in reasoning.lower())
            score += 0.3 * min(scientific_score / 2, 1)  # Normalize

        # Completeness (20%)
        if hasattr(pred, 'answer') and hasattr(pred, 'reasoning'):
            answer_length = len(str(pred.answer))
            reasoning_length = len(str(pred.reasoning))
            if answer_length > 20 and reasoning_length > 50:
                score += 0.2

        return score

    def extract_concepts(text):
        """Extract key concepts from text."""
        # Simple concept extraction
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter out common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        concepts = {word for word in words if len(word) > 3 and word not in stop_words}
        return concepts

    # Create baseline program
    baseline_qa = AdvancedQA()

    # Create MIPRO optimizer
    optimizer = MIPRO(
        metric=comprehensive_metric,
        num_candidates=12,  # Generate 12 instruction candidates
        init_temperature=1.0,  # Start with high creativity
        verbose=True  # Show optimization progress
    )

    # Compile with progress tracking
    print("\nStarting MIPRO optimization...")
    print("This may take a few minutes...\n")

    start_time = time.time()
    compiled_qa = optimizer.compile(
        baseline_qa,
        trainset=trainset,
        max_bootstrapped_demos=4
    )
    compile_time = time.time() - start_time

    print(f"\nOptimization completed in {compile_time:.1f} seconds")

    # Compare results
    print("\n" + "=" * 40)
    print("Performance Comparison")
    print("=" * 40)

    for example in testset:
        print(f"\nQuestion: {example.question}")
        print(f"Expected: {example.answer}")

        # Baseline
        baseline_result = baseline_qa(question=example.question)
        baseline_score = comprehensive_metric(example, baseline_result)
        print(f"\nBaseline Score: {baseline_score:.2f}")
        if hasattr(baseline_result, 'answer'):
            print(f"Baseline Answer: {baseline_result.answer}")

        # Optimized
        optimized_result = compiled_qa(question=example.question)
        optimized_score = comprehensive_metric(example, optimized_result)
        print(f"\nOptimized Score: {optimized_score:.2f}")
        if hasattr(optimized_result, 'answer'):
            print(f"Optimized Answer: {optimized_result.answer}")

        print("-" * 40)

# Example 2: Instruction Evolution
def instruction_evolution():
    """Demonstrate how MIPRO evolves instructions."""
    print("\n" + "=" * 60)
    print("Example 2: Instruction Evolution")
    print("=" * 60)

    class TextSummarizer(dspy.Module):
        def __init__(self):
            super().__init__()
            self.summarize = dspy.Predict("document -> summary")

        def forward(self, document):
            return self.summarize(document=document)

    # Training data for summarization
    trainset = [
        dspy.Example(
            document="The rapid advancement of artificial intelligence has transformed numerous industries. From healthcare to finance, AI applications are improving efficiency and accuracy. Machine learning algorithms can now detect diseases earlier, optimize investment portfolios, and even create art. However, concerns about job displacement and ethical implications remain significant challenges that society must address.",
            summary="AI is transforming industries like healthcare and finance with applications in disease detection and investment optimization, though job displacement and ethics are concerns."
        ),
        dspy.Example(
            document="Climate change represents one of the greatest challenges of our time. Rising global temperatures, melting ice caps, and extreme weather events are clear indicators of this crisis. Scientists warn that immediate action is needed to reduce carbon emissions and transition to renewable energy sources. Failure to act decisively could result in irreversible damage to ecosystems and human societies.",
            summary="Climate change, evidenced by rising temperatures and extreme weather, requires immediate action to reduce emissions and transition to renewables to prevent irreversible damage."
        ),
    ]

    # Evolution tracker
    class EvolutionTracker:
        def __init__(self):
            self.generations = []

        def track_generation(self, program, metrics, trace=None):
            """Track each generation of instructions."""
            generation = {
                "metrics": metrics,
                "timestamp": time.time()
            }

            # Extract current instruction if available
            if hasattr(program, 'summarize') and hasattr(program.summarize, 'instruction'):
                generation["instruction"] = str(program.summarize.instruction)
            elif trace and 'summarize' in trace:
                generation["instruction"] = str(trace['summarize'].instruction)

            self.generations.append(generation)

            print(f"Generation {len(self.generations)}:")
            print(f"  Score: {metrics.get('score', 'N/A')}")
            if "instruction" in generation:
                print(f"  Instruction: {generation['instruction'][:100]}...")
            print()

    # Create tracker
    tracker = EvolutionTracker()

    # Create MIPRO with callback
    optimizer = MIPRO(
        metric=lambda ex, pred, trace: rouge_score(ex.summary, pred.summary),
        num_candidates=8,
        init_temperature=1.2,  # Higher creativity
        verbose=True,
        callbacks=[tracker.track_generation]
    )

    def rouge_score(reference, candidate):
        """Simple ROUGE-like metric."""
        ref_words = set(str(reference).lower().split())
        cand_words = set(str(candidate).lower().split())
        overlap = len(ref_words & cand_words)
        precision = overlap / max(len(cand_words), 1)
        recall = overlap / max(len(ref_words), 1)
        if precision + recall == 0:
            return 0
        return 2 * precision * recall / (precision + recall)

    # Compile with evolution tracking
    print("Starting instruction evolution...\n")
    summarizer = TextSummarizer()
    compiled = optimizer.compile(summarizer, trainset=trainset, num_trials=3)

    # Show evolution
    print("\nInstruction Evolution Summary:")
    print("-" * 40)
    for i, gen in enumerate(tracker.generations):
        score = gen['metrics'].get('score', 'N/A')
        print(f"Gen {i+1}: Score = {score:.3f}")

# Example 3: Multi-Objective Optimization
def multi_objective_optimization():
    """Demonstrate multi-objective optimization with MIPRO."""
    print("\n" + "=" * 60)
    print("Example 3: Multi-Objective Optimization")
    print("=" * 60)

    class CreativeWriter(dspy.Module):
        def __init__(self):
            super().__init__()
            self.create = dspy.Predict("prompt -> creative_response")

        def forward(self, prompt):
            return self.create(prompt=prompt)

    # Creative writing training data
    trainset = [
        dspy.Example(
            prompt="Write about a mysterious door",
            creative_response="The ancient wooden door stood alone in the brick wall, seemingly leading nowhere. Its brass handle gleamed with an otherworldly light, beckoning the curious to discover what lay beyond."
        ),
        dspy.Example(
            prompt="Describe a futuristic city",
            creative_response="Towers of crystal and steel pierced the clouds, while flying vehicles weaved between buildings like metallic birds. Below, holographic advertisements painted the streets in dancing colors."
        ),
    ]

    # Multi-objective metric
    def creative_metric(example, pred, trace=None):
        metrics = {}

        # Creativity (40%)
        if hasattr(pred, 'creative_response'):
            response = str(pred.creative_response)
            # Check for creative vocabulary
            creative_words = ["mysterious", "ancient", "otherworldly", "futuristic", "crystal", "gleamed", "beckoning", "pierced", "weaved", "dancing"]
            creativity_score = sum(1 for word in creative_words if word in response.lower()) / 5  # Normalize
            metrics["creativity"] = min(creativity_score, 1)

        # Length (30%)
        if hasattr(pred, 'creative_response'):
            length = len(str(pred.creative_response))
            ideal_length = 100
            length_score = 1 - min(abs(length - ideal_length) / ideal_length, 1)
            metrics["length"] = length_score

        # Grammar/Readability (30%)
        if hasattr(pred, 'creative_response'):
            response = str(pred.creative_response)
            # Simple readability check
            sentences = response.count('.') + response.count('!') + response.count('?')
            if sentences > 0:
                avg_sentence_length = len(response.split()) / sentences
                readability_score = 1 - min(abs(avg_sentence_length - 15) / 15, 1)
                metrics["readability"] = readability_score
            else:
                metrics["readability"] = 0.5

        # Weighted combination
        total_score = (
            0.4 * metrics.get("creativity", 0) +
            0.3 * metrics.get("length", 0) +
            0.3 * metrics.get("readability", 0)
        )

        # Return combined score and individual metrics
        return total_score, metrics

    # Create optimizer
    optimizer = MIPRO(
        metric=creative_metric,
        num_candidates=15,  # More candidates for creative task
        init_temperature=1.3,  # High creativity
        verbose=True
    )

    # Compile
    print("Optimizing for creativity, length, and readability...\n")
    writer = CreativeWriter()
    compiled_writer = optimizer.compile(writer, trainset=trainset)

    # Test prompts
    test_prompts = [
        "Write about a magical forest",
        "Describe an alien marketplace",
        "Tell me about a time-traveling paradox"
    ]

    print("\nCreative Writing Results:")
    print("-" * 40)

    for prompt in test_prompts:
        result = compiled_writer(prompt=prompt)

        # Evaluate metrics
        _, metrics = creative_metric(None, result)

        print(f"\nPrompt: {prompt}")
        print(f"Response: {result.creative_response}")
        print(f"\nMetrics:")
        print(f"  Creativity: {metrics.get('creativity', 0):.2f}")
        print(f"  Length: {metrics.get('length', 0):.2f}")
        print(f"  Readability: {metrics.get('readability', 0):.2f}")
        print("-" * 40)

# Example 4: Advanced MIPRO Configuration
def advanced_configuration():
    """Demonstrate advanced MIPRO configuration options."""
    print("\n" + "=" * 60)
    print("Example 4: Advanced MIPRO Configuration")
    print("=" * 60)

    class CodeGenerator(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate = dspy.Predict("task, requirements -> code")

        def forward(self, task, requirements=""):
            return self.generate(task=task, requirements=requirements)

    # Code generation data
    trainset = [
        dspy.Example(
            task="Create a function to check if a number is prime",
            requirements="Handle edge cases, efficient",
            code="def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0: return False\n    return True"
        ),
        dspy.Example(
            task="Write a function to reverse a string",
            requirements="Pythonic, efficient",
            code="def reverse_string(s):\n    return s[::-1]"
        ),
    ]

    # Code quality metric
    def code_quality_metric(example, pred, trace=None):
        score = 0

        if hasattr(pred, 'code'):
            code = str(pred.code)

            # Syntax check (40%)
            try:
                compile(code, '<string>', 'exec')
                score += 0.4
            except:
                pass

            # Contains function definition (20%)
            if 'def ' in code:
                score += 0.2

            # Has docstring (20%)
            if '"""' in code or "'''" in code:
                score += 0.2

            # Reasonable length (20%)
            if 50 <= len(code) <= 500:
                score += 0.2

        return score

    # Advanced MIPRO configuration
    optimizer = MIPRO(
        metric=code_quality_metric,
        num_candidates=10,
        init_temperature=0.8,  # Lower for code generation
        auto="medium",  # Use medium preset
        verbose=True,
        # Advanced parameters
        breadth=8,  # Search breadth
        depth=2,    # Search depth
        max_labeled_demos=2,
        max_bootstrapped_demos=3
    )

    # Compile
    print("Compiling with advanced configuration...\n")
    code_gen = CodeGenerator()
    compiled = optimizer.compile(code_gen, trainset=trainset)

    # Test with new tasks
    test_tasks = [
        ("Create a function to find factorial", "Use recursion"),
        ("Write a function to merge two lists", "Preserve order"),
        ("Create a function to check palindrome", "Case insensitive")
    ]

    print("\nCode Generation Results:")
    print("-" * 40)

    for task, requirements in test_tasks:
        result = compiled(task=task, requirements=requirements)

        print(f"\nTask: {task}")
        print(f"Requirements: {requirements}")
        print(f"\nGenerated Code:\n{result.code}")

        # Check quality
        quality = code_quality_metric(None, result)
        print(f"\nQuality Score: {quality:.2f}")
        print("-" * 40)

# Example 5: MIPRO vs Other Optimizers
def optimizer_comparison():
    """Compare MIPRO with other optimizers."""
    print("\n" + "=" * 60)
    print("Example 5: MIPRO vs Other Optimizers")
    print("=" * 60)

    from dspy.teleprompter import BootstrapFewShot

    class SentimentAnalyzer(dspy.Module):
        def __init__(self):
            super().__init__()
            self.analyze = dspy.Predict("text -> sentiment")

        def forward(self, text):
            return self.analyze(text=text)

    # Sentiment analysis data
    trainset = [
        dspy.Example(text="I love this product!", sentiment="positive"),
        dspy.Example(text="This is terrible quality.", sentiment="negative"),
        dspy.Example(text="It works as expected.", sentiment="neutral"),
        dspy.Example(text="Outstanding service!", sentiment="positive"),
        dspy.Example(text="Would not recommend.", sentiment="negative"),
        dspy.Example(text="Average experience.", sentiment="neutral"),
    ]

    # Test data
    testset = [
        dspy.Example(text="Absolutely fantastic!", sentiment="positive"),
        dspy.Example(text="Complete waste of money.", sentiment="negative"),
        dspy.Example(text="It's okay.", sentiment="neutral"),
    ]

    # Simple metric
    def sentiment_metric(example, pred, trace=None):
        if not hasattr(pred, 'sentiment'):
            return 0
        return str(example.sentiment).lower() == str(pred.sentiment).lower()

    # Test different optimizers
    optimizers = {
        "Baseline": None,
        "BootstrapFewShot": BootstrapFewShot(
            metric=sentiment_metric,
            max_bootstrapped_demos=3
        ),
        "MIPRO": MIPRO(
            metric=sentiment_metric,
            num_candidates=8,
            auto="light"  # Faster for comparison
        )
    }

    results = {}

    for name, optimizer in optimizers.items():
        print(f"\nTesting {name}...")
        start_time = time.time()

        if optimizer:
            # Compile
            analyzer = SentimentAnalyzer()
            compiled = optimizer.compile(analyzer, trainset=trainset)
            compile_time = time.time() - start_time
        else:
            # Baseline
            compiled = SentimentAnalyzer()
            compile_time = 0

        # Evaluate
        correct = 0
        for example in testset:
            result = compiled(text=example.text)
            if hasattr(result, 'sentiment'):
                if sentiment_metric(example, result):
                    correct += 1

        accuracy = correct / len(testset)

        results[name] = {
            "accuracy": accuracy,
            "compile_time": compile_time
        }

        print(f"  Accuracy: {accuracy:.0%}")
        print(f"  Compile Time: {compile_time:.1f}s")

    # Summary
    print("\n" + "=" * 40)
    print("Comparison Summary")
    print("=" * 40)

    for name, result in results.items():
        print(f"{name:15} | Accuracy: {result['accuracy']:>5.0%} | Time: {result['compile_time']:>5.1f}s")

# Main execution
def run_all_examples():
    """Run all MIPRO optimization examples."""
    print("DSPy MIPRO Optimization Examples")
    print("Demonstrating advanced instruction and example optimization\n")

    try:
        basic_mipro_optimization()
        instruction_evolution()
        multi_objective_optimization()
        advanced_configuration()
        optimizer_comparison()

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("All MIPRO examples completed!")
    print("=" * 60)

if __name__ == "__main__":
    run_all_examples()
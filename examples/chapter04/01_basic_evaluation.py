"""
Basic Evaluation in DSPy
========================

This example demonstrates the fundamentals of evaluating DSPy modules,
including creating datasets, defining metrics, and running evaluations.

Requirements:
    - dspy-ai
    - An LLM API key (OpenAI, Anthropic, etc.)

Usage:
    python 01_basic_evaluation.py
"""

import dspy
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# =============================================================================
# 1. Configure DSPy
# =============================================================================

def setup_dspy():
    """Configure DSPy with your preferred LLM."""
    # Use OpenAI by default
    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    dspy.configure(lm=lm)
    print("DSPy configured with GPT-4o-mini")


# =============================================================================
# 2. Create a Simple Module
# =============================================================================

class SimpleQA(dspy.Signature):
    """Answer questions with short, factual responses."""
    question: str = dspy.InputField(desc="The question to answer")
    answer: str = dspy.OutputField(desc="A short, factual answer")


# =============================================================================
# 3. Create an Evaluation Dataset
# =============================================================================

def create_qa_dataset():
    """Create a simple QA evaluation dataset."""
    examples = [
        # Geography questions
        dspy.Example(
            question="What is the capital of France?",
            answer="Paris"
        ).with_inputs("question"),

        dspy.Example(
            question="What is the capital of Japan?",
            answer="Tokyo"
        ).with_inputs("question"),

        dspy.Example(
            question="What is the capital of Australia?",
            answer="Canberra"
        ).with_inputs("question"),

        # Science questions
        dspy.Example(
            question="What is H2O commonly known as?",
            answer="water"
        ).with_inputs("question"),

        dspy.Example(
            question="What planet is known as the Red Planet?",
            answer="Mars"
        ).with_inputs("question"),

        dspy.Example(
            question="What is the chemical symbol for gold?",
            answer="Au"
        ).with_inputs("question"),

        # Math questions
        dspy.Example(
            question="What is 15 multiplied by 4?",
            answer="60"
        ).with_inputs("question"),

        dspy.Example(
            question="What is the square root of 144?",
            answer="12"
        ).with_inputs("question"),

        # History questions
        dspy.Example(
            question="In what year did World War II end?",
            answer="1945"
        ).with_inputs("question"),

        dspy.Example(
            question="Who wrote Romeo and Juliet?",
            answer="Shakespeare"
        ).with_inputs("question"),
    ]

    return examples


# =============================================================================
# 4. Define Evaluation Metrics
# =============================================================================

def exact_match_metric(example, pred, trace=None):
    """
    Check if the prediction exactly matches the expected answer.

    Args:
        example: The Example with expected 'answer' field
        pred: The Prediction with predicted 'answer' field
        trace: Optional trace for optimization

    Returns:
        bool: True if exact match (case-insensitive)
    """
    expected = example.answer.lower().strip()
    predicted = pred.answer.lower().strip()
    return expected == predicted


def contains_answer_metric(example, pred, trace=None):
    """
    Check if the prediction contains the expected answer.

    More lenient than exact match - allows for additional context.
    """
    expected = example.answer.lower().strip()
    predicted = pred.answer.lower().strip()
    return expected in predicted


def flexible_match_metric(example, pred, trace=None):
    """
    Flexible matching that handles various answer formats.

    Returns a score between 0 and 1.
    """
    expected = example.answer.lower().strip()
    predicted = pred.answer.lower().strip()

    # Exact match
    if expected == predicted:
        return 1.0

    # Contains match
    if expected in predicted or predicted in expected:
        return 0.8

    # Word overlap
    expected_words = set(expected.split())
    predicted_words = set(predicted.split())
    overlap = expected_words & predicted_words

    if overlap:
        return len(overlap) / max(len(expected_words), len(predicted_words))

    return 0.0


# =============================================================================
# 5. Run Basic Evaluation
# =============================================================================

def run_basic_evaluation():
    """Demonstrate basic evaluation workflow."""
    print("\n" + "="*60)
    print("BASIC EVALUATION DEMO")
    print("="*60)

    # Create module and dataset
    qa_module = dspy.Predict(SimpleQA)
    devset = create_qa_dataset()

    print(f"\nDataset size: {len(devset)} examples")

    # Run evaluation with exact match
    print("\n--- Exact Match Evaluation ---")
    evaluate_exact = dspy.Evaluate(
        devset=devset,
        metric=exact_match_metric,
        num_threads=4,
        display_progress=True
    )
    exact_score = evaluate_exact(qa_module)
    print(f"Exact Match Score: {exact_score:.1f}%")

    # Run evaluation with contains match
    print("\n--- Contains Match Evaluation ---")
    evaluate_contains = dspy.Evaluate(
        devset=devset,
        metric=contains_answer_metric,
        num_threads=4,
        display_progress=True
    )
    contains_score = evaluate_contains(qa_module)
    print(f"Contains Match Score: {contains_score:.1f}%")

    # Run evaluation with flexible match
    print("\n--- Flexible Match Evaluation ---")
    evaluate_flexible = dspy.Evaluate(
        devset=devset,
        metric=flexible_match_metric,
        num_threads=4,
        display_progress=True
    )
    flexible_score = evaluate_flexible(qa_module)
    print(f"Flexible Match Score: {flexible_score:.1f}%")

    return {
        'exact': exact_score,
        'contains': contains_score,
        'flexible': flexible_score
    }


# =============================================================================
# 6. Detailed Evaluation with Results
# =============================================================================

def run_detailed_evaluation():
    """Run evaluation with detailed results."""
    print("\n" + "="*60)
    print("DETAILED EVALUATION")
    print("="*60)

    qa_module = dspy.Predict(SimpleQA)
    devset = create_qa_dataset()

    # Run with return_outputs to see predictions
    evaluate = dspy.Evaluate(
        devset=devset,
        metric=flexible_match_metric,
        num_threads=4,
        display_progress=True,
        return_outputs=True
    )

    result = evaluate(qa_module)

    print(f"\nOverall Score: {result.score:.1f}%")
    print("\nDetailed Results:")
    print("-" * 60)

    for i, (example, pred, score) in enumerate(result.results[:5]):
        print(f"\nExample {i+1}:")
        print(f"  Question: {example.question}")
        print(f"  Expected: {example.answer}")
        print(f"  Got: {pred.answer}")
        print(f"  Score: {score:.2f}")

    return result


# =============================================================================
# 7. Manual Evaluation Loop
# =============================================================================

def run_manual_evaluation():
    """Demonstrate manual evaluation loop for more control."""
    print("\n" + "="*60)
    print("MANUAL EVALUATION LOOP")
    print("="*60)

    qa_module = dspy.Predict(SimpleQA)
    devset = create_qa_dataset()

    scores = []
    results = []

    for i, example in enumerate(devset):
        # Get prediction
        pred = qa_module(**example.inputs())

        # Calculate score
        score = flexible_match_metric(example, pred)
        scores.append(score)

        # Store result
        results.append({
            'question': example.question,
            'expected': example.answer,
            'predicted': pred.answer,
            'score': score
        })

        # Print progress
        status = "PASS" if score >= 0.8 else "FAIL"
        print(f"[{status}] Q: {example.question[:40]}... Score: {score:.2f}")

    # Calculate aggregate
    avg_score = sum(scores) / len(scores) if scores else 0
    pass_rate = sum(1 for s in scores if s >= 0.8) / len(scores) * 100

    print("\n" + "-"*60)
    print(f"Average Score: {avg_score:.2f}")
    print(f"Pass Rate (>= 0.8): {pass_rate:.1f}%")

    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Setup
    setup_dspy()

    # Run demonstrations
    print("\n" + "#"*60)
    print("# DSPy Basic Evaluation Examples")
    print("#"*60)

    # Basic evaluation
    basic_results = run_basic_evaluation()

    # Detailed evaluation
    detailed_result = run_detailed_evaluation()

    # Manual evaluation
    manual_results = run_manual_evaluation()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Basic Evaluation Scores:")
    print(f"  - Exact Match: {basic_results['exact']:.1f}%")
    print(f"  - Contains Match: {basic_results['contains']:.1f}%")
    print(f"  - Flexible Match: {basic_results['flexible']:.1f}%")
    print("\nEvaluation complete!")

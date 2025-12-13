"""
Evaluation Loops in DSPy
========================

This example demonstrates various evaluation workflows including
parallel evaluation, progress tracking, and detailed analysis.

Requirements:
    - dspy-ai
    - An LLM API key (OpenAI, Anthropic, etc.)

Usage:
    python 04_evaluation_loops.py
"""

import dspy
import os
import time
from typing import List, Dict, Any
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# =============================================================================
# Setup
# =============================================================================

def setup_dspy():
    """Configure DSPy with your preferred LLM."""
    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    dspy.configure(lm=lm)
    print("DSPy configured")


# Signature and module
class QASignature(dspy.Signature):
    """Answer questions concisely."""
    question: str = dspy.InputField(desc="Question to answer")
    answer: str = dspy.OutputField(desc="Concise answer")


# Sample dataset
def create_sample_dataset() -> List[dspy.Example]:
    """Create a sample evaluation dataset."""
    examples = [
        dspy.Example(question="What is the capital of France?", answer="Paris", category="geography").with_inputs("question"),
        dspy.Example(question="What is 7 * 8?", answer="56", category="math").with_inputs("question"),
        dspy.Example(question="Who wrote 1984?", answer="George Orwell", category="literature").with_inputs("question"),
        dspy.Example(question="What is the chemical symbol for gold?", answer="Au", category="science").with_inputs("question"),
        dspy.Example(question="What year did WWII end?", answer="1945", category="history").with_inputs("question"),
        dspy.Example(question="What is the largest planet?", answer="Jupiter", category="science").with_inputs("question"),
        dspy.Example(question="What is 15% of 200?", answer="30", category="math").with_inputs("question"),
        dspy.Example(question="Who painted the Mona Lisa?", answer="Leonardo da Vinci", category="art").with_inputs("question"),
        dspy.Example(question="What is the capital of Japan?", answer="Tokyo", category="geography").with_inputs("question"),
        dspy.Example(question="What is the square root of 81?", answer="9", category="math").with_inputs("question"),
    ]
    return examples


# Metric
def flexible_match(example, pred, trace=None):
    """Flexible matching metric."""
    expected = example.answer.lower().strip()
    predicted = pred.answer.lower().strip()

    if expected == predicted:
        return 1.0
    if expected in predicted or predicted in expected:
        return 0.8
    return 0.0


# =============================================================================
# 1. Basic Evaluation with dspy.Evaluate
# =============================================================================

def basic_evaluation_demo(module, devset):
    """Demonstrate basic dspy.Evaluate usage."""
    print("\n" + "="*60)
    print("1. BASIC EVALUATION")
    print("="*60)

    evaluate = dspy.Evaluate(
        devset=devset,
        metric=flexible_match,
        num_threads=4,
        display_progress=True
    )

    score = evaluate(module)
    print(f"\nFinal Score: {score:.1f}%")

    return score


# =============================================================================
# 2. Evaluation with Detailed Results
# =============================================================================

def detailed_evaluation_demo(module, devset):
    """Demonstrate evaluation with detailed results."""
    print("\n" + "="*60)
    print("2. DETAILED EVALUATION")
    print("="*60)

    evaluate = dspy.Evaluate(
        devset=devset,
        metric=flexible_match,
        num_threads=4,
        display_progress=True,
        return_outputs=True
    )

    result = evaluate(module)

    print(f"\nAggregate Score: {result.score:.1f}%")
    print("\nPer-Example Results:")
    print("-" * 60)

    for i, (example, pred, score) in enumerate(result.results):
        status = "PASS" if score >= 0.8 else "FAIL"
        print(f"[{status}] Q: {example.question[:35]:35} | Expected: {example.answer[:10]:10} | Got: {pred.answer[:15]:15} | Score: {score:.2f}")

    return result


# =============================================================================
# 3. Parallel vs Sequential Evaluation
# =============================================================================

def parallel_comparison_demo(module, devset):
    """Compare parallel vs sequential evaluation speed."""
    print("\n" + "="*60)
    print("3. PARALLEL VS SEQUENTIAL")
    print("="*60)

    # Sequential (1 thread)
    print("\nRunning sequential evaluation (1 thread)...")
    start = time.time()
    evaluate_seq = dspy.Evaluate(
        devset=devset,
        metric=flexible_match,
        num_threads=1,
        display_progress=True
    )
    score_seq = evaluate_seq(module)
    time_seq = time.time() - start

    # Parallel (4 threads)
    print("\nRunning parallel evaluation (4 threads)...")
    start = time.time()
    evaluate_par = dspy.Evaluate(
        devset=devset,
        metric=flexible_match,
        num_threads=4,
        display_progress=True
    )
    score_par = evaluate_par(module)
    time_par = time.time() - start

    print("\nResults:")
    print(f"  Sequential: {score_seq:.1f}% in {time_seq:.2f}s")
    print(f"  Parallel:   {score_par:.1f}% in {time_par:.2f}s")
    print(f"  Speedup:    {time_seq/time_par:.2f}x")

    return {'sequential': time_seq, 'parallel': time_par}


# =============================================================================
# 4. Manual Evaluation Loop
# =============================================================================

def manual_evaluation_demo(module, devset):
    """Demonstrate manual evaluation loop."""
    print("\n" + "="*60)
    print("4. MANUAL EVALUATION LOOP")
    print("="*60)

    results = {
        'scores': [],
        'predictions': [],
        'by_category': defaultdict(list)
    }

    for i, example in enumerate(devset):
        # Get prediction
        pred = module(**example.inputs())

        # Calculate score
        score = flexible_match(example, pred)

        # Store results
        results['scores'].append(score)
        results['predictions'].append({
            'question': example.question,
            'expected': example.answer,
            'predicted': pred.answer,
            'score': score
        })

        # Track by category
        category = getattr(example, 'category', 'unknown')
        results['by_category'][category].append(score)

        # Progress
        print(f"[{i+1}/{len(devset)}] {example.question[:40]:40} Score: {score:.2f}")

    # Summary
    avg_score = sum(results['scores']) / len(results['scores'])
    print("\n" + "-"*60)
    print(f"Average Score: {avg_score:.2f}")

    print("\nBy Category:")
    for category, scores in results['by_category'].items():
        cat_avg = sum(scores) / len(scores)
        print(f"  {category}: {cat_avg:.2f} ({len(scores)} examples)")

    return results


# =============================================================================
# 5. Error Analysis
# =============================================================================

def error_analysis_demo(module, devset):
    """Demonstrate error analysis."""
    print("\n" + "="*60)
    print("5. ERROR ANALYSIS")
    print("="*60)

    evaluate = dspy.Evaluate(
        devset=devset,
        metric=flexible_match,
        num_threads=4,
        return_outputs=True
    )

    result = evaluate(module)

    # Categorize errors
    errors = {
        'complete_miss': [],
        'partial_match': [],
        'success': []
    }

    for example, pred, score in result.results:
        if score >= 0.8:
            errors['success'].append((example, pred, score))
        elif score > 0:
            errors['partial_match'].append((example, pred, score))
        else:
            errors['complete_miss'].append((example, pred, score))

    print(f"\nSuccess: {len(errors['success'])} ({100*len(errors['success'])/len(devset):.1f}%)")
    print(f"Partial: {len(errors['partial_match'])} ({100*len(errors['partial_match'])/len(devset):.1f}%)")
    print(f"Miss:    {len(errors['complete_miss'])} ({100*len(errors['complete_miss'])/len(devset):.1f}%)")

    if errors['complete_miss']:
        print("\nComplete Misses:")
        for example, pred, score in errors['complete_miss'][:3]:
            print(f"  Q: {example.question}")
            print(f"  Expected: {example.answer}")
            print(f"  Got: {pred.answer}")
            print()

    return errors


# =============================================================================
# 6. A/B Testing Workflow
# =============================================================================

def ab_testing_demo(devset):
    """Demonstrate A/B testing between module versions."""
    print("\n" + "="*60)
    print("6. A/B TESTING")
    print("="*60)

    # Module A: Simple Predict
    module_a = dspy.Predict(QASignature)

    # Module B: Chain of Thought (might perform better on reasoning)
    module_b = dspy.ChainOfThought(QASignature)

    evaluate = dspy.Evaluate(
        devset=devset,
        metric=flexible_match,
        num_threads=4,
        display_progress=True
    )

    print("\nEvaluating Module A (Predict)...")
    score_a = evaluate(module_a)

    print("\nEvaluating Module B (ChainOfThought)...")
    score_b = evaluate(module_b)

    print("\n" + "-"*60)
    print("A/B TEST RESULTS")
    print("-"*60)
    print(f"Module A (Predict):        {score_a:.1f}%")
    print(f"Module B (ChainOfThought): {score_b:.1f}%")
    print(f"Difference:                {score_b - score_a:+.1f}%")

    if score_b > score_a:
        print("\nRecommendation: Module B (ChainOfThought) performs better")
    elif score_a > score_b:
        print("\nRecommendation: Module A (Predict) performs better")
    else:
        print("\nRecommendation: Both modules perform equally")

    return {'module_a': score_a, 'module_b': score_b}


# =============================================================================
# 7. Comprehensive Evaluation Report
# =============================================================================

def comprehensive_report_demo(module, devset):
    """Generate a comprehensive evaluation report."""
    print("\n" + "="*60)
    print("7. COMPREHENSIVE EVALUATION REPORT")
    print("="*60)

    # Run detailed evaluation
    evaluate = dspy.Evaluate(
        devset=devset,
        metric=flexible_match,
        num_threads=4,
        return_outputs=True
    )
    result = evaluate(module)

    # Collect statistics
    scores = [score for _, _, score in result.results]
    by_category = defaultdict(list)

    for example, pred, score in result.results:
        category = getattr(example, 'category', 'unknown')
        by_category[category].append(score)

    # Report
    print("\n" + "="*60)
    print("                    EVALUATION REPORT")
    print("="*60)

    print(f"\nOverall Score: {result.score:.1f}%")
    print(f"Total Examples: {len(devset)}")
    print(f"Pass Rate (>=80%): {100 * sum(1 for s in scores if s >= 0.8) / len(scores):.1f}%")

    print("\nScore Distribution:")
    print(f"  Perfect (100%): {sum(1 for s in scores if s == 1.0)}")
    print(f"  High (80-99%):  {sum(1 for s in scores if 0.8 <= s < 1.0)}")
    print(f"  Medium (50-79%): {sum(1 for s in scores if 0.5 <= s < 0.8)}")
    print(f"  Low (1-49%):    {sum(1 for s in scores if 0 < s < 0.5)}")
    print(f"  Zero (0%):      {sum(1 for s in scores if s == 0)}")

    print("\nPerformance by Category:")
    print("-"*40)
    for category, cat_scores in sorted(by_category.items()):
        avg = sum(cat_scores) / len(cat_scores)
        print(f"  {category:15} {avg*100:6.1f}%  ({len(cat_scores)} examples)")

    print("\nBest Performing Examples:")
    best = sorted(result.results, key=lambda x: -x[2])[:3]
    for example, pred, score in best:
        print(f"  [{score:.0%}] {example.question[:50]}")

    print("\nWorst Performing Examples:")
    worst = sorted(result.results, key=lambda x: x[2])[:3]
    for example, pred, score in worst:
        print(f"  [{score:.0%}] {example.question[:50]}")

    print("\n" + "="*60)

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all evaluation demos."""
    setup_dspy()

    # Create module and dataset
    module = dspy.Predict(QASignature)
    devset = create_sample_dataset()

    print("\n" + "#"*60)
    print("# DSPy Evaluation Loops Demo")
    print("#"*60)

    # Run demos
    basic_evaluation_demo(module, devset)
    detailed_evaluation_demo(module, devset)
    parallel_comparison_demo(module, devset)
    manual_evaluation_demo(module, devset)
    error_analysis_demo(module, devset)
    ab_testing_demo(devset)
    comprehensive_report_demo(module, devset)

    print("\n" + "#"*60)
    print("# ALL DEMOS COMPLETE")
    print("#"*60)


if __name__ == "__main__":
    main()

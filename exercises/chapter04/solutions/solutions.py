"""
Chapter 4: Evaluation - Exercise Solutions
==========================================

Complete solutions for all Chapter 4 exercises.
"""

import dspy
import random
from typing import List, Tuple, Dict, Any
from collections import defaultdict, Counter
from difflib import SequenceMatcher
from datetime import datetime


# =============================================================================
# EXERCISE 1: Building a Quality Dataset
# =============================================================================

def exercise01_solution() -> Tuple[List, List, List]:
    """
    Create a balanced sentiment analysis dataset.

    Returns:
        Tuple of (trainset, devset, testset)
    """
    examples = []

    # Positive examples (10+)
    positive_texts = [
        ("This product is amazing! Best purchase ever!", 0.95),
        ("Excellent quality and fast shipping.", 0.90),
        ("I love it! Exactly what I needed.", 0.92),
        ("Highly recommend to everyone.", 0.88),
        ("Fantastic customer service.", 0.85),
        ("Great value for money.", 0.82),
        ("Works perfectly, very satisfied.", 0.90),
        ("Exceeded my expectations!", 0.93),
        ("Will definitely buy again.", 0.87),
        ("Beautiful design and great functionality.", 0.89),
    ]

    # Negative examples (10+)
    negative_texts = [
        ("Terrible product, complete waste of money.", 0.95),
        ("Would not recommend. Very disappointed.", 0.90),
        ("Broke after one week. Poor quality.", 0.88),
        ("Customer service was unhelpful.", 0.85),
        ("Does not work as advertised.", 0.92),
        ("Worst purchase I've ever made.", 0.93),
        ("Cheaply made and overpriced.", 0.87),
        ("Returned it immediately.", 0.84),
        ("Save your money, don't buy this.", 0.91),
        ("Extremely frustrating experience.", 0.86),
    ]

    # Neutral examples (5+)
    neutral_texts = [
        ("It's okay, nothing special.", 0.75),
        ("Does what it says, no more no less.", 0.80),
        ("Average product for the price.", 0.70),
        ("Neither good nor bad.", 0.85),
        ("Mixed feelings about this one.", 0.65),
        ("Has pros and cons.", 0.60),
    ]

    # Edge cases (5+)
    edge_cases = [
        # Sarcastic (should be negative but sounds positive)
        ("Oh great, another broken product. Just what I needed.", "negative", 0.70),
        # Mixed sentiment
        ("The food was delicious but the service was terrible.", "neutral", 0.55),
        # Very short
        ("Meh.", "neutral", 0.60),
        # Question format
        ("Is this even worth the money?", "negative", 0.65),
        # Emoji heavy
        ("Love it! Best ever!", "positive", 0.80),
        # Ambiguous
        ("It works.", "neutral", 0.50),
    ]

    # Create examples
    for text, confidence in positive_texts:
        examples.append(dspy.Example(
            text=text,
            sentiment="positive",
            confidence=confidence
        ).with_inputs("text"))

    for text, confidence in negative_texts:
        examples.append(dspy.Example(
            text=text,
            sentiment="negative",
            confidence=confidence
        ).with_inputs("text"))

    for text, confidence in neutral_texts:
        examples.append(dspy.Example(
            text=text,
            sentiment="neutral",
            confidence=confidence
        ).with_inputs("text"))

    for text, sentiment, confidence in edge_cases:
        examples.append(dspy.Example(
            text=text,
            sentiment=sentiment,
            confidence=confidence
        ).with_inputs("text"))

    print(f"Total examples: {len(examples)}")

    # Check balance
    sentiments = Counter(ex.sentiment for ex in examples)
    print(f"Distribution: {dict(sentiments)}")

    # Shuffle with fixed seed
    random.Random(42).shuffle(examples)

    # Split 60/20/20
    n = len(examples)
    train_end = int(n * 0.6)
    dev_end = int(n * 0.8)

    trainset = examples[:train_end]
    devset = examples[train_end:dev_end]
    testset = examples[dev_end:]

    print(f"Splits: Train={len(trainset)}, Dev={len(devset)}, Test={len(testset)}")

    # Validate
    for split_name, split_data in [("train", trainset), ("dev", devset), ("test", testset)]:
        for ex in split_data:
            assert ex.inputs(), f"Missing inputs in {split_name}"
            assert hasattr(ex, 'text') and ex.text, f"Missing text in {split_name}"
            assert hasattr(ex, 'sentiment'), f"Missing sentiment in {split_name}"

    print("Validation passed!")

    return trainset, devset, testset


# =============================================================================
# EXERCISE 2: Designing a Comprehensive Metric
# =============================================================================

def exercise02_solution(example, pred, trace=None):
    """
    Comprehensive QA quality metric.

    Evaluates:
    - Correctness (40%): Does answer contain expected information?
    - Completeness (30%): Are all key points addressed?
    - Conciseness (20%): Is answer appropriately brief?
    - Format (10%): Is answer well-formed?

    Args:
        example: dspy.Example with 'question', 'answer', optionally 'key_points'
        pred: Prediction with 'answer' field
        trace: Optional trace for optimization

    Returns:
        float: Quality score between 0 and 1
    """
    # Handle missing fields
    expected = getattr(example, 'answer', '')
    predicted = getattr(pred, 'answer', '')

    if not expected or not predicted:
        return 0.0

    # Normalize for comparison
    expected_lower = expected.lower().strip()
    predicted_lower = predicted.lower().strip()

    # 1. CORRECTNESS (40%)
    if expected_lower == predicted_lower:
        correctness_score = 1.0
    elif expected_lower in predicted_lower:
        correctness_score = 0.9
    elif predicted_lower in expected_lower:
        correctness_score = 0.7
    else:
        # Check word overlap
        expected_words = set(expected_lower.split())
        predicted_words = set(predicted_lower.split())
        overlap = expected_words & predicted_words
        if overlap:
            correctness_score = len(overlap) / len(expected_words) * 0.6
        else:
            correctness_score = 0.0

    # 2. COMPLETENESS (30%)
    key_points = getattr(example, 'key_points', [])
    if key_points:
        found = sum(1 for kp in key_points if kp.lower() in predicted_lower)
        completeness_score = found / len(key_points)
    else:
        # No key points specified, use correctness as proxy
        completeness_score = correctness_score

    # 3. CONCISENESS (20%)
    word_count = len(predicted.split())
    expected_word_count = len(expected.split())

    if expected_word_count > 0:
        # Allow 0.5x to 3x the expected length
        min_words = max(3, expected_word_count * 0.5)
        max_words = expected_word_count * 3

        if min_words <= word_count <= max_words:
            conciseness_score = 1.0
        elif word_count < min_words:
            conciseness_score = 0.5  # Too short
        else:
            # Penalize verbosity
            excess_ratio = word_count / max_words
            conciseness_score = max(0.3, 1.0 - (excess_ratio - 1) * 0.3)
    else:
        conciseness_score = 0.5 if 3 <= word_count <= 50 else 0.3

    # 4. FORMAT (10%)
    format_score = 0.0

    # Not empty
    if predicted.strip():
        format_score += 0.4

    # Starts with capital letter
    if predicted and predicted[0].isupper():
        format_score += 0.2

    # Ends with punctuation
    if predicted and predicted.strip()[-1] in '.!?':
        format_score += 0.2

    # No repeated consecutive words
    words = predicted.split()
    has_repeats = any(words[i] == words[i+1] for i in range(len(words)-1)) if len(words) > 1 else False
    if not has_repeats:
        format_score += 0.2

    # Combine with weights
    final_score = (
        0.4 * correctness_score +
        0.3 * completeness_score +
        0.2 * conciseness_score +
        0.1 * format_score
    )

    # For optimization, require high threshold
    if trace is not None:
        return final_score >= 0.7

    return final_score


# =============================================================================
# EXERCISE 3: Comprehensive Evaluation Pipeline
# =============================================================================

def exercise03_solution(module, devset, metric, category_field=None):
    """
    Comprehensive evaluation with detailed analysis.

    Args:
        module: DSPy module to evaluate
        devset: Evaluation dataset
        metric: Metric function
        category_field: Optional field name for category breakdown

    Returns:
        dict: Detailed evaluation results
    """
    results = {
        'aggregate_score': 0.0,
        'total_examples': len(devset),
        'successful': 0,
        'failed': 0,
        'errors': 0,
        'by_category': defaultdict(list),
        'error_analysis': defaultdict(int),
        'all_results': [],
        'best_examples': [],
        'worst_examples': []
    }

    for i, example in enumerate(devset):
        try:
            # Get prediction
            pred = module(**example.inputs())

            # Calculate score
            score = metric(example, pred)

            # Store result
            result = {
                'index': i,
                'example': example,
                'prediction': pred,
                'score': score
            }
            results['all_results'].append(result)

            # Track success/failure
            if score >= 0.8:
                results['successful'] += 1
            else:
                results['failed'] += 1

                # Categorize error
                if score == 0:
                    results['error_analysis']['complete_miss'] += 1
                elif score < 0.5:
                    results['error_analysis']['low_quality'] += 1
                else:
                    results['error_analysis']['partial_match'] += 1

            # Track by category
            if category_field:
                category = getattr(example, category_field, 'unknown')
                results['by_category'][category].append(score)

            # Progress update
            if (i + 1) % 10 == 0 or i == len(devset) - 1:
                current_avg = sum(r['score'] for r in results['all_results']) / len(results['all_results'])
                print(f"Progress: {i+1}/{len(devset)} | Current avg: {current_avg:.2f}")

        except Exception as e:
            results['errors'] += 1
            results['error_analysis']['exception'] += 1
            results['all_results'].append({
                'index': i,
                'example': example,
                'prediction': None,
                'score': 0,
                'error': str(e)
            })

    # Calculate aggregate score
    scores = [r['score'] for r in results['all_results']]
    results['aggregate_score'] = sum(scores) / len(scores) if scores else 0

    # Convert category scores to averages
    results['by_category'] = {
        cat: sum(scores) / len(scores)
        for cat, scores in results['by_category'].items()
    }

    # Sort for best/worst
    sorted_results = sorted(results['all_results'], key=lambda x: x['score'], reverse=True)
    results['best_examples'] = sorted_results[:5]
    results['worst_examples'] = sorted_results[-5:]

    # Convert error_analysis to dict
    results['error_analysis'] = dict(results['error_analysis'])

    return results


# =============================================================================
# EXERCISE 4: Preventing Data Leakage
# =============================================================================

def exercise04_solution(
    data: List[dspy.Example],
    key_field: str = 'question',
    similarity_threshold: float = 0.85,
    train_ratio: float = 0.6,
    dev_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List, List, List, Dict]:
    """
    Split data while preventing various forms of leakage.

    Args:
        data: List of dspy.Example objects
        key_field: Field to use for similarity comparison
        similarity_threshold: Threshold for considering items similar
        train_ratio: Fraction for training set
        dev_ratio: Fraction for dev set
        seed: Random seed

    Returns:
        Tuple of (trainset, devset, testset, stats)
    """
    stats = {
        'original_count': len(data),
        'duplicates_removed': 0,
        'similarity_groups': 0,
        'items_grouped': 0
    }

    def get_key(example):
        return getattr(example, key_field, '')

    def similar(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    # Step 1: Remove exact duplicates
    seen_keys = set()
    unique_data = []
    for ex in data:
        key = get_key(ex)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_data.append(ex)
        else:
            stats['duplicates_removed'] += 1

    print(f"After dedup: {len(unique_data)} examples")

    # Step 2: Group similar items
    groups = []
    assigned = set()

    for i, ex1 in enumerate(unique_data):
        if i in assigned:
            continue

        group = [ex1]
        key1 = get_key(ex1)

        for j, ex2 in enumerate(unique_data[i+1:], i+1):
            if j in assigned:
                continue

            key2 = get_key(ex2)
            if similar(key1, key2) >= similarity_threshold:
                group.append(ex2)
                assigned.add(j)

        if len(group) > 1:
            stats['items_grouped'] += len(group)

        groups.append(group)
        assigned.add(i)

    stats['similarity_groups'] = len(groups)
    print(f"Created {len(groups)} groups")

    # Step 3: Shuffle groups
    rng = random.Random(seed)
    rng.shuffle(groups)

    # Step 4: Assign groups to splits
    n_groups = len(groups)
    train_end = int(n_groups * train_ratio)
    dev_end = int(n_groups * (train_ratio + dev_ratio))

    train_groups = groups[:train_end]
    dev_groups = groups[train_end:dev_end]
    test_groups = groups[dev_end:]

    # Step 5: Flatten groups
    trainset = [ex for g in train_groups for ex in g]
    devset = [ex for g in dev_groups for ex in g]
    testset = [ex for g in test_groups for ex in g]

    # Shuffle within splits
    rng.shuffle(trainset)
    rng.shuffle(devset)
    rng.shuffle(testset)

    stats['final_counts'] = {
        'train': len(trainset),
        'dev': len(devset),
        'test': len(testset)
    }

    return trainset, devset, testset, stats


def verify_no_leakage(trainset, devset, testset, key_field='question', threshold=0.85):
    """Verify no similar items across splits."""
    def similar(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def get_key(ex):
        return getattr(ex, key_field, '')

    issues = []

    # Sample for efficiency (check first 50 from each)
    train_sample = trainset[:50]
    dev_sample = devset[:50]
    test_sample = testset[:50]

    # Check train vs dev
    for train_ex in train_sample:
        for dev_ex in dev_sample:
            sim = similar(get_key(train_ex), get_key(dev_ex))
            if sim >= threshold:
                issues.append({
                    'type': 'train-dev',
                    'similarity': sim,
                    'train_text': get_key(train_ex)[:50],
                    'dev_text': get_key(dev_ex)[:50]
                })

    # Check train vs test
    for train_ex in train_sample:
        for test_ex in test_sample:
            sim = similar(get_key(train_ex), get_key(test_ex))
            if sim >= threshold:
                issues.append({
                    'type': 'train-test',
                    'similarity': sim,
                    'train_text': get_key(train_ex)[:50],
                    'test_text': get_key(test_ex)[:50]
                })

    # Check dev vs test
    for dev_ex in dev_sample:
        for test_ex in test_sample:
            sim = similar(get_key(dev_ex), get_key(test_ex))
            if sim >= threshold:
                issues.append({
                    'type': 'dev-test',
                    'similarity': sim,
                    'dev_text': get_key(dev_ex)[:50],
                    'test_text': get_key(test_ex)[:50]
                })

    return issues


# =============================================================================
# EXERCISE 5: Evaluation Report Generator
# =============================================================================

def exercise05_solution(
    module_name: str,
    results: Dict[str, Any],
    historical: List[Dict] = None,
    output_path: str = None
) -> str:
    """
    Generate comprehensive evaluation report in Markdown.

    Args:
        module_name: Name of the module
        results: Results from comprehensive_evaluation()
        historical: Optional list of past results for trend analysis
        output_path: Optional path to save report

    Returns:
        str: Markdown formatted report
    """
    report = []

    # Header
    report.append(f"# Evaluation Report: {module_name}")
    report.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Total Examples**: {results.get('total_examples', 'N/A')}")

    # Executive Summary
    report.append("\n## Executive Summary\n")

    score = results.get('aggregate_score', 0)
    status = "PASS" if score >= 0.8 else "NEEDS IMPROVEMENT" if score >= 0.6 else "FAIL"

    report.append(f"**Overall Score**: {score*100:.1f}%")
    report.append(f"**Status**: {status}")

    # Key findings
    report.append("\n### Key Findings\n")
    if score >= 0.8:
        report.append("- Module meets quality standards")
    elif score >= 0.6:
        report.append("- Module shows promise but needs improvement")
    else:
        report.append("- Module requires significant improvements")

    if results.get('by_category'):
        best_cat = max(results['by_category'].items(), key=lambda x: x[1])
        worst_cat = min(results['by_category'].items(), key=lambda x: x[1])
        report.append(f"- Best category: {best_cat[0]} ({best_cat[1]*100:.1f}%)")
        report.append(f"- Needs work: {worst_cat[0]} ({worst_cat[1]*100:.1f}%)")

    # Performance Metrics
    report.append("\n## Performance Metrics\n")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    report.append(f"| Overall Score | {score*100:.1f}% |")
    report.append(f"| Successful | {results.get('successful', 'N/A')} |")
    report.append(f"| Failed | {results.get('failed', 'N/A')} |")
    report.append(f"| Errors | {results.get('errors', 0)} |")

    # Score Distribution
    if results.get('all_results'):
        scores = [r['score'] for r in results['all_results']]
        report.append("\n### Score Distribution\n")
        report.append("```")
        bins = [
            ("Perfect (100%)", sum(1 for s in scores if s == 1.0)),
            ("High (80-99%)", sum(1 for s in scores if 0.8 <= s < 1.0)),
            ("Medium (50-79%)", sum(1 for s in scores if 0.5 <= s < 0.8)),
            ("Low (1-49%)", sum(1 for s in scores if 0 < s < 0.5)),
            ("Zero (0%)", sum(1 for s in scores if s == 0))
        ]
        max_count = max(b[1] for b in bins) or 1
        for label, count in bins:
            bar = "#" * int(count / max_count * 30)
            report.append(f"{label:15} {count:4} {bar}")
        report.append("```")

    # Category Breakdown
    if results.get('by_category'):
        report.append("\n## Performance by Category\n")
        report.append("| Category | Score | Status |")
        report.append("|----------|-------|--------|")

        for cat, cat_score in sorted(results['by_category'].items(), key=lambda x: -x[1]):
            status_icon = "Pass" if cat_score >= 0.8 else "Warn" if cat_score >= 0.6 else "Fail"
            report.append(f"| {cat} | {cat_score*100:.1f}% | {status_icon} |")

    # Trend Analysis
    if historical:
        report.append("\n## Trend Analysis\n")
        report.append("```")
        for i, hist in enumerate(historical[-5:]):  # Last 5 results
            h_score = hist.get('score', 0)
            bar = "#" * int(h_score * 30)
            report.append(f"Run {len(historical)-4+i}: {h_score*100:5.1f}% {bar}")
        # Current
        bar = "#" * int(score * 30)
        report.append(f"Current:  {score*100:5.1f}% {bar}")
        report.append("```")

        # Trend direction
        if len(historical) >= 2:
            prev_score = historical[-1].get('score', 0)
            diff = score - prev_score
            if diff > 0.05:
                report.append(f"\nImprovement of {diff*100:.1f}% from previous run")
            elif diff < -0.05:
                report.append(f"\nRegression of {abs(diff)*100:.1f}% from previous run")
            else:
                report.append("\nStable performance compared to previous run")

    # Error Analysis
    if results.get('error_analysis'):
        report.append("\n## Error Analysis\n")
        report.append("| Error Type | Count |")
        report.append("|------------|-------|")
        for error_type, count in results['error_analysis'].items():
            report.append(f"| {error_type} | {count} |")

    # Sample Errors
    if results.get('worst_examples'):
        report.append("\n### Sample Failures\n")
        for i, worst in enumerate(results['worst_examples'][:3]):
            ex = worst.get('example', {})
            pred = worst.get('prediction', {})
            report.append(f"**Failure {i+1}** (Score: {worst.get('score', 0):.2f})")
            report.append(f"- Input: {getattr(ex, 'question', str(ex))[:80]}...")
            report.append(f"- Expected: {getattr(ex, 'answer', 'N/A')[:50]}...")
            if pred:
                report.append(f"- Got: {getattr(pred, 'answer', str(pred))[:50]}...")
            report.append("")

    # Recommendations
    report.append("\n## Recommendations\n")

    if score >= 0.9:
        report.append("1. Module is performing well - consider deploying")
        report.append("2. Continue monitoring for regression")
    elif score >= 0.7:
        report.append("1. Review and address failing categories")
        report.append("2. Consider optimizing with more training data")
        report.append("3. Analyze error patterns for targeted improvements")
    else:
        report.append("1. **Priority**: Address fundamental issues before deployment")
        report.append("2. Review signature and module design")
        report.append("3. Increase training data quantity and quality")
        report.append("4. Consider using ChainOfThought for complex tasks")

    # Footer
    report.append("\n---")
    report.append(f"*Report generated by DSPy Evaluation Framework*")

    # Join and optionally save
    full_report = "\n".join(report)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(full_report)
        print(f"Report saved to {output_path}")

    return full_report


# =============================================================================
# TEST ALL SOLUTIONS
# =============================================================================

def test_all_solutions():
    """Test all exercise solutions."""
    print("="*60)
    print("TESTING ALL EXERCISE SOLUTIONS")
    print("="*60)

    # Exercise 1
    print("\n--- Exercise 1: Dataset Creation ---")
    train, dev, test = exercise01_solution()
    assert len(train) > 0, "Training set should not be empty"
    assert len(dev) > 0, "Dev set should not be empty"
    assert len(test) > 0, "Test set should not be empty"
    print("Exercise 1 PASSED")

    # Exercise 2
    print("\n--- Exercise 2: Custom Metric ---")
    example = dspy.Example(
        question="What is the capital of France?",
        answer="Paris",
        key_points=["Paris", "France", "capital"]
    ).with_inputs("question")

    class MockPred:
        def __init__(self, answer):
            self.answer = answer

    # Test cases
    assert exercise02_solution(example, MockPred("Paris")) >= 0.9, "Perfect match should score high"
    assert exercise02_solution(example, MockPred("The capital of France is Paris.")) >= 0.7, "Good answer should score well"
    assert exercise02_solution(example, MockPred("London")) < 0.3, "Wrong answer should score low"
    print("Exercise 2 PASSED")

    # Exercise 3
    print("\n--- Exercise 3: Comprehensive Evaluation ---")
    # Would need actual module to fully test

    # Exercise 4
    print("\n--- Exercise 4: Data Leakage Prevention ---")
    test_data = [
        dspy.Example(question="What is machine learning?", answer="...").with_inputs("question"),
        dspy.Example(question="What is machine learning?", answer="...").with_inputs("question"),  # Duplicate
        dspy.Example(question="What is ML?", answer="...").with_inputs("question"),  # Similar
        dspy.Example(question="What is deep learning?", answer="...").with_inputs("question"),
        dspy.Example(question="What is AI?", answer="...").with_inputs("question"),
    ]
    train, dev, test, stats = exercise04_solution(test_data)
    assert stats['duplicates_removed'] >= 1, "Should remove duplicates"
    print(f"Stats: {stats}")
    print("Exercise 4 PASSED")

    # Exercise 5
    print("\n--- Exercise 5: Report Generation ---")
    sample_results = {
        'aggregate_score': 0.82,
        'total_examples': 100,
        'successful': 75,
        'failed': 25,
        'errors': 0,
        'by_category': {'geography': 0.90, 'science': 0.75},
        'error_analysis': {'wrong_answer': 20, 'incomplete': 5},
        'all_results': [{'score': 0.8}] * 100,
        'worst_examples': [{'example': dspy.Example(question="test", answer="ans").with_inputs("question"),
                          'prediction': MockPred("wrong"), 'score': 0.2}]
    }
    report = exercise05_solution("Test Module", sample_results)
    assert "# Evaluation Report" in report, "Report should have header"
    assert "82" in report, "Report should contain score"
    print("Exercise 5 PASSED")

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    test_all_solutions()

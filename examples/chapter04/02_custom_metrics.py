"""
Custom Metrics in DSPy
======================

This example demonstrates how to design and implement custom evaluation
metrics for various DSPy tasks.

Requirements:
    - dspy-ai
    - An LLM API key (OpenAI, Anthropic, etc.)

Usage:
    python 02_custom_metrics.py
"""

import dspy
import os
import re
from typing import List, Set
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


# =============================================================================
# 1. Boolean Metrics
# =============================================================================

def exact_match(example, pred, trace=None):
    """
    Simple exact match metric.

    Returns True if prediction exactly matches expected answer.
    """
    expected = example.answer.lower().strip()
    predicted = pred.answer.lower().strip()
    return expected == predicted


def normalized_match(example, pred, trace=None):
    """
    Match after normalization (removing punctuation, extra spaces).
    """
    def normalize(text):
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Lowercase and normalize whitespace
        text = ' '.join(text.lower().split())
        return text

    expected = normalize(example.answer)
    predicted = normalize(pred.answer)
    return expected == predicted


def any_correct_answer(example, pred, trace=None):
    """
    Check if prediction matches any of multiple correct answers.

    Expects example.answers to be a list of acceptable answers.
    """
    answers = getattr(example, 'answers', [example.answer])
    predicted = pred.answer.lower().strip()

    return any(
        ans.lower().strip() in predicted or predicted in ans.lower().strip()
        for ans in answers
    )


# =============================================================================
# 2. Numeric Score Metrics
# =============================================================================

def word_overlap_score(example, pred, trace=None):
    """
    Calculate word overlap between expected and predicted answers.

    Returns float between 0 and 1.
    """
    def get_words(text):
        return set(text.lower().split())

    expected_words = get_words(example.answer)
    predicted_words = get_words(pred.answer)

    if not expected_words:
        return 0.0

    overlap = expected_words & predicted_words
    return len(overlap) / len(expected_words)


def jaccard_similarity(example, pred, trace=None):
    """
    Calculate Jaccard similarity between word sets.

    Returns float between 0 and 1.
    """
    def get_words(text):
        return set(text.lower().split())

    expected_words = get_words(example.answer)
    predicted_words = get_words(pred.answer)

    if not expected_words and not predicted_words:
        return 1.0
    if not expected_words or not predicted_words:
        return 0.0

    intersection = len(expected_words & predicted_words)
    union = len(expected_words | predicted_words)

    return intersection / union


def length_ratio_score(example, pred, trace=None):
    """
    Score based on how close the predicted length is to expected.

    Returns float between 0 and 1.
    """
    expected_len = len(example.answer)
    predicted_len = len(pred.answer)

    if expected_len == 0:
        return 0.0 if predicted_len > 0 else 1.0

    ratio = min(predicted_len, expected_len) / max(predicted_len, expected_len)
    return ratio


# =============================================================================
# 3. Composite Metrics
# =============================================================================

def comprehensive_qa_metric(example, pred, trace=None):
    """
    Comprehensive metric combining multiple quality dimensions.

    Evaluates:
    - Correctness (40%): Does answer contain expected information?
    - Completeness (30%): Are all key points addressed?
    - Conciseness (20%): Is answer appropriately brief?
    - Format (10%): Is answer well-formatted?
    """
    # 1. Correctness (40%)
    expected = example.answer.lower()
    predicted = pred.answer.lower()
    if expected in predicted:
        correctness = 1.0
    elif any(word in predicted for word in expected.split()):
        correctness = 0.5
    else:
        correctness = 0.0

    # 2. Completeness (30%)
    key_points = getattr(example, 'key_points', [])
    if key_points:
        found = sum(1 for kp in key_points if kp.lower() in predicted)
        completeness = found / len(key_points)
    else:
        completeness = 1.0 if correctness > 0 else 0.0

    # 3. Conciseness (20%)
    word_count = len(pred.answer.split())
    if 5 <= word_count <= 50:
        conciseness = 1.0
    elif word_count < 3:
        conciseness = 0.3
    elif word_count > 100:
        conciseness = 0.5
    else:
        conciseness = 0.7

    # 4. Format (10%)
    format_score = 0.0
    if pred.answer.strip():  # Not empty
        format_score += 0.5
    if pred.answer[0].isupper() if pred.answer else False:  # Capitalized
        format_score += 0.25
    if pred.answer.strip().endswith('.') if pred.answer else False:  # Ends with period
        format_score += 0.25

    # Combine
    final = (
        0.4 * correctness +
        0.3 * completeness +
        0.2 * conciseness +
        0.1 * format_score
    )

    # For optimization, require high threshold
    if trace is not None:
        return final >= 0.7

    return final


def multi_aspect_metric(example, pred, trace=None):
    """
    Multi-aspect evaluation returning detailed scores.
    """
    scores = {
        'accuracy': 1.0 if example.answer.lower() in pred.answer.lower() else 0.0,
        'word_overlap': word_overlap_score(example, pred),
        'length_appropriate': length_ratio_score(example, pred),
    }

    # Overall score
    overall = sum(scores.values()) / len(scores)

    if trace is not None:
        return overall >= 0.6

    return overall


# =============================================================================
# 4. Domain-Specific Metrics
# =============================================================================

def sentiment_accuracy(example, pred, trace=None):
    """
    Evaluate sentiment prediction accuracy.

    Expects example.sentiment and pred.sentiment fields.
    """
    expected = getattr(example, 'sentiment', '').lower().strip()
    predicted = getattr(pred, 'sentiment', '').lower().strip()

    # Direct match
    if expected == predicted:
        return 1.0

    # Partial match (e.g., "very positive" matches "positive")
    sentiment_map = {
        'positive': ['positive', 'good', 'great', 'excellent'],
        'negative': ['negative', 'bad', 'poor', 'terrible'],
        'neutral': ['neutral', 'mixed', 'okay', 'average']
    }

    for sentiment_class, variants in sentiment_map.items():
        if expected in variants and predicted in variants:
            return 0.8

    return 0.0


def entity_extraction_f1(example, pred, trace=None):
    """
    Calculate F1 score for entity extraction.

    Expects example.entities and pred.entities as lists.
    """
    expected: Set[str] = set(
        e.lower().strip()
        for e in getattr(example, 'entities', [])
    )
    predicted: Set[str] = set(
        e.lower().strip()
        for e in getattr(pred, 'entities', [])
    )

    if not expected and not predicted:
        return 1.0
    if not expected or not predicted:
        return 0.0

    true_positives = len(expected & predicted)
    precision = true_positives / len(predicted) if predicted else 0
    recall = true_positives / len(expected) if expected else 0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def classification_accuracy(example, pred, trace=None):
    """
    Evaluate classification accuracy with confidence.

    Returns higher scores for confident correct predictions.
    """
    expected_label = getattr(example, 'label', '').lower().strip()
    predicted_label = getattr(pred, 'label', '').lower().strip()
    confidence = getattr(pred, 'confidence', 0.5)

    if expected_label == predicted_label:
        # Reward confident correct predictions
        return 0.5 + (0.5 * confidence)
    else:
        # Penalize confident incorrect predictions
        return 0.0


# =============================================================================
# 5. Trace-Aware Metrics
# =============================================================================

def optimization_aware_metric(example, pred, trace=None):
    """
    Metric that behaves differently during optimization vs evaluation.
    """
    # Calculate base score
    exact = example.answer.lower().strip() == pred.answer.lower().strip()
    contains = example.answer.lower() in pred.answer.lower()

    if trace is not None:
        # OPTIMIZATION MODE: Be strict
        # Only accept examples that are exactly right
        # These will be used as demonstrations
        return exact

    # EVALUATION MODE: Return nuanced score
    if exact:
        return 1.0
    elif contains:
        return 0.7
    else:
        return word_overlap_score(example, pred)


# =============================================================================
# Demonstration
# =============================================================================

def demo_metrics():
    """Demonstrate all metric types."""
    setup_dspy()

    # Create test examples
    qa_example = dspy.Example(
        question="What is the capital of France?",
        answer="Paris",
        key_points=["Paris", "capital", "France"]
    ).with_inputs("question")

    # Mock predictions
    class MockPred:
        def __init__(self, answer, **kwargs):
            self.answer = answer
            for k, v in kwargs.items():
                setattr(self, k, v)

    # Test predictions
    predictions = [
        MockPred("Paris"),
        MockPred("The capital of France is Paris."),
        MockPred("Paris is the capital city of France, known for the Eiffel Tower."),
        MockPred("London"),
        MockPred("I'm not sure, maybe Paris?"),
    ]

    print("\n" + "="*70)
    print("METRIC COMPARISON")
    print("="*70)

    metrics = [
        ("Exact Match", exact_match),
        ("Normalized Match", normalized_match),
        ("Word Overlap", word_overlap_score),
        ("Jaccard Similarity", jaccard_similarity),
        ("Length Ratio", length_ratio_score),
        ("Comprehensive QA", comprehensive_qa_metric),
    ]

    # Header
    print(f"\n{'Prediction':<50} | ", end="")
    for name, _ in metrics:
        print(f"{name[:12]:>12} | ", end="")
    print()
    print("-" * 140)

    # Evaluate each prediction
    for pred in predictions:
        display_answer = pred.answer[:45] + "..." if len(pred.answer) > 45 else pred.answer
        print(f"{display_answer:<50} | ", end="")

        for name, metric_fn in metrics:
            score = metric_fn(qa_example, pred)
            if isinstance(score, bool):
                print(f"{'T' if score else 'F':>12} | ", end="")
            else:
                print(f"{score:>12.2f} | ", end="")
        print()

    # Test trace behavior
    print("\n" + "="*70)
    print("TRACE-AWARE METRIC BEHAVIOR")
    print("="*70)

    test_pred = MockPred("The capital is Paris")

    print(f"\nPrediction: '{test_pred.answer}'")
    print(f"Expected: '{qa_example.answer}'")
    print(f"\nWith trace=None (evaluation): {optimization_aware_metric(qa_example, test_pred, trace=None):.2f}")
    print(f"With trace='something' (optimization): {optimization_aware_metric(qa_example, test_pred, trace='opt')}")


if __name__ == "__main__":
    demo_metrics()

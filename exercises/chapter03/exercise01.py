"""
Exercise 1: Basic Module Usage
Solution for Exercise 1 from Chapter 3

Task: Create a text classifier using dspy.Predict
- Create a signature for text classification
- Initialize a Predict module with examples
- Test the classifier with sample texts
"""

import dspy
from typing import List

def create_text_classifier():
    """Create and configure a text classifier module."""

    # Step 1: Define the signature for text classification
    class TextClassifierSignature(dspy.Signature):
        """Classify text into categories."""
        text = dspy.InputField(desc="Text to classify", type=str)
        category = dspy.OutputField(desc="Text category", type=str)
        confidence = dspy.OutputField(desc="Classification confidence", type=float)

    # Step 2: Create training examples
    training_examples = [
        dspy.Example(
            text="I love this product! It works perfectly.",
            category="positive",
            confidence=0.95
        ),
        dspy.Example(
            text="This is terrible. Worst purchase ever.",
            category="negative",
            confidence=0.90
        ),
        dspy.Example(
            text="It's okay, nothing special.",
            category="neutral",
            confidence=0.70
        ),
        dspy.Example(
            text="Amazing quality and great value!",
            category="positive",
            confidence=0.92
        ),
        dspy.Example(
            text="Poor customer service and slow delivery.",
            category="negative",
            confidence=0.88
        )
    ]

    # Step 3: Initialize the Predict module with examples
    classifier = dspy.Predict(
        TextClassifierSignature,
        demos=training_examples
    )

    return classifier

def test_classifier(classifier, test_texts: List[str]):
    """Test the classifier with sample texts."""

    print("\nTesting Text Classifier:")
    print("-" * 40)

    results = []
    for i, text in enumerate(test_texts, 1):
        try:
            # Classify the text
            result = classifier(text=text)

            print(f"\nTest {i}:")
            print(f"Text: {text}")
            print(f"Category: {result.category}")
            print(f"Confidence: {result.confidence:.2f}")

            results.append({
                'text': text,
                'category': result.category,
                'confidence': result.confidence
            })

        except Exception as e:
            print(f"\nTest {i}: Error processing '{text}'")
            print(f"Error: {e}")
            results.append({
                'text': text,
                'category': 'error',
                'confidence': 0.0,
                'error': str(e)
            })

    return results

def main():
    """Main function to run Exercise 1."""

    print("=" * 60)
    print("Exercise 1: Basic Module Usage")
    print("Creating and testing a text classifier using dspy.Predict")
    print("=" * 60)

    # Create the classifier
    classifier = create_text_classifier()

    # Test texts
    test_texts = [
        "The item arrived on time and works as expected.",
        "Completely disappointed with this purchase.",
        "Average quality, but the price is reasonable.",
        "Outstanding product! Highly recommended!",
        "Not worth the money at all."
    ]

    # Test the classifier
    results = test_classifier(classifier, test_texts)

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"Total texts tested: {len(results)}")

    categories = {}
    for result in results:
        cat = result['category']
        categories[cat] = categories.get(cat, 0) + 1

    print(f"Categories found: {list(categories.keys())}")
    for cat, count in categories.items():
        print(f"  - {cat}: {count}")

    # Calculate average confidence (excluding errors)
    valid_results = [r for r in results if r['category'] != 'error']
    if valid_results:
        avg_confidence = sum(r['confidence'] for r in valid_results) / len(valid_results)
        print(f"\nAverage confidence: {avg_confidence:.2f}")

    print("=" * 60)

if __name__ == "__main__":
    main()
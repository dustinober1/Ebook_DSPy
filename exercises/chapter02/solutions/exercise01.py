"""
Exercise 1 Solutions: Basic Signature Creation

This file contains solutions for Exercise 1 on creating basic DSPy signatures.
"""

# Exercise 1.1: Simple Question Answering
# Solution:
qa_signature = "question, context -> answer"

# Alternative with more context:
qa_signature_detailed = "question, context, question_type -> answer, confidence, sources"

# Exercise 1.2: Text Classification
# Solution:
classification_signature = "text, categories -> classification, confidence"

# More comprehensive version:
classification_signature_detailed = "text, categories, classification_criteria -> primary_category, confidence, category_scores, reasoning"

# Exercise 1.3: Text Transformation
# Solution:
transformation_signature = "informal_text -> formal_text"

# More robust version:
transformation_signature_comprehensive = "informal_text, target_audience, formality_level -> formal_text, transformations_applied, style_score"

# Design Choices Explanation:
design_explanation = """
1. Question Answering Signature:
   - Basic: Just question and context for simple use cases
   - Added question_type to help the model understand what kind of answer is needed
   - Added confidence to indicate answer reliability
   - Added sources to enable verification

2. Text Classification Signature:
   - Categories as input allows flexible classification schemes
   - Confidence score helps with thresholding and decision making
   - Added classification_criteria for nuanced classification
   - Category_scores show confidence across all categories
   - Reasoning provides explainability

3. Text Transformation Signature:
   - Simple version for straightforward transformations
   - Target_audience helps adapt tone and terminology
   - Formality_level allows different degrees of formality
   - Transformations_applied shows what changes were made
   - Style_score evaluates the quality of transformation

Key Principles Applied:
- Start simple, add complexity as needed
- Include metadata (confidence, scores) for better decision making
- Consider the downstream use of outputs
- Add explainability where helpful
"""

# Additional Example Usage:

def demonstrate_basic_signatures():
    """Show how to use these basic signatures with DSPy."""

    import dspy

    # Example 1: Question Answering
    print("=== Question Answering Example ===")
    qa_predictor = dspy.Predict(qa_signature)
    result = qa_predictor(
        question="What is the capital of France?",
        context="European geography and major cities"
    )
    print(f"Answer: {result.answer}")

    # Example 2: Text Classification
    print("\n=== Text Classification Example ===")
    classifier = dspy.Predict(classification_signature)
    result = classifier(
        text="I love this product! It works exactly as advertised.",
        categories="positive, negative, neutral"
    )
    print(f"Classification: {result.classification}")
    print(f"Confidence: {result.confidence}")

    # Example 3: Text Transformation
    print("\n=== Text Transformation Example ===")
    transformer = dspy.Predict(transformation_signature)
    result = transformer(
        informal_text="hey u wanna grab lunch later?"
    )
    print(f"Formal: {result.formal_text}")

if __name__ == "__main__":
    demonstrate_basic_signatures()
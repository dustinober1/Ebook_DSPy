"""
Basic DSPy Signatures Examples

This file demonstrates fundamental DSPy signature concepts including:
- Simple string-based signatures
- Basic input/output patterns
- Common signature types
- Using signatures with DSPy modules
"""

import dspy

# Example 1: Simple Question Answering
def example_simple_qa():
    """Demonstrate a basic QA signature."""

    # Define the signature as a string
    qa_signature = "question, context -> answer"

    # Create a predictor with this signature
    qa_predictor = dspy.Predict(qa_signature)

    # Use the predictor
    result = qa_predictor(
        question="What is the capital of France?",
        context="European geography and capitals"
    )

    print("=== Simple QA Example ===")
    print(f"Question: What is the capital of France?")
    print(f"Answer: {result.answer}")
    print()

# Example 2: Text Classification
def example_text_classification():
    """Demonstrate a text classification signature."""

    # Define classification signature
    classification_signature = "text, categories -> classification, confidence"

    # Create predictor
    classifier = dspy.Predict(classification_signature)

    # Classify a sample
    result = classifier(
        text="I love this product! It works exactly as advertised.",
        categories="positive, negative, neutral"
    )

    print("=== Text Classification Example ===")
    print(f"Text: I love this product! It works exactly as advertised.")
    print(f"Classification: {result.classification}")
    print(f"Confidence: {result.confidence}")
    print()

# Example 3: Text Summarization
def example_summarization():
    """Demonstrate a summarization signature."""

    # Define summarization signature
    summary_signature = "long_document, summary_length -> short_summary, key_points"

    # Create predictor
    summarizer = dspy.Predict(summary_signature)

    # Sample document
    document = """
    Artificial intelligence has revolutionized numerous industries in recent years.
    From healthcare to finance, AI applications are improving efficiency and accuracy.
    Machine learning algorithms can now detect diseases earlier, optimize investment
    portfolios, and even create art. As technology continues to advance, we can expect
    even more innovative applications of AI in our daily lives.
    """

    # Generate summary
    result = summarizer(
        long_document=document,
        summary_length="brief"
    )

    print("=== Summarization Example ===")
    print(f"Original: {document[:100]}...")
    print(f"Summary: {result.short_summary}")
    print(f"Key Points: {result.key_points}")
    print()

# Example 4: Translation
def example_translation():
    """Demonstrate a translation signature."""

    # Define translation signature
    translation_signature = "source_text, source_language, target_language -> translated_text, confidence"

    # Create translator
    translator = dspy.Predict(translation_signature)

    # Translate sample text
    result = translator(
        source_text="Hello, how are you today?",
        source_language="English",
        target_language="Spanish"
    )

    print("=== Translation Example ===")
    print(f"Original: Hello, how are you today?")
    print(f"Translated: {result.translated_text}")
    print(f"Confidence: {result.confidence}")
    print()

# Example 5: Information Extraction
def example_information_extraction():
    """Demonstrate an information extraction signature."""

    # Define extraction signature
    extraction_signature = "document, entities_to_extract -> extracted_entities, relationships"

    # Create extractor
    extractor = dspy.Predict(extraction_signature)

    # Sample document
    document = """
    Apple Inc., headquartered in Cupertino, California, was founded by Steve Jobs in 1976.
    The company went public in 1980 and is now led by CEO Tim Cook.
    """

    # Extract entities
    result = extractor(
        document=document,
        entities_to_extract="companies, people, locations, dates"
    )

    print("=== Information Extraction Example ===")
    print(f"Document: {document}")
    print(f"Extracted Entities: {result.extracted_entities}")
    print(f"Relationships: {result.relationships}")
    print()

# Example 6: Sentiment Analysis
def example_sentiment_analysis():
    """Demonstrate a sentiment analysis signature."""

    # Define sentiment signature
    sentiment_signature = "text -> sentiment, sentiment_score, emotional_indicators"

    # Create sentiment analyzer
    analyzer = dspy.Predict(sentiment_signature)

    # Analyze sentiment
    texts = [
        "This movie was absolutely fantastic!",
        "I'm disappointed with the service quality.",
        "The product works as expected."
    ]

    print("=== Sentiment Analysis Examples ===")
    for text in texts:
        result = analyzer(text=text)
        print(f"Text: {text}")
        print(f"Sentiment: {result.sentiment}")
        print(f"Score: {result.sentiment_score}")
        print(f"Emotions: {result.emotional_indicators}")
        print("-" * 50)

# Example 7: Chained Signatures
def example_chained_signatures():
    """Demonstrate chaining multiple signatures."""

    # Define signatures for a pipeline
    extractor_signature = "document -> key_topics, summary"
    analyzer_signature = "topics, summary -> insights, recommendations"

    # Create modules
    extractor = dspy.Predict(extractor_signature)
    analyzer = dspy.Predict(analyzer_signature)

    # Process document through pipeline
    document = """
    The company's quarterly report shows a 15% increase in revenue compared to last year.
    Customer satisfaction has improved, but operational costs have risen by 8%.
    The market share in the premium segment has grown significantly.
    """

    # Step 1: Extract information
    extraction_result = extractor(document=document)

    # Step 2: Analyze extracted information
    analysis_result = analyzer(
        topics=extraction_result.key_topics,
        summary=extraction_result.summary
    )

    print("=== Chained Signatures Example ===")
    print(f"Document: {document}")
    print(f"Topics: {extraction_result.key_topics}")
    print(f"Summary: {extraction_result.summary}")
    print(f"Insights: {analysis_result.insights}")
    print(f"Recommendations: {analysis_result.recommendations}")
    print()

# Example 8: Signature with Multiple Inputs
def example_multiple_inputs():
    """Demonstrate signatures with multiple complex inputs."""

    # Define signature
    analysis_signature = """
    customer_review, product_info, purchase_history, customer_demographics ->
    sentiment_score, key_complaints, product_suggestions, retention_risk
    """

    # Create analyzer
    analyzer = dspy.Predict(analysis_signature)

    # Complex input data
    result = analyzer(
        customer_review="The product stopped working after just one week of use. Very disappointed.",
        product_info="Wireless headphones, $199, 1-year warranty",
        purchase_history="Previous purchases: 2 similar products, 1 returned",
        customer_demographics="Age: 35, Loyalty: Gold member, 5 years with brand"
    )

    print("=== Multiple Inputs Example ===")
    print(f"Sentiment Score: {result.sentiment_score}")
    print(f"Key Complaints: {result.key_complaints}")
    print(f"Product Suggestions: {result.product_suggestions}")
    print(f"Retention Risk: {result.retention_risk}")
    print()

def run_all_examples():
    """Run all basic signature examples."""

    print("Running Basic DSPy Signature Examples...\n")
    print("=" * 60)

    example_simple_qa()
    example_text_classification()
    example_summarization()
    example_translation()
    example_information_extraction()
    example_sentiment_analysis()
    example_chained_signatures()
    example_multiple_inputs()

    print("=" * 60)
    print("All examples completed!")

if __name__ == "__main__":
    run_all_examples()
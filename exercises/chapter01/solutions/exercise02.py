"""
Exercise 2: Create Custom Signatures
===================================
Solution for creating different types of DSPy signatures for various NLP tasks.
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

# Configure a language model for testing
lm = dspy.OpenAI(model="gpt-4o-mini")
dspy.configure(lm=lm)


class Translate(dspy.Signature):
    """Translate text from English to Spanish."""
    text = dspy.InputField(desc="English text to translate")
    target_language = dspy.InputField(desc="Target language (default: Spanish)")
    translated_text = dspy.OutputField(desc="Text translated to the target language")


class AnalyzeSentiment(dspy.Signature):
    """Analyze the sentiment of a given text."""
    text = dspy.InputField(desc="Text to analyze for sentiment")
    sentiment = dspy.OutputField(desc="Sentiment classification: positive, negative, or neutral")
    confidence = dspy.OutputField(desc="Confidence score from 0.0 to 1.0")


class Summarize(dspy.Signature):
    """Create a brief summary of the given text."""
    text = dspy.InputField(desc="Text to summarize")
    summary = dspy.OutputField(desc="Concise summary of the main points")
    summary_length = dspy.OutputField(desc="Number of sentences in the summary")


class ExtractEntities(dspy.Signature):
    """Extract named entities from the given text."""
    text = dspy.InputField(desc="Text containing entities to extract")
    entities = dspy.OutputField(desc="Named entities (people, places, organizations) found in the text")
    entity_types = dspy.OutputField(desc="Types of entities found (e.g., PERSON, LOCATION, ORGANIZATION)")


def test_signatures():
    """Test each signature with sample inputs."""

    print("Testing Custom Signatures")
    print("=========================\n")

    # Test Translation
    print("1. Testing Translation Signature:")
    translate = dspy.Predict(Translate)
    result = translate(text="Hello, how are you today?", target_language="Spanish")
    print(f"   Input: 'Hello, how are you today?'")
    print(f"   Output: '{result.translated_text}'\n")

    # Test Sentiment Analysis
    print("2. Testing Sentiment Analysis Signature:")
    analyze_sentiment = dspy.Predict(AnalyzeSentiment)
    result = analyze_sentiment(text="I love this product! It works amazingly well.")
    print(f"   Input: 'I love this product! It works amazingly well.'")
    print(f"   Sentiment: {result.sentiment}")
    print(f"   Confidence: {result.confidence}\n")

    # Test Summarization
    print("3. Testing Summarization Signature:")
    summarize = dspy.Predict(Summarize)
    long_text = """
    Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines
    that can perform tasks that typically require human intelligence. These tasks include learning,
    reasoning, problem-solving, perception, and language understanding. AI has made significant advances
    in recent years, with applications in healthcare, finance, transportation, and entertainment.
    Machine learning, a subset of AI, enables systems to learn and improve from experience without
    being explicitly programmed. Deep learning, a further subset, uses neural networks with multiple
    layers to process complex patterns in data.
    """
    result = summarize(text=long_text)
    print(f"   Input: Long text about AI (164 words)")
    print(f"   Summary: '{result.summary}'")
    print(f"   Length: {result.summary_length} sentences\n")

    # Test Entity Extraction
    print("4. Testing Entity Extraction Signature:")
    extract_entities = dspy.Predict(ExtractEntities)
    sample_text = "Apple Inc., headquartered in Cupertino, California, was founded by Steve Jobs and Steve Wozniak in 1976."
    result = extract_entities(text=sample_text)
    print(f"   Input: '{sample_text}'")
    print(f"   Entities: {result.entities}")
    print(f"   Entity Types: {result.entity_types}\n")

    print("All signatures tested successfully!")


def demonstrate_signature_anatomy():
    """Demonstrate the key components of DSPy signatures."""

    print("\n\nSignature Anatomy Guide")
    print("=======================")
    print("\nKey components of a DSPy signature:")
    print("\n1. Class Definition:")
    print("   class SignatureName(dspy.Signature):")
    print("       \"\"\"Docstring explaining the task.\"\"\"")

    print("\n2. Input Fields:")
    print("   field_name = dspy.InputField(desc='Description of what this field represents')")

    print("\n3. Output Fields:")
    print("   field_name = dspy.OutputField(desc='Description of expected output')")

    print("\n4. Best Practices:")
    print("   - Use descriptive field names")
    print("   - Provide clear descriptions for all fields")
    print("   - Include a helpful docstring")
    print("   - Keep signatures focused on single tasks")
    print("   - Consider using ChainOfThought for complex reasoning")


if __name__ == "__main__":
    test_signatures()
    demonstrate_signature_anatomy()
"""
Exercise 5: Multi-Step Classification Pipeline
================================================
Solution for building a multi-step text processing pipeline using DSPy modules.
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

# Configure language model
lm = dspy.OpenAI(model="gpt-4o")
dspy.configure(lm=lm)


# Define signatures for each step of the pipeline
class ExtractTopic(dspy.Signature):
    """Extract the main topic or theme from the given text."""
    text = dspy.InputField(desc="Text to analyze")
    topic = dspy.OutputField(desc="Main topic or theme of the text (1-3 words)")


class ClassifySentiment(dspy.Signature):
    """Classify the sentiment of the given text."""
    text = dspy.InputField(desc="Text to classify")
    sentiment = dspy.OutputField(desc="Sentiment: positive, negative, or neutral")
    reasoning = dspy.OutputField(desc="Brief explanation for the sentiment classification")


class DetermineAudience(dspy.Signature):
    """Determine the intended audience for the text."""
    text = dspy.InputField(desc="Text to analyze")
    audience = dspy.OutputField(desc="Intended audience: general, technical, or academic")
    reasoning = dspy.OutputField(desc="Explanation for the audience determination")


class TailoredSummary(dspy.Signature):
    """Generate a summary tailored to the specific audience."""
    text = dspy.InputField(desc="Original text to summarize")
    audience = dspy.InputField(desc="Target audience for the summary")
    summary = dspy.OutputField(desc="Summary tailored to the specified audience")


class TextAnalysisPipeline(dspy.Module):
    """Multi-step text analysis pipeline that processes text through multiple stages."""

    def __init__(self):
        super().__init__()
        # Initialize modules for each step
        self.extract_topic = dspy.Predict(ExtractTopic)
        self.classify_sentiment = dspy.ChainOfThought(ClassifySentiment)
        self.determine_audience = dspy.ChainOfThought(DetermineAudience)
        self.generate_summary = dspy.ChainOfThought(TailoredSummary)

    def forward(self, text):
        """Process text through the pipeline and return all results."""
        # Step 1: Extract the main topic
        topic_result = self.extract_topic(text=text)
        topic = topic_result.topic

        # Step 2: Classify the sentiment
        sentiment_result = self.classify_sentiment(text=text)
        sentiment = sentiment_result.sentiment
        sentiment_reasoning = sentiment_result.reasoning

        # Step 3: Determine the intended audience
        audience_result = self.determine_audience(text=text)
        audience = audience_result.audience
        audience_reasoning = audience_result.reasoning

        # Step 4: Generate a tailored summary
        summary_result = self.generate_summary(text=text, audience=audience)
        summary = summary_result.summary

        # Return all results in a structured format
        return dspy.Prediction(
            topic=topic,
            sentiment=sentiment,
            sentiment_reasoning=sentiment_reasoning,
            audience=audience,
            audience_reasoning=audience_reasoning,
            summary=summary
        )


def display_pipeline_results(text, results):
    """Display the pipeline results in a formatted way."""

    print(f"\nProcessing: \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    print(f"\n📌 Topic: {results.topic}")
    print(f"\n😊 Sentiment: {results.sentiment.capitalize()}")
    if hasattr(results, 'sentiment_reasoning'):
        print(f"   Reasoning: {results.sentiment_reasoning}")

    print(f"\n👥 Audience: {results.audience.capitalize()}")
    if hasattr(results, 'audience_reasoning'):
        print(f"   Reasoning: {results.audience_reasoning}")

    print(f"\n📝 Summary (for {results.audience} audience):")
    print(f"   {results.summary}")

    print("\n" + "-"*60)


def test_pipeline():
    """Test the pipeline with different types of text."""

    print("\nMulti-Step Text Analysis Pipeline")
    print("==================================\n")

    # Create and initialize the pipeline
    pipeline = TextAnalysisPipeline()

    # Test texts
    test_texts = [
        {
            "text": "Machine learning models require large datasets and computational power. "
                   "Recent advances in transformer architectures have revolutionized NLP tasks. "
                   "These models use attention mechanisms to process sequential data effectively.",
            "description": "Technical ML/NLP content"
        },
        {
            "text": "I absolutely love this new restaurant! The food was amazing and the "
                   "service was excellent. The atmosphere was cozy and welcoming. "
                   "Can't wait to go back again!",
            "description": "Positive personal review"
        },
        {
            "text": "The economic indicators suggest a potential downturn in the housing market. "
                   "Analysts recommend caution in real estate investments. "
                   "Interest rates continue to rise, affecting mortgage affordability.",
            "description": "Economic/financial analysis"
        },
        {
            "text": "Climate change poses significant challenges to biodiversity conservation. "
                   "Research indicates that rising temperatures and changing precipitation patterns "
                   "are affecting ecosystems globally. Immediate action is required to mitigate impacts.",
            "description": "Academic/scientific content"
        },
        {
            "text": "Just finished my morning run! The weather was perfect today - not too hot, "
                   "not too cold. Feeling energized and ready to tackle the day ahead. "
                   "Exercise really does improve your mood!",
            "description": "Casual personal update"
        }
    ]

    # Process each text
    for i, test_case in enumerate(test_texts, 1):
        print(f"\n{'='*70}")
        print(f"TEST CASE {i}: {test_case['description']}")
        print(f"{'='*70}")

        # Run the pipeline
        results = pipeline(text=test_case['text'])

        # Display results
        display_pipeline_results(test_case['text'], results)


def demonstrate_pipeline_architecture():
    """Explain the pipeline architecture and design choices."""

    print("\n\nPipeline Architecture")
    print("=====================")
    print("\n🔄 Multi-Step Processing:")
    print("   1. Topic Extraction → Identifies the main theme")
    print("   2. Sentiment Analysis → Determines emotional tone")
    print("   3. Audience Determination → Identifies target readers")
    print("   4. Summary Generation → Creates tailored summary")

    print("\n🛠️ DSPy Modules Used:")
    print("   • dspy.Predict: For simple, direct predictions (topic extraction)")
    print("   • dspy.ChainOfThought: For reasoning-based tasks")

    print("\n📊 Design Decisions:")
    print("   • Modular design: Each step is independent and reusable")
    print("   • Chained outputs: Results flow between steps naturally")
    print("   • Mixed modules: Choose the right tool for each task")
    print("   • Structured output: All results returned together")

    print("\n🎯 Benefits:")
    print("   • Clear separation of concerns")
    print("   • Easy to modify individual steps")
    print("   • Can add new steps without changing existing ones")
    print("   • Results are comprehensive and interconnected")


if __name__ == "__main__":
    test_pipeline()
    demonstrate_pipeline_architecture()
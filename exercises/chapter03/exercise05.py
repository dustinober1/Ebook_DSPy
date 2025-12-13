"""
Exercise 5: Module Composition
Solution for Exercise 5 from Chapter 3

Task: Build a pipeline that combines multiple modules
- Create a text processing pipeline
- Chain multiple modules together
- Handle data flow between modules
"""

import dspy
from typing import List, Dict, Any, Optional
from datetime import datetime

class TextProcessingPipeline(dspy.Module):
    """A pipeline that processes text through multiple stages."""

    def __init__(self):
        super().__init__()

        # Stage 1: Preprocessing
        self.preprocessor = dspy.Predict(
            "raw_text -> cleaned_text"
        )

        # Stage 2: Analysis modules (parallel)
        self.sentiment_analyzer = dspy.Predict(
            "text -> sentiment confidence"
        )
        self.topic_classifier = dspy.Predict(
            "text -> primary_topic secondary_topics"
        )
        self.entity_extractor = dspy.Predict(
            "text -> people organizations locations"
        )

        # Stage 3: Summarization
        self.summarizer = dspy.ChainOfThought(
            "text sentiment topic -> summary"
        )

        # Stage 4: Quality assessment
        self.quality_assessor = dspy.Predict(
            "text summary -> quality_score issues"
        )

    def forward(self, raw_text: str) -> dspy.Prediction:
        """Process text through the complete pipeline."""

        # Initialize pipeline state
        pipeline_state = {
            "input_text": raw_text,
            "timestamp": datetime.now().isoformat(),
            "stage_results": {}
        }

        try:
            # Stage 1: Preprocessing
            print("Stage 1: Preprocessing...")
            preprocessed = self.preprocessor(raw_text=raw_text)
            pipeline_state["stage_results"]["preprocessing"] = {
                "cleaned_text": preprocessed.cleaned_text,
                "word_count": len(preprocessed.cleaned_text.split()),
                "char_count": len(preprocessed.cleaned_text)
            }

            # Stage 2: Parallel Analysis
            print("Stage 2: Running parallel analysis...")
            text_for_analysis = preprocessed.cleaned_text

            # Run all analyses
            sentiment = self.sentiment_analyzer(text=text_for_analysis)
            topics = self.topic_classifier(text=text_for_analysis)
            entities = self.entity_extractor(text=text_for_analysis)

            pipeline_state["stage_results"]["analysis"] = {
                "sentiment": {
                    "label": sentiment.sentiment,
                    "confidence": float(sentiment.confidence) if sentiment.confidence else 0.0
                },
                "topics": {
                    "primary": topics.primary_topic,
                    "secondary": topics.secondary_topics.split(", ") if topics.secondary_topics else []
                },
                "entities": {
                    "people": entities.people.split(", ") if entities.people else [],
                    "organizations": entities.organizations.split(", ") if entities.organizations else [],
                    "locations": entities.locations.split(", ") if entities.locations else []
                }
            }

            # Stage 3: Context-aware Summarization
            print("Stage 3: Generating summary...")
            summary_result = self.summarizer(
                text=text_for_analysis,
                sentiment=sentiment.sentiment,
                topic=topics.primary_topic
            )

            pipeline_state["stage_results"]["summarization"] = {
                "summary": summary_result.summary,
                "reasoning": summary_result.reasoning if hasattr(summary_result, 'reasoning') else "Generated summary"
            }

            # Stage 4: Quality Assessment
            print("Stage 4: Assessing quality...")
            quality = self.quality_assessor(
                text=text_for_analysis,
                summary=summary_result.summary
            )

            pipeline_state["stage_results"]["quality"] = {
                "score": float(quality.quality_score) if quality.quality_score else 0.0,
                "issues": quality.issues.split(", ") if quality.issues else []
            }

            # Compile final result
            return dspy.Prediction(
                original_text=raw_text,
                cleaned_text=preprocessed.cleaned_text,
                sentiment=sentiment.sentiment,
                confidence=float(sentiment.confidence) if sentiment.confidence else 0.0,
                primary_topic=topics.primary_topic,
                secondary_topics=topics.secondary_topics.split(", ") if topics.secondary_topics else [],
                entities=pipeline_state["stage_results"]["analysis"]["entities"],
                summary=summary_result.summary,
                quality_score=float(quality.quality_score) if quality.quality_score else 0.0,
                quality_issues=quality.issues.split(", ") if quality.issues else [],
                processing_stages=list(pipeline_state["stage_results"].keys()),
                success=True
            )

        except Exception as e:
            return dspy.Prediction(
                original_text=raw_text,
                error=str(e),
                success=False,
                completed_stages=list(pipeline_state["stage_results"].keys())
            )

class AdvancedPipeline(dspy.Module):
    """An advanced pipeline with conditional routing."""

    def __init__(self):
        super().__init__()

        # Route classifier
        self.router = dspy.Predict(
            "text -> route"
        )

        # Specialized processors for different routes
        self.email_processor = dspy.Predict(
            "email -> category urgency action_items"
        )
        self.news_processor = dspy.Predict(
            "news_article -> headline summary key_points sentiment"
        )
        self.review_processor = dspy.Predict(
            "review -> rating pros cons recommendation"
        )

        # Common post-processor
        self.post_processor = dspy.Predict(
            "processed_data route_type -> final_output"
        )

    def forward(self, text: str, route_hint: Optional[str] = None) -> dspy.Prediction:
        """Process text with conditional routing."""

        # Determine route
        if route_hint:
            route = route_hint
        else:
            routing_result = self.router(text=text)
            route = routing_result.route.lower()

        # Route to appropriate processor
        try:
            if "email" in route or "@" in text:
                # Email processing route
                result = self.email_processor(email=text)
                processed_data = {
                    "type": "email",
                    "category": result.category,
                    "urgency": result.urgency,
                    "action_items": result.action_items.split(", ") if result.action_items else []
                }

            elif "news" in route or any(word in text.lower() for word in ["breaking", "report", "announced"]):
                # News processing route
                result = self.news_processor(news_article=text)
                processed_data = {
                    "type": "news",
                    "headline": result.headline,
                    "summary": result.summary,
                    "key_points": result.key_points.split("\n") if result.key_points else [],
                    "sentiment": result.sentiment
                }

            elif "review" in route or any(word in text.lower() for word in ["rating", "recommend", "poor", "excellent"]):
                # Review processing route
                result = self.review_processor(review=text)
                processed_data = {
                    "type": "review",
                    "rating": result.rating,
                    "pros": result.pros.split(", ") if result.pros else [],
                    "cons": result.cons.split(", ") if result.cons else [],
                    "recommendation": result.recommendation
                }

            else:
                # Default processing
                processed_data = {
                    "type": "general",
                    "message": "Text processed with default pipeline",
                    "route_used": route
                }

            # Post-process the result
            final_result = self.post_processor(
                processed_data=str(processed_data),
                route_type=processed_data.get("type", "unknown")
            )

            return dspy.Prediction(
                original_text=text,
                route_taken=route,
                processed_data=processed_data,
                final_output=final_result.final_output,
                success=True
            )

        except Exception as e:
            return dspy.Prediction(
                original_text=text,
                error=str(e),
                success=False
            )

def test_text_pipeline():
    """Test the basic text processing pipeline."""

    pipeline = TextProcessingPipeline()

    # Test texts
    test_texts = [
        """
        Apple Inc. announced today that they will be opening a new research facility
        in Austin, Texas. Tim Cook, the CEO, stated that this decision reflects
        the company's commitment to innovation and creating jobs in the United States.
        The facility will focus on artificial intelligence and machine learning research.
        This is excellent news for the tech industry and local economy!
        """,
        """
        I'm very disappointed with this product. The quality is poor and it broke
        after just one week of use. Customer service was unhelpful and rude.
        I would not recommend this to anyone. Complete waste of money.
        """,
        """
        Climate change is one of the most pressing challenges of our time.
        Scientists from around the world, including Dr. Jane Smith from Stanford
        and Professor Zhang Liu from Beijing University, have published a comprehensive
        report on rising global temperatures. The findings are concerning.
        """
    ]

    print("=" * 60)
    print("Testing Text Processing Pipeline")
    print("=" * 60)

    for i, text in enumerate(test_texts, 1):
        print(f"\nTest Text {i}:")
        print("-" * 40)

        result = pipeline(raw_text=text)

        if result.success:
            print(f"Cleaned Text Preview: {result.cleaned_text[:100]}...")
            print(f"Sentiment: {result.sentiment} (Confidence: {result.confidence:.2f})")
            print(f"Primary Topic: {result.primary_topic}")
            print(f"Quality Score: {result.quality_score:.2f}")

            if result.entities["people"]:
                print(f"People: {', '.join(result.entities['people'][:2])}")
            if result.entities["organizations"]:
                print(f"Organizations: {', '.join(result.entities['organizations'])}")
            if result.entities["locations"]:
                print(f"Locations: {', '.join(result.entities['locations'])}")

            print(f"\nSummary: {result.summary[:150]}...")
        else:
            print(f"Error: {result.error}")

        print("-" * 20)

def test_advanced_pipeline():
    """Test the advanced pipeline with conditional routing."""

    pipeline = AdvancedPipeline()

    # Test texts for different routes
    test_cases = [
        {
            "text": "Subject: Urgent Meeting Tomorrow\n\nHi team, just a reminder about tomorrow's meeting at 2 PM. Please review the attached documents and prepare your updates. Action items: John to present Q3 results, Sarah to discuss the new project timeline.",
            "hint": "email"
        },
        {
            "text": "BREAKING: Tech Giant Announces Record Profits\n\nSilicon Valley, CA - In a surprise announcement today, leading tech company exceeded expectations with quarterly revenue of $50 billion, driven by strong cloud services and AI product adoption. The stock rose 15% in after-hours trading.",
            "hint": "news"
        },
        {
            "text": "Product Rating: 2/5\n\nPros: Nice design, affordable price\nCons: Poor build quality, battery life terrible, customer service unresponsive\nRecommendation: Look elsewhere - this product doesn't deliver on its promises.",
            "hint": "review"
        }
    ]

    print("\n" + "=" * 60)
    print("Testing Advanced Pipeline with Conditional Routing")
    print("=" * 60)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i} ({test_case['hint']}):")
        print("-" * 40)

        result = pipeline(text=test_case["text"], route_hint=test_case["hint"])

        if result.success:
            print(f"Route Taken: {result.route_taken}")
            print(f"Processed Type: {result.processed_data.get('type', 'unknown')}")

            # Show type-specific results
            if result.processed_data.get("type") == "email":
                print(f"Category: {result.processed_data.get('category')}")
                print(f"Urgency: {result.processed_data.get('urgency')}")
                if result.processed_data.get("action_items"):
                    print(f"Action Items: {', '.join(result.processed_data['action_items'][:2])}")

            elif result.processed_data.get("type") == "news":
                print(f"Headline: {result.processed_data.get('headline')}")
                print(f"Sentiment: {result.processed_data.get('sentiment')}")

            elif result.processed_data.get("type") == "review":
                print(f"Rating: {result.processed_data.get('rating')}")
                print(f"Recommendation: {result.processed_data.get('recommendation')}")

            print(f"\nFinal Output: {result.final_output[:100]}...")
        else:
            print(f"Error: {result.error}")

        print("-" * 20)

def main():
    """Main function to run Exercise 5."""

    print("\n" + "=" * 60)
    print("Exercise 5: Module Composition")
    print("Building pipelines that combine multiple DSPy modules")
    print("=" * 60)

    # Test basic pipeline
    print("\n1. Basic Text Processing Pipeline:")
    test_text_pipeline()

    # Test advanced pipeline
    print("\n\n2. Advanced Pipeline with Conditional Routing:")
    test_advanced_pipeline()

    print("\n" + "=" * 60)
    print("Exercise 5 Completed Successfully!")
    print("Demonstrated module composition and data flow between stages.")
    print("=" * 60)

if __name__ == "__main__":
    main()
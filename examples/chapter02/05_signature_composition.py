"""
Signature Composition and Chaining Examples

This file demonstrates how to compose and chain DSPy signatures to build
complex workflows from simple, reusable components.
"""

import dspy
from typing import List, Dict, Optional, Union, Any
import json

# Example 1: Building Blocks - Simple, reusable signatures

class TextCleaner(dspy.Signature):
    """Clean and preprocess text data."""

    raw_text = dspy.InputField(
        desc="Raw, unprocessed text",
        type=str,
        prefix="📝 Raw Text:\n"
    )

    cleaning_options = dspy.InputField(
        desc="Cleaning options and preferences",
        type=Dict[str, Union[bool, str]],
        prefix="⚙️ Cleaning Options:\n"
    )

    cleaned_text = dspy.OutputField(
        desc="Cleaned and preprocessed text",
        type=str,
        prefix="✨ Cleaned Text:\n"
    )

    cleaning_summary = dspy.OutputField(
        desc="Summary of cleaning operations performed",
        type=Dict[str, Union[int, List[str]]],
        prefix="📋 Cleaning Summary:\n"
    )

class EntityExtractor(dspy.Signature):
    """Extract named entities from text."""

    text = dspy.InputField(
        desc="Text to extract entities from",
        type=str,
        prefix="📄 Text:\n"
    )

    entity_types = dspy.InputField(
        desc="Types of entities to extract",
        type=List[str],
        prefix="🏷️ Entity Types:\n"
    )

    entities = dspy.OutputField(
        desc="Extracted entities with positions and types",
        type=List[Dict[str, Union[str, int]]],
        prefix="🏷️ Extracted Entities:\n"
    )

    entity_count = dspy.OutputField(
        desc="Count of each entity type found",
        type=Dict[str, int],
        prefix="📊 Entity Count:\n"
    )

class SentimentAnalyzer(dspy.Signature):
    """Analyze sentiment in text."""

    text = dspy.InputField(
        desc="Text to analyze",
        type=str,
        prefix="📄 Text:\n"
    )

    analysis_granularity = dspy.InputField(
        desc="Level of analysis (document, sentence, aspect)",
        type=str,
        default="document",
        prefix="🔍 Granularity: "
    )

    sentiment = dspy.OutputField(
        desc="Overall sentiment classification",
        type=str,
        prefix="😊 Sentiment: "
    )

    confidence = dspy.OutputField(
        desc="Confidence in sentiment classification",
        type=float,
        prefix="🎯 Confidence: "
    )

    emotional_indicators = dspy.OutputField(
        desc="Specific emotions detected",
        type=List[str],
        prefix="💭 Emotions: "
    )

class TopicClassifier(dspy.Signature):
    """Classify text into predefined topics."""

    text = dspy.InputField(
        desc="Text to classify",
        type=str,
        prefix="📄 Text:\n"
    )

    topic_taxonomy = dspy.InputField(
        desc="Available topics and their descriptions",
        type=Dict[str, str],
        prefix="🗂️ Topic Taxonomy:\n"
    )

    primary_topic = dspy.OutputField(
        desc="Primary topic classification",
        type=str,
        prefix="🎯 Primary Topic: "
    )

    topic_confidence = dspy.OutputField(
        desc="Confidence in topic classification",
        type=float,
        prefix="📊 Confidence: "
    )

    related_topics = dspy.OutputField(
        desc="Secondary topics with scores",
        type=List[Dict[str, Union[str, float]]],
        prefix="🔗 Related Topics:\n"
    )

class Summarizer(dspy.Signature):
    """Generate summaries of text."""

    text = dspy.InputField(
        desc="Text to summarize",
        type=str,
        prefix="📄 Text:\n"
    )

    summary_length = dspy.InputField(
        desc="Desired summary length (brief, medium, detailed)",
        type=str,
        default="medium",
        prefix="📏 Length: "
    )

    summary_type = dspy.InputField(
        desc="Type of summary (extractive, abstractive, hybrid)",
        type=str,
        default="abstractive",
        prefix="🔧 Type: "
    )

    summary = dspy.OutputField(
        desc="Generated summary",
        type=str,
        prefix="📝 Summary:\n"
    )

    key_points = dspy.OutputField(
        desc="Key points extracted from text",
        type=List[str],
        prefix="💡 Key Points:\n"
    )

    compression_ratio = dspy.OutputField(
        desc="Ratio of original to summary length",
        type=float,
        prefix="📊 Compression: "
    )

# Example 2: Composed Signatures - Combining building blocks

class DocumentAnalyzer(dspy.Signature):
    """Comprehensive document analysis using multiple components."""

    document_text = dspy.InputField(
        desc="Full document text to analyze",
        type=str,
        prefix="📚 Document:\n"
    )

    analysis_scope = dspy.InputField(
        desc="Scope of analysis to perform",
        type=List[str],
        prefix="🎯 Analysis Scope:\n"
    )

    document_metadata = dspy.InputField(
        desc="Metadata about the document",
        type=Dict[str, Union[str, int]],
        optional=True,
        prefix="📋 Metadata:\n"
    )

    # Component analysis results
    text_quality = dspy.OutputField(
        desc="Text quality and preprocessing results",
        type=Dict[str, Union[str, Dict[str, Any]]],
        prefix="✅ Text Quality:\n"
    )

    extracted_entities = dspy.OutputField(
        desc="Named entities found in document",
        type=Dict[str, Union[List[Dict[str, Union[str, int]]], Dict[str, int]]],
        prefix="🏷️ Entities:\n"
    )

    sentiment_analysis = dspy.OutputField(
        desc="Sentiment analysis results",
        type=Dict[str, Union[str, float, List[str]]],
        prefix="😊 Sentiment:\n"
    )

    topic_classification = dspy.OutputField(
        desc="Topic classification results",
        type=Dict[str, Union[str, float, List[Dict[str, Union[str, float]]]]],
        prefix="🗂️ Topics:\n"
    )

    document_summary = dspy.OutputField(
        desc="Document summary and key points",
        type=Dict[str, Union[str, List[str], float]],
        prefix="📝 Summary:\n"
    )

    overall_insights = dspy.OutputField(
        desc="Combined insights from all analyses",
        type=Dict[str, Union[str, List[str], Dict[str, Any]]],
        prefix="💡 Overall Insights:\n"
    )

# Example 3: Sequential Processing Pipeline

class SequentialProcessor:
    """Processor that executes signatures in sequence."""

    def __init__(self):
        # Initialize all component modules
        self.cleaner = dspy.Predict(TextCleaner)
        self.entity_extractor = dspy.Predict(EntityExtractor)
        self.sentiment_analyzer = dspy.Predict(SentimentAnalyzer)
        self.topic_classifier = dspy.Predict(TopicClassifier)
        self.summarizer = dspy.Predict(Summarizer)
        self.document_analyzer = dspy.Predict(DocumentAnalyzer)

    def process_document(self, document_text: str, analysis_scope: List[str]) -> Dict[str, Any]:
        """Process document through sequential pipeline."""

        results = {}

        # Step 1: Clean text
        if "cleaning" in analysis_scope:
            clean_result = self.cleaner(
                raw_text=document_text,
                cleaning_options={"remove_html": True, "normalize_whitespace": True}
            )
            cleaned_text = clean_result.cleaned_text
            results["cleaning"] = clean_result.cleaning_summary
        else:
            cleaned_text = document_text

        # Step 2: Extract entities
        if "entities" in analysis_scope:
            entity_result = self.entity_extractor(
                text=cleaned_text,
                entity_types=["person", "organization", "location", "date"]
            )
            results["entities"] = {
                "extracted": entity_result.entities,
                "counts": entity_result.entity_count
            }

        # Step 3: Analyze sentiment
        if "sentiment" in analysis_scope:
            sentiment_result = self.sentiment_analyzer(
                text=cleaned_text,
                analysis_granularity="document"
            )
            results["sentiment"] = {
                "overall": sentiment_result.sentiment,
                "confidence": sentiment_result.confidence,
                "emotions": sentiment_result.emotional_indicators
            }

        # Step 4: Classify topics
        if "topics" in analysis_scope:
            topics = {
                "business": "Business and finance topics",
                "technology": "Technology and innovation",
                "health": "Health and medicine",
                "politics": "Politics and government",
                "sports": "Sports and athletics"
            }
            topic_result = self.topic_classifier(
                text=cleaned_text,
                topic_taxonomy=topics
            )
            results["topics"] = {
                "primary": topic_result.primary_topic,
                "confidence": topic_result.topic_confidence,
                "related": topic_result.related_topics
            }

        # Step 5: Generate summary
        if "summary" in analysis_scope:
            summary_result = self.summarizer(
                text=cleaned_text,
                summary_length="medium"
            )
            results["summary"] = {
                "text": summary_result.summary,
                "key_points": summary_result.key_points,
                "compression": summary_result.compression_ratio
            }

        return results

# Example 4: Parallel Processing

class ParallelProcessor:
    """Processor that can run independent analyses in parallel."""

    def __init__(self):
        # Initialize modules
        self.entity_extractor = dspy.Predict(EntityExtractor)
        self.sentiment_analyzer = dspy.Predict(SentimentAnalyzer)
        self.topic_classifier = dspy.Predict(TopicClassifier)

    def process_parallel(self, text: str) -> Dict[str, Any]:
        """Process text with parallel independent analyses."""

        # These could be run in parallel in a real implementation
        # For demonstration, we run them sequentially

        # Parallel task 1: Entity extraction
        entity_result = self.entity_extractor(
            text=text,
            entity_types=["person", "organization", "location"]
        )

        # Parallel task 2: Sentiment analysis
        sentiment_result = self.sentiment_analyzer(
            text=text,
            analysis_granularity="document"
        )

        # Parallel task 3: Topic classification
        topics = {
            "positive": "Positive and uplifting content",
            "negative": "Negative and critical content",
            "neutral": "Neutral and factual content"
        }
        topic_result = self.topic_classifier(
            text=text,
            topic_taxonomy=topics
        )

        # Combine results
        return {
            "entities": {
                "found": entity_result.entities,
                "count": len(entity_result.entities)
            },
            "sentiment": {
                "classification": sentiment_result.sentiment,
                "confidence": sentiment_result.confidence,
                "emotions": sentiment_result.emotional_indicators
            },
            "topics": {
                "primary": topic_result.primary_topic,
                "confidence": topic_result.topic_confidence
            },
            "processing_metadata": {
                "parallel_tasks": 3,
                "processing_time": "optimized for parallel execution"
            }
        }

# Example 5: Conditional Processing

class ConditionalProcessor:
    """Processor that adapts workflow based on input characteristics."""

    def __init__(self):
        self.entity_extractor = dspy.Predict(EntityExtractor)
        self.sentiment_analyzer = dspy.Predict(SentimentAnalyzer)
        self.summarizer = dspy.Predict(Summarizer)

    def analyze_conditionally(self, text: str) -> Dict[str, Any]:
        """Analyze text with conditional processing logic."""

        results = {"text_length": len(text), "processing_steps": []}

        # Conditional logic based on text length
        if len(text) < 100:
            # Short text: quick analysis
            results["processing_steps"].append("quick_analysis")
            results["is_short"] = True
        elif len(text) < 1000:
            # Medium text: standard analysis
            results["processing_steps"].append("standard_analysis")
            results["is_short"] = False

            # Always do sentiment for medium texts
            sentiment_result = self.sentiment_analyzer(text=text)
            results["sentiment"] = sentiment_result.sentiment
        else:
            # Long text: comprehensive analysis
            results["processing_steps"].append("comprehensive_analysis")
            results["is_short"] = False

            # Extract entities from long texts
            entity_result = self.entity_extractor(
                text=text,
                entity_types=["person", "organization", "location", "date"]
            )
            results["entities"] = entity_result.entities

            # Generate summary for long texts
            summary_result = self.summarizer(
                text=text,
                summary_length="brief"
            )
            results["summary"] = summary_result.summary

        # Conditional sentiment analysis
        if "!" in text or "?" in text or any(word in text.lower() for word in ["angry", "happy", "sad", "excited"]):
            results["processing_steps"].append("sentiment_analysis")
            sentiment_result = self.sentiment_analyzer(text=text)
            results["sentiment"] = {
                "classification": sentiment_result.sentiment,
                "confidence": sentiment_result.confidence
            }

        # Conditional entity extraction
        if any(keyword in text.lower() for keyword in ["mr.", "mrs.", "dr.", "inc.", "ltd.", "corporation"]):
            if "entity_extraction" not in results["processing_steps"]:
                results["processing_steps"].append("entity_extraction")
                entity_result = self.entity_extractor(
                    text=text,
                    entity_types=["person", "organization"]
                )
                results["quick_entities"] = entity_result.entities[:5]  # Top 5

        return results

# Example 6: Hierarchical Composition

class HierarchicalAnalyzer:
    """Demonstrates hierarchical composition of signatures."""

    def __init__(self):
        # Initialize base processors
        self.sequential = SequentialProcessor()
        self.parallel = ParallelProcessor()
        self.conditional = ConditionalProcessor()

        # Initialize high-level analyzer
        self.document_analyzer = dspy.Predict(DocumentAnalyzer)

    def analyze_at_scale(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze multiple documents with appropriate strategies."""

        results = {
            "total_documents": len(documents),
            "processing_strategies": {},
            "aggregate_insights": {},
            "document_results": []
        }

        for doc in documents:
            doc_result = {
                "id": doc.get("id", "unknown"),
                "title": doc.get("title", "Untitled"),
                "text": doc.get("text", ""),
                "strategy_used": None,
                "analysis": None
            }

            # Choose processing strategy based on document characteristics
            text_length = len(doc_result["text"])

            if text_length < 200:
                # Short documents: conditional processing
                doc_result["strategy_used"] = "conditional"
                doc_result["analysis"] = self.conditional.analyze_conditionally(doc_result["text"])
            elif text_length < 2000:
                # Medium documents: parallel processing
                doc_result["strategy_used"] = "parallel"
                doc_result["analysis"] = self.parallel.process_parallel(doc_result["text"])
            else:
                # Long documents: sequential comprehensive processing
                doc_result["strategy_used"] = "sequential"
                doc_result["analysis"] = self.sequential.process_document(
                    doc_result["text"],
                    ["entities", "sentiment", "topics", "summary"]
                )

            results["document_results"].append(doc_result)

        # Aggregate insights across all documents
        all_sentiments = []
        all_topics = []
        total_entities = 0

        for doc_result in results["document_results"]:
            analysis = doc_result["analysis"]

            # Collect sentiment
            if "sentiment" in analysis:
                if isinstance(analysis["sentiment"], dict) and "classification" in analysis["sentiment"]:
                    all_sentiments.append(analysis["sentiment"]["classification"])
                elif isinstance(analysis["sentiment"], str):
                    all_sentiments.append(analysis["sentiment"])

            # Collect topics
            if "topics" in analysis and "primary" in analysis["topics"]:
                all_topics.append(analysis["topics"]["primary"])

            # Count entities
            if "entities" in analysis:
                if isinstance(analysis["entities"], dict) and "count" in analysis["entities"]:
                    total_entities += analysis["entities"]["count"]
                elif isinstance(analysis["entities"], list):
                    total_entities += len(analysis["entities"])

        # Calculate aggregates
        results["aggregate_insights"] = {
            "sentiment_distribution": {
                sentiment: all_sentiments.count(sentiment) / len(all_sentiments)
                for sentiment in set(all_sentiments)
            } if all_sentiments else {},
            "topic_distribution": {
                topic: all_topics.count(topic) / len(all_topics)
                for topic in set(all_topics)
            } if all_topics else {},
            "total_entities_extracted": total_entities,
            "average_entities_per_doc": total_entities / len(documents) if documents else 0
        }

        results["processing_strategies"] = {
            strategy: len([d for d in results["document_results"] if d["strategy_used"] == strategy])
            for strategy in set(d["strategy_used"] for d in results["document_results"])
        }

        return results

def demonstrate_composition():
    """Demonstrate various signature composition patterns."""

    print("DSPy Signature Composition Examples\n")
    print("=" * 60)

    # Initialize processors
    sequential = SequentialProcessor()
    parallel = ParallelProcessor()
    conditional = ConditionalProcessor()
    hierarchical = HierarchicalAnalyzer()

    # Example 1: Sequential Processing
    print("\n1. Sequential Processing")
    print("-" * 40)
    document = """
    Apple Inc. announced record Q4 earnings today, exceeding analyst expectations.
    The company reported revenue of $94.9 billion, up 8% from last year.
    CEO Tim Cook expressed optimism about the company's future growth prospects.
    """

    seq_result = sequential.process_document(
        document,
        ["cleaning", "entities", "sentiment", "topics", "summary"]
    )
    print(f"Extracted {seq_result['entities']['counts'].get('total', 0)} entities")
    print(f"Sentiment: {seq_result['sentiment']['overall']}")
    print(f"Primary Topic: {seq_result['topics']['primary']}")

    # Example 2: Parallel Processing
    print("\n2. Parallel Processing")
    print("-" * 40)
    para_result = parallel.process_parallel(
        "The customer service was excellent! They resolved my issue quickly and professionally."
    )
    print(f"Entities Found: {para_result['entities']['count']}")
    print(f"Sentiment: {para_result['sentiment']['classification']}")
    print(f"Parallel Tasks: {para_result['processing_metadata']['parallel_tasks']}")

    # Example 3: Conditional Processing
    print("\n3. Conditional Processing")
    print("-" * 40)
    short_text = "I love this product!"
    long_text = """
    Microsoft Corporation today announced a strategic partnership with OpenAI
    to accelerate the development of artificial intelligence technologies.
    The multi-billion dollar investment will focus on advancing AI research
    and developing safe, beneficial AI applications for enterprise customers.
    Satya Nadella, CEO of Microsoft, emphasized the importance of responsible
    AI development and the potential to transform industries.
    """

    cond_short = conditional.analyze_conditionally(short_text)
    cond_long = conditional.analyze_conditionally(long_text)

    print(f"Short Text Steps: {cond_short['processing_steps']}")
    print(f"Long Text Steps: {cond_long['processing_steps']}")
    print(f"Long Text Summary Length: {len(cond_long.get('summary', ''))}")

    # Example 4: Hierarchical Composition
    print("\n4. Hierarchical Composition")
    print("-" * 40)
    documents = [
        {
            "id": 1,
            "title": "Tech News",
            "text": "Google releases new AI model with impressive capabilities."
        },
        {
            "id": 2,
            "title": "Financial Report",
            "text": "Q3 earnings exceeded expectations with 15% growth year-over-year."
        },
        {
            "id": 3,
            "title": "Customer Review",
            "text": "This is absolutely terrible! Worst product ever!"
        }
    ]

    hier_result = hierarchical.analyze_at_scale(documents)
    print(f"Processed {hier_result['total_documents']} documents")
    print(f"Strategies Used: {hier_result['processing_strategies']}")
    print(f"Sentiment Distribution: {hier_result['aggregate_insights']['sentiment_distribution']}")

    print("\n" + "=" * 60)
    print("Composition examples completed!")

    # Key takeaways
    print("\n💡 Key Composition Patterns:")
    print("• Sequential: Step-by-step processing pipeline")
    print("• Parallel: Independent analyses running simultaneously")
    print("• Conditional: Adaptive processing based on input")
    print("• Hierarchical: Multi-level composition for complex systems")

if __name__ == "__main__":
    demonstrate_composition()
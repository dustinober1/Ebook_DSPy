"""
Exercise 7: Complete Project
Solution for Exercise 7 from Chapter 3

Task: Build a complete customer feedback analysis system
- Multiple analysis stages
- Custom modules
- Composition patterns
- Performance optimization
"""

import dspy
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

class FeedbackPreprocessor(dspy.Module):
    """Preprocess customer feedback text."""

    def __init__(self):
        super().__init__()
        self.cleaner = dspy.Predict(
            "raw_text -> cleaned_text language"
        )

    def forward(self, feedback: str) -> dspy.Prediction:
        """Clean and standardize feedback text."""

        # Basic text cleaning
        cleaned = re.sub(r'\s+', ' ', feedback.strip())
        cleaned = re.sub(r'[^\w\s.!?]', '', cleaned)

        # Use LLM for advanced cleaning
        result = self.cleaner(raw_text=cleaned)

        # Add metadata
        word_count = len(result.cleaned_text.split())
        char_count = len(result.cleaned_text)

        return dspy.Prediction(
            original_text=feedback,
            cleaned_text=result.cleaned_text,
            language=result.language,
            word_count=word_count,
            char_count=char_count,
            has_emoji=bool(re.search(r'[\U0001F600-\U0001F64F]', feedback))
        )

class SentimentAnalyzer(dspy.Module):
    """Analyze sentiment with detailed scoring."""

    def __init__(self):
        super().__init__()
        self.analyzer = dspy.ChainOfThought(
            "text -> sentiment emotion_intensity"
        )

    def forward(self, text: str) -> dspy.Prediction:
        """Analyze sentiment and emotional intensity."""

        result = self.analyzer(text=text)

        # Parse sentiment score
        sentiment_map = {
            "positive": 0.8,
            "negative": 0.2,
            "neutral": 0.5
        }
        base_score = sentiment_map.get(result.sentiment.lower(), 0.5)

        # Adjust based on intensity
        try:
            intensity = float(result.emotion_intensity) / 10
            final_score = base_score * (0.5 + intensity)
        except:
            final_score = base_score

        return dspy.Prediction(
            sentiment=result.sentiment,
            sentiment_score=round(min(1.0, final_score), 2),
            intensity=result.emotion_intensity
        )

class TopicClassifier(dspy.Module):
    """Classify feedback into topics."""

    def __init__(self):
        super().__init__()
        self.classifier = dspy.Predict(
            "text -> primary_topic secondary_topics"
        )

    def forward(self, text: str) -> dspy.Prediction:
        """Classify feedback into topic categories."""

        result = self.classifier(text=text)

        # Parse topics
        primary = result.primary_topic.strip()
        secondary = [t.strip() for t in result.secondary_topics.split(",") if t.strip()] if result.secondary_topics else []

        return dspy.Prediction(
            primary_topic=primary,
            secondary_topics=secondary,
            topic_count=1 + len(secondary)
        )

class IssueExtractor(dspy.Module):
    """Extract specific issues mentioned in feedback."""

    def __init__(self):
        super().__init__()
        self.extractor = dspy.Predict(
            "text -> issues severity"
        )

    def forward(self, text: str) -> dspy.Prediction:
        """Extract issues and assess severity."""

        result = self.extractor(text=text)

        # Parse issues
        issues = [i.strip() for i in result.issues.split(";") if i.strip()] if result.issues else []

        # Map severity to numeric
        severity_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        severity_score = severity_map.get(result.severity.lower(), 2) if result.severity else 2

        return dspy.Prediction(
            issues=issues,
            issue_count=len(issues),
            severity_level=result.severity,
            severity_score=severity_score
        )

class FeedbackSummarizer(dspy.Module):
    """Generate concise summaries of feedback."""

    def __init__(self):
        super().__init__()
        self.summarizer = dspy.ChainOfThought(
            "text sentiment topics -> summary key_points"
        )

    def forward(self, text: str, sentiment: str, topics: List[str]) -> dspy.Prediction:
        """Generate feedback summary."""

        topics_str = ", ".join([sentiment] + topics[:3])
        result = self.summarizer(text=text, sentiment=sentiment, topics=topics_str)

        return dspy.Prediction(
            summary=result.summary,
            key_points=[p.strip() for p in result.key_points.split("\n") if p.strip()]
        )

class CustomerFeedbackSystem(dspy.Module):
    """Complete customer feedback analysis system."""

    def __init__(self):
        super().__init__()

        # Initialize all modules
        self.preprocessor = FeedbackPreprocessor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.topic_classifier = TopicClassifier()
        self.issue_extractor = IssueExtractor()
        self.summarizer = FeedbackSummarizer()

        # Performance tracking
        self.processed_count = 0
        self.start_time = datetime.now()

        # Cache for frequently analyzed patterns
        self.pattern_cache = {}

    def analyze_feedback(self, feedback: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze a single feedback entry."""

        start_time = datetime.now()

        # Initialize result
        result = {
            "feedback_id": metadata.get("id") if metadata else f"fb_{self.processed_count}",
            "timestamp": start_time.isoformat(),
            "metadata": metadata or {}
        }

        try:
            # Stage 1: Preprocessing
            preprocessed = self.preprocessor(feedback=feedback)
            result["preprocessing"] = {
                "word_count": preprocessed.word_count,
                "language": preprocessed.language,
                "has_emoji": preprocessed.has_emoji
            }

            # Use cleaned text for further analysis
            text_to_analyze = preprocessed.cleaned_text

            # Stage 2: Parallel Analysis
            with ThreadPoolExecutor(max_workers=3) as executor:
                # Submit parallel tasks
                sentiment_future = executor.submit(self.sentiment_analyzer, text_to_analyze)
                topic_future = executor.submit(self.topic_classifier, text_to_analyze)
                issue_future = executor.submit(self.issue_extractor, text_to_analyze)

                # Collect results
                sentiment = sentiment_future.result()
                topics = topic_future.result()
                issues = issue_future.result()

            # Store analysis results
            result["sentiment"] = {
                "label": sentiment.sentiment,
                "score": sentiment.sentiment_score,
                "intensity": sentiment.intensity
            }

            result["topics"] = {
                "primary": topics.primary_topic,
                "secondary": topics.secondary_topics,
                "count": topics.topic_count
            }

            result["issues"] = {
                "identified": issues.issues,
                "count": issues.issue_count,
                "severity": issues.severity_level,
                "score": issues.severity_score
            }

            # Stage 3: Summarization
            summary = self.summarizer(
                text=text_to_analyze,
                sentiment=sentiment.sentiment,
                topics=[topics.primary_topic] + topics.secondary_topics[:2]
            )

            result["summary"] = {
                "text": summary.summary,
                "key_points": summary.key_points
            }

            # Stage 4: Priority calculation
            priority = self._calculate_priority(sentiment.sentiment_score, issues.severity_score, preprocessed.word_count)
            result["priority"] = priority

            # Stage 5: Recommendations
            recommendations = self._generate_recommendations(sentiment.sentiment, issues.issues, topics.primary_topic)
            result["recommendations"] = recommendations

            # Update counters
            self.processed_count += 1

            result["success"] = True
            result["processing_time"] = (datetime.now() - start_time).total_seconds()

        except Exception as e:
            result["error"] = str(e)
            result["success"] = False
            result["processing_time"] = (datetime.now() - start_time).total_seconds()

        return result

    def analyze_batch(self, feedback_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze multiple feedback entries efficiently."""

        start_time = datetime.now()
        results = []

        # Process each feedback
        for feedback_data in feedback_list:
            result = self.analyze_feedback(
                feedback=feedback_data["text"],
                metadata=feedback_data.get("metadata", {})
            )
            results.append(result)

        # Generate batch insights
        batch_insights = self._generate_batch_insights(results)

        return {
            "results": results,
            "batch_insights": batch_insights,
            "total_processed": len(results),
            "successful": sum(1 for r in results if r.get("success", False)),
            "batch_processing_time": (datetime.now() - start_time).total_seconds(),
            "system_stats": self.get_system_stats()
        }

    def _calculate_priority(self, sentiment_score: float, severity_score: int, word_count: int) -> Dict[str, Any]:
        """Calculate feedback priority based on multiple factors."""

        # Lower sentiment = higher priority
        sentiment_factor = 1 - sentiment_score

        # Severity factor
        severity_factor = severity_score / 4

        # Length factor (detailed feedback gets slightly higher priority)
        length_factor = min(word_count / 100, 1) * 0.1

        # Calculate final priority score
        priority_score = (sentiment_factor * 0.5 + severity_factor * 0.4 + length_factor)

        # Determine priority level
        if priority_score > 0.7:
            level = "critical"
        elif priority_score > 0.5:
            level = "high"
        elif priority_score > 0.3:
            level = "medium"
        else:
            level = "low"

        return {
            "score": round(priority_score, 2),
            "level": level,
            "factors": {
                "sentiment": round(sentiment_factor, 2),
                "severity": round(severity_factor, 2),
                "length": round(length_factor, 2)
            }
        }

    def _generate_recommendations(self, sentiment: str, issues: List[str], topic: str) -> List[str]:
        """Generate actionable recommendations based on analysis."""

        recommendations = []

        # Sentiment-based recommendations
        if sentiment == "negative":
            recommendations.append("Immediate follow-up with customer recommended")
            recommendations.append("Review issue resolution process")
        elif sentiment == "positive":
            recommendations.append("Consider asking for public review/testimonial")
            recommendations.append("Identify what worked well for replication")

        # Issue-based recommendations
        if issues:
            if "service" in str(issues).lower():
                recommendations.append("Review customer service training")
            if "product" in str(issues).lower():
                recommendations.append("Forward to product team for review")
            if "delivery" in str(issues).lower():
                recommendations.append("Investigate shipping/logistics issues")

        # Topic-based recommendations
        topic_recommendations = {
            "pricing": "Review pricing strategy and competitors",
            "quality": "Quality assurance review recommended",
            "usability": "Consider UX/UI improvements",
            "features": "Evaluate feature requests for roadmap"
        }

        if topic.lower() in topic_recommendations:
            recommendations.append(topic_recommendations[topic.lower()])

        return recommendations[:3]  # Return top 3 recommendations

    def _generate_batch_insights(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate insights from batch analysis."""

        successful_results = [r for r in results if r.get("success", False)]

        if not successful_results:
            return {"error": "No successful analyses in batch"}

        # Sentiment distribution
        sentiments = [r["sentiment"]["label"] for r in successful_results]
        sentiment_dist = {
            "positive": sentiments.count("positive"),
            "negative": sentiments.count("negative"),
            "neutral": sentiments.count("neutral")
        }

        # Topic distribution
        topics = [r["topics"]["primary"] for r in successful_results]
        topic_dist = {}
        for topic in topics:
            topic_dist[topic] = topic_dist.get(topic, 0) + 1

        # Priority distribution
        priorities = [r["priority"]["level"] for r in successful_results]
        priority_dist = {
            "critical": priorities.count("critical"),
            "high": priorities.count("high"),
            "medium": priorities.count("medium"),
            "low": priorities.count("low")
        }

        # Average metrics
        avg_sentiment = sum(r["sentiment"]["score"] for r in successful_results) / len(successful_results)
        avg_severity = sum(r["issues"]["score"] for r in successful_results) / len(successful_results)

        return {
            "sentiment_distribution": sentiment_dist,
            "topic_distribution": topic_dist,
            "priority_distribution": priority_dist,
            "average_sentiment_score": round(avg_sentiment, 2),
            "average_severity_score": round(avg_severity, 2),
            "total_issues_identified": sum(r["issues"]["count"] for r in successful_results),
            "recommendations_generated": sum(len(r.get("recommendations", [])) for r in successful_results)
        }

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics."""

        uptime = datetime.now() - self.start_time

        return {
            "feedback_processed": self.processed_count,
            "uptime_hours": uptime.total_seconds() / 3600,
            "cache_size": len(self.pattern_cache),
            "average_processing_rate": self.processed_count / (uptime.total_seconds() / 60) if uptime.total_seconds() > 0 else 0
        }

def test_feedback_system():
    """Test the complete customer feedback system."""

    system = CustomerFeedbackSystem()

    # Sample feedback data
    test_feedback = [
        {
            "text": """
            I'm extremely disappointed with the recent changes to your mobile app.
            The new interface is confusing and I can't find half the features I used daily.
            Customer support was unhelpful when I contacted them. Please fix these issues ASAP.
            """,
            "metadata": {"customer_id": "C12345", "date": "2024-12-12", "channel": "app"}
        },
        {
            "text": """
            Amazing service! The delivery was faster than expected and the product quality exceeded my expectations.
            The customer service team was very helpful with my questions. Will definitely order again!
            """,
            "metadata": {"customer_id": "C67890", "date": "2024-12-11", "channel": "email"}
        },
        {
            "text": """
            The product is okay, does what it's supposed to do. Nothing special but not terrible either.
            Price seems reasonable for what you get.
            """,
            "metadata": {"customer_id": "C11111", "date": "2024-12-10", "channel": "survey"}
        },
        {
            "text": """
            Critical issue: The billing system overcharged me twice this month!
            I've been trying to resolve this for weeks with no response.
            This is unacceptable and might force me to switch providers.
            """,
            "metadata": {"customer_id": "C22222", "date": "2024-12-12", "channel": "phone"}
        },
        {
            "text": """
            Love the new features in the latest update! Especially the improved search functionality.
            It's so much easier to find what I'm looking for now. Keep up the great work!
            """,
            "metadata": {"customer_id": "C33333", "date": "2024-12-09", "channel": "in-app"}
        }
    ]

    print("=" * 60)
    print("Customer Feedback Analysis System")
    print("=" * 60)

    # Analyze batch
    batch_result = system.analyze_batch(test_feedback)

    # Display results
    print(f"\nBatch Analysis Results:")
    print(f"Total Processed: {batch_result['total_processed']}")
    print(f"Successful: {batch_result['successful']}")
    print(f"Processing Time: {batch_result['batch_processing_time']:.2f}s")

    # Show individual analysis for first feedback
    print("\n" + "-" * 40)
    print("Sample Analysis (First Feedback):")
    print("-" * 40)

    if batch_result["results"]:
        first_result = batch_result["results"][0]
        if first_result.get("success"):
            print(f"Sentiment: {first_result['sentiment']['label']} (Score: {first_result['sentiment']['score']})")
            print(f"Primary Topic: {first_result['topics']['primary']}")
            print(f"Issues Found: {first_result['issues']['count']}")
            print(f"Priority: {first_result['priority']['level']} (Score: {first_result['priority']['score']})")
            print(f"\nSummary: {first_result['summary']['text']}")
            print(f"\nRecommendations:")
            for rec in first_result.get("recommendations", []):
                print(f"  - {rec}")

    # Show batch insights
    print("\n" + "-" * 40)
    print("Batch Insights:")
    print("-" * 40)

    insights = batch_result["batch_insights"]
    print(f"Sentiment Distribution: {insights['sentiment_distribution']}")
    print(f"Average Sentiment Score: {insights['average_sentiment_score']}")
    print(f"Topic Distribution: {insights['topic_distribution']}")
    print(f"Priority Distribution: {insights['priority_distribution']}")
    print(f"Total Issues Identified: {insights['total_issues_identified']}")

    # Show system stats
    print("\n" + "-" * 40)
    print("System Statistics:")
    print("-" * 40)

    stats = batch_result["system_stats"]
    print(f"Total Feedback Processed: {stats['feedback_processed']}")
    print(f"Average Processing Rate: {stats['average_processing_rate']:.1f} feedbacks/minute")

def main():
    """Main function to run Exercise 7."""

    print("\n" = *60)
    print("Exercise 7: Complete Customer Feedback Analysis System")
    print("Comprehensive system with multiple analysis stages and optimization")
    print("=" * 60)

    # Run the system test
    test_feedback_system()

    print("\n" + "=" * 60)
    print("Exercise 7 Completed Successfully!")
    print("System features demonstrated:")
    print("  - Multi-stage analysis pipeline")
    print("  - Parallel processing for efficiency")
    print("  - Priority calculation and recommendations")
    print("  - Batch insights and statistics")
    print("  - Performance tracking")
    print("=" * 60)

if __name__ == "__main__":
    main()
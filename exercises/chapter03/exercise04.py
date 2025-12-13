"""
Exercise 4: Custom Module Development
Solution for Exercise 4 from Chapter 3

Task: Create a custom module for document analysis
- Implement a custom DSPy module
- Include multiple processing stages
- Add state management
"""

import dspy
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

class DocumentAnalyzer(dspy.Module):
    """Custom module for comprehensive document analysis."""

    def __init__(self):
        super().__init__()

        # Initialize internal modules
        self.summarizer = dspy.ChainOfThought(
            "document -> summary"
        )
        self.entity_extractor = dspy.Predict(
            "document -> entities"
        )
        self.sentiment_analyzer = dspy.Predict(
            "text -> sentiment"
        )
        self.keyword_extractor = dspy.Predict(
            "document -> keywords"
        )

        # Initialize state
        self.processed_documents = []
        self.analysis_history = []

    def forward(self, document: str, analysis_type: str = "full") -> dspy.Prediction:
        """
        Analyze a document with multiple processing stages.

        Args:
            document: The text document to analyze
            analysis_type: Type of analysis ("full", "summary", "sentiment", "entities")
        """

        # Store processing timestamp
        timestamp = datetime.now().isoformat()

        # Initialize results dictionary
        analysis_results = {
            "document": document,
            "timestamp": timestamp,
            "analysis_type": analysis_type
        }

        try:
            # Stage 1: Basic preprocessing
            preprocessed = self._preprocess_document(document)
            analysis_results["preprocessing"] = {
                "word_count": preprocessed["word_count"],
                "sentence_count": preprocessed["sentence_count"],
                "char_count": len(document),
                "language": self._detect_language(document)
            }

            # Stage 2: Summary generation
            if analysis_type in ["full", "summary"]:
                summary_result = self.summarizer(document=document)
                analysis_results["summary"] = summary_result.summary

            # Stage 3: Entity extraction
            if analysis_type in ["full", "entities"]:
                entity_result = self.entity_extractor(document=document)
                entities = self._parse_entities(entity_result.entities)
                analysis_results["entities"] = {
                    "people": entities["people"],
                    "organizations": entities["organizations"],
                    "locations": entities["locations"],
                    "dates": entities["dates"]
                }

            # Stage 4: Sentiment analysis
            if analysis_type in ["full", "sentiment"]:
                sentiment_result = self.sentiment_analyzer(text=document)
                analysis_results["sentiment"] = {
                    "label": sentiment_result.sentiment,
                    "score": self._extract_sentiment_score(sentiment_result.sentiment)
                }

            # Stage 5: Keyword extraction
            if analysis_type == "full":
                keyword_result = self.keyword_extractor(document=document)
                keywords = self._parse_keywords(keyword_result.keywords)
                analysis_results["keywords"] = keywords

            # Stage 6: Custom analysis
            analysis_results["custom_metrics"] = self._calculate_custom_metrics(document)

            # Store in history
            self.analysis_history.append(analysis_results)
            self.processed_documents.append({
                "timestamp": timestamp,
                "document_preview": document[:100] + "...",
                "analysis_type": analysis_type
            })

            return dspy.Prediction(
                **analysis_results,
                success=True,
                documents_processed=len(self.processed_documents)
            )

        except Exception as e:
            return dspy.Prediction(
                document=document,
                timestamp=timestamp,
                error=str(e),
                success=False
            )

    def _preprocess_document(self, document: str) -> Dict[str, int]:
        """Basic document preprocessing."""
        # Count words
        words = document.split()
        word_count = len(words)

        # Count sentences (simple heuristic)
        sentences = re.split(r'[.!?]+', document)
        sentence_count = len([s for s in sentences if s.strip()])

        return {
            "word_count": word_count,
            "sentence_count": sentence_count
        }

    def _detect_language(self, text: str) -> str:
        """Simple language detection based on common words."""
        # Very basic implementation
        if any(word in text.lower() for word in ["the", "and", "is", "are"]):
            return "English"
        elif any(word in text.lower() for word in ["le", "et", "est", "dans"]):
            return "French"
        elif any(word in text.lower() for word in ["el", "la", "es", "en"]):
            return "Spanish"
        else:
            return "Unknown"

    def _parse_entities(self, entities_text: str) -> Dict[str, List[str]]:
        """Parse entities from the extraction result."""
        # This is a simplified parser - in practice, you'd use NER
        entities = {
            "people": [],
            "organizations": [],
            "locations": [],
            "dates": []
        }

        # Simple regex-based extraction
        # Dates (YYYY-MM-DD or Month DD, YYYY)
        date_pattern = r'\b\d{4}-\d{2}-\d{2}\b|\b[A-Z][a-z]+ \d{1,2}, \d{4}\b'
        entities["dates"] = re.findall(date_pattern, entities_text)

        # Organizations (words starting with capital letters)
        org_pattern = r'\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company|Technologies)\b'
        entities["organizations"] = re.findall(org_pattern, entities_text)

        # People (simple pattern: Title + Name)
        person_pattern = r'\b(?:Mr|Mrs|Dr|Prof)\. [A-Z][a-z]+ [A-Z][a-z]+\b'
        entities["people"] = re.findall(person_pattern, entities_text)

        # Locations (Cities, Countries)
        location_pattern = r'\b(?:New York|London|Paris|Tokyo|Berlin|San Francisco)\b'
        entities["locations"] = re.findall(location_pattern, entities_text)

        return entities

    def _extract_sentiment_score(self, sentiment_text: str) -> float:
        """Extract a numeric sentiment score from sentiment text."""
        sentiment_lower = sentiment_text.lower()

        if any(word in sentiment_lower for word in ["positive", "good", "excellent"]):
            return 0.8
        elif any(word in sentiment_lower for word in ["negative", "bad", "terrible"]):
            return 0.2
        elif any(word in sentiment_lower for word in ["neutral", "okay", "average"]):
            return 0.5
        else:
            return 0.5  # Default to neutral

    def _parse_keywords(self, keywords_text: str) -> List[str]:
        """Parse keywords from the extraction result."""
        # Split by common delimiters
        keywords = re.split(r'[,;]\s*', keywords_text)
        # Clean and filter
        cleaned_keywords = [
            kw.strip().lower()
            for kw in keywords
            if kw.strip() and len(kw.strip()) > 2
        ]
        return cleaned_keywords[:10]  # Return top 10 keywords

    def _calculate_custom_metrics(self, document: str) -> Dict[str, float]:
        """Calculate custom document metrics."""
        words = document.split()

        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

        # Readability score (simplified)
        avg_sentence_length = len(words) / (document.count('.') + document.count('!') + document.count('?') or 1)
        readability = max(0, 100 - avg_sentence_length * 2)

        # Complexity score (based on word variety)
        unique_words = set(word.lower() for word in words)
        complexity = len(unique_words) / len(words) if words else 0

        return {
            "avg_word_length": round(avg_word_length, 2),
            "readability_score": round(readability, 2),
            "complexity_score": round(complexity, 2)
        }

    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get the history of analyzed documents."""
        return self.processed_documents

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about processed documents."""
        if not self.processed_documents:
            return {"total_documents": 0}

        analysis_types = {}
        for doc in self.processed_documents:
            atype = doc["analysis_type"]
            analysis_types[atype] = analysis_types.get(atype, 0) + 1

        return {
            "total_documents": len(self.processed_documents),
            "analysis_types": analysis_types,
            "first_analysis": self.processed_documents[0]["timestamp"],
            "latest_analysis": self.processed_documents[-1]["timestamp"]
        }

def test_document_analyzer():
    """Test the document analyzer with sample documents."""

    analyzer = DocumentAnalyzer()

    # Sample documents
    test_documents = [
        {
            "text": """
            Artificial Intelligence: A New Era

            Artificial Intelligence (AI) has revolutionized the technology industry in 2024.
            Companies like Google, Microsoft, and OpenAI are leading the innovation.
            Dr. Sarah Johnson from MIT published groundbreaking research on neural networks
            last month. The study shows significant improvements in natural language
            processing capabilities.

            The positive impact of AI on healthcare is particularly noteworthy.
            Medical professionals are using AI for diagnosis and treatment planning.
            """,
            "type": "full"
        },
        {
            "text": """
            Climate Change Report

            Global temperatures have risen by 1.5°C since pre-industrial levels.
            This change is primarily caused by greenhouse gas emissions from
            human activities. The report emphasizes the urgent need for
            renewable energy adoption.

            New York, London, and Tokyo have committed to carbon neutrality
            by 2050. These cities are investing heavily in solar and wind energy.
            """,
            "type": "summary"
        },
        {
            "text": """
            Product Review: Terrible Experience

            I purchased this product on January 15, 2024, and it's the worst
            purchase I've ever made. The quality is poor and customer service
            is non-existent. I do not recommend this to anyone. Very disappointed.
            """,
            "type": "sentiment"
        }
    ]

    print("=" * 60)
    print("Document Analyzer Test")
    print("=" * 60)

    results = []

    for i, doc_data in enumerate(test_documents, 1):
        print(f"\nAnalyzing Document {i} (Type: {doc_data['type']}):")
        print("-" * 40)

        result = analyzer.forward(document=doc_data["text"], analysis_type=doc_data["type"])
        results.append(result)

        if result.success:
            print(f"Word Count: {result.preprocessing.word_count}")
            print(f"Language: {result.preprocessing.language}")

            if hasattr(result, 'summary'):
                print(f"\nSummary: {result.summary[:100]}...")

            if hasattr(result, 'sentiment'):
                print(f"\nSentiment: {result.sentiment.label} (Score: {result.sentiment.score})")

            if hasattr(result, 'entities'):
                print(f"\nEntities Found:")
                for entity_type, entities in result.entities.items():
                    if entities:
                        print(f"  - {entity_type.title()}: {', '.join(entities[:3])}")

            if hasattr(result, 'custom_metrics'):
                print(f"\nCustom Metrics:")
                print(f"  - Avg Word Length: {result.custom_metrics.avg_word_length}")
                print(f"  - Readability: {result.custom_metrics.readability_score}")
        else:
            print(f"Error: {result.error}")

    # Show statistics
    stats = analyzer.get_statistics()
    print(f"\n\nAnalyzer Statistics:")
    print(f"Total Documents: {stats['total_documents']}")
    print(f"Analysis Types: {stats['analysis_types']}")

def main():
    """Main function to run Exercise 4."""

    print("\n" + "=" * 60)
    print("Exercise 4: Custom Module Development")
    print("Creating a custom document analysis module with state management")
    print("=" * 60)

    # Run the test
    test_document_analyzer()

    print("\n" + "=" * 60)
    print("Exercise 4 Completed Successfully!")
    print("Custom module with multiple processing stages and state management.")
    print("=" * 60)

if __name__ == "__main__":
    main()
"""
Custom Module Examples

This file demonstrates creating and using custom DSPy modules:
- Extending dspy.Module base class
- Implementing custom forward methods
- Module parameter management
- Complex custom architectures
- Real-world custom module patterns
"""

import dspy
import re
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# Example 1: Simple Custom Module
class TextCleaner(dspy.Module):
    """Custom module for cleaning and preprocessing text."""

    def __init__(self):
        super().__init__()
        # Define internal processing modules
        self.signature = dspy.Signature("raw_text -> cleaned_text")
        self.cleaner = dspy.Predict(self.signature)

    def forward(self, raw_text):
        """Clean text using multiple processing steps."""
        # Basic text cleaning
        cleaned = raw_text.strip()

        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)

        # Remove special characters if needed
        # Use LLM for more sophisticated cleaning
        result = self.cleaner(raw_text=cleaned)

        return dspy.Prediction(
            original_text=raw_text,
            cleaned_text=result.cleaned_text,
            word_count=len(result.cleaned_text.split()),
            char_count=len(result.cleaned_text)
        )

# Example 2: Advanced Custom Module with Multiple Steps
class SentimentAnalysisPipeline(dspy.Module):
    """Custom module for comprehensive sentiment analysis."""

    def __init__(self):
        super().__init__()

        # Define signatures for different steps
        self.emotion_sig = dspy.Signature("text -> emotions")
        self.sentiment_sig = dspy.Signature("text -> sentiment_score")
        self.aspects_sig = dspy.Signature("text -> aspects")
        self.summary_sig = dspy.Signature(
            "emotions, sentiment_score, aspects -> sentiment_summary"
        )

        # Create modules
        self.emotion_detector = dspy.Predict(self.emotion_sig)
        self.sentiment_scorer = dspy.Predict(self.sentiment_sig)
        self.aspect_extractor = dspy.Predict(self.aspects_sig)
        self.summarizer = dspy.Predict(self.summary_sig)

    def forward(self, text):
        """Process text through complete sentiment pipeline."""
        # Step 1: Detect emotions
        emotions = self.emotion_detector(text=text)

        # Step 2: Score sentiment
        sentiment = self.sentiment_scorer(text=text)

        # Step 3: Extract aspects
        aspects = self.aspect_extractor(text=text)

        # Step 4: Generate summary
        summary = self.summarizer(
            emotions=emotions.emotions,
            sentiment_score=sentiment.sentiment_score,
            aspects=aspects.aspects
        )

        return dspy.Prediction(
            text=text,
            emotions=emotions.emotions,
            sentiment_score=sentiment.sentiment_score,
            aspects=aspects.aspects,
            summary=summary.sentiment_summary,
            processed_at=datetime.now().isoformat()
        )

# Example 3: Custom Module with Conditional Logic
class AdaptiveProcessor(dspy.Module):
    """Custom module that adapts processing based on input."""

    def __init__(self):
        super().__init__()

        # Different processors for different types
        self.classifier_sig = dspy.Signature("text -> text_type")
        self.formal_sig = dspy.Signature("text -> formal_response")
        self.casual_sig = dspy.Signature("text -> casual_response")
        self.technical_sig = dspy.Signature("text -> technical_analysis")

        # Modules
        self.classifier = dspy.Predict(self.classifier_sig)
        self.formal_processor = dspy.Predict(self.formal_sig)
        self.casual_processor = dspy.Predict(self.casual_sig)
        self.technical_processor = dspy.Predict(self.technical_sig)

    def forward(self, text, context="general"):
        """Adaptively process text based on its type."""
        # Classify the text
        classification = self.classifier(text=text)

        # Route to appropriate processor
        text_type = classification.text_type.lower()

        if "formal" in text_type or "academic" in text_type:
            result = self.formal_processor(text=text)
            processor_used = "formal"
        elif "casual" in text_type or "informal" in text_type:
            result = self.casual_processor(text=text)
            processor_used = "casual"
        elif "technical" in text_type or "scientific" in text_type:
            result = self.technical_processor(text=text)
            processor_used = "technical"
        else:
            # Default processing
            result = self.casual_processor(text=text)
            processor_used = "default"

        return dspy.Prediction(
            original_text=text,
            processed_text=result.get('formal_response') or result.get('casual_response') or result.get('technical_analysis'),
            text_type=classification.text_type,
            processor_used=processor_used,
            context=context
        )

# Example 4: Custom Module with Internal State
class ConversationMemory(dspy.Module):
    """Custom module that maintains conversation context."""

    def __init__(self, max_history=5):
        super().__init__()
        self.max_history = max_history
        self.conversation_history = []

        # Processing signatures
        self.context_sig = dspy.Signature("current_message, history -> contextualized_response")
        self.summary_sig = dspy.Signature("conversation_history -> conversation_summary")

        # Modules
        self.contextualizer = dspy.Predict(self.context_sig)
        self.summarizer = dspy.Predict(self.summary_sig)

    def forward(self, message):
        """Process message with conversation context."""
        # Add to history
        self.conversation_history.append({
            "message": message,
            "timestamp": datetime.now().isoformat()
        })

        # Limit history size
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

        # Create history string for context
        history_str = json.dumps(self.conversation_history[:-1], indent=2)

        # Generate contextualized response
        response = self.contextualizer(
            current_message=message,
            history=history_str
        )

        # Generate summary periodically
        if len(self.conversation_history) % 3 == 0:
            summary = self.summarizer(
                conversation_history=json.dumps(self.conversation_history, indent=2)
            )
            conversation_summary = summary.conversation_summary
        else:
            conversation_summary = None

        return dspy.Prediction(
            message=message,
            response=response.contextualized_response,
            history_length=len(self.conversation_history),
            conversation_summary=conversation_summary,
            timestamp=datetime.now().isoformat()
        )

# Example 5: Custom Module with Ensemble Processing
class EnsembleClassifier(dspy.Module):
    """Custom module that combines multiple classifiers."""

    def __init__(self):
        super().__init__()

        # Multiple classifier signatures
        self.nlp_sig = dspy.Signature("text -> nlp_classification")
        self.keyword_sig = dspy.Signature("text -> keyword_classification")
        self.semantic_sig = dspy.Signature("text -> semantic_classification")
        self.ensemble_sig = dspy.Signature(
            "nlp_result, keyword_result, semantic_result -> final_classification"
        )

        # Classifier modules
        self.nlp_classifier = dspy.Predict(self.nlp_sig)
        self.keyword_classifier = dspy.Predict(self.keyword_sig)
        self.semantic_classifier = dspy.Predict(self.semantic_sig)
        self.ensemble = dspy.Predict(self.ensemble_sig)

    def forward(self, text):
        """Classify text using ensemble of methods."""
        # Get classifications from all methods
        nlp_result = self.nlp_classifier(text=text)
        keyword_result = self.keyword_classifier(text=text)
        semantic_result = self.semantic_classifier(text=text)

        # Ensemble the results
        final = self.ensemble(
            nlp_result=nlp_result.nlp_classification,
            keyword_result=keyword_result.keyword_classification,
            semantic_result=semantic_result.semantic_classification
        )

        # Calculate confidence based on agreement
        classifications = [
            nlp_result.nlp_classification,
            keyword_result.keyword_classification,
            semantic_result.semantic_classification
        ]

        # Simple confidence calculation
        most_common = max(set(classifications), key=classifications.count)
        confidence = classifications.count(most_common) / len(classifications)

        return dspy.Prediction(
            text=text,
            nlp_classification=nlp_result.nlp_classification,
            keyword_classification=keyword_result.keyword_classification,
            semantic_classification=semantic_result.semantic_classification,
            final_classification=final.final_classification,
            confidence=confidence,
            agreement_score=confidence
        )

# Example 6: Custom Module with Validation
class ValidatedQA(dspy.Module):
    """Custom QA module with answer validation."""

    def __init__(self):
        super().__init__()

        self.qa_sig = dspy.Signature("question -> answer")
        self.validation_sig = dspy.Signature("question, answer -> validation_score")
        self.refinement_sig = dspy.Signature("question, initial_answer, validation -> refined_answer")

        self.qa_module = dspy.Predict(self.qa_sig)
        self.validator = dspy.Predict(self.validation_sig)
        self.refiner = dspy.Predict(self.refinement_sig)

    def forward(self, question, min_confidence=0.7, max_attempts=3):
        """Generate and validate answer with refinement."""
        attempts = 0
        current_answer = None
        validation_score = 0

        while attempts < max_attempts and validation_score < min_confidence:
            attempts += 1

            if attempts == 1:
                # First attempt
                qa_result = self.qa_module(question=question)
                current_answer = qa_result.answer
            else:
                # Refine previous answer
                refine_result = self.refiner(
                    question=question,
                    initial_answer=current_answer,
                    validation=f"Previous validation score: {validation_score}"
                )
                current_answer = refine_result.refined_answer

            # Validate the answer
            validation = self.validator(
                question=question,
                answer=current_answer
            )

            try:
                validation_score = float(validation.validation_score)
            except:
                validation_score = 0.5  # Default if parsing fails

        return dspy.Prediction(
            question=question,
            answer=current_answer,
            validation_score=validation_score,
            attempts=attempts,
            is_valid=validation_score >= min_confidence
        )

# Example 7: Custom Module with Tool Integration
class CalculatorPlus(dspy.Module):
    """Custom calculator module with extended functionality."""

    def __init__(self):
        super().__init__()

        self.calc_sig = dspy.Signature("expression -> result")
        self.word_problem_sig = dspy.Signature("word_problem -> mathematical_expression")
        self.verification_sig = dspy.Signature("expression, result -> verification")

        self.calculator = dspy.Predict(self.calc_sig)
        self.word_parser = dspy.Predict(self.word_problem_sig)
        self.verifier = dspy.Predict(self.verification_sig)

    def forward(self, input_text):
        """Process mathematical input, supporting expressions and word problems."""
        # Determine if it's a word problem or direct expression
        if any(word in input_text.lower() for word in ['what', 'calculate', 'find', 'how many', 'how much']):
            # Parse word problem
            parsed = self.word_parser(word_problem=input_text)
            expression = parsed.mathematical_expression
            input_type = "word_problem"
        else:
            expression = input_text
            input_type = "expression"

        # Calculate result
        calc_result = self.calculator(expression=expression)

        # Verify calculation
        verification = self.verifier(
            expression=expression,
            result=calc_result.result
        )

        # Extract numeric result for additional processing
        try:
            numeric_result = self._safe_eval(calc_result.result)
        except:
            numeric_result = None

        return dspy.Prediction(
            input_text=input_text,
            input_type=input_type,
            expression=expression,
            result=calc_result.result,
            numeric_result=numeric_result,
            verification=verification.verification,
            timestamp=datetime.now().isoformat()
        )

    def _safe_eval(self, expression):
        """Safely evaluate mathematical expressions."""
        import math
        allowed_names = {
            "__builtins__": {},
            "abs": abs, "min": min, "max": max, "round": round,
            "pow": pow, "sqrt": math.sqrt, "log": math.log,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "pi": math.pi, "e": math.e
        }
        return eval(expression, allowed_names, {})

# Example 8: Custom Module with Metrics Tracking
class MetricsTracker(dspy.Module):
    """Custom module that tracks processing metrics."""

    def __init__(self):
        super().__init__()
        self.metrics = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "avg_processing_time": 0,
            "total_processing_time": 0
        }

        self.process_sig = dspy.Signature("input -> output")
        self.processor = dspy.Predict(self.process_sig)

    def forward(self, data):
        """Process data while tracking metrics."""
        start_time = datetime.now()

        try:
            result = self.processor(input=data)
            success = True
            output = result.output
        except Exception as e:
            success = False
            output = f"Error: {str(e)}"

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Update metrics
        self.metrics["total_processed"] += 1
        if success:
            self.metrics["successful"] += 1
        else:
            self.metrics["failed"] += 1

        self.metrics["total_processing_time"] += processing_time
        self.metrics["avg_processing_time"] = (
            self.metrics["total_processing_time"] / self.metrics["total_processed"]
        )

        return dspy.Prediction(
            input=data,
            output=output,
            success=success,
            processing_time=processing_time,
            metrics=self.metrics.copy()
        )

# Demonstration Functions
def demonstrate_simple_custom_module():
    """Demonstrate basic custom module usage."""

    print("=" * 60)
    print("Example 1: Simple Custom Module - TextCleaner")
    print("=" * 60)

    cleaner = TextCleaner()

    texts = [
        "  This    text  has   extra   spaces!  ",
        "\t\nNewlines\nand\ttabs\n\nshould\tbe\n cleaned  \t\n",
        "   Mixed    whitespace    issues   "
    ]

    print("\nText Cleaning Examples:")
    print("-" * 40)

    for text in texts:
        result = cleaner(raw_text=text)
        print(f"\nOriginal: {repr(text)}")
        print(f"Cleaned: {repr(result.cleaned_text)}")
        print(f"Words: {result.word_count}, Characters: {result.char_count}")

def demonstrate_advanced_custom_module():
    """Demonstrate advanced custom module with pipeline."""

    print("\n" + "=" * 60)
    print("Example 2: Advanced Custom Module - SentimentAnalysisPipeline")
    print("=" * 60)

    analyzer = SentimentAnalysisPipeline()

    texts = [
        "I absolutely love this product! It's amazing and works perfectly.",
        "This is terrible. Worst purchase I've ever made. Complete waste of money.",
        "It's okay, nothing special but does the job."
    ]

    print("\nSentiment Analysis Pipeline:")
    print("-" * 40)

    for text in texts:
        result = analyzer(text=text)
        print(f"\nText: {text}")
        print(f"Emotions: {result.emotions}")
        print(f"Sentiment Score: {result.sentiment_score}")
        print(f"Aspects: {result.aspects}")
        print(f"Summary: {result.summary}")

def demonstrate_adaptive_processor():
    """Demonstrate adaptive processing module."""

    print("\n" + "=" * 60)
    print("Example 3: Adaptive Custom Module - AdaptiveProcessor")
    print("=" * 60)

    processor = AdaptiveProcessor()

    texts = [
        ("Dear Sir/Madam, I am writing to inquire about...", "formal"),
        ("Hey what's up? Check out this cool thing!", "casual"),
        ("The neural network utilizes backpropagation for gradient descent optimization.", "technical")
    ]

    print("\nAdaptive Processing:")
    print("-" * 40)

    for text, context in texts:
        result = processor(text=text, context=context)
        print(f"\nText: {text[:50]}...")
        print(f"Detected Type: {result.text_type}")
        print(f"Processor Used: {result.processor_used}")
        print(f"Context: {result.context}")

def demonstrate_conversation_memory():
    """Demonstrate conversation memory module."""

    print("\n" + "=" * 60)
    print("Example 4: Stateful Custom Module - ConversationMemory")
    print("=" * 60)

    memory = ConversationMemory(max_history=3)

    messages = [
        "Hi, my name is Alice",
        "I'm looking for a good restaurant",
        "Do you have any recommendations?",
        "Actually, I prefer Italian food",
        "Thanks for the help!"
    ]

    print("\nConversation with Memory:")
    print("-" * 40)

    for msg in messages:
        result = memory(message=msg)
        print(f"\nMessage: {msg}")
        print(f"Response: {result.response[:80]}...")
        print(f"History Length: {result.history_length}")
        if result.conversation_summary:
            print(f"Summary: {result.conversation_summary}")

def demonstrate_ensemble_classifier():
    """Demonstrate ensemble classification module."""

    print("\n" + "=" * 60)
    print("Example 5: Ensemble Custom Module - EnsembleClassifier")
    print("=" * 60)

    classifier = EnsembleClassifier()

    texts = [
        "This movie was fantastic and thrilling!",
        "The stock market is showing bearish trends.",
        "I need to fix the bug in my Python code.",
        "Let's meet at the restaurant at 7 PM."
    ]

    print("\nEnsemble Classification:")
    print("-" * 40)

    for text in texts:
        result = classifier(text=text)
        print(f"\nText: {text}")
        print(f"NLP: {result.nlp_classification}")
        print(f"Keyword: {result.keyword_classification}")
        print(f"Semantic: {result.semantic_classification}")
        print(f"Final: {result.final_classification}")
        print(f"Confidence: {result.confidence:.2f}")

def demonstrate_validated_qa():
    """Demonstrate validated QA module."""

    print("\n" + "=" * 60)
    print("Example 6: Custom Module with Validation - ValidatedQA")
    print("=" * 60)

    qa = ValidatedQA(min_confidence=0.8)

    questions = [
        "What is the capital of France?",
        "How many planets are in our solar system?",
        "Who wrote Romeo and Juliet?"
    ]

    print("\nValidated Question Answering:")
    print("-" * 40)

    for question in questions:
        result = qa(question=question)
        print(f"\nQuestion: {question}")
        print(f"Answer: {result.answer}")
        print(f"Validation Score: {result.validation_score:.2f}")
        print(f"Attempts: {result.attempts}")
        print(f"Is Valid: {result.is_valid}")

def demonstrate_calculator_plus():
    """Demonstrate calculator with extended functionality."""

    print("\n" + "=" * 60)
    print("Example 7: Custom Module with Tools - CalculatorPlus")
    print("=" * 60)

    calc = CalculatorPlus()

    inputs = [
        "2 + 3 * 4",
        "What is 15% of 200?",
        "If I buy 3 items at $25 each, how much do I pay?",
        "sqrt(16) + pow(2, 3)"
    ]

    print("\nExtended Calculator:")
    print("-" * 40)

    for inp in inputs:
        result = calc(input_text=inp)
        print(f"\nInput: {inp}")
        print(f"Type: {result.input_type}")
        print(f"Expression: {result.expression}")
        print(f"Result: {result.result}")
        if result.numeric_result is not None:
            print(f"Numeric: {result.numeric_result}")

def demonstrate_metrics_tracking():
    """Demonstrate metrics tracking module."""

    print("\n" + "=" * 60)
    print("Example 8: Custom Module with Metrics - MetricsTracker")
    print("=" * 60)

    tracker = MetricsTracker()

    data_points = [
        "Process this data",
        "Analyze this information",
        "Summarize this content",
        "Error: This might cause issues",
        "Final data point"
    ]

    print("\nMetrics Tracking:")
    print("-" * 40)

    for data in data_points:
        result = tracker(data=data)
        print(f"\nInput: {data}")
        print(f"Success: {result.success}")
        print(f"Processing Time: {result.processing_time:.3f}s")
        print(f"Total Processed: {result.metrics['total_processed']}")
        print(f"Success Rate: {result.metrics['successful']}/{result.metrics['total_processed']}")
        print(f"Avg Time: {result.metrics['avg_processing_time']:.3f}s")

# Main execution
def run_all_examples():
    """Run all custom module examples."""

    print("DSPy Custom Module Examples")
    print("These examples demonstrate various custom module patterns and architectures.")
    print("=" * 60)

    try:
        demonstrate_simple_custom_module()
        demonstrate_advanced_custom_module()
        demonstrate_adaptive_processor()
        demonstrate_conversation_memory()
        demonstrate_ensemble_classifier()
        demonstrate_validated_qa()
        demonstrate_calculator_plus()
        demonstrate_metrics_tracking()

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_examples()
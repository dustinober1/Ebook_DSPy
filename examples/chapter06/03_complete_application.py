"""
Complete Multi-Application System

This file demonstrates a comprehensive application that combines multiple DSPy components
to build a document processing and analysis system for a real-world use case.

The system includes:
- Document classification
- Entity extraction
- RAG-based Q&A
- Intelligent agent for document assistance
- Complete pipeline integration
"""

import dspy
from dspy.teleprompter import BootstrapFewShot, MIPRO
from typing import List, Dict, Any, Optional, Tuple
import time
from datetime import datetime
import json

# Configure language model (placeholder)
dspy.settings.configure(lm=dspy.LM(model="gpt-3.5-turbo", api_key="your-key"))

# Document Classifiers
class DocumentClassifier(dspy.Module):
    """Classifies documents into different categories."""

    def __init__(self, categories):
        super().__init__()
        self.categories = categories
        self.classify = dspy.Predict(
            f"document_text, categories[{', '.join(categories)}] -> category, confidence"
        )

    def forward(self, document_text):
        result = self.classify(
            document_text=document_text,
            categories=", ".join(self.categories)
        )

        return dspy.Prediction(
            category=result.category,
            confidence=float(result.confidence)
        )

# Entity Extractor
class DocumentEntityExtractor(dspy.Module):
    """Extracts entities from documents."""

    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(
            "document_text -> entities, relationships"
        )

    def forward(self, document_text):
        result = self.extract(document_text=document_text)

        return dspy.Prediction(
            entities=result.entities,
            relationships=result.relationships,
            reasoning=result.rationale
        )

# RAG System for Document Q&A
class DocumentQA(dspy.Module):
    """Answers questions about documents using RAG."""

    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=5)
        self.answer = dspy.ChainOfThought("context, question -> answer, confidence")

    def forward(self, question, document_context=None):
        if document_context:
            # Use provided context
            context = document_context
        else:
            # Retrieve relevant documents
            retrieved = self.retrieve(question=question)
            context = "\n".join(retrieved.passages)

        result = self.answer(context=context, question=question)

        return dspy.Prediction(
            answer=result.answer,
            confidence=float(result.confidence),
            context=context
        )

# Document Assistant Agent
class DocumentAssistantAgent(dspy.Module):
    """Intelligent agent for document assistance."""

    def __init__(self):
        super().__init__()
        self.understand_request = dspy.Predict(
            "user_request -> request_type, entities, urgency"
        )
        self.generate_response = dspy.ChainOfThought(
            "request_type, entities, document_info -> helpful_response, action_suggestions"
        )
        self.memory = []

    def forward(self, user_request, document_info=None):
        understanding = self.understand_request(user_request=user_request)

        result = self.generate_response(
            request_type=understanding.request_type,
            entities=understanding.entities,
            document_info=document_info or "No document provided"
        )

        # Store interaction
        self.memory.append({
            "timestamp": datetime.now().isoformat(),
            "request": user_request,
            "response": result.helpful_response,
            "type": understanding.request_type
        })

        return dspy.Prediction(
            response=result.helpful_response,
            action_suggestions=result.action_suggestions,
            request_type=understanding.request_type,
            memory_size=len(self.memory)
        )

# Complete Document Processing Pipeline
class DocumentProcessingPipeline(dspy.Module):
    """Complete pipeline for document processing and analysis."""

    def __init__(self, categories):
        super().__init__()
        self.categories = categories
        self.classifier = DocumentClassifier(categories)
        self.extractor = DocumentEntityExtractor()
        self.qa_system = DocumentQA()
        self.agent = DocumentAssistantAgent()
        self.processed_documents = []

    def forward(self, document_text, user_query=None):
        start_time = time.time()

        # Step 1: Classify document
        classification = self.classifier(document_text=document_text)

        # Step 2: Extract entities
        extraction = self.extractor(document_text=document_text)

        # Step 3: Process user query if provided
        qa_result = None
        if user_query:
            qa_result = self.qa_system(
                question=user_query,
                document_context=document_text
            )

        # Step 4: Generate assistance
        assistance = self.agent(
            user_request=user_query or "Analyze this document",
            document_info=f"Type: {classification.category}, Entities: {extraction.entities}"
        )

        # Compile results
        processing_time = time.time() - start_time

        document_summary = {
            "id": len(self.processed_documents),
            "category": classification.category,
            "confidence": classification.confidence,
            "entities": extraction.entities,
            "relationships": extraction.relationships,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }

        self.processed_documents.append(document_summary)

        return dspy.Prediction(
            document_summary=document_summary,
            classification=classification,
            extraction=extraction,
            qa_result=qa_result,
            assistance=assistance.response,
            action_suggestions=assistance.action_suggestions,
            total_documents_processed=len(self.processed_documents)
        )

    def get_statistics(self):
        """Get processing statistics."""
        if not self.processed_documents:
            return "No documents processed yet"

        stats = {
            "total_documents": len(self.processed_documents),
            "category_distribution": {},
            "average_confidence": 0,
            "total_processing_time": 0,
            "average_processing_time": 0,
            "total_entities_extracted": 0
        }

        for doc in self.processed_documents:
            # Category distribution
            cat = doc["category"]
            stats["category_distribution"][cat] = stats["category_distribution"].get(cat, 0) + 1

            # Confidence
            stats["average_confidence"] += doc["confidence"]

            # Processing time
            stats["total_processing_time"] += doc["processing_time"]

            # Entities
            stats["total_entities_extracted"] += len(doc["entities"].split(", ")) if doc["entities"] else 0

        stats["average_confidence"] /= len(self.processed_documents)
        stats["average_processing_time"] = stats["total_processing_time"] / len(self.processed_documents)

        return stats

# Example Documents
sample_documents = [
    {
        "text": """
        ACME Corporation announces merger with TechStart Inc.
        The $500 million deal was finalized on June 15, 2024.
        CEO John Smith stated that this merger will create significant value for shareholders.
        The combined company will have over 10,000 employees worldwide.
        """,
        "expected_category": "Business News",
        "query": "What is the value of the merger?"
    },
    {
        "text": """
        Research Study: Machine Learning in Healthcare
        Authors: Dr. Sarah Johnson, Dr. Michael Chen
        Published: May 2024 in Nature Medicine
        Abstract: Our study demonstrates 94% accuracy in early disease detection using ML algorithms.
        The research involved 50,000 patients across 20 hospitals.
        Funding provided by National Institute of Health (NIH) Grant #12345.
        """,
        "expected_category": "Research Paper",
        "query": "What was the accuracy of the ML algorithm?"
    },
    {
        "text": """
        SERVICE AGREEMENT
        Between: CloudTech Solutions (Provider) and Global Retail Inc (Client)
        Effective Date: January 1, 2024

        Terms:
        1. Provider agrees to maintain 99.9% uptime
        2. Client agrees to pay $50,000 monthly
        3. Contract duration: 3 years
        4. Termination requires 90-day notice

        Signed:
        Provider: Jane Doe, CEO
        Client: Robert Wilson, CTO
        """,
        "expected_category": "Legal Contract",
        "query": "What is the monthly payment amount?"
    },
    {
        "text": """
        Product Review: SmartWatch Pro 2024
        Rating: 4.5/5 stars
        Review by: Tech Enthusiast Magazine

        The SmartWatch Pro 2024 offers impressive features:
        - Heart rate monitoring with 98% accuracy
        - 7-day battery life
        - Water resistance up to 50 meters
        - Price: $399

        Minor issues: Limited app compatibility
        Overall: Excellent value for money
        """,
        "expected_category": "Product Review",
        "query": "What is the battery life?"
    },
    {
        "text": """
        EMPLOYEE HANDBOOK UPDATE
        Company: InnovateTech Solutions
        Date: June 2024

        New Policies:
        1. Remote work policy: 3 days office, 2 days remote
        2. Health benefits: Additional mental health coverage
        3. Professional development: $2000 annual budget
        4. Holiday schedule: 15 days PTO + 10 company holidays

        Contact HR with questions: hr@innovatetech.com
        """,
        "expected_category": "HR Document",
        "query": "How many vacation days do employees get?"
    }
]

# Demo Functions
def demo_document_classification():
    """Demonstrate document classification."""
    print("\n" + "=" * 60)
    print("Document Classification Demo")
    print("=" * 60)

    categories = ["Business News", "Research Paper", "Legal Contract", "Product Review", "HR Document"]
    classifier = DocumentClassifier(categories)

    print("\nClassification Results:")
    for doc in sample_documents:
        result = classifier(document_text=doc["text"])
        status = "✓" if result.category == doc["expected_category"] else "✗"
        print(f"{status} Expected: {doc['expected_category']}, Got: {result.category} ({result.confidence:.2f})")

def demo_entity_extraction():
    """Demonstrate entity extraction."""
    print("\n" + "=" * 60)
    print("Entity Extraction Demo")
    print("=" * 60)

    extractor = DocumentEntityExtractor()

    for doc in sample_documents[:2]:  # Show first 2 examples
        print(f"\nDocument: {doc['expected_category']}")
        result = extractor(document_text=doc["text"])
        print(f"Entities: {result.entities}")
        print(f"Relationships: {result.relationships}")

def demo_document_qa():
    """Demonstrate document Q&A."""
    print("\n" + "=" * 60)
    print("Document Q&A Demo")
    print("=" * 60)

    qa = DocumentQA()

    for doc in sample_documents[:3]:  # Test first 3 documents
        print(f"\nDocument: {doc['expected_category'][:30]}...")
        print(f"Question: {doc['query']}")
        result = qa(
            question=doc['query'],
            document_context=doc['text']
        )
        print(f"Answer: {result.answer}")
        print(f"Confidence: {result.confidence:.2f}")

def demo_agent_interaction():
    """Demonstrate agent interaction."""
    print("\n" + "=" * 60)
    print("Document Assistant Agent Demo")
    print("=" * 60)

    agent = DocumentAssistantAgent()

    requests = [
        "Summarize this contract for me",
        "What should I be aware of in this research paper?",
        "Can you explain the key terms in this agreement?"
    ]

    for request in requests:
        print(f"\nUser: {request}")
        result = agent(user_request=request)
        print(f"Agent: {result.response}")
        print(f"Type: {result.request_type}")

def demo_complete_pipeline():
    """Demonstrate complete processing pipeline."""
    print("\n" + "=" * 60)
    print("Complete Pipeline Demo")
    print("=" * 60)

    categories = ["Business News", "Research Paper", "Legal Contract", "Product Review", "HR Document"]
    pipeline = DocumentProcessingPipeline(categories)

    print("\nProcessing Documents:")
    for i, doc in enumerate(sample_documents, 1):
        print(f"\n--- Document {i} ---")
        result = pipeline(document_text=doc["text"], user_query=doc["query"])

        print(f"Category: {result.document_summary['category']}")
        print(f"Confidence: {result.document_summary['confidence']:.2f}")
        print(f"Entities: {result.document_summary['entities'][:50]}...")
        if result.qa_result:
            print(f"Q&A: {result.qa_result.answer}")
        print(f"Processing Time: {result.document_summary['processing_time']:.3f}s")

    # Show statistics
    stats = pipeline.get_statistics()
    print("\n" + "=" * 40)
    print("Processing Statistics")
    print("=" * 40)
    print(json.dumps(stats, indent=2))

def demo_batch_processing():
    """Demonstrate batch processing with optimization."""
    print("\n" + "=" * 60)
    print("Batch Processing with Optimization")
    print("=" * 60)

    # Create optimized pipeline
    categories = ["Business News", "Research Paper", "Legal Contract", "Product Review", "HR Document"]

    # Training data for optimization
    trainset = [
        dspy.Example(
            document_text="Apple reports quarterly earnings of $100 billion",
            category="Business News",
            confidence=0.9
        ),
        # ... more training examples
    ]

    # Optimize classifier
    def classification_metric(example, pred, trace=None):
        correct = example.category == pred.category
        confidence_match = abs(example.confidence - pred.confidence) < 0.2
        return correct and confidence_match

    optimizer = BootstrapFewShot(
        metric=classification_metric,
        max_bootstrapped_demos=3
    )

    print("\nOptimizing pipeline...")
    # In practice, you would optimize the actual components
    # optimized_pipeline = optimizer.compile(pipeline, trainset=trainset)
    optimized_pipeline = DocumentProcessingPipeline(categories)

    # Process all documents
    print("Batch processing all documents...")
    start_time = time.time()

    results = []
    for doc in sample_documents:
        result = optimized_pipeline(document_text=doc["text"])
        results.append(result)

    total_time = time.time() - start_time

    print(f"\nBatch Processing Complete:")
    print(f"- Documents processed: {len(results)}")
    print(f"- Total time: {total_time:.2f}s")
    print(f"- Average time per document: {total_time/len(results):.3f}s")
    print(f"- Documents per second: {len(results)/total_time:.1f}")

# Main execution
def run_complete_demo():
    """Run the complete application demo."""
    print("DSPy Complete Application Demo")
    print("Document Processing and Analysis System\n")

    try:
        demo_document_classification()
        demo_entity_extraction()
        demo_document_qa()
        demo_agent_interaction()
        demo_complete_pipeline()
        demo_batch_processing()

    except Exception as e:
        print(f"\nError running demo: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Complete application demo finished!")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("✓ Document classification")
    print("✓ Entity extraction")
    print("✓ RAG-based Q&A")
    print("✓ Intelligent agent assistance")
    print("✓ Complete pipeline integration")
    print("✓ Performance optimization")
    print("✓ Batch processing")
    print("✓ Statistics and monitoring")

if __name__ == "__main__":
    run_complete_demo()
# Chapter 6 Exercises: Building Real-World Applications

## Overview

These exercises provide hands-on practice building complete, production-ready applications with DSPy. You'll work with all the concepts learned in previous chapters—signatures, modules, evaluation, and optimization—to solve real-world problems.

## Exercise 1: Build a Customer Support RAG System

### Objective
Create a complete RAG system for customer support that can answer questions about product documentation and policies.

### Problem
You need to build a customer support bot that can answer questions based on a knowledge base of product documentation, FAQs, and support policies. The system should provide accurate answers with sources.

### Starter Code
```python
import dspy
from dspy.teleprompter import BootstrapFewShot
from typing import List, Dict, Any

class CustomerSupportRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=5)
        # TODO: Add appropriate modules for answering questions

    def forward(self, question):
        # TODO: Implement RAG pipeline
        pass

# Sample knowledge base
knowledge_base = [
    "Product returns must be initiated within 30 days of purchase.",
    "Free shipping is available for orders over $50.",
    "Customer support is available 24/7 via phone and chat.",
    "Warranty covers manufacturing defects for 1 year.",
    "Account deletion requires email verification."
]

# TODO: Implement this function
def create_support_rag(knowledge_base):
    """Create and optimize a customer support RAG system."""
    pass
```

### Tasks
1. Complete the CustomerSupportRAG class implementation
2. Add modules for question understanding and answer generation
3. Implement the create_support_rag function
4. Create training data for optimization
5. Optimize the system using BootstrapFewShot
6. Test with support-related questions

### Hints
- Use ChainOfThought for complex reasoning about policies
- Include confidence scores in predictions
- Consider using MIPRO for better optimization
- Add source citations to answers

### Expected Output
```
Question: "What is the return policy?"
Answer: "Product returns must be initiated within 30 days of purchase."
Sources: [Document 1: "Product returns must be initiated..."]
Confidence: 0.95
```

---

## Exercise 2: Multi-hop Research Assistant

### Objective
Build a system that can answer complex research questions by gathering information from multiple sources and connecting the dots.

### Problem
Create a research assistant that can answer questions requiring information synthesis from multiple documents, such as "What are the relationships between AI companies and their founders?"

### Starter Code
```python
import dspy
from dspy.teleprompter import MIPRO

class ResearchAssistant(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=5)
        # TODO: Add modules for multi-hop reasoning

    def forward(self, research_question):
        # TODO: Implement multi-hop search and synthesis
        pass

# TODO: Implement this function
def create_research_assistant():
    """Create a multi-hop research assistant."""
    pass

# TODO: Implement this function
def evaluate_research_quality(question, answer, ground_truth):
    """Evaluate the quality of research answers."""
    pass
```

### Tasks
1. Implement multi-hop search logic
2. Add modules for connecting information across documents
3. Create a comprehensive evaluation metric
4. Optimize with MIPRO for complex reasoning
5. Test with multi-hop questions
6. Track reasoning paths

### Hints
- Track visited entities to avoid loops
- Use entity extraction to identify relationships
- Implement a maximum hop limit to prevent infinite searching
- Consider using graphs to represent relationships

### Expected Output
```
Research Question: "How are Google's founders connected to other tech companies?"
Research Path:
1. Founders: Larry Page, Sergey Brin
2. Education: Stanford University
3. Connections: Other Stanford alumni in tech
4. Investments: Alphabet portfolio companies

Answer: "Google's founders Larry Page and Sergey Brin met at Stanford..."
Sources: [5 documents, 12 entities, 3 hops]
```

---

## Exercise 3: Multi-label Document Classifier

### Objective
Build a sophisticated classifier that can assign multiple labels to documents based on their content.

### Problem
News articles often cover multiple topics (e.g., technology, business, international). Build a classifier that can identify all relevant topics for each document.

### Starter Code
```python
import dspy
from typing import List

class MultiLabelClassifier(dspy.Module):
    def __init__(self, possible_labels):
        super().__init__()
        self.possible_labels = possible_labels
        # TODO: Add modules for multi-label classification

    def forward(self, document):
        # TODO: Implement multi-label classification
        pass

# Possible labels
labels = [
    "Technology", "Business", "Politics", "Health", "Science",
    "Sports", "Entertainment", "International", "Finance", "Education"
]

# TODO: Implement this function
def train_multilabel_classifier(trainset):
    """Train a multi-label classifier."""
    pass
```

### Tasks
1. Implement multi-label classification logic
2. Handle label dependencies (some labels co-occur)
3. Create appropriate training data
4. Implement evaluation metrics for multi-label
5. Optimize with appropriate DSPy optimizer
6. Test on real news headlines

### Hints
- Use threshold-based prediction for multiple labels
- Consider label correlations during prediction
- Precision and recall are important for multi-label
- F1-score needs micro and macro averaging

### Expected Output
```
Document: "Apple announces new AI features and stock rises"
Predicted Labels:
- Technology (0.95 confidence)
- Business (0.88 confidence)
- Finance (0.72 confidence)

Evaluation: Micro-F1: 0.85, Macro-F1: 0.78
```

---

## Exercise 4: Contract Information Extractor

### Objective
Build an entity extraction system specifically designed for legal contracts that can identify key terms, parties, dates, and obligations.

### Problem
Legal contracts contain critical information that needs to be extracted and organized. Build a system that can parse contracts and extract structured information.

### Starter Code
```python
import dspy
from typing import Dict, List

class ContractExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        # TODO: Add modules for contract analysis

    def forward(self, contract_text):
        # TODO: Implement contract information extraction
        pass

# TODO: Implement this function
def extract_contract_entities(contract_text):
    """Extract structured information from a contract."""
    pass

# TODO: Implement this function
def validate_extraction(extracted_info, contract_text):
    """Validate extracted information against original text."""
    pass
```

### Tasks
1. Identify contract-specific entity types
2. Implement extraction for parties, dates, amounts, obligations
3. Add validation to ensure extracted info is accurate
4. Create a relationship extractor for contract clauses
5. Handle different contract types (employment, sales, NDAs)
6. Generate summary of key terms

### Hints
- Legal language has specific patterns and terminology
- Dates and amounts have specific formats in contracts
- Parties often have defined terms (e.g., "Client", "Provider")
- Obligations are often expressed as conditional statements

### Expected Output
```
Contract Analysis:
Parties:
- Provider: TechCorp Inc.
- Client: Global Solutions Ltd.

Key Dates:
- Effective Date: January 1, 2024
- Termination: 12 months after effective date

Financial Terms:
- Payment: $50,000 per month
- Penalty: 10% for late payment

Obligations:
- Provider: Deliver software updates quarterly
- Client: Provide feedback within 14 days
```

---

## Exercise 5: Autonomous Customer Service Agent

### Objective
Create an intelligent agent that can handle customer service interactions from start to finish, including understanding requests, taking actions, and learning from feedback.

### Problem
Build a customer service agent that can handle various types of requests (billing, technical support, general inquiries) and escalate when necessary.

### Starter Code
```python
import dspy
from dspy.teleprompter import BootstrapFewShot

class CustomerServiceAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        # TODO: Add modules for perception, decision, action

    def forward(self, customer_message, session_context=None):
        # TODO: Implement complete agent workflow
        pass

# TODO: Implement this function
def create_agent_session():
    """Initialize an agent session with memory."""
    pass

# TODO: Implement this function
def handle_conversation(agent, conversation):
    """Process a complete conversation with the agent."""
    pass
```

### Tasks
1. Implement intent classification for customer messages
2. Add modules for handling different types of requests
3. Implement escalation logic for complex issues
4. Add memory to maintain conversation context
5. Include satisfaction measurement
6. Optimize with real conversation data

### Hints
- Sentiment analysis helps prioritize urgent issues
- Track resolution time for performance metrics
- Maintain consistency in responses
- Learn from successful resolutions

### Expected Output
```
Session ID: 12345
Customer: "My order hasn't arrived yet"
Agent Intent: Order Inquiry
Action Taken: Checked order status, provided tracking
Resolution: Found package delayed, expedited shipping
Customer Satisfaction: Positive
Time to Resolution: 3 minutes
```

---

## Exercise 6: Code Review Assistant

### Objective
Build an automated code review assistant that can analyze code for bugs, security issues, style violations, and suggest improvements.

### Problem
Code reviews are time-consuming but essential. Build an assistant that can automatically identify common issues and suggest improvements.

### Starter Code
```python
import dspy
from typing import Dict, List

class CodeReviewAssistant(dspy.Module):
    def __init__(self):
        super().__init__()
        # TODO: Add modules for code analysis

    def forward(self, code, language="python"):
        # TODO: Implement comprehensive code review
        pass

# TODO: Implement this function
def analyze_code_quality(code):
    """Analyze code for various quality aspects."""
    pass

# TODO: Implement this function
def suggest_improvements(code, issues_found):
    """Suggest specific improvements for identified issues."""
    pass
```

### Tasks
1. Implement analysis for different code quality aspects
2. Detect common bugs and anti-patterns
3. Check for security vulnerabilities
4. Ensure adherence to coding standards
5. Suggest specific improvements
6. Generate a comprehensive review report

### Hints
- Different languages have different patterns and issues
- Consider cyclomatic complexity for code complexity
- Check for common security issues (SQL injection, XSS)
- Style guides vary by project

### Expected Output
```
Code Review Report:
File: utils.py

Issues Found:
1. Security: SQL injection vulnerability in query (Line 45)
2. Performance: O(n²) complexity in loop (Line 67)
3. Style: Line too long (Line 23, 120 characters)
4. Bug: Unhandled exception in try block (Line 89)

Improvements Suggested:
- Use parameterized queries for database access
- Consider using a set for O(1) lookup
- Break long line into multiple lines
- Add specific exception handling

Overall Score: 7/10
```

---

## Exercise 7: Integrated Multi-Application System

### Objective
Combine multiple applications from this chapter into a comprehensive system that can handle complex, real-world scenarios.

### Problem
Create a document processing pipeline that can classify documents, extract entities, answer questions about them, and generate reports.

### Starter Code
```python
import dspy
from typing import Dict, Any

class DocumentProcessingPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        # TODO: Initialize all components

    def forward(self, document):
        # TODO: Implement complete processing pipeline
        pass

# TODO: Implement this function
def create_pipeline():
    """Create and configure the complete pipeline."""
    pass

# TODO: Implement this function
def process_document_collection(pipeline, documents):
    """Process a collection of documents through the pipeline."""
    pass
```

### Tasks
1. Integrate classifier, extractor, and RAG components
2. Add document preprocessing and cleaning
3. Implement cross-component communication
4. Create a unified evaluation framework
5. Add caching for efficiency
6. Generate comprehensive reports

### Hints
- Components should share context and results
- Optimize each component before integration
- Consider bottlenecks in the pipeline
- Batch processing can improve efficiency

### Expected Output
```
Processing Report:
Documents Processed: 100
Classification Accuracy: 92%
Entity Extraction F1: 0.89
QA System Accuracy: 85%

Processing Time: 12.3 seconds
Average per Document: 0.123 seconds

Top Categories Identified:
- Contracts (35%)
- Invoices (28%)
- Reports (22%)
- Others (15%)

Most Common Entities:
- Dates: 1,234 extracted
- Organizations: 456 extracted
- Monetary Values: 234 extracted
```

---

## Exercise Solutions Approach

After completing these exercises, you'll have:

1. **Complete Applications**: Six production-ready applications
2. **Integration Experience**: Understanding how to combine DSPy components
3. **Optimization Skills**: Experience with different optimizers
4. **Evaluation Expertise**: Comprehensive metrics for different tasks
5. **Real-World Readiness**: Systems that handle complexity and edge cases

### Key Learning Points

- Start simple and iterate
- Validate each component before integration
- Use appropriate evaluation metrics for each task
- Optimize based on specific requirements
- Consider performance and scalability
- Handle errors gracefully

### Advanced Challenges

1. Add web interfaces to your applications
- Implement real-time processing
- Add support for multiple languages
- Create deployment configurations
- Add monitoring and logging
- Implement A/B testing for different approaches

Good luck building your DSPy applications! These exercises will give you hands-on experience with all the concepts learned throughout this book.
# Chapter 2: Signatures - Exercise Problems

This document contains all exercises for Chapter 2 on DSPy Signatures. Each exercise builds on concepts from the chapter and helps you practice signature design and implementation.

---

## Exercise 1: Basic Signature Creation

### Objective
Create basic string-based signatures for common NLP tasks.

### Tasks

#### Task 1.1: Simple Question Answering
Create a signature for answering questions about a given text. The signature should include the question and relevant context as inputs, and return an answer as output.

**Requirements:**
- Use clear, descriptive field names
- Follow the proper `input -> output` format
- Consider what additional context might be helpful

#### Task 1.2: Text Classification
Create a signature for classifying text into categories. The signature should include both the classification and a confidence score as outputs.

**Requirements:**
- Support multiple category options
- Include confidence scoring
- Consider edge cases (unknown categories)

#### Task 1.3: Text Transformation
Create a signature for transforming informal text to formal text. Consider what additional context might make the transformation more accurate.

**Requirements:**
- Handle various formality levels
- Consider domain-specific transformations
- Include metadata about the transformation

### Expected Output Format
```python
# Your answers here
qa_signature = "________________________________________"

classification_signature = "________________________________________"

transformation_signature = "________________________________________"

# Explanation of design choices
explanation = """
Explain why you structured each signature this way:
- What trade-offs did you consider?
- Why did you include certain fields?
- How would each signature handle edge cases?
"""
```

---

## Exercise 2: Typed Signatures

### Objective
Convert string signatures to typed signatures with proper field definitions.

### Tasks

#### Task 2.1: Convert Basic to Typed
Convert this basic signature to a typed signature class:
```
"customer_review, product_category -> sentiment_score, key_points"
```

**Requirements:**
- Use `dspy.Signature` as the base class
- Add appropriate field types
- Include descriptive field descriptions
- Add helpful prefixes for prompting

#### Task 2.2: Create Email Processing Signature
Create a comprehensive signature for processing incoming emails.

**Requirements:**
- Input: email text, sender information, priority level
- Output: category, response needed, urgency, action items
- Use appropriate types for all fields
- Include at least 2 optional fields

#### Task 2.3: Add Validation
Create a validation function that checks a signature class for:
- At least 2 input fields and 2 output fields
- All fields have descriptions
- Proper type annotations

### Expected Output Format
```python
import dspy
from typing import List, Dict, Optional, Union

class CustomerReviewAnalyzer(dspy.Signature):
    """Analyze customer reviews for sentiment and key points."""

    # TODO: Add input fields
    # TODO: Add output fields
    pass

class EmailProcessor(dspy.Signature):
    """Process and categorize incoming emails."""

    # TODO: Implement complete signature
    pass

def validate_signature(signature_class):
    """Validate a signature class."""
    # TODO: Implement validation logic
    pass
```

---

## Exercise 3: Complex Multi-Field Signatures

### Objective
Design and implement complex signatures with multiple interconnected fields.

### Scenario
You're building a document analysis system for a legal firm that needs to:
1. Extract key information from legal documents
2. Identify risks and obligations
3. Generate summaries for different stakeholders
4. Suggest amendments

### Tasks

#### Task 3.1: Design Complete Signature
Create a comprehensive signature for legal document analysis.

**Requirements:**
- Include at least 4 input fields
- Include at least 6 output fields
- Use appropriate types for all fields
- Include helpful descriptions

#### Task 3.2: Create Helper Signatures
Break down the complex task into smaller, reusable signatures:
- Clause extraction signature
- Risk assessment signature
- Amendment suggestion signature

#### Task 3.3: Demonstrate Composition
Show how smaller signatures can be composed into the main one. Create a function that demonstrates the workflow.

### Expected Output Format
```python
class LegalDocumentAnalyzer(dspy.Signature):
    """Comprehensive legal document analysis."""

    # TODO: Define input fields
    # TODO: Define output fields
    pass

class ClauseExtractor(dspy.Signature):
    """Extract and categorize legal clauses."""

    # TODO: Implement
    pass

def analyze_document_workflow(document_text, document_type, jurisdiction):
    """Demonstrate signature composition."""

    # TODO: Show how to chain signatures
    pass
```

---

## Exercise 4: Domain-Specific Signatures

### Objective
Create specialized signatures for a specific domain of your choice.

### Choose ONE Domain:
1. **Healthcare**: Patient triage and diagnosis assistance
2. **Finance**: Investment analysis and recommendation
3. **Education**: Personalized learning path generation
4. **E-commerce**: Product recommendation engine
5. **Customer Support**: Ticket classification and response

### Tasks

#### Task 4.1: Define Domain Context
- Choose your domain
- Describe the specific use case
- Identify key requirements

#### Task 4.2: Create Primary Signature
- Design signature for your domain
- Include at least 4 inputs and 4 outputs
- Use domain-appropriate terminology

#### Task 4.3: Create Supporting Signatures
- Create at least 2 helper signatures
- Show how they support the main task

#### Task 4.4: Create Usage Example
- Provide realistic example inputs
- Describe expected outputs
- Show how signatures work together

### Expected Output Format
```python
# Domain definition
DOMAIN = "your_chosen_domain"
DOMAIN_DESCRIPTION = """
Describe your domain and use case here
"""

# Primary signature
class DomainSignature(dspy.Signature):
    """Primary signature for your domain."""

    # TODO: Implement
    pass

# Supporting signatures
class SupportingSignature1(dspy.Signature):
    """Supporting signature 1."""
    # TODO: Implement
    pass

def demonstrate_usage():
    """Show how your signatures work."""

    # TODO: Provide example
    pass
```

---

## Exercise 5: Signature Refactoring

### Objective
Improve an existing poorly designed signature.

### Problem Signature
```python
class BadSignature(dspy.Signature):
    """Poorly designed signature that needs improvement."""

    data = dspy.InputField()
    info = dspy.InputField()

    result = dspy.OutputField()
    other = dspy.OutputField()
```

### Tasks

#### Task 5.1: Identify Problems
List all issues with the BadSignature:
- Vague field names
- Missing context
- Unclear outputs
- No type information

#### Task 5.2: Refactor for Specific Use Case
Assume this is for analyzing customer feedback. Create an improved version:
- Be specific about inputs and outputs
- Add proper types, descriptions, and prefixes
- Handle edge cases and errors

#### Task 5.3: Create Unit Tests
Write test cases for your improved signature:
- Test with valid data
- Test edge cases
- Verify output structure

#### Task 5.4: Document Improvements
Explain how your refactored version addresses the original problems.

### Expected Output Format
```python
# Problems identified
problems_with_bad_signature = [
    "List each problem here"
]

# Refactored signature
class ImprovedCustomerFeedbackAnalyzer(dspy.Signature):
    """Improved signature for analyzing customer feedback."""

    # TODO: Implement improved version
    pass

# Unit tests
def test_improved_signature():
    """Test the improved signature."""

    # TODO: Create test cases
    pass

# Improvements documentation
improvements_made = """
1. First improvement explained
2. Second improvement explained
...etc
"""
```

---

## Exercise 6: Comprehensive Project

### Objective
Build a complete signature-based system for a real-world scenario.

### Scenario
You're building an AI-powered assistant for job seekers that:
1. Analyzes job descriptions
2. Matches them with user profiles
3. Identifies skill gaps
4. Suggests improvements to resumes
5. Generates application materials

### Tasks

#### Task 6.1: Design System Architecture
List all signatures needed for this system and how they connect.

#### Task 6.2: Implement Core Signatures
Create 5 core signatures:
- JobDescriptionAnalyzer
- UserProfileMatcher
- SkillGapIdentifier
- ResumeImprover
- ApplicationMaterialGenerator

#### Task 6.3: Create Main Workflow
Implement a main class that orchestrates all signatures:
- Chain signatures together
- Handle errors and edge cases
- Provide meaningful outputs

#### Task 6.4: Design Evaluation Metrics
Create metrics to evaluate system performance:
- Accuracy of job analysis
- Quality of matches
- Usefulness of suggestions
- Overall user satisfaction

#### Task 6.5: Extension Tasks (Optional)
Add advanced features:
- Caching for repeated analyses
- User preferences and personalization
- Analytics to track performance
- Feedback loop for improvement

### Expected Output Format
```python
# System architecture
system_architecture = """
Describe your system architecture here
"""

# Core signatures
class JobDescriptionAnalyzer(dspy.Signature):
    """Analyze job descriptions to extract requirements."""
    # TODO: Implement
    pass

# ... other signatures

# Main workflow class
class JobSeekerAssistant:
    """Main system that orchestrates all signatures."""

    def __init__(self):
        # TODO: Initialize all signature modules
        pass

    def process_job_application(self, job_description, user_profile, user_resume):
        """Process a complete job application."""
        # TODO: Implement workflow
        pass

# Evaluation metrics
def evaluate_system_performance(test_cases):
    """Evaluate the complete system."""
    # TODO: Define and implement metrics
    pass
```

---

## Submission Guidelines

### What to Submit
1. Complete implementations for all chosen exercises
2. Clear explanations of design decisions
3. Comments explaining complex parts
4. Test cases demonstrating functionality

### How to Submit
1. Create a new directory: `exercises/chapter02/your_username/`
2. Create separate files for each exercise
3. Include a README.md with explanations
4. Ensure all code is runnable

### Evaluation Criteria
- **Correctness**: Does the code work as expected?
- **Clarity**: Is the code well-documented and easy to understand?
- **Completeness**: Are all requirements met?
- **Best Practices**: Does the code follow DSPy best practices?
- **Creativity**: Are the solutions thoughtful and well-designed?

### Getting Help
- Review the chapter content
- Check the examples in `examples/chapter02/`
- Look at the provided solutions
- Ask questions in the community forum

Remember: The goal is not just to complete the exercises, but to understand the concepts deeply. Take your time, experiment, and learn!
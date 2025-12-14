# Chapter 2 Exercises

## Prerequisites

- **Chapter 2 Content**: Complete understanding of all signature concepts
- **Required Knowledge**: Basic Python programming, understanding of data types
- **Difficulty Level**: Intermediate
- **Estimated Time**: 2-3 hours

## Exercise Overview

This chapter includes 6 hands-on exercises to practice working with DSPy signatures. Each exercise builds on concepts from the chapter:

1. **Basic Signature Creation** - Practice fundamental signature syntax
2. **Typed Signatures** - Work with field types and descriptions
3. **Complex Multi-Field Signatures** - Handle multiple inputs and outputs
4. **Domain-Specific Signatures** - Create signatures for real-world applications
5. **Signature Refactoring** - Improve and optimize existing signatures
6. **Comprehensive Project** - Build a complete signature-based system

---

## Exercise 1: Basic Signature Creation

### Objective
Create basic string-based signatures for common NLP tasks.

### Instructions

1. **Simple Question Answering**
   - Create a signature for answering questions about a given text
   - Use clear, descriptive field names
   - Follow the proper `input -> output` format

2. **Text Classification**
   - Create a signature for classifying text into categories
   - Include both the classification and confidence score as outputs

3. **Text Transformation**
   - Create a signature for transforming informal text to formal text
   - Consider what additional context might be helpful

### Tasks

```python
# Task 1: Create a QA signature
# Your answer should be in the format: "input_fields -> output_fields"

qa_signature = "________________________________________"


# Task 2: Create a classification signature
classification_signature = "________________________________________"


# Task 3: Create a transformation signature
transformation_signature = "________________________________________"


# Task 4: Explain your design choices
# Why did you structure each signature this way?
# What trade-offs did you consider?

explanation = """
________________________________________
________________________________________
________________________________________
"""
```

### Validation Questions
- Does each signature clearly separate inputs from outputs?
- Are the field names descriptive?
- Would another developer understand what each signature does?
- Are all necessary inputs included?

---

## Exercise 2: Typed Signatures

### Objective
Convert string signatures to typed signatures with proper field definitions.

### Instructions

Convert the following basic signatures to typed DSPy signature classes:

### Tasks

```python
import dspy
from typing import List, Dict, Optional, Union

# Task 1: Convert this basic signature to a typed class
# Basic: "customer_review, product_category -> sentiment_score, key_points"

class CustomerReviewAnalyzer(dspy.Signature):
    """Analyze customer reviews for sentiment and key points."""

    # TODO: Add input fields with proper types and descriptions

    # TODO: Add output fields with proper types and descriptions
    pass


# Task 2: Create a signature for email processing
# Requirements:
# - Input: email text, sender information, priority level
# - Output: category, response needed, urgency, action items

class EmailProcessor(dspy.Signature):
    """Process and categorize incoming emails."""

    # TODO: Implement the complete signature

    pass


# Task 3: Add field prefixes to improve prompting
# Modify the EmailProcessor to include helpful prefixes

class EnhancedEmailProcessor(EmailProcessor):
    """Enhanced email processor with better prompting."""

    # TODO: Redefine fields with helpful prefixes
    pass


# Task 4: Create a validation function
def validate_signature(signature_class):
    """Validate that a signature has required components."""
    # TODO: Implement validation logic
    # Check for at least 2 input fields and 2 output fields
    # Ensure all fields have descriptions
    pass
```

### Challenge
Add optional fields and default values to your typed signatures.

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

```python
# Task 1: Design the complete signature
# Include all necessary inputs and outputs for the document analysis system

class LegalDocumentAnalyzer(dspy.Signature):
    """Comprehensive legal document analysis and review."""

    # TODO: Define input fields
    # Consider: document text, document type, jurisdiction, review focus

    # TODO: Define output fields
    # Consider: executive summary, key clauses, risks, obligations, amendments

    pass


# Task 2: Create helper signatures for specific tasks
# Break down the complex task into smaller, reusable signatures

class ClauseExtractor(dspy.Signature):
    """Extract and categorize legal clauses from documents."""

    # TODO: Implement focused signature for clause extraction
    pass


class RiskAssessor(dspy.Signature):
    """Assess legal and financial risks in contracts."""

    # TODO: Implement focused signature for risk assessment
    pass


# Task 3: Demonstrate signature composition
# Show how smaller signatures can be composed into the main one

def analyze_document(document_text, document_type, jurisdiction):
    """Demonstrate how to use the composed signatures."""

    # TODO: Show how to chain multiple signatures
    # Use ClauseExtractor first, then RiskAssessor, then main analyzer

    pass
```

### Evaluation Criteria
- **Completeness**: All necessary inputs/outputs included
- **Modularity**: Can be broken into reusable components
- **Clarity**: Field names and descriptions are unambiguous
- **Flexibility**: Can handle different document types

---

## Exercise 4: Domain-Specific Signatures

### Objective
Create specialized signatures for a specific domain of your choice.

### Choose ONE domain:
1. **Healthcare**: Patient triage and diagnosis assistance
2. **Finance**: Investment analysis and recommendation
3. **Education**: Personalized learning path generation
4. **E-commerce**: Product recommendation engine
5. **Customer Support**: Ticket classification and response generation

### Tasks

```python
# Task 1: Define the domain context
DOMAIN = "____________________"  # Your chosen domain
DOMAIN_DESCRIPTION = """
________________________________________
________________________________________
________________________________________
"""

# Task 2: Create primary signature for your domain

class DomainSignature(dspy.Signature):
    """Primary signature for [Your Domain]."""

    # TODO: Implement domain-specific signature
    # Include at least 4 inputs and 4 outputs
    # Use appropriate types and descriptions

    pass


# Task 3: Create supporting signatures
# Create at least 2 helper signatures that support the main task

class SupportingSignature1(dspy.Signature):
    """Supporting signature 1."""
    # TODO: Implement
    pass

class SupportingSignature2(dspy.Signature):
    """Supporting signature 2."""
    # TODO: Implement
    pass


# Task 4: Create usage example
def demonstrate_usage():
    """Show how your signatures would be used in practice."""

    # TODO: Provide a realistic example
    # Include sample inputs and expected outputs

    example_inputs = {
        # TODO: Add example inputs
    }

    expected_outputs = {
        # TODO: Describe expected outputs
    }

    return example_inputs, expected_outputs
```

### Bonus
Add error handling and validation outputs to your domain signatures.

---

## Exercise 5: Signature Refactoring

### Objective
Improve an existing poorly designed signature.

### Problem Signature
```python
# This signature has multiple issues:
# - Vague field names
# - Missing context
# - Unclear outputs
# - No type information

class BadSignature(dspy.Signature):
    """Poorly designed signature that needs improvement."""

    data = dspy.InputField()
    info = dspy.InputField()

    result = dspy.OutputField()
    other = dspy.OutputField()
```

### Tasks

```python
# Task 1: Identify the problems
# List all issues with the BadSignature

problems_with_bad_signature = [
    "________________________________________",
    "________________________________________",
    "________________________________________",
    "________________________________________",
]

# Task 2: Refactor the signature
# Create an improved version based on a specific use case
# Assume this is for analyzing customer feedback

class ImprovedCustomerFeedbackAnalyzer(dspy.Signature):
    """Improved signature for analyzing customer feedback."""

    # TODO: Implement the improved signature
    # Be specific about inputs and outputs
    # Add proper types, descriptions, and prefixes

    pass


# Task 3: Create unit tests for your signature
def test_improved_signature():
    """Test the improved signature with sample data."""

    # TODO: Create test cases
    # Test with valid data
    # Test edge cases
    # Verify output structure

    test_cases = [
        # TODO: Add test cases
    ]

    return test_cases


# Task 4: Document the improvements
# Explain how your refactored version addresses the original problems

improvements_made = """
1. ________________________________________
2. ________________________________________
3. ________________________________________
4. ________________________________________
5. ________________________________________
"""
```

### Reflection
What principles did you apply during refactoring?

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

```python
# Task 1: Design the system architecture
# List all signatures needed for this system

system_signatures = [
    "1. JobDescriptionAnalyzer",
    "2. UserProfileMatcher",
    "3. SkillGapIdentifier",
    "4. ResumeImprover",
    "5. ApplicationMaterialGenerator"
]

# Task 2: Implement the core signatures

class JobDescriptionAnalyzer(dspy.Signature):
    """Analyze job descriptions to extract requirements and preferences."""

    # TODO: Implement
    pass

class UserProfileMatcher(dspy.Signature):
    """Match user profiles against job requirements."""

    # TODO: Implement
    pass

class SkillGapAnalyzer(dspy.Signature):
    """Identify gaps between user skills and job requirements."""

    # TODO: Implement
    pass

class ResumeImprover(dspy.Signature):
    """Suggest improvements to user's resume for specific job."""

    # TODO: Implement
    pass

class ApplicationMaterialGenerator(dspy.Signature):
    """Generate personalized application materials."""

    # TODO: Implement
    pass

# Task 3: Create the main workflow
class JobSeekerAssistant:
    """Main system that orchestrates all signatures."""

    def __init__(self):
        # TODO: Initialize all signature modules
        pass

    def process_job_application(self, job_description, user_profile, user_resume):
        """Process a complete job application."""

        # TODO: Implement the workflow
        # Chain signatures together
        # Handle errors and edge cases

        results = {
            "job_analysis": None,
            "match_score": None,
            "skill_gaps": None,
            "resume_improvements": None,
            "application_materials": None
        }

        return results

# Task 4: Create evaluation metrics
def evaluate_system_performance(test_cases):
    """Evaluate the complete system on test cases."""

    # TODO: Define evaluation metrics
    # - Accuracy of job analysis
    # - Quality of matches
    # - Usefulness of suggestions
    # - Overall user satisfaction

    metrics = {
        "accuracy": 0.0,
        "completeness": 0.0,
        "usefulness": 0.0,
        "user_satisfaction": 0.0
    }

    return metrics
```

### Extension Tasks (Optional)

1. **Add caching** for repeated analyses
2. **Implement user preferences** and personalization
3. **Create analytics** to track system performance
4. **Add support for multiple languages**
5. **Implement a feedback loop** for continuous improvement

---

## Solutions and Explanations

### Solution Guidelines

Solutions for these exercises are available in the `exercises/chapter02/solutions/` directory. Each solution includes:

1. **Complete implementation** of all tasks
2. **Explanation** of design choices
3. **Alternative approaches** and their trade-offs
4. **Common pitfalls** to avoid
5. **Extension ideas** for further practice

### Self-Assessment Checklist

For each exercise, check:
- ✅ All requirements are met
- ✅ Code is well-documented
- ✅ Signatures are reusable and modular
- ✅ Field names are descriptive
- ✅ Types and descriptions are appropriate
- ✅ Error handling is considered
- ✅ Performance implications are understood

### Further Practice

1. **Create your own domain-specific signatures** based on your interests
2. **Contribute to DSPy's signature library** with useful patterns
3. **Build a complete application** using only signatures
4. **Write tests** for signature-based systems
5. **Optimize signatures** for specific LLM providers

## Summary

These exercises cover:
- Basic signature syntax and structure
- Typed signatures with rich metadata
- Complex multi-field signatures
- Domain-specific design patterns
- Refactoring and improvement techniques
- Building complete signature-based systems

By completing these exercises, you've mastered the fundamentals of DSPy signatures and are ready to explore DSPy modules in Chapter 3.

## Next Steps

- Check your solutions against the provided answers
- Experiment with different signature designs
- Practice creating signatures for your own use cases
- Proceed to Chapter 3: Modules to learn how to use signatures with DSPy modules

## Resources

- [Solution Code](../../exercises/chapter02/solutions/) - Complete implementations
- [DSPy Documentation](https://dspy-docs.vercel.app/) - Official documentation
- [Community Forum](https://github.com/stanfordnlp/dspy/discussions) - Ask questions and share ideas
- [Example Gallery](../05-practical-examples.md) - More real-world examples
# Typed Signatures

## Prerequisites

- **Previous Section**: [Signature Syntax](./02-signature-syntax.md) - Understanding of basic signature syntax
- **Required Knowledge**: Familiarity with data types and programming type systems
- **Difficulty Level**: Intermediate to Advanced
- **Estimated Reading Time**: 35 minutes

## Learning Objectives

By the end of this section, you will:
- Understand the benefits of typed signatures in DSPy
- Learn how to define fields with types and descriptions
- Master the creation of structured, type-safe signatures
- Be able to validate and constrain signature inputs/outputs

## What are Typed Signatures?

Typed signatures extend basic DSPy signatures with:
- **Type information** - Specifying expected data types
- **Field descriptions** - Adding documentation for each field
- **Constraints** - Defining valid value ranges or formats
- **Validation rules** - Ensuring data quality and consistency
 
 Typed signatures transform simple string signatures into rich, self-documenting specifications that provide better type safety, validation, and developer experience.
 
 > **How DSPy Uses Types**: DSPy uses your type definitions in two ways:
 > 1. **Prompting**: It translates types into natural language instructions for the LM (e.g., `int` becomes "an integer").
 > 2. **Validation**: When using modules like `TypedPredictor`, it enforces these types at runtime, retrying if the LM fails to match the schema.


## Creating Typed Signatures

### Basic Typed Signature Class

```python
import dspy

class QuestionAnswering(dspy.Signature):
    """Answer questions based on provided context."""

    question = dspy.InputField(desc="The question to be answered")
    context = dspy.InputField(desc="Background information relevant to the question")
    answer = dspy.OutputField(desc="A comprehensive answer to the question")
    confidence = dspy.OutputField(desc="Confidence score from 0 to 1", type=float)
```

### Field Types and Descriptions

DSPy provides several field types:

```python
class DocumentAnalysis(dspy.Signature):
    """Analyze documents for key information."""

    # Input fields with types and descriptions
    document_text = dspy.InputField(
        desc="The full text of the document to analyze",
        type=str,
        prefix="Document: "
    )

    analysis_type = dspy.InputField(
        desc="Type of analysis to perform (e.g., 'sentiment', 'topics', 'entities')",
        type=str,
        prefix="Analysis Type: "
    )

    # Output fields with constraints
    summary = dspy.OutputField(
        desc="Brief summary of the document (max 200 words)",
        type=str,
        prefix="Summary: "
    )

    key_points = dspy.OutputField(
        desc="List of main points extracted from the document",
        type=list,
        prefix="Key Points: "
    )

    sentiment_score = dspy.OutputField(
        desc="Sentiment analysis score from -1 (negative) to 1 (positive)",
        type=float,
        prefix="Sentiment Score: "
    )
```

## Field Options

### 1. Type Specification

```python
class CustomerSupport(dspy.Signature):
    """Process customer support tickets."""

    ticket_id = dspy.InputField(type=str)           # String
    urgency_level = dspy.InputField(type=int)       # Integer
    is_premium_customer = dspy.InputField(type=bool) # Boolean
    issue_tags = dspy.InputField(type=list)         # List
    metadata = dspy.InputField(type=dict)           # Dictionary

    resolution_time = dspy.OutputField(type=float)  # Float
    resolution_steps = dspy.OutputField(type=list)  # List
    success_flag = dspy.OutputField(type=bool)      # Boolean
```

### 2. Descriptions and Documentation

```python
class FinancialAnalysis(dspy.Signature):
    """Analyze financial data and generate insights."""

    revenue_data = dspy.InputField(
        desc="Monthly revenue figures for the past 24 months",
        prefix="Revenue Data: "
    )

    expense_categories = dspy.InputField(
        desc="Breakdown of expenses by category (e.g., salaries, marketing, operations)",
        prefix="Expense Categories: "
    )

    growth_forecast = dspy.OutputField(
        desc="Predicted growth rate for next 6 quarters with assumptions",
        prefix="Growth Forecast: "
    )

    risk_factors = dspy.OutputField(
        desc="List of potential risks and their impact assessment",
        prefix="Risk Assessment: "
    )
```

### 3. Prefix and Formatting

```python
class ReportGenerator(dspy.Signature):
    """Generate various types of business reports."""

    data_source = dspy.InputField(
        desc="Source of data for the report",
        prefix="📊 Data Source: "
    )

    report_type = dspy.InputField(
        desc="Type of report to generate (e.g., 'weekly', 'monthly', 'quarterly')",
        prefix="📋 Report Type: "
    )

    executive_summary = dspy.OutputField(
        desc="Brief overview for executives (2-3 paragraphs)",
        prefix="🎯 Executive Summary:\n"
    )

    detailed_analysis = dspy.OutputField(
        desc="In-depth analysis with supporting data",
        prefix="📈 Detailed Analysis:\n"
    )
```

## Complex Typed Signatures

### Nested Structures

```python
class ProjectPlanning(dspy.Signature):
    """Create detailed project plans with milestones and resources."""

    project_requirements = dspy.InputField(
        desc="Detailed requirements and scope of the project",
        type=str
    )

    timeline = dspy.InputField(
        desc="Desired timeline and key dates",
        type=str
    )

    budget = dspy.InputField(
        desc="Available budget and financial constraints",
        type=float
    )

    project_plan = dspy.OutputField(
        desc="Comprehensive project plan with phases",
        type=dict,
        prefix="Project Plan: "
    )

    milestones = dspy.OutputField(
        desc="List of key milestones with dates and dependencies",
        type=list,
        prefix="Milestones: "
    )

    resource_allocation = dspy.OutputField(
        desc="Required resources and assignment strategy",
        type=dict,
        prefix="Resource Allocation: "
    )

    risk_assessment = dspy.OutputField(
        desc="Potential risks and mitigation strategies",
        type=dict,
        prefix="Risk Assessment: "
    )
```

### Conditional Fields

```python
class MedicalDiagnosis(dspy.Signature):
    """Assist in medical diagnosis based on symptoms and history."""

    patient_symptoms = dspy.InputField(
        desc="List of current symptoms and their duration",
        type=str
    )

    medical_history = dspy.InputField(
        desc="Patient's relevant medical history",
        type=str
    )

    vital_signs = dspy.InputField(
        desc="Recent vital signs measurements",
        type=dict
    )

    preliminary_diagnosis = dspy.OutputField(
        desc="Most likely diagnoses with confidence scores",
        type=list,
        prefix="Preliminary Diagnosis: "
    )

    recommended_tests = dspy.OutputField(
        desc="Medical tests to confirm diagnosis",
        type=list,
        prefix="Recommended Tests: "
    )

    urgency_level = dspy.OutputField(
        desc="Urgency of medical attention (1-5 scale)",
        type=int,
        prefix="Urgency Level: "
    )

    follow_up_plan = dspy.OutputField(
        desc="Recommended follow-up actions and timeline",
        type=str,
        prefix="Follow-up Plan: "
    )
```

## Using Typed Signatures

### Creating Modules with Typed Signatures

```python
import dspy

# Create a module with a typed signature
analyzer = dspy.Predict(DocumentAnalysis)

# Use the module
result = analyzer(
    document_text="The company reported strong quarterly earnings...",
    analysis_type="financial"
)

# Access typed results
print(f"Summary: {result.summary}")
print(f"Sentiment: {result.sentiment_score}")
print(f"Key Points: {result.key_points}")
```

### Chain of Typed Operations

```python
# Define multiple typed signatures
class TextExtraction(dspy.Signature):
    """Extract key information from text."""
    raw_text = dspy.InputField(desc="Raw text to process", type=str)
    entities = dspy.OutputField(desc="Named entities found", type=list)
    topics = dspy.OutputField(desc="Main topics covered", type=list)

class Summarization(dspy.Signature):
    """Create structured summaries."""
    original_text = dspy.InputField(desc="Text to summarize", type=str)
    summary_length = dspy.InputField(desc="Desired summary length", type=str)
    executive_summary = dspy.OutputField(desc="Brief overview", type=str)
    key_insights = dspy.OutputField(desc="Main insights", type=list)

# Chain them together
extractor = dspy.Predict(TextExtraction)
summarizer = dspy.Predict(Summarization)

# Process text through the chain
extracted = extractor(raw_text=document_text)
summary = summarizer(
    original_text=document_text,
    summary_length="brief"
)
```

## Validation and Constraints

### Custom Field Validation

```python
from typing import Literal, Optional
import dspy

class SentimentAnalysis(dspy.Signature):
    """Analyze sentiment with strict output constraints."""

    text_to_analyze = dspy.InputField(
        desc="Text to analyze for sentiment",
        type=str
    )

    sentiment_label = dspy.OutputField(
        desc="Sentiment classification",
        type=Literal["positive", "negative", "neutral"]
    )

    confidence_score = dspy.OutputField(
        desc="Confidence in classification",
        type=float,
        # Additional validation in practice
        # validator=lambda x: 0 <= x <= 1
    )

    emotional_indicators = dspy.OutputField(
        desc="Emotions detected in the text",
        type=list,
        prefix="Emotions: "
    )
```

### Default Values and Optional Fields

```python
class TaskManagement(dspy.Signature):
    """Manage tasks with priorities and deadlines."""

    task_title = dspy.InputField(
        desc="Title of the task",
        type=str
    )

    task_description = dspy.InputField(
        desc="Detailed description of the task",
        type=str
    )

    priority = dspy.InputField(
        desc="Priority level (1-5, where 5 is highest)",
        type=int,
        default=3
    )

    due_date = dspy.InputField(
        desc="Due date for task completion",
        type=str,
        optional=True
    )

    estimated_hours = dspy.OutputField(
        desc="Estimated time to complete",
        type=float
    )

    suggested_breakdown = dspy.OutputField(
        desc="Suggested subtasks",
        type=list,
        optional=True
    )
```

## Advanced Type Features

### Enumerated Types

```python
from enum import Enum

class DocumentType(str, Enum):
    CONTRACT = "contract"
    INVOICE = "invoice"
    REPORT = "report"
    EMAIL = "email"
    MANUAL = "manual"

class DocumentProcessor(dspy.Signature):
    """Process different types of documents appropriately."""

    document_content = dspy.InputField(
        desc="Content of the document",
        type=str
    )

    document_type = dspy.InputField(
        desc="Type of document being processed",
        type=DocumentType
    )

    processed_content = dspy.OutputField(
        desc="Processed document content",
        type=str
    )

    extracted_fields = dspy.OutputField(
        desc="Fields extracted based on document type",
        type=dict
    )
```

### Union Types

```python
from typing import Union

class FlexibleAnalyzer(dspy.Signature):
    """Analyze data that can come in different formats."""

    input_data = dspy.InputField(
        desc="Data to analyze (can be text, JSON, or list)",
        type=Union[str, list, dict]
    )

    analysis_type = dspy.InputField(
        desc="Type of analysis to perform",
        type=str
    )

    analysis_result = dspy.OutputField(
        desc="Result of the analysis",
        type=Union[str, dict, list]
    )
```

## Best Practices for Typed Signatures

### 1. Be Specific with Types

```python
# Too generic
class DataProcessor(dspy.Signature):
    data = dspy.InputField(type=object)
    result = dspy.OutputField(type=object)

# Specific and helpful
class DataProcessor(dspy.Signature):
    customer_data = dspy.InputField(
        desc="Customer information including name, email, and purchase history",
        type=dict
    )
    personalized_offer = dspy.OutputField(
        desc="Tailored offer based on customer profile",
        type=dict
    )
```

### 2. Use Descriptions as Documentation

```python
# Clear documentation
class CodeReviewer(dspy.Signature):
    """Review code for quality, security, and best practices."""

    code_snippet = dspy.InputField(
        desc="Code to review (include comments if available)",
        type=str
    )

    programming_language = dspy.InputField(
        desc="Language/framework the code is written in",
        type=str
    )

    review_comments = dspy.OutputField(
        desc="Specific feedback on code quality and improvements",
        type=str
    )

    security_issues = dspy.OutputField(
        desc="List of potential security vulnerabilities found",
        type=list
    )

    style_score = dspy.OutputField(
        desc="Code style rating from 1-10",
        type=int
    )
```

### 3. Structure Output for Easy Processing

```python
# Machine-readable output
class DataExtractor(dspy.Signature):
    """Extract structured data from unstructured text."""

    unstructured_text = dspy.InputField(
        desc="Text containing embedded data",
        type=str
    )

    extraction_schema = dspy.InputField(
        desc="Schema defining what data to extract",
        type=dict
    )

    extracted_data = dspy.OutputField(
        desc="Extracted data matching the schema",
        type=dict
    )

    extraction_confidence = dspy.OutputField(
        desc="Confidence score for each extracted field",
        type=dict
    )

    unextractable_sections = dspy.OutputField(
        desc="Text sections that couldn't be parsed",
        type=list
    )
```

### 4. Use Prefixes for Better Prompting

```python
class MeetingSummarizer(dspy.Signature):
    """Create comprehensive meeting summaries."""

    meeting_transcript = dspy.InputField(
        desc="Full transcript of the meeting",
        prefix="📝 Meeting Transcript:\n",
        type=str
    )

    participant_list = dspy.InputField(
        desc="List of meeting attendees",
        prefix="👥 Participants: ",
        type=str
    )

    executive_summary = dspy.OutputField(
        desc="Brief summary for busy executives",
        prefix="🎯 Executive Summary:\n",
        type=str
    )

    action_items = dspy.OutputField(
        desc="Decisions and next steps with owners",
        prefix="✅ Action Items:\n",
        type=list
    )
```

## Error Handling with Typed Signatures

```python
class RobustProcessor(dspy.Signature):
    """Process data with comprehensive error handling."""

    input_data = dspy.InputField(
        desc="Data to process",
        type=str
    )

    processing_result = dspy.OutputField(
        desc="Successful processing result",
        type=str,
        optional=True
    )

    error_message = dspy.OutputField(
        desc="Description of any errors encountered",
        type=str,
        optional=True
    )

    success_flag = dspy.OutputField(
        desc="True if processing succeeded",
        type=bool
    )

    fallback_result = dspy.OutputField(
        desc="Alternative result if main processing fails",
        type=str,
        optional=True
    )
```

## Summary

Typed signatures provide:
- **Type Safety**: Clear specification of expected data types
- **Documentation**: Self-documenting field descriptions
- **Validation**: Built-in structure and constraint validation
- **Developer Experience**: Better IDE support and autocomplete
- **Maintainability**: Easier to understand and modify complex signatures

Key advantages:
1. **Explicit contracts** between inputs and outputs
2. **Rich metadata** for each field
3. **Type checking** and validation capabilities
4. **Better prompting** through prefixes and formatting
5. **Easier debugging** with clear field definitions

Typed signatures transform DSPy from a simple prompting tool into a structured, type-safe framework for building reliable LLM applications.

## Key Takeaways

1. **Typed signatures add structure** - Define fields with types, descriptions, and constraints
2. **Use InputField/OutputField** - Specialized field classes with rich options
3. **Include descriptions** - They serve as documentation and help the model
4. **Leverage type hints** - They provide validation and improve developer experience
5. **Structure outputs** - Design them for easy programmatic consumption

## Further Reading

- [Next Section: Advanced Signatures](./04-advanced-signatures.md) - Complex signature patterns
- [DSPy Module Documentation](https://dspy-docs.vercel.app/docs/modules) - Using signatures with modules
- [Practical Examples](./05-practical-examples.md) - Real-world typed signature applications
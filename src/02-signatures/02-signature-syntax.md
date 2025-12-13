# Signature Syntax

## Prerequisites

- **Previous Section**: [Understanding Signatures](./01-understanding-signatures.md) - Grasp of signature concepts
- **Required Knowledge**: Basic understanding of string formatting and data types
- **Difficulty Level**: Intermediate
- **Estimated Reading Time**: 30 minutes

## Learning Objectives

By the end of this section, you will:
- Master the basic string-based signature syntax in DSPy
- Understand how to structure complex multi-field signatures
- Learn best practices for naming and formatting signatures
- Be able to write signatures for various task types

## Basic Signature Syntax

DSPy signatures use a simple, intuitive string format:
```
input_fields -> output_fields
```

The arrow (`->`) separates inputs from outputs, clearly showing the transformation.

### Simple Examples

```python
# Basic question answering
"question -> answer"

# Text summarization
"long_document -> short_summary"

# Translation
"source_text, target_language -> translated_text"
```

## Field Separation

Fields are separated by commas. Use descriptive names that clearly indicate the field's purpose.

### Input Fields
```python
# Single input
"question -> answer"

# Multiple inputs
"question, context -> answer"

# Many inputs with clear names
"customer_email, company_policy, urgency_level -> response, escalation_needed"
```

### Output Fields
```python
# Single output
"text -> sentiment"

# Multiple outputs
"review -> sentiment, key_points, overall_rating"

# Structured outputs
"meeting_notes -> action_items, decisions, follow_up_tasks, attendees"
```

## Naming Conventions

### 1. Use Descriptive Names
```python
# Poor naming
"a, b, c -> d, e"

# Good naming
"product_description, customer_segment, price_point -> recommended_marketing_message, target_audience_fit"
```

### 2. Be Consistent
```python
# Inconsistent
"text, doc, article -> summary, brief"

# Consistent
"document_text -> document_summary"
"meeting_transcript -> meeting_summary"
"email_thread -> email_summary"
```

### 3. Use Underscores for Multi-word Names
```python
# Camel case (less readable)
"customerFeedback -> analysisResult"

# Underscores (recommended)
"customer_feedback -> analysis_result"
```

### 4. Include Units or Types When Helpful
```python
"temperature_celsius, humidity_percent -> comfort_level, hvac_recommendation"
```

## Complex Signatures

### Multiple Inputs and Outputs

```python
# Comprehensive analysis
"financial_report, fiscal_year, industry_type -> revenue_growth, profit_margins, risk_factors, investment_recommendation"

# Content processing
"raw_article, target_audience, desired_tone -> processed_article, readability_score, engagement_prediction"
```

### Conditional Logic in Signatures

While signatures don't contain conditional logic, they can imply it through field design:

```python
# Implies conditional processing
"support_ticket, customer_tier, issue_type -> resolution_steps, estimated_time, escalation_required, customer_satisfaction_prediction"
```

## Common Patterns

### 1. Transformation Pattern
```
input -> output
```
```python
"raw_data -> cleaned_data"
"informal_text -> formal_text"
"technical_specification -> user_friendly_description"
```

### 2. Analysis Pattern
```
data -> analysis, metadata
```
```python
"product_review -> sentiment_score, key_aspects, recommendation_strength"
"sales_data -> trend_analysis, seasonal_patterns, forecast"
```

### 3. Generation Pattern
```
requirements, constraints -> generated_content, validation"
```
```python
"topic, audience, length -> blog_post, seo_score, readability_metrics"
"requirements, tech_stack -> implementation_plan, estimated_effort, risk_assessment"
```

### 4. Extraction Pattern
```
source_document -> extracted_field1, extracted_field2, extracted_field3"
```
```python
"invoice_document -> vendor_name, invoice_number, total_amount, due_date"
"job_description -> required_skills, experience_level, salary_range, company_benefits"
```

## Signature Examples by Domain

### Customer Service
```python
"customer_complaint, product_info -> resolution_steps, apology_template, compensation_suggestion"

"support_ticket, customer_history, issue_type -> solution_recommendation, estimated_resolution_time, satisfaction_prediction"
```

### Healthcare
```python
"patient_symptoms, medical_history -> possible_conditions, urgency_level, recommended_tests"

"clinical_notes, research_guidelines -> treatment_plan, success_probability, alternative_options"
```

### Finance
```python
"market_data, risk_tolerance -> investment_portfolio, expected_return, risk_assessment"

"financial_statement, accounting_standards -> revenue_recognition, compliance_status, red_flags"
```

### Legal
```python
"contract_document, jurisdiction -> key_clauses, potential_risks, amendment_suggestions"

"case_law, legal_question -> relevant_precedents, success_probability, argument_strategy"
```

### Education
```python
"student_essay, rubric -> grade, feedback_points, improvement_suggestions"

"lesson_plan, student_level -> learning_objectives, assessment_methods, differentiation_strategies"
```

## Advanced Syntax Features

### Field Descriptions (Inline Documentation)

For complex signatures, you can add inline documentation:

```python
# With inline descriptions
"question, context[provided_background] -> answer[concise_response], confidence[score_0_to_1]"

# Multiple descriptive fields
"meeting_transcript, attendees_list, meeting_date -> action_items[with_owners_and_due_dates], decisions[with_reasoning], follow_up_email_draft"
```

### Type Hints (Informal)

While DSPy doesn't enforce types, you can include them as documentation:

```python
"customer_feedback:string[int], sentiment:label -> category:label, priority:number[1-5]"
```

## Signature Validation

DSPy provides built-in validation for signatures:

### Valid Signatures
```python
# Valid: Clear input/output separation
"text -> summary"

# Valid: Multiple fields
"article, author, publication_date -> abstract, key_findings, citation_format"

# Valid: Descriptive names
"product_features, customer_needs -> value_proposition, competitive_advantages"
```

### Invalid Signatures
```python
# Invalid: No arrow separator
"text summary"

# Invalid: Multiple arrows
"input -> intermediate -> output"

# Invalid: Empty inputs or outputs
" -> output"
"input -> "
```

## Best Practices

### 1. Start Simple, Expand as Needed
```python
# Start with this
"question -> answer"

# Then expand if needed
"question, context -> answer, confidence, sources"
```

### 2. Include All Necessary Context
```python
# Missing context
"email -> response"

# Better with context
"customer_email, previous_interactions, company_policy -> personalized_response, escalation_needed"
```

### 3. Think About Output Usage
```python
# If you need structured data
"interview_transcript -> key_skills, experience_years, cultural_fit_score, recommendation"

# If you need natural language
"interview_transcript -> candidate_assessment_summary"
```

### 4. Consider the Model's Perspective
Design signatures that make it clear what the model should produce:

```python
# Ambiguous
"data -> analysis"

# Clear
"sales_data_monthly -> trend_analysis_growth_percentage, top_performing_products, seasonality_notes"
```

### 5. Use Consistent Terminology
```python
# Confusing mix
"doc -> abstract, summary, brief"

# Consistent terminology
"article -> executive_summary, main_points, conclusion"
```

## Debugging Signatures

### Common Issues

1. **Vague Field Names**
   ```python
   # Problem
   "info -> result"

   # Solution
   "customer_review_text -> sentiment_analysis_score"
   ```

2. **Missing Required Context**
   ```python
   # Problem
   "question -> answer"

   # Solution
   "question, domain_knowledge, answer_length -> answer, confidence"
   ```

3. **Too Many Outputs**
   ```python
   # Problem: Overly complex
   "document -> summary, sentiment, entities, topics, language, quality, recommendations, actions"

   # Solution: Split into multiple signatures
   "document -> summary, main_topics"
   "document -> sentiment, emotional_tone"
   "document -> named_entities, relationships"
   ```

### Testing Your Signature

Before implementing, ask:
- Does each input have a clear purpose?
- Is each output necessary and distinct?
- Would another developer understand this?
- Can I create test cases for this signature?

## Signatures in Practice

### Using Signatures with DSPy Modules

```python
import dspy

# Define a signature
qa_signature = "question, context -> answer, confidence"

# Use it with a module
qa_module = dspy.Predict(qa_signature)

# Call with structured data
result = qa_module(
    question="What is the capital of France?",
    context="European geography, countries and capitals"
)
```

### Chaining Signatures

```python
# Define a pipeline
summarizer = dspy.Predict("document -> summary")
analyzer = dspy.Predict("summary -> key_insights")

# Chain them
doc_summary = summarizer(document=document_text)
insights = analyzer(summary=doc_summary.summary)
```

## Summary

Signature syntax in DSPy is:
- **Simple**: Uses clear `input -> output` format
- **Flexible**: Supports multiple fields and complex transformations
- **Expressive**: Can represent sophisticated AI tasks
- **Composable**: Enables building complex workflows

Key syntax rules:
- Use `->` to separate inputs from outputs
- Separate fields with commas
- Use descriptive, consistent naming
- Include all necessary context
- Keep signatures focused and testable

In the next section, we'll explore typed signatures that add rich metadata and constraints to our signatures.

## Key Takeaways

1. **Syntax is simple but powerful**: `input1, input2 -> output1, output2`
2. **Naming matters**: Clear, descriptive names prevent ambiguity
3. **Include context**: All relevant inputs should be specified
4. **Think about outputs**: Structure them for easy consumption
5. **Test your signatures**: Ensure they're unambiguous and complete

## Further Reading

- [Next Section: Typed Signatures](./03-typed-signatures.md) - Adding type information and constraints
- [Practical Examples](./05-practical-examples.md) - See signatures in real-world applications
- [Chapter 3: Modules](../03-modules/) - Using signatures with DSPy modules
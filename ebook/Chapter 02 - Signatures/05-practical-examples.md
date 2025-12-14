# Practical Examples

## Prerequisites

- **Previous Section**: [Advanced Signatures](./04-advanced-signatures.md) - Understanding of advanced signature patterns
- **Required Knowledge**: Understanding of real-world use cases and domain requirements
- **Difficulty Level**: Intermediate to Advanced
- **Estimated Reading Time**: 45 minutes

## Learning Objectives

By the end of this section, you will see:
- Real-world signature implementations across multiple domains
- How signatures solve practical business problems
- Best practices for designing signatures for specific use cases
- Performance considerations and optimization patterns

## Example 1: Customer Support Automation

### Basic Ticket Classification

```python
import dspy
from typing import List, Dict, Optional

class TicketClassifier(dspy.Signature):
    """Classify customer support tickets automatically."""

    ticket_text = dspy.InputField(
        desc="Full text of the customer support ticket",
        type=str,
        prefix="📄 Ticket Text:\n"
    )

    customer_info = dspy.InputField(
        desc="Customer information (tier, history, etc.)",
        type=dict,
        prefix="👤 Customer Info:\n"
    )

    category = dspy.OutputField(
        desc="Primary category of the ticket",
        type=str,
        prefix="🏷️ Category: "
    )

    urgency = dspy.OutputField(
        desc="Urgency level (1-5, 5 being highest)",
        type=int,
        prefix="⚡ Urgency: "
    )

    suggested_response_type = dspy.OutputField(
        desc="Type of response needed",
        type=str,
        prefix="💬 Response Type: "
    )

    escalation_needed = dspy.OutputField(
        desc="Whether escalation to senior support is needed",
        type=bool,
        prefix="🚨 Escalation: "
    )

# Usage
classifier = dspy.Predict(TicketClassifier)

result = classifier(
    ticket_text="My order #12345 hasn't arrived and it's been 2 weeks. The tracking shows it's still at the warehouse.",
    customer_info={"tier": "premium", "previous_issues": 2, "account_age": "2 years"}
)
```

### Advanced Support Workflow

```python
class SupportWorkflow(dspy.Signature):
    """Complete support ticket processing workflow."""

    ticket_data = dspy.InputField(
        desc="Complete ticket information including history",
        type=Dict[str, Union[str, int, List[str]]]
    )

    knowledge_base = dspy.InputField(
        desc="Relevant knowledge base articles",
        type=List[Dict[str, str]],
        optional=True
    )

    # Stage 1: Analysis
    ticket_analysis = dspy.OutputField(
        desc="Detailed analysis of the ticket",
        type=Dict[str, Union[str, int, List[str]]],
        prefix="🔍 Ticket Analysis:\n"
    )

    # Stage 2: Solution Generation
    solution_suggestions = dspy.OutputField(
        desc="Possible solutions to the customer's problem",
        type=List[Dict[str, Union[str, float, str]]],
        prefix="💡 Solutions:\n"
    )

    # Stage 3: Response Generation
    personalized_response = dspy.OutputField(
        desc="Personalized response for the customer",
        type=str,
        prefix="✉️ Response:\n"
    )

    # Stage 4: Internal Actions
    internal_actions = dspy.OutputField(
        desc="Actions needed from support team",
        type=List[Dict[str, Union[str, bool, Dict[str, str]]]],
        prefix="⚙️ Internal Actions:\n"
    )

    # Stage 5: Follow-up
    follow_up_plan = dspy.OutputField(
        desc="Follow-up actions and timing",
        type=Dict[str, Union[str, int, List[str]]],
        prefix="📅 Follow-up:\n"
    )
```

## Example 2: Healthcare Clinical Decision Support

### Symptom Analysis

```python
class SymptomAnalyzer(dspy.Signature):
    """Analyze patient symptoms and suggest possible conditions."""

    patient_symptoms = dspy.InputField(
        desc="List of current symptoms with duration",
        type=List[str],
        prefix="🩺 Symptoms:\n"
    )

    patient_demographics = dspy.InputField(
        desc="Age, gender, and relevant demographic information",
        type=Dict[str, Union[str, int]],
        prefix="👤 Demographics:\n"
    )

    medical_history = dspy.InputField(
        desc="Relevant past medical conditions",
        type=str,
        prefix="📋 Medical History:\n"
    )

    vital_signs = dspy.InputField(
        desc="Current vital signs",
        type=Dict[str, Union[int, float, str]],
        prefix="📊 Vital Signs:\n"
    )

    possible_conditions = dspy.OutputField(
        desc="Possible conditions with probability scores",
        type=List[Dict[str, Union[str, float, List[str]]]],
        prefix="🔍 Possible Conditions:\n"
    )

    recommended_tests = dspy.OutputField(
        desc="Diagnostic tests to consider",
        type=List[Dict[str, Union[str, str, bool]]],
        prefix="🧪 Recommended Tests:\n"
    )

    urgency_level = dspy.OutputField(
        desc="Medical urgency assessment",
        type=str,
        prefix="🚨 Urgency Level: "
    )

    differential_diagnosis = dspy.OutputField(
        desc="Differential diagnosis reasoning",
        type=str,
        prefix="🤔 Differential Diagnosis:\n"
    )
```

### Treatment Planner

```python
class TreatmentPlanner(dspy.Signature):
    """Generate treatment plans based on diagnosis."""

    confirmed_diagnosis = dspy.InputField(
        desc="Confirmed medical diagnosis",
        type=str,
        prefix="🏥 Diagnosis:\n"
    )

    patient_profile = dspy.InputField(
        desc="Complete patient profile including allergies and preferences",
        type=Dict[str, Union[str, List[str], Dict[str, str]]]
    )

    treatment_guidelines = dspy.InputField(
        desc="Medical treatment guidelines for the condition",
        type=str,
        prefix="📚 Guidelines:\n"
    )

    primary_treatment = dspy.OutputField(
        desc="Primary treatment recommendation",
        type=Dict[str, Union[str, List[str], int, Dict[str, str]]],
        prefix="💊 Primary Treatment:\n"
    )

    alternative_treatments = dspy.OutputField(
        desc="Alternative treatment options",
        type=List[Dict[str, Union[str, List[str], float]]],
        prefix="🔄 Alternatives:\n"
    )

    monitoring_plan = dspy.OutputField(
        desc="Monitoring and follow-up plan",
        type=Dict[str, Union[str, List[str], int]],
        prefix="📈 Monitoring Plan:\n"
    )

    lifestyle_recommendations = dspy.OutputField(
        desc="Lifestyle and self-care recommendations",
        type=List[str],
        prefix="🏃 Lifestyle:\n"
    )

    contraindications = dspy.OutputField(
        desc="Treatments to avoid and why",
        type=List[Dict[str, str]],
        prefix="⚠️ Contraindications:\n"
    )
```

## Example 3: Financial Document Analysis

### Contract Risk Analysis

```python
class ContractRiskAnalyzer(dspy.Signature):
    """Analyze contracts for financial and legal risks."""

    contract_text = dspy.InputField(
        desc="Full text of the contract",
        type=str,
        prefix="📄 Contract Text:\n"
    )

    contract_type = dspy.InputField(
        desc="Type of contract (e.g., loan, lease, service)",
        type=str,
        prefix="📋 Contract Type: "
    )

    party_information = dspy.InputField(
        desc="Information about all parties involved",
        type=Dict[str, str],
        prefix="🏢 Parties:\n"
    )

    risk_assessment = dspy.OutputField(
        desc="Overall risk assessment",
        type=Dict[str, Union[str, int, float, List[str]]],
        prefix="⚠️ Risk Assessment:\n"
    )

    key_obligations = dspy.OutputField(
        desc="Key obligations and liabilities",
        type=List[Dict[str, Union[str, int, float]]],
        prefix="📝 Key Obligations:\n"
    )

    problematic_clauses = dspy.OutputField(
        desc="Potentially problematic clauses with explanations",
        type=List[Dict[str, Union[str, int, str]]],
        prefix="⚡ Problematic Clauses:\n"
    )

    negotiation_points = dspy.OutputField(
        desc "Suggested points for negotiation",
        type=List[Dict[str, str]],
        prefix="💼 Negotiation Points:\n"
    )

    compliance_requirements = dspy.OutputField(
        desc="Regulatory compliance requirements",
        type=List[Dict[str, Union[str, List[str]]]],
        prefix="🔒 Compliance:\n"
    )
```

### Investment Analysis

```python
class InvestmentAnalyzer(dspy.Signature):
    """Analyze investment opportunities and risks."""

    company_data = dspy.InputField(
        desc="Company financial and operational data",
        type=Dict[str, Union[str, int, float, List[str]]]
    )

    market_conditions = dspy.InputField(
        desc="Current market and economic conditions",
        type=Dict[str, Union[str, float, List[str]]]
    )

    investment_amount = dspy.InputField(
        desc="Proposed investment amount",
        type=float,
        prefix="💰 Investment Amount: "
    )

    investment_analysis = dspy.OutputField(
        desc="Comprehensive investment analysis",
        type=Dict[str, Union[str, float, int, List[str]]],
        prefix="📊 Analysis:\n"
    )

    risk_factors = dspy.OutputField(
        desc="Key risk factors and mitigation strategies",
        type=List[Dict[str, Union[str, int, List[str]]]],
        prefix="⚠️ Risk Factors:\n"
    )

    projected_returns = dspy.OutputField(
        desc="Projected returns under different scenarios",
        type=Dict[str, Union[float, List[float], Dict[str, float]]],
        prefix="📈 Projected Returns:\n"
    )

    investment_recommendation = dspy.OutputField(
        desc="Final investment recommendation with rationale",
        type=Dict[str, Union[str, int, float]],
        prefix="✅ Recommendation:\n"
    )

    exit_strategy = dspy.OutputField(
        desc="Potential exit strategies and timing",
        type=List[Dict[str, Union[str, int, float]]],
        prefix="🚪 Exit Strategy:\n"
    )
```

## Example 4: Legal Document Processing

### Contract Review System

```python
class ContractReviewer(dspy.Signature):
    """Review and analyze legal contracts."""

    contract_content = dspy.InputField(
        desc="Full contract text",
        type=str,
        prefix="📜 Contract:\n"
    )

    contract_category = dspy.InputField(
        desc="Category of contract (e.g., employment, vendor, partnership)",
        type=str,
        prefix="📂 Category: "
    )

    jurisdiction = dspy.InputField(
        desc="Governing jurisdiction",
        type=str,
        prefix="⚖️ Jurisdiction: "
    )

    review_focus = dspy.InputField(
        desc="Specific areas to focus review on",
        type=List[str],
        prefix="🎯 Focus Areas:\n"
    )

    executive_summary = dspy.OutputField(
        desc="Brief summary for non-legal stakeholders",
        type=str,
        prefix="📝 Executive Summary:\n"
    )

    key_terms = dspy.OutputField(
        desc="Important terms and their implications",
        type=List[Dict[str, Union[str, str, int]]],
        prefix="🔑 Key Terms:\n"
    )

    compliance_issues = dspy.OutputField(
        desc="Potential compliance and regulatory issues",
        type=List[Dict[str, Union[str, int, List[str]]]],
        prefix="🚨 Compliance Issues:\n"
    )

    amendments_suggested = dspy.OutputField(
        desc="Suggested amendments and changes",
        type=List[Dict[str, Union[str, str, str]]],
        prefix="✏️ Suggested Amendments:\n"
    )

    risk_rating = dspy.OutputField(
        desc="Overall risk rating and justification",
        type=Dict[str, Union[str, int, List[str]]],
        prefix="⚡ Risk Rating:\n"
    )

    next_steps = dspy.OutputField(
        desc="Recommended next steps",
        type=List[Dict[str, Union[str, bool, str]]],
        prefix="➡️ Next Steps:\n"
    )
```

## Example 5: Educational Content Generation

### Personalized Learning Path

```python
class LearningPathGenerator(dspy.Signature):
    """Generate personalized learning paths for students."""

    student_profile = dspy.InputField(
        desc="Student's learning profile, preferences, and history",
        type=Dict[str, Union[str, int, float, List[str]]],
        prefix="👨‍🎓 Student Profile:\n"
    )

    learning_objectives = dspy.InputField(
        desc="Learning objectives to achieve",
        type=List[str],
        prefix="🎯 Objectives:\n"
    )

    available_resources = dspy.InputField(
        desc="Available learning resources and materials",
        type=List[Dict[str, Union[str, int, float, List[str]]]],
        prefix="📚 Resources:\n"
    )

    time_constraints = dspy.InputField(
        desc="Available time and deadlines",
        type=Dict[str, Union[int, str, List[str]]],
        prefix="⏰ Time Constraints:\n"
    )

    learning_path = dspy.OutputField(
        desc="Personalized learning path with milestones",
        type=Dict[str, Union[str, List[Dict[str, Union[str, int, float]]]]],
        prefix="🛤️ Learning Path:\n"
    )

    recommended_materials = dspy.OutputField(
        desc="Recommended learning materials with priorities",
        type=List[Dict[str, Union[str, float, int, List[str]]]],
        prefix="📖 Materials:\n"
    )

    assessment_plan = dspy.OutputField(
        desc="Plan for assessing progress and mastery",
        type=Dict[str, Union[str, List[Dict[str, Union[str, int]]]]],
        prefix="📝 Assessment Plan:\n"
    )

    adaptations = dspy.OutputField(
        desc="Adaptations for different learning styles",
        type=List[Dict[str, Union[str, List[str]]]],
        prefix="🔄 Adaptations:\n"
    )

    motivation_strategies = dspy.OutputField(
        desc="Strategies to maintain student engagement",
        type=List[str],
        prefix="💪 Motivation:\n"
    )
```

### Quiz and Assessment Generator

```python
class AssessmentGenerator(dspy.Signature):
    """Generate quizzes and assessments for learning content."""

    subject_content = dspy.InputField(
        desc="Content to assess understanding of",
        type=str,
        prefix="📚 Content:\n"
    )

    assessment_type = dspy.InputField(
        desc="Type of assessment (quiz, exam, assignment)",
        type=str,
        prefix="📝 Type: "
    )

    difficulty_level = dspy.InputField(
        desc="Desired difficulty level (1-5)",
        type=int,
        prefix="📊 Difficulty: "
    )

    learning_objectives = dspy.InputField(
        desc="Specific learning objectives to test",
        type=List[str],
        prefix="🎯 Objectives:\n"
    )

    assessment_items = dspy.OutputField(
        desc="Generated assessment items",
        type=List[Dict[str, Union[str, List[str], Dict[str, Union[str, int, float]]]]],
        prefix="❓ Questions:\n"
    )

    rubric = dspy.OutputField(
        desc="Grading rubric for the assessment",
        type=Dict[str, Union[str, List[Dict[str, Union[str, int]]]]],
        prefix="📏 Rubric:\n"
    )

    time_allocation = dspy.OutputField(
        desc="Suggested time for each section",
        type=Dict[str, Union[int, float]],
        prefix="⏱️ Time Allocation:\n"
    )

    answer_key = dspy.OutputField(
        desc="Complete answer key with explanations",
        type=Dict[str, Union[str, List[str]]],
        prefix="🔑 Answer Key:\n"
    )
```

## Example 6: E-commerce Product Recommendations

### Product Recommendation Engine

```python
class ProductRecommender(dspy.Signature):
    """Generate personalized product recommendations."""

    customer_profile = dspy.InputField(
        desc="Customer's purchase history, preferences, and demographics",
        type=Dict[str, Union[str, int, float, List[str], List[Dict[str, Union[str, int, float]]]]],
        prefix="👤 Customer Profile:\n"
    )

    browsing_session = dspy.InputField(
        desc="Current browsing session data",
        type=Dict[str, Union[str, List[str], int, float]],
        prefix="🖥️ Current Session:\n"
    )

    inventory_data = dspy.InputField(
        desc="Available products with details",
        type=List[Dict[str, Union[str, float, int, List[str], Dict[str, Union[str, float]]]]],
        prefix="📦 Available Products:\n"
    )

    context = dspy.InputField(
        desc="Context (season, promotions, events)",
        type=Dict[str, Union[str, List[str], Dict[str, Union[str, float]]]],
        prefix="🌟 Context:\n"
    )

    recommendations = dspy.OutputField(
        desc="Personalized product recommendations",
        type=List[Dict[str, Union[str, float, int, List[str], Dict[str, Union[str, float]]]]],
        prefix="🎯 Recommendations:\n"
    )

    reasoning = dspy.OutputField(
        desc="Reasoning behind each recommendation",
        type=List[str],
        prefix="💭 Reasoning:\n"
    )

    cross_sell_opportunities = dspy.OutputField(
        desc="Cross-selling opportunities",
        type=List[Dict[str, Union[str, float, List[str]]]],
        prefix="🔄 Cross-sell:\n"
    )

    upsell_suggestions = dspy.OutputField(
        desc="Upsell suggestions with value proposition",
        type=List[Dict[str, Union[str, str, float]]],
        prefix="⬆️ Upsell:\n"
    )

    personalization_score = dspy.OutputField(
        desc="How personalized the recommendations are",
        type=float,
        prefix="🎨 Personalization Score: "
    )
```

## Example 7: Research Paper Analysis

### Literature Review Assistant

```python
class LiteratureAnalyzer(dspy.Signature):
    """Analyze research papers and generate insights."""

    paper_content = dspy.InputField(
        desc="Full text or abstract of research paper",
        type=str,
        prefix="📄 Paper Content:\n"
    )

    paper_metadata = dspy.InputField(
        desc="Paper metadata (authors, journal, year, etc.)",
        type=Dict[str, Union[str, int, List[str]]],
        prefix="📋 Metadata:\n"
    )

    research_area = dspy.InputField(
        desc="Research area and specific subfields",
        type=str,
        prefix="🔬 Research Area: "
    )

    analysis_depth = dspy.InputField(
        desc="Depth of analysis required (brief, detailed, comprehensive)",
        type=str,
        prefix="📊 Analysis Depth: "
    )

    key_contributions = dspy.OutputField(
        desc="Main contributions of the paper",
        type=List[Dict[str, Union[str, int, List[str]]]],
        prefix="💡 Key Contributions:\n"
    )

    methodology_summary = dspy.OutputField(
        desc="Summary of research methodology",
        type=str,
        prefix="🔧 Methodology:\n"
    )

    findings_and_results = dspy.OutputField(
        desc="Main findings and experimental results",
        type=Dict[str, Union[str, List[str], Dict[str, Union[str, float]]]],
        prefix="📈 Findings:\n"
    )

    limitations = dspy.OutputField(
        desc="Limitations and weaknesses identified",
        type=List[Dict[str, Union[str, str]]],
        prefix="⚠️ Limitations:\n"
    )

    future_research = dspy.OutputField(
        desc="Suggestions for future research directions",
        type=List[str],
        prefix="🔮 Future Research:\n"
    )

    related_works = dspy.OutputField(
        desc="Key related works and how this paper relates",
        type=List[Dict[str, Union[str, str, List[str]]]],
        prefix="📚 Related Works:\n"
    )

    novelty_assessment = dspy.OutputField(
        desc="Assessment of novelty and innovation",
        type=Dict[str, Union[str, int, float, List[str]]],
        prefix="✨ Novelty:\n"
    )
```

## Example 8: Software Code Review

### Code Quality Analyzer

```python
class CodeReviewer(dspy.Signature):
    """Review code for quality, security, and best practices."""

    code_snippet = dspy.InputField(
        desc="Code to review",
        type=str,
        prefix="💻 Code:\n"
    )

    programming_language = dspy.InputField(
        desc="Programming language and version",
        type=str,
        prefix="🔤 Language: "
    )

    code_context = dspy.InputField(
        desc "Context: what the code does and where it's used",
        type=str,
        prefix="📝 Context:\n"
    )

    review_criteria = dspy.InputField(
        desc="Specific criteria to focus on",
        type=List[str],
        prefix="🎯 Review Criteria:\n"
    )

    quality_assessment = dspy.OutputField(
        desc="Overall code quality assessment",
        type=Dict[str, Union[int, float, str, List[str]]],
        prefix="📊 Quality Assessment:\n"
    )

    security_issues = dspy.OutputField(
        desc="Security vulnerabilities and concerns",
        type=List[Dict[str, Union[str, int, List[str]]]],
        prefix="🔒 Security Issues:\n"
    )

    performance_considerations = dspy.OutputField(
        desc="Performance-related feedback",
        type=List[Dict[str, Union[str, str, List[str]]]],
        prefix="⚡ Performance:\n"
    )

    best_practices = dspy.OutputField(
        desc="Best practices adherence and improvements",
        type=List[Dict[str, Union[str, List[str]]]],
        prefix="✨ Best Practices:\n"
    )

    suggested_improvements = dspy.OutputField(
        desc="Specific code improvements with examples",
        type=List[Dict[str, Union[str, str, bool]]],
        prefix="🔧 Improvements:\n"
    )

    code_score = dspy.OutputField(
        desc="Overall code score (1-10)",
        type=int,
        prefix="📏 Score: "
    )

    learning_resources = dspy.OutputField(
        desc="Learning resources for improvements",
        type=List[Dict[str, Union[str, str]]],
        prefix="📚 Learning Resources:\n"
    )
```

## Performance Patterns

### Batch Processing for Efficiency

```python
class BatchTextProcessor(dspy.Signature):
    """Process multiple text items efficiently in batches."""

    text_batch = dspy.InputField(
        desc="List of texts to process",
        type=List[str],
        prefix="📝 Text Batch:\n"
    )

    processing_task = dspy.InputField(
        desc="Type of processing to perform on each text",
        type=str,
        prefix="🎯 Task: "
    )

    batch_size = dspy.InputField(
        desc="Size of the batch",
        type=int,
        prefix="📊 Batch Size: "
    )

    processing_results = dspy.OutputField(
        desc="Results for each text in the batch",
        type=List[Dict[str, Union[str, int, float, List[str]]]],
        prefix="📋 Results:\n"
    )

    batch_summary = dspy.OutputField(
        desc="Summary of batch processing",
        type=Dict[str, Union[int, float, str]],
        prefix="📈 Summary:\n"
    )

    failed_items = dspy.OutputField(
        desc="Items that failed processing with error messages",
        type=List[Dict[str, Union[str, int]]],
        optional=True
    )

    processing_time = dspy.OutputField(
        desc="Time taken for batch processing",
        type=float,
        prefix="⏱️ Processing Time: "
    )
```

## Key Takeaways

1. **Domain-Specific Design**: Tailor signatures to your specific domain requirements
2. **Comprehensive Coverage**: Include all relevant inputs and outputs for complete solutions
3. **Clear Structure**: Use prefixes and clear field descriptions for better prompting
4. **Modular Approach**: Break complex tasks into smaller, reusable signatures
5. **Error Handling**: Include validation and error outputs for robust applications

## Further Reading

- [Next Section: Exercises](./06-exercises.md) - Practice implementing these patterns
- [Chapter 3: Modules](../03-modules/) - Using these signatures with DSPy modules
- [Chapter 6: Real-World Applications](../06-real-world-applications/) - Building complete applications
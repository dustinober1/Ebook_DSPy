"""
Typed DSPy Signatures Examples

This file demonstrates advanced DSPy signature concepts including:
- Typed signature classes
- Field descriptions and prefixes
- Type hints and validation
- Complex data structures
- Optional fields and defaults
"""

import dspy
from typing import List, Dict, Optional, Union, Literal

# Example 1: Customer Support Ticket Analysis
class CustomerSupportTicket(dspy.Signature):
    """Analyze customer support tickets for classification and routing."""

    # Input fields with types and descriptions
    ticket_text = dspy.InputField(
        desc="Full text of the customer support ticket",
        type=str,
        prefix="📄 Ticket Content:\n"
    )

    customer_tier = dspy.InputField(
        desc="Customer's loyalty tier (bronze, silver, gold, platinum)",
        type=str,
        prefix="🏆 Customer Tier: "
    )

    previous_interactions = dspy.InputField(
        desc="Number of previous support interactions",
        type=int,
        default=0,
        prefix="📞 Previous Interactions: "
    )

    issue_category = dspy.InputField(
        desc="Broad category of the issue (technical, billing, general)",
        type=str,
        optional=True,
        prefix="🏷️ Issue Category: "
    )

    # Output fields with types and descriptions
    priority_level = dspy.OutputField(
        desc="Priority level from 1 (low) to 5 (critical)",
        type=int,
        prefix="⚡ Priority Level: "
    )

    sentiment = dspy.OutputField(
        desc="Customer sentiment (positive, neutral, negative, angry)",
        type=str,
        prefix="😊 Sentiment: "
    )

    resolution_steps = dspy.OutputField(
        desc="Recommended steps to resolve the issue",
        type=List[str],
        prefix="🔧 Resolution Steps:\n"
    )

    escalation_needed = dspy.OutputField(
        desc="Whether escalation to senior support is required",
        type=bool,
        prefix="🚨 Escalation Required: "
    )

    estimated_resolution_time = dspy.OutputField(
        desc="Estimated time to resolution in minutes",
        type=float,
        prefix="⏱️ Estimated Resolution Time: "
    )

# Example 2: Financial Document Analysis
class FinancialAnalyzer(dspy.Signature):
    """Analyze financial documents for insights and risks."""

    document_content = dspy.InputField(
        desc="Content of the financial document",
        type=str,
        prefix="📊 Financial Document:\n"
    )

    document_type = dspy.InputField(
        desc="Type of financial document (10-K, 10-Q, earnings_report, etc.)",
        type=str,
        prefix="📋 Document Type: "
    )

    fiscal_year = dspy.InputField(
        desc="Fiscal year the document covers",
        type=int,
        prefix="📅 Fiscal Year: "
    )

    company_industry = dspy.InputField(
        desc="Industry sector of the company",
        type=str,
        prefix="🏭 Industry: "
    )

    # Analysis outputs
    revenue_trend = dspy.OutputField(
        desc="Revenue trend analysis (increasing, decreasing, stable)",
        type=str,
        prefix="📈 Revenue Trend: "
    )

    profitability_metrics = dspy.OutputField(
        desc="Key profitability metrics and ratios",
        type=Dict[str, Union[float, str]],
        prefix="💰 Profitability:\n"
    )

    risk_factors = dspy.OutputField(
        desc="Identified risk factors with severity scores",
        type=List[Dict[str, Union[str, int, float]]],
        prefix="⚠️ Risk Factors:\n"
    )

    investment_recommendation = dspy.OutputField(
        desc="Investment recommendation (buy, hold, sell) with confidence",
        type=Dict[str, Union[str, float]],
        prefix="💼 Investment Recommendation:\n"
    )

    compliance_issues = dspy.OutputField(
        desc="Potential compliance or regulatory issues",
        type=List[str],
        prefix="⚖️ Compliance Issues:\n"
    )

# Example 3: Medical Symptom Analysis
class MedicalSymptomAnalyzer(dspy.Signature):
    """Analyze patient symptoms for potential conditions."""

    patient_symptoms = dspy.InputField(
        desc="List of symptoms with duration and severity",
        type=List[Dict[str, Union[str, int]]],
        prefix="🩺 Symptoms:\n"
    )

    patient_age = dspy.InputField(
        desc="Patient's age in years",
        type=int,
        prefix="👤 Age: "
    )

    patient_gender = dspy.InputField(
        desc="Patient's gender",
        type=str,
        prefix="⚥ Gender: "
    )

    medical_history = dspy.InputField(
        desc="Relevant medical history and conditions",
        type=str,
        optional=True,
        prefix="📋 Medical History:\n"
    )

    current_medications = dspy.InputField(
        desc="List of current medications",
        type=List[str],
        optional=True,
        prefix="💊 Current Medications:\n"
    )

    # Medical analysis outputs
    possible_conditions = dspy.OutputField(
        desc="List of possible conditions with probability scores",
        type=List[Dict[str, Union[str, float, List[str]]]],
        prefix="🔍 Possible Conditions:\n"
    )

    urgency_level = dspy.OutputField(
        desc="Medical urgency level (routine, urgent, emergency)",
        type=str,
        prefix="🚨 Urgency Level: "
    )

    recommended_tests = dspy.OutputField(
        desc="Recommended diagnostic tests",
        type=List[Dict[str, Union[str, str, bool]]],
        prefix="🧪 Recommended Tests:\n"
    )

    lifestyle_recommendations = dspy.OutputField(
        desc="Lifestyle changes that may help",
        type=List[str],
        prefix="🏃 Lifestyle Recommendations:\n"
    )

    follow_up_timeline = dspy.OutputField(
        desc="Recommended follow-up timeline",
        type=str,
        prefix="📅 Follow-up: "
    )

# Example 4: Educational Content Generator
class EducationalContentGenerator(dspy.Signature):
    """Generate educational content based on learning objectives."""

    topic = dspy.InputField(
        desc="Main topic to teach",
        type=str,
        prefix="📚 Topic: "
    )

    target_audience = dspy.InputField(
        desc="Target audience (beginner, intermediate, advanced)",
        type=str,
        prefix="👥 Target Audience: "
    )

    learning_objectives = dspy.InputField(
        desc="Specific learning objectives to achieve",
        type=List[str],
        prefix="🎯 Learning Objectives:\n"
    )

    content_length = dspy.InputField(
        desc="Desired content length in words",
        type=int,
        default=500,
        prefix="📝 Content Length: "
    )

    teaching_style = dspy.InputField(
        desc="Teaching style (visual, hands-on, theoretical, practical)",
        type=str,
        default="practical",
        prefix="🎨 Teaching Style: "
    )

    # Generated content outputs
    lesson_content = dspy.OutputField(
        desc="Main educational content",
        type=str,
        prefix="📖 Lesson Content:\n"
    )

    key_concepts = dspy.OutputField(
        desc="Key concepts covered in the lesson",
        type=List[str],
        prefix="🔑 Key Concepts:\n"
    )

    practice_exercises = dspy.OutputField(
        desc="Practice exercises to reinforce learning",
        type=List[Dict[str, Union[str, int, Dict[str, str]]]],
        prefix="✏️ Practice Exercises:\n"
    )

    assessment_questions = dspy.OutputField(
        desc="Questions to assess understanding",
        type=List[Dict[str, Union[str, List[str], str]]],
        prefix="❓ Assessment Questions:\n"
    )

    difficulty_rating = dspy.OutputField(
        desc="Actual difficulty rating from 1-10",
        type=int,
        prefix="📊 Difficulty Rating: "
    )

# Example 5: Product Review Analyzer
class ProductReviewAnalyzer(dspy.Signature):
    """Analyze product reviews for insights and improvements."""

    review_text = dspy.InputField(
        desc="Full text of the product review",
        type=str,
        prefix="💬 Review Text:\n"
    )

    product_category = dspy.InputField(
        desc="Category of the product",
        type=str,
        prefix="🏷️ Product Category: "
    )

    product_name = dspy.InputField(
        desc="Name of the specific product",
        type=str,
        prefix="📦 Product Name: "
    )

    reviewer_demographics = dspy.InputField(
        desc="Information about the reviewer",
        type=Dict[str, Union[str, int]],
        optional=True,
        prefix="👤 Reviewer Info:\n"
    )

    rating_given = dspy.InputField(
        desc="Rating given by reviewer (1-5 stars)",
        type=int,
        optional=True,
        prefix="⭐ Rating: "
    )

    # Analysis outputs
    sentiment_score = dspy.OutputField(
        desc="Sentiment score from -1 (very negative) to 1 (very positive)",
        type=float,
        prefix="😊 Sentiment Score: "
    )

    mentioned_features = dspy.OutputField(
        desc="Product features mentioned in the review",
        type=List[Dict[str, Union[str, str, int]]],
        prefix="🔧 Mentioned Features:\n"
    )

    competitor_comparisons = dspy.OutputField(
        desc="Comparisons to competitor products",
        type=List[Dict[str, Union[str, str]]],
        prefix="⚖️ Competitor Comparisons:\n"
    )

    improvement_suggestions = dspy.OutputField(
        desc="Suggestions for product improvement",
        type=List[Dict[str, Union[str, int]]],
        prefix="💡 Improvement Suggestions:\n"
    )

    purchase_intent = dspy.OutputField(
        desc="Likelihood this review influences others to purchase",
        type=Dict[str, Union[float, str]],
        prefix="🛒 Purchase Intent Impact:\n"
    )

# Example 6: Complex Signature with Union Types
class DataProcessor(dspy.Signature):
    """Process various types of data with flexible inputs and outputs."""

    input_data = dspy.InputField(
        desc="Data to process (can be text, JSON, or structured data)",
        type=Union[str, dict, list],
        prefix="📊 Input Data:\n"
    )

    processing_type = dspy.InputField(
        desc="Type of processing to perform",
        type=Literal["extract", "transform", "analyze", "validate"],
        prefix="⚙️ Processing Type: "
    )

    output_format = dspy.InputField(
        desc="Desired output format",
        type=Literal["text", "json", "markdown", "summary"],
        default="text",
        prefix="📋 Output Format: "
    )

    validation_schema = dspy.InputField(
        desc="Schema for validation (if validation processing type)",
        type=dict,
        optional=True,
        prefix="✅ Validation Schema:\n"
    )

    # Processing results
    processed_data = dspy.OutputField(
        desc="Processed version of input data",
        type=Union[str, dict, list],
        prefix="✨ Processed Data:\n"
    )

    processing_summary = dspy.OutputField(
        desc="Summary of what processing was performed",
        type=str,
        prefix="📝 Processing Summary:\n"
    )

    validation_results = dspy.OutputField(
        desc="Results of validation (if applicable)",
        type=Dict[str, Union[bool, List[str], str]],
        optional=True,
        prefix="✅ Validation Results:\n"
    )

    metadata = dspy.OutputField(
        desc="Metadata about processing including time and statistics",
        type=Dict[str, Union[int, float, str]],
        prefix="📈 Processing Metadata:\n"
    )

    errors_encountered = dspy.OutputField(
        desc="Any errors encountered during processing",
        type=List[Dict[str, Union[str, str]]],
        optional=True,
        prefix="❌ Errors:\n"
    )

def demonstrate_typed_signatures():
    """Demonstrate various typed signature examples."""

    print("Demonstrating Typed DSPy Signatures...\n")
    print("=" * 60)

    # Example 1: Customer Support
    print("\n1. Customer Support Ticket Analysis")
    print("-" * 40)
    support_analyzer = dspy.Predict(CustomerSupportTicket)
    result = support_analyzer(
        ticket_text="My premium account features are not working properly. This is the third time I've contacted support.",
        customer_tier="gold",
        previous_interactions=2
    )
    print(f"Priority: {result.priority_level}")
    print(f"Sentiment: {result.sentiment}")
    print(f"Escalation: {result.escalation_needed}")

    # Example 2: Financial Analysis
    print("\n2. Financial Document Analysis")
    print("-" * 40)
    financial_analyzer = dspy.Predict(FinancialAnalyzer)
    result = financial_analyzer(
        document_content="The company reported Q3 2024 revenue of $500M, up 15% YoY...",
        document_type="earnings_report",
        fiscal_year=2024,
        company_industry="technology"
    )
    print(f"Revenue Trend: {result.revenue_trend}")
    print(f"Investment Recommendation: {result.investment_recommendation}")

    # Example 3: Medical Analysis
    print("\n3. Medical Symptom Analysis")
    print("-" * 40)
    medical_analyzer = dspy.Predict(MedicalSymptomAnalyzer)
    result = medical_analyzer(
        patient_symptoms=[
            {"symptom": "headache", "duration": "3 days", "severity": 6},
            {"symptom": "fever", "duration": "1 day", "severity": 4}
        ],
        patient_age=35,
        patient_gender="male"
    )
    print(f"Urgency: {result.urgency_level}")
    print(f"Possible Conditions: {len(result.possible_conditions)} identified")

    # Example 4: Educational Content
    print("\n4. Educational Content Generation")
    print("-" * 40)
    content_generator = dspy.Predict(EducationalContentGenerator)
    result = content_generator(
        topic="Machine Learning Basics",
        target_audience="beginner",
        learning_objectives=["Understand ML concepts", "Know different types of ML"],
        content_length=300
    )
    print(f"Key Concepts: {len(result.key_concepts)} concepts covered")
    print(f"Difficulty: {result.difficulty_rating}/10")

    # Example 5: Product Review
    print("\n5. Product Review Analysis")
    print("-" * 40)
    review_analyzer = dspy.Predict(ProductReviewAnalyzer)
    result = review_analyzer(
        review_text="Great laptop! Fast performance but battery life could be better.",
        product_category="electronics",
        product_name="UltraBook Pro",
        rating_given=4
    )
    print(f"Sentiment Score: {result.sentiment_score}")
    print(f"Features Mentioned: {len(result.mentioned_features)}")

    print("\n" + "=" * 60)
    print("Typed signature examples completed!")

if __name__ == "__main__":
    demonstrate_typed_signatures()
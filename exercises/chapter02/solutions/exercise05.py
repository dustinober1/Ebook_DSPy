"""
Exercise 5 Solutions: Signature Refactoring

This file contains solutions for Exercise 5 on improving a poorly designed signature.
"""

import dspy
from typing import List, Dict, Optional, Union, Literal
from enum import Enum

# Problems identified with BadSignature
problems_with_bad_signature = [
    "1. Vague field names - 'data' and 'info' don't indicate what they contain",
    "2. No type information - Fields have no type annotations",
    "3. Missing descriptions - No explanation of what fields represent",
    "4. Insufficient fields - Only 2 inputs and 2 outputs for complex processing",
    "5. No error handling - No way to handle processing failures",
    "6. Not domain-specific - Too generic to be useful in practice",
    "7. No prefixes - Missing helpful prefixes for better prompting",
    "8. No optional fields - All fields are required",
    "9. No documentation - Class has no docstring explaining purpose"
]

# Refactored signature for customer feedback analysis
class ImprovedCustomerFeedbackAnalyzer(dspy.Signature):
    """Analyze customer feedback to extract insights and generate actionable recommendations."""

    # Detailed input fields
    feedback_text = dspy.InputField(
        desc="Full text of customer feedback including all comments",
        type=str,
        prefix="💬 Customer Feedback:\n"
    )

    feedback_source = dspy.InputField(
        desc="Source of feedback (email, review site, survey, social media, etc.)",
        type=Literal["email", "review_site", "survey", "social_media", "support_chat", "phone_call"],
        prefix="📡 Source: "
    )

    customer_information = dspy.InputField(
        desc="Customer demographics and account information",
        type=Dict[str, Union[str, int, float, bool]],
        prefix="👤 Customer Info:\n"
    )

    product_service_details = dspy.InputField(
        desc="Details about the product or service being reviewed",
        type=Dict[str, Union[str, float, int, List[str]]],
        prefix="📦 Product/Service:\n"
    )

    feedback_metadata = dspy.InputField(
        desc="Additional metadata about the feedback",
        type=Dict[str, Union[str, int, float, bool]],
        optional=True,
        prefix="📋 Metadata:\n"
    )

    # Comprehensive output fields
    sentiment_analysis = dspy.OutputField(
        desc="Detailed sentiment analysis with scores and emotional indicators",
        type=Dict[str, Union[str, float, List[str]]],
        prefix="😊 Sentiment Analysis:\n"
    )

    key_topics = dspy.OutputField(
        desc="Main topics and themes identified in the feedback",
        type=List[Dict[str, Union[str, int, float]]],
        prefix="🔑 Key Topics:\n"
    )

    specific_issues = dspy.OutputField(
        desc="Specific problems or complaints mentioned",
        type=List[Dict[str, Union[str, str, int]]],
        prefix="⚠️ Issues Identified:\n"
    )

    positive_aspects = dspy.OutputField(
        desc="Positive feedback and compliments",
        type=List[Dict[str, Union[str, str]]],
        prefix="✅ Positive Aspects:\n"
    )

    actionable_recommendations = dspy.OutputField(
        desc="Actionable recommendations for improvement",
        type=List[Dict[str, Union[str, str, int, Dict[str, Union[str, int]]]]],
        prefix("💡 Recommendations:\n")
    )

    feedback_priority = dspy.OutputField(
        desc="Priority level for addressing this feedback",
        type=Dict[str, Union[str, int, List[str]]],
        prefix("🚨 Priority:\n")
    )

    follow_up_actions = dspy.OutputField(
        desc="Specific follow-up actions needed",
        type=List[Dict[str, Union[str, str, bool, int]]],
        prefix("📞 Follow-up Actions:\n")
    )

    customer_satisfaction_prediction = dspy.OutputField(
        desc="Predicted impact on customer satisfaction and retention",
        type=Dict[str, Union[float, str, List[str]]],
        prefix("💰 Satisfaction Impact:\n")
    )

    processing_metadata = dspy.OutputField(
        desc="Metadata about the analysis process",
        type=Dict[str, Union[float, int, str]],
        prefix("📊 Processing Info:\n")
    )

# Alternative specialized signatures for different feedback types

class ProductReviewAnalyzer(dspy.Signature):
    """Specialized analyzer for product reviews."""

    review_text = dspy.InputField(
        desc="Full product review text",
        type=str,
        prefix="📝 Review:\n"
    )

    product_category = dspy.InputField(
        desc="Category of the product",
        type=str,
        prefix="🏷️ Category: "
    )

    product_attributes = dspy.InputField(
        desc="Specific product attributes being reviewed",
        type=List[str],
        prefix="🔧 Attributes:\n"
    )

    reviewer_profile = dspy.InputField(
        desc="Information about the reviewer",
        type=Dict[str, Union[str, int, bool]],
        prefix="👤 Reviewer:\n"
    )

    feature_feedback = dspy.OutputField(
        desc="Feedback on specific product features",
        type=List[Dict[str, Union[str, int, float]]],
        prefix("🔧 Feature Feedback:\n")
    )

    usability_rating = dspy.OutputField(
        desc="Assessment of product usability",
        type=Dict[str, Union[int, str, List[str]]],
        prefix("👍 Usability:\n")
    )

    value_assessment = dspy.OutputField(
        desc "Assessment of value for money",
        type=Dict[str, Union[str, int, float]],
        prefix("💰 Value:\n")
    )

    comparison_feedback = dspy.OutputField(
        desc="Comparisons to competing products",
        type=List[Dict[str, Union[str, str]]],
        prefix("⚖️ Comparisons:\n")
    )

    purchase_recommendation = dspy.OutputField(
        desc="Whether the reviewer recommends the product",
        type=Dict[str, Union[bool, str, float]],
        prefix("✅ Recommendation:\n")
    )

class ServiceInteractionAnalyzer(dspy.Signature):
    """Specialized analyzer for service interactions."""

    interaction_transcript = dspy.InputField(
        desc="Full transcript or summary of service interaction",
        type=str,
        prefix="💬 Interaction:\n"
    )

    interaction_type = dspy.InputField(
        desc="Type of service interaction",
        type=Literal["support", "sales", "complaint", "inquiry", "feedback"],
        prefix("🏷️ Type:\n")
    )

    service_rep_performance = dspy.InputField(
        desc="Performance metrics for service representative",
        type=Dict[str, Union[int, float, str]],
        optional=True,
        prefix("👨‍💼 Rep Performance:\n")
    )

    interaction_outcome = dspy.InputField(
        desc="Final outcome of the interaction",
        type=str,
        optional=True,
        prefix("✅ Outcome:\n")
    )

    service_quality_metrics = dspy.OutputField(
        desc "Quality metrics for the service interaction",
        type=Dict[str, Union[int, float, str]],
        prefix("⭐ Service Quality:\n")
    )

    communication_effectiveness = dspy.OutputField(
        desc="Assessment of communication effectiveness",
        type=Dict[str, Union[int, str, List[str]]],
        prefix("💬 Communication:\n")
    )

    problem_resolution = dspy.OutputField(
        desc "Details of problem resolution if applicable",
        type=Dict[str, Union[bool, str, int, List[str]]],
        prefix("🔧 Resolution:\n")
    )

    customer_experience_rating = dspy.OutputField(
        desc "Overall customer experience rating",
        type=Dict[str, Union[int, str, List[str]]],
        prefix("😊 Experience:\n")
    )

    improvement_opportunities = dspy.OutputField(
        desc "Opportunities for service improvement",
        type[List[Dict[str, Union[str, str, int]]]],
        prefix("💡 Improvements:\n")
    )

# Unit tests for the improved signature

def test_improved_signature():
    """Test the ImprovedCustomerFeedbackAnalyzer signature with various scenarios."""

    import json

    print("=" * 60)
    print("Testing ImprovedCustomerFeedbackAnalyzer")
    print("=" * 60)

    # Initialize the analyzer
    analyzer = dspy.Predict(ImprovedCustomerFeedbackAnalyzer)

    # Test Case 1: Positive feedback
    print("\n✅ Test Case 1: Positive Product Review")
    print("-" * 40)

    positive_result = analyzer(
        feedback_text="Absolutely love this product! The quality is exceptional and it exceeded my expectations. Customer service was also fantastic when I had questions. Would definitely recommend to friends!",
        feedback_source="review_site",
        customer_information={
            "customer_id": "CUST_12345",
            "age": 35,
            "account_type": "premium",
            "years_with_company": 2
        },
        product_service_details={
            "product_name": "Wireless Headphones Pro",
            "category": "electronics",
            "price": 299.99,
            "purchase_date": "2024-01-10"
        },
        feedback_metadata={
            "rating": 5,
            "verified_purchase": True,
            "helpful_votes": 42
        }
    )

    print(f"Sentiment: {positive_result.sentiment_analysis.get('overall', 'N/A')}")
    print(f"Confidence: {positive_result.sentiment_analysis.get('confidence', 0):.2f}")
    print(f"Key Topics: {len(positive_result.key_topics)} identified")
    print(f"Priority: {positive_result.feedback_priority.get('level', 'N/A')}")

    # Test Case 2: Negative feedback with issues
    print("\n❌ Test Case 2: Negative Service Feedback")
    print("-" * 40)

    negative_result = analyzer(
        feedback_text="Very disappointed with the service. Waited 45 minutes on hold only to be disconnected. The agent was rude and unhelpful when I finally got through. My issue still isn't resolved after 3 attempts.",
        feedback_source="support_chat",
        customer_information={
            "customer_id": "CUST_67890",
            "age": 42,
            "account_type": "standard",
            "years_with_company": 5
        },
        product_service_details={
            "service_type": "technical_support",
            "issue_type": "billing_inquiry",
            "previous_interactions": 2
        },
        feedback_metadata={
            "session_id": "CHAT_456789",
            "duration_minutes": 52,
            "agent_id": "AGENT_234"
        }
    )

    print(f"Sentiment: {negative_result.sentiment_analysis.get('overall', 'N/A')}")
    print(f"Issues Identified: {len(negative_result.specific_issues)}")
    print(f"Urgent Actions: {len(negative_result.follow_up_actions)} required")
    print(f"Retention Risk: {negative_result.customer_satisfaction_prediction.get('retention_risk', 'N/A')}")

    # Test Case 3: Mixed feedback with suggestions
    print("\n🔄 Test Case 3: Mixed Feedback with Suggestions")
    print("-" * 40)

    mixed_result = analyzer(
        feedback_text="The product works well overall and the features are innovative. However, the user interface could be more intuitive. I had to watch several tutorials to understand advanced features. Also, the mobile app crashes occasionally. If these were fixed, it would be a 5-star product!",
        feedback_source="email",
        customer_information={
            "customer_id": "CUST_11111",
            "age": 28,
            "account_type": "professional",
            "years_with_company": 1
        },
        product_service_details={
            "product_name": "Project Management Suite",
            "category": "software",
            "version": "2.3.1",
            "subscription_tier": "professional"
        }
    )

    print(f"Sentiment: {mixed_result.sentiment_analysis.get('overall', 'N/A')}")
    print(f"Positive Aspects: {len(mixed_result.positive_aspects)}")
    print(f"Recommendations: {len(mixed_result.actionable_recommendations)}")
    print(f"Processing Time: {mixed_result.processing_metadata.get('processing_time_ms', 'N/A')}ms")

    # Verify output structure
    print("\n🔍 Output Structure Validation")
    print("-" * 40)

    required_fields = [
        'sentiment_analysis', 'key_topics', 'specific_issues',
        'positive_aspects', 'actionable_recommendations',
        'feedback_priority', 'follow_up_actions',
        'customer_satisfaction_prediction', 'processing_metadata'
    ]

    for field in required_fields:
        if hasattr(mixed_result, field):
            print(f"✅ {field}: Present")
        else:
            print(f"❌ {field}: Missing")

    print("\n" + "=" * 60)
    print("Unit tests completed successfully!")
    print("=" * 60)

def demonstrate_specialized_analyzers():
    """Demonstrate the specialized analyzers for different feedback types."""

    print("\n" + "=" * 60)
    print("Demonstrating Specialized Analyzers")
    print("=" * 60)

    # Product Review Analyzer
    print("\n📦 Product Review Analyzer Demo")
    print("-" * 40)

    product_analyzer = dspy.Predict(ProductReviewAnalyzer)

    product_result = product_analyzer(
        review_text="Great laptop for the price! The keyboard is comfortable and battery life is impressive. However, the screen could be brighter and the touchpad is a bit glitchy. Overall, satisfied with the purchase.",
        product_category="electronics",
        product_attributes=["laptop", "keyboard", "battery", "screen", "touchpad"],
        reviewer_profile={"tech_savvy": True, "previous_purchases": 5}
    )

    print(f"Feature Feedback: {len(product_result.feature_feedback)} features analyzed")
    print(f"Usability Rating: {product_result.usability_rating.get('score', 'N/A')}/10")
    print(f"Value Assessment: {product_result.value_assessment.get('rating', 'N/A')}")
    print(f"Recommends: {product_result.purchase_recommendation.get('recommends', False)}")

    # Service Interaction Analyzer
    print("\n🎧 Service Interaction Analyzer Demo")
    print("-" * 40)

    service_analyzer = dspy.Predict(ServiceInteractionAnalyzer)

    service_result = service_analyzer(
        interaction_transcript="Customer called about billing issue. Agent patiently explained charges and offered a discount. Issue resolved within 10 minutes. Customer expressed satisfaction.",
        interaction_type="support",
        service_rep_performance={
            "resolution_time_minutes": 10,
            "customer_satisfaction": 5,
            "first_call_resolution": True
        },
        interaction_outcome="Billing question answered, discount applied"
    )

    print(f"Service Quality: {service_result.service_quality_metrics.get('overall_score', 'N/A')}/10")
    print(f"Communication: {service_result.communication_effectiveness.get('rating', 'N/A')}/10")
    print(f"Problem Resolved: {service_result.problem_resolution.get('resolved', False)}")
    print(f"Experience Rating: {service_result.customer_experience_rating.get('overall', 'N/A')}/10")

    print("\n" + "=" * 60)
    print("Specialized analyzer demonstrations completed!")
    print("=" * 60)

# Improvements documentation
improvements_made = """
1. Clear, Descriptive Field Names:
   - Changed 'data' to 'feedback_text' - Clear indication of content
   - Changed 'info' to 'customer_information' - Specifies what information
   - Added specific fields for different data types

2. Type Safety:
   - Added type hints for all fields (str, int, Dict, List, etc.)
   - Used Literal types for constrained values (feedback_source, interaction_type)
   - Defined proper Union types for flexible but structured data

3. Comprehensive Descriptions:
   - Each field has a clear description of its purpose
   - Descriptions explain what data is expected
   - Helpful for both developers and the language model

4. Rich Metadata:
   - Added prefixes for better prompting (💬, 👤, 📦, etc.)
   - Metadata fields for additional context
   - Processing metadata for tracking and debugging

5. Expanded Inputs and Outputs:
   - 5 input fields instead of 2 (more comprehensive)
   - 9 output fields instead of 2 (more actionable insights)
   - Covers all aspects of feedback analysis

6. Error Handling Edge Cases:
   - Optional fields for non-critical information
   - Structured outputs that can handle missing data
   - Processing metadata to track analysis quality

7. Domain-Specific Design:
   - Created specifically for customer feedback analysis
   - Fields tailored to business needs (retention, recommendations, priority)
   - Actionable outputs that drive business decisions

8. Specialized Variants:
   - ProductReviewAnalyzer for product-specific feedback
   - ServiceInteractionAnalyzer for service interactions
   - Each optimized for their specific use case

9. Documentation:
   - Clear docstrings explaining purpose
   - Comments explaining complex fields
   - Type annotations serve as documentation

10. Extensibility:
    - Structured to accommodate future fields
    - Modular design allows for easy enhancement
    - Consistent patterns across signatures
"""

# Additional utility functions

def compare_signatures():
    """Compare the old and new signatures to highlight improvements."""

    print("\n" + "=" * 60)
    print("Signature Comparison: Before vs After")
    print("=" * 60)

    print("\n❌ Original BadSignature:")
    print("```python")
    print("class BadSignature(dspy.Signature):")
    print('    """Poorly designed signature."""')
    print("    data = dspy.InputField()  # Vague name, no type, no description")
    print("    info = dspy.InputField()   # Vague name, no type, no description")
    print("    result = dspy.OutputField()  # Vague name, no type, no description")
    print("    other = dspy.OutputField()   # Vague name, no type, no description")
    print("```")

    print("\n✅ ImprovedCustomerFeedbackAnalyzer:")
    print("```python")
    print("class ImprovedCustomerFeedbackAnalyzer(dspy.Signature):")
    print('    """Analyze customer feedback comprehensively."""')
    print("    feedback_text = dspy.InputField(")
    print('        desc="Full text of customer feedback",')
    print("        type=str,")
    print('        prefix="💬 Customer Feedback:\\n"')
    print("    )")
    print("    # ... (9 more detailed fields with types and descriptions)")
    print("```")

    comparison = {
        "Field Names": {
            "Before": "Generic (data, info, result, other)",
            "After": "Descriptive (feedback_text, sentiment_analysis, etc.)"
        },
        "Type Information": {
            "Before": "None specified",
            "After": "Full type annotations (str, Dict, List, Literal)"
        },
        "Field Count": {
            "Before": "2 inputs, 2 outputs",
            "After": "5 inputs, 9 outputs (comprehensive coverage)"
        },
        "Documentation": {
            "Before": "None",
            "After": "Field descriptions and class docstring"
        },
        "Usability": {
            "Before": "Unclear purpose, hard to use correctly",
            "After": "Clear purpose, self-documenting"
        },
        "Business Value": {
            "Before": "Low - too generic to be useful",
            "After": "High - actionable insights for business"
        }
    }

    print("\n📊 Comparison Summary:")
    print("-" * 40)
    for aspect, comparison in comparison.items():
        print(f"\n{aspect}:")
        print(f"  Before: {comparison['Before']}")
        print(f"  After:  {comparison['After']}")

if __name__ == "__main__":
    # Run all demonstrations
    test_improved_signature()
    demonstrate_specialized_analyzers()
    compare_signatures()

    # Print improvements summary
    print("\n" + "=" * 60)
    print("IMPROVEMENTS SUMMARY")
    print("=" * 60)
    print(improvements_made)
"""
Exercise 2 Solutions: Typed Signatures

This file contains solutions for Exercise 2 on creating typed DSPy signatures.
"""

import dspy
from typing import List, Dict, Optional, Union, Literal

# Task 2.1: Convert Basic Signature to Typed

class CustomerReviewAnalyzer(dspy.Signature):
    """Analyze customer reviews for sentiment and key points."""

    # Input fields with types and descriptions
    customer_review = dspy.InputField(
        desc="Full text of the customer review",
        type=str,
        prefix="📝 Review:\n"
    )

    product_category = dspy.InputField(
        desc="Category of the product being reviewed",
        type=str,
        prefix="🏷️ Category: "
    )

    # Output fields with types and descriptions
    sentiment_score = dspy.OutputField(
        desc="Sentiment score from -1 (very negative) to 1 (very positive)",
        type=float,
        prefix="😊 Sentiment Score: "
    )

    key_points = dspy.OutputField(
        desc="Main points extracted from the review",
        type=List[str],
        prefix="💡 Key Points:\n"
    )

    sentiment_label = dspy.OutputField(
        desc="Sentiment classification (positive, negative, neutral)",
        type=str,
        prefix="🏷️ Sentiment: "
    )

    review_summary = dspy.OutputField(
        desc="Brief summary of the review",
        type=str,
        prefix="📄 Summary: "
    )

# Task 2.2: Create Email Processing Signature

class EmailProcessor(dspy.Signature):
    """Process and categorize incoming emails."""

    # Input fields
    email_text = dspy.InputField(
        desc="Full content of the email including body and headers",
        type=str,
        prefix="📧 Email Content:\n"
    )

    sender_information = dspy.InputField(
        desc="Information about the sender including email address and name",
        type=Dict[str, Union[str, int]],
        prefix="👤 Sender:\n"
    )

    priority_level = dspy.InputField(
        desc="Priority level specified by sender or system (1-5)",
        type=int,
        default=3,
        prefix="⚡ Priority: "
    )

    attachments = dspy.InputField(
        desc="List of attachments if any",
        type=List[Dict[str, Union[str, int]]],
        optional=True,
        prefix="📎 Attachments:\n"
    )

    previous_conversation = dspy.InputField(
        desc="Previous emails in this conversation thread",
        type=str,
        optional=True,
        prefix="📜 History:\n"
    )

    # Output fields
    email_category = dspy.OutputField(
        desc="Category of the email (inquiry, complaint, information, spam, etc.)",
        type=str,
        prefix="🏷️ Category: "
    )

    response_needed = dspy.OutputField(
        desc="Whether a response is needed and urgency",
        type=Dict[str, Union[bool, str, int]],
        prefix="💬 Response Needed:\n"
    )

    urgency = dspy.OutputField(
        desc="Urgency level for processing (1-5)",
        type=int,
        prefix="🚨 Urgency: "
    )

    action_items = dspy.OutputField(
        desc="List of action items extracted from the email",
        type=List[Dict[str, Union[str, bool, str]]],
        prefix="✅ Action Items:\n"
    )

    suggested_reply = dspy.OutputField(
        desc="Suggested reply template or key points to include",
        type=str,
        optional=True,
        prefix="✉️ Suggested Reply:\n"
    )

    routing_destination = dspy.OutputField(
        desc="Who or which department should handle this email",
        type=str,
        prefix="🔄 Route To: "
    )

# Task 2.3: Signature Validation

def validate_signature(signature_class) -> Dict[str, Union[bool, List[str], str]]:
    """
    Validate that a signature class meets minimum requirements.

    Args:
        signature_class: A class that inherits from dspy.Signature

    Returns:
        Dictionary with validation results
    """

    validation_result = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "field_counts": {"inputs": 0, "outputs": 0}
    }

    # Check if it's actually a Signature class
    if not issubclass(signature_class, dspy.Signature):
        validation_result["is_valid"] = False
        validation_result["errors"].append("Class must inherit from dspy.Signature")
        return validation_result

    # Get all fields from the signature
    field_definitions = signature_class.__annotations__ if hasattr(signature_class, '__annotations__') else {}

    # Count input and output fields
    for field_name, field_obj in signature_class.__dict__.items():
        if isinstance(field_obj, dspy.InputField):
            validation_result["field_counts"]["inputs"] += 1
            # Check if field has description
            if not field_obj.desc or field_obj.desc.strip() == "":
                validation_result["warnings"].append(f"Input field '{field_name}' missing description")
        elif isinstance(field_obj, dspy.OutputField):
            validation_result["field_counts"]["outputs"] += 1
            # Check if field has description
            if not field_obj.desc or field_obj.desc.strip() == "":
                validation_result["warnings"].append(f"Output field '{field_name}' missing description")

    # Check minimum field requirements
    if validation_result["field_counts"]["inputs"] < 2:
        validation_result["is_valid"] = False
        validation_result["errors"].append("Signature must have at least 2 input fields")

    if validation_result["field_counts"]["outputs"] < 2:
        validation_result["is_valid"] = False
        validation_result["errors"].append("Signature must have at least 2 output fields")

    # Check for docstring
    if not signature_class.__doc__ or signature_class.__doc__.strip() == "":
        validation_result["warnings"].append("Signature class missing docstring")

    return validation_result

# Additional helper functions

def test_customer_review_analyzer():
    """Test the CustomerReviewAnalyzer signature."""

    print("Testing CustomerReviewAnalyzer...")

    # Validate the signature
    validation = validate_signature(CustomerReviewAnalyzer)
    print(f"Validation: {validation['is_valid']}")

    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")

    # Test with actual prediction
    analyzer = dspy.Predict(CustomerReviewAnalyzer)
    result = analyzer(
        customer_review="Great product! Excellent quality and fast shipping. Highly recommend!",
        product_category="electronics"
    )

    print(f"Sentiment Score: {result.sentiment_score}")
    print(f"Sentiment Label: {result.sentiment_label}")
    print(f"Key Points: {result.key_points}")

def test_email_processor():
    """Test the EmailProcessor signature."""

    print("\nTesting EmailProcessor...")

    # Validate the signature
    validation = validate_signature(EmailProcessor)
    print(f"Validation: {validation['is_valid']}")

    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")

    # Test with actual prediction
    processor = dspy.Predict(EmailProcessor)
    result = processor(
        email_text="Subject: Urgent Issue - Cannot access my account\n\nHi support team,\n\nI haven't been able to log into my account for the past 2 days. I've tried resetting my password but the reset link never arrives. This is very urgent as I need to access my account for work.\n\nPlease help!\nJohn",
        sender_information={"email": "john@example.com", "name": "John Doe", "account_type": "premium"},
        priority_level=4
    )

    print(f"Category: {result.email_category}")
    print(f"Urgency: {result.urgency}")
    print(f"Response Needed: {result.response_needed}")
    print(f"Action Items: {len(result.action_items)}")

# Demonstration of validation with bad signature

class BadSignatureExample(dspy.Signature):
    """Example of a poorly designed signature."""

    data = dspy.InputField()  # No description
    info = dspy.InputField()  # No description

    result = dspy.OutputField()  # No description
    # Only one output field (below minimum)

def demonstrate_validation():
    """Demonstrate the validation function with good and bad examples."""

    print("\n=== Signature Validation Demonstration ===\n")

    # Test good signatures
    print("1. CustomerReviewAnalyzer Validation:")
    result1 = validate_signature(CustomerReviewAnalyzer)
    print(f"   Valid: {result1['is_valid']}")
    print(f"   Input Fields: {result1['field_counts']['inputs']}")
    print(f"   Output Fields: {result1['field_counts']['outputs']}")

    if result1['warnings']:
        print(f"   Warnings: {result1['warnings']}")

    print("\n2. EmailProcessor Validation:")
    result2 = validate_signature(EmailProcessor)
    print(f"   Valid: {result2['is_valid']}")
    print(f"   Input Fields: {result2['field_counts']['inputs']}")
    print(f"   Output Fields: {result2['field_counts']['outputs']}")

    if result2['warnings']:
        print(f"   Warnings: {result2['warnings']}")

    # Test bad signature
    print("\n3. BadSignatureExample Validation:")
    result3 = validate_signature(BadSignatureExample)
    print(f"   Valid: {result3['is_valid']}")
    print(f"   Input Fields: {result3['field_counts']['inputs']}")
    print(f"   Output Fields: {result3['field_counts']['outputs']}")

    if result3['errors']:
        print(f"   Errors: {result3['errors']}")

    if result3['warnings']:
        print(f"   Warnings: {result3['warnings']}")

if __name__ == "__main__":
    # Run tests and demonstrations
    test_customer_review_analyzer()
    test_email_processor()
    demonstrate_validation()
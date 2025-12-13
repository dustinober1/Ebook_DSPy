"""
Real-World DSPy Signature Applications

This file demonstrates practical, production-ready signature implementations
for real business scenarios across multiple domains.
"""

import dspy
from typing import List, Dict, Optional, Union, Literal
from enum import Enum
import json

# Example 1: Customer Service Intelligence System
class ServiceLevel(str, Enum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"

class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class CustomerIntelligenceAnalyzer(dspy.Signature):
    """Analyze customer interactions for insights and actions."""

    # Customer and interaction data
    interaction_text = dspy.InputField(
        desc="Full customer interaction transcript",
        type=str,
        prefix="📝 Interaction Transcript:\n"
    )

    customer_profile = dspy.InputField(
        desc="Customer profile including history, tier, and value",
        type=Dict[str, Union[str, int, float, List[str]]],
        prefix="👤 Customer Profile:\n"
    )

    interaction_context = dspy.InputField(
        desc="Context of interaction (channel, time, previous contacts)",
        type=Dict[str, Union[str, int, List[Dict[str, str]]]],
        prefix="🏛️ Interaction Context:\n"
    )

    # Intelligence outputs
    customer_sentiment = dspy.OutputField(
        desc="Customer sentiment analysis with confidence",
        type=Dict[str, Union[str, float, List[str]]],
        prefix="😊 Sentiment Analysis:\n"
    )

    intent_classification = dspy.OutputField(
        desc="Primary and secondary intents of the customer",
        type=Dict[str, Union[str, float, List[str]]],
        prefix="🎯 Intent Classification:\n"
    )

    risk_assessment = dspy.OutputField(
        desc="Churn risk and escalation risk assessments",
        type=Dict[str, Union[float, str, List[str]]],
        prefix="⚠️ Risk Assessment:\n"
    )

    value_opportunity = dspy.OutputField(
        desc="Upsell, cross-sell, and loyalty opportunities",
        type=Dict[str, Union[str, float, List[str], Dict[str, str]]],
        prefix="💰 Value Opportunities:\n"
    )

    recommended_actions = dspy.OutputField(
        desc="Immediate and follow-up actions for service team",
        type=List[Dict[str, Union[str, bool, int, Dict[str, str]]]],
        prefix="⚡ Recommended Actions:\n"
    )

    quality_metrics = dspy.OutputField(
        desc="Service quality metrics and improvement areas",
        type=Dict[str, Union[int, float, str, List[str]]],
        prefix="📊 Quality Metrics:\n"
    )

# Example 2: Healthcare Triage and Diagnosis Assistant
class UrgencyLevel(str, Enum):
    ROUTINE = "routine"
    URGENT = "urgent"
    EMERGENCY = "emergency"

class MedicalTriageSystem(dspy.Signature):
    """AI-powered medical triage and preliminary diagnosis system."""

    # Patient presentation
    chief_complaint = dspy.InputField(
        desc="Primary reason for medical consultation",
        type=str,
        prefix="🩺 Chief Complaint:\n"
    )

    symptom_details = dspy.InputField(
        desc="Detailed symptom description with onset, duration, severity",
        type=List[Dict[str, Union[str, int, float]]],
        prefix="📋 Symptom Details:\n"
    )

    patient_demographics = dspy.InputField(
        desc="Age, gender, and relevant demographic information",
        type=Dict[str, Union[str, int]],
        prefix="👤 Demographics:\n"
    )

    medical_history = dspy.InputField(
        desc="Relevant past medical history and conditions",
        type=str,
        optional=True,
        prefix="📚 Medical History:\n"
    )

    vital_signs = dspy.InputField(
        desc="Current vital signs if available",
        type=Dict[str, Union[int, float, str]],
        optional=True,
        prefix="📊 Vital Signs:\n"
    )

    current_medications = dspy.InputField(
        desc="List of current medications and supplements",
        type=List[str],
        optional=True,
        prefix="💊 Medications:\n"
    )

    # Triage and analysis outputs
    urgency_classification = dspy.OutputField(
        desc="Urgency level with reasoning and timeframe",
        type=Dict[str, Union[str, List[str]]],
        prefix="🚨 Urgency Classification:\n"
    )

    differential_diagnosis = dspy.OutputField(
        desc="Top possible diagnoses with probability scores",
        type=List[Dict[str, Union[str, float, List[str]]]],
        prefix="🔍 Differential Diagnosis:\n"
    )

    red_flags = dspy.OutputField(
        desc="Critical symptoms requiring immediate attention",
        type=List[Dict[str, Union[str, str]]],
        prefix="🚩 Red Flags:\n"
    )

    recommended_workup = dspy.OutputField(
        desc="Recommended diagnostic tests and examinations",
        type=List[Dict[str, Union[str, str, bool, int]]],
        prefix="🧪 Recommended Workup:\n"
    )

    initial_treatment = dspy.OutputField(
        desc="Initial treatment recommendations and comfort measures",
        type=Dict[str, Union[str, List[str], Dict[str, str]]],
        prefix="💊 Initial Treatment:\n"
    )

    patient_education = dspy.OutputField(
        desc="Patient education points and when to seek further care",
        type=List[Dict[str, Union[str, str]]],
        prefix="📚 Patient Education:\n"
    )

    follow_up_recommendations = dspy.OutputField(
        desc="Follow-up timeline and monitoring instructions",
        type=Dict[str, Union[str, int, List[str]]],
        prefix="📅 Follow-up Recommendations:\n"
    )

# Example 3: Financial Risk Assessment Engine
class RiskCategory(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class FinancialRiskAssessment(dspy.Signature):
    """Comprehensive financial risk assessment for lending and investment."""

    # Entity information
    entity_profile = dspy.InputField(
        desc="Complete profile of individual or company",
        type=Dict[str, Union[str, int, float, List[str], Dict[str, Any]]],
        prefix="🏢 Entity Profile:\n"
    )

    financial_statements = dspy.InputField(
        desc="Recent financial statements and performance metrics",
        type=Dict[str, Union[float, int, str, Dict[str, float]]],
        prefix="📊 Financial Statements:\n"
    )

    credit_history = dspy.InputField(
        desc="Credit history, scores, and payment patterns",
        type=Dict[str, Union[int, float, List[Dict[str, Union[str, int]]]]],
        prefix="💳 Credit History:\n"
    )

    loan_request = dspy.InputField(
        desc="Details of loan or investment request",
        type=Dict[str, Union[float, int, str, List[str]]],
        prefix="💰 Request Details:\n"
    )

    market_conditions = dspy.InputField(
        desc="Current market and economic conditions",
        type=Dict[str, Union[str, float, List[str]]],
        prefix="🌍 Market Conditions:\n"
    )

    # Risk assessment outputs
    overall_risk_score = dspy.OutputField(
        desc="Overall risk score (0-100) with explanation",
        type=Dict[str, Union[float, str, List[str]]],
        prefix="⚠️ Overall Risk Score:\n"
    )

    risk_category = dspy.OutputField(
        desc="Risk category with qualifying factors",
        type=Dict[str, Union[str, List[str]]],
        prefix="📊 Risk Category:\n"
    )

    specific_risks = dspy.OutputField(
        desc="Specific risks identified with severity levels",
        type=List[Dict[str, Union[str, int, float, List[str]]]],
        prefix="🔍 Specific Risks:\n"
    )

    mitigating_factors = dspy.OutputField(
        desc="Factors that reduce risk",
        type=List[Dict[str, Union[str, str, int]]],
        prefix="🛡️ Mitigating Factors:\n"
    )

    recommended_terms = dspy.OutputField(
        desc="Recommended loan/investment terms based on risk",
        type=Dict[str, Union[float, str, List[str], Dict[str, Union[float, str]]]],
        prefix="📋 Recommended Terms:\n"
    )

    approval_probability = dspy.OutputField(
        desc="Probability of approval with confidence intervals",
        type=Dict[str, Union[float, List[float], str]],
        prefix="✅ Approval Probability:\n"
    )

    monitoring_requirements = dspy.OutputField(
        desc="Ongoing monitoring and covenant requirements",
        type=List[Dict[str, Union[str, str, int]]],
        prefix="📈 Monitoring Requirements:\n"
    )

# Example 4: Legal Document Compliance Checker
class ComplianceArea(str, Enum):
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    CCPA = "ccpa"
    INDUSTRY_SPECIFIC = "industry_specific"

class LegalComplianceChecker(dspy.Signature):
    """Check legal documents for compliance across multiple regulations."""

    document_content = dspy.InputField(
        desc="Full text of the legal document",
        type=str,
        prefix="📄 Document Content:\n"
    )

    document_type = dspy.InputField(
        desc="Type of legal document (contract, policy, agreement, etc.)",
        type=str,
        prefix="📋 Document Type: "
    )

    jurisdiction = dspy.InputField(
        desc="Legal jurisdiction governing the document",
        type=str,
        prefix="⚖️ Jurisdiction: "
    )

    applicable_regulations = dspy.InputField(
        desc="List of regulations that apply to this document",
        type=List[ComplianceArea],
        prefix="🏛️ Applicable Regulations:\n"
    )

    industry_sector = dspy.InputField(
        desc="Industry sector for industry-specific compliance",
        type=str,
        optional=True,
        prefix="🏭 Industry Sector: "
    )

    # Compliance analysis outputs
    compliance_status = dspy.OutputField(
        desc="Overall compliance status with breakdown by regulation",
        type=Dict[str, Union[bool, str, List[str]]],
        prefix="✅ Compliance Status:\n"
    )

    violations_identified = dspy.OutputField(
        desc="Specific violations with severity and remediation steps",
        type=List[Dict[str, Union[str, int, List[str], Dict[str, str]]]],
        prefix="⚠️ Violations:\n"
    )

    missing_elements = dspy.OutputField(
        desc="Required elements missing from the document",
        type=List[Dict[str, Union[str, str]]],
        prefix="❌ Missing Elements:\n"
    )

    recommended_amendments = dspy.OutputField(
        desc="Specific amendments to ensure compliance",
        type=List[Dict[str, Union[str, str, Dict[str, str]]]],
        prefix="✏️ Recommended Amendments:\n"
    )

    risk_assessment = dspy.OutputField(
        desc="Legal and financial risk assessment",
        type=Dict[str, Union[str, float, List[str]]],
        prefix="⚖️ Risk Assessment:\n"
    )

    implementation_timeline = dspy.OutputField(
        desc="Timeline to achieve full compliance",
        type=Dict[str, Union[int, str, List[Dict[str, Union[str, int]]]]],
        prefix="📅 Implementation Timeline:\n"
    )

# Example 5: E-commerce Personalization Engine
class RecommendationType(str, Enum):
    PRODUCT = "product"
    CONTENT = "content"
    SERVICE = "service"
    EXPERIENCE = "experience"

class EcommercePersonalizer(dspy.Signature):
    """Hyper-personalized recommendations for e-commerce platforms."""

    # Customer and session data
    customer_id = dspy.InputField(
        desc="Unique customer identifier",
        type=str,
        prefix="👤 Customer ID: "
    )

    browsing_behavior = dspy.InputField(
        desc="Current browsing session data and patterns",
        type=Dict[str, Union[str, int, float, List[str], List[Dict[str, Any]]]],
        prefix="🖥️ Browsing Behavior:\n"
    )

    purchase_history = dspy.InputField(
        desc="Historical purchase data and patterns",
        type=List[Dict[str, Union[str, float, int, Dict[str, Any]]]],
        prefix="🛒 Purchase History:\n"
    )

    customer_preferences = dspy.InputField(
        desc="Explicit preferences and saved items",
        type=Dict[str, Union[List[str], Dict[str, Any], str]],
        prefix="❤️ Preferences:\n"
    )

    contextual_data = dspy.InputField(
        desc="Contextual factors (time, location, device, campaign)",
        type=Dict[str, Union[str, float, int, List[str]]],
        prefix="🌍 Context:\n"
    )

    inventory_data = dspy.InputField(
        desc="Available inventory and real-time stock levels",
        type=Dict[str, Union[List[Dict[str, Any]], Dict[str, float]]],
        prefix="📦 Inventory Data:\n"
    )

    # Personalization outputs
    personalized_recommendations = dspy.OutputField(
        desc="Hyper-personalized product recommendations with scores",
        type=List[Dict[str, Union[str, float, int, List[str], Dict[str, Union[str, float]]]]],
        prefix="🎯 Recommendations:\n"
    )

    recommendation_reasoning = dspy.OutputField(
        desc="Explanation for each recommendation category",
        type=Dict[str, Union[str, List[str], Dict[str, str]]],
        prefix="💭 Reasoning:\n"
    )

    pricing_strategy = dspy.OutputField(
        desc="Personalized pricing and discount recommendations",
        type=Dict[str, Union[float, str, List[Dict[str, Union[float, str]]]]],
        prefix="💰 Pricing Strategy:\n"
    )

    engagement_prediction = dspy.OutputField(
        desc="Predicted engagement metrics for recommendations",
        type=Dict[str, Union[float, List[float], str]],
        prefix="📈 Engagement Prediction:\n"
    )

    segmentation_insights = dspy.OutputField(
        desc="Customer segment insights and behavioral patterns",
        type=Dict[str, Union[str, List[str], Dict[str, Union[str, float]]]],
        prefix="👥 Segmentation Insights:\n"
    )

    a_b_test_suggestions = dspy.OutputField(
        desc="A/B test suggestions for optimization",
        type=List[Dict[str, Union[str, Dict[str, Any]]]],
        prefix="🧪 A/B Test Suggestions:\n"
    )

def demonstrate_real_world_applications():
    """Demonstrate real-world signature applications."""

    print("Real-World DSPy Signature Applications\n")
    print("=" * 60)

    # Initialize all the systems
    service_analyzer = dspy.Predict(CustomerIntelligenceAnalyzer)
    medical_triage = dspy.Predict(MedicalTriageSystem)
    risk_assessor = dspy.Predict(FinancialRiskAssessment)
    compliance_checker = dspy.Predict(LegalComplianceChecker)
    ecommerce_personalizer = dspy.Predict(EcommercePersonalizer)

    # Example 1: Customer Service Intelligence
    print("\n1. Customer Service Intelligence")
    print("-" * 40)
    service_result = service_analyzer(
        interaction_text="Customer is frustrated about delayed order and threatening to cancel subscription...",
        customer_profile={"tier": "gold", "value": 5000, "years_active": 3, "previous_issues": 1},
        interaction_context={"channel": "phone", "time": "2024-01-15 14:30", "previous_contacts": 2}
    )
    print(f"Sentiment: {service_result.customer_sentiment.get('overall', 'unknown')}")
    print(f"Churn Risk: {service_result.risk_assessment.get('churn_risk', 'unknown')}")
    print(f"Urgent Actions: {len([a for a in service_result.recommended_actions if a.get('urgent')])}")

    # Example 2: Medical Triage
    print("\n2. Medical Triage System")
    print("-" * 40)
    triage_result = medical_triage(
        chief_complaint="Chest pain and shortness of breath",
        symptom_details=[
            {"symptom": "chest_pain", "duration": "2 hours", "severity": 7},
            {"symptom": "shortness_of_breath", "duration": "1 hour", "severity": 6}
        ],
        patient_demographics={"age": 65, "gender": "male"},
        vital_signs={"heart_rate": 110, "blood_pressure": "160/95"}
    )
    print(f"Urgency: {triage_result.urgency_classification.get('level', 'unknown')}")
    print(f"Red Flags: {len(triage_result.red_flags)} identified")
    print(f"Top Diagnosis: {triage_result.differential_diagnosis[0].get('condition', 'unknown') if triage_result.differential_diagnosis else 'None'}")

    # Example 3: Financial Risk Assessment
    print("\n3. Financial Risk Assessment")
    print("-" * 40)
    risk_result = risk_assessor(
        entity_profile={"type": "business", "industry": "retail", "years_in_business": 5},
        financial_statements={"annual_revenue": 2000000, "profit_margin": 0.08, "debt_to_equity": 0.4},
        credit_history={"credit_score": 720, "late_payments": 1, "defaults": 0},
        loan_request={"amount": 500000, "purpose": "expansion", "term_months": 36},
        market_conditions={"interest_rate": 0.05, "growth_rate": 0.03}
    )
    print(f"Risk Score: {risk_result.overall_risk_score.get('score', 0)}/100")
    print(f"Risk Category: {risk_result.risk_category.get('category', 'unknown')}")
    print(f"Approval Probability: {risk_result.approval_probability.get('percentage', 0)}%")

    # Example 4: Legal Compliance
    print("\n4. Legal Compliance Check")
    print("-" * 40)
    compliance_result = compliance_checker(
        document_content="Privacy Policy: We collect user data for marketing purposes...",
        document_type="privacy_policy",
        jurisdiction="California",
        applicable_regulations=["gdpr", "ccpa"],
        industry_sector="technology"
    )
    print(f"Compliance Status: {compliance_result.compliance_status.get('overall', 'unknown')}")
    print(f"Violations: {len(compliance_result.violations_identified)}")
    print(f"Required Amendments: {len(compliance_result.recommended_amendments)}")

    # Example 5: E-commerce Personalization
    print("\n5. E-commerce Personalization")
    print("-" * 40)
    personalization_result = ecommerce_personalizer(
        customer_id="CUST_12345",
        browsing_behavior={
            "session_duration": 1800,
            "pages_viewed": 12,
            "categories": ["electronics", "gaming"],
            "searches": ["gaming laptop", "wireless mouse"]
        },
        purchase_history=[
            {"product": "gaming_headset", "date": "2024-01-01", "price": 99.99},
            {"product": "mechanical_keyboard", "date": "2023-12-15", "price": 149.99}
        ],
        customer_preferences={"brands": ["logitech", "razer"], "price_range": [50, 500]},
        contextual_data={"device": "desktop", "time_of_day": "evening", "campaign": "summer_sale"},
        inventory_data={"total_products": 1000, "categories_in_stock": 50}
    )
    print(f"Recommendations Generated: {len(personalization_result.personalized_recommendations)}")
    print(f"Top Category: {personalization_result.segmentation_insights.get('primary_segment', 'unknown')}")
    print(f"Engagement Score: {personalization_result.engagement_prediction.get('overall_score', 0)}/10")

    print("\n" + "=" * 60)
    print("All real-world applications demonstrated successfully!")

    # Additional insights
    print("\n💡 Key Insights:")
    print("• Typed signatures provide structure and reliability")
    print("• Domain-specific signatures capture business logic")
    print("• Complex systems can be built by composing signatures")
    print("• Real-world applications require comprehensive input/output handling")

if __name__ == "__main__":
    demonstrate_real_world_applications()
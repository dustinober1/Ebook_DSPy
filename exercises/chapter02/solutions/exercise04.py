"""
Exercise 4 Solutions: Domain-Specific Signatures - Healthcare Domain

This file contains a complete solution for Exercise 4, implementing a healthcare
domain-specific system for patient triage and diagnosis assistance.
"""

import dspy
from typing import List, Dict, Optional, Union, Literal
from enum import Enum
import json

# Domain definition
DOMAIN = "Healthcare - Patient Triage and Diagnosis Assistance"
DOMAIN_DESCRIPTION = """
This system assists healthcare professionals in:
1. Triage patients based on symptoms and severity
2. Provide preliminary diagnosis suggestions
3. Recommend diagnostic tests and procedures
4. Suggest initial treatment options
5. Identify red flags requiring immediate attention

The system is designed to support, not replace, medical professionals.
All recommendations should be validated by qualified healthcare providers.
"""

# Enums for type safety
class UrgencyLevel(str, Enum):
    ROUTINE = "routine"
    URGENT = "urgent"
    EMERGENCY = "emergency"
    CRITICAL = "critical"

class SymptomSeverity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    EXTREME = "extreme"

class DiagnosticTest(str, Enum):
    BLOOD_TEST = "blood_test"
    IMAGING_XRAY = "imaging_xray"
    IMAGING_CT = "imaging_ct"
    IMAGING_MRI = "imaging_mri"
    ECG = "ecg"
    URINE_TEST = "urine_test"
    BIOPSY = "biopsy"
    SWAB_TEST = "swab_test"

# Task 4.1 & 4.2: Primary Healthcare Signature

class PatientTriageSystem(dspy.Signature):
    """Comprehensive patient triage and preliminary diagnosis system."""

    # Patient presentation inputs
    chief_complaint = dspy.InputField(
        desc="Primary reason for medical consultation in patient's own words",
        type=str,
        prefix="🩺 Chief Complaint:\n"
    )

    symptom_details = dspy.InputField(
        desc="Detailed list of symptoms with onset, duration, severity, and progression",
        type=List[Dict[str, Union[str, int, float]]],
        prefix="📋 Symptom Details:\n"
    )

    patient_demographics = dspy.InputField(
        desc="Patient demographic information including age, gender, and relevant details",
        type=Dict[str, Union[str, int]],
        prefix="👤 Patient Demographics:\n"
    )

    medical_history = dspy.InputField(
        desc="Relevant past medical history, chronic conditions, and previous surgeries",
        type=str,
        optional=True,
        prefix="📚 Medical History:\n"
    )

    current_medications = dspy.InputField(
        desc="List of current medications, supplements, and dosages",
        type=List[Dict[str, Union[str, int, float]]],
        optional=True,
        prefix="💊 Current Medications:\n"
    )

    vital_signs = dspy.InputField(
        desc="Current vital signs including BP, HR, temperature, respiratory rate, oxygen saturation",
        type=Dict[str, Union[int, float, str]],
        optional=True,
        prefix="📊 Vital Signs:\n"
    )

    environmental_factors = dspy.InputField(
        desc="Environmental and lifestyle factors relevant to presentation",
        type=Dict[str, Union[str, bool, List[str]]],
        optional=True,
        prefix="🌍 Environmental Factors:\n"
    )

    # Triage outputs
    urgency_classification = dspy.OutputField(
        desc="Urgency level with detailed reasoning and timeframe for care",
        type=Dict[str, Union[str, List[str], int]],
        prefix="🚨 Urgency Classification:\n"
    )

    differential_diagnosis = dspy.OutputField(
        desc="Top differential diagnoses with probability scores and supporting evidence",
        type=List[Dict[str, Union[str, float, List[str], Dict[str, Union[str, bool]]]]],
        prefix="🔍 Differential Diagnosis:\n"
    )

    red_flags = dspy.OutputField(
        desc="Critical symptoms and findings requiring immediate attention",
        type=List[Dict[str, Union[str, str, List[str]]]],
        prefix="🚩 Red Flags:\n"
    )

    recommended_tests = dspy.OutputField(
        desc="Diagnostic tests to consider with urgency and rationale",
        type=List[Dict[str, Union[str, str, int, bool, List[str]]]],
        prefix="🧪 Recommended Tests:\n"
    )

    initial_treatment = dspy.OutputField(
        desc="Immediate treatment recommendations and comfort measures",
        type=Dict[str, Union[str, List[str], Dict[str, Union[str, str, int]]]]],
        prefix="💊 Initial Treatment:\n"
    )

    patient_education = dspy.OutputField(
        desc="Patient education points and self-care instructions",
        type=Dict[str, Union[List[str], Dict[str, Union[str, List[str]]]]],
        prefix="📚 Patient Education:\n"
    )

    follow_up_plan = dspy.OutputField(
        desc="Follow-up timeline, monitoring instructions, and when to seek further care",
        type=Dict[str, Union[str, int, List[str], Dict[str, Union[str, bool]]]],
        prefix="📅 Follow-up Plan:\n"
    )

# Task 4.3: Supporting Signatures

class SymptomAnalyzer(dspy.Signature):
    """Analyze individual symptoms for patterns and significance."""

    symptom = dspy.InputField(
        desc="Single symptom to analyze",
        type=str,
        prefix="💢 Symptom:\n"
    )

    symptom_context = dspy.InputField(
        desc="Context including onset, duration, triggers, and alleviating factors",
        type=Dict[str, Union[str, int, List[str]]],
        prefix="📝 Context:\n"
    )

    patient_context = dspy.InputField(
        desc="Relevant patient characteristics (age, gender, medical history)",
        type=Dict[str, Union[str, int, List[str]]],
        prefix="👤 Patient Context:\n"
    )

    symptom_significance = dspy.OutputField(
        desc="Clinical significance and concern level of the symptom",
        type=Dict[str, Union[str, int, List[str]]],
        prefix="⚠️ Significance:\n"
    )

    associated_conditions = dspy.OutputField(
        desc="Possible conditions associated with this symptom",
        type=List[Dict[str, Union[str, float, str]]],
        prefix="🏥 Associated Conditions:\n"
    )

    additional_questions = dspy.OutputField(
        desc="Clarifying questions to ask about this symptom",
        type=List[str],
        prefix="❓ Additional Questions:\n"
    )

class MedicationInteractionChecker(dspy.Signature):
    """Check for potential medication interactions and contraindications."""

    medications = dspy.InputField(
        desc="List of current medications with dosages",
        type=List[Dict[str, Union[str, float, int]]],
        prefix="💊 Medications:\n"
    )

    patient_conditions = dspy.InputField(
        desc="Patient's medical conditions and comorbidities",
        type=List[str],
        prefix="🏥 Conditions:\n"
    )

    new_medication = dspy.InputField(
        desc="New medication being considered (optional)",
        type=str,
        optional=True,
        prefix="➕ New Medication: "
    )

    interactions = dspy.OutputField(
        desc="Potential drug-drug interactions with severity levels",
        type=List[Dict[str, Union[str, str, int, List[str]]]],
        prefix("⚠️ Drug Interactions:\n")
    )

    contraindications = dspy.OutputField(
        desc="Contraindications and warnings for current conditions",
        type=List[Dict[str, Union[str, str, List[str]]]],
        prefix("🚫 Contraindications:\n")
    )

    recommendations = dspy.OutputField(
        desc="Recommendations for monitoring or medication adjustments",
        type=List[Dict[str, Union[str, str, bool]]],
        prefix("💡 Recommendations:\n")
    )

class DiagnosticTestSelector(dspy.Signature):
    """Select appropriate diagnostic tests based on clinical presentation."""

    clinical_presentation = dspy.InputField(
        desc="Summary of clinical presentation and key findings",
        type=str,
        prefix="🏥 Clinical Presentation:\n"
    )

    suspected_conditions = dspy.InputField(
        desc="List of suspected conditions with confidence levels",
        type=List[Dict[str, Union[str, float]]],
        prefix="🔍 Suspected Conditions:\n"
    )

    test_preferences = dspy.InputField(
        desc="Patient preferences and constraints for testing",
        type=Dict[str, Union[bool, str, List[str]]],
        optional=True,
        prefix="👤 Preferences:\n"
    )

    resource_constraints = dspy.InputField(
        desc="Available resources and facility capabilities",
        type=Dict[str, Union[bool, List[str]]],
        optional=True,
        prefix="🏥 Available Resources:\n"
    )

    test_recommendations = dspy.OutputField(
        desc="Recommended tests with priority, urgency, and rationale",
        type=List[Dict[str, Union[str, int, bool, float, List[str]]]],
        prefix("🧪 Test Recommendations:\n")
    )

    testing_sequence = dspy.OutputField(
        desc="Optimal sequence for performing recommended tests",
        type=List[Dict[str, Union[str, List[str], int]]],
        prefix("📋 Testing Sequence:\n")
    )

    alternative_tests = dspy.OutputField(
        desc="Alternative tests if primary recommendations not feasible",
        type=List[Dict[str, Union[str, List[str]]]],
        prefix("🔄 Alternative Options:\n")
    )

# Task 4.4: Usage Example and Workflow

class HealthcareWorkflowSystem:
    """Complete healthcare workflow system integrating all signatures."""

    def __init__(self):
        """Initialize all DSPy modules."""
        self.triage_system = dspy.Predict(PatientTriageSystem)
        self.symptom_analyzer = dspy.Predict(SymptomAnalyzer)
        self.medication_checker = dspy.Predict(MedicationInteractionChecker)
        self.test_selector = dspy.Predict(DiagnosticTestSelector)

    def process_patient_case(self,
                            chief_complaint: str,
                            symptoms: List[Dict[str, Union[str, int, float]]],
                            demographics: Dict[str, Union[str, int]],
                            medical_history: Optional[str] = None,
                            medications: Optional[List[Dict[str, Union[str, int, float]]]] = None,
                            vital_signs: Optional[Dict[str, Union[int, float, str]]] = None,
                            environmental_factors: Optional[Dict[str, Union[str, bool, List[str]]]] = None) -> Dict[str, Any]:
        """
        Process a complete patient case through the healthcare workflow.

        Returns:
            Comprehensive analysis and recommendations
        """

        results = {
            "case_id": f"CASE_{hash(chief_complaint) % 10000:04d}",
            "timestamp": "2024-01-15T10:30:00Z",
            "patient_demographics": demographics
        }

        # Step 1: Analyze individual symptoms
        print("Analyzing symptoms...")
        symptom_analyses = []
        for symptom in symptoms:
            analysis = self.symptom_analyzer(
                symptom=symptom["description"],
                symptom_context={
                    "onset": symptom.get("onset", "unknown"),
                    "duration": symptom.get("duration", "unknown"),
                    "severity": symptom.get("severity", 5),
                    "triggers": symptom.get("triggers", []),
                    "alleviating_factors": symptom.get("alleviating_factors", [])
                },
                patient_context={
                    "age": demographics.get("age", 0),
                    "gender": demographics.get("gender", "unknown"),
                    "conditions": medical_history or "None"
                }
            )
            symptom_analyses.append({
                "symptom": symptom["description"],
                "significance": analysis.symptom_significance,
                "associated_conditions": analysis.associated_conditions
            })

        results["symptom_analyses"] = symptom_analyses

        # Step 2: Check medication interactions if medications provided
        if medications:
            print("Checking medication interactions...")
            conditions_list = []
            if medical_history:
                # Simple extraction - would be more sophisticated in practice
                conditions_list = [medical_history]  # In reality, parse actual conditions

            med_check = self.medication_checker(
                medications=medications,
                patient_conditions=conditions_list
            )
            results["medication_analysis"] = {
                "interactions": med_check.interactions,
                "contraindications": med_check.contraindications,
                "recommendations": med_check.recommendations
            }

        # Step 3: Comprehensive triage and diagnosis
        print("Performing comprehensive triage...")
        triage_result = self.triage_system(
            chief_complaint=chief_complaint,
            symptom_details=symptoms,
            patient_demographics=demographics,
            medical_history=medical_history or "None reported",
            current_medications=medications or [],
            vital_signs=vital_signs or {},
            environmental_factors=environmental_factors or {}
        )

        results["triage"] = {
            "urgency": triage_result.urgency_classification,
            "diagnosis": triage_result.differential_diagnosis,
            "red_flags": triage_result.red_flags,
            "treatment": triage_result.initial_treatment,
            "education": triage_result.patient_education,
            "follow_up": triage_result.follow_up_plan
        }

        # Step 4: Select diagnostic tests
        print("Selecting diagnostic tests...")
        clinical_summary = f"""
        Patient: {demographics.get('age', 'unknown')}-year-old {demographics.get('gender', 'unknown')}
        Chief Complaint: {chief_complaint}
        Key Findings: {', '.join([s['description'] for s in symptoms[:3]])}
        """
        suspected_conditions = [
            {"condition": d.get("condition", "unknown"), "confidence": d.get("confidence", 0.5)}
            for d in triage_result.differential_diagnosis[:5]
        ]

        test_selection = self.test_selector(
            clinical_presentation=clinical_summary.strip(),
            suspected_conditions=suspected_conditions,
            test_preferences={
                "minimally_invasive": True,
                "insurance_coverage": "standard"
            },
            resource_constraints={
                "has_imaging": True,
                "has_lab": True,
                "specialist_available": False
            }
        )

        results["diagnostic_plan"] = {
            "recommended_tests": test_selection.test_recommendations,
            "testing_sequence": test_selection.testing_sequence,
            "alternatives": test_selection.alternative_tests
        }

        # Step 5: Generate care recommendations
        results["care_recommendations"] = self._generate_care_recommendations(results)

        return results

    def _generate_care_recommendations(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate integrated care recommendations from all analyses."""

        recommendations = {
            "immediate_actions": [],
            "monitoring_parameters": [],
            "patient_instructions": [],
            "follow_up_schedule": {},
            "when_to_seek_care": []
        }

        # Extract urgency
        urgency = analysis_results["triage"]["urgency"].get("level", "routine")

        # Immediate actions based on urgency
        if urgency in ["emergency", "critical"]:
            recommendations["immediate_actions"].extend([
                "Proceed to nearest emergency department",
                "Continuous vital sign monitoring",
                "IV access establishment"
            ])
        elif urgency == "urgent":
            recommendations["immediate_actions"].extend([
                "Schedule urgent appointment within 24 hours",
                "Begin symptomatic treatment",
                "Arrange necessary diagnostic tests"
            ])

        # Add medication-related recommendations
        if "medication_analysis" in analysis_results:
            med_analysis = analysis_results["medication_analysis"]
            if med_analysis["interactions"]:
                recommendations["patient_instructions"].append(
                    "Review all medications with healthcare provider due to potential interactions"
                )
            if med_analysis["contraindications"]:
                recommendations["patient_instructions"].append(
                    "Certain current medications may be contraindicated - seek medical review"
                )

        # Add monitoring based on red flags
        red_flags = analysis_results["triage"].get("red_flags", [])
        for flag in red_flags:
            if "fever" in flag.get("description", "").lower():
                recommendations["monitoring_parameters"].append("Temperature monitoring every 4 hours")
            if "pain" in flag.get("description", "").lower():
                recommendations["monitoring_parameters"].append("Pain score assessment regularly")

        # Add when to seek care instructions
        recommendations["when_to_seek_care"].extend([
            "If symptoms worsen or new symptoms develop",
            "If experiencing severe pain or shortness of breath",
            "If fever persists for more than 48 hours",
            "If unable to tolerate oral fluids"
        ])

        return recommendations

def demonstrate_healthcare_system():
    """Demonstrate the healthcare system with a realistic case."""

    print("=" * 60)
    print("Healthcare Patient Triage System Demonstration")
    print("=" * 60)

    # Initialize system
    system = HealthcareWorkflowSystem()

    # Sample patient case
    chief_complaint = "Chest pain and shortness of breath for 2 hours"

    symptoms = [
        {
            "description": "Chest pain",
            "onset": "2 hours ago",
            "duration": "continuous",
            "severity": 8,
            "location": "center of chest",
            "radiation": "to left arm",
            "character": "pressure-like"
        },
        {
            "description": "Shortness of breath",
            "onset": "2 hours ago",
            "duration": "continuous",
            "severity": 7,
            "worse_with": "exertion",
            "better_with": "rest"
        },
        {
            "description": "Sweating",
            "onset": "1 hour ago",
            "severity": 5,
            "description": "profuse sweating"
        }
    ]

    demographics = {
        "age": 65,
        "gender": "male",
        "weight": 180,
        "height": 72
    }

    medical_history = "Hypertension, Type 2 Diabetes, High Cholesterol, Former smoker (quit 5 years ago)"

    medications = [
        {"name": "Lisinopril", "dosage": "10mg", "frequency": "daily"},
        {"name": "Metformin", "dosage": "500mg", "frequency": "twice daily"},
        {"name": "Atorvastatin", "dosage": "20mg", "frequency": "daily"}
    ]

    vital_signs = {
        "blood_pressure": "160/95",
        "heart_rate": 110,
        "temperature": "98.6°F",
        "respiratory_rate": 22,
        "oxygen_saturation": 94%
    }

    # Process the case
    print(f"\n🏥 Processing patient: {demographics['age']}-year-old {demographics['gender']}")
    print(f"Chief Complaint: {chief_complaint}\n")

    results = system.process_patient_case(
        chief_complaint=chief_complaint,
        symptoms=symptoms,
        demographics=demographics,
        medical_history=medical_history,
        medications=medications,
        vital_signs=vital_signs
    )

    # Display key results
    print("\n" + "=" * 60)
    print("TRIAGE RESULTS")
    print("=" * 60)

    # Urgency
    urgency = results["triage"]["urgency"]
    print(f"\n🚨 Urgency Level: {urgency.get('level', 'unknown').upper()}")
    print(f"   Reasoning: {urgency.get('reasoning', 'Not provided')}")
    print(f"   Timeframe: {urgency.get('timeframe', 'Not specified')}")

    # Top diagnosis
    diagnoses = results["triage"]["diagnosis"]
    if diagnoses:
        top_dx = diagnoses[0]
        print(f"\n🔍 Top Differential Diagnosis:")
        print(f"   Condition: {top_dx.get('condition', 'Unknown')}")
        print(f"   Confidence: {top_dx.get('confidence', 0):.1%}")
        print(f"   Supporting: {', '.join(top_dx.get('supporting_evidence', [])[:2])}")

    # Red flags
    red_flags = results["triage"].get("red_flags", [])
    if red_flags:
        print(f"\n🚩 Red Flags ({len(red_flags)}):")
        for flag in red_flags[:3]:
            print(f"   • {flag.get('description', 'Unspecified')}")

    # Diagnostic tests
    tests = results["diagnostic_plan"]["recommended_tests"]
    if tests:
        print(f"\n🧪 Recommended Diagnostic Tests:")
        for test in tests[:3]:
            print(f"   • {test.get('name', 'Unknown')} - {test.get('urgency', 'routine')} priority")

    # Care recommendations
    care = results["care_recommendations"]
    if care["immediate_actions"]:
        print(f"\n⚡ Immediate Actions:")
        for action in care["immediate_actions"]:
            print(f"   • {action}")

    # Follow-up
    follow_up = results["triage"]["follow_up"]
    print(f"\n📅 Follow-up Plan:")
    print(f"   Timeline: {follow_up.get('timeline', 'Not specified')}")
    print(f"   Monitoring: {', '.join(follow_up.get('monitoring_instructions', [])[:2])}")

    print("\n" + "=" * 60)
    print("Note: This is a demonstration system only.")
    print("All medical decisions should be made by qualified healthcare professionals.")
    print("=" * 60)

if __name__ == "__main__":
    demonstrate_healthcare_system()
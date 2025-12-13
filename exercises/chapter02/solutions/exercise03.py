"""
Exercise 3 Solutions: Complex Multi-Field Signatures

This file contains solutions for Exercise 3 on creating complex, multi-field signatures
for legal document analysis.
"""

import dspy
from typing import List, Dict, Optional, Union, Literal
from datetime import datetime

# Task 3.1: Complete Legal Document Analyzer

class LegalDocumentAnalyzer(dspy.Signature):
    """Comprehensive legal document analysis and review."""

    # Input fields
    document_text = dspy.InputField(
        desc="Full text of the legal document",
        type=str,
        prefix="📄 Legal Document:\n"
    )

    document_type = dspy.InputField(
        desc="Type of legal document (contract, agreement, policy, etc.)",
        type=Literal["contract", "agreement", "policy", "terms", "privacy", "nda", "other"],
        prefix="📋 Document Type: "
    )

    jurisdiction = dspy.InputField(
        desc="Legal jurisdiction governing the document",
        type=str,
        prefix="⚖️ Jurisdiction: "
    )

    parties_involved = dspy.InputField(
        desc="List of parties involved in the agreement",
        type=List[Dict[str, str]],
        prefix="👥 Parties:\n"
    )

    effective_date = dspy.InputField(
        desc="Date when the document takes effect",
        type=str,
        prefix="📅 Effective Date: "
    )

    review_focus = dspy.InputField(
        desc="Specific areas to focus on during review",
        type=List[str],
        optional=True,
        prefix="🎯 Review Focus:\n"
    )

    # Output fields
    executive_summary = dspy.OutputField(
        desc="Brief summary for non-legal stakeholders",
        type=str,
        prefix="📝 Executive Summary:\n"
    )

    key_obligations = dspy.OutputField(
        desc="Key obligations imposed by the document",
        type=List[Dict[str, Union[str, List[str], Dict[str, str]]]],
        prefix="📋 Key Obligations:\n"
    )

    critical_clauses = dspy.OutputField(
        desc="Most important clauses with explanations",
        type=List[Dict[str, Union[str, int, List[str]]]],
        prefix("🔑 Critical Clauses:\n")
    )

    risk_assessment = dspy.OutputField(
        desc="Assessment of legal and business risks",
        type=Dict[str, Union[float, str, List[Dict[str, Union[str, int, float]]]]],
        prefix="⚠️ Risk Assessment:\n"
    )

    compliance_issues = dspy.OutputField(
        desc="Potential compliance and regulatory issues",
        type=List[Dict[str, Union[str, int, List[str]]]],
        prefix="🚨 Compliance Issues:\n"
    )

    amendment_suggestions = dspy.OutputField(
        desc="Suggested amendments and improvements",
        type=List[Dict[str, Union[str, str, Dict[str, str]]]],
        prefix="✏️ Amendment Suggestions:\n"
    )

    overall_rating = dspy.OutputField(
        desc="Overall document quality and fairness rating",
        type=Dict[str, Union[int, str, List[str]]],
        prefix="⭐ Overall Rating:\n"
    )

# Task 3.2: Helper Signatures

class ClauseExtractor(dspy.Signature):
    """Extract and categorize legal clauses from documents."""

    document_text = dspy.InputField(
        desc="Text of the legal document",
        type=str,
        prefix="📄 Document Text:\n"
    )

    clause_types = dspy.InputField(
        desc="Types of clauses to extract",
        type=List[Literal[
            "liability", "termination", "payment", "confidentiality",
            "intellectual_property", "dispute_resolution", "indemnification",
            "force_majeure", "governing_law", "assignment", "non_compete"
        ]],
        prefix="🏷️ Clause Types:\n"
    )

    extracted_clauses = dspy.OutputField(
        desc="Extracted clauses with types and importance scores",
        type=List[Dict[str, Union[str, int, float, Dict[str, Union[int, str]]]]],
        prefix="📋 Extracted Clauses:\n"
    )

    clause_summary = dspy.OutputField(
        desc="Summary of clauses found by type",
        type=Dict[str, Union[int, List[str]]],
        prefix="📊 Clause Summary:\n"
    )

    missing_clauses = dspy.OutputField(
        desc="Important clause types that were not found",
        type=List[Dict[str, Union[str, str]]],
        prefix="❌ Missing Clauses:\n"
    )

class RiskAssessor(dspy.Signature):
    """Assess legal and financial risks in contracts."""

    clauses = dspy.InputField(
        desc="List of clauses from the document",
        type=List[Dict[str, Union[str, Dict[str, Any]]]],
        prefix="📋 Clauses:\n"
    )

    business_context = dspy.InputField(
        desc="Business context and risk tolerance",
        type=Dict[str, Union[str, float]],
        prefix="🏢 Business Context:\n"
    )

    risk_categories = dspy.InputField(
        desc="Categories of risks to assess",
        type=List[Literal[
            "financial", "legal", "operational", "reputational",
            "regulatory", "compliance", "strategic"
        ]],
        prefix="⚠️ Risk Categories:\n"
    )

    identified_risks = dspy.OutputField(
        desc="Specific risks identified with severity and likelihood",
        type=List[Dict[str, Union[str, int, float, List[str]]]],
        prefix="🚨 Identified Risks:\n"
    )

    risk_mitigation = dspy.OutputField(
        desc="Strategies to mitigate identified risks",
        type=Dict[str, List[Dict[str, Union[str, str, int]]]],
        prefix="🛡️ Risk Mitigation:\n"
    )

    overall_risk_score = dspy.OutputField(
        desc="Overall risk assessment score",
        type=Dict[str, Union[float, str, List[str]]],
        prefix="📊 Overall Risk Score:\n"
    )

class AmendmentGenerator(dspy.Signature):
    """Generate specific amendments to improve documents."""

    issues_identified = dspy.InputField(
        desc="List of issues found in the document",
        type=List[Dict[str, Union[str, int, List[str]]]],
        prefix="❌ Issues:\n"
    )

    improvement_goals = dspy.InputField(
        desc="Goals for the amendments (fairness, clarity, protection)",
        type=List[str],
        prefix="🎯 Improvement Goals:\n"
    )

    proposed_amendments = dspy.OutputField(
        desc="Specific text amendments with explanations",
        type=List[Dict[str, Union[str, str, Dict[str, Union[str, int]]]]],
        prefix="✏️ Proposed Amendments:\n"
    )

    negotiation_points = dspy.OutputField(
        desc "Key points for negotiation based on amendments",
        type=List[Dict[str, Union[str, int, bool]]],
        prefix="💼 Negotiation Points:\n"
    )

    impact_assessment = dspy.OutputField(
        desc="Impact of amendments on each party",
        type=Dict[str, Dict[str, Union[str, int, float]]],
        prefix="📈 Impact Assessment:\n"
    )

# Task 3.3: Signature Composition Workflow

class LegalDocumentProcessor:
    """Complete legal document processing pipeline."""

    def __init__(self):
        """Initialize all DSPy modules."""
        self.clause_extractor = dspy.Predict(ClauseExtractor)
        self.risk_assessor = dspy.Predict(RiskAssessor)
        self.amendment_generator = dspy.Predict(AmendmentGenerator)
        self.document_analyzer = dspy.Predict(LegalDocumentAnalyzer)

    def analyze_document_complete(self,
                                 document_text: str,
                                 document_type: str,
                                 jurisdiction: str,
                                 parties_involved: List[Dict[str, str]],
                                 effective_date: str,
                                 business_context: Optional[Dict[str, Union[str, float]]] = None,
                                 review_focus: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Complete document analysis workflow.

        Returns:
            Dictionary containing all analysis results
        """

        results = {
            "document_metadata": {
                "type": document_type,
                "jurisdiction": jurisdiction,
                "parties": parties_involved,
                "effective_date": effective_date,
                "analysis_timestamp": datetime.now().isoformat()
            }
        }

        # Step 1: Extract clauses
        print("Step 1: Extracting clauses...")
        clause_types = [
            "liability", "termination", "payment", "confidentiality",
            "intellectual_property", "dispute_resolution", "indemnification",
            "force_majeure", "governing_law"
        ]

        clause_result = self.clause_extractor(
            document_text=document_text,
            clause_types=clause_types
        )

        results["clause_extraction"] = {
            "clauses": clause_result.extracted_clauses,
            "summary": clause_result.clause_summary,
            "missing": clause_result.missing_clauses
        }

        # Step 2: Assess risks
        print("Step 2: Assessing risks...")
        if business_context is None:
            business_context = {
                "risk_tolerance": "medium",
                "industry": "general",
                "deal_value": "standard"
            }

        risk_categories = ["financial", "legal", "operational", "reputational", "compliance"]

        risk_result = self.risk_assessor(
            clauses=clause_result.extracted_clauses,
            business_context=business_context,
            risk_categories=risk_categories
        )

        results["risk_assessment"] = {
            "identified_risks": risk_result.identified_risks,
            "mitigation_strategies": risk_result.risk_mitigation,
            "overall_score": risk_result.overall_risk_score
        }

        # Step 3: Generate amendments
        print("Step 3: Generating amendment suggestions...")
        improvement_goals = ["fairness", "clarity", "protection", "compliance"]

        # Compile issues from risk assessment
        issues_for_amendment = []
        for risk in risk_result.identified_risks:
            issues_for_amendment.append({
                "type": risk.get("category", "general"),
                "description": risk.get("description", "Unspecified risk"),
                "severity": risk.get("severity", 3),
                "clauses_involved": risk.get("clauses", [])
            })

        if issues_for_amendment:
            amendment_result = self.amendment_generator(
                issues_identified=issues_for_amendment,
                improvement_goals=improvement_goals
            )

            results["amendments"] = {
                "proposed": amendment_result.proposed_amendments,
                "negotiation_points": amendment_result.negotiation_points,
                "impact": amendment_result.impact_assessment
            }

        # Step 4: Comprehensive analysis
        print("Step 4: Performing comprehensive analysis...")
        comprehensive_result = self.document_analyzer(
            document_text=document_text,
            document_type=document_type,
            jurisdiction=jurisdiction,
            parties_involved=parties_involved,
            effective_date=effective_date,
            review_focus=review_focus or []
        )

        results["comprehensive_analysis"] = {
            "executive_summary": comprehensive_result.executive_summary,
            "key_obligations": comprehensive_result.key_obligations,
            "critical_clauses": comprehensive_result.critical_clauses,
            "compliance_issues": comprehensive_result.compliance_issues,
            "overall_rating": comprehensive_result.overall_rating
        }

        # Add cross-references
        results["cross_references"] = {
            "clauses_to_risks": self._map_clauses_to_risks(
                clause_result.extracted_clauses,
                risk_result.identified_risks
            ),
            "amendments_to_risks": self._map_amendments_to_risks(
                results.get("amendments", {}).get("proposed", []),
                risk_result.identified_risks
            )
        }

        return results

    def _map_clauses_to_risks(self,
                             clauses: List[Dict[str, Any]],
                             risks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Map identified clauses to specific risks."""
        mappings = []
        for risk in risks:
            for clause_id in risk.get("clauses", []):
                clause = next((c for c in clauses if c.get("id") == clause_id), None)
                if clause:
                    mappings.append({
                        "clause_type": clause.get("type", "unknown"),
                        "risk_description": risk.get("description", "unspecified"),
                        "risk_severity": risk.get("severity", 0)
                    })
        return mappings

    def _map_amendments_to_risks(self,
                                amendments: List[Dict[str, Any]],
                                risks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Map proposed amendments to the risks they address."""
        mappings = []
        for amendment in amendments:
            # This would be more sophisticated in practice
            # For now, assume amendments address the issues they were generated from
            mappings.append({
                "amendment_type": amendment.get("type", "general"),
                "addresses_risk": amendment.get("addresses", "unspecified"),
                "risk_mitigation": amendment.get("explanation", "")[:100] + "..."
            })
        return mappings

# Example usage and testing

def demonstrate_legal_analysis():
    """Demonstrate the complete legal document analysis system."""

    print("=" * 60)
    print("Legal Document Analysis System Demonstration")
    print("=" * 60)

    # Initialize processor
    processor = LegalDocumentProcessor()

    # Sample document (simplified)
    sample_document = """
    SERVICE AGREEMENT

    This Service Agreement ("Agreement") is entered into on January 1, 2024
    between TechCorp Inc. ("Service Provider") and ClientXYZ LLC ("Client").

    1. SERVICES
    Service Provider shall provide software development services as outlined in Exhibit A.

    2. PAYMENT TERMS
    Client shall pay Service Provider $50,000 upon signing and $50,000 upon completion.

    3. LIABILITY
    Service Provider's total liability under this Agreement shall not exceed
    the total fees paid by Client.

    4. TERMINATION
    Either party may terminate this Agreement with 30 days written notice.

    5. CONFIDENTIALITY
    Both parties shall maintain confidentiality of all proprietary information.
    """

    # Document metadata
    doc_type = "contract"
    jurisdiction = "California"
    parties = [
        {"name": "TechCorp Inc.", "role": "Service Provider", "type": "corporation"},
        {"name": "ClientXYZ LLC", "role": "Client", "type": "llc"}
    ]
    effective_date = "2024-01-01"

    # Business context
    business_context = {
        "risk_tolerance": "low",
        "industry": "technology",
        "deal_value": 100000
    }

    # Review focus
    focus_areas = ["liability", "payment_terms", "termination"]

    # Perform analysis
    print("\n🔍 Starting comprehensive document analysis...\n")
    results = processor.analyze_document_complete(
        document_text=sample_document,
        document_type=doc_type,
        jurisdiction=jurisdiction,
        parties_involved=parties,
        effective_date=effective_date,
        business_context=business_context,
        review_focus=focus_areas
    )

    # Display results summary
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS SUMMARY")
    print("=" * 60)

    # Clause summary
    print(f"\n📋 Clauses Extracted: {len(results['clause_extraction']['clauses'])}")
    print(f"   - Payment clauses: {results['clause_extraction']['summary'].get('payment', 0)}")
    print(f"   - Liability clauses: {results['clause_extraction']['summary'].get('liability', 0)}")
    print(f"   - Missing clauses: {len(results['clause_extraction']['missing'])}")

    # Risk summary
    risks = results['risk_assessment']['identified_risks']
    print(f"\n⚠️ Risks Identified: {len(risks)}")
    for risk in risks[:3]:  # Show top 3
        print(f"   - {risk.get('category', 'General')}: {risk.get('severity', 'N/A')} severity")

    # Overall risk score
    overall_score = results['risk_assessment']['overall_score']
    print(f"\n📊 Overall Risk Score: {overall_score.get('score', 'N/A')}/10")

    # Amendments
    if 'amendments' in results:
        print(f"\n✏️ Amendment Suggestions: {len(results['amendments']['proposed'])}")
        for amendment in results['amendments']['proposed'][:2]:  # Show top 2
            print(f"   - {amendment.get('type', 'General')} amendment")

    # Overall rating
    rating = results['comprehensive_analysis']['overall_rating']
    print(f"\n⭐ Overall Document Rating: {rating.get('score', 'N/A')}/10")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

if __name__ == "__main__":
    demonstrate_legal_analysis()
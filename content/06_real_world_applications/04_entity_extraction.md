# Entity Extraction: Mining Structured Information from Text

## Introduction

Entity extraction (also known as Named Entity Recognition or NER) is the process of identifying and categorizing specific pieces of information from unstructured text. This critical technology powers everything from resume parsing and contract analysis to medical record processing and financial document extraction. DSPy provides robust tools for building sophisticated entity extraction systems that can handle complex, real-world scenarios.

## Understanding Entity Extraction

### Common Entity Types

1. **Person**: Names of individuals (John Smith, Dr. Sarah Johnson)
2. **Organization**: Companies, institutions (Google, Microsoft, Stanford University)
3. **Location**: Places, addresses (New York, 123 Main Street)
4. **Date/Time**: Temporal expressions (January 15, 2024, 3:30 PM)
5. **Money**: Monetary values ($50,000, €1.2 million)
6. **Product**: Commercial products (iPhone 15, Toyota Camry)
7. **Event**: Named events (World War II, Olympics 2024)
8. **Custom**: Domain-specific entities (Medical codes, Legal references)

### Real-World Applications

- **Resume Processing**: Extract skills, experience, education
- **Contract Analysis**: Identify parties, dates, clauses
- **Medical Records**: Extract diagnoses, medications, procedures
- **Financial Documents**: Extract amounts, dates, companies
- **News Articles**: Identify people, organizations, events
- **Customer Reviews**: Extract products, features, sentiments

## Building Entity Extractors

### Basic Entity Extractor

```python
import dspy
from typing import List, Dict, Any

class BasicEntityExtractor(dspy.Module):
    def __init__(self, entity_types):
        super().__init__()
        self.entity_types = entity_types
        types_str = ", ".join(entity_types)
        self.extract = dspy.Predict(
            f"text, entity_types[{types_str}] -> entities"
        )

    def forward(self, text):
        result = self.extract(
            text=text,
            entity_types=", ".join(self.entity_types)
        )

        # Parse the extracted entities
        entities = self._parse_entities(result.entities)

        return dspy.Prediction(
            entities=entities,
            raw_output=result.entities
        )

    def _parse_entities(self, entities_text):
        """Parse raw entity text into structured format."""
        entities = []
        if not entities_text:
            return entities

        # Simple parsing - assumes format: "TYPE: entity1, entity2"
        lines = entities_text.strip().split('\n')
        for line in lines:
            if ':' in line:
                entity_type, entity_list = line.split(':', 1)
                entity_type = entity_type.strip()
                for entity in entity_list.split(','):
                    entity = entity.strip()
                    if entity:
                        entities.append({
                            "text": entity,
                            "type": entity_type,
                            "confidence": 0.8  # Default confidence
                        })

        return entities
```

### Advanced Entity Extractor with Context

```python
class AdvancedEntityExtractor(dspy.Module):
    def __init__(self, entity_types, context_window=100):
        super().__init__()
        self.entity_types = entity_types
        self.context_window = context_window
        types_str = ", ".join(entity_types)

        self.find_entities = dspy.ChainOfThought(
            f"text, context, entity_types[{types_str}] -> entities_with_positions"
        )

        self.validate_entities = dspy.Predict(
            "entity, text_context -> is_valid, corrected_entity, confidence"
        )

        self.disambiguate = dspy.Predict(
            "entity, context, possible_meanings -> disambiguated_entity, reasoning"
        )

    def forward(self, text, document_context=None):
        if document_context:
            context = document_context[-self.context_window:]
        else:
            context = text

        # Find entities with positions
        extraction = self.find_entities(
            text=text,
            context=context,
            entity_types=", ".join(self.entity_types)
        )

        # Parse and validate entities
        entities = []
        for entity_info in self._parse_entities_with_positions(extraction.entities_with_positions):
            # Validate each entity
            validation = self.validate_entities(
                entity=entity_info["text"],
                text_context=text[max(0, entity_info["start"]-50):entity_info["end"]+50]
            )

            if validation.is_valid.lower() == "yes":
                # Disambiguate if needed
                if entity_info["type"] in ["PERSON", "ORGANIZATION"]:
                    disambiguation = self.disambiguate(
                        entity=validation.corrected_entity,
                        context=context,
                        possible_meanings="Multiple possible matches"
                    )
                    final_entity = disambiguation.disambiguated_entity
                else:
                    final_entity = validation.corrected_entity

                entities.append({
                    "text": final_entity,
                    "type": entity_info["type"],
                    "start": entity_info["start"],
                    "end": entity_info["end"],
                    "confidence": float(validation.confidence),
                    "original": entity_info["text"]
                })

        return dspy.Prediction(
            entities=entities,
            extraction_reasoning=extraction.rationale
        )

    def _parse_entities_with_positions(self, entities_text):
        """Parse entities with their positions."""
        # Assuming format: "TYPE: entity (start-end), entity (start-end)"
        entities = []
        lines = entities_text.strip().split('\n')

        for line in lines:
            if ':' in line:
                entity_type, entities_list = line.split(':', 1)
                entity_type = entity_type.strip()

                for entity_match in entities_list.split(','):
                    entity_match = entity_match.strip()
                    if '(' in entity_match and ')' in entity_match:
                        entity_text = entity_match[:entity_match.rfind('(')].strip()
                        positions = entity_match[entity_match.rfind('(')+1:-1].split('-')
                        if len(positions) == 2:
                            entities.append({
                                "text": entity_text,
                                "type": entity_type,
                                "start": int(positions[0]),
                                "end": int(positions[1])
                            })

        return entities
```

### Relation Extractor

```python
class RelationExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract_relations = dspy.ChainOfThought(
            "entities, text -> relations"
        )

    def forward(self, text, entities):
        # Prepare entities context
        entities_context = "\n".join([
            f"{e['type']}: {e['text']} (position: {e.get('start', 'N/A')})"
            for e in entities
        ])

        # Extract relations between entities
        relations = self.extract_relations(
            entities=entities_context,
            text=text
        )

        parsed_relations = self._parse_relations(relations.relations)

        return dspy.Prediction(
            relations=parsed_relations,
            reasoning=relations.rationale
        )

    def _parse_relations(self, relations_text):
        """Parse relations text into structured format."""
        relations = []
        if not relations_text:
            return relations

        # Assuming format: "SUBJECT -> RELATION -> OBJECT"
        lines = relations_text.strip().split('\n')
        for line in lines:
            if '->' in line:
                parts = [p.strip() for p in line.split('->')]
                if len(parts) == 3:
                    relations.append({
                        "subject": parts[0],
                        "relation": parts[1],
                        "object": parts[2]
                    })

        return relations
```

## Specialized Entity Extraction Applications

### Resume/CV Parser

```python
class ResumeParser(dspy.Module):
    def __init__(self):
        super().__init__()
        self.contact_info = dspy.Predict(
            "resume_text -> name, email, phone, location, linkedin"
        )

        self.extract_sections = dspy.Predict(
            "resume_text -> work_experience, education, skills, certifications"
        )

        self.parse_experience = dspy.ChainOfThought(
            "experience_section -> detailed_experiences"
        )

        self.parse_education = dspy.Predict(
            "education_section -> schools, degrees, graduation_years"
        )

    def forward(self, resume_text):
        # Extract contact information
        contact = self.contact_info(resume_text=resume_text)

        # Identify and extract sections
        sections = self.extract_sections(resume_text=resume_text)

        # Parse work experience
        experience_details = []
        if sections.work_experience:
            exp_parsed = self.parse_experience(experience_section=sections.work_experience)
            experience_details = self._parse_experience_details(exp_parsed.detailed_experiences)

        # Parse education
        education_details = []
        if sections.education:
            edu_parsed = self.parse_education(education_section=sections.education)
            education_details = self._parse_education_details(edu_parsed)

        # Parse skills
        skills = []
        if sections.skills:
            skills = [s.strip() for s in sections.skills.split(',')]

        return dspy.Prediction(
            contact_info={
                "name": contact.name,
                "email": contact.email,
                "phone": contact.phone,
                "location": contact.location,
                "linkedin": contact.linkedin
            },
            work_experience=experience_details,
            education=education_details,
            skills=skills,
            certifications=sections.certifications.split(',') if sections.certifications else []
        )

    def _parse_experience_details(self, experience_text):
        """Parse detailed work experience."""
        experiences = []
        # Parse each job entry
        for job in experience_text.split('\n\n'):
            if job.strip():
                experiences.append(self._parse_single_job(job))
        return experiences

    def _parse_single_job(self, job_text):
        """Parse a single job entry."""
        # Simple parsing - in practice, would be more sophisticated
        lines = job_text.split('\n')
        title_company = lines[0] if lines else ""
        return {
            "title_company": title_company,
            "details": lines[1:] if len(lines) > 1 else []
        }

    def _parse_education_details(self, education_text):
        """Parse education information."""
        schools = []
        for school in education_text.schools.split(','):
            schools.append({"name": school.strip()})
        return schools
```

### Contract Analyzer

```python
class ContractAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract_parties = dspy.Predict(
            "contract_text -> parties, roles"
        )

        self.extract_dates = dspy.Predict(
            "contract_text -> effective_date, termination_date, key_dates"
        )

        self.extract_financials = dspy.Predict(
            "contract_text -> payment_terms, amounts, penalties"
        )

        self.identify_clauses = dspy.ChainOfThought(
            "contract_text -> important_clauses, obligations"
        )

    def forward(self, contract_text):
        # Extract parties involved
        parties = self.extract_parties(contract_text=contract_text)

        # Extract important dates
        dates = self.extract_dates(contract_text=contract_text)

        # Extract financial information
        financials = self.extract_financials(contract_text=contract_text)

        # Identify key clauses
        clauses = self.identify_clauses(contract_text=contract_text)

        return dspy.Prediction(
            parties={
                "entities": parties.parties.split(','),
                "roles": parties.roles
            },
            dates={
                "effective_date": dates.effective_date,
                "termination_date": dates.termination_date,
                "key_dates": dates.key_dates.split(',') if dates.key_dates else []
            },
            financials={
                "payment_terms": financials.payment_terms,
                "amounts": financials.amounts.split(',') if financials.amounts else [],
                "penalties": financials.penalties
            },
            clauses={
                "important": self._parse_clauses(clauses.important_clauses),
                "obligations": self._parse_obligations(clauses.obligations)
            },
            reasoning=clauses.rationale
        )

    def _parse_clauses(self, clauses_text):
        """Parse contract clauses."""
        return [c.strip() for c in clauses_text.split(';') if c.strip()]

    def _parse_obligations(self, obligations_text):
        """Parse contractual obligations."""
        obligations = []
        for obligation in obligations_text.split('\n'):
            if obligation.strip():
                obligations.append(obligation.strip())
        return obligations
```

### Medical Record Processor

```python
class MedicalRecordProcessor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract_patient_info = dspy.Predict(
            "medical_record -> patient_name, age, gender, id"
        )

        self.extract_diagnoses = dspy.Predict(
            "medical_record -> diagnoses, icd_codes, symptoms"
        )

        self.extract_medications = dspy.Predict(
            "medical_record -> medications, dosages, frequencies"
        )

        self.extract_procedures = dspy.Predict(
            "medical_record -> procedures, dates, providers"
        )

        self.extract_vitals = dspy.Predict(
            "medical_record -> vital_signs, values, dates"
        )

    def forward(self, medical_record):
        # Extract patient demographics
        patient = self.extract_patient_info(medical_record=medical_record)

        # Extract medical information
        diagnoses = self.extract_diagnoses(medical_record=medical_record)
        medications = self.extract_medications(medical_record=medical_record)
        procedures = self.extract_procedures(medical_record=medical_record)
        vitals = self.extract_vitals(medical_record=medical_record)

        return dspy.Prediction(
            patient_info={
                "name": patient.patient_name,
                "age": patient.age,
                "gender": patient.gender,
                "medical_id": patient.id
            },
            medical_info={
                "diagnoses": self._parse_medical_list(diagnoses.diagnoses),
                "icd_codes": diagnoses.icd_codes.split(',') if diagnoses.icd_codes else [],
                "symptoms": self._parse_medical_list(diagnoses.symptoms)
            },
            medications=self._parse_medications(medications.medications, medications.dosages, medications.frequencies),
            procedures=self._parse_procedures(procedures.procedures, procedures.dates, procedures.providers),
            vitals=self._parse_vitals(vitals.vital_signs, vitals.values, vitals.dates)
        )

    def _parse_medical_list(self, list_text):
        """Parse comma-separated medical items."""
        return [item.strip() for item in list_text.split(',') if item.strip()]

    def _parse_medications(self, meds_text, dosages_text, frequencies_text):
        """Parse medication information."""
        medications = []
        meds = meds_text.split(',') if meds_text else []
        dosages = dosages_text.split(',') if dosages_text else []
        frequencies = frequencies_text.split(',') if frequencies_text else []

        for i, med in enumerate(meds):
            medication = {"name": med.strip()}
            if i < len(dosages):
                medication["dosage"] = dosages[i].strip()
            if i < len(frequencies):
                medication["frequency"] = frequencies[i].strip()
            medications.append(medication)

        return medications

    def _parse_procedures(self, procedures_text, dates_text, providers_text):
        """Parse procedure information."""
        procedures = []
        proc_list = procedures_text.split(';') if procedures_text else []
        dates = dates_text.split(';') if dates_text else []
        providers = providers_text.split(';') if providers_text else []

        for i, proc in enumerate(proc_list):
            procedure = {"name": proc.strip()}
            if i < len(dates):
                procedure["date"] = dates[i].strip()
            if i < len(providers):
                procedure["provider"] = providers[i].strip()
            procedures.append(procedure)

        return procedures

    def _parse_vitals(self, vitals_text, values_text, dates_text):
        """Parse vital signs information."""
        vitals = {}
        vitals_list = vitals_text.split(',') if vitals_text else []
        values = values_text.split(',') if values_text else []

        for i, vital in enumerate(vitals_list):
            key = vital.strip()
            if i < len(values):
                vitals[key] = values[i].strip()

        return vitals
```

### Financial Document Analyzer

```python
class FinancialAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract_companies = dspy.Predict(
            "document_text -> companies, stock_symbols, exchanges"
        )

        self.extract_financials = dspy.Predict(
            "document_text -> revenues, profits, expenses, assets"
        )

        self.extract_transactions = dspy.Predict(
            "document_text -> transactions, amounts, dates, parties"
        )

        self.identify_risks = dspy.ChainOfThought(
            "document_text, financial_data -> risk_factors, concerns"
        )

    def forward(self, document_text):
        # Extract company information
        companies = self.extract_companies(document_text=document_text)

        # Extract financial data
        financials = self.extract_financials(document_text=document_text)

        # Extract transactions
        transactions = self.extract_transactions(document_text=document_text)

        # Identify risks
        risks = self.identify_risks(
            document_text=document_text,
            financial_data=str(financials)
        )

        return dspy.Prediction(
            entities={
                "companies": companies.companies.split(',') if companies.companies else [],
                "stock_symbols": companies.stock_symbols.split(',') if companies.stock_symbols else [],
                "exchanges": companies.exchanges.split(',') if companies.exchanges else []
            },
            financials={
                "revenues": self._parse_financial_amounts(financials.revenues),
                "profits": self._parse_financial_amounts(financials.profits),
                "expenses": self._parse_financial_amounts(financials.expenses),
                "assets": self._parse_financial_amounts(financials.assets)
            },
            transactions=self._parse_transactions(
                transactions.transactions,
                transactions.amounts,
                transactions.dates,
                transactions.parties
            ),
            risks={
                "factors": self._parse_list(risks.risk_factors),
                "concerns": self._parse_list(risks.concerns)
            },
            reasoning=risks.rationale
        )

    def _parse_financial_amounts(self, amounts_text):
        """Parse financial amounts."""
        if not amounts_text:
            return []
        return [amount.strip() for amount in amounts_text.split(',')]

    def _parse_transactions(self, trans_text, amounts_text, dates_text, parties_text):
        """Parse transaction data."""
        transactions = []
        trans_list = trans_text.split('|') if trans_text else []
        amounts = amounts_text.split('|') if amounts_text else []
        dates = dates_text.split('|') if dates_text else []
        parties = parties_text.split('|') if parties_text else []

        for i, trans in enumerate(trans_list):
            transaction = {"description": trans.strip()}
            if i < len(amounts):
                transaction["amount"] = amounts[i].strip()
            if i < len(dates):
                transaction["date"] = dates[i].strip()
            if i < len(parties):
                transaction["parties"] = parties[i].strip()
            transactions.append(transaction)

        return transactions

    def _parse_list(self, list_text):
        """Parse semicolon-separated lists."""
        if not list_text:
            return []
        return [item.strip() for item in list_text.split(';') if item.strip()]
```

## Optimizing Entity Extraction

### Using BootstrapFewShot for Entity Extraction

```python
class OptimizedEntityExtractor(dspy.Module):
    def __init__(self, entity_types):
        super().__init__()
        self.entity_types = entity_types
        types_str = ", ".join(entity_types)
        self.extract = dspy.ChainOfThought(
            f"text, entity_types[{types_str}] -> entities_with_confidence"
        )

    def forward(self, text):
        result = self.extract(
            text=text,
            entity_types=", ".join(self.entity_types)
        )

        entities = self._parse_entities_with_confidence(result.entities_with_confidence)

        return dspy.Prediction(
            entities=entities,
            reasoning=result.rationale
        )

    def _parse_entities_with_confidence(self, entities_text):
        """Parse entities with confidence scores."""
        entities = []
        # Format: "ENTITY_TYPE: entity1 (0.9), entity2 (0.8)"
        lines = entities_text.strip().split('\n')
        for line in lines:
            if ':' in line:
                entity_type, entity_list = line.split(':', 1)
                entity_type = entity_type.strip()

                for entity_match in entity_list.split(','):
                    entity_match = entity_match.strip()
                    if '(' in entity_match and ')' in entity_match:
                        entity_text = entity_match[:entity_match.rfind('(')].strip()
                        confidence = float(entity_match[entity_match.rfind('(')+1:-1])
                        entities.append({
                            "text": entity_text,
                            "type": entity_type,
                            "confidence": confidence
                        })

        return entities

# Training data
entity_trainset = [
    dspy.Example(
        text="Apple Inc. announced Q2 earnings of $24.6 billion on April 27, 2023.",
        entities=[
            {"text": "Apple Inc.", "type": "ORGANIZATION", "confidence": 0.95},
            {"text": "$24.6 billion", "type": "MONEY", "confidence": 0.90},
            {"text": "April 27, 2023", "type": "DATE", "confidence": 0.95}
        ]
    ),
    # ... more examples
]

# Evaluation metric
def entity_extraction_metric(example, pred, trace=None):
    """Calculate F1 score for entity extraction."""
    pred_entities = set((e["text"], e["type"]) for e in pred.entities)
    true_entities = set((e["text"], e["type"]) for e in example.entities)

    # Precision and Recall
    tp = len(pred_entities & true_entities)
    fp = len(pred_entities - true_entities)
    fn = len(true_entities - pred_entities)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1 Score
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

# Optimize
optimizer = BootstrapFewShot(
    metric=entity_extraction_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=4
)
optimized_extractor = optimizer.compile(
    OptimizedEntityExtractor(["PERSON", "ORGANIZATION", "LOCATION", "DATE", "MONEY"]),
    trainset=entity_trainset
)
```

## Best Practices

### 1. Handle Entity Ambiguity

```python
def disambiguate_entity(entity, context):
    """Disambiguate entities based on context."""
    # Example: "Apple" could be company or fruit
    if entity.lower() == "apple":
        if any(word in context.lower() for word in ["inc", "corp", "company", "stock", "earnings"]):
            return "Apple Inc."
        elif any(word in context.lower() for word in ["fruit", "food", "eat", "tree"]):
            return "apple (fruit)"
    return entity
```

### 2. Validate Extracted Entities

```python
def validate_entity(entity, entity_type):
    """Validate entity based on type-specific rules."""
    if entity_type == "DATE":
        # Validate date format
        import re
        return bool(re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', entity))
    elif entity_type == "EMAIL":
        # Validate email format
        return "@" in entity and "." in entity.split("@")[-1]
    elif entity_type == "PHONE":
        # Validate phone format
        import re
        return bool(re.match(r'[\d\-\+\(\)\s]+', entity))
    return True
```

### 3. Handle Nested Entities

```python
def resolve_nested_entities(entities):
    """Resolve overlapping or nested entities."""
    # Sort by start position, then by length (longer first)
    sorted_entities = sorted(
        entities,
        key=lambda e: (e.get("start", 0), -len(e["text"]))
    )

    resolved = []
    for entity in sorted_entities:
        # Check for overlap
        overlap = False
        for existing in resolved:
            if (entity.get("start", 0) < existing.get("end", float('inf')) and
                entity.get("end", float('inf')) > existing.get("start", 0)):
                overlap = True
                break

        if not overlap:
            resolved.append(entity)

    return resolved
```

## Evaluation Techniques

### Comprehensive Entity Evaluation

```python
def evaluate_entity_extraction(extractor, testset):
    """Comprehensive evaluation of entity extraction."""
    results = {
        "precision": [],
        "recall": [],
        "f1": [],
        "type_wise": {}
    }

    for example in testset:
        prediction = extractor(text=example.text)

        # Calculate precision, recall, F1
        pred_entities = set((e["text"], e["type"]) for e in prediction.entities)
        true_entities = set((e["text"], e["type"]) for e in example.entities)

        tp = len(pred_entities & true_entities)
        fp = len(pred_entities - true_entities)
        fn = len(true_entities - pred_entities)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results["precision"].append(precision)
        results["recall"].append(recall)
        results["f1"].append(f1)

        # Type-wise evaluation
        for entity_type in set(e["type"] for e in example.entities):
            if entity_type not in results["type_wise"]:
                results["type_wise"][entity_type] = {"tp": 0, "fp": 0, "fn": 0}

            pred_type = set(e for e in pred_entities if e[1] == entity_type)
            true_type = set(e for e in true_entities if e[1] == entity_type)

            results["type_wise"][entity_type]["tp"] += len(pred_type & true_type)
            results["type_wise"][entity_type]["fp"] += len(pred_type - true_type)
            results["type_wise"][entity_type]["fn"] += len(true_type - pred_type)

    # Calculate averages
    for key in ["precision", "recall", "f1"]:
        results[key] = sum(results[key]) / len(results[key])

    # Calculate type-wise metrics
    for entity_type, counts in results["type_wise"].items():
        precision = counts["tp"] / (counts["tp"] + counts["fp"]) if (counts["tp"] + counts["fp"]) > 0 else 0
        recall = counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results["type_wise"][entity_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    return results
```

## Key Takeaways

1. **Entity extraction transforms** unstructured text into structured data
2. **Different applications** require different entity types and extraction strategies
3. **Context is crucial** for accurate entity extraction and disambiguation
4. **Optimization improves** extraction accuracy and consistency
5. **Real-world systems** must handle ambiguity, validation, and edge cases
6. **Comprehensive evaluation** includes precision, recall, and type-wise metrics

## Next Steps

In the next section, we'll explore **Intelligent Agents**, showing how to build autonomous systems that can reason, plan, and execute complex tasks independently.
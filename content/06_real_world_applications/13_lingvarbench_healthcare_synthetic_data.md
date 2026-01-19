# LingVarBench: Synthetic Healthcare Transcript Generation

## Overview

**LingVarBench** is a groundbreaking framework for generating synthetic healthcare phone transcripts that addresses the critical challenge of data privacy in medical NLP. By leveraging DSPy's **SIMBA (Stochastic Introspective Mini-Batch Ascent)** optimizer, LingVarBench creates high-quality, HIPAA-compliant synthetic data that preserves the linguistic patterns and clinical relevance of real healthcare conversations while eliminating privacy risks.

## Key Innovation

The framework introduces a novel approach to synthetic data generation that:
- Maintains clinical accuracy and linguistic diversity
- Preserves patient privacy through complete HIPAA compliance
- Enables robust NER model training without access to real patient data
- Achieves 90%+ accuracy on real healthcare transcripts using only synthetic data

## Architecture

### 1. Data Generation Pipeline

```python
import dspy
from typing import List, Dict, Optional
import random
from dataclasses import dataclass

@dataclass
class MedicalEntity:
    """Represents a medical entity with protected health information."""
    entity_type: str  # MEDICATION, CONDITION, PROCEDURE, etc.
    original_text: str
    deidentified_text: str
    confidence: float

class SyntheticTranscriptGenerator(dspy.Module):
    """Generates synthetic healthcare transcripts using DSPy optimization."""

    def __init__(self, entity_types: List[str]):
        super().__init__()
        self.entity_types = entity_types

        # Initialize SIMBA optimizer for prompt synthesis
        self.simba_optimizer = SIMBAOptimizer(
            population_size=20,
            generations=50,
            mutation_rate=0.1,
            crossover_rate=0.8
        )

        # Core generation module
        self.transcript_generator = dspy.ChainOfThought(
            """Generate a realistic phone conversation between a patient
            and healthcare provider about {topic}.

            Requirements:
            - Include {num_entities} medical entities from: {entity_types}
            - Maintain natural conversation flow
            - Use appropriate medical terminology
            - Create realistic patient symptoms/concerns
            - Ensure provider responses are clinically appropriate
            - Deidentify all PHI while preserving meaning

            Entities to include: {required_entities}

            Generate conversation:
            {conversation}"""
        )

        # Entity deidentification module
        self.deidentifier = dspy.Predict(
            "conversation_with_phi -> conversation_without_phi"
        )

    def generate_transcript(self, topic: str, num_entities: int = 5) -> Dict:
        """Generate a synthetic healthcare transcript."""
        # Select random entities for this transcript
        required_entities = random.sample(self.entity_types,
                                       min(num_entities, len(self.entity_types)))

        # Generate conversation with entities
        conversation = self.transcript_generator(
            topic=topic,
            num_entities=num_entities,
            entity_types=", ".join(self.entity_types),
            required_entities=", ".join(required_entities)
        )

        # Deidentify PHI while preserving entity types
        deidentified = self.deidentifier(
            conversation_with_phi=conversation.conversation
        )

        return {
            "original": conversation.conversation,
            "deidentified": deidentified.conversation_without_phi,
            "entities": self._extract_entities(deidentified.conversation_without_phi),
            "phi_removed": True
        }

    def _extract_entities(self, transcript: str) -> List[MedicalEntity]:
        """Extract and classify medical entities from the transcript."""
        # Implementation depends on your entity extraction approach
        pass
```

### 2. SIMBA Optimization for Prompt Synthesis

```python
class SIMBAOptimizer:
    """Stochastic Introspective Mini-Batch Ascent for prompt optimization."""

    def __init__(self, population_size: int = 20, generations: int = 50,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def optimize_prompt(self, base_prompt: str,
                       evaluation_data: List[Dict]) -> str:
        """Optimize prompt using evolutionary approach."""

        # Initialize population with prompt variations
        population = self._initialize_population(base_prompt)

        for generation in range(self.generations):
            # Evaluate fitness of each prompt
            fitness_scores = []
            for prompt in population:
                score = self._evaluate_prompt(prompt, evaluation_data)
                fitness_scores.append(score)

            # Select best performers
            selected = self._select_parents(population, fitness_scores)

            # Create next generation through crossover and mutation
            population = self._create_generation(selected)

            # Track best performer
            best_idx = fitness_scores.index(max(fitness_scores))
            best_prompt = population[best_idx]
            best_score = max(fitness_scores)

            print(f"Generation {generation}: Best score = {best_score:.3f}")

        return best_prompt

    def _evaluate_prompt(self, prompt: str, data: List[Dict]) -> float:
        """Evaluate prompt quality on synthetic data generation metrics."""

        # Test prompt on evaluation data
        generated_samples = []
        for example in data[:10]:  # Sample for efficiency
            # Use prompt to generate transcript
            generator = dspy.ChainOfThought(prompt)
            result = generator(**example)
            generated_samples.append(result)

        # Calculate metrics
        entity_coverage = self._calculate_entity_coverage(generated_samples, data)
        naturalness_score = self._evaluate_naturalness(generated_samples)
        privacy_preservation = self._check_privacy_compliance(generated_samples)

        # Combine metrics with weights
        total_score = (
            0.4 * entity_coverage +
            0.4 * naturalness_score +
            0.2 * privacy_preservation
        )

        return total_score
```

### 3. Privacy-Preserving Entity Generation

```python
class HIPAACompliantEntityGenerator(dspy.Module):
    """Generates realistic medical entities while preserving privacy."""

    def __init__(self):
        super().__init__()

        # Entity generation templates
        self.medication_template = dspy.Predict(
            """Generate a realistic {medication_type} medication name.
            Requirements:
            - Must sound authentic but be synthetic
            - Follow pharmaceutical naming conventions
            - Not match any real medication

            Medication name:"""
        )

        self.condition_template = dspy.Predict(
            """Generate a realistic medical condition description.
            Requirements:
            - Describe common symptoms
            - Use appropriate medical terminology
            - Be specific enough for NER training
            - Not reveal any identifying information

            Condition description:"""
        )

        # PHI detection and removal
        self.phi_detector = dspy.ChainOfThought(
            """Analyze text for Protected Health Information (PHI).

            PHI types to detect:
            - Names (person, provider, facility)
            - Dates (birth, admission, visit)
            - Locations (address, city, state)
            - Contact information (phone, email)
            - ID numbers (SSN, MRN, insurance)

            Text: {text}

            List all PHI found and suggest replacements:"""
        )

    def generate_safe_entity(self, entity_type: str) -> str:
        """Generate a synthetic medical entity."""

        if entity_type == "MEDICATION":
            result = self.medication_template(medication_type="common")
            return result.medication_name

        elif entity_type == "CONDITION":
            result = self.condition_template()
            return result.condition_description

        # Additional entity types...

    def ensure_privacy_compliance(self, transcript: str) -> str:
        """Remove or replace all PHI from transcript."""

        phi_analysis = self.phi_detector(text=transcript)

        # Apply replacements for detected PHI
        cleaned_transcript = transcript
        for phi_item in phi_analysis.phi_found:
            replacement = self._generate_replacement(phi_item)
            cleaned_transcript = cleaned_transcript.replace(phi_item, replacement)

        return cleaned_transcript
```

## Evaluation Protocol

### 1. Synthetic-to-Real Transfer Learning

```python
class SyntheticToRealEvaluator:
    """Evaluates how well models trained on synthetic data perform on real data."""

    def __init__(self, synthetic_generator, real_dataset):
        self.synthetic_generator = synthetic_generator
        self.real_dataset = real_dataset

    def evaluate_transfer_learning(self, num_synthetic_samples: int = 1000):
        """Test transfer learning performance."""

        # Generate synthetic training data
        synthetic_data = []
        for _ in range(num_synthetic_samples):
            transcript = self.synthetic_generator.generate_transcript(
                topic=random.choice(["medication refill", "symptom inquiry",
                                  "appointment scheduling", "lab results"])
            )
            synthetic_data.append(transcript)

        # Train NER model on synthetic data only
        synthetic_model = self._train_ner_model(synthetic_data)

        # Evaluate on real healthcare transcripts
        real_performance = self._evaluate_on_real_data(
            synthetic_model, self.real_dataset
        )

        # Compare with model trained on real data (if available)
        # real_to_real = self._train_and_evaluate_on_real()

        return {
            "synthetic_to_real_accuracy": real_performance,
            "synthetic_samples_used": len(synthetic_data),
            "entity_f1_scores": self._calculate_entity_f1(synthetic_model),
            "privacy_compliance": self._verify_no_phi_leakage(synthetic_data)
        }

    def _verify_no_phi_leakage(self, synthetic_data: List[Dict]) -> bool:
        """Ensure no real PHI is present in synthetic data."""

        # Check for common PHI patterns
        phi_patterns = [
            r'\b\d{2}/\d{2}/\d{4}\b',  # Dates
            r'\b\d{3}-\d{2}-\d{4}\b',   # SSN
            r'\b\d{10}\b',              # Phone numbers
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'  # Names (simplified)
        ]

        for sample in synthetic_data:
            text = sample["deidentified"]
            for pattern in phi_patterns:
                if re.search(pattern, text):
                    return False

        return True
```

### 2. Quality Metrics for Synthetic Data

```python
class SyntheticDataQualityMetrics:
    """Comprehensive metrics for evaluating synthetic healthcare data."""

    def __init__(self):
        self.naturalness_evaluator = dspy.Predict(
            "transcript -> naturalness_score(1-10)"
        )

        self.medical_accuracy_checker = dspy.ChainOfThought(
            """Verify medical accuracy in healthcare conversation.

            Check:
            - Symptoms match described conditions
            - Medications are appropriate for conditions
            - Provider advice is medically sound
            - Terminology is used correctly

            Conversation: {conversation}

            Medical accuracy assessment:"""
        )

    def evaluate_synthetic_sample(self, sample: Dict) -> Dict:
        """Evaluate quality of a single synthetic transcript."""

        transcript = sample["deidentified"]

        # Naturalness evaluation
        naturalness = self.naturalness_evaluator(transcript=transcript)

        # Medical accuracy check
        accuracy = self.medical_accuracy_checker(conversation=transcript)

        # Entity diversity
        entity_diversity = self._calculate_entity_diversity(sample["entities"])

        # Linguistic features
        linguistic_score = self._analyze_linguistic_features(transcript)

        return {
            "naturalness_score": naturalness.naturalness_score / 10.0,
            "medical_accuracy": self._parse_accuracy_score(accuracy),
            "entity_diversity": entity_diversity,
            "linguistic_appropriateness": linguistic_score,
            "overall_quality": self._calculate_overall_score(
                naturalness.naturalness_score / 10.0,
                self._parse_accuracy_score(accuracy),
                entity_diversity,
                linguistic_score
            )
        }

    def _calculate_entity_diversity(self, entities: List[Dict]) -> float:
        """Calculate diversity of entity types in transcript."""
        if not entities:
            return 0.0

        unique_types = set(e["entity_type"] for e in entities)
        return len(unique_types) / len(entities)

    def _analyze_linguistic_features(self, transcript: str) -> float:
        """Analyze linguistic appropriateness for healthcare context."""

        # Check for appropriate formality level
        # Verify medical terminology usage
        # Assess conversation flow

        # Simplified implementation
        features = {
            "formality": self._check_formality(transcript),
            "terminology": self._check_medical_terminology(transcript),
            "flow": self._check_conversation_flow(transcript)
        }

        return sum(features.values()) / len(features)
```

## Implementation Guide

### 1. Setting Up LingVarBench

```python
# Initialize the framework
entity_types = [
    "MEDICATION", "CONDITION", "PROCEDURE",
    "SYMPTOM", "DEVICE", "MEASUREMENT"
]

generator = SyntheticTranscriptGenerator(entity_types)

# Optimize prompts using SIMBA
training_scenarios = [
    {"topic": "medication refill", "num_entities": 3},
    {"topic": "symptom inquiry", "num_entities": 4},
    {"topic": "appointment scheduling", "num_entities": 2}
]

optimized_generator = simba_optimizer.optimize_prompt(
    base_prompt=generator.transcript_generator.prompt,
    evaluation_data=training_scenarios
)
```

### 2. Training NER Models with Synthetic Data

```python
# Generate synthetic training set
synthetic_train_data = []
for i in range(5000):
    sample = generator.generate_transcript(
        topic=random.choice(list_of_topics),
        num_entities=random.randint(2, 6)
    )

    # Convert to NER training format
    ner_sample = convert_to_ner_format(sample)
    synthetic_train_data.append(ner_sample)

# Train model using only synthetic data
ner_model = train_ner_model(synthetic_train_data)

# Evaluate on real healthcare transcripts (unseen during training)
test_performance = evaluate_model(ner_model, real_test_set)

print(f"Performance on real data: {test_performance['f1']:.3f}")
# Expected: >0.90 F1 score as demonstrated in the paper
```

### 3. Integration with Existing DSPy Pipelines

```python
class HealthcareNERPipeline(dspy.Module):
    """Complete NER pipeline for healthcare transcripts."""

    def __init__(self, synthetic_generator=None):
        super().__init__()

        # Use synthetic data for training if real data unavailable
        if synthetic_generator is None:
            synthetic_generator = SyntheticTranscriptGenerator(entity_types)

        self.ner_extractor = dspy.Predict(
            """Extract medical entities from healthcare transcript.

            Entity types: {entity_types}

            Transcript: {transcript}

            Extracted entities:"""
        )

        self.entity_classifier = dspy.ChainOfThought(
            """Classify extracted medical entities.

            Entity: {entity}
            Context: {context}

            Classification (type and confidence):"""
        )

    def forward(self, transcript: str) -> Dict:
        """Extract and classify medical entities."""

        # Extract entities
        extraction_result = self.ner_extractor(
            transcript=transcript,
            entity_types=", ".join(entity_types)
        )

        # Classify each entity
        classified_entities = []
        for entity in extraction_result.extracted_entities:
            classification = self.entity_classifier(
                entity=entity,
                context=transcript
            )
            classified_entities.append({
                "text": entity,
                "type": classification.entity_type,
                "confidence": classification.confidence,
                "reasoning": classification.reasoning
            })

        return {
            "entities": classified_entities,
            "transcript": transcript,
            "phi_compliant": True
        }
```

## Key Results from Paper

1. **Synthetic Data Quality**: Achieved 92% naturalness score in human evaluations
2. **Privacy Preservation**: 0% PHI leakage in 10,000 generated transcripts
3. **Transfer Learning**: Models trained on synthetic data achieved 91.3% F1 on real data
4. **Data Efficiency**: 50% less training data needed compared to real data training
5. **Cost Reduction**: 80% lower cost than manual data annotation

## Best Practices

1. **Always verify HIPAA compliance** before deployment
2. **Use diverse seed examples** for SIMBA optimization
3. **Regularly evaluate** on real data to prevent synthetic-to-real gap
4. **Combine with real data** when available for best performance
5. **Implement robust PHI detection** as a safety layer

## Limitations and Considerations

- Synthetic data may not capture rare medical conditions
- Requires careful prompt engineering to maintain medical accuracy
- May need domain expert validation for critical applications
- Performance can vary with different medical specialties

## Conclusion

LingVarBench demonstrates the power of combining DSPy's optimization capabilities with privacy-preserving synthetic data generation. The SIMBA optimizer enables automatic creation of high-quality prompts that generate realistic healthcare transcripts without compromising patient privacy. This approach opens new possibilities for medical NLP research while maintaining strict HIPAA compliance.
# Retrieval-Augmented Guardrails for AI Systems

## Overview

Retrieval-Augmented Guardrails provide a sophisticated approach to ensuring AI safety and accuracy, particularly in high-stakes domains like healthcare. This application demonstrates how DSPy can be used to build a comprehensive guardrail system that evaluates AI-generated responses against domain-specific knowledge and similar past interactions.

## Key Concepts

### Error Taxonomy

A clinically grounded error ontology comprising:

1. **Clinical Accuracy** (15 codes)
   - Factual medical errors
   - Dosage/miscalculation errors
   - Outdated information

2. **Completeness** (12 codes)
   - Missing critical information
   - Incomplete follow-up instructions
   - Omitted precautions

3. **Appropriateness** (10 codes)
   - Workflow violations
   - Scope creep
   - Resource allocation errors

4. **Communication Style** (12 codes)
   - Tone mismatches
   - Technical complexity issues
   - Cultural sensitivity

5. **Safety Concerns** (10 codes)
   - Red flag omissions
   - Emergency protocol violations
   - Patient safety compromises

### Retrieval-Augmented Evaluation Pipeline (RAEC)

The RAEC leverages semantically similar historical message-response pairs to improve evaluation quality by providing relevant context for error detection.

## Implementation

### Basic Setup

```python
import dspy
from dspy.datasets import PatientMessages
from dspy.teleprompters import BootstrapFewShot
from dspy.retrieve import FAISSRetriever

# Configure models
lm = dspy.OpenAI(model="gpt-4", api_key="your-api-key")
dspy.settings.configure(lm=lm)

# Initialize retriever for historical messages
retriever = FAISSRetriever(
    collection_name="patient_messages",
    embed_model="text-embedding-3-large"
)
```

### Error Detection Signatures

```python
class ErrorClassifier(dspy.Signature):
    """Classify errors in AI-generated patient messages."""

    message_context = dspy.InputField(desc="Patient's original message")
    ai_response = dspy.InputField(desc="AI-generated response")
    reference_context = dspy.InputField(desc="Similar historical responses")
    error_categories = dspy.OutputField(desc="List of potential error categories")
    confidence_scores = dspy.OutputField(desc="Confidence scores for each error")

class ErrorSeverityAssessor(dspy.Signature):
    """Assess the severity of detected errors."""

    error_description = dspy.InputField(desc="Description of detected error")
    clinical_context = dspy.InputField(desc="Relevant clinical context")
    severity_level = dspy.OutputField(desc="Severity: low/medium/high/critical")
    action_required = dspy.OutputField(desc="Required action to address error")
```

### Retrieval-Augmented Evaluator

```python
class RetrievalAugmentedEvaluator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.error_classifier = dspy.ChainOfThought(ErrorClassifier)
        self.severity_assessor = dspy.Predict(ErrorSeverityAssessor)
        self.retriever = retriever

    def forward(self, patient_message, ai_response):
        # Retrieve similar historical cases
        similar_cases = self.retriever.retrieve(
            query=patient_message,
            k=3
        )

        # Combine contexts
        reference_context = "\n".join([
            f"Similar Case {i+1}:\nPatient: {case.patient}\nResponse: {case.response}"
            for i, case in enumerate(similar_cases)
        ])

        # Classify potential errors
        classification = self.error_classifier(
            message_context=patient_message,
            ai_response=ai_response,
            reference_context=reference_context
        )

        # Assess severity for each detected error
        error_assessments = []
        errors = classification.error_categories.split(", ")
        confidences = [float(c) for c in classification.confidence_scores.split(", ")]

        for error, confidence in zip(errors, confidences):
            severity = self.severity_assessor(
                error_description=error,
                clinical_context=reference_context
            )

            error_assessments.append({
                "error": error,
                "confidence": confidence,
                "severity": severity.severity_level,
                "action": severity.action_required
            })

        return dspy.Prediction(
            errors=error_assessments,
            reference_cases=similar_cases,
            overall_safe=self._calculate_safety_score(error_assessments)
        )

    def _calculate_safety_score(self, error_assessments):
        """Calculate overall safety score based on errors."""
        critical_errors = sum(1 for e in error_assessments if e["severity"] == "critical")
        high_errors = sum(1 for e in error_assessments if e["severity"] == "high")

        if critical_errors > 0:
            return "unsafe"
        elif high_errors > 2:
            return "requires_review"
        elif high_errors > 0:
            return "minor_issues"
        else:
            return "safe"
```

### Two-Stage Hierarchical Detection

```python
class StageOneScreening(dspy.Module):
    """First stage: Quick screening for obvious errors."""

    def __init__(self):
        super().__init__()
        self.screen_classifier = dspy.Predict(
            dspy.Signature(
                """Screen AI response for immediate safety concerns.

                patient_message: Patient's message
                ai_response: AI-generated response
                -> screen_result: safe/needs_review/unsafe
                immediate_concerns: List of immediate concerns if any
            """
            )
        )

    def forward(self, patient_message, ai_response):
        result = self.screen_classifier(
            patient_message=patient_message,
            ai_response=ai_response
        )

        return dspy.Prediction(
            screen_result=result.screen_result,
            immediate_concerns=result.immediate_concerns.split(", ") if result.immediate_concerns else []
        )

class StageTwoDetailedAnalysis(dspy.Module):
    """Second stage: Comprehensive error analysis."""

    def __init__(self):
        super().__init__()
        self.detailed_evaluator = RetrievalAugmentedEvaluator()

    def forward(self, patient_message, ai_response):
        # Only proceed if stage one passed
        stage_one = StageOneScreening()(patient_message, ai_response)

        if stage_one.screen_result == "unsafe":
            return dspy.Prediction(
                final_assessment="unsafe",
                critical_errors=stage_one.immediate_concerns
            )

        # Detailed analysis
        detailed_result = self.detailed_evaluator(
            patient_message=patient_message,
            ai_response=ai_response
        )

        return dspy.Prediction(
            final_assessment=detailed_result.overall_safe,
            detailed_errors=detailed_result.errors,
            reference_cases=detailed_result.reference_cases
        )
```

### Complete Guardrail Pipeline

```python
class PatientMessageGuardrail(dspy.Module):
    """Complete pipeline for evaluating AI-generated patient messages."""

    def __init__(self):
        super().__init__()
        self.stage_one = StageOneScreening()
        self.stage_two = StageTwoDetailedAnalysis()

    def forward(self, patient_message, ai_response):
        # Stage 1: Quick screening
        screening = self.stage_one(patient_message, ai_response)

        # Stage 2: Detailed analysis if needed
        if screening.screen_result != "safe":
            analysis = self.stage_two(patient_message, ai_response)
            return analysis

        return dspy.Prediction(
            final_assessment="safe",
            detailed_errors=[],
            approved=True
        )
```

## Training and Optimization

### Dataset Preparation

```python
class PatientMessageDataset:
    def __init__(self, messages_file):
        self.data = self._load_data(messages_file)

    def _load_data(self, file_path):
        """Load patient messages with error annotations."""
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                data.append({
                    "patient_message": item["message"],
                    "ai_response": item["ai_response"],
                    "error_codes": item["error_codes"],
                    "severity_levels": item["severity_levels"]
                })
        return data

    def create_trainset(self):
        """Create training examples for DSPy."""
        trainset = []
        for item in self.data:
            # Create examples with expected outputs
            example = dspy.Example(
                patient_message=item["patient_message"],
                ai_response=item["ai_response"],
                expected_errors=item["error_codes"],
                expected_severity=item["severity_levels"]
            ).with_inputs("patient_message", "ai_response")
            trainset.append(example)
        return trainset
```

### Optimization with DSPy

```python
def optimize_guardrail_pipeline(trainset):
    """Optimize the guardrail pipeline using DSPy."""

    def evaluation_metric(example, prediction, trace=None):
        """Custom metric for guardrail optimization."""

        # Check if critical errors were detected
        predicted_critical = [
            e for e in prediction.detailed_errors
            if e["severity"] in ["critical", "high"]
        ]

        expected_critical = [
            code for code, severity in zip(
                example.expected_errors, example.expected_severity
            )
            if severity in ["critical", "high"]
        ]

        # Calculate precision and recall for critical errors
        tp = len(set(predicted_critical) & set(expected_critical))
        fp = len(set(predicted_critical) - set(expected_critical))
        fn = len(set(expected_critical) - set(predicted_critical))

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)

        # F1 score with emphasis on recall (catching critical errors)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        return f1

    # Use BootstrapFewShot for optimization
    optimizer = BootstrapFewShot(
        metric=evaluation_metric,
        max_bootstrapped_demos=10,
        max_labeled_demos=5
    )

    optimized_pipeline = optimizer.compile(
        PatientMessageGuardrail(),
        trainset=trainset
    )

    return optimized_pipeline
```

## Performance Evaluation

### Metrics and Benchmarks

```python
def evaluate_guardrail_performance(pipeline, testset):
    """Evaluate guardrail pipeline performance."""

    results = {
        "total": len(testset),
        "correctly_classified": 0,
        "false_negatives": 0,
        "false_positives": 0,
        "concordance": 0,
        "f1_score": 0
    }

    for example in testset:
        prediction = pipeline(
            example.patient_message,
            example.ai_response
        )

        # Track classifications
        if prediction.final_assessment == "safe" and not example.expected_errors:
            results["correctly_classified"] += 1
        elif prediction.final_assessment != "safe" and example.expected_errors:
            results["correctly_classified"] += 1
        elif prediction.final_assessment == "safe" and example.expected_errors:
            results["false_negatives"] += 1
        else:
            results["false_positives"] += 1

    # Calculate metrics
    results["accuracy"] = results["correctly_classified"] / results["total"]
    results["f1_score"] = calculate_f1_score(results)

    return results

def calculate_f1_score(results):
    """Calculate F1 score from evaluation results."""
    precision = results["correctly_classified"] / (
        results["correctly_classified"] + results["false_positives"] + 1e-6
    )
    recall = results["correctly_classified"] / (
        results["correctly_classified"] + results["false_negatives"] + 1e-6
    )
    return 2 * precision * recall / (precision + recall + 1e-6)
```

## Performance Results

### Improvement with Retrieval-Augmentation

Based on the published results:

- **Concordance Improvement**: 50% vs 33% (without retrieval)
- **F1 Score**: 0.500 vs 0.256 (without retrieval)
- **Error Detection**: Significantly improved in:
  - Clinical completeness (+35%)
  - Workflow appropriateness (+28%)
  - Safety concern identification (+42%)

### Human Validation Results

```python
# Example validation results
human_validation = {
    "with_retrieval": {
        "concordance": 0.50,
        "precision": 0.48,
        "recall": 0.52,
        "f1": 0.500
    },
    "without_retrieval": {
        "concordance": 0.33,
        "precision": 0.32,
        "recall": 0.20,
        "f1": 0.256
    }
}
```

## Advanced Features

### Error Pattern Learning

```python
class ErrorPatternLearner(dspy.Module):
    """Learn and adapt to new error patterns."""

    def __init__(self):
        super().__init__()
        self.pattern_detector = dspy.ChainOfThought(
            dspy.Signature(
                """Identify new error patterns from misclassifications.

                misclassified_cases: Cases where errors were missed
                -> new_patterns: List of newly identified error patterns
                suggested_improvements: Improvements to detection rules
            """
            )
        )

    def learn_from_misclassifications(self, misclassifications):
        """Learn from cases where errors were missed."""
        result = self.pattern_detector(
            misclassified_cases=str(misclassifications)
        )

        return {
            "new_patterns": result.new_patterns.split(", "),
            "improvements": result.suggested_improvements.split("; ")
        }
```

### Adaptive Threshold Adjustment

```python
class AdaptiveThresholdManager:
    """Dynamically adjust detection thresholds based on performance."""

    def __init__(self, initial_threshold=0.5):
        self.threshold = initial_threshold
        self.performance_history = []

    def update_threshold(self, recent_performance):
        """Update threshold based on recent performance."""
        self.performance_history.append(recent_performance)

        # Keep only recent history
        if len(self.performance_history) > 10:
            self.performance_history.pop(0)

        # Adjust threshold if precision is too low
        avg_precision = np.mean([p["precision"] for p in self.performance_history])
        if avg_precision < 0.7:
            self.threshold += 0.05
        elif avg_precision > 0.9:
            self.threshold -= 0.02

        return self.threshold
```

## Real-World Deployment

### Integration with EHR Systems

```python
class EHRGuardrailIntegration:
    """Integrate guardrails with Electronic Health Record systems."""

    def __init__(self, guardrail_pipeline, ehr_api):
        self.guardrail = guardrail_pipeline
        self.ehr = ehr_api
        self.audit_log = []

    def screen_message(self, patient_id, message, ai_response):
        """Screen an AI-generated response before sending."""

        # Evaluate with guardrail
        result = self.guardrail(message, ai_response)

        # Log evaluation
        self.audit_log.append({
            "timestamp": datetime.now(),
            "patient_id": patient_id,
            "message": message,
            "ai_response": ai_response,
            "guardrail_result": result.final_assessment,
            "errors": result.detailed_errors
        })

        # Determine action
        if result.final_assessment == "safe":
            return {"action": "send", "message": ai_response}
        elif result.final_assessment == "requires_review":
            return {
                "action": "review",
                "message": ai_response,
                "concerns": result.detailed_errors,
                "reviewer": "clinician"
            }
        else:
            return {
                "action": "block",
                "reason": "Critical safety concerns detected",
                "errors": result.detailed_errors
            }
```

### Continuous Monitoring

```python
class GuardrailMonitor:
    """Monitor guardrail performance and trigger alerts."""

    def __init__(self):
        self.error_rates = {}
        self.alert_thresholds = {
            "critical_miss_rate": 0.05,
            "false_positive_rate": 0.30,
            "response_time_ms": 5000
        }

    def check_performance(self, recent_results):
        """Check if performance is within acceptable ranges."""
        alerts = []

        # Check critical error miss rate
        critical_misses = sum(
            1 for r in recent_results
            if r.has_critical_error and r.assessment == "safe"
        )
        critical_miss_rate = critical_misses / len(recent_results)

        if critical_miss_rate > self.alert_thresholds["critical_miss_rate"]:
            alerts.append({
                "type": "critical_miss_rate_high",
                "value": critical_miss_rate,
                "threshold": self.alert_thresholds["critical_miss_rate"]
            })

        return alerts
```

## Best Practices

### 1. Error Taxonomy Design
- Involve domain experts in taxonomy creation
- Regular updates based on new error patterns
- Balance granularity with usability
- Document clear criteria for each error type

### 2. Retrieval System Optimization
- Use high-quality embedding models
- Maintain up-to-date reference database
- Implement semantic similarity thresholds
- Cache frequent queries

### 3. Evaluation Protocol
- Include diverse error types in test set
- Conduct regular human validation
- Monitor performance drift over time
- Establish clear escalation procedures

### 4. System Integration
- Design for low-latency operation
- Implement proper audit trails
- Ensure compliance with healthcare regulations
- Provide clear feedback to end users

## Conclusion

The Retrieval-Augmented Guardrail system demonstrates how DSPy can be applied to build sophisticated AI safety mechanisms that:

- Significantly improve error detection through contextual understanding
- Provide hierarchical evaluation for efficiency
- Adapt to new error patterns through continuous learning
- Maintain high performance in real-world healthcare scenarios

This approach provides a robust framework for ensuring AI safety and reliability in critical applications, particularly where errors can have significant consequences.

## References

- Original paper: "Retrieval-Augmented Guardrails for AI-Drafted Patient-Portal Messages: Error Taxonomy Construction and Large-Scale Evaluation" (arXiv:2509.22565)
- Clinical error taxonomy documentation
- DSPy retrieval and optimization guides
# Human-Aligned Evaluation: Capturing What Really Matters

## Overview

Traditional evaluation metrics like BERTScore, ROUGE, and BLEU often fail to capture what truly matters to human users, especially in complex, nuanced tasks. Human-aligned evaluation focuses on creating evaluation systems that reflect actual human priorities and domain-specific quality requirements.

This section draws on real-world experiences from building evaluation systems for clinical summarization, demonstrating how to bridge the gap between automated metrics and human judgment.

## The Limitations of Standard Metrics

### Why Off-the-Shelf Metrics Fail

```python
# Example from clinical summarization
standard_metrics = {
    "bert_score": 87.19,  # High semantic similarity
    "rouge_2": 0.82,      # Good n-gram overlap
    # But missed critical clinical details!
}

# Human evaluation revealed:
# - Omitted key diagnoses
# - Missing treatment outcomes
# - Incomplete patient history
```

**Key Problems:**
1. **Context-blind**: Metrics don't understand task-specific requirements
2. **Surface-level**: Focus on lexical overlap, not meaningful content
3. **One-size-fits-all**: Can't adapt to different use cases or priorities
4. **Poor correlation**: Often weak correlation with actual human judgment

### The Correlation Crisis

Studies have shown concerning correlations between standard metrics and human judgment:

```python
# Real-world correlation data from summarization tasks
correlations = {
    "BERTScore": 0.14,      # Almost no correlation
    "ROUGE-2": 0.21,        # Weak correlation
    "BLEU": 0.18,          # Poor correlation
    "Human-aligned LLM": 0.28  # 2x better correlation
}
```

## Building Human-Aligned Evaluation Systems

### 1. Understand Your Quality Dimensions

First, identify what matters for your specific task:

```python
class ClinicalQualityDimensions:
    """Quality dimensions for clinical summarization."""

    FACTUAL_ACCURACY = "Is all information correct?"
    CLINICAL_COMPLETENESS = "Are all critical findings included?"
    CONCISENESS = "Is it appropriately brief?"
    CLINICAL_RELEVANCE = "Is information clinically significant?"
    TEMPORAL_ACCURACY = "Are timelines and sequences correct?"

    @classmethod
    def get_weights(cls):
        """Different weights for different clinical contexts."""
        return {
            "emergency": {
                cls.FACTUAL_ACCURACY: 0.5,
                cls.CLINICAL_COMPLETENESS: 0.3,
                cls.CONCISENESS: 0.1,
                cls.CLINICAL_RELEVANCE: 0.1
            },
            "routine_followup": {
                cls.FACTUAL_ACCURACY: 0.3,
                cls.CLINICAL_COMPLETENESS: 0.2,
                cls.CONCISENESS: 0.3,
                cls.CLINICAL_RELEVANCE: 0.2
            }
        }
```

### 2. Collect Granular Human Feedback

Use structured interfaces to capture nuanced human judgments:

```python
class HumanFeedbackCollector:
    """Collect structured human feedback for evaluation alignment."""

    def __init__(self, quality_dimensions):
        self.dimensions = quality_dimensions
        self.feedback_data = []

    def collect_feedback(self, example, prediction, context):
        """Collect human evaluation with detailed breakdown."""
        feedback = {
            "example_id": example.id,
            "prediction": prediction,
            "context": context,
            "ratings": {},
            "detailed_feedback": {},
            "overall_score": None
        }

        # Rate each dimension
        for dimension in self.dimensions:
            rating = input(f"Rate {dimension} (1-5): ")
            feedback["ratings"][dimension] = int(rating)

            # Collect specific feedback
            detail = input(f"Specific feedback for {dimension}: ")
            feedback["detailed_feedback"][dimension] = detail

        # Overall assessment
        feedback["overall_score"] = int(input("Overall quality (1-5): "))

        self.feedback_data.append(feedback)
        return feedback

    def analyze_patterns(self):
        """Identify common failure patterns from collected feedback."""
        patterns = {}

        for dimension in self.dimensions:
            low_scores = [
                f for f in self.feedback_data
                if f["ratings"][dimension] <= 2
            ]

            if low_scores:
                # Extract common issues from feedback
                issues = [
                    f["detailed_feedback"][dimension]
                    for f in low_scores
                ]
                patterns[dimension] = self._cluster_issues(issues)

        return patterns

    def _cluster_issues(self, issues):
        """Simple clustering of similar issues."""
        # In practice, use NLP clustering techniques
        from collections import Counter

        # Simple keyword-based clustering
        clusters = {}
        for issue in issues:
            keywords = issue.lower().split()[:3]  # First 3 words
            key = " ".join(keywords)
            if key not in clusters:
                clusters[key] = []
            clusters[key].append(issue)

        return clusters
```

### 3. LLM-as-a-Judge with Human-Guided Prompts

Create judges that encode human priorities:

```python
class HumanAlignedLLMJudge(dspy.Module):
    """LLM judge trained on human feedback patterns."""

    def __init__(self, quality_dimensions, weights=None):
        super().__init__()
        self.dimensions = quality_dimensions
        self.weights = weights or {d: 0.25 for d in quality_dimensions}

        # Create evaluation signature
        self.evaluation_signature = dspy.Signature(
            """Evaluate a clinical summary against reference text.

            Quality Dimensions to Assess:
            {dimensions}

            For each dimension:
            1. Rate from 0.0 (poor) to 1.0 (excellent)
            2. Provide specific justification
            3. Note any critical issues

            Reference Summary: {reference}
            Generated Summary: {candidate}
            Context: {context}
            """,
            dspy.InputField(desc="Reference summary"),
            dspy.InputField(desc="Generated summary"),
            dspy.InputField(desc="Additional context"),
            dspy.OutputField(desc="Factual accuracy score"),
            dspy.OutputField(desc="Completeness score"),
            dspy.OutputField(desc="Conciseness score"),
            dspy.OutputField(desc="Overall weighted score"),
            dspy.OutputField(desc="Detailed justification")
        )

        self.judge = dspy.ChainOfThought(self.evaluation_signature)

    def forward(self, reference, candidate, context=None):
        """Evaluate with human-aligned criteria."""
        # Format dimensions for prompt
        dim_text = "\n".join([
            f"- {dim}: {desc}"
            for dim, desc in self.dimensions.items()
        ])

        result = self.judge(
            dimensions=dim_text,
            reference=reference,
            candidate=candidate,
            context=context or "No additional context"
        )

        # Calculate weighted score
        scores = {
            "factual": getattr(result, 'factual_accuracy_score', 0),
            "completeness": getattr(result, 'completeness_score', 0),
            "conciseness": getattr(result, 'conciseness_score', 0)
        }

        weighted_score = sum(
            scores[dim] * self.weights.get(dim, 0.25)
            for dim in scores
        )

        return dspy.Prediction(
            scores=scores,
            weighted_score=weighted_score,
            justification=getattr(result, 'detailed_justification', ''),
            raw_result=result
        )
```

## Case Study: Clinical Summarization

### The Challenge

MultiClinSUM shared task: Multilingual clinical reports summarization where "quality" depends entirely on the use case:
- Clinician's quick review: Needs key findings only
- Patient understanding: Simplified language, no jargon
- Billing system: Specific codes and procedures

### Solution Implementation

```python
class ClinicalSummarizationEvaluator:
    """Complete human-aligned evaluation for clinical summarization."""

    def __init__(self):
        self.human_collector = HumanFeedbackCollector([
            "Factual Accuracy",
            "Clinical Completeness",
            "Conciseness",
            "Clinical Relevance"
        ])

        self.llm_judge = HumanAlignedLLMJudge(
            quality_dimensions={
                "Factual Accuracy": "All medical information is correct",
                "Clinical Completeness": "Critical findings not omitted",
                "Conciseness": "Appropriate length for quick review",
                "Clinical Relevance": "Information is clinically significant"
            },
            weights={
                "Factual Accuracy": 0.5,
                "Clinical Completeness": 0.3,
                "Conciseness": 0.1,
                "Clinical Relevance": 0.1
            }
        )

    def evaluate_system(self, system, test_set):
        """Comprehensive evaluation with multiple metrics."""
        results = {
            "standard_metrics": {},
            "human_aligned": {},
            "correlation_analysis": {}
        }

        # Collect predictions
        predictions = []
        for example in test_set:
            pred = system(example.document)
            predictions.append(pred)

        # Calculate standard metrics
        results["standard_metrics"] = self._calculate_standard_metrics(
            test_set, predictions
        )

        # Human-aligned evaluation
        for example, pred in zip(test_set, predictions):
            # LLM judge evaluation
            judge_result = self.llm_judge(
                reference=example.summary,
                candidate=pred.summary,
                context=example.context
            )

            # Store results
            example.judge_score = judge_result.weighted_score
            example.judge_breakdown = judge_result.scores

        # Calculate human-aligned scores
        results["human_aligned"] = {
            "llm_judge_avg": np.mean([e.judge_score for e in test_set]),
            "dimension_averages": self._calculate_dim_averages(test_set)
        }

        return results

    def validate_alignment(self, human_feedback_data):
        """Check if LLM judge aligns with human judgment."""
        correlations = {}

        for example in human_feedback_data:
            # Compare human overall score with LLM judge
            human_score = example.overall_score / 5.0  # Normalize to [0,1]
            llm_score = example.llm_judge_score

            # Calculate correlation
            correlations.append((human_score, llm_score))

        spearman_rho = self._calculate_spearman(correlations)

        return {
            "spearman_correlation": spearman_rho,
            "alignment_quality": "good" if spearman_rho > 0.5 else "needs_improvement",
            "recommendations": self._generate_alignment_recommendations(spearman_rho)
        }
```

### Results: The Power of Human Alignment

After implementing the human-aligned system:

| Metric | Before Optimization | After Optimization | Improvement |
|--------|-------------------|-------------------|-------------|
| BERTScore | 87.19 | 87.27 | +0.08 |
| **LLM Judge** | **53.90** | **68.07** | **+26.3%** |
| Human Alignment | ρ=0.14 | ρ=0.28 | **2x improvement** |

Key insights:
1. **BERTScore barely changed** - It wasn't measuring what mattered
2. **Human-aligned metric improved 26%** - Optimizing for the right target
3. **Correlation with humans doubled** - Better alignment with actual needs

## Integration with DSPy Optimization

### Using Human-Aligned Metrics for Compilation

```python
# Configure DSPy optimizer with human-aligned metric
def human_aligned_metric(gold, pred, trace=None):
    """Metric that captures clinical quality."""
    judge = HumanAlignedLLMJudge()
    result = judge(
        reference=gold.summary,
        candidate=pred.summary,
        context=getattr(gold, 'context', None)
    )
    return result.weighted_score > 0.7  # Threshold for acceptable quality

# Compile with human guidance
optimizer = dspy.BootstrapFewShot(
    metric=human_aligned_metric,
    max_bootstrapped_demos=5,
    max_labeled_demos=3
)

optimized_summarizer = optimizer.compile(
    ClinicalSummarizer(),
    trainset=training_examples_with_human_feedback
)
```

### Continuous Improvement Loop

```python
class ContinuousImprovementSystem:
    """System for ongoing evaluation and improvement."""

    def __init__(self):
        self.evaluator = ClinicalSummarizationEvaluator()
        self.feedback_collector = HumanFeedbackCollector()
        self.performance_history = []

    def deployment_cycle(self, current_model, new_data):
        """Continuous evaluation and retraining cycle."""
        # 1. Evaluate current performance
        current_results = self.evaluator.evaluate_system(
            current_model, new_data
        )

        # 2. Collect human feedback on edge cases
        edge_cases = self._identify_edge_cases(new_data, current_results)
        for case in edge_cases:
            self.feedback_collector.collect_feedback(
                case.example, case.prediction, case.context
            )

        # 3. Analyze patterns
        patterns = self.feedback_collector.analyze_patterns()

        # 4. Update evaluation criteria if needed
        if self._need_criteria_update(patterns):
            self._update_evaluation_criteria(patterns)

        # 5. Retrain with new insights
        if current_results["human_aligned"]["llm_judge_avg"] < 0.7:
            optimized_model = self._retrain_with_feedback(
                current_model,
                self.feedback_collector.feedback_data
            )
            return optimized_model

        return current_model

    def _identify_edge_cases(self, data, results):
        """Find cases where model performance is poor."""
        edge_cases = []

        for i, example in enumerate(data):
            if example.judge_score < 0.5:  # Poor performance
                edge_cases.append({
                    "example": example,
                    "prediction": example.generated_summary,
                    "context": example.context,
                    "score": example.judge_score
                })

        return edge_cases[:20]  # Top 20 worst cases
```

## Best Practices for Human-Aligned Evaluation

### 1. Start Clear, Stay Consistent

```python
# Good: Clear, actionable quality criteria
EVALUATION_RUBRIC = """
Factual Accuracy (50% weight):
- 1.0: All information verifiably correct
- 0.5: Minor inaccuracies that don't affect clinical meaning
- 0.0: Major errors that could impact care

Clinical Completeness (30% weight):
- 1.0: All critical findings included
- 0.5: Some findings missing but not critical
- 0.0: Critical information omitted
"""

# Bad: Vague, subjective criteria
BAD_RUBRIC = """
Rate the summary quality:
- Good: Looks nice
- Bad: Looks wrong
"""
```

### 2. Separate Training from Evaluation Data

```python
# Prevent leakage between optimization and evaluation
def create_strict_splits(data, train_ratio=0.6, dev_ratio=0.2):
    """Create splits with no overlap in patients or documents."""
    # Group by patient/document to prevent leakage
    patient_groups = {}
    for item in data:
        patient_id = item.get("patient_id", item["doc_id"])
        if patient_id not in patient_groups:
            patient_groups[patient_id] = []
        patient_groups[patient_id].append(item)

    patients = list(patient_groups.keys())
    random.shuffle(patients)

    # Split by patient, not by example
    train_cutoff = int(len(patients) * train_ratio)
    dev_cutoff = int(len(patients) * (train_ratio + dev_ratio))

    train_patients = patients[:train_cutoff]
    dev_patients = patients[train_cutoff:dev_cutoff]
    test_patients = patients[dev_cutoff:]

    # Create datasets
    trainset = []
    for p in train_patients:
        trainset.extend(patient_groups[p])

    # ... similar for dev and test

    return trainset, devset, testset
```

### 3. Version Control Everything

```python
class EvaluationVersionControl:
    """Track all components of evaluation system."""

    def __init__(self):
        self.versions = {}

    def snapshot_evaluation(self, version_name, components):
        """Save complete evaluation configuration."""
        snapshot = {
            "version": version_name,
            "timestamp": datetime.now(),
            "components": {
                "metric_prompt": components["metric_prompt"],
                "quality_dimensions": components["quality_dimensions"],
                "weights": components["weights"],
                "thresholds": components["thresholds"],
                "test_set_hash": self._hash_dataset(components["test_set"])
            }
        }

        self.versions[version_name] = snapshot

        # Save to file for reproducibility
        with open(f"evaluations/{version_name}.json", "w") as f:
            json.dump(snapshot, f, indent=2, default=str)

    def compare_versions(self, v1, v2):
        """Compare two evaluation versions."""
        return {
            "prompt_changes": self._diff_prompts(v1, v2),
            "weight_changes": self._diff_weights(v1, v2),
            "dimension_changes": self._diff_dimensions(v1, v2)
        }
```

## Exercises

1. **Identify Quality Dimensions**: For your task, list 3-5 key quality dimensions that standard metrics miss. Assign weights based on importance.

2. **Create Human Feedback Protocol**: Design a structured form for collecting human feedback on your task's outputs.

3. **Build LLM Judge**: Implement an LLM judge that evaluates outputs based on your quality dimensions.

4. **Validate Alignment**: Collect human judgments on 20 examples and calculate correlation with your LLM judge.

5. **Iterate and Improve**: Based on misalignments, refine your judge prompt and re-evaluate.

## Key Takeaways

1. **Standard metrics often fail** to capture what matters for complex tasks
2. **Human alignment is crucial** for building evaluation systems that reflect real needs
3. **LLM-as-a-judge bridges the gap** between automated metrics and human judgment
4. **Continuous feedback** drives ongoing improvement
5. **Context matters** - quality definitions must adapt to specific use cases

Remember: Good evaluation systems evolve with your understanding of the task and its real-world impact. Start simple, collect feedback, and iteratively refine what "quality" means for your specific context.

---

**References:**
- Explosion AI. (2025). Engineering a human-aligned LLM evaluation workflow with Prodigy and DSPy.
- Statsig. (2025). DSPy vs prompt engineering: Systematic vs manual tuning.
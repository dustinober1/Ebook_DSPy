# LLM-as-a-Judge for Context-Sensitive Evaluation

## Overview

**LLM-as-a-Judge** is a powerful evaluation paradigm that uses large language models to assess the quality and impact of model outputs. This approach is particularly valuable when traditional metrics fail to capture domain-specific nuances or real-world consequences.

This framework becomes essential in safety-critical domains like healthcare, where standard metrics such as Word Error Rate (WER) for Automatic Speech Recognition (ASR) correlate poorly with actual clinical risk. The approach demonstrates how LLMs can be trained to perform nuanced, context-aware evaluations that align with expert human judgment.

## When to Use LLM-as-a-Judge

### 1. Domain-Specific Impact Assessment

```python
# Standard metrics (WER, BLEU) fail to capture clinical meaning
standard_metrics = {
    "wer": 0.12,  # Low error rate
    "bleu": 0.85,  # High overlap
    # But missed critical negation: "no chest pain" → "chest pain"
}

# LLM-as-a-Judge captures actual impact
clinical_judge = ClinicalImpactJudge()
assessment = clinical_judge.evaluate(
    ground_truth="Patient reports no chest pain or shortness of breath",
    hypothesis="Patient reports chest pain or shortness of breath"
)
# Result: SIGNIFICANT_CLINICAL_IMPACT (2/2)
```

### 2. Nuanced Semantic Evaluation

Traditional metrics struggle with:
- Context-dependent meaning
- Domain-specific terminology
- Weighted importance of different errors
- Complex relationships between concepts

### 3. Multi-Dimensional Quality Assessment

```python
class MultiDimensionalJudge(dspy.Module):
    """Evaluates outputs across multiple quality dimensions."""

    def __init__(self, dimensions: List[str]):
        super().__init__()
        self.dimensions = dimensions
        self.judge = dspy.ChainOfThought(
            """Evaluate the {output} against {ground_truth}.

            Consider these dimensions:
            {dimensions}

            For each dimension, provide:
            - Score (1-5)
            - Justification
            - Impact severity"""
        )

    def forward(self, output: str, ground_truth: str):
        evaluation = self.judge(
            output=output,
            ground_truth=ground_truth,
            dimensions=", ".join(self.dimensions)
        )
        return evaluation
```

## Implementation Framework

### 1. Core Judge Architecture

```python
import dspy
from typing import Dict, List, Tuple, Optional

class LLMJudge(dspy.Module):
    """Base class for LLM-as-a-Judge implementations."""

    def __init__(self,
                 prompt_template: str,
                 output_schema: type,
                 max_tokens: int = 1000):
        super().__init__()
        self.prompt_template = prompt_template
        self.output_schema = output_schema
        self.max_tokens = max_tokens

        # Initialize the judge with Chain of Thought for reasoning
        self.judge = dspy.ChainOfThought(
            self.prompt_template,
            max_tokens=self.max_tokens
        )

    def evaluate(self, ground_truth: str, hypothesis: str, **context) -> Dict:
        """Evaluate hypothesis against ground truth."""
        # Format the prompt with inputs
        prompt = self.prompt_template.format(
            ground_truth=ground_truth,
            hypothesis=hypothesis,
            **context
        )

        # Get LLM evaluation
        result = self.judge(
            ground_truth=ground_truth,
            hypothesis=hypothesis,
            **context
        )

        # Parse and validate output
        try:
            return self.parse_output(result)
        except Exception as e:
            return {
                "error": str(e),
                "raw_output": str(result),
                "evaluation": "PARSING_ERROR"
            }

    def parse_output(self, raw_output) -> Dict:
        """Parse LLM output into structured format."""
        # Implementation depends on output_schema
        # This is a generic implementation
        if hasattr(raw_output, 'reasoning'):
            return {
                "reasoning": raw_output.reasoning,
                "evaluation": getattr(raw_output, 'evaluation', None),
                "confidence": getattr(raw_output, 'confidence', 0.0)
            }
        return {"raw_output": str(raw_output)}
```

### 2. Clinical Impact Judge

```python
class ClinicalImpactJudge(LLMJudge):
    """Judge for assessing clinical impact of ASR errors."""

    # Define impact levels
    IMPACT_LEVELS = {
        0: "No Clinical Impact",
        1: "Minimal Clinical Impact",
        2: "Significant Clinical Impact"
    }

    def __init__(self):
        prompt_template = """
        You are an expert medical analyst. Your task is to assess the clinical impact
        of errors in an AI-generated transcription of a medical conversation.

        You will be given:
        1. ground_truth_conversation: The accurate, human-verified transcript
        2. transcription_conversation: The machine-generated transcript with errors

        Core Principle: Determine if a clinician reading the transcription would
        make different medical decisions than if they read the ground truth.

        Provide:
        1. reasoning: Step-by-step analysis of differences
        2. clinical_impact: Single integer (0, 1, or 2)

        Impact Levels:
        - 0: No Clinical Impact (cosmetic errors only)
        - 1: Minimal Clinical Impact (non-critical ambiguities)
        - 2: Significant Clinical Impact (could affect diagnosis/treatment)

        Ground Truth: {ground_truth}
        Transcription: {hypothesis}
        Context: {context}
        """

        super().__init__(
            prompt_template=prompt_template,
            output_schema=dict
        )

    def evaluate(self, ground_truth: str, hypothesis: str,
                  context: Optional[str] = None) -> Dict:
        """Evaluate clinical impact with structured output."""
        result = super().evaluate(
            ground_truth=ground_truth,
            hypothesis=hypothesis,
            context=context or "No additional context"
        )

        # Normalize the impact level
        if 'clinical_impact' in result:
            try:
                impact = int(result['clinical_impact'])
                result['clinical_impact'] = min(max(impact, 0), 2)
                result['impact_label'] = self.IMPACT_LEVELS[result['clinical_impact']]
            except:
                result['clinical_impact'] = -1
                result['impact_label'] = "UNKNOWN"

        return result
```

### 3. Specialized Judges for Different Domains

```python
class CodeQualityJudge(LLMJudge):
    """Judge for evaluating code quality and correctness."""

    def __init__(self):
        prompt_template = """
        Evaluate the generated code against the reference implementation.

        Consider:
        - Correctness: Does it produce the right output?
        - Efficiency: Is it optimal in time/space complexity?
        - Readability: Is it clean and maintainable?
        - Edge Cases: Does it handle unusual inputs?

        Provide scores (1-5) for each dimension and overall assessment.

        Reference: {ground_truth}
        Generated: {hypothesis}
        """

        super().__init__(prompt_template, dict)

class CreativeWritingJudge(LLMJudge):
    """Judge for evaluating creative writing quality."""

    def __init__(self):
        prompt_template = """
        Evaluate the creative writing piece against criteria:

        - Creativity and originality
        - Engagement and flow
        - Character development (if applicable)
        - Plot coherence
        - Language quality

        Reference Piece: {ground_truth}
        Generated Piece: {hypothesis}
        Writing Style: {style}
        """

        super().__init__(prompt_template, dict)

class FactualAccuracyJudge(LLMJudge):
    """Judge for checking factual accuracy in generated text."""

    def __init__(self):
        prompt_template = """
        Fact-check the generated text against verified information.

        For each factual claim:
        - Is it accurate?
        - Is it properly attributed?
        - Is any important context missing?

        Flag any hallucinations or misstatements.

        Verified Information: {ground_truth}
        Generated Text: {hypothesis}
        """

        super().__init__(prompt_template, dict)
```

## Training and Optimization

### 1. Using GEPA for Prompt Optimization

> **Note**: **GEPA** stands for *Generative Evolutionary Prompt Adjustment*. It is an advanced optimizer technique covered in detail in [Chapter 5: Optimizers](../05-optimizers/00-chapter-intro.md).

```python
from gepa import GEPAOptimizer

class OptimizedJudge:
    """Train LLM judge using GEPA for prompt optimization."""

    def __init__(self, base_judge_class, training_data: List[Dict]):
        self.base_judge_class = base_judge_class
        self.training_data = training_data
        self.optimized_prompt = None
        self.trained_judge = None

    def optimize_prompt(self, num_iterations: int = 10):
        """Optimize the judge's prompt using GEPA."""

        # Initialize GEPA optimizer
        optimizer = GEPAOptimizer(
            population_size=10,
            generations=num_iterations,
            objectives=["accuracy", "robustness"],
            reflection_model="gpt-4"
        )

        # Define initial prompt
        initial_prompt = self.base_judge_class.__init__.__doc__

        # Create evaluation function
        def evaluate_prompt(prompt: str) -> Dict:
            # Create temporary judge with new prompt
            temp_judge = self.base_judge_class()
            temp_judge.prompt_template = prompt

            # Evaluate on training data
            correct = 0
            total = len(self.training_data)

            for example in self.training_data:
                result = temp_judge.evaluate(**example)
                if result.get('evaluation') == example.get('expected'):
                    correct += 1

            return {
                "accuracy": correct / total,
                "robustness": self._calculate_robustness(temp_judge)
            }

        # Run optimization
        best_prompt = optimizer.compile(
            program=initial_prompt,
            trainset=self.training_data,
            evalset=self.training_data
        )

        self.optimized_prompt = best_prompt
        self.trained_judge = self.base_judge_class()
        self.trained_judge.prompt_template = self.optimized_prompt

    def _calculate_robustness(self, judge) -> float:
        """Calculate robustness across diverse examples."""
        # Test on edge cases and variations
        edge_cases = self._generate_edge_cases()
        consistent_results = 0

        for case in edge_cases:
            result1 = judge.evaluate(**case)
            # Slight variation
            case_variant = self._add_noise(case)
            result2 = judge.evaluate(**case_variant)

            if self._results_consistent(result1, result2):
                consistent_results += 1

        return consistent_results / len(edge_cases)
```

### 2. Cost-Sensitive Training

```python
class CostSensitiveTraining:
    """Train judge with cost-sensitive loss function."""

    def __init__(self, cost_matrix: Dict[Tuple[int, int], float]):
        self.cost_matrix = cost_matrix
        # Example: cost_matrix[(true, pred)] = penalty

    def calculate_loss(self, predictions: List[int],
                       labels: List[int]) -> float:
        """Calculate weighted loss based on cost matrix."""
        total_cost = 0.0

        for pred, true in zip(predictions, labels):
            cost = self.cost_matrix.get((true, pred), 0.0)
            total_cost += cost

        return total_cost / len(predictions)

# Example cost matrix for clinical impact
clinical_cost_matrix = {
    (0, 0): 1.2,   # Correctly identify no impact
    (0, 1): 0.3,   # Over-predict minimal impact
    (0, 2): -1.0,  # Over-predict significant impact
    (1, 0): 0.3,   # Under-predict minimal impact
    (1, 1): 1.5,   # Correctly identify minimal impact
    (1, 2): 0.5,   # Over-predict significant impact
    (2, 0): -1.2,  # Miss significant impact (worst)
    (2, 1): 0.4,   # Under-predict significance
    (2, 2): 1.5    # Correctly identify significant impact
}
```

## Best Practices

### 1. Prompt Design

```python
# Good: Clear, structured prompts with explicit criteria
GOOD_PROMPT_TEMPLATE = """
You are evaluating [task_type] outputs.

Evaluation Criteria:
1. [Criterion 1]: [Clear definition]
2. [Criterion 2]: [Clear definition]
3. [Criterion 3]: [Clear definition]

For each criterion:
- Provide a score (1-5)
- Give specific justification
- Note any concerns

Output Format:
{
  "scores": {{
    "criterion_1": score,
    "criterion_2": score,
    "criterion_3": score
  }},
  "justifications": {{
    "criterion_1": "explanation",
    "criterion_2": "explanation",
    "criterion_3": "explanation"
  }},
  "overall_assessment": "summary",
  "confidence": 0.0-1.0
}
"""

# Bad: Vague, unstructured evaluation
BAD_PROMPT_TEMPLATE = """
Is this output good?
Output: {hypothesis}
Reference: {ground_truth}
"""
```

### 2. Handling Bias

```python
class UnbiasedJudge:
    """Judge with bias mitigation strategies."""

    def __init__(self, base_judge, bias_detectors: List[callable]):
        self.base_judge = base_judge
        self.bias_detectors = bias_detectors

    def evaluate(self, *args, **kwargs):
        # Get initial evaluation
        result = self.base_judge.evaluate(*args, **kwargs)

        # Check for various biases
        for detector in self.bias_detectors:
            bias_score = detector(result, *args, **kwargs)
            if bias_score > 0.7:  # High bias detected
                result["bias_warning"] = f"High {detector.__name__} detected"
                result["bias_score"] = bias_score

        return result

def length_bias_detector(result, hypothesis, **kwargs):
    """Detect bias towards longer/shorter outputs."""
    if len(hypothesis) > 500:
        return 0.8  # Likely favoring longer outputs
    return 0.1

def positivity_bias_detector(result, hypothesis, **kwargs):
    """Detect bias towards overly positive evaluations."""
    positive_words = ["excellent", "perfect", "outstanding"]
    count = sum(1 for word in positive_words if word in str(result).lower())
    if count > 2:
        return 0.7
    return 0.1
```

### 3. Ensemble of Judges

```python
class JudgeEnsemble:
    """Combine multiple judges for more robust evaluation."""

    def __init__(self, judges: List[LLMJudge], weights: Optional[List[float]] = None):
        self.judges = judges
        self.weights = weights or [1.0] * len(judges)

    def evaluate(self, *args, **kwargs):
        """Get evaluations from all judges and combine."""
        evaluations = []

        for judge in self.judges:
            eval_result = judge.evaluate(*args, **kwargs)
            evaluations.append(eval_result)

        # Combine results
        combined = self._combine_evaluations(evaluations)

        # Calculate confidence based on agreement
        combined["agreement_score"] = self._calculate_agreement(evaluations)
        combined["individual_evaluations"] = evaluations

        return combined

    def _combine_evaluations(self, evaluations: List[Dict]) -> Dict:
        """Combine multiple evaluation results."""
        # Simple averaging for numeric scores
        combined = {}

        if all('evaluation' in e for e in evaluations):
            # For classification tasks
            scores = [e['evaluation'] for e in evaluations]
            combined['evaluation'] = round(sum(scores) / len(scores))
            combined['vote_distribution'] = {
                score: scores.count(score) for score in set(scores)
            }

        return combined

    def _calculate_agreement(self, evaluations: List[Dict]) -> float:
        """Calculate how much judges agree with each other."""
        if len(evaluations) < 2:
            return 1.0

        agreements = 0
        total_comparisons = 0

        for i in range(len(evaluations)):
            for j in range(i + 1, len(evaluations)):
                if evaluations[i].get('evaluation') == evaluations[j].get('evaluation'):
                    agreements += 1
                total_comparisons += 1

        return agreements / total_comparisons if total_comparisons > 0 else 0.0
```

## Integration with DSPy Evaluation

### 1. Custom Metrics

```python
class LLMJudgeMetric(dspy.Metric):
    """DSPy metric that uses LLM judge for evaluation."""

    def __init__(self, judge: LLMJudge):
        self.judge = judge

    def __call__(self, example, prediction, trace=None):
        """Evaluate prediction using LLM judge."""
        # Extract relevant fields from example and prediction
        ground_truth = example.outputs()
        hypothesis = prediction.get('output', str(prediction))

        # Add context if available
        context = example.get('context', None)

        # Get judge evaluation
        result = self.judge.evaluate(
            ground_truth=ground_truth,
            hypothesis=hypothesis,
            context=context
        )

        # Convert to numeric score
        if 'evaluation' in result:
            return result['evaluation'] / 2.0  # Normalize to [0, 1]

        # Fallback to confidence score
        return result.get('confidence', 0.0)

# Usage in DSPy evaluation
clinical_metric = LLMJudgeMetric(ClinicalImpactJudge())

evaluate = dspy.Evaluate(
    devset=test_set,
    metric=clinical_metric,
    num_threads=1  # LLM judges may be expensive
)
```

### 2. Progressive Evaluation

```python
class ProgressiveEvaluator:
    """Multi-stage evaluation using different judges."""

    def __init__(self):
        self.stages = [
            ("quick_filter", QuickFilterJudge()),  # Fast, cheap
            ("detailed", DetailedJudge()),        # Slower, thorough
            ("expert", ExpertJudge())              # Slowest, most accurate
        ]

    def evaluate(self, examples, predictions):
        """Progressively evaluate with increasing detail."""
        results = {}

        for stage_name, judge in self.stages:
            stage_results = []

            for example, pred in zip(examples, predictions):
                # Skip if already filtered out
                if stage_name != "quick_filter" and \
                   results.get("quick_filter", {}).get(pred.id, True) == False:
                    stage_results.append(False)
                    continue

                result = judge.evaluate(
                    ground_truth=example.outputs(),
                    hypothesis=pred.get('output', str(pred))
                )

                passed = result.get('evaluation', True)
                stage_results.append(passed)

            results[stage_name] = dict(zip([p.id for p in predictions],
                                           stage_results))

        return results
```

## Exercises

1. **Implement Domain-Specific Judge**: Create an LLM judge for evaluating responses in your specific domain (e.g., legal documents, scientific papers, customer service).

2. **Compare with Traditional Metrics**: Evaluate a dataset using both traditional metrics (WER, BLEU) and an LLM judge. Compare the correlation with human judgments.

3. **Optimize with GEPA**: Take a basic judge prompt and optimize it using GEPA on a small labeled dataset.

4. **Create Ensemble**: Build an ensemble of judges with different specializations and evaluate their combined performance.

5. **Bias Analysis**: Implement bias detection for your judge and analyze potential biases in evaluations.
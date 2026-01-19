# Defining Metrics

## Prerequisites

- **Chapter 1-3**: DSPy Fundamentals, Signatures, and Modules
- **Previous Sections**: Why Evaluation Matters, Creating Datasets
- **Required Knowledge**: Basic Python functions
- **Difficulty Level**: Intermediate-Advanced
- **Estimated Reading Time**: 35 minutes

## Learning Objectives

By the end of this section, you will be able to:
- Understand the anatomy of DSPy metric functions
- Use built-in metrics for common tasks
- Create custom metrics for specific needs
- Design composite metrics that capture multiple quality dimensions
- Use the trace parameter for optimization-aware metrics

## Metric Function Anatomy

A DSPy metric is a Python function that evaluates prediction quality:

```python
def metric(example, pred, trace=None):
    """
    Evaluate prediction quality.

    Args:
        example: The original Example with inputs AND expected outputs
        pred: The Prediction (module output) to evaluate
        trace: Optional trace info (used during optimization)

    Returns:
        bool or float: Score indicating quality (True/False or 0.0-1.0)
    """
    # Compare prediction to expected output
    return pred.answer == example.answer
```

### The Three Parameters

#### 1. `example` - The Ground Truth

Contains both inputs and expected outputs:

```python
example = dspy.Example(
    question="What is 2+2?",  # Input
    answer="4"                 # Expected output (ground truth)
).with_inputs("question")

# In metric:
def metric(example, pred, trace=None):
    ground_truth = example.answer  # Access expected output
    input_question = example.question  # Can also access input
```

#### 2. `pred` - The Model's Prediction

The output from your DSPy module:

```python
# Module produces prediction
module = dspy.Predict("question -> answer")
pred = module(question="What is 2+2?")

# In metric:
def metric(example, pred, trace=None):
    model_output = pred.answer  # Access predicted output
```

#### 3. `trace` - Optimization Context

Indicates whether metric is being used for optimization:

```python
def metric(example, pred, trace=None):
    # Calculate score
    score = calculate_similarity(example.answer, pred.answer)

    if trace is not None:
        # During optimization: return boolean for filtering
        return score >= 0.9  # Only accept very good examples

    # During evaluation: return actual score
    return score
```

## Built-in Metrics

DSPy provides several ready-to-use metrics:

### SemanticF1

Measures semantic overlap between answers:

```python
from dspy.evaluate import SemanticF1

# Initialize metric
metric = SemanticF1(decompositional=True)

# Use in evaluation
example = dspy.Example(
    question="What is photosynthesis?",
    response="The process by which plants convert sunlight to energy"
).with_inputs("question")

pred = module(question=example.question)

# Returns F1 score based on semantic similarity
score = metric(example, pred)
print(f"Semantic F1: {score}")
```

### Exact Match

Simple string equality:

```python
def exact_match(example, pred, trace=None):
    """Exact string match metric."""
    return example.answer.strip().lower() == pred.answer.strip().lower()
```

### Answer Correctness

For QA tasks with known correct answers:

```python
def answer_correctness(example, pred, trace=None):
    """Check if predicted answer contains the correct answer."""
    correct = example.answer.lower()
    predicted = pred.answer.lower()
    return correct in predicted or predicted in correct
```

## Creating Custom Metrics

### Simple Boolean Metrics

Return True/False for pass/fail:

```python
def sentiment_accuracy(example, pred, trace=None):
    """Check if sentiment prediction matches ground truth."""
    return example.sentiment == pred.sentiment

def label_match(example, pred, trace=None):
    """Check if classification label matches."""
    expected = example.label.lower().strip()
    predicted = pred.label.lower().strip()
    return expected == predicted
```

### Numeric Metrics

Return scores between 0 and 1:

```python
def partial_match(example, pred, trace=None):
    """Score based on word overlap."""
    expected_words = set(example.answer.lower().split())
    predicted_words = set(pred.answer.lower().split())

    if not expected_words:
        return 0.0

    overlap = expected_words.intersection(predicted_words)
    return len(overlap) / len(expected_words)

def length_ratio(example, pred, trace=None):
    """Score based on answer length appropriateness."""
    expected_len = len(example.answer)
    predicted_len = len(pred.answer)

    if expected_len == 0:
        return 0.0

    ratio = min(predicted_len, expected_len) / max(predicted_len, expected_len)
    return ratio
```

### Domain-Specific Metrics

Metrics tailored to your application:

```python
# Medical diagnosis accuracy
def diagnosis_metric(example, pred, trace=None):
    """Evaluate medical diagnosis predictions."""
    # Primary diagnosis must match
    primary_correct = example.primary_diagnosis == pred.primary_diagnosis

    # Check if any differential diagnosis is correct
    differential_overlap = any(
        d in example.differential_diagnoses
        for d in pred.differential_diagnoses
    )

    # Urgency assessment
    urgency_correct = example.urgency_level == pred.urgency_level

    # Weighted combination
    score = (
        0.5 * primary_correct +
        0.3 * differential_overlap +
        0.2 * urgency_correct
    )

    return score

# Code generation correctness
def code_correctness(example, pred, trace=None):
    """Evaluate generated code."""
    try:
        # Try to execute the generated code
        exec(pred.code)

        # Check if output matches expected
        # (In practice, you'd capture and compare output)
        return True
    except Exception:
        return False

# Entity extraction F1
def entity_f1(example, pred, trace=None):
    """Calculate F1 score for entity extraction."""
    expected = set(example.entities)
    predicted = set(pred.entities)

    if not expected and not predicted:
        return 1.0
    if not expected or not predicted:
        return 0.0

    true_positives = len(expected & predicted)
    precision = true_positives / len(predicted) if predicted else 0
    recall = true_positives / len(expected) if expected else 0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
```

## Composite Metrics

Combine multiple quality dimensions:

### Weighted Combination

```python
def comprehensive_qa_metric(example, pred, trace=None):
    """
    Comprehensive QA evaluation combining multiple factors.
    """
    # 1. Answer correctness (most important)
    correct = example.answer.lower() in pred.answer.lower()
    correctness_score = 1.0 if correct else 0.0

    # 2. Answer completeness
    expected_len = len(example.answer)
    predicted_len = len(pred.answer)
    completeness = min(1.0, predicted_len / max(expected_len, 1))

    # 3. Relevance (answer mentions key terms from question)
    question_words = set(example.question.lower().split())
    answer_words = set(pred.answer.lower().split())
    relevance = len(question_words & answer_words) / max(len(question_words), 1)

    # 4. Confidence (if available)
    confidence_score = getattr(pred, 'confidence', 0.5)

    # Weighted combination
    final_score = (
        0.5 * correctness_score +
        0.2 * completeness +
        0.2 * relevance +
        0.1 * confidence_score
    )

    # For optimization, require high threshold
    if trace is not None:
        return final_score >= 0.8

    return final_score
```

### Checklist-Based Metrics

```python
def quality_checklist(example, pred, trace=None):
    """
    Evaluate against a quality checklist.
    """
    checks = {
        "has_answer": len(pred.answer.strip()) > 0,
        "not_too_short": len(pred.answer) >= 10,
        "not_too_long": len(pred.answer) <= 500,
        "contains_expected": example.expected_keyword in pred.answer.lower(),
        "no_apology": "sorry" not in pred.answer.lower(),
        "no_uncertainty": "i don't know" not in pred.answer.lower(),
    }

    # Count passed checks
    passed = sum(checks.values())
    total = len(checks)

    if trace is not None:
        # For optimization, all checks must pass
        return passed == total

    # For evaluation, return ratio of passed checks
    return passed / total
```

### Multi-Aspect Metrics

```python
def multi_aspect_metric(example, pred, trace=None):
    """
    Return detailed scores for multiple aspects.
    During evaluation, returns overall score.
    Can also be used for detailed analysis.
    """
    scores = {
        "accuracy": calculate_accuracy(example, pred),
        "fluency": calculate_fluency(pred.answer),
        "relevance": calculate_relevance(example.question, pred.answer),
        "safety": calculate_safety(pred.answer),
    }

    # Overall score (weighted average)
    weights = {"accuracy": 0.4, "fluency": 0.2, "relevance": 0.3, "safety": 0.1}
    overall = sum(scores[k] * weights[k] for k in scores)

    if trace is not None:
        return overall >= 0.7

    return overall


# Helper functions
def calculate_accuracy(example, pred):
    return 1.0 if example.answer.lower() in pred.answer.lower() else 0.0

def calculate_fluency(text):
    # Simple fluency check (could use language model)
    words = text.split()
    if len(words) < 3:
        return 0.5
    return 1.0

def calculate_relevance(question, answer):
    q_words = set(question.lower().split())
    a_words = set(answer.lower().split())
    overlap = len(q_words & a_words)
    return min(1.0, overlap / max(len(q_words), 1))

def calculate_safety(text):
    unsafe_terms = ["harmful", "dangerous", "illegal"]
    return 0.0 if any(term in text.lower() for term in unsafe_terms) else 1.0
```

## The Trace Parameter Deep Dive

The `trace` parameter enables different behavior during optimization vs. evaluation:

### Why Trace Matters

```python
# During optimization (trace is not None)
# - DSPy is looking for good examples to bootstrap
# - Metric should return boolean (True = good example)
# - Be stricter to get high-quality demonstrations

# During evaluation (trace is None)
# - You want to measure actual performance
# - Metric should return actual score (float or bool)
# - Be accurate, not strict
```

### Trace-Aware Metric Pattern

```python
def smart_metric(example, pred, trace=None):
    """
    Metric that behaves differently during optimization vs evaluation.
    """
    # Calculate detailed score
    exact = example.answer.lower() == pred.answer.lower()
    partial = example.answer.lower() in pred.answer.lower()
    length_ok = 0.5 <= len(pred.answer) / len(example.answer) <= 2.0

    if trace is not None:
        # OPTIMIZATION MODE
        # Be strict - only accept perfect examples
        # These will be used as demonstrations
        return exact and length_ok

    # EVALUATION MODE
    # Return nuanced score
    if exact:
        return 1.0
    elif partial and length_ok:
        return 0.7
    elif partial:
        return 0.5
    else:
        return 0.0
```

### Using Trace for Debugging

```python
def debugging_metric(example, pred, trace=None):
    """
    Metric that logs information when tracing.
    """
    score = example.answer.lower() in pred.answer.lower()

    if trace is not None:
        # Log during optimization for debugging
        print(f"Expected: {example.answer}")
        print(f"Got: {pred.answer}")
        print(f"Score: {score}")
        print("---")

    return score
```

## Common Metric Patterns

### Pattern 1: Exact Match with Normalization

```python
def normalized_exact_match(example, pred, trace=None):
    """Exact match after normalization."""
    def normalize(text):
        return text.lower().strip().replace(".", "").replace(",", "")

    return normalize(example.answer) == normalize(pred.answer)
```

### Pattern 2: Contains Expected

```python
def contains_expected(example, pred, trace=None):
    """Check if prediction contains the expected answer."""
    expected = example.answer.lower()
    predicted = pred.answer.lower()
    return expected in predicted
```

### Pattern 3: Any of Multiple Correct Answers

```python
def any_correct(example, pred, trace=None):
    """Accept any of multiple correct answers."""
    # example.answers is a list of acceptable answers
    predicted = pred.answer.lower().strip()
    return any(
        ans.lower().strip() in predicted
        for ans in example.answers
    )
```

### Pattern 4: Threshold-Based

```python
def threshold_metric(example, pred, trace=None, threshold=0.8):
    """Apply threshold to continuous score."""
    # Calculate similarity score
    score = calculate_similarity(example.answer, pred.answer)

    if trace is not None:
        return score >= threshold

    return score
```

### Pattern 5: Multi-Field Match

```python
def multi_field_metric(example, pred, trace=None):
    """Evaluate multiple output fields."""
    scores = []

    # Check each output field
    if hasattr(example, 'sentiment'):
        scores.append(example.sentiment == pred.sentiment)

    if hasattr(example, 'category'):
        scores.append(example.category == pred.category)

    if hasattr(example, 'confidence'):
        scores.append(abs(example.confidence - pred.confidence) < 0.1)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)
```

## Metric Design Best Practices

### 1. Capture What Actually Matters

```python
# BAD: Metric that doesn't capture real quality
def bad_metric(example, pred, trace=None):
    return len(pred.answer) > 10  # Length doesn't mean quality!

# GOOD: Metric that captures task-specific quality
def good_metric(example, pred, trace=None):
    return (
        example.key_fact in pred.answer and
        pred.answer.endswith(".") and  # Complete sentence
        len(pred.answer.split()) >= 5   # Substantive answer
    )
```

### 2. Be Robust to Formatting

```python
def robust_metric(example, pred, trace=None):
    """Handle formatting variations."""
    def clean(text):
        return " ".join(text.lower().split())

    return clean(example.answer) == clean(pred.answer)
```

### 3. Handle Edge Cases

```python
def safe_metric(example, pred, trace=None):
    """Handle missing or empty values."""
    expected = getattr(example, 'answer', '')
    predicted = getattr(pred, 'answer', '')

    if not expected or not predicted:
        return 0.0

    return expected.lower() in predicted.lower()
```

### 4. Make Metrics Interpretable

```python
def interpretable_metric(example, pred, trace=None):
    """Return score with clear meaning."""
    checks = {
        "correct": example.answer.lower() in pred.answer.lower(),
        "complete": len(pred.answer) >= 50,
        "relevant": any(word in pred.answer.lower()
                       for word in example.question.lower().split()),
    }

    # Log which checks failed (useful for debugging)
    failed = [k for k, v in checks.items() if not v]
    if failed and trace is None:  # Only log during evaluation
        print(f"Failed checks: {failed}")

    return sum(checks.values()) / len(checks)
```

## Specialized Metrics for Long-form Content Generation

When evaluating long-form articles like Wikipedia entries, we need specialized metrics that go beyond simple answer correctness. These metrics assess comprehensiveness, factual accuracy, and verifiability.

### Topic Coverage Evaluation

Measures how comprehensively the generated content covers the topic:

```python
def topic_coverage_rouge(example, pred, trace=None):
    """
    Evaluate topic coverage using ROUGE metrics against reference articles.

    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures
    overlap between generated and reference content.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("Install rouge_score: pip install rouge-score")
        return 0.0

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Score against reference content
    scores = scorer.score(
        example.reference_content,
        pred.article_content
    )

    # Use ROUGE-L as primary metric (measures longest common subsequence)
    rouge_l_score = scores['rougeL'].fmeasure

    if trace is not None:
        # During optimization, require good coverage
        return rouge_l_score >= 0.4

    return rouge_l_score

def comprehensive_topic_coverage(example, pred, trace=None):
    """
    More comprehensive topic coverage evaluation.

    Checks coverage of multiple aspects:
    1. Key entities mentioned
    2. Important concepts covered
    3. Topic depth across sections
    """
    # Extract key entities from reference
    reference_entities = set(example.get('key_entities', []))
    generated_text = pred.article_content.lower()

    # Check entity coverage
    entities_covered = sum(
        1 for entity in reference_entities
        if entity.lower() in generated_text
    )
    entity_coverage = entities_covered / len(reference_entities) if reference_entities else 0

    # Check section coverage (if outline provided)
    if hasattr(pred, 'outline') and pred.outline:
        expected_sections = set(example.get('required_sections', []))
        generated_sections = set(s['title'].lower() for s in pred.outline)

        section_coverage = len(expected_sections & generated_sections) / len(expected_sections)
    else:
        section_coverage = 0.5  # Default if no outline

    # Check concept coverage
    reference_concepts = set(example.get('key_concepts', []))
    concepts_covered = sum(
        1 for concept in reference_concepts
        if concept.lower() in generated_text
    )
    concept_coverage = concepts_covered / len(reference_concepts) if reference_concepts else 0

    # Weighted combination
    overall_coverage = (
        0.4 * entity_coverage +
        0.4 * concept_coverage +
        0.2 * section_coverage
    )

    if trace is not None:
        return overall_coverage >= 0.6

    return overall_coverage
```

### Factual Accuracy (FactScore)

FactScore is a metric specifically designed to evaluate factual accuracy in long-form generation:

```python
class FactScoreMetric:
    """
    FactScore: Evaluates factual accuracy by breaking down content into
    atomic claims and verifying each against a knowledge source.
    """

    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        self.claim_extractor = dspy.Predict(
            "text -> atomic_claims"
        )
        self.fact_checker = dspy.ChainOfThought(
            "claim, context -> is_factual, confidence, correction"
        )

    def __call__(self, example, pred, trace=None):
        """
        Calculate FactScore for generated content.

        Returns the average of factual claim scores.
        """
        # Extract atomic claims from generated content
        claims_result = self.claim_extractor(
            text=pred.article_content
        )
        claims = self._parse_claims(claims_result.atomic_claims)

        if not claims:
            return 0.0

        # Verify each claim
        claim_scores = []
        for claim in claims:
            verification = self.fact_checker(
                claim=claim,
                context=example.get('reference_documents', '')
            )

            # Convert confidence to score
            if verification.is_factual.lower() == 'true':
                score = float(verification.confidence)
            else:
                score = 0.0

            claim_scores.append(score)

        # Calculate FactScore (average of verified claims)
        fact_score = sum(claim_scores) / len(claim_scores)

        if trace is not None:
            # During optimization, require high factual accuracy
            return fact_score >= 0.7

        return fact_score

    def _parse_claims(self, claims_text: str) -> List[str]:
        """Parse atomic claims from extracted text."""
        claims = []
        lines = claims_text.strip().split('\n')
        for line in lines:
            if line.strip() and (line.strip().startswith('-') or line.strip().startswith('•')):
                claim = line.strip().lstrip('- •').strip()
                if claim.endswith('.'):
                    claim = claim[:-1]
                claims.append(claim)
        return claims

# Usage
fact_scorer = FactScoreMetric()
def factscore_metric(example, pred, trace=None):
    """Wrapper for FactScore metric."""
    return fact_scorer(example, pred, trace)
```

### Verifiability Assessment

Measures how well claims in the generated content can be verified with citations:

```python
def verifiability_metric(example, pred, trace=None):
    """
    Measures the fraction of sentences that can be verified
    using retrieved evidence or citations.
    """
    sentences = _split_into_sentences(pred.article_content)

    if not sentences:
        return 0.0

    verifiable_count = 0

    for sentence in sentences:
        # Check if sentence has citation
        has_citation = bool(re.search(r'\[\d+\]|\[.*?\]', sentence))

        # Check if sentence is factual claim
        is_factual = _is_factual_claim(sentence)

        # Check if supporting evidence exists
        if hasattr(pred, 'citations') and pred.citations:
            has_evidence = _check_evidence_support(
                sentence,
                pred.citations,
                example.get('reference_documents', '')
            )
        else:
            has_evidence = False

        # Sentence is verifiable if it has citation OR supporting evidence
        if is_factual and (has_citation or has_evidence):
            verifiable_count += 1

    verifiability = verifiable_count / len(sentences)

    if trace is not None:
        # During optimization, require high verifiability
        return verifiability >= 0.6

    return verifiability

def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    import re
    # Simple sentence splitting
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

def _is_factual_claim(sentence: str) -> bool:
    """Determine if a sentence makes a factual claim."""
    factual_indicators = [
        'according to', 'research shows', 'studies indicate',
        'data suggests', 'reported', 'found that', 'demonstrates',
        'proved', 'discovered', 'measured', 'calculated'
    ]

    # Check for numbers (statistics)
    has_numbers = bool(re.search(r'\d+', sentence))

    # Check for factual indicators
    has_indicators = any(ind in sentence.lower() for ind in factual_indicators)

    # Check for specific entities (often indicates facts)
    has_entities = bool(re.search(r'[A-Z][a-z]+ [A-Z][a-z]+', sentence))

    return has_numbers or has_indicators or has_entities

def _check_evidence_support(sentence: str,
                           citations: List[str],
                           reference_docs: str) -> bool:
    """Check if sentence has supporting evidence in references."""
    # Simple check - in practice would use semantic similarity
    sentence_words = set(sentence.lower().split())

    for citation in citations:
        # Extract cited content (simplified)
        cited_content = _extract_citation_content(citation, reference_docs)

        if cited_content:
            cited_words = set(cited_content.lower().split())
            overlap = len(sentence_words & cited_words) / len(sentence_words)

            if overlap > 0.3:  # 30% overlap threshold
                return True

    return False

def _extract_citation_content(citation: str, reference_docs: str) -> str:
    """Extract content for a specific citation."""
    # Simplified - would need proper citation parsing
    if citation in reference_docs:
        return reference_docs.split(citation)[1].split('\n')[0]
    return ""
```

### Citation Quality Metrics

```python
def citation_quality_metric(example, pred, trace=None):
    """
    Evaluates the quality and appropriateness of citations in the article.
    """
    if not hasattr(pred, 'citations') or not pred.citations:
        return 0.0

    total_score = 0.0

    for citation in pred.citations:
        # Check citation format
        format_score = _check_citation_format(citation)

        # Check citation relevance
        relevance_score = _check_citation_relevance(
            citation,
            pred.article_content,
            example.get('reference_documents', '')
        )

        # Check source credibility (if available)
        credibility_score = _check_source_credibility(citation)

        # Combine scores
        citation_score = (
            0.3 * format_score +
            0.5 * relevance_score +
            0.2 * credibility_score
        )

        total_score += citation_score

    average_score = total_score / len(pred.citations)

    if trace is not None:
        return average_score >= 0.7

    return average_score

def _check_citation_format(citation: str) -> float:
    """Check if citation follows expected format."""
    # Check for common citation formats
    patterns = [
        r'\[\d+\]',  # Numeric [1]
        r'\([A-Z][a-z]+, \d{4}\)',  # APA (Smith, 2023)
        r'\([A-Z][a-z]+ et al\., \d{4}\)',  # APA et al.
    ]

    for pattern in patterns:
        if re.search(pattern, citation):
            return 1.0

    return 0.5  # Partial score for unrecognized format

def _check_citation_relevance(citation: str,
                             content: str,
                             references: str) -> float:
    """Check how relevant the citation is to the content."""
    # Simplified - would use semantic similarity in practice
    citation_text = _extract_citation_text(citation, references)

    if not citation_text:
        return 0.0

    # Find where citation is used in content
    citation_context = _find_citation_context(citation, content)

    if not citation_context:
        return 0.0

    # Calculate word overlap
    context_words = set(citation_context.lower().split())
    citation_words = set(citation_text.lower().split())

    overlap = len(context_words & citation_words)
    return min(1.0, overlap / 10)  # Normalize by expected overlap

def _check_source_credibility(citation: str) -> float:
    """Check the credibility of the cited source."""
    # List of credible sources (simplified)
    credible_domains = [
        'nature.com', 'science.org', 'cell.com',
        'arxiv.org', 'scholar.google.com',
        'gov', 'edu', 'ieee.org', 'acm.org'
    ]

    # Extract domain if URL is present
    if 'http' in citation:
        from urllib.parse import urlparse
        try:
            domain = urlparse(citation).netloc
            if any(cred in domain for cred in credible_domains):
                return 1.0
            return 0.5  # Partial for other domains
        except:
            return 0.5

    # For non-URL citations, assume academic source
    return 0.8
```

### Composite Long-form Quality Metric

```python
def longform_composite_metric(example, pred, trace=None):
    """
    Composite metric for evaluating long-form article quality.

    Combines multiple aspects:
    - Topic coverage (ROUGE)
    - Factual accuracy (FactScore)
    - Verifiability
    - Citation quality
    - Coherence and flow
    """
    # Individual component scores
    coverage_score = topic_coverage_rouge(example, pred, trace)
    factual_score = factscore_metric(example, pred, trace)
    verifiability_score = verifiability_metric(example, pred, trace)
    citation_score = citation_quality_metric(example, pred, trace)

    # Coherence score (simplified)
    coherence_score = _evaluate_coherence(pred.article_content)

    # Weighted combination for final score
    final_score = (
        0.25 * coverage_score +
        0.30 * factual_score +
        0.20 * verifiability_score +
        0.15 * citation_score +
        0.10 * coherence_score
    )

    if trace is not None:
        # During optimization, require good overall quality
        return final_score >= 0.6

    return final_score

def _evaluate_coherence(text: str) -> float:
    """Evaluate text coherence and flow."""
    sentences = _split_into_sentences(text)

    if len(sentences) < 2:
        return 1.0

    coherence_scores = []

    # Check transitions between consecutive sentences
    for i in range(len(sentences) - 1):
        current = sentences[i]
        next_sent = sentences[i + 1]

        # Check for transition words
        transitions = ['however', 'therefore', 'furthermore', 'consequently',
                      'moreover', 'in addition', 'in contrast', 'similarly']

        has_transition = any(trans in next_sent.lower() for trans in transitions)

        # Check for pronoun reference to previous sentence
        current_words = set(current.lower().split())
        next_words = set(next_sent.lower().split())

        # Common coherence indicators
        pronouns = {'it', 'they', 'this', 'that', 'these', 'those'}
        pronoun_reference = bool(pronouns & next_words)

        # Topic continuity
        topic_overlap = len(current_words & next_words) / len(current_words | next_words)

        # Score for this transition
        transition_score = (
            0.4 * (1.0 if has_transition else 0.0) +
            0.3 * (1.0 if pronoun_reference else 0.0) +
            0.3 * topic_overlap
        )

        coherence_scores.append(transition_score)

    # Average coherence across all transitions
    return sum(coherence_scores) / len(coherence_scores)
```

## Summary

Effective metrics are the key to meaningful evaluation:

1. **Understand the anatomy**: example, pred, trace parameters
2. **Use built-in metrics** when appropriate (SemanticF1, etc.)
3. **Create custom metrics** for domain-specific needs
4. **Combine multiple aspects** with composite metrics
5. **Use trace appropriately** for optimization vs. evaluation
6. **Employ specialized metrics** for long-form content (ROUGE, FactScore, Verifiability)

### Key Takeaways

1. **Metrics define success** - They determine what optimization improves
2. **The trace parameter** enables optimization-aware behavior
3. **Custom metrics** capture domain-specific quality
4. **Composite metrics** address multiple dimensions
5. **Robustness matters** - Handle edge cases gracefully
6. **Long-form content requires specialized evaluation** beyond simple accuracy

## Next Steps

- [Next Section: Evaluation Loops](./04-evaluation-loops.md) - Run systematic evaluations
- [Best Practices](./05-best-practices.md) - Avoid common pitfalls
- [Examples](../../examples/chapter04/) - See metrics in action

## Further Reading

- [DSPy Metrics Documentation](https://dspy.ai/learn/evaluation/metrics)
- [Evaluation Metrics for NLP](https://huggingface.co/spaces/evaluate-metric)
- [Custom Metrics in Machine Learning](https://scikit-learn.org/stable/modules/model_evaluation.html)

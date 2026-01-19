# Classification Tasks: Building Robust Text Categorization Systems

## Introduction

Text classification is one of the most common and practical applications of natural language processing. From spam detection to sentiment analysis, topic categorization to intent recognition, classification systems power countless real-world applications. DSPy provides powerful tools for building sophisticated classifiers that can handle the complexity and nuances of real-world text data.

## Understanding Text Classification

### Classification Types

1. **Binary Classification**: Two classes (spam/not spam, positive/negative)
2. **Multi-class Classification**: Multiple exclusive categories (news topics, product categories)
3. **Multi-label Classification**: Multiple non-exclusive categories (tags, topics)
4. **Hierarchical Classification**: Nested categories (product taxonomy)

### Real-World Applications

- **Content Moderation**: Detecting inappropriate content
- **Customer Support**: Routing tickets to appropriate departments
- **Market Analysis**: Categorizing news and social media posts
- **Document Management**: Organizing documents by type and topic
- **Intent Recognition**: Understanding user goals in chatbots

## Building Basic Classifiers

### Simple Binary Classifier

```python
import dspy

class BinaryClassifier(dspy.Module):
    def __init__(self, positive_class="positive", negative_class="negative"):
        super().__init__()
        self.positive_class = positive_class
        self.negative_class = negative_class
        self.classify = dspy.Predict("text -> classification, confidence")

    def forward(self, text):
        result = self.classify(text=text)

        # Normalize classification
        classification = result.classification.lower()
        if self.positive_class in classification:
            label = self.positive_class
        elif self.negative_class in classification:
            label = self.negative_class
        else:
            # Fallback based on confidence
            label = self.positive_class if float(result.confidence) > 0.5 else self.negative_class

        return dspy.Prediction(
            classification=label,
            confidence=result.confidence,
            raw_prediction=result.classification
        )
```

### Multi-class Classifier

```python
class MultiClassClassifier(dspy.Module):
    def __init__(self, categories):
        super().__init__()
        self.categories = categories
        categories_str = ", ".join(categories)
        self.classify = dspy.Predict(
            f"text, categories[{categories_str}] -> classification, confidence, reasoning"
        )

    def forward(self, text):
        result = self.classify(text=text, categories=", ".join(self.categories))

        # Ensure classification is in categories
        if result.classification not in self.categories:
            # Find closest match
            result.classification = self._find_closest_category(
                result.classification,
                self.categories
            )

        return dspy.Prediction(
            classification=result.classification,
            confidence=result.confidence,
            reasoning=result.reasoning
        )

    def _find_closest_category(self, prediction, categories):
        """Find the closest matching category using simple string similarity."""
        best_match = categories[0]
        best_score = 0

        for cat in categories:
            if cat.lower() in prediction.lower():
                return cat  # Exact substring match
            # Simple similarity check
            common_words = set(cat.lower().split()) & set(prediction.lower().split())
            score = len(common_words)
            if score > best_score:
                best_score = score
                best_match = cat

        return best_match
```

### Multi-label Classifier

```python
class MultiLabelClassifier(dspy.Module):
    def __init__(self, possible_labels):
        super().__init__()
        self.possible_labels = possible_labels
        labels_str = ", ".join(possible_labels)
        self.extract_labels = dspy.Predict(
            f"text, possible_labels[{labels_str}] -> labels, explanation"
        )

    def forward(self, text):
        result = self.extract_labels(
            text=text,
            possible_labels=", ".join(self.possible_labels)
        )

        # Parse and filter labels
        predicted_labels = []
        for label in result.labels.split(","):
            label = label.strip().lower()
            for possible in self.possible_labels:
                if possible.lower() in label or label in possible.lower():
                    if possible not in predicted_labels:
                        predicted_labels.append(possible)

        return dspy.Prediction(
            labels=predicted_labels,
            explanation=result.explanation,
            raw_output=result.labels
        )
```

### When Multi-label Becomes Extreme

When your label space grows from hundreds to thousands or millions of labels, you're entering the domain of **Extreme Multi-Label Classification (XML)**. Standard multi-label approaches become infeasible due to:

- **Computational Complexity**: O(|L|) per instance becomes prohibitive
- **Memory Constraints**: Storing millions of label embeddings and classifiers
- **Data Sparsity**: Most label pairs rarely co-occur
- **Inference Latency**: Real-time requirements cannot be met

For these extreme scenarios, DSPy provides specialized XML techniques that we explore in depth in **[Extreme Multi-Label Classification](08-extreme-multilabel-classification.md)**. These include:

- Efficient label indexing and similarity search
- Hierarchical label organization
- Zero-shot XML for handling new labels
- Specialized evaluation metrics (P@k, nDCG@k, PS@k)
- Memory-efficient streaming processors

## Advanced Classification Techniques

### Hierarchical Classification

```python
class HierarchicalClassifier(dspy.Module):
    def __init__(self, hierarchy):
        """
        Hierarchy format:
        {
            "Technology": ["AI/ML", "Web Dev", "Mobile"],
            "Business": ["Finance", "Marketing", "Management"],
            "Science": ["Physics", "Chemistry", "Biology"]
        }
        """
        super().__init__()
        self.hierarchy = hierarchy
        self.root_categories = list(hierarchy.keys())

        # Level 1: Root classifier
        self.root_classifier = dspy.Predict(
            f"text, root_categories[{', '.join(self.root_categories)}] -> root_category"
        )

        # Level 2: Sub-category classifiers
        self.sub_classifiers = {}
        for root, subs in hierarchy.items():
            subs_str = ", ".join(subs)
            self.sub_classifiers[root] = dspy.Predict(
                f"text, sub_categories[{subs_str}] -> sub_category"
            )

    def forward(self, text):
        # First level classification
        root_result = self.root_classifier(
            text=text,
            root_categories=", ".join(self.root_categories)
        )

        root_category = root_result.root_category
        if root_category not in self.hierarchy:
            root_category = self._find_closest_root(root_result.root_category)

        # Second level classification
        if root_category in self.sub_classifiers:
            sub_result = self.sub_classifiers[root_category](
                text=text,
                sub_categories=", ".join(self.hierarchy[root_category])
            )
            sub_category = sub_result.sub_category
        else:
            sub_category = "Unknown"

        return dspy.Prediction(
            root_category=root_category,
            sub_category=sub_category,
            full_path=f"{root_category} > {sub_category}"
        )

    def _find_closest_root(self, prediction):
        best_match = self.root_categories[0]
        best_score = 0
        prediction = prediction.lower()

        for root in self.root_categories:
            if root.lower() in prediction:
                return root
            # Simple similarity
            score = len(set(root.lower().split()) & set(prediction.split()))
            if score > best_score:
                best_score = score
                best_match = root

        return best_match
```

### Few-shot Classification with Examples

```python
class FewShotClassifier(dspy.Module):
    def __init__(self, categories):
        super().__init__()
        self.categories = categories
        categories_str = ", ".join(categories)
        self.classify_with_examples = dspy.ChainOfThought(
            f"text, examples, categories[{categories_str}] -> classification, similar_examples, confidence"
        )

    def forward(self, text, examples=None):
        if examples is None:
            examples = []

        # Format examples for the prompt
        examples_text = "\n".join([
            f"Example {i+1}: {ex.text}\nCategory: {ex.category}"
            for i, ex in enumerate(examples[:5])  # Limit examples
        ])

        result = self.classify_with_examples(
            text=text,
            examples=examples_text,
            categories=", ".join(self.categories)
        )

        return dspy.Prediction(
            classification=result.classification,
            similar_examples=result.similar_examples,
            confidence=result.confidence,
            reasoning=result.rationale
        )
```

### Confidence-aware Classification

```python
class ConfidenceClassifier(dspy.Module):
    def __init__(self, categories, confidence_threshold=0.7):
        super().__init__()
        self.categories = categories
        self.confidence_threshold = confidence_threshold
        self.classify = dspy.Predict(
            f"text, categories[{', '.join(categories)}] -> classification, confidence, uncertainty_analysis"
        )
        self.request_clarification = dspy.Predict("text, uncertainty -> clarification_question")

    def forward(self, text):
        result = self.classify(
            text=text,
            categories=", ".join(self.categories)
        )

        confidence = float(result.confidence)

        # Handle low confidence cases
        if confidence < self.confidence_threshold:
            clarification = self.request_clarification(
                text=text,
                uncertainty=result.uncertainty_analysis
            )
            return dspy.Prediction(
                classification="UNCERTAIN",
                confidence=confidence,
                clarification_needed=clarification.clarification_question,
                uncertainty_analysis=result.uncertainty_analysis
            )

        return dspy.Prediction(
            classification=result.classification,
            confidence=confidence,
            uncertainty_analysis=result.uncertainty_analysis
        )
```

## Real-World Classification Applications

### Customer Support Ticket Router

```python
class SupportTicketRouter(dspy.Module):
    def __init__(self):
        super().__init__()
        # Define departments and priorities
        self.departments = [
            "Technical Support", "Billing", "Sales", "Account Management",
            "Product Feedback", "Bug Reports", "Feature Requests"
        ]

        self.priorities = ["Low", "Medium", "High", "Critical", "Urgent"]

        # Classifiers
        self.department_classifier = dspy.ChainOfThought(
            f"ticket_text, departments[{', '.join(self.departments)}] -> department, reasoning"
        )

        self.priority_classifier = dspy.Predict(
            f"ticket_text, priorities[{', '.join(self.priorities)}] -> priority, urgency_factors"
        )

        self.extract_details = dspy.Predict(
            "ticket_text -> product, issue_type, customer_tier"
        )

    def forward(self, ticket_text):
        # Extract basic details
        details = self.extract_details(ticket_text=ticket_text)

        # Classify department
        dept_result = self.department_classifier(
            ticket_text=ticket_text,
            departments=", ".join(self.departments)
        )

        # Classify priority
        priority_result = self.priority_classifier(
            ticket_text=ticket_text,
            priorities=", ".join(self.priorities)
        )

        return dspy.Prediction(
            department=dept_result.department,
            priority=priority_result.priority,
            product=details.product,
            issue_type=details.issue_type,
            customer_tier=details.customer_tier,
            urgency_factors=priority_result.urgency_factors,
            department_reasoning=dept_result.reasoning
        )

# Example usage
router = SupportTicketRouter()
ticket = "My premium account is charged twice this month and I can't access my reports"
routing = router(ticket_text=ticket)

print(f"Department: {routing.department}")  # Billing
print(f"Priority: {routing.priority}")     # High (premium customer)
```

### Content Moderation System

```python
class ContentModerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.categories = [
            "Safe", "Hate Speech", "Spam", "Inappropriate Content",
            "Misinformation", "Harassment", "Violence", "Self-harm"
        ]

        self.moderate = dspy.ChainOfThought(
            f"content, categories[{', '.join(self.categories)}] -> category, severity, explanation"
        )

        self.extract_entities = dspy.Predict("content -> mentioned_users, links, keywords")

        self.check_context = dspy.Predict(
            "content, user_history, platform_context -> contextual_factors"
        )

    def forward(self, content, user_history=None, platform_context=None):
        # Extract entities
        entities = self.extract_entities(content=content)

        # Check context if available
        if user_history and platform_context:
            context = self.check_context(
                content=content,
                user_history=user_history,
                platform_context=platform_context
            )
            contextual_factors = context.contextual_factors
        else:
            contextual_factors = "No additional context"

        # Moderate content
        moderation = self.moderate(
            content=content,
            categories=", ".join(self.categories)
        )

        # Determine action based on category and severity
        action = self._determine_action(
            moderation.category,
            float(moderation.severity) if moderation.severity else 0
        )

        return dspy.Prediction(
            category=moderation.category,
            severity=moderation.severity,
            action=action,
            explanation=moderation.explanation,
            mentioned_users=entities.mentioned_users,
            links=entities.links,
            contextual_factors=contextual_factors,
            reasoning=moderation.rationale
        )

    def _determine_action(self, category, severity):
        """Determine moderation action based on category and severity."""
        if category == "Safe":
            return "Allow"
        elif category in ["Hate Speech", "Violence", "Self-harm"]:
            return "Remove"
        elif category in ["Spam", "Inappropriate Content"]:
            return "Remove" if severity > 0.7 else "Flag"
        else:
            return "Review"
```

### Product Review Sentiment and Aspect Classifier

```python
class ReviewAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.sentiments = ["Positive", "Negative", "Neutral", "Mixed"]
        self.aspects = [
            "Quality", "Price", "Service", "Delivery", "Features",
            "Usability", "Design", "Durability", "Value"
        ]

        self.analyze_sentiment = dspy.Predict(
            f"review, sentiments[{', '.join(self.sentiments)}] -> sentiment, sentiment_score"
        )

        self.extract_aspects = dspy.Predict(
            f"review, aspects[{', '.join(self.aspects)}] -> mentioned_aspects, aspect_sentiments"
        )

        self.summarize_review = dspy.Predict(
            "review, sentiment, aspects -> summary, key_points"
        )

    def forward(self, review_text):
        # Analyze overall sentiment
        sentiment_result = self.analyze_sentiment(
            review=review_text,
            sentiments=", ".join(self.sentiments)
        )

        # Extract aspects and their sentiments
        aspect_result = self.extract_aspects(
            review=review_text,
            aspects=", ".join(self.aspects)
        )

        # Generate summary
        summary_result = self.summarize_review(
            review=review_text,
            sentiment=sentiment_result.sentiment,
            aspects=aspect_result.mentioned_aspects
        )

        return dspy.Prediction(
            overall_sentiment=sentiment_result.sentiment,
            sentiment_score=sentiment_result.sentiment_score,
            mentioned_aspects=aspect_result.mentioned_aspects,
            aspect_sentiments=aspect_result.aspect_sentiments,
            summary=summary_result.summary,
            key_points=summary_result.key_points
        )
```

## Optimizing Classifiers

### Using BootstrapFewShot for Classification

```python
class OptimizedClassifier(dspy.Module):
    def __init__(self, categories):
        super().__init__()
        self.categories = categories
        categories_str = ", ".join(categories)
        self.classify = dspy.Predict(
            f"text, categories[{categories_str}] -> classification, confidence, reasoning"
        )

    def forward(self, text):
        result = self.classify(
            text=text,
            categories=", ".join(self.categories)
        )

        return dspy.Prediction(
            classification=result.classification,
            confidence=result.confidence,
            reasoning=result.reasoning
        )

# Training data
trainset = [
    dspy.Example(
        text="This product is amazing! Highly recommended.",
        classification="Positive",
        confidence=0.9
    ),
    dspy.Example(
        text="Terrible customer service. Would not buy again.",
        classification="Negative",
        confidence=0.85
    ),
    # ... more examples
]

# Evaluation metric
def classification_metric(example, pred, trace=None):
    correct = example.classification.lower() == pred.classification.lower()
    confidence_match = abs(float(example.confidence) - float(pred.confidence)) < 0.2
    return correct and confidence_match

# Optimize
optimizer = BootstrapFewShot(
    metric=classification_metric,
    max_bootstrapped_demos=5,
    max_labeled_demos=5
)
optimized_classifier = optimizer.compile(
    OptimizedClassifier(["Positive", "Negative", "Neutral"]),
    trainset=trainset
)
```

### KNNFewShot for Context-aware Classification

```python
class ContextAwareClassifier(dspy.Module):
    def __init__(self, categories):
        super().__init__()
        self.categories = categories
        self.classify = dspy.Predict(
            f"text, similar_examples, categories[{', '.join(categories)}] -> classification"
        )

    def forward(self, text, similar_examples):
        # Format similar examples
        examples_text = "\n".join([
            f"Similar: {ex.text} -> {ex.category}"
            for ex in similar_examples
        ])

        result = self.classify(
            text=text,
            similar_examples=examples_text,
            categories=", ".join(self.categories)
        )

        return dspy.Prediction(
            classification=result.classification,
            similar_examples_used=similar_examples
        )

# Use KNNFewShot to find similar examples during compilation
knn_optimizer = KNNFewShot(k=3)
context_classifier = knn_optimizer.compile(
    ContextAwareClassifier(["Tech", "Sports", "Politics", "Entertainment"]),
    trainset=classification_trainset
)
```

## Best Practices

### 1. Data Quality and Balance

```python
def prepare_balanced_dataset(raw_data, categories, samples_per_category=100):
    """Create a balanced dataset for training."""
    balanced = []
    category_counts = {cat: 0 for cat in categories}

    for item in raw_data:
        cat = item.category
        if cat in category_counts and category_counts[cat] < samples_per_category:
            balanced.append(item)
            category_counts[cat] += 1

    return balanced
```

### 2. Handle Class Imbalance

```python
class BalancedClassifier(dspy.Module):
    def __init__(self, categories, class_weights=None):
        super().__init__()
        self.categories = categories
        self.class_weights = class_weights or {cat: 1.0 for cat in categories}

    def adjust_prediction(self, prediction, confidence):
        """Adjust confidence based on class weights."""
        if prediction in self.class_weights:
            adjusted_conf = confidence * self.class_weights[prediction]
            return min(adjusted_conf, 1.0)
        return confidence
```

### 3. Error Analysis and Iteration

```python
def analyze_errors(classifier, testset):
    """Analyze classification errors to improve the system."""
    errors = []

    for example in testset:
        prediction = classifier(text=example.text)
        if prediction.classification != example.category:
            errors.append({
                "text": example.text,
                "predicted": prediction.classification,
                "actual": example.category,
                "confidence": prediction.confidence
            })

    # Analyze error patterns
    return analyze_error_patterns(errors)
```

## Evaluation Metrics

### Comprehensive Classification Evaluation

```python
def evaluate_comprehensive(classifier, testset):
    """Evaluate classifier with multiple metrics."""
    from collections import defaultdict

    results = defaultdict(list)

    for example in testset:
        pred = classifier(text=example.text)

        # Basic accuracy
        is_correct = pred.classification == example.category
        results["accuracy"].append(is_correct)

        # Confidence calibration
        results["confidence"].append(float(pred.confidence))

        # Per-category metrics
        results[f"category_{example.category}"].append(is_correct)

    # Calculate metrics
    metrics = {
        "overall_accuracy": sum(results["accuracy"]) / len(results["accuracy"]),
        "average_confidence": sum(results["confidence"]) / len(results["confidence"]),
    }

    # Add per-category accuracy
    for cat in classifier.categories:
        cat_results = results.get(f"category_{cat}", [])
        if cat_results:
            metrics[f"{cat}_accuracy"] = sum(cat_results) / len(cat_results)

    return metrics
```

## Common Challenges and Solutions

### Challenge 1: Ambiguous Categories
**Solution**: Use confidence thresholds and ask for clarification.

### Challenge 2: Concept Drift
**Solution**: Implement continuous learning with new data.

### Challenge 3: Multi-label Complexity
**Solution**: Use threshold-based multi-label classification.

### Challenge 4: Imbalanced Classes
**Solution**: Use weighted loss functions and resampling.

## Key Takeaways

1. **DSPy enables flexible** and powerful text classification systems
2. **Different architectures** suit different classification needs
3. **Optimization significantly improves** classification performance
4. **Real-world applications** require handling of edge cases and uncertainty
5. **Evaluation must be comprehensive** and task-specific
6. **Continuous improvement** is key to maintaining performance

## Next Steps

In the next section, we'll explore **Entity Extraction**, demonstrating how to build systems that can identify and extract structured information from unstructured text.
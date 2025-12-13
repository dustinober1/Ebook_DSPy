# Extreme Multi-Label Classification (XML) Examples

This directory contains comprehensive examples demonstrating Extreme Multi-Label Classification (XML) techniques using DSPy. XML addresses the challenge of classifying items against millions of possible labels efficiently.

## Examples Overview

### 1. `xml-demo.py` - Complete XML System Demo
A complete, working demonstration of an XML system for Wikipedia article tagging.

**Features Demonstrated**:
- Hierarchical label organization
- Efficient candidate selection
- Multi-stage classification pipeline
- XML-specific evaluation metrics (P@k, nDCG@k)
- Performance and scalability analysis

**Run it**:
```bash
cd examples/chapter06
python xml-demo.py
```

### 2. Core Components

The demo implements key XML components:

#### Label Indexing
```python
# Efficient embedding-based similarity search
similar_categories = label_index.search_similar_labels(query_embedding, k=100)
```

#### Hierarchical Organization
```python
# Organize labels in tree structure for context
hierarchy = XMLHierarchy(category_tree)
parent_context = hierarchy.get_label_context("Machine Learning")
```

#### Candidate Selection
```python
# Multi-stage candidate filtering
candidates = selector.select_candidates(
    text="Deep learning advances...",
    max_candidates=1000
)
```

#### Evaluation Metrics
```python
# XML-specific metrics
metrics = evaluator.evaluate_dataset(test_data)
print(f"Precision@5: {metrics['avg_precision@5']:.3f}")
print(f"nDCG@10: {metrics['avg_ndcg@10']:.3f}")
```

## Key XML Concepts Demonstrated

### 1. Scalability Challenges
- Traditional O(|L|) classification is infeasible for millions of labels
- Memory constraints for storing all label embeddings
- Inference latency requirements for real-time applications

### 2. Efficient Solutions
- **Label Embedding Index**: Fast similarity search using FAISS
- **Hierarchical Organization**: Exploiting label taxonomy
- **Candidate Selection**: Reducing label space from millions to thousands
- **Streaming Processing**: Handling massive label spaces with limited memory

### 3. Specialized Techniques
- **Zero-shot XML**: Classifying with new/unseen labels
- **In-Context Learning**: Adapting to new domains with few examples
- **Propensity Scored Evaluation**: Handling label imbalance
- **Meta-Learning**: Rapid domain adaptation

## Real-World Applications

### E-commerce Product Categorization
```python
# Categorize products among millions of categories
product_classifier = ProductXMLClassifier(
    category_hierarchy=ecommerce_taxonomy,
    max_categories=5000
)

predictions = product_classifier.classify(
    product_description="Wireless noise-canceling headphones..."
)
```

### Content Tagging Systems
```python
# Automatic tagging for articles/videos
content_tagger = ContentXMLTagger(
    tag_space=wikipedia_categories,
    embedding_model="sentence-transformers"
)

tags = content_tagger.tag(
    content="Latest advances in quantum computing...",
    max_tags=20
)
```

### Ad Targeting
```python
# Match ads to user interests
ad_targeter = AdXMLSystem(
    keyword_space=ad_keywords,
    user_profiles=user_data
)

targeted_ads = ad_targeter.select_ads(
    user_query="buy running shoes",
    user_context={"location": "NYC", "age": 28}
)
```

## Performance Optimizations

### Memory Efficiency
```python
# Process labels in streams to handle massive spaces
streaming_processor = StreamingXMLProcessor(
    label_streams={
        "electronics": electronics_categories,
        "clothing": clothing_categories,
        "books": book_categories
    },
    batch_size=10000
)

results = streaming_processor.classify(text, classifier)
```

### Inference Speed
```python
# Multi-stage pipeline for fast inference
class FastXMLClassifier:
    def __init__(self):
        self.stage1 = embedding_filter  # Fast: reduces to 10K
        self.stage2 = cluster_selector  # Medium: reduces to 1K
        self.stage3 = neural_classifier  # Slow: final prediction

    def predict(self, text):
        # Stage 1: O(1) embedding lookup
        candidates1 = self.stage1.filter(text, k=10000)

        # Stage 2: O(√|L|) clustering
        candidates2 = self.stage2.select(text, candidates1, k=1000)

        # Stage 3: O(k) neural classification
        return self.stage3.classify(text, candidates2)
```

## Best Practices

### 1. Label Space Design
- Organize labels hierarchically when possible
- Use meaningful label descriptions for zero-shot
- Consider label correlations and co-occurrences
- Balance label frequencies to avoid extreme imbalance

### 2. Model Architecture
- Use multiple specialized modules for different stages
- Implement early stopping for candidate selection
- Cache intermediate results for efficiency
- Parallelize independent operations

### 3. Evaluation
- Use ranking metrics (P@k, nDCG@k) not just accuracy
- Measure inference latency and throughput
- Evaluate on tail labels separately
- Consider user satisfaction metrics

### 4. Production Deployment
- Monitor prediction confidence distributions
- Implement A/B testing for model updates
- Use online learning for label space evolution
- Set up alerts for performance degradation

## Advanced Topics Covered

### Zero-shot XML
```python
# Classify with completely new labels
zero_shot = ZeroShotXML()

new_predictions = zero_shot.predict_new_label(
    text="Introduction to transformer architectures",
    new_label="Vision Transformers",
    label_description="Neural networks for computer vision using transformer architecture"
)
```

### In-Context Learning
```python
# Adapt to new domain with few examples
icl = XMLInContextLearner(support_set=new_domain_examples)

adapted_classifier = icl.adapt_to_domain(
    base_classifier=general_xml_classifier,
    support_set=medical_articles[:50]
)
```

### Meta-Learning
```python
# Rapid adaptation to new tasks
meta_learner = XMLMetaLearner(base_classifier)

for task in new_domains:
    adapted = meta_learner.adapt_to_domain(
        support_set=task.support_set,
        query_example=task.example
    )
    # Use adapted classifier for this domain
```

## Scaling Considerations

### When to Use XML
- Label space > 10,000 labels
- Real-time inference requirements (< 100ms)
- Memory constraints (< 4GB for labels)
- Frequent label additions/updates

### When Traditional ML Suffices
- Label space < 1,000 labels
- Batch processing acceptable
- Labels are relatively static
- Simple binary/multi-class classification

## Resources

### Papers
- [Extreme Classification: A Survey](https://arxiv.org/abs/2009.09551)
- [DeepXML: A Deep Extreme Multi-Label Learning Framework](https://arxiv.org/abs/1909.03615)
- [LightXML: Extreme Multi-Label Text Classification](https://arxiv.org/abs/2004.06337)

### Libraries
- [FAISS](https://github.com/facebookresearch/faiss) - Efficient similarity search
- [XGBoost](https://github.com/dmlc/xgboost) - Tree-based XML methods
- [PyTorch](https://pytorch.org/) - Neural network implementations

### Datasets
- [Wikipedia-500K](https://github.com/layer6ai-Labs/xtreme-multilabel) - 500K Wikipedia categories
- [Amazon-3M](http://jmcauley.ucsd.edu/data/amazon/) - 3M product categories
- [EUR-Lex](https://www.cl.cam.ac.uk/research/nl/bea/eurolex/) - Legal document categories

## Exercises

To practice XML concepts, see:
- [XML Exercises](../../../exercises/chapter06/xml-exercises.md)
- Hands-on implementations of all components
- Real-world application challenges
- Performance optimization tasks

## Troubleshooting

### Common Issues

1. **Memory Errors**:
   - Use streaming processing for large label spaces
   - Implement label clustering
   - Reduce embedding dimensionality

2. **Slow Inference**:
   - Optimize candidate selection
   - Use approximate nearest neighbor search
   - Implement caching

3. **Poor Precision@k**:
   - Improve label embeddings
   - Use better candidate selection
   - Incorporate label hierarchy

4. **Tail Label Performance**:
   - Use propensity-scored evaluation
   - Implement resampling techniques
   - Use hierarchical boosting

### Debugging Tips

```python
# Analyze candidate quality
def debug_candidates(text, candidates, true_labels):
    candidates_set = set(candidates)
    recall_at_candidate = len(set(true_labels) & candidates_set) / len(true_labels)
    print(f"Candidate recall: {recall_at_candidate:.3f}")

# Check label embedding quality
def debug_embeddings(label_index, sample_labels):
    for label in sample_labels:
        similar = label_index.search_similar_labels(
            label_index.get_embedding(label), k=5
        )
        print(f"\nSimilar to '{label}':")
        for sim_label, score in similar:
            print(f"  {sim_label}: {score:.3f}")

# Profile inference time
import cProfile

def profile_inference(classifier, text):
    profiler = cProfile.Profile()
    profiler.enable()
    result = classifier(text)
    profiler.disable()
    profiler.print_stats(sort='cumulative')
```

## Contributing

To add new examples or improve existing ones:
1. Ensure code follows DSPy patterns
2. Include comprehensive documentation
3. Add performance benchmarks
4. Test with various label space sizes
5. Update this README with new examples

Remember: XML is about making the impossible (millions of labels) possible through intelligent algorithms and careful engineering!
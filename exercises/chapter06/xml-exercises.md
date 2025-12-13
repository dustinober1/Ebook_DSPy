# Chapter 6 Exercises: Extreme Multi-Label Classification

These exercises help you master Extreme Multi-Label Classification (XML) concepts and implement scalable solutions using DSPy.

## Exercise 1: XML Fundamentals

### Problem 1.1: Understanding XML Challenges

**Task**: Explain why traditional multi-label classification fails at extreme scales and identify the key challenges in XML.

**Requirements**:
1. List at least 5 major challenges specific to XML
2. For each challenge, explain why it becomes critical at scale
3. Provide a real-world example where each challenge manifests

**Solution Approach**:
- Consider computational complexity (O(|L|) problems)
- Think about memory requirements for millions of labels
- Analyze data sparsity issues
- Evaluate inference latency requirements
- Consider label correlation and hierarchy

### Problem 1.2: Label Space Analysis

**Task**: Analyze a label space to determine if it requires XML techniques.

**Dataset**:
```python
# Example dataset with varying label counts
datasets = {
    "news_topics": ["Sports", "Politics", "Technology", "Entertainment"],  # 4 labels
    "product_categories": [f"Category_{i}" for i in range(1000)],  # 1K labels
    "wikipedia_categories": [f"WP_Category_{i}" for i in range(2000000)],  # 2M labels
    "gene_ontology": [f"GO_{i}" for i in range(50000)]  # 50K labels
}
```

**Requirements**:
1. Calculate memory requirements for storing label embeddings (768-dim each)
2. Estimate inference time for O(|L|) approach at 100ms per label
3. Determine which datasets require XML techniques
4. Justify your decisions

## Exercise 2: Label Indexing and Search

### Problem 2.1: Implement Label Embedding Index

**Task**: Complete the implementation of an efficient label index for XML.

**Starter Code**:
```python
import numpy as np
import faiss

class XMLEmbeddingIndex:
    def __init__(self, labels, embedding_dim=768):
        self.labels = labels
        self.embedding_dim = embedding_dim
        # TODO: Initialize data structures

    def build_index(self, embeddings):
        """Build FAISS index for fast similarity search."""
        # TODO: Implement efficient index construction
        pass

    def search(self, query_embedding, k=100):
        """Find k most similar labels."""
        # TODO: Implement fast similarity search
        pass

    def update_index(self, label, new_embedding):
        """Update index with new label."""
        # TODO: Implement efficient update
        pass
```

**Requirements**:
1. Use FAISS for efficient similarity search
2. Support cosine similarity
3. Handle millions of labels
4. Support incremental updates

### Problem 2.2: Hierarchical Label Organization

**Task**: Implement a hierarchical label structure for Wikipedia categories.

**Sample Hierarchy**:
```python
wikipedia_hierarchy = {
    "Science": {
        "Computer Science": {
            "Artificial Intelligence": ["Machine Learning", "Deep Learning", "NLP"],
            "Software Engineering": ["Algorithms", "Databases", "Security"]
        },
        "Natural Sciences": {
            "Physics": ["Quantum Mechanics", "Thermodynamics", "Electromagnetism"],
            "Biology": ["Genetics", "Ecology", "Cell Biology"]
        }
    },
    "Technology": {
        "Computing": ["Hardware", "Software", "Networking"],
        "Engineering": ["Mechanical", "Electrical", "Civil"]
    }
}
```

**Requirements**:
1. Flatten hierarchy into label list
2. Build parent-child relationships
3. Calculate depth for each label
4. Implement efficient context retrieval

## Exercise 3: XML Classification Implementation

### Problem 3.1: Candidate Selection Strategy

**Task**: Implement efficient candidate selection for XML.

**Requirements**:
1. Use embedding similarity for initial filtering
2. Apply label clustering for group-based selection
3. Use DSPy for intelligent refinement
4. Ensure O(√|L|) or better complexity

**Starter Code**:
```python
class XMLCandidateSelector:
    def __init__(self, label_index, clusterer=None):
        self.label_index = label_index
        self.clusterer = clusterer

    def select_candidates(self, text_embedding, max_candidates=1000):
        """Select relevant candidate labels efficiently."""
        # TODO: Implement multi-stage selection
        candidates = []

        # Stage 1: Embedding-based similarity
        # TODO: Add code

        # Stage 2: Cluster-based selection
        # TODO: Add code

        # Stage 3: DSPy refinement
        # TODO: Add code

        return candidates[:max_candidates]
```

### Problem 3.2: Zero-Shot XML Classification

**Task**: Implement zero-shot classification for new labels.

**Scenario**: You have a trained XML classifier for 100K labels, but need to classify with 10 new labels that weren't in training.

**Requirements**:
1. Use label descriptions to understand new labels
2. Implement similarity-based scoring
3. Use few-shot examples if available
4. Combine multiple scoring strategies

## Exercise 4: XML Evaluation Metrics

### Problem 4.1: Implement XML Evaluation Suite

**Task**: Complete implementation of XML-specific evaluation metrics.

**Starter Code**:
```python
class XMLEvaluator:
    def __init__(self, k_values=[1, 3, 5, 10]):
        self.k_values = k_values

    def precision_at_k(self, true_labels, predicted_labels, k):
        """Calculate Precision@k."""
        # TODO: Implement Precision@k
        pass

    def ndcg_at_k(self, true_labels, predicted_labels, k):
        """Calculate Normalized DCG@k."""
        # TODO: Implement nDCG@k
        pass

    def ps_at_k(self, true_labels, predicted_labels, k, label_frequencies):
        """Calculate Propensity Scored Precision@k."""
        # TODO: Implement PS@k with label frequency weighting
        pass
```

**Requirements**:
1. Implement all XML metrics correctly
2. Handle edge cases (empty predictions, all predictions correct)
3. Support batch evaluation
4. Calculate confidence intervals

### Problem 4.2: Comparative Analysis

**Task**: Compare different evaluation metrics on synthetic data.

**Requirements**:
1. Generate synthetic XML predictions with varying quality
2. Calculate all evaluation metrics
3. Analyze metric behaviors
4. Identify metric correlations and differences

## Exercise 5: Advanced XML Techniques

### Problem 5.1: In-Context Learning for XML

**Task**: Implement an in-context learning system for XML adaptation.

**Scenario**: Adapt a general XML classifier to a specific domain (e.g., medical literature) using only 50 examples.

**Requirements**:
1. Select relevant examples from support set
2. Format examples effectively for prompting
3. Handle varying numbers of labels per example
4. Measure adaptation effectiveness

### Problem 5.2: Streaming XML Processor

**Task**: Design and implement a memory-efficient streaming processor for 10M labels.

**Requirements**:
1. Process labels in batches to manage memory
2. Maintain priority queue for top-k predictions
3. Support multiple label streams
4. Measure and optimize throughput

## Exercise 6: Real-World Application

### Problem 6.1: E-commerce Product Categorization

**Task**: Build an XML system for categorizing products in a large e-commerce platform.

**Dataset Characteristics**:
- 5M unique product categories
- Hierarchical category structure (3 levels deep)
- 10M products for training
- Average 3 categories per product

**Requirements**:
1. Design appropriate label indexing strategy
2. Implement efficient candidate selection
3. Handle category hierarchy effectively
4. Optimize for real-time inference (< 100ms)

### Problem 6.2: Wikipedia Automatic Tagging

**Task**: Create an automated tagging system for Wikipedia articles.

**Requirements**:
1. Use existing Wikipedia category hierarchy
2. Handle article metadata (title, sections, references)
3. Ensure tag notability and relevance
4. Validate against human tagging patterns

## Exercise 7: Performance Optimization

### Problem 7.1: Inference Optimization

**Task**: Optimize XML inference for production deployment.

**Benchmarks to Achieve**:
- Latency: < 50ms per classification
- Throughput: > 1000 classifications/second
- Memory: < 2GB RAM usage
- GPU utilization: > 80%

**Optimization Techniques to Explore**:
1. Batch processing
2. Caching strategies
3. Model quantization
4. Parallel processing

### Problem 7.2: Memory-Efficient Training

**Task**: Design a memory-efficient training strategy for XML models.

**Constraints**:
- Training set: 100M instances
- Label space: 1M labels
- Available RAM: 32GB
- GPU memory: 16GB

## Challenge Problem: End-to-End XML System

**Task**: Build a complete XML system from scratch for a real dataset.

**Dataset**: Amazon Product Category Dataset (can be downloaded from public sources)

**Requirements**:
1. Data preprocessing and label hierarchy construction
2. Efficient label indexing implementation
3. Multi-stage XML classifier
4. Comprehensive evaluation
5. Performance optimization
6. Documentation and reproducibility

**Evaluation Criteria**:
- Model performance (nDCG@10, PS@5)
- Inference efficiency (latency, throughput)
- Code quality and documentation
- Innovation in approach
- Reproducibility of results

## Bonus Exercises

### Bonus 1: XML visualization
Create visualizations for:
- Label distribution and hierarchy
- Prediction confidence analysis
- Error analysis and patterns
- Performance profiling

### Bonus 2: Research paper implementation
Implement a recent XML research paper in DSPy:
- Choose a paper from top ML conferences
- Reproduce the core ideas
- Compare with baseline approaches
- Analyze strengths and weaknesses

### Bonus 3: Cross-domain XML transfer
Implement transfer learning for XML:
- Pretrain on large source domain (e.g., Wikipedia)
- Fine-tune on smaller target domain
- Measure transfer effectiveness
- Analyze domain shift effects

## Solutions

Check the solutions directory for detailed implementations and explanations of these exercises. Each solution includes:
- Complete code implementation
- Performance benchmarks
- Detailed explanations
- Possible extensions and improvements

Remember to start with the fundamentals and gradually work your way up to the more advanced challenges. The key to mastering XML is understanding the trade-offs between accuracy, efficiency, and scalability.
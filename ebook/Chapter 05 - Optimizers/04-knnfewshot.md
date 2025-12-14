# KNNFewShot: Similarity-Based Example Selection

## Introduction

KNNFewShot is a DSPy optimizer that uses K-Nearest Neighbors algorithm to select the most relevant examples for each query. Unlike other optimizers that generate or optimize examples globally, KNNFewShot dynamically selects context-specific examples based on similarity to the current input.

## How KNNFewShot Works

### Core Concept
1. **Embed Training Examples**: Convert all training examples to vector representations
2. **Query Embedding**: Embed the new query/question
3. **Similarity Search**: Find K most similar training examples
4. **Dynamic Selection**: Use these examples as few-shot demonstrations

### Key Advantages
- **Context-aware**: Different examples for different queries
- **Scalable**: Efficient even with large training sets
- **Interpretable**: Easy to understand why examples were selected
- **Adaptable**: Works with any similarity metric

## Basic Usage

### Simple Example

```python
import dspy
from dspy.teleprompter import KNNFewShot

# 1. Define your program
class QAProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict("question, context -> answer")

    def forward(self, question):
        return self.generate(question=question)

# 2. Prepare training data
trainset = [
    dspy.Example(
        question="What is the capital of France?",
        answer="Paris",
        topic="geography"
    ),
    dspy.Example(
        question="Who wrote Romeo and Juliet?",
        answer="William Shakespeare",
        topic="literature"
    ),
    # ... many more examples
]

# 3. Create KNNFewShot optimizer
optimizer = KNNFewShot(k=3)  # Use 3 nearest neighbors

# 4. Compile the program
compiled_qa = optimizer.compile(QAProgram(), trainset=trainset)

# 5. Use with dynamic example selection
result = compiled_qa(question="What is the capital of Germany?")
print(result.answer)

# The 3 most similar geography questions were automatically included!
```

## Advanced Configuration

### Custom Similarity Metrics

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Custom embedding model
encoder = SentenceTransformer('all-MiniLM-L6-v2')

def custom_similarity(query, example):
    """Calculate similarity using embeddings."""
    query_emb = encoder.encode(query)
    example_emb = encoder.encode(example.question)

    # Cosine similarity
    similarity = np.dot(query_emb, example_emb) / (
        np.linalg.norm(query_emb) * np.linalg.norm(example_emb)
    )

    return similarity

# Use with custom similarity
optimizer = KNNFewShot(
    k=5,
    similarity_fn=custom_similarity,
    vectorizer=encoder.encode  # Direct embedding function
)
```

### Field-Specific Similarity

```python
def field_weighted_similarity(query, example):
    """Calculate similarity with field weights."""
    weights = {
        'question': 0.6,
        'topic': 0.3,
        'difficulty': 0.1
    }

    similarities = []
    for field, weight in weights.items():
        if hasattr(query, field) and hasattr(example, field):
            sim = text_similarity(getattr(query, field), getattr(example, field))
            similarities.append(weight * sim)

    return sum(similarities)

# Example with custom fields
class WeightedExample:
    def __init__(self, question, answer, topic, difficulty):
        self.question = question
        self.answer = answer
        self.topic = topic
        self.difficulty = difficulty

weighted_trainset = [
    WeightedExample(
        question="What is photosynthesis?",
        answer="Process by which plants convert sunlight to energy",
        topic="biology",
        difficulty="medium"
    ),
    # ... more examples
]

optimizer = KNNFewShot(
    k=4,
    similarity_fn=field_weighted_similarity
)
```

## KNNFewShot Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | 3 | Number of neighbors to retrieve |
| `vectorizer` | Callable | None | Function to convert examples to vectors |
| `similarity_fn` | Callable | None | Custom similarity function |
| `embedding_model` | str | "text-embedding-ada-002" | OpenAI embedding model |

### Advanced Parameters

```python
optimizer = KNNFewShot(
    k=5,
    vectorizer=my_vectorizer,
    similarity_fn=my_similarity,
    embedding_model="text-embedding-3-large",
    max_tokens=8192,          # Maximum tokens for context
    include_metadata=True,    # Include similarity scores
    cache_embeddings=True,    # Cache embeddings for speed
    diversity_boost=0.1,      # Encourage diverse selections
    exclude_self=True,        # Exclude exact matches
    batch_size=100           # Batch size for embedding
)
```

## Working with Different Data Types

### Text Classification

```python
class TextClassifier(dspy.Module):
    def __init__(self, categories):
        super().__init__()
        self.classify = dspy.Predict(
            f"text, similar_examples[{','.join(categories)}] -> category"
        )

    def forward(self, text):
        return self.classify(text=text)

# Training data with categories
classification_trainset = [
    dspy.Example(
        text="I love this product!",
        category="positive"
    ),
    dspy.Example(
        text="This is terrible quality.",
        category="negative"
    ),
    # ... more examples
]

# KNNFewShot for classification
optimizer = KNNFewShot(
    k=3,
    vectorizer=lambda x: x.text  # Use text field for similarity
)

classifier = optimizer.compile(
    TextClassifier(["positive", "negative", "neutral"]),
    trainset=classification_trainset
)

# Dynamic examples based on text similarity
result = classifier(text="This works great!")
print(result.category)  # Likely "positive" due to similar examples
```

### Multi-Modal Data

```python
class MultimodalRetriever(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Predict(
            "query, image_description, similar_examples -> response"
        )

    def forward(self, query, image_description):
        return self.retrieve(
            query=query,
            image_description=image_description
        )

# Multi-modal training data
multimodal_trainset = [
    dspy.Example(
        query="What color is the car?",
        image_description="A red sports car driving on a highway",
        response="The car is red"
    ),
    # ... more examples
]

def multimodal_similarity(query, example):
    """Combined text and image similarity."""
    text_sim = text_similarity(query.query, example.query)
    image_sim = text_similarity(query.image_description, example.image_description)

    return 0.6 * text_sim + 0.4 * image_sim

optimizer = KNNFewShot(
    k=4,
    similarity_fn=multimodal_similarity
)

retriever = optimizer.compile(MultimodalRetriever(), trainset=multimodal_trainset)
```

## Performance Optimization

### Caching Embeddings

```python
import pickle
from pathlib import Path

class CachedKNNFewShot:
    def __init__(self, k=3, cache_dir="./embeddings"):
        self.k = k
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.embedding_cache = {}

    def get_or_create_embedding(self, example):
        """Get cached embedding or create new one."""
        example_id = str(hash(str(example)))
        cache_file = self.cache_dir / f"{example_id}.pkl"

        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        else:
            embedding = create_embedding(example)
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
            return embedding

# Use cached version
optimizer = CachedKNNFewShot(k=5)
```

### Batch Processing

```python
class BatchKNNFewShot:
    def __init__(self, k=3, batch_size=100):
        self.k = k
        self.batch_size = batch_size
        self.embeddings = None

    def fit(self, trainset):
        """Pre-compute all embeddings."""
        self.trainset = trainset

        # Process in batches
        embeddings = []
        for i in range(0, len(trainset), self.batch_size):
            batch = trainset[i:i + self.batch_size]
            batch_embeddings = embed_batch(batch)
            embeddings.extend(batch_embeddings)

        self.embeddings = np.array(embeddings)

    def find_neighbors(self, query, k=None):
        """Find k nearest neighbors efficiently."""
        if self.embeddings is None:
            raise ValueError("Must call fit() first")

        k = k or self.k
        query_emb = embed_single(query)

        # Vectorized similarity computation
        similarities = np.dot(self.embeddings, query_emb)
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        return [self.trainset[i] for i in top_k_indices]

# Efficient for large datasets
optimizer = BatchKNNFewShot(k=5, batch_size=500)
optimizer.fit(trainset)
```

## Similarity Functions

### Text-Based Similarity

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def tfidf_similarity(query, example):
    """TF-IDF based similarity."""
    vectorizer = TfidfVectorizer().fit([query, example])
    vectors = vectorizer.transform([query, example])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

def jaccard_similarity(query, example):
    """Jaccard similarity on word sets."""
    q_words = set(query.lower().split())
    e_words = set(example.lower().split())

    intersection = len(q_words & e_words)
    union = len(q_words | e_words)

    return intersection / union if union > 0 else 0

def fuzzy_similarity(query, example):
    """Fuzzy matching with edit distance."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, query, example).ratio()
```

### Semantic Similarity

```python
def semantic_similarity(query, example, model=None):
    """Deep semantic similarity using embeddings."""
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode([query, example])
    query_emb, example_emb = embeddings[0], embeddings[1]

    # Cosine similarity
    return np.dot(query_emb, example_emb) / (
        np.linalg.norm(query_emb) * np.linalg.norm(example_emb)
    )

def domain_specific_similarity(query, example):
    """Similarity with domain-specific weighting."""
    # Domain keywords
    domain_keywords = {
        'medical': ['patient', 'diagnosis', 'treatment', 'symptom'],
        'legal': ['contract', 'law', 'court', 'legal'],
        'financial': ['investment', 'return', 'risk', 'portfolio']
    }

    # Check domain overlap
    query_domains = sum(1 for domain, words in domain_keywords.items()
                       if any(word in query.lower() for word in words))
    example_domains = sum(1 for domain, words in domain_keywords.items()
                         if any(word in example.lower() for word in words))

    # Domain similarity bonus
    domain_bonus = query_domains * example_domains * 0.1

    # Combine with semantic similarity
    base_sim = semantic_similarity(query, example)

    return base_sim + domain_bonus
```

## Best Practices

### 1. Choose Appropriate k Value

```python
def find_optimal_k(program, trainset, valset, k_values=[1, 3, 5, 7, 10]):
    """Find the best k value through validation."""
    results = {}

    for k in k_values:
        optimizer = KNNFewShot(k=k)
        compiled = optimizer.compile(program, trainset=trainset)
        score = evaluate(compiled, valset)
        results[k] = score

    best_k = max(results, key=results.get)
    return best_k, results

best_k, all_scores = find_optimal_k(my_program, trainset, valset)
print(f"Best k: {best_k}")
print(f"All scores: {all_scores}")
```

### 2. Clean and Normalize Data

```python
def clean_example(example):
    """Clean and normalize example text."""
    if hasattr(example, 'question'):
        example.question = normalize_text(example.question)
    if hasattr(example, 'answer'):
        example.answer = normalize_text(example.answer)
    return example

def normalize_text(text):
    """Normalize text for better similarity matching."""
    import re
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

clean_trainset = [clean_example(ex) for ex in raw_trainset]
```

### 3. Handle Data Imbalance

```python
class BalancedKNNFewShot:
    def __init__(self, k=3, balance_by='category'):
        self.k = k
        self.balance_by = balance_by

    def find_balanced_neighbors(self, query, trainset):
        """Find neighbors with balanced categories."""
        # Group by category
        by_category = {}
        for example in trainset:
            cat = getattr(example, self.balance_by, 'unknown')
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(example)

        # Select neighbors from different categories
        neighbors = []
        categories = list(by_category.keys())

        for i in range(self.k):
            category = categories[i % len(categories)]
            category_examples = by_category[category]

            # Find best in this category
            best = max(category_examples,
                      key=lambda ex: similarity(query, ex))
            neighbors.append(best)

        return neighbors
```

### 4. Monitor Similarity Quality

```python
def analyze_similarity_distribution(trainset, sample_size=1000):
    """Analyze similarity score distribution."""
    import random

    # Sample pairs
    pairs = random.sample(trainset, min(sample_size, len(trainset)))
    similarities = []

    for i in range(len(pairs)):
        for j in range(i + 1, min(i + 10, len(pairs))):
            sim = semantic_similarity(pairs[i].question, pairs[j].question)
            similarities.append(sim)

    # Statistics
    import numpy as np
    print(f"Mean similarity: {np.mean(similarities):.3f}")
    print(f"Std similarity: {np.std(similarities):.3f}")
    print(f"Median similarity: {np.median(similarities):.3f}")

    # Plot distribution
    import matplotlib.pyplot as plt
    plt.hist(similarities, bins=50, alpha=0.75)
    plt.title('Similarity Score Distribution')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.show()

# Use before training
analyze_similarity_distribution(trainset)
```

## Common Pitfalls and Solutions

### Pitfall 1: Poor Similarity Metric
```python
# Problem: Using raw text similarity for semantic tasks
optimizer = KNNFewShot(similarity_fn=lambda q, e: q in e)

# Solution: Use semantic similarity
optimizer = KNNFewShot(
    similarity_fn=lambda q, e: semantic_similarity(q.question, e.question)
)
```

### Pitfall 2: Large Context Window
```python
# Problem: Too many examples exceed context limit
optimizer = KNNFewShot(k=20)  # May exceed token limit

# Solution: Dynamic k based on content
def dynamic_k(query, examples):
    """Choose k based on content length."""
    avg_length = sum(len(str(ex)) for ex in examples) / len(examples)
    max_tokens = 4000  # Leave room for query

    k = max(1, min(5, max_tokens // (avg_length * 2)))
    return k

optimizer = KNNFewShot(k=dynamic_k)
```

### Pitfall 3: Overfitting to Training Data
```python
# Problem: Always selecting exact matches
def overfitting_similarity(query, example):
    return query.lower() == example.question.lower()

# Solution: Include some diversity
def diverse_similarity(query, example):
    base_sim = semantic_similarity(query, example)

    # Penalty for exact matches
    if query.lower() == example.question.lower():
        base_sim *= 0.9

    return base_sim
```

## Key Takeaways

1. KNNFewShot provides context-aware example selection
2. It's efficient and scalable for large datasets
3. Custom similarity functions can dramatically improve performance
4. Proper data cleaning and normalization is essential
5. Monitor similarity distributions to understand your data
6. Balance between similarity and diversity for best results

## Next Steps

In the next section, we'll explore fine-tuning, which adapts small language models for specific tasks within DSPy.
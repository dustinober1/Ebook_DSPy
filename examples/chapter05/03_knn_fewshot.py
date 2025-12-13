"""
KNNFewShot Optimization Examples

This file demonstrates KNNFewShot, DSPy's similarity-based optimizer that
dynamically selects relevant examples based on input similarity.

Examples include:
- Basic KNNFewShot usage
- Custom similarity functions
- Embedding-based similarity
- Field-specific similarity
- Performance optimization
"""

import dspy
from dspy.teleprompter import KNNFewShot
from typing import List, Dict, Any, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Configure language model (placeholder)
dspy.settings.configure(lm=dspy.LM(model="gpt-3.5-turbo", api_key="your-key"))

# Example 1: Basic KNNFewShot with Text Similarity
def basic_knn_optimization():
    """Demonstrate basic KNNFewShot for QA tasks."""
    print("=" * 60)
    print("Example 1: Basic KNNFewShot Optimization")
    print("=" * 60)

    class ContextualQA(dspy.Module):
        def __init__(self):
            super().__init__()
            self.answer = dspy.Predict("question, similar_examples -> answer")

        def forward(self, question):
            return self.answer(question=question)

    # Training data with different topics
    trainset = [
        # Math questions
        dspy.Example(
            question="What is 15 + 27?",
            answer="42",
            topic="mathematics"
        ),
        dspy.Example(
            question="If a box has 8 rows of 5 items, how many total?",
            answer="40",
            topic="mathematics"
        ),
        dspy.Example(
            question="What is 144 ÷ 12?",
            answer="12",
            topic="mathematics"
        ),
        # Geography questions
        dspy.Example(
            question="What is the capital of Japan?",
            answer="Tokyo",
            topic="geography"
        ),
        dspy.Example(
            question="Which country has the most population?",
            answer="China",
            topic="geography"
        ),
        dspy.Example(
            question="What continent is Brazil in?",
            answer="South America",
            topic="geography"
        ),
        # Science questions
        dspy.Example(
            question="What is the chemical symbol for gold?",
            answer="Au",
            topic="science"
        ),
        dspy.Example(
            question="How many planets in our solar system?",
            answer="8",
            topic="science"
        ),
    ]

    # Test questions
    test_questions = [
        "What is 25 × 4?",  # Similar to math
        "What is the capital of Italy?",  # Similar to geography
        "What is the chemical symbol for silver?",  # Similar to science
    ]

    # Simple text similarity function
    def text_similarity(query, example):
        """Calculate basic text similarity."""
        query_words = set(str(query).lower().split())
        example_words = set(str(example.question).lower().split())

        # Jaccard similarity
        intersection = len(query_words & example_words)
        union = len(query_words | example_words)

        return intersection / union if union > 0 else 0

    # Create KNNFewShot optimizer
    optimizer = KNNFewShot(
        k=3,
        similarity_fn=text_similarity
    )

    # Compile
    print("Compiling with KNNFewShot...")
    qa_system = ContextualQA()
    compiled_qa = optimizer.compile(qa_system, trainset=trainset)

    # Test with different questions
    print("\nTesting context-aware example selection:")
    print("-" * 40)

    for question in test_questions:
        result = compiled_qa(question=question)

        print(f"\nQuestion: {question}")
        print(f"Answer: {result.answer}")

        # Manually find similar examples to show what was likely selected
        similarities = [(ex, text_similarity(question, ex)) for ex in trainset]
        similarities.sort(key=lambda x: x[1], reverse=True)

        print(f"\nMost similar examples (k=3):")
        for i, (example, sim) in enumerate(similarities[:3]):
            print(f"  {i+1}. [{example.topic}] {example.question} (similarity: {sim:.2f})")

# Example 2: Semantic Similarity with Embeddings
def semantic_knn_optimization():
    """Demonstrate KNNFewShot with semantic similarity."""
    print("\n" + "=" * 60)
    print("Example 2: Semantic Similarity with Embeddings")
    print("=" * 60)

    class TopicClassifier(dspy.Module):
        def __init__(self, categories):
            super().__init__()
            self.categories = categories
            self.classify = dspy.Predict("text, similar_examples -> category")

        def forward(self, text):
            return self.classify(text=text)

    # Text classification training data
    trainset = [
        dspy.Example(
            text="The company reported record profits this quarter",
            category="business"
        ),
        dspy.Example(
            text="Stock prices surged after the earnings announcement",
            category="business"
        ),
        dspy.Example(
            text="The team won their fifth championship in a row",
            category="sports"
        ),
        dspy.Example(
            text="The quarterback threw a touchdown in the final seconds",
            category="sports"
        ),
        dspy.Example(
            text="New vaccine shows 95% effectiveness in trials",
            category="health"
        ),
        dspy.Example(
            text="The study revealed important health benefits of exercise",
            category="health"
        ),
    ]

    # Mock sentence transformer (in practice, use actual model)
    class MockSentenceTransformer:
        def encode(self, texts):
            """Mock embedding generation."""
            if isinstance(texts, str):
                texts = [texts]

            embeddings = []
            for text in texts:
                # Create deterministic mock embeddings based on text
                embedding = np.zeros(100)
                for i, char in enumerate(text[:100]):
                    embedding[i] = ord(char) / 255.0
                embeddings.append(embedding)

            return np.array(embeddings)

        def __call__(self, text):
            return self.encode(text)

    # Initialize model
    encoder = MockSentenceTransformer()

    def semantic_similarity(query, example):
        """Calculate semantic similarity using embeddings."""
        # Get embeddings
        query_emb = encoder.encode(str(query))
        example_emb = encoder.encode(str(example.text))

        # Reshape for cosine similarity
        query_emb = query_emb.reshape(1, -1)
        example_emb = example_emb.reshape(1, -1)

        # Calculate cosine similarity
        similarity = cosine_similarity(query_emb, example_emb)[0][0]

        return similarity

    # Create categories
    categories = ["business", "sports", "health"]

    # Create optimizer with semantic similarity
    optimizer = KNNFewShot(
        k=3,
        similarity_fn=semantic_similarity,
        vectorizer=encoder.encode  # Pre-compute embeddings
    )

    # Compile
    print("Compiling with semantic similarity...")
    classifier = TopicClassifier(categories)
    compiled_classifier = optimizer.compile(classifier, trainset=trainset)

    # Test texts
    test_texts = [
        "The corporation exceeded analyst expectations",
        "The athlete broke the world record",
        "Regular exercise reduces risk of heart disease"
    ]

    print("\nSemantic similarity classification:")
    print("-" * 40)

    for text in test_texts:
        result = compiled_classifier(text=text)

        print(f"\nText: {text}")
        print(f"Predicted category: {result.category}")

        # Show similarity scores
        similarities = [(ex, semantic_similarity(text, ex)) for ex in trainset]
        similarities.sort(key=lambda x: x[1], reverse=True)

        print("Top similar examples:")
        for i, (example, sim) in enumerate(similarities[:3]):
            print(f"  {i+1}. [{example.category}] {sim:.3f} - {example.text}")

# Example 3: Field-Weighted Similarity
def field_weighted_knn():
    """Demonstrate KNNFewShot with field-weighted similarity."""
    print("\n" + "=" * 60)
    print("Example 3: Field-Weighted Similarity")
    print("=" * 60)

    class MultiAttributeMatcher(dspy.Module):
        def __init__(self):
            super().__init__()
            self.match = dspy.Predict("query, similar_cases -> match")

        def forward(self, query):
            return self.match(query=query)

    # Training data with multiple fields
    trainset = [
        dspy.Example(
            query="Python developer job",
            skills="Python, Django, PostgreSQL",
            experience="3 years",
            location="Remote",
            match="backend_developer_v1"
        ),
        dspy.Example(
            query="Frontend React position",
            skills="React, TypeScript, CSS",
            experience="2 years",
            location="New York",
            match="frontend_developer_v1"
        ),
        dspy.Example(
            query="Senior data scientist role",
            skills="Python, ML, TensorFlow",
            experience="5 years",
            location="San Francisco",
            match="data_scientist_v1"
        ),
    ]

    def field_weighted_similarity(query, example):
        """Calculate similarity with field weights."""
        weights = {
            'skills': 0.5,       # Most important
            'experience': 0.3,   # Medium importance
            'location': 0.2      # Less important
        }

        total_similarity = 0
        total_weight = 0

        for field, weight in weights.items():
            if hasattr(query, field) and hasattr(example, field):
                # Simple word overlap for each field
                query_words = set(str(getattr(query, field)).lower().split())
                example_words = set(str(getattr(example, field)).lower().split())

                # Calculate overlap
                overlap = len(query_words & example_words)
                field_sim = overlap / max(len(query_words), 1)

                total_similarity += weight * field_sim
                total_weight += weight

        return total_similarity / total_weight if total_weight > 0 else 0

    # Create optimizer
    optimizer = KNNFewShot(
        k=2,
        similarity_fn=field_weighted_similarity
    )

    # Compile
    print("Compiling with field-weighted similarity...")
    matcher = MultiAttributeMatcher()
    compiled_matcher = optimizer.compile(matcher, trainset=trainset)

    # Test queries
    class TestQuery:
        def __init__(self, query, skills, experience, location):
            self.query = query
            self.skills = skills
            self.experience = experience
            self.location = location

    test_queries = [
        TestQuery(
            "Python backend position",
            "Python, Flask, MySQL",
            "4 years",
            "Remote"
        ),
        TestQuery(
            "React frontend job",
            "React, JavaScript, HTML",
            "3 years",
            "Boston"
        ),
    ]

    print("\nField-weighted matching results:")
    print("-" * 40)

    for test_q in test_queries:
        result = compiled_matcher(query=test_q.query)

        print(f"\nQuery: {test_q.query}")
        print(f"Skills: {test_q.skills}")
        print(f"Experience: {test_q.experience}")
        print(f"Location: {test_q.location}")
        print(f"Match found: {result.match}")

        # Show field-wise similarities
        print("\nField similarities:")
        for example in trainset:
            # Get field-wise scores
            query_skills = set(test_q.skills.lower().split(', '))
            ex_skills = set(example.skills.lower().split(', '))
            skills_sim = len(query_skills & ex_skills) / max(len(query_skills), 1)

            # For simplicity, use text similarity for other fields
            exp_sim = text_similarity(test_q.experience, example.experience)
            loc_sim = text_similarity(test_q.location, example.location)

            total_sim = field_weighted_similarity(test_q, example)

            print(f"  {example.match}:")
            print(f"    Skills: {skills_sim:.2f}, Exp: {exp_sim:.2f}, Loc: {loc_sim:.2f}")
            print(f"    Weighted Total: {total_sim:.2f}")

    def text_similarity(text1, text2):
        """Simple text similarity."""
        words1 = set(str(text1).lower().split())
        words2 = set(str(text2).lower().split())
        overlap = len(words1 & words2)
        return overlap / max(len(words1 | words2), 1)

# Example 4: Dynamic k Selection
def dynamic_k_optimization():
    """Demonstrate dynamic k selection based on content."""
    print("\n" + "=" * 60)
    print("Example 4: Dynamic k Selection")
    print("=" * 60)

    class AdaptiveRetriever(dspy.Module):
        def __init__(self):
            super().__init__()
            self.retrieve = dspy.Predict("query, retrieved_docs -> answer")

        def forward(self, query):
            return self.retrieve(query=query)

    # Document retrieval training data
    trainset = [
        dspy.Example(
            query="How to install Python on Windows",
            document="Download Python installer from python.org, run it, and add to PATH",
            category="installation"
        ),
        dspy.Example(
            query="Python list comprehension syntax",
            document="[expression for item in iterable if condition]",
            category="syntax"
        ),
        dspy.Example(
            query="Fix Python import error",
            document="Check if module is installed, verify file structure, or use sys.path",
            category="debugging"
        ),
        dspy.Example(
            query="Python virtual environment setup",
            document="Use 'python -m venv env' and 'source env/bin/activate'",
            category="installation"
        ),
    ]

    def dynamic_k_selection(query, train_examples, max_k=5):
        """Dynamically select k based on query complexity and content."""
        # Calculate all similarities first
        similarities = []
        for example in train_examples:
            sim = simple_text_similarity(query.query, example.query)
            similarities.append((example, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Determine k based on query characteristics
        query_text = str(query.query).lower()
        query_length = len(query_text.split())

        # Adjust k based on query complexity
        if query_length < 5:  # Very short query
            k = 5  # Use more examples for context
        elif query_length > 10:  # Long, specific query
            k = 2  # Fewer examples needed
        else:
            k = 3  # Default

        # Adjust based on similarity distribution
        if similarities[0][1] > 0.7:  # Very similar example found
            k = min(k, 2)  # Don't need many examples

        return k

    def simple_text_similarity(text1, text2):
        """Simple text similarity for demonstration."""
        words1 = set(str(text1).lower().split())
        words2 = set(str(text2).lower().split())
        overlap = len(words1 & words2)
        return overlap / max(len(words1 | words2), 1)

    # Create optimizer with dynamic k
    class DynamicKNNFewShot:
        def __init__(self, max_k=5):
            self.max_k = max_k

        def compile(self, program, trainset):
            # Store training examples
            self.trainset = trainset
            self.program = program
            return self

        def __call__(self, query):
            # Calculate dynamic k
            k = dynamic_k_selection(query, self.trainset, self.max_k)

            # Get k most similar examples
            similarities = []
            for example in self.trainset:
                sim = simple_text_similarity(query.query, example.query)
                similarities.append((example, sim))

            similarities.sort(key=lambda x: x[1], reverse=True)
            top_k = [ex[0] for ex in similarities[:k]]

            # Format examples for the model
            example_text = "\n".join([
                f"Q: {ex.query}\nA: {ex.document}"
                for ex in top_k
            ])

            # Call the program with examples
            return self.program(query=query.query, retrieved_docs=example_text)

    # Test with different query types
    retriever = AdaptiveRetriever()
    optimizer = DynamicKNNFewShot(max_k=5)
    compiled_retriever = optimizer.compile(retriever, trainset=trainset)

    test_queries = [
        "pip command",
        "list comprehension for filtering",
        "module not found python3",
        "django install virtual env"
    ]

    print("\nDynamic k retrieval results:")
    print("-" * 40)

    for query in test_queries:
        class TestQuery:
            def __init__(self, text):
                self.query = text

        test_q = TestQuery(query)
        result = compiled_retriever(test_q)

        # Calculate what k would be used
        k = dynamic_k_selection(test_q, trainset)

        print(f"\nQuery: {query}")
        print(f"Words: {len(query.split())}")
        print(f"Selected k: {k}")
        print(f"Answer: {result.answer}")

# Example 5: Performance Optimization
def performance_optimized_knn():
    """Demonstrate performance optimization techniques."""
    print("\n" + "=" * 60)
    print("Example 5: Performance Optimization")
    print("=" * 60)

    class FastClassifier(dspy.Module):
        def __init__(self):
            super().__init__()
            self.classify = dspy.Predict("text -> category")

        def forward(self, text):
            return self.classify(text=text)

    # Generate larger dataset
    categories = ["technology", "sports", "politics", "entertainment", "business"]
    trainset = []

    for i in range(100):  # 100 examples
        category = categories[i % len(categories)]
        trainset.append(dspy.Example(
            text=f"Example text about {category} number {i}",
            category=category
        ))

    # Optimized KNN with caching
    class CachedKNNFewShot:
        def __init__(self, k=5):
            self.k = k
            self.embedding_cache = {}
            self.vectorizer = TfidfVectorizer(max_features=1000)

        def compile(self, program, trainset):
            self.program = program
            self.trainset = trainset

            # Pre-compute embeddings for all training examples
            print("Pre-computing embeddings...")
            self.train_embeddings = self.vectorizer.fit_transform([
                str(ex.text) for ex in trainset
            ])

            return self

        def get_embedding(self, text):
            """Get cached embedding or compute new one."""
            if text not in self.embedding_cache:
                self.embedding_cache[text] = self.vectorizer.transform([text])
            return self.embedding_cache[text]

        def __call__(self, text):
            # Get query embedding (cached)
            query_embedding = self.get_embedding(str(text))

            # Calculate all similarities at once (vectorized)
            similarities = cosine_similarity(
                query_embedding,
                self.train_embeddings
            )[0]

            # Get top k indices
            top_k_indices = np.argsort(similarities)[-self.k:][::-1]

            # Get top k examples
            top_k_examples = [self.trainset[i] for i in top_k_indices]

            # For simplicity, return classification based on majority
            categories = [ex.category for ex in top_k_examples]
            most_common = max(set(categories), key=categories.count)

            return dspy.Prediction(category=most_common)

    # Performance comparison
    print("\nPerformance comparison:")
    print("-" * 40)

    # Test with naive approach
    def naive_knn(query, trainset, k=5):
        """Naive KNN without optimization."""
        similarities = []
        for example in trainset:
            # Simple similarity (not optimized)
            sim = len(str(query.text).split() & str(example.text).split())
            similarities.append((example, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    # Test queries
    test_texts = [
        dspy.Example(text="Technology news about AI", category="technology"),
        dspy.Example(text="Sports championship final", category="sports"),
        dspy.Example(text="Business earnings report", category="business"),
    ]

    # Time naive approach
    print("\nTesting naive KNN...")
    start_time = time.time()
    for test_text in test_texts[:2]:  # Test on subset
        naive_knn(test_text, trainset[:50])  # Use subset
    naive_time = time.time() - start_time

    # Time optimized approach
    print("\nTesting optimized KNN...")
    classifier = FastClassifier()
    optimizer = CachedKNNFewShot(k=5)
    compiled = optimizer.compile(classifier, trainset=trainset)

    start_time = time.time()
    for test_text in test_texts:
        result = compiled(text=test_text.text)
        print(f"  Query: {test_text.text[:30]}... -> {result.category}")
    optimized_time = time.time() - start_time

    print(f"\nTiming results:")
    print(f"Naive KNN (50 examples, 2 queries): {naive_time:.3f}s")
    print(f"Optimized KNN (100 examples, 3 queries): {optimized_time:.3f}s")
    print(f"Speedup: {(naive_time / optimized_time):.1f}x")

# Main execution
def run_all_examples():
    """Run all KNNFewShot examples."""
    print("DSPy KNNFewShot Optimization Examples")
    print("Demonstrating similarity-based example selection\n")

    try:
        basic_knn_optimization()
        semantic_knn_optimization()
        field_weighted_knn()
        dynamic_k_optimization()
        performance_optimized_knn()

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("All KNNFewShot examples completed!")
    print("=" * 60)

if __name__ == "__main__":
    run_all_examples()
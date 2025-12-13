# Chapter 6: Real-World Applications - Exercises

## Overview

These exercises focus on building practical DSPy applications for real-world scenarios. You'll work with RAG systems, classification pipelines, entity extraction, and agent-based systems.

**Difficulty**: Intermediate-Advanced
**Time Estimate**: 2-3 hours total
**Prerequisites**: Completion of Chapters 1-5

---

## Exercise 1: Build a Simple RAG System

**Objective**: Create a Retrieval-Augmented Generation system that answers questions based on a knowledge base.

**Requirements**:
- Use the `qa_pairs.json` dataset from assets/datasets/
- Implement document retrieval using simple text similarity
- Generate answers using DSPy's ChainOfThought module
- Evaluate on at least 3 test questions

**Starter Code**:
```python
import dspy
from assets.datasets import load_qa_pairs

class SimpleRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        # TODO: Initialize retriever and answer generator
        pass

    def forward(self, query):
        # TODO: Implement forward pass
        # 1. Retrieve relevant documents
        # 2. Augment query with context
        # 3. Generate answer
        pass

# Load data and test
data = load_qa_pairs()
rag = SimpleRAG()
# TODO: Test and evaluate
```

**Expected Output**: Answers to questions with supporting context

**Hints**:
- Use string similarity (e.g., cosine similarity on embeddings) for retrieval
- Keep context concise to avoid overwhelming the LM
- Evaluate with simple exact match or semantic similarity

**Advanced Challenge**: Implement BM25-based retrieval instead of embedding-based

---

## Exercise 2: Multi-Class Text Classification

**Objective**: Build a multi-class text classifier using DSPy for document categorization.

**Requirements**:
- Use `classification_data.csv` from assets/datasets/
- Create a classifier for at least 5 categories
- Achieve > 85% accuracy on test set
- Use optimization to improve performance

**Starter Code**:
```python
import dspy
from assets.datasets import load_classification_data

class DocumentClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        # TODO: Initialize classification signature
        pass

    def forward(self, text):
        # TODO: Predict document category
        pass

# Load and split data
data = load_classification_data()
train_set = data[:20]
test_set = data[20:]

# Create and optimize classifier
classifier = DocumentClassifier()

# TODO: Define metric and optimize
```

**Expected Output**: Predictions like "product_review", "customer_service", etc.

**Hints**:
- Include category descriptions in your signature
- Use BootstrapFewShot to find good in-context examples
- Consider the data distribution across classes

**Advanced Challenge**: Implement hierarchical classification (primary category → subcategory)

---

## Exercise 3: Named Entity Recognition Pipeline

**Objective**: Extract and classify entities from unstructured text.

**Requirements**:
- Use `entity_examples.json` from assets/datasets/
- Extract at least 8 entity types (PERSON, ORG, LOC, DATE, etc.)
- Return entity positions and confidence scores
- Handle overlapping entities appropriately

**Starter Code**:
```python
import dspy
from assets.datasets import load_entity_examples

class EntityExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        # TODO: Initialize entity extraction signature
        self.entity_types = [
            "PERSON", "ORG", "LOC", "DATE", "PRODUCT",
            "DURATION", "MEASUREMENT", "EVENT"
        ]

    def forward(self, text):
        # TODO: Extract entities with types and positions
        pass

# Test on examples
extractor = EntityExtractor()
examples = load_entity_examples()

# TODO: Evaluate extraction accuracy
```

**Expected Output**:
```json
{
  "text": "John Smith works at Google in Mountain View",
  "entities": [
    {"text": "John Smith", "type": "PERSON", "start": 0, "end": 10, "confidence": 0.95},
    {"text": "Google", "type": "ORG", "start": 20, "end": 26, "confidence": 0.98},
    {"text": "Mountain View", "type": "LOC", "start": 30, "end": 43, "confidence": 0.92}
  ]
}
```

**Hints**:
- Consider using structured output format (JSON) for entity representation
- Confidence scores should reflect LM certainty
- Test with entities that have special characters

**Advanced Challenge**: Implement entity linking (resolve entities to a knowledge base)

---

## Exercise 4: Multi-Hop Question Answering

**Objective**: Build a system that answers questions requiring multi-step reasoning over documents.

**Requirements**:
- Create questions requiring 2-3 reasoning hops
- Implement module composition for multi-hop reasoning
- Each hop should retrieve and reason over documents
- Track reasoning steps and intermediate answers

**Starter Code**:
```python
import dspy

class MultiHopQA(dspy.Module):
    def __init__(self):
        super().__init__()
        # TODO: Initialize multi-hop reasoning modules
        pass

    def forward(self, query, documents):
        # Step 1: Identify what information is needed
        # Step 2: Search for initial answer
        # Step 3: Identify follow-up questions
        # Step 4: Search for follow-up answers
        # Step 5: Synthesize final answer
        pass

# Test with multi-hop questions
questions = [
    "What is the CEO of the company that was acquired?",
    "Where did the person who founded the research group go to school?",
    # TODO: Add more multi-hop questions
]

# TODO: Evaluate reasoning quality
```

**Expected Output**: Final answer with reasoning trace showing each hop

**Hints**:
- Create intermediate reasoning steps using ChainOfThought
- Each hop should identify "what information do I need next?"
- Use the trace to debug reasoning failures

**Advanced Challenge**: Implement dynamic hop count (system decides how many hops needed)

---

## Exercise 5: Intelligent Search and Ranking

**Objective**: Build a search system that retrieves AND ranks documents by relevance.

**Requirements**:
- Implement two-stage retrieval (retrieve candidate docs, then rank)
- Use DSPy for semantic ranking, not just keyword matching
- Return top-K results with relevance scores
- Evaluate Mean Reciprocal Rank (MRR) and Normalized DCG

**Starter Code**:
```python
import dspy
from typing import List

class SmartSearch(dspy.Module):
    def __init__(self):
        super().__init__()
        # TODO: Initialize retriever and ranker
        pass

    def forward(self, query: str, documents: List[str]) -> List[dict]:
        # Step 1: Retrieve candidate documents
        candidates = self.retrieve(query, documents)

        # Step 2: Rerank candidates using semantic similarity
        ranked = self.rerank(query, candidates)

        # Return top-K with scores
        return ranked[:10]

# TODO: Evaluate on test queries
```

**Expected Output**:
```python
[
    {"rank": 1, "document": "...", "score": 0.95},
    {"rank": 2, "document": "...", "score": 0.87},
    # ...
]
```

**Hints**:
- Use embedding-based similarity for initial retrieval
- Use DSPy modules for semantic reranking
- Consider query expansion to improve recall

**Advanced Challenge**: Implement listwise ranking (jointly rank all candidates using a neural model)

---

## Exercise 6: Information Extraction from Documents

**Objective**: Extract structured information from unstructured documents (contracts, reports, emails).

**Requirements**:
- Use `domain_specific/legal_contracts.json` or business documents
- Extract key information (parties, dates, amounts, terms)
- Output as structured data (JSON/dict)
- Handle different document formats

**Starter Code**:
```python
import dspy
import json
from typing import Dict, Any

class ContractExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        # TODO: Initialize extraction modules for different fields
        pass

    def forward(self, document_text: str) -> Dict[str, Any]:
        # TODO: Extract structured information
        # Return: {
        #     "parties": [...],
        #     "date": "...",
        #     "amount": "...",
        #     "terms": {...},
        #     "key_clauses": [...]
        # }
        pass

# Test on sample contracts
contracts = load_contracts()

for contract in contracts[:3]:
    extracted = extractor(contract['text'])
    print(json.dumps(extracted, indent=2))
```

**Expected Output**: Structured JSON with all extracted fields

**Hints**:
- Define clear field definitions in your signatures
- Use JSON output formatting for structured results
- Validate extracted data types

**Advanced Challenge**: Implement confidence scores per field and reasoning chains

---

## Answer Key

Solutions are available in the `solutions/` directory:

- `exercise1_rag.py` - RAG implementation
- `exercise2_classification.py` - Text classifier
- `exercise3_ner.py` - Entity extractor
- `exercise4_multihop.py` - Multi-hop QA
- `exercise5_ranking.py` - Smart search
- `exercise6_extraction.py` - Contract extractor

## Review Checklist

Before submitting your solutions:

- [ ] Code runs without errors
- [ ] All requirements met
- [ ] Test cases pass
- [ ] Output format matches specifications
- [ ] Evaluation metrics reported
- [ ] Code is well-commented
- [ ] Advanced challenge attempted (optional)

## Additional Resources

- [Chapter 6: Real-World Applications](../src/06-real-world-applications/)
- [Chapter 9: Glossary](../src/09-appendices/04-glossary.md)
- [Sample Datasets](../assets/datasets/README.md)
- [DSPy GitHub Examples](https://github.com/stanfordnlp/dspy/tree/main/examples)

---

**Difficulty Level**: Intermediate-Advanced
**Estimated Time**: 2-3 hours
**Next Steps**: Move to Chapter 7 (Advanced Topics)

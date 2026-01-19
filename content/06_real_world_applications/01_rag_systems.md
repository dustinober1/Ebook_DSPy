# RAG Systems: Building Intelligent Document Q&A

## Introduction

Retrieval-Augmented Generation (RAG) is one of the most powerful and widely used applications of language models today. RAG systems combine the strengths of information retrieval with language generation to answer questions based on large collections of documents. DSPy provides excellent support for building sophisticated RAG systems that can handle real-world complexity.

## What is RAG?

### The RAG Architecture

RAG systems work in two main phases:

1. **Retrieval Phase**: Find relevant documents or passages from a knowledge base
2. **Generation Phase**: Generate answers using the retrieved context

```
Question → Retrieve Documents → Generate Answer → Response
```

### Why RAG Matters

- **Current Information**: Can answer questions about recent events
- **Verifiable**: Sources are cited and can be checked
- **Customizable**: Works with your specific documents
- **Cost-Effective**: No need to retrain models for new domains
- **Scalable**: Can handle millions of documents

## Building a Basic RAG System

### Core Components

```python
import dspy
from dspy.retrieve import ColBERTv2Retriever

class BasicRAG(dspy.Module):
    def __init__(self, num_passages=5):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(
            context=context,
            answer=prediction.answer,
            reasoning=prediction.rationale
        )
```

### Setting Up Document Collection

```python
# Sample document collection
documents = [
    "Python is a high-level programming language created by Guido van Rossum.",
    "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
    "Natural Language Processing (NLP) deals with interactions between computers and human language.",
    "Deep learning uses neural networks with multiple layers to learn from data.",
    "DSPy is a framework for programming language models."
]

# Create retriever (in practice, you'd use a proper vector database)
retriever = ColBERTv2Retriever(
    k=5,
    collection=documents
)
dspy.settings.configure(retriever=retriever)
```

### Using the Basic RAG System

```python
# Initialize and use the RAG system
rag = BasicRAG()

# Ask a question
question = "What is Python?"
result = rag(question=question)

print(f"Question: {question}")
print(f"Answer: {result.answer}")
print(f"Sources: {result.context}")
```

## Advanced RAG Techniques

### Multi-stage Retrieval

For complex queries, use multiple retrieval strategies:

```python
class AdvancedRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        # Initial broad retrieval
        self.initial_retrieve = dspy.Retrieve(k=20)
        # Rerank for precision
        self.rerank = dspy.Predict("query, documents -> ranked_documents")
        # Generate final answer
        self.generate = dspy.ChainOfThought("question, context -> answer")

    def forward(self, question):
        # Get initial candidates
        initial_docs = self.initial_retrieve(question).passages

        # Rerank based on question
        reranked = self.rerank(
            query=question,
            documents="\n".join(initial_docs)
        )

        # Use top documents for context
        context = reranked.ranked_documents.split("\n")[:5]

        # Generate answer
        prediction = self.generate(question=question, context="\n".join(context))

        return dspy.Prediction(
            answer=prediction.answer,
            context=context,
            reasoning=prediction.rationale
        )
```

### Query Understanding and Expansion

Improve retrieval by understanding and expanding queries:

```python
class SmartQueryRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        # Understand and expand the query
        self.query_processor = dspy.ChainOfThought("question -> keywords, expanded_query")
        # Retrieve with multiple queries
        self.retrieve = dspy.Retrieve(k=5)
        # Synthesize results
        self.synthesize = dspy.ChainOfThought("question, contexts -> answer")

    def forward(self, question):
        # Process the query
        processed = self.query_processor(question=question)

        # Retrieve with original and expanded query
        original_results = self.retrieve(question=question).passages
        expanded_results = self.retrieve(question=processed.expanded_query).passages

        # Combine and deduplicate results
        all_contexts = list(set(original_results + expanded_results))

        # Generate answer from combined context
        prediction = self.synthesize(
            question=question,
            contexts="\n\n".join(all_contexts[:8])
        )

        return dspy.Prediction(
            answer=prediction.answer,
            keywords=processed.keywords,
            expanded_query=processed.expanded_query,
            context=all_contexts[:8],
            reasoning=prediction.rationale
        )
```

### Conversational RAG

Handle follow-up questions and maintain context:

```python
class ConversationalRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        # Maintain conversation history
        self.context_manager = dspy.Predict(
            "conversation_history, new_question -> contextualized_question"
        )
        # RAG for contextualized question
        self.rag = dspy.ChainOfThought("context, question -> answer")
        # Track conversation
        self.conversation_history = []

    def forward(self, question):
        # Add question to history
        self.conversation_history.append({"user": question})

        # Contextualize based on history
        contextualized = self.context_manager(
            conversation_history=str(self.conversation_history),
            new_question=question
        )

        # Retrieve and generate answer
        context = dspy.Retrieve(k=5)(contextualized.contextualized_question).passages
        prediction = self.rag(
            context=context,
            question=contextualized.contextualized_question
        )

        # Add response to history
        self.conversation_history.append({"assistant": prediction.answer})

        return dspy.Prediction(
            answer=prediction.answer,
            context=context,
            contextualized_question=contextualized.contextualized_question
        )
```

## Optimizing RAG Systems

### Using BootstrapFewShot for RAG

```python
class OptimizedRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=5)
        self.generate_answer = dspy.Predict("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate_answer(context=context, question=question)

# Training data for optimization
trainset = [
    dspy.Example(
        question="What are the benefits of Python?",
        context="Python is known for its simplicity, readability, and large ecosystem.",
        answer="Python offers simplicity, readability, and has a large ecosystem of libraries."
    ),
    dspy.Example(
        question="How does machine learning work?",
        context="Machine learning uses algorithms to find patterns in data and make predictions.",
        answer="Machine learning works by using algorithms to identify patterns in data and make predictions based on those patterns."
    ),
    # ... more examples
]

# Define evaluation metric
def rag_metric(example, pred, trace=None):
    # Check if answer is grounded in context
    context_words = set(example.context.lower().split())
    answer_words = set(pred.answer.lower().split())
    overlap = len(context_words & answer_words) / max(len(answer_words), 1)

    # Check relevance (simplified)
    relevance = any(word in pred.answer.lower() for word in example.question.lower().split())

    return overlap * relevance

# Optimize with BootstrapFewShot
optimizer = BootstrapFewShot(metric=rag_metric, max_bootstrapped_demos=4)
optimized_rag = optimizer.compile(OptimizedRAG(), trainset=trainset)
```

### MIPRO for Complex RAG

For sophisticated RAG tasks, MIPRO can optimize both instructions and examples:

```python
class ComplexRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=8)
        self.analyze_question = dspy.Predict("question -> question_type, key_entities")
        self.generate_answer = dspy.ChainOfThought(
            "question, context, question_type, key_entities -> answer"
        )

    def forward(self, question):
        # Analyze the question
        analysis = self.analyze_question(question=question)

        # Retrieve relevant context
        context = self.retrieve(question).passages

        # Generate targeted answer
        prediction = self.generate_answer(
            question=question,
            context=context,
            question_type=analysis.question_type,
            key_entities=analysis.key_entities
        )

        return dspy.Prediction(
            answer=prediction.answer,
            question_type=analysis.question_type,
            key_entities=analysis.key_entities,
            context=context
        )

# Optimize with MIPRO for complex reasoning
mipro_optimizer = MIPRO(
    metric=rag_metric,
    num_candidates=10,
    init_temperature=0.7
)

optimized_complex_rag = mipro_optimizer.compile(ComplexRAG(), trainset=trainset)
```

## Real-World RAG Applications

### Document Q&A System

```python
class DocumentQASystem(dspy.Module):
    def __init__(self, document_collection):
        super().__init__()
        self.collection = document_collection
        self.retrieve = dspy.Retrieve(k=5)
        self.extract_answer = dspy.Predict(
            "context, question -> answer, confidence, evidence"
        )
        self.cite_sources = dspy.Predict("answer, context -> citations")

    def forward(self, question):
        # Retrieve relevant documents
        retrieved = self.retrieve(question=question)
        context = retrieved.passages

        # Extract answer with confidence
        extraction = self.extract_answer(context=context, question=question)

        # Generate citations
        citations = self.cite_sources(
            answer=extraction.answer,
            context=context
        )

        return dspy.Prediction(
            answer=extraction.answer,
            confidence=extraction.confidence,
            evidence=extraction.evidence,
            citations=citations.citations,
            sources=retrieved
        )

# Example usage
doc_qa = DocumentQASystem(my_document_collection)
result = doc_qa("What are the main challenges in implementing RAG?")

print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence}")
print(f"Sources: {result.citations}")
```

### Legal Document Analysis

```python
class LegalRAG(dspy.Module):
    def __init__(self, legal_documents):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=10)
        self.legal_analysis = dspy.ChainOfThought(
            "question, legal_context -> analysis, relevant_laws, precedents"
        )
        self.risk_assessment = dspy.Predict(
            "analysis, relevant_laws -> risk_level, recommendations"
        )

    def forward(self, legal_question):
        # Retrieve relevant legal documents
        legal_context = self.retrieve(question=legal_question).passages

        # Perform legal analysis
        analysis = self.legal_analysis(
            question=legal_question,
            legal_context=legal_context
        )

        # Assess risks
        assessment = self.risk_assessment(
            analysis=analysis.analysis,
            relevant_laws=analysis.relevant_laws
        )

        return dspy.Prediction(
            legal_analysis=analysis.analysis,
            relevant_laws=analysis.relevant_laws,
            precedents=analysis.precedents,
            risk_level=assessment.risk_level,
            recommendations=assessment.recommendations
        )
```

### Healthcare Information System

```python
class MedicalRAG(dspy.Module):
    def __init__(self, medical_documents):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=8)
        self.extract_symptoms = dspy.Predict("query -> symptoms, conditions")
        self.medical_analysis = dspy.ChainOfThought(
            "symptoms, medical_context -> possible_conditions, confidence"
        )
        self.disclaimer = "This is not medical advice. Please consult a healthcare professional."

    def forward(self, patient_query):
        # Extract symptoms and conditions
        extracted = self.extract_symptoms(query=patient_query)

        # Retrieve medical information
        medical_context = self.retrieve(
            question=f"{extracted.symptoms} {extracted.conditions}"
        ).passages

        # Analyze with medical context
        analysis = self.medical_analysis(
            symptoms=extracted.symptoms,
            medical_context=medical_context
        )

        return dspy.Prediction(
            symptoms=extracted.symptoms,
            possible_conditions=analysis.possible_conditions,
            confidence=analysis.confidence,
            disclaimer=self.disclaimer,
            reasoning=analysis.rationale
        )
```

## Best Practices for RAG Systems

### 1. Document Preprocessing

```python
def preprocess_documents(documents):
    """Clean and prepare documents for retrieval."""
    processed = []
    for doc in documents:
        # Remove irrelevant sections
        cleaned = remove_boilerplate(doc)
        # Split into manageable chunks
        chunks = chunk_document(cleaned, max_length=500)
        # Add metadata
        for i, chunk in enumerate(chunks):
            processed.append({
                "text": chunk,
                "source": doc.get("source", "unknown"),
                "chunk_id": i,
                "timestamp": doc.get("timestamp")
            })
    return processed
```

### 2. Hybrid Retrieval

```python
class HybridRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.keyword_search = dspy.Retrieve(k=5, search_type="keyword")
        self.semantic_search = dspy.Retrieve(k=5, search_type="semantic")
        self.merge_results = dspy.Predict("results1, results2 -> merged_results")

    def forward(self, question):
        # Get keyword results
        keyword_docs = self.keyword_search(question).passages

        # Get semantic results
        semantic_docs = self.semantic_search(question).passages

        # Merge and rank
        merged = self.merge_results(
            results1=keyword_docs,
            results2=semantic_docs
        )

        return dspy.Prediction(context=merged.merged_results)
```

### 3. Evaluation Metrics

```python
def evaluate_rag(rag_system, testset):
    """Comprehensive RAG evaluation."""
    results = []

    for example in testset:
        prediction = rag_system(question=example.question)

        # Answer quality
        answer_quality = evaluate_answer_quality(
            prediction.answer,
            example.expected_answer
        )

        # Retrieval quality
        retrieval_quality = evaluate_retrieval_quality(
            prediction.context,
            example.relevant_docs
        )

        # Grounding (is answer in context?)
        grounding_score = check_grounding(
            prediction.answer,
            prediction.context
        )

        results.append({
            "answer_quality": answer_quality,
            "retrieval_quality": retrieval_quality,
            "grounding": grounding_score,
            "overall": (answer_quality + retrieval_quality + grounding_score) / 3
        })

    return results
```

## Common Challenges and Solutions

### Challenge 1: Hallucination

**Problem**: The model generates answers not supported by the retrieved documents.

**Solution**:
```python
class FactCheckingRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=5)
        self.generate = dspy.Predict("context, question -> answer")
        self.verify = dspy.Predict("answer, context -> verification, corrections")

    def forward(self, question):
        context = self.retrieve(question).passages
        answer = self.generate(context=context, question=question)
        verification = self.verify(answer=answer.answer, context=context)

        if verification.verification == "needs_correction":
            answer.answer = verification.corrections

        return answer
```

### Challenge 2: Outdated Information

**Problem**: Information in the knowledge base becomes outdated.

**Solution**:
```python
class TimelyRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.check_recency = dspy.Predict("query -> needs_current_info")
        self.search_web = dspy.Predict("query -> current_info")
        self.rag = dspy.ChainOfThought("context, question, current_info -> answer")

    def forward(self, question):
        recency_check = self.check_recency(query=question)

        if recency_check.needs_current_info:
            current = self.search_web(query=question)
            current_info = current.current_info
        else:
            current_info = "N/A"

        context = self.retrieve(question).passages
        prediction = self.rag(
            context=context,
            question=question,
            current_info=current_info
        )

        return prediction
```

### Challenge 3: Query Understanding

**Problem**: Ambiguous or poorly formed queries lead to poor retrieval.

**Solution**:
```python
class QueryClarificationRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.clarify = dspy.Predict("question -> clarified_question, assumptions")
        self.rag = dspy.Predict("context, clarified_question -> answer")

    def forward(self, question):
        clarification = self.clarify(question=question)
        context = self.retrieve(question=clarification.clarified_question).passages
        answer = self.rag(
            context=context,
            clarified_question=clarification.clarified_question
        )

        return dspy.Prediction(
            answer=answer.answer,
            assumptions=clarification.assumptions,
            clarified_question=clarification.clarified_question
        )
```

## Key Takeaways

1. **RAG combines retrieval and generation** for knowledge-intensive tasks
2. **Context quality is crucial**—better retrieval leads to better answers
3. **Optimization significantly improves** RAG performance
4. **Real-world RAG systems** handle complexity through multiple stages
5. **Evaluation must be comprehensive**—check retrieval, generation, and grounding
6. **Common challenges** include hallucination, outdated info, and query ambiguity

## Next Steps

In the next section, we'll explore **Multi-hop Search**, which extends RAG concepts to handle complex queries that require reasoning across multiple documents and information sources.
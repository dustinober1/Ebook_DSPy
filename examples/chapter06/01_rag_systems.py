"""
RAG Systems Implementation Examples

This file demonstrates complete implementations of Retrieval-Augmented Generation systems,
including basic RAG, advanced techniques, and optimization strategies.

Examples include:
- Basic RAG system
- Query-aware RAG with expansion
- Conversational RAG with memory
- Domain-specific RAG applications
- RAG system optimization
"""

import dspy
from dspy.teleprompter import BootstrapFewShot, MIPRO
from typing import List, Dict, Any, Optional
import time

# Configure language model (placeholder)
dspy.settings.configure(lm=dspy.LM(model="gpt-3.5-turbo", api_key="your-key"))

# Example 1: Basic RAG System
class BasicRAG(dspy.Module):
    """Basic RAG system for document Q&A."""

    def __init__(self, num_passages=5):
        super().__init__()
        self.num_passages = num_passages
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        # Retrieve relevant passages
        retrieved = self.retrieve(question=question)
        context = retrieved.passages

        # Generate answer using retrieved context
        prediction = self.generate_answer(context="\n".join(context), question=question)

        return dspy.Prediction(
            question=question,
            answer=prediction.answer,
            context=context,
            reasoning=prediction.rationale
        )

def demo_basic_rag():
    """Demonstrate basic RAG system."""
    print("=" * 60)
    print("Example 1: Basic RAG System")
    print("=" * 60)

    # Create RAG system
    rag = BasicRAG(num_passages=3)

    # Sample questions
    questions = [
        "What are the main benefits of machine learning?",
        "How does natural language processing work?",
        "What is deep learning?"
    ]

    print("\nProcessing questions:")
    for question in questions:
        result = rag(question=question)
        print(f"\nQ: {question}")
        print(f"A: {result.answer}")
        print(f"Sources: {len(result.context)} passages retrieved")

# Example 2: Query-Aware RAG System
class QueryAwareRAG(dspy.Module):
    """RAG system that understands and expands queries."""

    def __init__(self):
        super().__init__()
        self.understand_query = dspy.ChainOfThought("question -> query_type, key_concepts")
        self.expand_query = dspy.Predict("query, key_concepts -> expanded_queries")
        self.retrieve = dspy.Retrieve(k=3)
        self.synthesize = dspy.ChainOfThought("question, contexts -> final_answer")

    def forward(self, question):
        # Understand the query
        understanding = self.understand_query(question=question)

        # Generate expanded queries
        expansion = self.expand_query(
            query=question,
            key_concepts=understanding.key_concepts
        )

        # Retrieve with multiple queries
        all_contexts = []

        # Original query
        original_results = self.retrieve(question=question)
        all_contexts.extend(original_results.passages)

        # Expanded queries
        for expanded in expansion.expanded_queries.split(";"):
            expanded = expanded.strip()
            if expanded and expanded != question:
                expanded_results = self.retrieve(question=expanded)
                all_contexts.extend(expanded_results.passages)

        # Remove duplicates
        unique_contexts = list(dict.fromkeys(all_contexts))

        # Synthesize final answer
        synthesis = self.synthesize(
            question=question,
            contexts="\n\n".join(unique_contexts[:5])
        )

        return dspy.Prediction(
            question=question,
            answer=synthesis.final_answer,
            query_type=understanding.query_type,
            key_concepts=understanding.key_concepts,
            contexts_used=unique_contexts[:5],
            reasoning=synthesis.rationale
        )

def demo_query_aware_rag():
    """Demonstrate query-aware RAG system."""
    print("\n" + "=" * 60)
    print("Example 2: Query-Aware RAG System")
    print("=" * 60)

    rag = QueryAwareRAG()

    # Complex query that benefits from expansion
    question = "How do AI and machine learning impact healthcare and medicine?"
    result = rag(question=question)

    print(f"\nQuestion: {result.question}")
    print(f"Query Type: {result.query_type}")
    print(f"Key Concepts: {result.key_concepts}")
    print(f"\nAnswer: {result.answer}")
    print(f"\nContexts Used: {len(result.contexts_used)}")

# Example 3: Conversational RAG System
class ConversationalRAG(dspy.Module):
    """RAG system that maintains conversation context."""

    def __init__(self):
        super().__init__()
        self.context_manager = dspy.Predict(
            "conversation_history, new_question -> contextualized_question"
        )
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.ChainOfThought("context, question -> answer")
        self.conversation_history = []

    def forward(self, question):
        # Add to conversation history
        self.conversation_history.append({"type": "user", "text": question})

        # Contextualize based on history
        history_text = "\n".join([
            f"{item['type']}: {item['text']}"
            for item in self.conversation_history[-3:]  # Last 3 exchanges
        ])

        contextualized = self.context_manager(
            conversation_history=history_text,
            new_question=question
        )

        # Retrieve with contextualized question
        retrieved = self.retrieve(question=contextualized.contextualized_question)
        context = retrieved.passages

        # Generate answer
        prediction = self.generate(context="\n".join(context), question=question)

        # Add response to history
        self.conversation_history.append({
            "type": "assistant",
            "text": prediction.answer
        })

        return dspy.Prediction(
            answer=prediction.answer,
            contextualized_question=contextualized.contextualized_question,
            conversation_turn=len(self.conversation_history) // 2
        )

def demo_conversational_rag():
    """Demonstrate conversational RAG system."""
    print("\n" + "=" * 60)
    print("Example 3: Conversational RAG System")
    print("=" * 60)

    rag = ConversationalRAG()

    # Simulate a conversation
    conversation = [
        "What is quantum computing?",
        "How does it differ from classical computing?",
        "What are its practical applications?"
    ]

    print("\nSimulated Conversation:")
    for i, question in enumerate(conversation, 1):
        print(f"\nTurn {i}")
        print(f"User: {question}")
        result = rag(question=question)
        print(f"Assistant: {result.answer}")
        print(f"(Contextualized: {result.contextualized_question})")

# Example 4: Domain-Specific RAG for Customer Support
class CustomerSupportRAG(dspy.Module):
    """RAG system specialized for customer support."""

    def __init__(self):
        super().__init__()
        self.classify_intent = dspy.Predict("question -> intent_category, urgency")
        self.retrieve = dspy.Retrieve(k=4)
        self.generate_response = dspy.ChainOfThought(
            "intent, context, urgency -> response, action_needed, empathy_level"
        )
        self.suggest_solutions = dspy.Predict("response, intent -> solutions, next_steps")

    def forward(self, question, customer_tier="standard"):
        # Classify intent and urgency
        classification = self.classify_intent(question=question)

        # Retrieve relevant support documents
        retrieved = self.retrieve(
            query=f"{classification.intent_category} {question}"
        )

        # Generate empathetic response
        response_gen = self.generate_response(
            intent=classification.intent_category,
            context="\n".join(retrieved.passages),
            urgency=classification.urgency
        )

        # Suggest specific solutions
        solutions = self.suggest_solutions(
            response=response_gen.response,
            intent=classification.intent_category
        )

        return dspy.Prediction(
            response=response_gen.response,
            intent=classification.intent_category,
            urgency=classification.urgency,
            action_needed=response_gen.action_needed,
            empathy_level=response_gen.empathy_level,
            solutions=solutions.solutions,
            next_steps=solutions.next_steps
        )

def demo_customer_support_rag():
    """Demonstrate customer support RAG system."""
    print("\n" + "=" * 60)
    print("Example 4: Customer Support RAG System")
    print("=" * 60)

    support_rag = CustomerSupportRAG()

    # Customer support queries
    support_queries = [
        "My order hasn't arrived yet and it's been 2 weeks!",
        "How do I cancel my subscription?",
        "The app keeps crashing on my iPhone"
    ]

    for query in support_queries:
        print(f"\nCustomer: {query}")
        result = support_rag(query=query)
        print(f"\nIntent: {result.intent} (Urgency: {result.urgency})")
        print(f"Response: {result.response}")
        print(f"Empathy Level: {result.empathy_level}")
        print(f"Solutions: {result.solutions}")
        print(f"Next Steps: {result.next_steps}")

# Example 5: RAG System Optimization
def optimize_rag_system():
    """Demonstrate RAG system optimization with BootstrapFewShot."""
    print("\n" + "=" * 60)
    print("Example 5: RAG System Optimization")
    print("=" * 60)

    # Training data for optimization
    trainset = [
        dspy.Example(
            question="What are the benefits of cloud computing?",
            context="Cloud computing offers scalability, cost savings, and accessibility.",
            answer="Cloud computing provides scalability, cost-effectiveness, and remote access to resources."
        ),
        dspy.Example(
            question="How does encryption protect data?",
            context="Encryption scrambles data using algorithms and keys.",
            answer="Encryption protects data by scrambling it using mathematical algorithms and secret keys."
        ),
        # ... more training examples
    ]

    # Define evaluation metric
    def rag_metric(example, pred, trace=None):
        """Custom metric for RAG evaluation."""
        # Check if answer is grounded in context
        if hasattr(pred, 'context') and pred.context:
            context_words = set(" ".join(pred.context).lower().split())
            answer_words = set(pred.answer.lower().split())
            overlap = len(context_words & answer_words)
            grounding = overlap / max(len(answer_words), 1)
        else:
            grounding = 0

        # Check relevance
        question_words = set(example.question.lower().split())
        answer_relevance = len(question_words & answer_words) / max(len(question_words), 1)

        return 0.6 * grounding + 0.4 * answer_relevance

    # Create base RAG system
    class OptimizedRAG(dspy.Module):
        def __init__(self):
            super().__init__()
            self.retrieve = dspy.Retrieve(k=3)
            self.generate = dspy.ChainOfThought("context, question -> answer")

        def forward(self, question):
            retrieved = self.retrieve(question=question)
            prediction = self.generate(
                context="\n".join(retrieved.passages),
                question=question
            )
            return dspy.Prediction(
                answer=prediction.answer,
                context=retrieved.passages
            )

    # Optimize with BootstrapFewShot
    optimizer = BootstrapFewShot(
        metric=rag_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=2
    )

    print("\nOptimizing RAG system...")
    start_time = time.time()
    optimized_rag = optimizer.compile(OptimizedRAG(), trainset=trainset)
    optimization_time = time.time() - start_time

    print(f"Optimization completed in {optimization_time:.2f} seconds")

    # Test optimized system
    test_questions = [
        "What makes machine learning models accurate?",
        "How do APIs enable software integration?"
    ]

    print("\nTesting optimized system:")
    for question in test_questions:
        result = optimized_rag(question=question)
        score = rag_metric(
            dspy.Example(question=question),
            dspy.Prediction(answer=result.answer, context=result.context)
        )
        print(f"\nQ: {question}")
        print(f"A: {result.answer}")
        print(f"Quality Score: {score:.2f}")

# Example 6: RAG System with Caching
class CachedRAG(dspy.Module):
    """RAG system with caching for efficiency."""

    def __init__(self):
        super().__init__()
        self.cache = {}
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        # Check cache first
        if question in self.cache:
            print(f"Cache hit for: {question}")
            return self.cache[question]

        # Generate response
        retrieved = self.retrieve(question=question)
        prediction = self.generate(
            context="\n".join(retrieved.passages),
            question=question
        )

        result = dspy.Prediction(
            answer=prediction.answer,
            context=retrieved.passages
        )

        # Cache the result
        self.cache[question] = result

        return result

def demo_cached_rag():
    """Demonstrate cached RAG system."""
    print("\n" + "=" * 60)
    print("Example 6: Cached RAG System")
    print("=" * 60)

    cached_rag = CachedRAG()

    # Ask the same question twice
    question = "What are the principles of good software design?"

    print("\nFirst query (cache miss):")
    result1 = cached_rag(question=question)
    print(f"Answer: {result1.answer}")

    print("\nSecond query (cache hit):")
    result2 = cached_rag(question=question)
    print(f"Answer: {result2.answer}")

    print(f"\nCache size: {len(cached_rag.cache)}")

# Main execution
def run_all_examples():
    """Run all RAG system examples."""
    print("DSPy RAG Systems Examples")
    print("Demonstrating various RAG implementations and techniques\n")

    try:
        demo_basic_rag()
        demo_query_aware_rag()
        demo_conversational_rag()
        demo_customer_support_rag()
        optimize_rag_system()
        demo_cached_rag()

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("All RAG system examples completed!")
    print("=" * 60)

if __name__ == "__main__":
    run_all_examples()
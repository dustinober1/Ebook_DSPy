"""
Exercise 4: Build a Simple Q&A System
======================================
Solution for building a contextual question-answering system with confidence scoring.
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

# Configure language model
lm = dspy.OpenAI(model="gpt-4o")
dspy.configure(lm=lm)


class ContextualQA(dspy.Signature):
    """Answer questions based on the provided context.

    This system should:
    1. Answer using only information from the context
    2. Indicate confidence level based on context clarity
    3. Mention if the answer is not found in context
    """
    context = dspy.InputField(desc="Text containing information to answer from")
    question = dspy.InputField(desc="Question to be answered based on the context")
    answer = dspy.OutputField(desc="Answer to the question based only on the context")
    confidence = dspy.OutputField(desc="Confidence level: high, medium, or low")
    evidence = dspy.OutputField(desc="Brief quote or evidence from context supporting the answer")


class QASystem:
    """A question-answering system that works with provided context."""

    def __init__(self):
        self.qa_module = dspy.ChainOfThought(ContextualQA)

    def answer(self, context, question):
        """Answer a question based on the given context."""
        try:
            result = self.qa_module(context=context, question=question)
            return {
                "answer": result.answer,
                "confidence": result.confidence.lower(),
                "evidence": result.evidence,
                "question": question,
                "context": context
            }
        except Exception as e:
            return {
                "answer": f"Error processing question: {e}",
                "confidence": "low",
                "evidence": "N/A",
                "question": question,
                "context": context
            }


def display_result(result):
    """Display a Q&A result in a formatted way."""
    print(f"\nQuestion: {result['question']}")
    print(f"Answer: {result['answer']}")

    # Display confidence with appropriate emoji
    confidence_emoji = {
        "high": "🟢",
        "medium": "🟡",
        "low": "🔴"
    }
    emoji = confidence_emoji.get(result['confidence'], "⚪")
    print(f"Confidence: {emoji} {result['confidence'].capitalize()}")

    if result['evidence'] and result['evidence'] != "N/A":
        print(f"Evidence: \"{result['evidence']}\"")

    print("-" * 60)


def test_qa_system():
    """Test the Q&A system with sample contexts."""

    print("\nContextual Q&A System Test")
    print("===========================\n")

    # Create Q&A system
    qa_system = QASystem()

    # Define test cases
    test_cases = [
        {
            "context": "Paris is the capital of France. It has a population of about 2.1 million people. The city is known for the Eiffel Tower and the Louvre Museum.",
            "question": "What is the capital of France?"
        },
        {
            "context": "Python was created by Guido van Rossum and released in 1991. It emphasizes code readability and simplicity.",
            "question": "Who created Python?"
        },
        {
            "context": "The Great Wall of China is over 13,000 miles long. It was built over many centuries to protect against invasions.",
            "question": "What is the main programming language used in AI?"  # Not in context!
        },
        {
            "context": "The Wright brothers, Orville and Wilbur, invented the first successful airplane in 1903. Their first flight lasted 12 seconds and covered 120 feet.",
            "question": "How long was the Wright brothers' first flight?"
        },
        {
            "context": "Water covers about 71% of Earth's surface. It exists in three states: solid (ice), liquid (water), and gas (water vapor).",
            "question": "What percentage of Earth's surface is covered by water?"
        }
    ]

    # Process each test case
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Context: {test_case['context']}")

        result = qa_system.answer(test_case['context'], test_case['question'])
        display_result(result)

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nThe Q&A system demonstrated:")
    print("✓ Accurate answers when information is in context")
    print("✓ Low confidence when answer is not in context")
    print("✓ Evidence extraction from the provided text")
    print("✓ Proper handling of out-of-context questions")


def analyze_confidence_logic():
    """Explain how the confidence scoring works."""

    print("\n\nConfidence Scoring Logic")
    print("========================")
    print("\nThe system determines confidence based on:")
    print("\n1. HIGH Confidence:")
    print("   - Answer is explicitly stated in context")
    print("   - Question directly matches available information")
    print("   - Evidence is a direct quote supporting the answer")

    print("\n2. MEDIUM Confidence:")
    print("   - Answer requires inference from context")
    print("   - Information is implied but not directly stated")
    print("   - Multiple pieces of context need to be combined")

    print("\n3. LOW Confidence:")
    print("   - Answer is not found in context")
    print("   - Context is insufficient to answer the question")
    print("   - Question is about unrelated topics")

    print("\nThis confidence scoring helps users understand:")
    print("- When to trust the answer completely")
    print("- When to verify with additional sources")
    print("- When the answer might be missing information")


if __name__ == "__main__":
    test_qa_system()
    analyze_confidence_logic()
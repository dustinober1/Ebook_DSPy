"""
Exercise 3: ReAct Agent with Tools
Solution for Exercise 3 from Chapter 3

Task: Create a ReAct agent that can answer questions using web search
- Implement mock web search tool
- Create ReAct agent with reasoning
- Handle multi-step queries
"""

import dspy
import json
from typing import List, Dict, Any, Optional

# Mock web search tool for demonstration
class MockWebSearchTool:
    """Mock web search tool that simulates search results."""

    def __init__(self):
        # Mock knowledge base
        self.knowledge_base = {
            "python programming": {
                "answer": "Python is a high-level programming language created by Guido van Rossum, first released in 1991. It's known for its simple syntax and extensive libraries.",
                "source": "Official Python Documentation"
            },
            "machine learning": {
                "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
                "source": "Machine Learning Textbook"
            },
            "climate change": {
                "answer": "Climate change refers to significant changes in global temperature and weather patterns. While climate variations are natural, human activities have been the main driver since the mid-20th century.",
                "source": "IPCC Reports"
            },
            "renewable energy": {
                "answer": "Renewable energy is energy from sources that are naturally replenishing but flow-limited. Examples include solar, wind, geothermal, and hydropower.",
                "source": "U.S. Department of Energy"
            },
            "quantum computing": {
                "answer": "Quantum computing is a type of computation that uses quantum phenomena like superposition and entanglement to perform operations on data.",
                "source": "Quantum Computing Institute"
            }
        }

    def search(self, query: str) -> Dict[str, Any]:
        """Simulate web search and return results."""

        # Check if we have relevant information
        query_lower = query.lower()
        for key, info in self.knowledge_base.items():
            if key in query_lower or any(word in query_lower for word in key.split()):
                return {
                    "query": query,
                    "results": [
                        {
                            "title": f"Information about {key}",
                            "snippet": info["answer"][:200] + "...",
                            "url": f"https://example.com/{key.replace(' ', '-')}",
                            "source": info["source"]
                        }
                    ],
                    "total_results": 1,
                    "found": True
                }

        # Return no results if no match
        return {
            "query": query,
            "results": [],
            "total_results": 0,
            "found": False
        }

class QuestionAnsweringAgent:
    """ReAct agent for answering questions using web search."""

    def __init__(self):
        # Initialize tools
        self.search_tool = MockWebSearchTool()

        # Define the ReAct signature
        class QuestionAnsweringSignature(dspy.Signature):
            """Answer questions by reasoning and using web search when needed."""
            question = dspy.InputField(desc="User's question", type=str)
            thought = dspy.OutputField(desc="Current thinking process", type=str)
            action = dspy.OutputField(desc="Action to take (search/answer)", type=str)
            search_query = dspy.OutputField(desc="Search query if needed", type=str)
            observation = dspy.OutputField(desc("Result from search", type=str)
            answer = dspy.OutputField(desc("Final answer", type=str)

        # Create few-shot examples
        qa_examples = [
            dspy.Example(
                question="What is Python programming?",
                thought="The user is asking about Python programming. I should search for information about Python.",
                action="search",
                search_query="Python programming language",
                observation="Python is a high-level programming language created by Guido van Rossum...",
                answer="Python is a high-level programming language created by Guido van Rossum, first released in 1991."
            ),
            dspy.Example(
                question="How does photosynthesis work?",
                thought="This is about biology. I should search for information about photosynthesis.",
                action="search",
                search_query="photosynthesis process plants",
                observation="Photosynthesis is the process by which plants convert sunlight into energy...",
                answer="Photosynthesis is the process where plants use sunlight, water, and carbon dioxide to create glucose and oxygen."
            )
        ]

        # Initialize the ReAct module
        self.agent = dspy.React(
            signature=QuestionAnsweringSignature,
            tools={"search": self.search_tool.search},
            demos=qa_examples
        )

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question using the ReAct agent."""

        try:
            # Simulate the ReAct process
            result = self._simulate_react_process(question)

            return {
                'question': question,
                'answer': result['answer'],
                'thought_process': result['thought_process'],
                'sources_used': result.get('sources', []),
                'success': True
            }

        except Exception as e:
            return {
                'question': question,
                'error': str(e),
                'success': False
            }

    def _simulate_react_process(self, question: str) -> Dict[str, Any]:
        """Simulate the ReAct (Reasoning and Acting) process."""

        # Initial thought
        thought = f"User asks: {question}. Let me analyze what information I need."

        # Determine if search is needed
        question_lower = question.lower()
        needs_search = not any(keyword in question_lower for keyword in ["what is", "who are you", "how are you"])

        sources = []
        answer = ""

        if needs_search:
            # Action: Search
            action = "search"
            search_query = question[:50]  # Use first 50 chars as query

            # Observation: Get search results
            search_results = self.search_tool.search(search_query)

            if search_results['found'] and search_results['results']:
                observation = f"Found {search_results['total_results']} relevant result(s)."
                top_result = search_results['results'][0]
                observation += f" Top result: {top_result['snippet']}"
                sources.append(top_result['source'])

                # Final answer based on search
                answer = search_results['results'][0]['snippet']
            else:
                observation = "No relevant information found in search."
                answer = "I couldn't find specific information about that in my search."
        else:
            # Direct answer without search
            action = "answer"
            observation = "Answering based on general knowledge."
            answer = self._generate_direct_answer(question)

        thought += f" I will {action} for this question."

        return {
            'answer': answer,
            'thought_process': thought,
            'action': action if needs_search else 'direct_answer',
            'sources': sources
        }

    def _generate_direct_answer(self, question: str) -> str:
        """Generate a direct answer without search."""

        if "who are you" in question.lower():
            return "I am an AI assistant designed to help answer your questions using web search capabilities."
        elif "how are you" in question.lower():
            return "I'm functioning well and ready to help you with your questions!"
        else:
            return "That's an interesting question. Let me think about the best way to answer it."

def test_question_agent():
    """Test the question answering agent."""

    agent = QuestionAnsweringAgent()

    # Test questions
    test_questions = [
        "What is machine learning?",
        "How does renewable energy work?",
        "Tell me about quantum computing",
        "Who are you?",
        "What causes climate change?",
        "Explain Python programming"
    ]

    print("=" * 60)
    print("Question Answering Agent Test")
    print("=" * 60)

    results = []

    for i, question in enumerate(test_questions, 1):
        print(f"\nQuestion {i}: {question}")

        result = agent.answer_question(question)
        results.append(result)

        if result['success']:
            print(f"Answer: {result['answer']}")
            if result['sources_used']:
                print(f"Sources: {', '.join(result['sources_used'])}")
            print(f"(Thought process: {result['thought_process'][:100]}...)")
        else:
            print(f"Error: {result['error']}")

        print("-" * 40)

    # Summary
    successful = sum(1 for r in results if r['success'])
    print(f"\nSummary:")
    print(f"Total questions: {len(results)}")
    print(f"Successfully answered: {successful}")
    print(f"Success rate: {successful/len(results)*100:.1f}%")

def create_interactive_qa():
    """Create an interactive question-answering session."""

    agent = QuestionAnsweringAgent()

    print("\n" + "=" * 60)
    print("Interactive Question Answering")
    print("Type 'quit' to exit")
    print("=" * 60)

    while True:
        question = input("\nAsk a question: ").strip()

        if question.lower() == 'quit':
            print("Goodbye!")
            break

        if not question:
            print("Please enter a valid question.")
            continue

        result = agent.answer_question(question)

        if result['success']:
            print(f"\nAnswer: {result['answer']}")
            if result['sources_used']:
                print(f"\nSources: {', '.join(result['sources_used'])}")
        else:
            print(f"\nSorry, I couldn't answer that question: {result['error']}")

def main():
    """Main function to run Exercise 3."""

    print("\n" + "=" * 60)
    print("Exercise 3: ReAct Agent with Tools")
    print("Creating a question-answering agent with web search capability")
    print("=" * 60)

    # Run the test
    test_question_agent()

    # Option to run interactive session
    response = input("\nWould you like to ask your own questions? (y/n): ")
    if response.lower().startswith('y'):
        create_interactive_qa()

if __name__ == "__main__":
    main()
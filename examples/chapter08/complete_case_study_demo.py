"""
Chapter 8: Complete Case Studies Demonstration

This file demonstrates all four case studies from Chapter 8 working together
in an integrated AI platform.
"""

import os
import json
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

import dspy
from dspy import Module, Predict, Signature, InputField, OutputField

# Mock implementations for demonstration
# In a real implementation, these would be the actual classes from the case studies


# ============ CASE STUDY 1: ENTERPRISE RAG SYSTEM ============

class MockRAGSystem:
    """Mock implementation of Enterprise RAG System."""

    def __init__(self):
        self.documents = [
            "DSPy is a framework for programming with language models.",
            "It provides tools for building RAG systems, chatbots, and more.",
            "DSPy uses signatures to define input/output specifications.",
            "Modules are the building blocks of DSPy programs.",
            "Optimizers improve model performance through training."
        ]

    def query(self, user_id: str, question: str, **kwargs) -> Dict:
        """Query the RAG system."""
        # Simple keyword matching for demo
        relevant_docs = [
            doc for doc in self.documents
            if any(word.lower() in doc.lower() for word in question.lower().split())
        ]

        return {
            "answer": f"Based on {len(relevant_docs)} documents: {relevant_docs[0] if relevant_docs else 'No relevant information found.'}",
            "sources": [{"id": i, "title": f"Doc {i}"} for i in range(len(relevant_docs))],
            "confidence": min(len(relevant_docs) / 3.0, 1.0)
        }


# ============ CASE STUDY 2: CUSTOMER SUPPORT CHATBOT ============

class MockChatbot:
    """Mock implementation of Customer Support Chatbot."""

    def __init__(self):
        self.responses = {
            "greeting": "Hello! How can I help you today?",
            "order": "I can help you with your order. Please provide your order number.",
            "technical": "Let me connect you with our technical support team.",
            "complaint": "I'm sorry to hear that. Let me help resolve your issue."
        }

    def process_message(self, session_id: str, message: str, **kwargs) -> Dict:
        """Process a chat message."""
        # Simple intent detection
        if any(word in message.lower() for word in ["hello", "hi", "help"]):
            intent = "greeting"
        elif "order" in message.lower():
            intent = "order"
        elif any(word in message.lower() for word in ["broken", "error", "issue"]):
            intent = "complaint"
        elif "technical" in message.lower():
            intent = "technical"
        else:
            intent = "general"

        return {
            "response": self.responses.get(intent, "I understand. How else can I help?"),
            "intent": intent,
            "confidence": 0.85,
            "metadata": {"session_id": session_id}
        }


# ============ CASE STUDY 3: AI CODE ASSISTANT ============

class MockCodeAssistant:
    """Mock implementation of AI Code Assistant."""

    def __init__(self):
        self.templates = {
            "python": {
                "function": "def {name}({params}):\n    '''{description}'''\n    # TODO: Implement\n    pass"
            }
        }

    def process_code_request(self, request: Dict) -> Dict:
        """Process a code assistance request."""
        req_type = request.get("type", "generate")

        if req_type == "generate":
            prompt = request.get("prompt", "").lower()
            if "function" in prompt:
                return {
                    "code": "def calculate_sum(a, b):\n    '''Calculate sum of two numbers'''\n    return a + b",
                    "language": "python",
                    "explanation": "This function adds two numbers and returns the result."
                }
            else:
                return {
                    "code": "# Generated code based on your request",
                    "language": "python",
                    "explanation": "This code addresses your programming needs."
                }

        return {"response": "Code processing complete"}


# ============ CASE STUDY 4: DATA ANALYSIS PIPELINE ============

class MockDataPipeline:
    """Mock implementation of Automated Data Analysis Pipeline."""

    def __init__(self):
        self.insights = [
            "Sales increased by 15% compared to last quarter",
            "Customer satisfaction scores improved by 10%",
            "Product returns decreased by 5%"
        ]

    def run_pipeline(self, trigger: Dict) -> Dict:
        """Run the data analysis pipeline."""
        if trigger["type"] == "query":
            query = trigger.get("query", "").lower()
            if "sales" in query:
                insights = [self.insights[0]]
            elif "satisfaction" in query:
                insights = [self.insights[1]]
            elif "returns" in query:
                insights = [self.insights[2]]
            else:
                insights = self.insights
        else:
            insights = self.insights

        return {
            "insights": insights,
            "statistics": {"mean": 100, "std": 15, "count": 1000},
            "recommendations": ["Continue current strategy", "Focus on customer retention"]
        }


# ============ INTEGRATED AI PLATFORM ============

@dataclass
class PlatformRequest:
    """Unified request format for the integrated platform."""
    user_id: str
    session_id: str
    query: str
    request_type: str = "auto"
    context: Optional[Dict] = None
    timestamp: Optional[datetime] = None

class IntegratedAIPlatform:
    """Integrated platform combining all four case studies."""

    def __init__(self):
        # Initialize all components
        self.rag_system = MockRAGSystem()
        self.chatbot = MockChatbot()
        self.code_assistant = MockCodeAssistant()
        self.data_pipeline = MockDataPipeline()

        # Analytics tracking
        self.analytics = {
            "total_requests": 0,
            "requests_by_type": {},
            "response_times": [],
            "user_sessions": {}
        }

    def process_request(self, request: PlatformRequest) -> Dict:
        """Process a unified platform request."""
        start_time = time.time()

        # Update analytics
        self.analytics["total_requests"] += 1
        if request.user_id not in self.analytics["user_sessions"]:
            self.analytics["user_sessions"][request.user_id] = {
                "sessions": set(),
                "requests": 0
            }
        self.analytics["user_sessions"][request.user_id]["sessions"].add(request.session_id)
        self.analytics["user_sessions"][request.user_id]["requests"] += 1

        # Determine request type if auto
        if request.request_type == "auto":
            request_type = self._classify_request(request.query)
        else:
            request_type = request.request_type

        # Route to appropriate component
        result = self._route_request(request, request_type)

        # Calculate response time
        response_time = time.time() - start_time
        self.analytics["response_times"].append(response_time)
        self.analytics["requests_by_type"][request_type] = \
            self.analytics["requests_by_type"].get(request_type, 0) + 1

        # Add metadata
        result["metadata"] = {
            "request_type": request_type,
            "response_time": response_time,
            "timestamp": datetime.now().isoformat(),
            "platform": "Integrated AI Platform"
        }

        return result

    def _classify_request(self, query: str) -> str:
        """Classify the request type."""
        query_lower = query.lower()

        # Check for code-related keywords
        if any(word in query_lower for word in ["code", "function", "programming", "python"]):
            return "code"

        # Check for data analysis keywords
        elif any(word in query_lower for word in ["data", "analyze", "report", "insights"]):
            return "data"

        # Check for knowledge/QA keywords
        elif any(word in query_lower for word in ["what", "how", "explain", "tell me"]):
            return "rag"

        # Default to chatbot
        else:
            return "chat"

    def _route_request(self, request: PlatformRequest, request_type: str) -> Dict:
        """Route request to appropriate component."""
        if request_type == "rag":
            return self.rag_system.query(
                user_id=request.user_id,
                question=request.query
            )

        elif request_type == "chat":
            return self.chatbot.process_message(
                session_id=request.session_id,
                message=request.query
            )

        elif request_type == "code":
            return self.code_assistant.process_code_request({
                "type": "generate",
                "prompt": request.query
            })

        elif request_type == "data":
            return self.data_pipeline.run_pipeline({
                "type": "query",
                "query": request.query
            })

        else:
            return {"error": f"Unknown request type: {request_type}"}

    def get_analytics(self) -> Dict:
        """Get platform analytics."""
        avg_response_time = (
            sum(self.analytics["response_times"]) / len(self.analytics["response_times"])
            if self.analytics["response_times"] else 0
        )

        return {
            "total_requests": self.analytics["total_requests"],
            "unique_users": len(self.analytics["user_sessions"]),
            "avg_response_time": avg_response_time,
            "requests_by_type": self.analytics["requests_by_type"],
            "top_users": sorted(
                [(uid, data["requests"]) for uid, data in self.analytics["user_sessions"].items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }


# ============ INTERACTIVE DEMONSTRATION ============

def interactive_demo():
    """Interactive demonstration of the integrated platform."""
    print("\n" + "="*80)
    print("INTEGRATED AI PLATFORM DEMONSTRATION")
    print("="*80)
    print("\nThis demo combines all four case studies from Chapter 8:")
    print("1. Enterprise RAG System")
    print("2. Customer Support Chatbot")
    print("3. AI-Powered Code Assistant")
    print("4. Automated Data Analysis Pipeline")
    print("\nType 'quit' to exit the demo.\n")

    # Initialize platform
    platform = IntegratedAIPlatform()

    # Demo scenarios
    scenarios = [
        {
            "description": "Knowledge Query (RAG)",
            "query": "What is DSPy?",
            "expected_type": "rag"
        },
        {
            "description": "Customer Support (Chatbot)",
            "query": "Hi, I need help with my order",
            "expected_type": "chat"
        },
        {
            "description": "Code Generation (Code Assistant)",
            "query": "Write a Python function to calculate the factorial",
            "expected_type": "code"
        },
        {
            "description": "Data Analysis (Data Pipeline)",
            "query": "Show me sales data analysis",
            "expected_type": "data"
        }
    ]

    # Run demo scenarios
    print("\nRunning demo scenarios...\n")
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*60}")
        print(f"Scenario {i}: {scenario['description']}")
        print(f"{'='*60}")
        print(f"Query: {scenario['query']}")

        # Create request
        request = PlatformRequest(
            user_id=f"demo_user_{i}",
            session_id=f"session_{i}",
            query=scenario["query"]
        )

        # Process request
        result = platform.process_request(request)

        # Display result
        print(f"\nDetected Type: {result['metadata']['request_type']}")
        print(f"Response Time: {result['metadata']['response_time']:.2f}s")

        if result['metadata']['request_type'] == 'rag':
            print(f"\nAnswer: {result['answer']}")
            print(f"Sources: {len(result['sources'])} documents")
        elif result['metadata']['request_type'] == 'chat':
            print(f"\nChatbot Response: {result['response']}")
            print(f"Intent: {result['intent']}")
        elif result['metadata']['request_type'] == 'code':
            print(f"\nGenerated Code:\n{result['code']}")
            print(f"\nExplanation: {result['explanation']}")
        elif result['metadata']['request_type'] == 'data':
            print(f"\nInsights:")
            for insight in result['insights']:
                print(f"• {insight}")
            print(f"\nRecommendations:")
            for rec in result['recommendations']:
                print(f"• {rec}")

    # Show analytics
    print(f"\n{'='*60}")
    print("PLATFORM ANALYTICS")
    print(f"{'='*60}")

    analytics = platform.get_analytics()
    print(f"\nTotal Requests: {analytics['total_requests']}")
    print(f"Unique Users: {analytics['unique_users']}")
    print(f"Average Response Time: {analytics['avg_response_time']:.2f}s")
    print(f"\nRequests by Type:")
    for req_type, count in analytics['requests_by_type'].items():
        print(f"• {req_type}: {count}")

    # Interactive mode
    print(f"\n{'='*60}")
    print("INTERACTIVE MODE")
    print(f"{'='*60}")
    print("\nEnter your queries or 'quit' to exit:")

    user_id = "interactive_user"
    session_id = f"session_{int(time.time())}"

    while True:
        try:
            query = input("\nQuery: ").strip()
            if query.lower() == 'quit':
                break

            if not query:
                continue

            # Create and process request
            request = PlatformRequest(
                user_id=user_id,
                session_id=session_id,
                query=query
            )

            result = platform.process_request(request)

            # Display result
            print(f"\n[{result['metadata']['request_type'].upper()}] {result['metadata']['response_time']:.2f}s")

            if result['metadata']['request_type'] == 'rag':
                print(f"Answer: {result['answer']}")
            elif result['metadata']['request_type'] == 'chat':
                print(f"Response: {result['response']}")
            elif result['metadata']['request_type'] == 'code':
                print(f"Code:\n{result['code']}")
            elif result['metadata']['request_type'] == 'data':
                print(f"Insights: {', '.join(result['insights'][:2])}")

        except KeyboardInterrupt:
            print("\n\nExiting demo...")
            break
        except Exception as e:
            print(f"\nError: {e}")

    print("\n\nDemo complete! 🎉")
    print("\nThis demonstrates how DSPy can be used to build complex,")
    print("integrated AI systems that solve real-world problems.")


# ============ PERFORMANCE BENCHMARKS ============

def run_performance_benchmarks():
    """Run performance benchmarks for the integrated platform."""
    print("\n" + "="*80)
    print("PERFORMANCE BENCHMARKS")
    print("="*80)

    platform = IntegratedAIPlatform()

    # Test queries
    test_queries = [
        ("What is machine learning?", "rag"),
        ("Hello, I need help", "chat"),
        ("Create a sorting algorithm", "code"),
        ("Analyze customer trends", "data")
    ] * 10  # 10 rounds

    print(f"\nRunning {len(test_queries)} test queries...\n")

    results = []
    for i, (query, expected_type) in enumerate(test_queries):
        request = PlatformRequest(
            user_id=f"bench_user_{i}",
            session_id=f"bench_session",
            query=query
        )

        start_time = time.time()
        result = platform.process_request(request)
        end_time = time.time()

        results.append({
            "query": query,
            "expected": expected_type,
            "actual": result["metadata"]["request_type"],
            "response_time": end_time - start_time,
            "correct": result["metadata"]["request_type"] == expected_type
        })

    # Calculate metrics
    total_time = sum(r["response_time"] for r in results)
    avg_time = total_time / len(results)
    accuracy = sum(1 for r in results if r["correct"]) / len(results) * 100

    print(f"\nBenchmark Results:")
    print(f"Total queries: {len(results)}")
    print(f"Average response time: {avg_time:.3f}s")
    print(f"Classification accuracy: {accuracy:.1f}%")
    print(f"Fastest query: {min(r['response_time'] for r in results):.3f}s")
    print(f"Slowest query: {max(r['response_time'] for r in results):.3f}s")

    # Classification breakdown
    print(f"\nClassification Breakdown:")
    for req_type in ["rag", "chat", "code", "data"]:
        count = sum(1 for r in results if r["actual"] == req_type)
        print(f"• {req_type}: {count} queries")


# ============ MAIN EXECUTION ============

def main():
    """Main function to run the demonstration."""
    print("\n" + "="*80)
    print("CHAPTER 8: COMPLETE CASE STUDIES DEMONSTRATION")
    print("="*80)
    print("\nThis demo showcases all four case studies from Chapter 8:")
    print("\n1. Enterprise RAG System - Knowledge retrieval and Q&A")
    print("2. Customer Support Chatbot - Conversational AI")
    print("3. AI-Powered Code Assistant - Code generation and help")
    print("4. Automated Data Analysis Pipeline - Data insights and reports")
    print("\nAll components are integrated into a unified platform!")

    # Show menu
    while True:
        print("\n" + "-"*80)
        print("Select an option:")
        print("1. Interactive Demo (Recommended)")
        print("2. Performance Benchmarks")
        print("3. Component-by-Component Demo")
        print("4. Exit")
        print("-"*80)

        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == "1":
            interactive_demo()
        elif choice == "2":
            run_performance_benchmarks()
        elif choice == "3":
            component_demo()
        elif choice == "4":
            print("\nThank you for exploring DSPy case studies!")
            break
        else:
            print("\nInvalid choice. Please try again.")


def component_demo():
    """Demonstrate each component individually."""
    print("\n" + "="*80)
    print("COMPONENT-BY-COMPONENT DEMO")
    print("="*80)

    # Initialize all components
    rag = MockRAGSystem()
    chatbot = MockChatbot()
    code_ass = MockCodeAssistant()
    data_pipe = MockDataPipeline()

    print("\n1. RAG System Demo:")
    rag_result = rag.query("user_1", "What is DSPy?")
    print(f"   Query: What is DSPy?")
    print(f"   Answer: {rag_result['answer'][:100]}...")

    print("\n2. Chatbot Demo:")
    chat_result = chatbot.process_message("session_1", "Hello, I need help")
    print(f"   Message: Hello, I need help")
    print(f"   Response: {chat_result['response']}")
    print(f"   Intent: {chat_result['intent']}")

    print("\n3. Code Assistant Demo:")
    code_result = code_ass.process_code_request({
        "type": "generate",
        "prompt": "Create a function"
    })
    print(f"   Prompt: Create a function")
    print(f"   Code:\n   {code_result['code']}")

    print("\n4. Data Pipeline Demo:")
    data_result = data_pipe.run_pipeline({
        "type": "query",
        "query": "sales report"
    })
    print(f"   Query: sales report")
    print(f"   Insights: {data_result['insights'][0]}")


if __name__ == "__main__":
    main()
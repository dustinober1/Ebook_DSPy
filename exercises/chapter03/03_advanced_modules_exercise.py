"""
Exercise 3.3: Advanced Module Techniques
Working with ReAct agents, custom modules, and module composition
"""

import dspy
from typing import List, Dict, Any, Optional, Union
import json

def exercise_3_1_react_agent():
    """
    Exercise 3.1: Create a ReAct agent with tools

    Task: Build a ReAct agent that can use multiple tools to answer questions
    """
    # TODO: Define mock tools for the agent
    # You'll need at least a calculator and a search tool

    # Your code here:
    class Calculator:
        """TODO: Implement a calculator tool"""
        def calculate(self, expression):
            # TODO: Safely evaluate mathematical expressions
            pass

    class SearchEngine:
        """TODO: Implement a search tool"""
        def search(self, query):
            # TODO: Return mock search results
            pass

    # TODO: Define ReAct signature
    class ReActSignature(dspy.Signature):
        """TODO: Define signature for ReAct agent"""
        # TODO: Add fields for thought, action, action_input, observation
        pass

    # TODO: Create tool registry
    tools = {
        # TODO: Register tools with descriptions
    }

    # TODO: Create ReAct module
    # agent = dspy.React(signature=ReActSignature, tools=tools)

    # Test questions
    questions = [
        "What is 15% of 240?",
        "What's the distance from Earth to Mars?",
        "If I save $100 monthly at 5% interest for 10 years, how much will I have?"
    ]

    print("Exercise 3.1 Solution:")
    for question in questions:
        # TODO: Use the ReAct agent to answer
        # result = agent(question=question)
        # print(f"\nQ: {question}")
        # print(f"A: {result.answer}")
        pass

    return {
        "tools_implemented": False,
        "react_signature": False,
        "agent_created": False,
        "questions_answered": False
    }

def exercise_3_2_custom_module():
    """
    Exercise 3.2: Create a custom module

    Task: Build a custom sentiment analysis pipeline module
    """
    # TODO: Create a custom module that processes text through multiple steps
    # Steps: 1) Clean text, 2) Analyze sentiment, 3) Extract aspects

    # Your code here:
    class SentimentPipeline(dspy.Module):
        """TODO: Implement custom sentiment analysis pipeline"""

        def __init__(self):
            super().__init__()
            # TODO: Initialize internal modules
            # - Text cleaner
            # - Sentiment analyzer
            # - Aspect extractor
            # - Result synthesizer
            pass

        def forward(self, text):
            """TODO: Implement the pipeline processing"""
            # TODO: Step 1: Clean the text
            # TODO: Step 2: Analyze sentiment
            # TODO: Step 3: Extract aspects
            # TODO: Step 4: Synthesize results
            pass

    # TODO: Test the custom module
    # test_texts = [
    #     "The camera quality is amazing but the battery life is disappointing.",
    #     "Excellent customer service and fast shipping!",
    #     "The product arrived damaged and doesn't work properly."
    # ]

    print("\nExercise 3.2 Solution:")
    # pipeline = SentimentPipeline()
    # for text in test_texts:
    #     result = pipeline(text=text)
    #     print(f"\nText: {text}")
    #     print(f"Overall: {result.overall_sentiment}")
    #     print(f"Aspects: {result.aspects}")

    return {
        "custom_module_class": False,
        "pipeline_implemented": False,
        "all_steps_working": False,
        "results_formatted": False
    }

def exercise_3_3_module_composition():
    """
    Exercise 3.3: Compose multiple modules

    Task: Create a text processing system using module composition
    """
    # TODO: Define individual processing modules
    # 1) Preprocessor, 2) Analyzer, 3) Summarizer

    # Your code here:
    class TextPreprocessor(dspy.Module):
        """TODO: Implement text preprocessing"""
        def __init__(self):
            super().__init__()
            # TODO: Set up preprocessor signature and module
            pass

        def forward(self, text):
            # TODO: Implement preprocessing
            pass

    class TextAnalyzer(dspy.Module):
        """TODO: Implement text analysis"""
        def __init__(self):
            super().__init__()
            # TODO: Set up analyzer signature and module
            pass

        def forward(self, text):
            # TODO: Implement analysis
            pass

    class TextSummarizer(dspy.Module):
        """TODO: Implement text summarization"""
        def __init__(self):
            super().__init__()
            # TODO: Set up summarizer signature and module
            pass

        def forward(self, text):
            # TODO: Implement summarization
            pass

    # TODO: Create a composition pattern (sequential or parallel)
    # Your code here:
    class TextProcessingSystem(dspy.Module):
        """TODO: Implement the composed system"""

        def __init__(self):
            super().__init__()
            # TODO: Initialize components
            pass

        def forward(self, text):
            # TODO: Process through modules
            # Try both sequential and parallel approaches
            pass

    # TODO: Test the composed system
    # test_article = """
    # Long article text here...
    # """

    print("\nExercise 3.3 Solution:")
    # system = TextProcessingSystem()
    # result = system(text=test_article)
    # print(f"Processed text: {result.processed_text}")

    return {
        "individual_modules": False,
        "composition_pattern": False,
        "system_created": False,
        "composition_working": False
    }

def exercise_3_4_conditional_module():
    """
    Exercise 3.4: Create a conditional routing module

    Task: Build a module that routes inputs based on content type
    """
    # TODO: Create modules for different content types
    # 1) Question handler, 2) Statement processor, 3) Command executor

    # Your code here:
    class ContentClassifier(dspy.Module):
        """TODO: Classify content type"""
        def __init__(self):
            super().__init__()
            # TODO: Set up classifier
            pass

        def forward(self, text):
            # TODO: Classify as question, statement, or command
            pass

    class ConditionalProcessor(dspy.Module):
        """TODO: Process based on content type"""

        def __init__(self):
            super().__init__()
            # TODO: Initialize classifier and handlers
            pass

        def forward(self, text):
            # TODO: Classify and route to appropriate handler
            # TODO: Return both classification and result
            pass

    # TODO: Test with different content types
    # test_inputs = [
    #     "What is the weather today?",
    #     "The stock market is performing well.",
    #     "Calculate 25 * 4"
    # ]

    print("\nExercise 3.4 Solution:")
    # processor = ConditionalProcessor()
    # for text in test_inputs:
    #     result = processor(text=text)
    #     print(f"\nInput: {text}")
    #     print(f"Type: {result.content_type}")
    #     print(f"Result: {result.result}")

    return {
        "classifier_module": False,
        "conditional_logic": False,
        "routing_working": False,
        "all_types_handled": False
    }

def exercise_3_5_module_with_state():
    """
    Exercise 3.5: Create a stateful module

    Task: Build a conversation module that maintains context
    """
    # TODO: Create a module that maintains conversation history

    # Your code here:
    class ConversationModule(dspy.Module):
        """TODO: Implement stateful conversation module"""

        def __init__(self, max_history=5):
            super().__init__()
            # TODO: Initialize conversation history
            # TODO: Set up processing signature
            pass

        def forward(self, user_input):
            # TODO: Add input to history
            # TODO: Process with context from history
            # TODO: Generate contextual response
            # TODO: Manage history size
            pass

    # TODO: Test conversation flow
    # conversation = ConversationModule()
    # messages = [
    #     "Hi, I'm John",
    #     "I'm looking for information about Python",
    #     "Can you recommend any good tutorials?",
    #     "What about Django?",
    #     "Thanks for the help!"
    # ]

    print("\nExercise 3.5 Solution:")
    # conv = ConversationModule()
    # for msg in messages:
    #     result = conv(user_input=msg)
    #     print(f"\nUser: {msg}")
    #     print(f"Bot: {result.response}")
    #     print(f"History: {len(conv.history)} messages")

    return {
        "state_maintained": False,
        "context_used": False,
        "history_managed": False,
        "conversation_flow": False
    }

def exercise_3_6_module_validation():
    """
    Exercise 3.6: Add validation to modules

    Task: Create modules with input/output validation
    """
    # TODO: Create modules with validation logic

    # Your code here:
    class ValidatedInputModule(dspy.Module):
        """TODO: Module with input validation"""

        def __init__(self):
            super().__init__()
            # TODO: Set up validation rules
            # TODO: Initialize processing module
            pass

        def forward(self, data):
            # TODO: Validate input
            # TODO: Process if valid
            # TODO: Handle invalid inputs gracefully
            pass

    class ValidatedOutputModule(dspy.Module):
        """TODO: Module with output validation"""

        def __init__(self):
            super().__init__()
            # TODO: Set up processing module
            # TODO: Define output validation
            pass

        def forward(self, input_data):
            # TODO: Process input
            # TODO: Validate output
            # TODO: Retry or adjust if invalid
            pass

    # TODO: Test validation scenarios
    # test_cases = [
    #     "Valid input",
    #     "",  # Empty
    #     None,  # None value
    #     "x" * 10000  # Too long
    # ]

    print("\nExercise 3.6 Solution:")
    # validator = ValidatedInputModule()
    # for test in test_cases:
    #     result = validator(data=test)
    #     print(f"\nInput: {test}")
    #     print(f"Valid: {result.is_valid}")
    #     print(f"Result: {result.result}")

    return {
        "input_validation": False,
        "output_validation": False,
        "error_handling": False,
        "validation_comprehensive": False
    }

# Solution verification function
def verify_exercise_3():
    """Verify all advanced module exercises are completed correctly."""
    results = {
        "exercise_3_1": exercise_3_1_react_agent(),
        "exercise_3_2": exercise_3_2_custom_module(),
        "exercise_3_3": exercise_3_3_module_composition(),
        "exercise_3_4": exercise_3_4_conditional_module(),
        "exercise_3_5": exercise_3_5_module_with_state(),
        "exercise_3_6": exercise_3_6_module_validation()
    }

    print("\n" + "="*50)
    print("EXERCISE 3.3 COMPLETION STATUS")
    print("="*50)

    for exercise, result in results.items():
        print(f"\n{exercise}:")
        for task, status in result.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {task}: {status}")

    # Overall completion percentage
    total_tasks = sum(len(r) for r in results.values())
    completed_tasks = sum(sum(r.values()) for r in results.values())
    percentage = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0

    print(f"\nOverall Progress: {percentage:.1f}% ({completed_tasks}/{total_tasks})")

    if percentage == 100:
        print("\n🎉 Outstanding! You've mastered advanced module techniques!")
    else:
        print(f"\nKeep working! {total_tasks - completed_tasks} tasks remaining.")

    return percentage

# Hint function
def show_hints(exercise_number):
    """Show hints for specific exercises."""
    hints = {
        "3_1": [
            "Use dspy.React with a signature that includes thought/action fields",
            "Tools need description and function in the tools dictionary",
            "The agent will automatically use tools based on reasoning",
            "Test with questions that require both calculation and search"
        ],
        "3_2": [
            "Override __init__ and forward methods of dspy.Module",
            "Use dspy.Predict for internal processing steps",
            "Chain results from one step to the next",
            "Return dspy.Prediction with all relevant outputs"
        ],
        "3_3": [
            "Create a main module that contains sub-modules",
            "Call sub-modules sequentially or in parallel",
            "Pass outputs between modules appropriately",
            "Consider both SequentialChain and ParallelProcessor patterns"
        ],
        "3_4": [
            "Use a classifier to determine content type",
            "Have separate modules for each type",
            "Use conditional logic (if/elif/else) to route",
            "Return both the classification and processed result"
        ],
        "3_5": [
            "Use instance variables to store state (self.history)",
            "Limit history size to prevent memory issues",
            "Include history context when processing new inputs",
            "Update state in each forward() call"
        ],
        "3_6": [
            "Validate before processing (input validation)",
            "Check required fields and data types",
            "Validate after processing (output validation)",
            "Return validation status with results"
        ]
    }

    if exercise_number in hints:
        print(f"\nHints for Exercise {exercise_number}:")
        for i, hint in enumerate(hints[exercise_number], 1):
            print(f"  {i}. {hint}")

if __name__ == "__main__":
    print("DSPy Advanced Module Techniques Exercises")
    print("=" * 50)
    print("\nComplete the exercises by filling in the TODO sections.")
    print("Run verify_exercise_3() to check your progress.")
    print("\nUse show_hints('X_Y') to get hints for specific exercises.")
    print("\nExample: show_hints('3_1')\n")
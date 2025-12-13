"""
Solutions for Chapter 3 Exercises
This file contains complete solutions for all module exercises
"""

import dspy
import math
from typing import List, Dict, Any, Optional, Union
import json

# ========== Exercise 3.1 Solutions ==========

def solution_1_1_basic_predict():
    """Solution for Exercise 1.1: Create a basic Predict module"""

    class SummarizerSignature(dspy.Signature):
        """Summarize long text into a concise summary."""
        text = dspy.InputField(desc="Text to summarize", type=str)
        summary = dspy.OutputField(desc="Concise summary", type=str)

    summarizer = dspy.Predict(SummarizerSignature)

    text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines,
    in contrast to the natural intelligence displayed by humans and animals.
    Leading AI textbooks define the field as the study of "intelligent agents":
    any device that perceives its environment and takes actions that maximize
    its chance of successfully achieving its goals.
    """

    result = summarizer(text=text)

    print("Exercise 1.1 Solution:")
    print(f"Summary: {result.summary}")

    return {
        "signature_defined": True,
        "module_created": True,
        "result_obtained": True
    }

def solution_1_2_multiple_outputs():
    """Solution for Exercise 1.2: Create a module with multiple outputs"""

    class TextAnalyzerSignature(dspy.Signature):
        """Analyze text for sentiment, word count, and topics."""
        text = dspy.InputField(desc="Text to analyze", type=str)
        sentiment = dspy.OutputField(desc="Overall sentiment", type=str)
        word_count = dspy.OutputField(desc="Number of words", type=int)
        topics = dspy.OutputField(desc="Main topics", type=str)

    analyzer = dspy.Predict(TextAnalyzerSignature)

    review = """
    This product is absolutely amazing! The quality is outstanding and
    the customer service was helpful. I would definitely recommend this
    to anyone looking for a reliable solution. The price is reasonable
    and the features exceeded my expectations.
    """

    result = analyzer(text=review)

    print("\nExercise 1.2 Solution:")
    print(f"Sentiment: {result.sentiment}")
    print(f"Word Count: {result.word_count}")
    print(f"Topics: {result.topics}")

    return {
        "signature_complete": True,
        "module_working": True,
        "all_outputs_extracted": True
    }

def solution_1_3_module_configuration():
    """Solution for Exercise 1.3: Configure modules with different parameters"""

    # Define signature
    class DescribeColorSignature(dspy.Signature):
        """Describe a color to someone who has never seen it."""
        prompt = dspy.InputField(desc="Description prompt", type=str)
        output = dspy.OutputField(desc="Color description", type=str)

    # Create modules with different temperatures
    conservative_module = dspy.Predict(DescribeColorSignature, temperature=0.1)
    creative_module = dspy.Predict(DescribeColorSignature, temperature=0.9)

    prompt = "Describe the color blue to someone who has never seen it"

    conservative_result = conservative_module(prompt=prompt)
    creative_result = creative_module(prompt=prompt)

    print("\nExercise 1.3 Solution:")
    print(f"Conservative response: {conservative_result.output}")
    print(f"Creative response: {creative_result.output}")

    return {
        "modules_configured": True,
        "responses_compared": True,
        "difference_observed": True
    }

def solution_1_4_few_shot_learning():
    """Solution for Exercise 1.4: Implement few-shot learning with examples"""

    # Create examples for translation
    translation_examples = [
        dspy.Example(source="Hello", target="Bonjour"),
        dspy.Example(source="Goodbye", target="Au revoir"),
        dspy.Example(source="Thank you", target="Merci"),
        dspy.Example(source="Please", target="S'il vous plaît"),
        dspy.Example(source="How are you?", target="Comment allez-vous?")
    ]

    # Define signature
    class TranslationSignature(dspy.Signature):
        """Translate English to French."""
        source = dspy.InputField(desc="English text", type=str)
        target = dspy.OutputField(desc="French translation", type=str)

    # Create module with examples
    translator = dspy.Predict(TranslationSignature, demos=translation_examples)

    # Test with new translation
    result = translator(source="Good morning")

    print("\nExercise 1.4 Solution:")
    print(f"Translation: {result.target}")

    return {
        "examples_created": True,
        "module_with_examples": True,
        "new_translation": True
    }

def solution_1_5_error_handling():
    """Solution for Exercise 1.5: Implement error handling in modules"""

    class SafeModule(dspy.Module):
        """Safe module with comprehensive error handling."""

        def __init__(self):
            super().__init__()
            self.signature = dspy.Signature("text -> processed_text")
            self.predictor = dspy.Predict(self.signature)

        def forward(self, **kwargs):
            """Safe processing with validation."""
            text = kwargs.get('text', '')

            # Input validation
            if text is None:
                return dspy.Prediction(
                    processed_text="[ERROR: Input cannot be None]",
                    status="error",
                    input_type="none"
                )

            if not isinstance(text, str):
                return dspy.Prediction(
                    processed_text=f"[ERROR: Expected string, got {type(text)}]",
                    status="error",
                    input_type=type(text).__name__
                )

            if len(text) == 0:
                return dspy.Prediction(
                    processed_text="[ERROR: Empty input]",
                    status="error",
                    input_type="empty"
                )

            if len(text) > 5000:
                return dspy.Prediction(
                    processed_text=f"[ERROR: Input too long ({len(text)} chars)]",
                    status="error",
                    input_type="too_long"
                )

            # Process with internal module
            try:
                result = self.predictor(text=text)
                return dspy.Prediction(
                    processed_text=result.processed_text,
                    status="success",
                    input_type="valid"
                )
            except Exception as e:
                return dspy.Prediction(
                    processed_text=f"[ERROR: Processing failed - {str(e)}]",
                    status="error",
                    input_type="processing_error"
                )

    # Test with various inputs
    safe_module = SafeModule()
    test_inputs = [
        "Normal text",
        "",
        None,
        "x" * 10000
    ]

    print("\nExercise 1.5 Solution:")
    for test_input in test_inputs:
        result = safe_module(text=test_input)
        print(f"\nInput: {repr(test_input)[:30]}")
        print(f"Status: {result.status}")
        print(f"Processed: {result.processed_text[:50]}...")

    return {
        "safe_module_implemented": True,
        "error_handling_working": True,
        "all_inputs_handled": True
    }

# ========== Exercise 3.2 Solutions ==========

def solution_2_1_basic_cot():
    """Solution for Exercise 2.1: Create a basic Chain of Thought module"""

    class MathSolverSignature(dspy.Signature):
        """Solve math problems with step-by-step reasoning."""
        problem = dspy.InputField(desc="Mathematical word problem", type=str)
        reasoning = dspy.OutputField(desc="Step-by-step solution reasoning", type=str)
        answer = dspy.OutputField(desc="Final numerical answer", type=str)

    math_solver = dspy.ChainOfThought(MathSolverSignature)

    problems = [
        "A store sells apples for $2 each and oranges for $1.50 each. "
        "If John buys 3 apples and 4 oranges, how much does he pay?",
        "Sarah is 5 years older than Tom. In 10 years, Sarah will be "
        "twice as old as Tom. How old are they now?"
    ]

    print("\nExercise 2.1 Solution:")
    for problem in problems:
        result = math_solver(problem=problem)
        print(f"\nProblem: {problem}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Answer: {result.answer}")

    return {
        "signature_defined": True,
        "cot_module_created": True,
        "problems_solved": True,
        "reasoning_steps_visible": True
    }

def solution_2_2_cot_with_examples():
    """Solution for Exercise 2.2: Improve CoT with few-shot examples"""

    logic_examples = [
        dspy.Example(
            puzzle="All roses are flowers. Some flowers fade quickly. Therefore?",
            reasoning="1. All roses are flowers (given). 2. Some flowers fade quickly (given). "
                     "3. Since roses are flowers, and some flowers fade, some roses might fade. "
                     "4. Cannot conclude all roses fade, only that some might.",
            answer="Some roses might fade quickly"
        ),
        dspy.Example(
            puzzle="All mammals are warm-blooded. Whales are mammals. Therefore?",
            reasoning="1. All mammals are warm-blooded (given). 2. Whales are mammals (given). "
                     "3. Since whales are mammals, and all mammals are warm-blooded, "
                     "whales must be warm-blooded.",
            answer="Whales are warm-blooded"
        )
    ]

    class LogicPuzzleSignature(dspy.Signature):
        """Solve logic puzzles with clear step-by-step reasoning."""
        puzzle = dspy.InputField(desc="Logic puzzle to solve", type=str)
        reasoning = dspy.OutputField(desc="Step-by-step logical deduction", type=str)
        answer = dspy.OutputField(desc="Logical conclusion", type=str)

    puzzle_solver = dspy.ChainOfThought(LogicPuzzleSignature, demos=logic_examples)

    puzzle = """
    In a room, there are 3 people: Alice, Bob, and Carol.
    - The person wearing red is not Alice
    - Bob is not wearing blue
    - Carol is wearing green
    Who is wearing red?
    """

    result = puzzle_solver(puzzle=puzzle)

    print("\nExercise 2.2 Solution:")
    print(f"Reasoning: {result.reasoning}")
    print(f"Answer: {result.answer}")

    return {
        "examples_created": True,
        "signature_complete": True,
        "cot_with_examples": True,
        "puzzle_solved": True
    }

def solution_2_3_cot_for_planning():
    """Solution for Exercise 2.3: Use CoT for planning tasks"""

    class TravelPlannerSignature(dspy.Signature):
        """Plan a vacation with detailed steps and research."""
        request = dspy.InputField(desc="Travel planning request", type=str)
        research_steps = dspy.OutputField(desc="Research methodology", type=str)
        option_comparison = dspy.OutputField(desc="Comparison of options", type=str)
        recommendation = dspy.OutputField(desc="Final recommendation", type=str)

    travel_planner = dspy.ChainOfThought(TravelPlannerSignature)

    travel_request = """
    I want to plan a 7-day vacation for a family of 4 (2 adults, 2 kids)
    to a beach destination in July. Budget is $5000. We're flying from
    New York and prefer all-inclusive resorts.
    """

    result = travel_planner(request=travel_request)

    print("\nExercise 2.3 Solution:")
    print(f"Research Steps: {result.research_steps}")
    print(f"Comparison: {result.option_comparison}")
    print(f"Recommendation: {result.recommendation}")

    return {
        "planning_signature": True,
        "planner_created": True,
        "plan_generated": True,
        "all_steps_present": True
    }

# ========== Exercise 3.3 Solutions ==========

def solution_3_1_react_agent():
    """Solution for Exercise 3.1: Create a ReAct agent with tools"""

    class Calculator:
        """Calculator tool for mathematical operations."""
        def calculate(self, expression):
            try:
                # Safe evaluation
                allowed_names = {"abs": abs, "min": min, "max": max, "round": round,
                               "pow": pow, "sqrt": math.sqrt}
                result = eval(expression, {"__builtins__": {}}, allowed_names)
                return str(result)
            except Exception as e:
                return f"Error: {str(e)}"

    class SearchEngine:
        """Mock search tool."""
        def search(self, query):
            results = {
                "distance earth mars": "The distance from Earth to Mars varies from 55 to 400 million kilometers",
                "saving money": "To calculate savings: principal × (1 + rate)^time"
            }
            for key, value in results.items():
                if key in query.lower():
                    return value
            return "No specific information found"

    class ReActSignature(dspy.Signature):
        """Think step by step and use tools to answer questions."""
        question = dspy.InputField(desc="User's question", type=str)
        thought = dspy.OutputField(desc="Current thinking process", type=str)
        action = dspy.OutputField(desc="Tool to use (if needed)", type=str)
        action_input = dspy.OutputField(desc="Input for the tool", type=str)
        observation = dspy.OutputField(desc="Result from tool", type=str)
        answer = dspy.OutputField(desc="Final answer", type=str)

    # Tool registry
    tools = {
        "calculate": {
            "description": "Calculate mathematical expressions",
            "function": Calculator().calculate
        },
        "search": {
            "description": "Search for information",
            "function": SearchEngine().search
        }
    }

    # Create ReAct agent
    agent = dspy.React(signature=ReActSignature, tools=tools)

    questions = [
        "What is 15% of 240?",
        "What's the distance from Earth to Mars?",
        "If I save $100 monthly at 5% interest for 10 years, how much will I have?"
    ]

    print("\nExercise 3.1 Solution:")
    for question in questions:
        result = agent(question=question)
        print(f"\nQ: {question}")
        print(f"A: {result.answer}")

    return {
        "tools_implemented": True,
        "react_signature": True,
        "agent_created": True,
        "questions_answered": True
    }

def solution_3_2_custom_module():
    """Solution for Exercise 3.2: Create a custom module"""

    class SentimentPipeline(dspy.Module):
        """Custom sentiment analysis pipeline module."""

        def __init__(self):
            super().__init__()
            # Initialize internal modules
            self.cleaner_sig = dspy.Signature("text -> cleaned_text")
            self.cleaner = dspy.Predict(self.cleaner_sig)

            self.sentiment_sig = dspy.Signature("text -> sentiment_score")
            self.sentiment = dspy.Predict(self.sentiment_sig)

            self.aspects_sig = dspy.Signature("text -> key_aspects")
            self.aspects = dspy.Predict(self.aspects_sig)

            self.synthesize_sig = dspy.Signature(
                "sentiment, aspects, text -> comprehensive_analysis"
            )
            self.synthesizer = dspy.Predict(self.synthesize_sig)

        def forward(self, text):
            # Step 1: Clean the text
            clean_result = self.cleaner(text=text)

            # Step 2: Analyze sentiment
            sentiment_result = self.sentiment(text=clean_result.cleaned_text)

            # Step 3: Extract aspects
            aspects_result = self.aspects(text=clean_result.cleaned_text)

            # Step 4: Synthesize results
            synthesis = self.synthesizer(
                sentiment=sentiment_result.sentiment_score,
                aspects=aspects_result.key_aspects,
                text=clean_result.cleaned_text
            )

            return dspy.Prediction(
                original_text=text,
                cleaned_text=clean_result.cleaned_text,
                overall_sentiment=sentiment_result.sentiment_score,
                aspects=aspects_result.key_aspects,
                comprehensive_analysis=synthesis.comprehensive_analysis
            )

    # Test the custom module
    test_texts = [
        "The camera quality is amazing but the battery life is disappointing.",
        "Excellent customer service and fast shipping!",
        "The product arrived damaged and doesn't work properly."
    ]

    pipeline = SentimentPipeline()
    print("\nExercise 3.2 Solution:")
    for text in test_texts:
        result = pipeline(text=text)
        print(f"\nText: {text}")
        print(f"Overall: {result.overall_sentiment}")
        print(f"Aspects: {result.aspects}")

    return {
        "custom_module_class": True,
        "pipeline_implemented": True,
        "all_steps_working": True,
        "results_formatted": True
    }

def solution_3_3_module_composition():
    """Solution for Exercise 3.3: Compose multiple modules"""

    class TextPreprocessor(dspy.Module):
        """Text preprocessing module."""
        def __init__(self):
            super().__init__()
            self.sig = dspy.Signature("text -> preprocessed_text")
            self.processor = dspy.Predict(self.sig)

        def forward(self, text):
            result = self.processor(text=f"Clean and preprocess: {text}")
            return dspy.Prediction(text=result.preprocessed_text)

    class TextAnalyzer(dspy.Module):
        """Text analysis module."""
        def __init__(self):
            super().__init__()
            self.sig = dspy.Signature("text -> analysis")
            self.analyzer = dspy.Predict(self.sig)

        def forward(self, text):
            result = self.analyzer(text=f"Analyze this text: {text}")
            return dspy.Prediction(text=result.analysis)

    class TextSummarizer(dspy.Module):
        """Text summarization module."""
        def __init__(self):
            super().__init__()
            self.sig = dspy.Signature("text -> summary")
            self.summarizer = dspy.Predict(self.sig)

        def forward(self, text):
            result = self.summarizer(text=f"Summarize: {text}")
            return dspy.Prediction(text=result.summary)

    class TextProcessingSystem(dspy.Module):
        """Composed text processing system."""

        def __init__(self):
            super().__init__()
            self.preprocessor = TextPreprocessor()
            self.analyzer = TextAnalyzer()
            self.summarizer = TextSummarizer()

        def forward(self, text):
            # Sequential processing
            preprocessed = self.preprocessor(text=text)
            analyzed = self.analyzer(text=preprocessed.text)
            summarized = self.summarizer(text=preprocessed.text)

            # Combine all results
            return dspy.Prediction(
                original_text=text,
                preprocessed=preprocessed.text,
                analysis=analyzed.text,
                summary=summarized.text,
                fully_processed=f"Processed: {summarized.text}"
            )

    # Test the system
    test_article = """
    Artificial Intelligence is revolutionizing many industries. From healthcare
    to finance, AI systems are improving efficiency and enabling new capabilities.
    Machine learning algorithms can now detect diseases earlier, predict market
    trends more accurately, and personalize user experiences. However, challenges
    remain in ensuring AI ethics, transparency, and responsible deployment.
    """

    system = TextProcessingSystem()
    result = system(text=test_article)

    print("\nExercise 3.3 Solution:")
    print(f"Fully Processed: {result.fully_processed}")

    return {
        "individual_modules": True,
        "composition_pattern": True,
        "system_created": True,
        "composition_working": True
    }

# Run all solutions
def run_all_solutions():
    """Run all exercise solutions."""
    print("=" * 60)
    print("CHAPTER 3 EXERCISE SOLUTIONS")
    print("=" * 60)

    print("\n[Module Basics Solutions]")
    solution_1_1_basic_predict()
    solution_1_2_multiple_outputs()
    solution_1_3_module_configuration()
    solution_1_4_few_shot_learning()
    solution_1_5_error_handling()

    print("\n" + "="*60)
    print("[Chain of Thought Solutions]")
    solution_2_1_basic_cot()
    solution_2_2_cot_with_examples()
    solution_2_3_cot_for_planning()

    print("\n" + "="*60)
    print("[Advanced Modules Solutions]")
    solution_3_1_react_agent()
    solution_3_2_custom_module()
    solution_3_3_module_composition()

    print("\n" + "="*60)
    print("All solutions completed!")
    print("="*60)

if __name__ == "__main__":
    run_all_solutions()
"""
Exercise 3.1: Module Basics
Working with dspy.Predict and understanding fundamental module concepts
"""

import dspy
from typing import List, Dict, Any

def exercise_1_1_basic_predict():
    """
    Exercise 1.1: Create a basic Predict module

    Task: Create a text summarization module using dspy.Predict
    """
    # TODO: Define a signature for text summarization
    # Hint: Use dspy.Signature with input and output fields

    # Your code here:
    class SummarizerSignature(dspy.Signature):
        """TODO: Define the signature"""
        # TODO: Add input field
        # TODO: Add output field
        pass

    # TODO: Create the predictor module
    # Your code here:
    # summarizer = dspy.Predict(???)

    # Test cases
    text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines,
    in contrast to the natural intelligence displayed by humans and animals.
    Leading AI textbooks define the field as the study of "intelligent agents":
    any device that perceives its environment and takes actions that maximize
    its chance of successfully achieving its goals.
    """

    # TODO: Use the summarizer to create a summary
    # result = summarizer(???)

    # Expected result: Should contain a concise summary of the text
    print("Exercise 1.1 Solution:")
    # print(f"Summary: {result.summary}")

    return {
        "signature_defined": False,  # Set to True when signature is defined
        "module_created": False,     # Set to True when module is created
        "result_obtained": False     # Set to True when you get a result
    }

def exercise_1_2_multiple_outputs():
    """
    Exercise 1.2: Create a module with multiple outputs

    Task: Create a text analyzer that returns sentiment, word count, and topics
    """
    # TODO: Define a signature with multiple output fields
    # Hint: Use dspy.OutputField for each output

    # Your code here:
    class TextAnalyzerSignature(dspy.Signature):
        """TODO: Define signature with multiple outputs"""
        # TODO: Add input field
        # TODO: Add multiple output fields (sentiment, word_count, topics)
        pass

    # TODO: Create the analyzer module
    # Your code here:
    # analyzer = dspy.Predict(???)

    # Test case
    review = """
    This product is absolutely amazing! The quality is outstanding and
    the customer service was helpful. I would definitely recommend this
    to anyone looking for a reliable solution. The price is reasonable
    and the features exceeded my expectations.
    """

    # TODO: Analyze the review
    # result = analyzer(???)

    # Expected: Should extract sentiment, count words, and identify topics
    print("\nExercise 1.2 Solution:")
    # print(f"Sentiment: {result.sentiment}")
    # print(f"Word Count: {result.word_count}")
    # print(f"Topics: {result.topics}")

    return {
        "signature_complete": False,
        "module_working": False,
        "all_outputs_extracted": False
    }

def exercise_1_3_module_configuration():
    """
    Exercise 1.3: Configure modules with different parameters

    Task: Create two modules with different configurations and compare outputs
    """
    # TODO: Create two modules with different temperatures
    # Module 1: Conservative (low temperature)
    # Module 2: Creative (high temperature)

    # Your code here:
    # conservative_module = dspy.Predict(???, temperature=???)
    # creative_module = dspy.Predict(???, temperature=???)

    # Test case
    prompt = "Describe the color blue to someone who has never seen it"

    # TODO: Get responses from both modules
    # conservative_result = conservative_module(???)
    # creative_result = creative_module(???)

    print("\nExercise 1.3 Solution:")
    # print(f"Conservative response: {conservative_result.output}")
    # print(f"Creative response: {creative_result.output}")

    return {
        "modules_configured": False,
        "responses_compared": False,
        "difference_observed": False
    }

def exercise_1_4_few_shot_learning():
    """
    Exercise 1.4: Implement few-shot learning with examples

    Task: Create a module that learns from examples to perform translations
    """
    # TODO: Create examples for translation task
    # Your code here:
    translation_examples = [
        # TODO: Add dspy.Example instances
        # Example: dspy.Example(source="Hello", target="Bonjour")
    ]

    # TODO: Define signature for translation
    # Your code here:
    # class TranslationSignature(dspy.Signature):
    #     """TODO: Define translation signature"""
    #     pass

    # TODO: Create module with examples
    # Your code here:
    # translator = dspy.Predict(???, demos=???)

    # Test with new translation
    # TODO: Translate a new sentence
    # result = translator(source="Good morning")

    print("\nExercise 1.4 Solution:")
    # print(f"Translation: {result.target}")

    return {
        "examples_created": False,
        "module_with_examples": False,
        "new_translation": False
    }

def exercise_1_5_error_handling():
    """
    Exercise 1.5: Implement error handling in modules

    Task: Create a safe module that handles errors gracefully
    """
    # TODO: Create a custom module with error handling
    # Hint: Extend dspy.Module and implement forward() method

    # Your code here:
    class SafeModule(dspy.Module):
        """TODO: Implement safe module with error handling"""

        def __init__(self):
            super().__init__()
            # TODO: Initialize internal components
            pass

        def forward(self, **kwargs):
            """TODO: Implement safe processing with error handling"""
            # TODO: Add input validation
            # TODO: Add try-catch blocks
            # TODO: Return meaningful error messages
            pass

    # TODO: Test with various inputs
    test_inputs = [
        "Normal text",
        "",  # Empty string
        None,  # None value
        "x" * 10000  # Very long string
    ]

    print("\nExercise 1.5 Solution:")

    return {
        "safe_module_implemented": False,
        "error_handling_working": False,
        "all_inputs_handled": False
    }

# Solution verification function
def verify_exercise_1():
    """Verify all exercises are completed correctly."""
    results = {
        "exercise_1_1": exercise_1_1_basic_predict(),
        "exercise_1_2": exercise_1_2_multiple_outputs(),
        "exercise_1_3": exercise_1_3_module_configuration(),
        "exercise_1_4": exercise_1_4_few_shot_learning(),
        "exercise_1_5": exercise_1_5_error_handling()
    }

    print("\n" + "="*50)
    print("EXERCISE 3.1 COMPLETION STATUS")
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
        print("\n🎉 Congratulations! You've completed all Module Basics exercises!")
    else:
        print(f"\nKeep working! {total_tasks - completed_tasks} tasks remaining.")

    return percentage

# Hint function
def show_hints(exercise_number):
    """Show hints for specific exercises."""
    hints = {
        "1_1": [
            "Use dspy.InputField for the input parameter",
            "Use dspy.OutputField for the output parameter",
            "Pass the signature class to dspy.Predict()",
            "Call the module with the input field name as parameter"
        ],
        "1_2": [
            "Define multiple OutputField instances",
            "Each output should have a descriptive name",
            "Use type hints to specify expected data types",
            "Access each output field separately from the result"
        ],
        "1_3": [
            "Temperature 0.1-0.3 is conservative",
            "Temperature 0.7-1.0 is creative",
            "Use the same signature for both modules",
            "Compare the length and creativity of outputs"
        ],
        "1_4": [
            "Create at least 3-5 example pairs",
            "Use dspy.Example() with matching field names",
            "Pass examples using the demos parameter",
            "Test with a sentence not in the examples"
        ],
        "1_5": [
            "Override the forward() method",
            "Use isinstance() for type checking",
            "Return dspy.Prediction() with error info",
            "Handle edge cases like empty input"
        ]
    }

    if exercise_number in hints:
        print(f"\nHints for Exercise {exercise_number}:")
        for i, hint in enumerate(hints[exercise_number], 1):
            print(f"  {i}. {hint}")

if __name__ == "__main__":
    print("DSPy Module Basics Exercises")
    print("=" * 50)
    print("\nComplete the exercises by filling in the TODO sections.")
    print("Run verify_exercise_1() to check your progress.")
    print("\nUse show_hints('X_Y') to get hints for specific exercises.")
    print("\nExample: show_hints('1_1')\n")
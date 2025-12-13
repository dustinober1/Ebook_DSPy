"""
Exercise 3.2: Chain of Thought Module
Working with dspy.ChainOfThought for step-by-step reasoning
"""

import dspy
import math
from typing import List, Dict, Any

def exercise_2_1_basic_cot():
    """
    Exercise 2.1: Create a basic Chain of Thought module

    Task: Create a math word problem solver using Chain of Thought
    """
    # TODO: Define a signature for math problem solving
    # Include fields for reasoning steps

    # Your code here:
    class MathSolverSignature(dspy.Signature):
        """TODO: Define signature for step-by-step math problem solving"""
        # TODO: Add input field for the problem
        # TODO: Add output field for reasoning steps
        # TODO: Add output field for the final answer
        pass

    # TODO: Create the ChainOfThought module
    # Your code here:
    # math_solver = dspy.ChainOfThought(???)

    # Test cases
    problems = [
        "A store sells apples for $2 each and oranges for $1.50 each. "
        "If John buys 3 apples and 4 oranges, how much does he pay?",
        "Sarah is 5 years older than Tom. In 10 years, Sarah will be "
        "twice as old as Tom. How old are they now?"
    ]

    print("Exercise 2.1 Solution:")
    for problem in problems:
        # TODO: Solve each problem using the ChainOfThought module
        # result = math_solver(problem=???)
        # print(f"\nProblem: {problem}")
        # print(f"Reasoning: {result.reasoning}")
        # print(f"Answer: {result.answer}")

        pass  # Remove this line when implementing

    return {
        "signature_defined": False,
        "cot_module_created": False,
        "problems_solved": False,
        "reasoning_steps_visible": False
    }

def exercise_2_2_cot_with_examples():
    """
    Exercise 2.2: Improve CoT with few-shot examples

    Task: Create a logical puzzle solver with demonstration examples
    """
    # TODO: Create examples showing good reasoning patterns
    # Your code here:
    logic_examples = [
        # TODO: Add dspy.Example instances with reasoning
        # Example: dspy.Example(
        #     puzzle="All roses are flowers. Some flowers fade quickly. Therefore?",
        #     reasoning="1. All roses are flowers (given). 2. Some flowers fade quickly (given). "
        #              "3. Since roses are flowers, and some flowers fade, some roses might fade.",
        #     answer="Some roses might fade quickly"
        # )
    ]

    # TODO: Define signature for logical puzzles
    # Your code here:
    class LogicPuzzleSignature(dspy.Signature):
        """TODO: Define signature for logical puzzle solving"""
        pass

    # TODO: Create CoT module with examples
    # Your code here:
    # puzzle_solver = dspy.ChainOfThought(???, demos=???)

    # Test puzzle
    puzzle = """
    In a room, there are 3 people: Alice, Bob, and Carol.
    - The person wearing red is not Alice
    - Bob is not wearing blue
    - Carol is wearing green
    Who is wearing red?
    """

    # TODO: Solve the puzzle with reasoning
    # result = puzzle_solver(puzzle=puzzle)

    print("\nExercise 2.2 Solution:")
    # print(f"Puzzle: {puzzle}")
    # print(f"Reasoning: {result.reasoning}")
    # print(f"Answer: {result.answer}")

    return {
        "examples_created": False,
        "signature_complete": False,
        "cot_with_examples": False,
        "puzzle_solved": False
    }

def exercise_2_3_cot_for_planning():
    """
    Exercise 2.3: Use CoT for planning tasks

    Task: Create a travel planning assistant using Chain of Thought
    """
    # TODO: Define a signature for travel planning
    # Include steps for research, comparison, and decision

    # Your code here:
    class TravelPlannerSignature(dspy.Signature):
        """TODO: Define signature for travel planning with steps"""
        # TODO: Add input field for travel request
        # TODO: Add output fields for research steps
        # TODO: Add output field for options comparison
        # TODO: Add output field for final recommendation
        pass

    # TODO: Create the planner module
    # Your code here:
    # travel_planner = dspy.ChainOfThought(???)

    # Test case
    travel_request = """
    I want to plan a 7-day vacation for a family of 4 (2 adults, 2 kids)
    to a beach destination in July. Budget is $5000. We're flying from
    New York and prefer all-inclusive resorts.
    """

    # TODO: Create a travel plan
    # result = travel_planner(request=travel_request)

    print("\nExercise 2.3 Solution:")
    # print(f"Request: {travel_request}")
    # print(f"Research Steps: {result.research_steps}")
    # print(f"Comparison: {result.comparison}")
    # print(f"Recommendation: {result.recommendation}")

    return {
        "planning_signature": False,
        "planner_created": False,
        "plan_generated": False,
        "all_steps_present": False
    }

def exercise_2_4_cot_for_error_analysis():
    """
    Exercise 2.4: Use CoT for error analysis and debugging

    Task: Create a code debugging assistant
    """
    # TODO: Define signature for code debugging
    # Should identify errors, explain causes, and suggest fixes

    # Your code here:
    class CodeDebuggerSignature(dspy.Signature):
        """TODO: Define signature for code debugging"""
        pass

    # TODO: Create debugger with examples
    # Your code here:
    # debugger = dspy.ChainOfThought(???)

    # Test code with errors
    buggy_code = """
    def calculate_average(numbers):
        total = 0
        for num in numbers:
            total += num
        return total / len(numbers)

    # This will cause an error when called with empty list
    result = calculate_average([])
    print(f"Average: {result}")
    """

    # TODO: Debug the code
    # result = debugger(code=buggy_code)

    print("\nExercise 2.4 Solution:")
    # print(f"Error Identified: {result.error}")
    # print(f"Reasoning: {result.reasoning}")
    # print(f"Fix Suggested: {result.fix}")

    return {
        "debugger_signature": False,
        "debugger_working": False,
        "error_identified": False,
        "fix_provided": False
    }

def exercise_2_5_cot_temperature_control():
    """
    Exercise 2.5: Experiment with temperature in CoT modules

    Task: Compare CoT outputs at different temperature settings
    """
    # TODO: Create CoT modules with different temperatures
    # Module 1: Low temperature (0.1)
    # Module 2: Medium temperature (0.5)
    # Module 3: High temperature (0.9)

    # Your code here:
    # cot_creative = dspy.ChainOfThought(???, temperature=???)
    # cot_balanced = dspy.ChainOfThought(???, temperature=???)
    # cot_precise = dspy.ChainOfThought(???, temperature=???)

    # Creative task
    creative_prompt = """
    Come up with a unique business idea that combines AI with sustainable agriculture.
    Explain the concept, target market, and potential challenges.
    """

    # TODO: Generate responses with different temperatures
    # creative_result = cot_creative(prompt=creative_prompt)
    # balanced_result = cot_balanced(prompt=creative_prompt)
    # precise_result = cot_precise(prompt=creative_prompt)

    print("\nExercise 2.5 Solution:")
    print("Comparing outputs at different temperatures:")
    # print(f"\nHigh Temperature (0.9):")
    # print(f"Idea: {creative_result.idea}")
    # print(f"\nMedium Temperature (0.5):")
    # print(f"Idea: {balanced_result.idea}")
    # print(f"\nLow Temperature (0.1):")
    # print(f"Idea: {precise_result.idea}")

    return {
        "modules_with_temps": False,
        "outputs_generated": False,
        "differences_observed": False,
        "creativity_assessed": False
    }

def exercise_2_6_cot_for_complex_reasoning():
    """
    Exercise 2.6: Multi-step complex reasoning

    Task: Create a module for ethical dilemma analysis
    """
    # TODO: Define comprehensive signature for ethical analysis
    # Should consider stakeholders, principles, consequences, etc.

    # Your code here:
    class EthicalAnalyzerSignature(dspy.Signature):
        """TODO: Define signature for ethical dilemma analysis"""
        # TODO: Add field for stakeholders identification
        # TODO: Add field for ethical principles
        # TODO: Add field for consequence analysis
        # TODO: Add field for recommendation
        pass

    # TODO: Create analyzer with structured examples
    # Your code here:
    # ethical_analyzer = dspy.ChainOfThought(???, demos=???)

    # Test dilemma
    dilemma = """
    A self-driving car must choose between hitting a pedestrian or swerving
    into a wall that will injure the passenger. The car has 1 passenger.
    The pedestrian is a child. There is no time to brake safely.
    What should the car do and why?
    """

    # TODO: Analyze the dilemma
    # result = ethical_analyzer(dilemma=dilemma)

    print("\nExercise 2.6 Solution:")
    # print(f"Stakeholders: {result.stakeholders}")
    # print(f"Ethical Principles: {result.principles}")
    # print(f"Consequences: {result.consequences}")
    # print(f"Recommendation: {result.recommendation}")

    return {
        "ethical_signature": False,
        "analyzer_created": False,
        "dilemma_analyzed": False,
        "all_considerations": False
    }

# Solution verification function
def verify_exercise_2():
    """Verify all Chain of Thought exercises are completed correctly."""
    results = {
        "exercise_2_1": exercise_2_1_basic_cot(),
        "exercise_2_2": exercise_2_2_cot_with_examples(),
        "exercise_2_3": exercise_2_3_cot_for_planning(),
        "exercise_2_4": exercise_2_4_cot_for_error_analysis(),
        "exercise_2_5": exercise_2_5_cot_temperature_control(),
        "exercise_2_6": exercise_2_6_cot_for_complex_reasoning()
    }

    print("\n" + "="*50)
    print("EXERCISE 3.2 COMPLETION STATUS")
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
        print("\n🎉 Excellent! You've mastered Chain of Thought modules!")
    else:
        print(f"\nKeep practicing! {total_tasks - completed_tasks} tasks remaining.")

    return percentage

# Hint function
def show_hints(exercise_number):
    """Show hints for specific exercises."""
    hints = {
        "2_1": [
            "Include 'reasoning' as an OutputField in the signature",
            "ChainOfThought automatically generates reasoning steps",
            "Access reasoning through result.reasoning",
            "Make sure to show both reasoning and final answer"
        ],
        "2_2": [
            "Create 2-3 examples with detailed step-by-step reasoning",
            "Show how to eliminate possibilities in logic puzzles",
            "Use the demos parameter to provide examples",
            "Reasoning should be clear and sequential"
        ],
        "2_3": [
            "Plan should include research, options, comparison, recommendation",
            "Consider budget, constraints, and preferences",
            "Show how the reasoning leads to the final plan",
            "Include multiple options before making recommendation"
        ],
        "2_4": [
            "Identify specific line numbers or sections with errors",
            "Explain why the error occurs (edge cases, logic flaws)",
            "Provide corrected code or clear fix instructions",
            "Consider multiple potential issues in the code"
        ],
        "2_5": [
            "Higher temperature = more creative but less focused",
            "Lower temperature = more predictable but less creative",
            "Test the same prompt with different temperatures",
            "Compare length, creativity, and coherence of outputs"
        ],
        "2_6": [
            "Consider multiple ethical frameworks (utilitarian, deontological)",
            "Identify all affected parties (stakeholders)",
            "Analyze short and long-term consequences",
            "Provide nuanced recommendation with justification"
        ]
    }

    if exercise_number in hints:
        print(f"\nHints for Exercise {exercise_number}:")
        for i, hint in enumerate(hints[exercise_number], 1):
            print(f"  {i}. {hint}")

if __name__ == "__main__":
    print("DSPy Chain of Thought Module Exercises")
    print("=" * 50)
    print("\nComplete the exercises by filling in the TODO sections.")
    print("Run verify_exercise_2() to check your progress.")
    print("\nUse show_hints('X_Y') to get hints for specific exercises.")
    print("\nExample: show_hints('2_1')\n")
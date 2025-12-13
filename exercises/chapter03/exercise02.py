"""
Exercise 2: Chain of Thought Implementation
Solution for Exercise 2 from Chapter 3

Task: Build a math problem solver using ChainOfThought
- Create a math problem signature
- Implement step-by-step reasoning
- Handle different types of math problems
"""

import dspy
import math
from typing import List, Dict, Any

class MathProblemSolver:
    """Math problem solver using Chain of Thought reasoning."""

    def __init__(self):
        # Define the signature for math problems
        class MathSolverSignature(dspy.Signature):
            """Solve math problems with step-by-step reasoning."""
            problem = dspy.InputField(desc="Mathematical problem to solve", type=str)
            reasoning = dspy.OutputField(desc="Step-by-step reasoning process", type=str)
            calculations = dspy.OutputField(desc("Detailed calculations", type=str)
            answer = dspy.OutputField(desc="Final answer", type=str)

        # Create examples for different math problem types
        math_examples = [
            # Percentage problems
            dspy.Example(
                problem="What is 25% of 200?",
                reasoning="1. Convert percentage to decimal: 25% = 0.25\n2. Multiply by the number",
                calculations="0.25 × 200 = 50",
                answer="50"
            ),
            # Area problems
            dspy.Example(
                problem="What is the area of a rectangle with length 8 and width 5?",
                reasoning="1. Use rectangle area formula: A = length × width\n2. Multiply the dimensions",
                calculations="8 × 5 = 40",
                answer="40 square units"
            ),
            # Speed problems
            dspy.Example(
                problem="If a car travels 120 miles in 2 hours, what is its average speed?",
                reasoning="1. Use speed formula: speed = distance ÷ time\n2. Calculate the speed",
                calculations="120 miles ÷ 2 hours = 60 mph",
                answer="60 miles per hour"
            )
        ]

        # Initialize the ChainOfThought module
        self.solver = dspy.ChainOfThought(
            MathSolverSignature,
            demos=math_examples
        )

    def solve_problem(self, problem: str) -> Dict[str, Any]:
        """Solve a math problem with step-by-step reasoning."""

        try:
            # Use the solver to get the solution
            result = self.solver(problem=problem)

            return {
                'problem': problem,
                'reasoning': result.reasoning,
                'calculations': result.calculations,
                'answer': result.answer,
                'success': True
            }

        except Exception as e:
            return {
                'problem': problem,
                'error': str(e),
                'success': False
            }

    def solve_batch(self, problems: List[str]) -> List[Dict[str, Any]]:
        """Solve multiple math problems."""

        results = []
        for problem in problems:
            result = self.solve_problem(problem)
            results.append(result)

        return results

def test_math_solver():
    """Test the math problem solver with various types of problems."""

    solver = MathProblemSolver()

    # Test problems of different types
    test_problems = [
        # Percentage problems
        "What is 15% of 300?",
        "A shirt costs $40. If it's on sale for 20% off, what is the sale price?",

        # Geometry problems
        "What is the area of a circle with radius 5?",
        "A triangle has a base of 10 and height of 6. What is its area?",

        # Speed/distance problems
        "If I drive 60 mph for 3 hours, how far will I travel?",
        "A train travels 150 miles in 2.5 hours. What is its average speed?",

        # Arithmetic problems
        "What is 345 + 678?",
        "What is 144 ÷ 12?",

        # Word problems
        "John has 25 apples. He gives 8 to his friend. How many does he have left?",
        "A bakery sells 120 donuts per day. How many donuts do they sell in a week?"
    ]

    print("=" * 60)
    print("Math Problem Solver Test")
    print("=" * 60)

    results = solver.solve_batch(test_problems)

    # Display results
    for i, result in enumerate(results, 1):
        print(f"\nProblem {i}:")
        print(f"Problem: {result['problem']}")

        if result['success']:
            print(f"\nReasoning:")
            print(result['reasoning'])
            print(f"\nCalculations:")
            print(result['calculations'])
            print(f"\nAnswer: {result['answer']}")
        else:
            print(f"Error: {result['error']}")

        print("-" * 40)

    # Summary
    successful = sum(1 for r in results if r['success'])
    print(f"\nSummary:")
    print(f"Total problems: {len(results)}")
    print(f"Successfully solved: {successful}")
    print(f"Success rate: {successful/len(results)*100:.1f}%")

def create_interactive_solver():
    """Create an interactive math problem solver."""

    solver = MathProblemSolver()

    print("\n" + "=" * 60)
    print("Interactive Math Problem Solver")
    print("Type 'quit' to exit")
    print("=" * 60)

    while True:
        problem = input("\nEnter a math problem: ").strip()

        if problem.lower() == 'quit':
            print("Goodbye!")
            break

        if not problem:
            print("Please enter a valid problem.")
            continue

        result = solver.solve_problem(problem)

        if result['success']:
            print("\nSolution:")
            print(f"Reasoning: {result['reasoning']}")
            print(f"Calculations: {result['calculations']}")
            print(f"Answer: {result['answer']}")
        else:
            print(f"\nError solving problem: {result['error']}")

def main():
    """Main function to run Exercise 2."""

    print("\n" + "=" * 60)
    print("Exercise 2: Chain of Thought Implementation")
    print("Building a math problem solver with step-by-step reasoning")
    print("=" * 60)

    # Run the test
    test_math_solver()

    # Option to run interactive solver
    response = input("\nWould you like to try the interactive solver? (y/n): ")
    if response.lower().startswith('y'):
        create_interactive_solver()

if __name__ == "__main__":
    main()
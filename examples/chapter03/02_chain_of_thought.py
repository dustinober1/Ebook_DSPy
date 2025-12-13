"""
Chain of Thought Module Examples

This file demonstrates advanced reasoning capabilities using DSPy's ChainOfThought module:
- Mathematical problem solving
- Logical reasoning
- Analytical tasks
- Step-by-step explanations
- Complex reasoning patterns
"""

import dspy
import math
from typing import List, Dict, Any, Union

# Example 1: Mathematical Problem Solving
def demonstrate_mathematical_reasoning():
    """Demonstrate step-by-step mathematical problem solving."""

    print("=" * 60)
    print("Example 1: Mathematical Problem Solving")
    print("=" * 60)

    # Define a detailed math problem signature
    class MathSolver(dspy.Signature):
        """Solve mathematical problems with step-by-step reasoning."""
        problem = dspy.InputField(desc="Mathematical problem to solve", type=str)
        given_info = dspy.OutputField(desc="Information given in the problem", type=str)
        approach = dspy.OutputField(desc="Mathematical approach to solve", type=str)
        calculations = dspy.OutputField(desc="Step-by-step calculations", type=str)
        answer = dspy.OutputField(desc="Final answer", type=str)
        verification = dspy.OutputField(desc="Check the answer", type=str)

    # Create examples showing good reasoning
    math_examples = [
        dspy.Example(
            problem="A rope is 12 meters long and cut into 3 equal pieces. Each piece is then cut in half. How many pieces total?",
            given_info="Initial rope length: 12 meters, Number of first cuts: 3, Each piece is cut in half",
            approach="1. Start with 1 rope\n2. First cutting: 1 rope → 3 pieces\n3. Second cutting: Each piece → 2 pieces",
            calculations="3 pieces × 2 = 6 pieces total",
            answer="6 pieces",
            verification="Check: 6 pieces × (12m/6) = 12m ✓"
        ),
        dspy.Example(
            problem="John saves $300 per month. How much does he save in 2 years?",
            given_info="Monthly savings: $300, Time period: 2 years",
            approach="1. Calculate months in 2 years: 2 × 12 = 24 months\n2. Multiply monthly savings by number of months",
            calculations="24 months × $300/month = $7,200",
            answer="$7,200",
            verification="Check: $7,200 ÷ $300 = 24 months = 2 years ✓"
        )
    ]

    # Create solver with examples
    solver = dspy.ChainOfThought(MathSolver, demos=math_examples)

    # Test with complex problems
    problems = [
        "A store sells t-shirts for $20 each. They offer a buy 3 get 1 free deal. How much do 12 t-shirts cost?",
        "If I drive 60 miles at 30 mph, and the return trip takes 2 hours, how fast was I driving back?",
        "A tank has 1000 liters. It's losing water at 2% per hour. How much water remains after 5 hours?"
    ]

    print("\nMathematical Problem Solving:")
    print("-" * 40)

    for i, problem in enumerate(problems, 1):
        print(f"\nProblem {i}:")
        print(f"Problem: {problem}")
        result = solver(problem=problem)

        print(f"Given Info: {result.given_info}")
        print(f"Approach: {result.approach}")
        print(f"Calculations: {result.calculations}")
        print(f"Answer: {result.answer}")
        print(f"Verification: {result.verification}")
        print("-" * 20)

# Example 2: Logical Reasoning
def demonstrate_logical_reasoning():
    """Demonstrate logical puzzle solving with Chain of Thought."""

    print("\n" + "=" * 60)
    print("Example 2: Logical Reasoning")
    print("=" * 60)

    # Define logic puzzle signature
    class LogicPuzzleSolver(dspy.Signature):
        """Solve logic puzzles with step-by-step reasoning."""
        puzzle = dspy.InputField(desc="Logic puzzle to solve", type=str)
        known_facts = dspy.OutputField(desc="Facts given in the puzzle", type=str)
        logical_steps = dspy.OutputField(desc="Step-by-step logical deduction", type=str)
        assumptions = dspy.OutputField(desc="Assumptions made", type=str)
        conclusion = dspy.OutputField(desc="Logical conclusion", type=str)
        confidence = dspy.OutputField(desc="Confidence level", type=int)

    # Create examples for logical reasoning
    logic_examples = [
        dspy.Example(
            puzzle="Three friends: Alice, Bob, Carol. One is a doctor, one is a teacher, one is an engineer. Alice is not the doctor. The engineer is not Carol. Bob is not the teacher. Who is the engineer?",
            known_facts="3 people with 3 professions. Alice ≠ doctor. Engineer ≠ Carol. Bob ≠ teacher.",
            logical_steps="1. Since Alice ≠ doctor, Alice is either teacher or engineer\n2. Since Bob ≠ teacher, Bob is either doctor or engineer\n3. Since engineer ≠ Carol, Carol is not the engineer\n4. Alice could be teacher or engineer\n5. If Alice were teacher, then Bob would be doctor and Carol engineer, but Bob ≠ teacher doesn't prevent this",
            assumptions="Each person has exactly one profession",
            conclusion="We need more information to uniquely determine who is the engineer",
            confidence=3
        )
    ]

    # Create solver
    solver = dspy.ChainOfThought(LogicPuzzleSolver, demos=logic_examples)

    # Test with complex logic puzzles
    puzzles = [
        "In a race, you pass the person in 2nd place. What place are you in?",
        """
        A farmer has a fox, a chicken, and a bag of grain. He needs to cross a river with a small boat.
        He can only take one item at a time. The fox cannot be left alone with the chicken,
        and the chicken cannot be left alone with the grain. How does he get everything across?
        """,
        """
        Four houses painted different colors: red, blue, green, yellow.
        The red house is not next to the blue house.
        The green house is between the red and yellow houses.
        The yellow house is on the end.
        Which houses are next to each other?
        """
    ]

    print("\nLogical Reasoning:")
    print("-" * 40)

    for i, puzzle in enumerate(puzzles, 1):
        print(f"\nPuzzle {i}:")
        print(f"Puzzle: {puzzle}")
        result = solver(puzzle=puzzle)

        print(f"Facts: {result.known_facts}")
        print(f"Steps: {result.logical_steps}")
        print(f"Conclusion: {result.conclusion}")
        print(f"Confidence: {result.confidence}/10")
        print("-" * 20)

# Example 3: Analytical Reasoning
def demonstrate_analytical_reasoning():
    """Demonstrate data analysis and pattern recognition."""

    print("\n" + "=" * 60)
    print("Example 3: Analytical Reasoning")
    print("=" * 60)

    # Define analysis signature
    class DataAnalyzer(dspy.Signature):
        """Analyze data patterns and draw insights."""
        data = dspy.InputField(desc="Data to analyze", type=str)
        observations = dspy.OutputField(desc="Key observations from data", type=str)
        patterns = dspy.OutputField(desc="Patterns or trends identified", type=str)
        insights = dspy.OutputField(desc="Deep insights from analysis", type=str)
        limitations = dspy.OutputField(desc="Limitations of analysis", type=str)
        recommendations = dspy.OutputField(desc="Actionable recommendations", type=str)

    # Create analyzer
    analyzer = dspy.ChainOfThought(DataAnalyzer)

    # Test with different data types
    data_sets = [
        {
            "type": "Sales Data",
            "data": """
            Quarterly Sales:
            Q1: $100k, Q2: $120k, Q3: $110k, Q4: $150k
            Products: Electronics 40%, Clothing 30%, Home 20%, Other 10%
            Customer Types: New 30%, Returning 70%
            """,
            "focus": "sales_trends"
        },
        {
            "type": "Temperature Data",
            "data": """
            Daily Temperature (°C):
            Mon: 18, Tue: 20, Wed: 22, Thu: 25, Fri: 24, Sat: 21, Sun: 19
            Weekly Average: 20.4
            Max: 25 on Thursday, Min: 18 on Monday
            """,
            "focus": "weather_patterns"
        }
    ]

    print("\nData Analysis:")
    print("-" * 40)

    for data_set in data_sets:
        print(f"\nAnalyzing {data_set['type']}:")
        result = analyzer(data=data_set["data"])

        print(f"Observations: {result.observations}")
        print(f"Patterns: {result.patterns}")
        print(f"Insights: {result.insights}")
        print(f"Limitations: {result.limitations}")
        print(f"Recommendations: {result.recommendations}")
        print("-" * 20)

# Example 4: Scientific Reasoning
def demonstrate_scientific_reasoning():
    """Demonstrate scientific method and hypothesis testing."""

    print("\n" + "=" * 60)
    print("Example 4: Scientific Reasoning")
    print("=" * 60)

    class HypothesisTester(dspy.Signature):
        """Test hypotheses using scientific reasoning."""
        observation = dspy.InputField(desc="Observation or question", type=str)
        hypothesis = dspy.OutputField(desc="Testable hypothesis", type=str)
        test_method = dspy.OutputField(desc="How to test the hypothesis", type=str)
        predictions = dspy.OutputField(desc="Expected outcomes", type=str)
        evaluation = dspy.OutputField(desc="Evaluation of hypothesis", type=str)
        conclusion = dspy.OutputField(desc="Scientific conclusion", type=str)

    # Create examples with scientific method
    science_examples = [
        dspy.Example(
            observation="Plants in the dark grow taller than those in the light",
            hypothesis="Plants grow taller in the dark because they're trying to reach light (etiolation)",
            test_method="1. Grow plants in dark with artificial light\n2. Grow plants in light with dark periods\n3. Measure growth rates",
            predictions="Dark plants should grow faster initially but may have weaker stems\nLight plants should grow slower but be stronger",
            evaluation="Compare growth rates and stem strength between groups",
            conclusion="Further experimentation needed to determine cause-effect relationship"
        )
    ]

    # Create tester
    tester = dspy.ChainOfThought(HypothesisTester, demos=science_examples)

    # Scientific observations to test
    observations = [
        "Students who study music tend to have better math scores",
        "Cities with more coffee shops per capita have higher startup rates",
        "Companies with 4-day work weeks report higher employee satisfaction"
    ]

    print("\nScientific Method:")
    print("-" * 40)

    for obs in observations:
        print(f"\nObservation: {obs}")
        result = tester(observation=obs)

        print(f"Hypothesis: {result.hypothesis}")
        print(f"Test Method: {result.test_method}")
        print(f"Predictions: {result.predictions}")
        print(f"Evaluation: {result.evaluation}")
        print(f"Conclusion: {result.conclusion}")
        print("-" * 20)

# Example 5: Ethical Reasoning
def demonstrate_ethical_reasoning():
    """Demonstrate ethical analysis and moral reasoning."""

    print("\n" + "=" * 60)
    print("Example 5: Ethical Reasoning")
    print("=" * 60)

    class EthicalAnalyzer(dspy.Signature):
        """Analyze ethical dilemmas and provide reasoned recommendations."""
        scenario = dspy.InputField(desc="Ethical scenario to analyze", type=str)
        stakeholders = dspy.OutputField(desc="Affected stakeholders", type=str)
        ethical_principles = dspy.OutputField(desc="Relevant ethical principles", type=str)
        pros_cons = dspy.OutputField(desc="Pros and cons analysis", type=str)
        alternative_approaches = dspy.OutputField(desc="Alternative approaches", type=str)
        recommendation = dspy.OutputField(desc="Ethical recommendation", type=str)
        reasoning = dspy.OutputField(desc="Ethical reasoning process", type=str)

    analyzer = dspy.ChainOfThought(EthicalAnalyzer)

    ethical_scenarios = [
        {
            "title": "Autonomous Vehicle Ethics",
            "scenario": "A self-driving car must choose between hitting a pedestrian or swerving into a wall that will injure passengers. There is no time to brake safely."
        },
        {
            "title": "Data Privacy vs Public Health",
            "scenario": "A city has collected smartphone location data to track disease spread. Should this data be shared with researchers to potentially save lives?"
        },
        {
            "title": "AI Decision Making",
            "scenario": "An AI system can make optimal resource allocation decisions that would maximize efficiency but might displace some workers. Should it be deployed?"
        }
    ]

    print("\nEthical Analysis:")
    print("-" * 40)

    for scenario in ethical_scenarios:
        print(f"\n{scenario['title']}:")
        print(f"Scenario: {scenario['scenario']}")
        result = analyzer(scenario=scenario['scenario'])

        print(f"Stakeholders: {result.stakeholders}")
        print(f"Principles: {result.ethical_principles}")
        print(f"Pros/Cons: {result.pros_cons}")
        print(f"Alternatives: {result.alternative_approaches}")
        print(f"Recommendation: {result.recommendation}")
        print("-" * 20)

# Example 6: Creative Problem Solving
def demonstrate_creative_reasoning():
    """Demonstrate creative and innovative problem solving."""

    print("\n" + "=" * 60)
    print("Example 6: Creative Problem Solving")
    print("=" * 60)

    class CreativeSolver(dspy.Signature):
        """Solve problems with creative approaches."""
        problem = dspy.InputField(desc="Problem that requires creative solution", type=str)
        brainstorming = dspy.OutputField(desc="Initial brainstorming session", type=str)
        constraints = dspy.OutputField(desc="Constraints and limitations", type=str)
        approaches = dspy.OutputField(desc="Multiple solution approaches", type=str)
        evaluation = dspy.OutputField(desc="Approach evaluation", type=str)
        recommended_solution = dspy.OutputField(desc="Recommended solution", type=str)
        innovation = dspy.OutputField(desc="Innovative aspects of solution", type=str)

    solver = dspy.ChainOfThought(CreativeSolver, temperature=0.8)

    creative_problems = [
        "How to reduce plastic waste in a city of 1 million people with limited budget?",
        "Design a social media platform that promotes well-being and reduces addiction",
        "Create a new educational system that adapts to each student's learning style",
        "Solve traffic congestion in a growing city using innovative approaches"
    ]

    print("\nCreative Problem Solving:")
    print("-" * 40)

    for i, problem in enumerate(creative_problems, 1):
        print(f"\nProblem {i}:")
        print(f"Problem: {problem}")
        result = solver(problem=problem)

        print(f"Brainstorming: {result.brainstorming}")
        print(f"Constraints: {result.constraints}")
        print(f"Approaches: {result.approaches}")
        print(f"Innovation: {result.innovation}")
        print(f"Recommended: {result.recommended_solution}")
        print("-" * 20)

# Example 7: Diagnostic Reasoning
def demonstrate_diagnostic_reasoning():
    """Demonstrate step-by-step diagnostic reasoning."""

    print("\n" + "=" * 60)
    print("Example 7: Diagnostic Reasoning")
    print("=" * 60)

    class DiagnosticSystem(dspy.Signature):
        """Diagnose and solve technical problems systematically."""
        symptoms = dspy.InputField(desc="System symptoms", type=str)
        possible_causes = dspy.OutputField(desc="List of potential causes", type=str)
        tests = dspy.OutputField(desc="Diagnostic tests to run", type=str)
        test_results = dspy.OutputField(desc="Results from diagnostic tests", type=str)
        root_cause = dspy.OutputField(desc="Identified root cause", type=str)
        solution = dspy.OutputField(desc="Step-by-step solution", type=str)
        verification = dspy.OutputField(desc="Verification of fix", type=str)

    # Create diagnostic examples
    diagnostic_examples = [
        dspy.Example(
            symptoms="Website loads slowly, shows 503 errors during peak traffic",
            possible_causes="1. Database server is overloaded\n2. Code has memory leak\n3. Network bandwidth is saturated\n4. Third-party API calls blocking",
            tests="1. Check database performance metrics\n2. Profile application memory usage\n3. Monitor network usage during peak\n4. Check API response times",
            test_results="1. Database response time spikes during peak\n2. Memory usage stays normal\n3. Network usage at 90% capacity\n4. API calls occasionally timeout",
            root_cause="Network bandwidth saturation during peak traffic",
            solution="1. Implement CDN for static resources\n2. Optimize database queries and add indexes\n3. Implement API response caching\n4. Load balance across multiple servers",
            verification="Monitor website speed after CDN implementation"
        )
    ]

    # Note: The ServerError example would be a proper dspy.Example in practice

    # Create diagnostic system
    diagnostic_system = dspy.ChainOfThought(DiagnosticSystem)

    # Test with technical problems
    technical_problems = [
        "My Python script crashes with MemoryError when processing large datasets",
        "The API rate limit is hit when making 1000 requests per minute",
        "The JSON serialization is failing for nested objects with circular references"
    ]

    print("\nDiagnostic Reasoning:")
    print("-" * 40)

    for i, symptoms in enumerate(technical_problems, 1):
        print(f"\nProblem {i}:")
        print(f"Symptoms: {symptoms}")
        result = diagnostic_system(symptoms=symptoms)

        print(f"Causes: {result.possible_causes}")
        print(f"Tests: {result.tests}")
        print(f"Root Cause: {result.root_cause}")
        print(f"Solution: {result.solution}")
        print(f"Verification: {result.verification}")
        print("-" * 20)

# Main execution function
def run_all_examples():
    """Run all Chain of Thought examples."""

    print("DSPy Chain of Thought Examples")
    print("These examples demonstrate various reasoning capabilities.")
    print("=" * 60)

    try:
        demonstrate_mathematical_reasoning()
        demonstrate_logical_reasoning()
        demonstrate_analytical_reasoning()
        demonstrate_scientific_reasoning()
        demonstrate_ethical_reasoning()
        demonstrate_creative_reasoning()
        demonstrate_diagnostic_reasoning()

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_examples()
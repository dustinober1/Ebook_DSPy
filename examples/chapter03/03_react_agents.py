"""
ReAct Agent Module Examples

This file demonstrates DSPy's ReAct (Reason+Act) agent capabilities:
- Tool usage and function calling
- Multi-step problem solving
- Information gathering and synthesis
- Complex task decomposition
- Interactive agent workflows
"""

import dspy
import json
import math
from typing import List, Dict, Any, Union, Optional
from datetime import datetime, timedelta

# Mock tools for demonstration
class CalculatorTool:
    """Simple calculator tool for arithmetic operations."""

    def calculate(self, expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            # Safe evaluation of basic math expressions
            allowed_names = {
                "abs": abs, "min": min, "max": max, "round": round,
                "pow": pow, "sqrt": math.sqrt, "log": math.log,
                "sin": math.sin, "cos": math.cos, "tan": math.tan
            }
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

class WeatherTool:
    """Mock weather information tool."""

    def get_weather(self, city: str, date: str = None) -> str:
        """Get weather information for a city."""
        # Mock weather data
        weather_data = {
            "New York": {"temp": "22°C", "condition": "Partly cloudy", "humidity": "65%"},
            "London": {"temp": "18°C", "condition": "Rainy", "humidity": "80%"},
            "Tokyo": {"temp": "26°C", "condition": "Sunny", "humidity": "55%"},
            "Sydney": {"temp": "15°C", "condition": "Windy", "humidity": "70%"}
        }

        data = weather_data.get(city, {"temp": "Unknown", "condition": "Unknown", "humidity": "Unknown"})
        return f"Weather in {city}: {data['temp']}, {data['condition']}, Humidity: {data['humidity']}"

class SearchTool:
    """Mock search tool for information retrieval."""

    def search(self, query: str) -> str:
        """Search for information."""
        # Mock search results
        search_results = {
            "Python population": "As of 2024, Montenegro has a population of approximately 620,000 people.",
            "Python programming": "Python is a high-level programming language created by Guido van Rossum.",
            "Eiffel Tower height": "The Eiffel Tower is 330 meters tall (1,083 feet) including antennas.",
            "Mars distance": "The distance from Earth to Mars varies from 55 to 400 million kilometers."
        }

        for key, value in search_results.items():
            if key.lower() in query.lower():
                return value

        return f"No specific information found for: {query}"

class ReActAgentWithTools:
    """Example of a ReAct agent with multiple tools."""

    def __init__(self):
        # Initialize tools
        self.calculator = CalculatorTool()
        self.weather = WeatherTool()
        self.search = SearchTool()

        # Tool registry for the agent
        self.tools = {
            "calculate": {
                "description": "Calculate mathematical expressions",
                "function": self.calculator.calculate
            },
            "weather": {
                "description": "Get weather information for a city",
                "function": self.weather.get_weather
            },
            "search": {
                "description": "Search for information on the internet",
                "function": self.search.search
            }
        }

    def create_react_signature(self):
        """Create a signature for the ReAct agent."""
        class ReActSignature(dspy.Signature):
            """Think step by step and use tools to answer questions."""
            question = dspy.InputField(desc="User's question", type=str)
            thought = dspy.OutputField(desc="Current thinking process", type=str)
            action = dspy.OutputField(desc="Tool to use (if needed)", type=str)
            action_input = dspy.OutputField(desc="Input for the tool", type=str)
            observation = dspy.OutputField(desc="Result from tool", type=str)
            answer = dspy.OutputField(desc="Final answer", type=str)

        return ReActSignature

    def execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Execute a tool and return the result."""
        if tool_name in self.tools:
            try:
                return self.tools[tool_name]["function"](tool_input)
            except Exception as e:
                return f"Tool execution error: {str(e)}"
        else:
            return f"Unknown tool: {tool_name}"

# Example 1: Mathematical Problem Solving with Tools
def demonstrate_mathematical_react():
    """Demonstrate ReAct agent solving math problems."""

    print("=" * 60)
    print("Example 1: Mathematical Problem Solving with ReAct")
    print("=" * 60)

    # Create agent
    agent = ReActAgentWithTools()
    signature = agent.create_react_signature()

    # Create few-shot examples
    math_examples = [
        dspy.Example(
            question="What is 15% of 240?",
            thought="I need to calculate 15% of 240. I'll use the calculator tool.",
            action="calculate",
            action_input="240 * 0.15",
            observation="36.0",
            answer="15% of 240 is 36."
        ),
        dspy.Example(
            question="What's the area of a circle with radius 5?",
            thought="I need to calculate π * r^2 where r=5. I'll use the calculator.",
            action="calculate",
            action_input="math.pi * 5 * 5",
            observation="78.53981633974483",
            answer="The area of a circle with radius 5 is approximately 78.54 square units."
        )
    ]

    # Create ReAct module with examples
    math_solver = dspy.React(signature=signature, tools=agent.tools, demos=math_examples)

    # Test problems
    problems = [
        "A company's revenue grew from $1M to $1.5M in one year. What is the percentage growth?",
        "If I invest $1000 at 5% annual compound interest for 10 years, how much will I have?",
        "A rectangular room is 12m by 8m. How much flooring is needed for the entire room?",
        "What's the volume of a sphere with radius 3 units?"
    ]

    print("\nMathematical Problem Solving:")
    print("-" * 40)

    for i, problem in enumerate(problems, 1):
        print(f"\nProblem {i}:")
        print(f"Question: {problem}")

        # Simulate ReAct steps
        try:
            result = math_solver(question=problem)
            print(f"Thought: {result.thought}")
            print(f"Action: {result.action}")
            if result.action:
                observation = agent.execute_tool(result.action, result.action_input)
                print(f"Observation: {observation}")
            print(f"Answer: {result.answer}")
        except Exception as e:
            print(f"Error: {e}")
            # Provide fallback solution
            if "percentage" in problem.lower():
                print("Answer: The percentage growth is 50%.")
            elif "invest" in problem.lower():
                print("Answer: You'll have approximately $1,628.89 after 10 years.")
            elif "flooring" in problem.lower():
                print("Answer: You need 96 square meters of flooring.")
            elif "sphere" in problem.lower():
                print("Answer: The volume is approximately 113.1 cubic units.")

        print("-" * 20)

# Example 2: Information Gathering and Synthesis
def demonstrate_information_react():
    """Demonstrate ReAct agent for research tasks."""

    print("\n" + "=" * 60)
    print("Example 2: Information Gathering and Synthesis")
    print("=" * 60)

    # Create agent with search capabilities
    agent = ReActAgentWithTools()

    class ResearchSignature(dspy.Signature):
        """Research agent for information gathering."""
        query = dspy.InputField(desc="Research query", type=str)
        research_steps = dspy.OutputField(desc="Step-by-step research process", type=str)
        findings = dspy.OutputField(desc="Key findings from research", type=str)
        synthesis = dspy.OutputField(desc="Synthesized answer", type=str)
        sources = dspy.OutputField(desc="Source information", type=str)

    # Create research examples
    research_examples = [
        dspy.Example(
            query="What is the population of Montenegro?",
            research_steps="1. Search for Montenegro population data\n2. Verify the information",
            findings="Montenegro has approximately 620,000 people as of 2024",
            synthesis="Montenegro is a small European country with a population of about 620,000 people.",
            sources="Population statistics database"
        )
    ]

    research_agent = dspy.React(signature=ResearchSignature, tools=agent.tools, demos=research_examples)

    # Research queries
    queries = [
        "How tall is the Eiffel Tower and when was it built?",
        "What's the current weather in London and what's typical for this season?",
        "Find information about Python programming language history"
    ]

    print("\nResearch Tasks:")
    print("-" * 40)

    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")

        try:
            result = research_agent(query=query)
            print(f"Research Steps: {result.research_steps}")
            print(f"Findings: {result.findings}")
            print(f"Answer: {result.synthesis}")
            print(f"Sources: {result.sources}")
        except Exception as e:
            print(f"Error: {e}")
            # Provide fallback responses
            if "Eiffel Tower" in query:
                print("Answer: The Eiffel Tower is 330 meters tall and was built in 1889.")
            elif "London" in query:
                print("Answer: Current weather in London: 18°C, Rainy. This is typical for London's weather.")
            elif "Python" in query:
                print("Answer: Python was created by Guido van Rossum and first released in 1991.")

        print("-" * 20)

# Example 3: Multi-Tool Problem Solving
def demonstrate_multi_tool_react():
    """Demonstrate ReAct agent using multiple tools together."""

    print("\n" + "=" * 60)
    print("Example 3: Multi-Tool Problem Solving")
    print("=" * 60)

    agent = ReActAgentWithTools()

    class ComplexTaskSignature(dspy.Signature):
        """Agent for complex tasks requiring multiple tools."""
        task = dspy.InputField(desc="Complex task description", type=str)
        plan = dspy.OutputField(desc="Step-by-step plan", type=str)
        execution = dspy.OutputField(desc="Execution steps and results", type=str)
        final_result = dspy.OutputField(desc="Final answer", type=str)

    complex_agent = dspy.React(signature=ComplexTaskSignature, tools=agent.tools)

    # Complex tasks
    tasks = [
        "Calculate the total travel distance from New York to London to Tokyo. Also check the weather in each city.",
        "If I save $1000 monthly at 5% interest, how much will I have in 5 years? Also search for investment tips.",
        "Compare the areas of a circle with radius 10 and a square with side length 18."
    ]

    print("\nComplex Multi-Tool Tasks:")
    print("-" * 40)

    for i, task in enumerate(tasks, 1):
        print(f"\nTask {i}: {task}")

        # Simulate multi-step process
        print("\nExecution Plan:")

        if "travel distance" in task.lower():
            print("1. Search for distances between cities")
            print("2. Calculate total distance")
            print("3. Get weather for each city")

            # Execute steps
            distance1 = "5570 km"  # Mock NY to London
            distance2 = "9560 km"  # Mock London to Tokyo
            total = str(5570 + 9560) + " km"

            weather_ny = agent.execute_tool("weather", "New York")
            weather_london = agent.execute_tool("weather", "London")
            weather_tokyo = agent.execute_tool("weather", "Tokyo")

            print(f"\nExecution Results:")
            print(f"NY to London: {distance1}")
            print(f"London to Tokyo: {distance2}")
            print(f"Total distance: {total}")
            print(f"\nWeather:")
            print(f"- {weather_ny}")
            print(f"- {weather_london}")
            print(f"- {weather_tokyo}")

        elif "save $1000" in task.lower():
            print("1. Calculate compound interest for 5 years")
            print("2. Search for investment tips")

            future_value = agent.execute_tool("calculate", "1000 * 12 * 5 * 1.05**5")
            tips = agent.execute_tool("search", "investment tips diversification")

            print(f"\nExecution Results:")
            print(f"Future value: ~${future_value}")
            print(f"Investment tips: {tips}")

        elif "compare areas" in task.lower():
            print("1. Calculate area of circle (π × r²)")
            print("2. Calculate area of square (s²)")
            print("3. Compare the results")

            circle_area = agent.execute_tool("calculate", "math.pi * 10 * 10")
            square_area = agent.execute_tool("calculate", "18 * 18")

            print(f"\nExecution Results:")
            print(f"Circle area: ~{circle_area} sq units")
            print(f"Square area: {square_area} sq units")
            print(f"Winner: {'Square' if float(square_area) > float(circle_area) else 'Circle'}")

        print("-" * 20)

# Example 4: Interactive Agent Workflow
def demonstrate_interactive_react():
    """Demonstrate interactive ReAct agent with user feedback."""

    print("\n" + "=" * 60)
    print("Example 4: Interactive Agent Workflow")
    print("=" * 60)

    class InteractiveSignature(dspy.Signature):
        """Interactive agent that can ask clarifying questions."""
        user_request = dspy.InputField(desc="User's initial request", type=str)
        clarification_needed = dspy.OutputField(desc="Questions for user", type=str)
        action_taken = dspy.OutputField(desc("Actions performed", type=str)
        result = dspy.OutputField(desc("Result of actions", type=str)
        follow_up = dspy.OutputField(desc("Follow-up suggestions", type=str)

    interactive_agent = dspy.React(signature=InteractiveSignature, tools=ReActAgentWithTools().tools)

    # Scenarios that might need clarification
    scenarios = [
        "Help me plan a vacation",
        "I need to solve a math problem",
        "Tell me about the weather"
    ]

    print("\nInteractive Scenarios:")
    print("-" * 40)

    for scenario in scenarios:
        print(f"\nUser: {scenario}")

        # Simulate interaction
        if "vacation" in scenario.lower():
            clarifications = [
                "Where would you like to go?",
                "When are you planning to travel?",
                "What's your budget range?"
            ]
            print(f"\nAgent Questions: {clarifications[0]}")
            print(f"Agent: Once you tell me the destination, I can check the weather and help calculate costs!")

        elif "math problem" in scenario.lower():
            clarifications = [
                "What specific math problem do you need help with?",
                "Is it algebra, geometry, or calculus?",
                "Do you need just the answer or the steps too?"
            ]
            print(f"\nAgent Questions: {clarifications[0]}")
            print(f"Agent: I have tools to calculate complex expressions and show step-by-step solutions.")

        elif "weather" in scenario.lower():
            clarifications = [
                "Which city's weather would you like to know?",
                "Do you need current conditions or a forecast?"
            ]
            print(f"\nAgent Questions: {clarifications[0]}")
            print(f"Agent: I can get weather information for most major cities.")

        print("-" * 20)

# Example 5: Tool Error Handling
def demonstrate_react_error_handling():
    """Demonstrate how ReAct agents handle tool errors."""

    print("\n" + "=" * 60)
    print("Example 5: Tool Error Handling")
    print("=" * 60)

    agent = ReActAgentWithTools()

    # Test error scenarios
    error_scenarios = [
        ("calculate", "2 / 0"),  # Division by zero
        ("calculate", "invalid expression"),  # Invalid syntax
        ("weather", "Nonexistent City"),  # Unknown city
        ("search", ""),  # Empty query
        ("unknown_tool", "test")  # Unknown tool
    ]

    print("\nError Handling Tests:")
    print("-" * 40)

    for tool, input_val in error_scenarios:
        print(f"\nTesting {tool} with input: {repr(input_val)}")
        result = agent.execute_tool(tool, input_val)
        print(f"Result: {result}")
        print("-" * 20)

# Example 6: Custom Tool Integration
def demonstrate_custom_tools():
    """Demonstrate creating and integrating custom tools."""

    print("\n" + "=" * 60)
    print("Example 6: Custom Tool Integration")
    print("=" * 60)

    class DateTool:
        """Tool for date calculations."""

        def days_until(self, date_str: str) -> str:
            """Calculate days until a specific date."""
            try:
                target_date = datetime.strptime(date_str, "%Y-%m-%d")
                today = datetime.now()
                delta = target_date - today
                return f"{delta.days} days until {date_str}"
            except:
                return f"Invalid date format. Use YYYY-MM-DD"

    class TextTool:
        """Tool for text processing."""

        def word_count(self, text: str) -> str:
            """Count words in text."""
            return f"Word count: {len(text.split())}"

        def sentiment_score(self, text: str) -> str:
            """Mock sentiment analysis."""
            positive_words = ["good", "great", "amazing", "excellent", "love"]
            negative_words = ["bad", "terrible", "awful", "hate", "worst"]

            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)

            if pos_count > neg_count:
                return "Sentiment: Positive"
            elif neg_count > pos_count:
                return "Sentiment: Negative"
            else:
                return "Sentiment: Neutral"

    # Create agent with custom tools
    class CustomToolAgent:
        def __init__(self):
            self.date_tool = DateTool()
            self.text_tool = TextTool()

            self.custom_tools = {
                "days_until": {
                    "description": "Calculate days until a date (YYYY-MM-DD format)",
                    "function": self.date_tool.days_until
                },
                "word_count": {
                    "description": "Count words in text",
                    "function": self.text_tool.word_count
                },
                "sentiment": {
                    "description": "Analyze text sentiment",
                    "function": self.text_tool.sentiment_score
                }
            }

    custom_agent = CustomToolAgent()

    # Test custom tools
    test_cases = [
        ("days_until", "2024-12-25"),
        ("word_count", "This is a sample text with several words."),
        ("sentiment", "I absolutely love this amazing product!")
    ]

    print("\nCustom Tool Tests:")
    print("-" * 40)

    for tool_name, input_val in test_cases:
        print(f"\nTool: {tool_name}")
        print(f"Input: {input_val}")
        result = custom_agent.custom_tools[tool_name]["function"](input_val)
        print(f"Output: {result}")
        print("-" * 20)

# Main execution
def run_all_examples():
    """Run all ReAct agent examples."""

    print("DSPy ReAct Agent Examples")
    print("These examples demonstrate tool-using agents with reasoning capabilities.")
    print("=" * 60)

    try:
        demonstrate_mathematical_react()
        demonstrate_information_react()
        demonstrate_multi_tool_react()
        demonstrate_interactive_react()
        demonstrate_react_error_handling()
        demonstrate_custom_tools()

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_examples()
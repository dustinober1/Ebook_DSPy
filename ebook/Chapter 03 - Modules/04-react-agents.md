# ReAct Agents

## Prerequisites

- **Previous Section**: [Chain of Thought Module](./03-chainofthought.md) - Understanding of reasoning modules
- **Chapter 2**: Signatures - Familiarity with signature design
- **Required Knowledge**: Concept of agents and tool usage
- **Difficulty Level**: Intermediate to Advanced
- **Estimated Reading Time**: 45 minutes

## Learning Objectives

By the end of this section, you will:
- Master the `dspy.ReAct` module for building tool-using agents
- Understand the ReAct (Reasoning and Acting) paradigm
- Learn to integrate external tools and APIs with DSPy
- Design agents that can search, calculate, and interact with systems
- Build sophisticated agentic workflows for complex tasks

## Introduction to ReAct

ReAct (Reasoning and Acting) is a paradigm that enables language models to use external tools by interleaving reasoning traces with task-specific actions. Unlike simple Chain of Thought where the model only "thinks," ReAct agents can both think and act.

### The ReAct Cycle

```
Think (Reason) → Act (Use Tool) → Observe (Get Result) → Think (Reason) → ...
```

This cycle allows agents to:
1. **Reason** about what to do next
2. **Act** by calling external tools
3. **Observe** the results of those actions
4. **Reason** about the observations
5. **Repeat** until the task is complete

### Why ReAct Matters

**Traditional LLM Limitation:**
```
Q: What's the current stock price of Apple?
A: I don't have access to real-time data.
```

**With ReAct:**
```
Think: I need to find the current stock price of Apple. I should use a stock price tool.
Act: Search for "AAPL stock price"
Observe: Apple (AAPL) is trading at $178.52
Think: I found the stock price. I can now answer the user.
Answer: Apple (AAPL) is currently trading at $178.52.
```

## Basic ReAct Implementation

### Simple ReAct Example
```python
import dspy

# Define a ReAct signature
class BasicReAct(dspy.Signature):
    """Use tools to answer questions."""
    question = dspy.InputField(desc="Question to answer", type=str)
    reasoning = dspy.OutputField(desc="Step-by-step reasoning", type=str)
    answer = dspy.OutputField(desc="Final answer", type=str)

# Create a ReAct module with tools
agent = dspy.ReAct(BasicReAct, tools=[dspy.WebSearch()])

# Use the agent
result = agent(
    question="What was the last movie directed by Christopher Nolan?"
)

print("Reasoning:")
print(result.reasoning)
print("\nAnswer:")
print(result.answer)
```

### The ReAct Workflow
```python
# ReAct automatically generates a trace like:
"""
Thought 1: I need to find information about Christopher Nolan's latest movie.
Action 1: Search[Christopher Nolan latest movie 2023 2024]
Observation 1: Oppenheimer (2023) is Christopher Nolan's most recent film.
Thought 2: I have found that Oppenheimer is his latest movie. I can now answer.
Action 2: Finish[Oppenheimer (2023) is Christopher Nolan's most recent film.]
"""
```

## Built-in Tools

### 1. Web Search
```python
# Web search tool
search_tool = dspy.WebSearch()

# Create ReAct agent with search
researcher = dspy.ReAct(
    "query -> reasoning, answer",
    tools=[search_tool]
)

# Research a topic
result = researcher(
    query="What are the main advantages of quantum computing?"
)
```

### 2. Calculator
```python
# Calculator tool
calc_tool = dspy.Calculator()

# Create math agent
math_agent = dspy.ReAct(
    "math_problem -> reasoning, solution",
    tools=[calc_tool]
)

# Solve complex math
result = math_agent(
    math_problem="Calculate the compound interest on $10,000 at 5% for 3 years."
)
```

### 3. Multiple Tools
```python
# Combine multiple tools
agent = dspy.ReAct(
    "complex_query -> reasoning, detailed_answer",
    tools=[
        dspy.WebSearch(),
        dspy.Calculator(),
        dspy.ProgramInterpreter()  # For code execution
    ]
)

# Query requiring multiple tools
result = agent(
    complex_query="What's the population of Tokyo, and what's the per capita GDP "
                   "if the GDP is $2 trillion?"
)
```

## Custom Tools

### Creating Your Own Tool
```python
from dspy.predict.react import Tool

class WeatherTool(Tool):
    name = "weather"
    description = "Get current weather for a location"
    parameters = {
        "location": "The city name",
        "units": "Temperature units (celsius or fahrenheit, default: celsius)"
    }

    def forward(self, location, units="celsius"):
        """Simulate weather API call."""
        # In reality, this would call a real weather API
        import random
        temp = random.uniform(-10, 35) if units == "celsius" else random.uniform(14, 95)

        return f"Weather in {location}: {temp:.1f}°{units[0].upper()}, cloudy"

# Use custom tool
weather_agent = dspy.ReAct(
    "weather_question -> reasoning, weather_info",
    tools=[WeatherTool()]
)

result = weather_agent(
    weather_question="What's the weather like in London?"
)
```

### API Integration Tool
```python
class APITool(Tool):
    """Template for API integration tools."""

    def __init__(self, api_config):
        super().__init__()
        self.api_config = api_config

    def make_api_call(self, endpoint, params=None):
        """Make API call with error handling."""
        import requests

        try:
            response = requests.get(
                f"{self.api_config['base_url']}/{endpoint}",
                params=params,
                headers=self.api_config.get('headers', {})
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return f"API Error: {str(e)}"

# Example: GitHub API tool
class GitHubTool(APITool):
    name = "github"
    description = "Search GitHub repositories"
    parameters = {
        "query": "Search query for repositories",
        "sort": "Sort order (stars, forks, updated)"
    }

    def __init__(self, github_token=None):
        config = {
            "base_url": "https://api.github.com",
            "headers": {"Authorization": f"token {github_token}"} if github_token else {}
        }
        super().__init__(config)

    def forward(self, query, sort="stars"):
        """Search GitHub repositories."""
        return self.make_api_call(
            "search/repositories",
            params={"q": query, "sort": sort, "per_page": 5}
        )

# Use GitHub tool
github_agent = dspy.ReAct(
    "github_search -> reasoning, repo_info",
    tools=[GitHubTool()]
)

result = github_agent(
    github_search="Find popular machine learning libraries on GitHub"
)
```

## Advanced ReAct Patterns

### 1. Multi-Step Research
```python
class ResearchAgent(dspy.Signature):
    """Conduct comprehensive research on a topic."""
    research_topic = dspy.InputField(desc="Topic to research", type=str)
    research_depth = dspy.InputField(desc="How deep to research", type=str)
    findings = dspy.OutputField(desc="Key findings from research", type=str)
    sources = dspy.OutputField(desc="Sources used", type=str)
    gaps = dspy.OutputField(desc="Information gaps identified", type=str)
    next_steps = dspy.OutputField(desc="Suggested further research", type=str)

# Enhanced research agent
researcher = dspy.ReAct(
    ResearchAgent,
    tools=[
        dspy.WebSearch(max_results=10),
        dspy.WebPageScraper(),  # Custom tool for extracting content
        dspy.ProgramInterpreter()  # For data analysis
    ],
    max_steps=10  # Allow more thinking/acting cycles
)

# Deep research
result = researcher(
    research_topic="The impact of AI on job markets",
    research_depth="comprehensive"
)

print(f"Key Findings: {result.findings}")
print(f"Sources: {result.sources}")
```

### 2. Data Analysis Agent
```python
class DataAnalysisAgent(dspy.Signature):
    """Analyze data and generate insights."""
    dataset_description = dspy.InputField(desc="Description of dataset", type=str)
    analysis_goal = dspy.InputField(desc="What to learn from data", type=str)
    data_exploration = dspy.OutputField(desc="Steps taken to explore data", type=str)
    insights = dspy.OutputField(desc="Key insights discovered", type=str)
    visualizations = dspy.OutputField(desc="Suggested visualizations", type=str)
    limitations = dspy.OutputField(desc="Analysis limitations", type=str)

# Data analysis agent with programming capability
data_analyst = dspy.ReAct(
    DataAnalysisAgent,
    tools=[
        dspy.ProgramInterpreter(),  # For Python execution
        dspy.Calculator(),
        dspy.FileOperation()  # Custom tool for file operations
    ]
)

result = data_analyst(
    dataset_description="Sales data with columns: date, product, region, amount",
    analysis_goal="Find top performing products and regions"
)
```

### 3. Decision Support Agent
```python
class DecisionAgent(dspy.Signature):
    """Help make informed decisions."""
    decision_context = dspy.InputField(desc("Context for the decision", type=str)
    options = dspy.InputField(desc("Available options", type=str)
    criteria = dspy.InputField(desc("Decision criteria", type=str)
    analysis = dspy.OutputField(desc("Analysis of each option", type=str)
    recommendation = dspy.OutputField(desc("Recommended choice", type=str)
    confidence = dspy.OutputField(desc("Confidence in recommendation", type=str)
    risks = dspy.OutputField(desc("Potential risks", type=str)

# Decision agent with research capability
decision_helper = dspy.ReAct(
    DecisionAgent,
    tools=[
        dspy.WebSearch(),  # Research options
        dspy.Calculator(),  # Calculate metrics
        dspy.ComparisonTool()  # Custom comparison tool
    ]
)

result = decision_helper(
    decision_context="Choosing a cloud provider for our startup",
    options="AWS, Google Cloud, Azure",
    criteria="Cost, scalability, ease of use, support"
)
```

## Building Complex Agent Workflows

### 1. Agent Orchestration
```python
class OrchestratorAgent(dspy.Signature):
    """Orchestrate multiple specialized agents."""
    task = dspy.InputField(desc("Complex task to complete", type=str)
    subtasks = dspy.OutputField(desc("Identified subtasks", type=str)
    agent_assignments = dspy.OutputField(desc("Which agent handles each subtask", type=str)
    coordination = dspy.OutputField(desc="How agents coordinate", type=str)
    final_result = dspy.OutputField(desc="Combined result from all agents", type=str)

# Specialized agents
researcher = dspy.ReAct(
    "research_query -> research_result",
    tools=[dspy.WebSearch()],
    max_steps=5
)

analyst = dspy.ReAct(
    "analysis_query -> analysis_result",
    tools=[dspy.ProgramInterpreter(), dspy.Calculator()],
    max_steps=5
)

writer = dspy.ReAct(
    "writing_task -> written_content",
    tools=[dspy.WebSearch()],  # For research while writing
    max_steps=3
)

# Orchestrator
orchestrator = dspy.ReAct(
    OrchestratorAgent,
    tools=[
        Tool(researcher),  # Sub-agents as tools
        Tool(analyst),
        Tool(writer)
    ]
)
```

### 2. Hierarchical Agents
```python
class ManagerAgent(dspy.Signature):
    """Manage and delegate to worker agents."""
    objective = dspy.InputField(desc("High-level objective", type=str)
    delegation_plan = dspy.OutputField(desc("How to delegate work", type=str)
    monitoring = dspy.OutputField(desc("How to monitor progress", type=str)
    integration = dspy.OutputField(desc("How to integrate results", type=str)
    final_report = dspy.OutputField(desc("Complete report", type=str)

# Worker agents with specific expertise
market_researcher = dspy.ReAct(
    "market_research_task -> market_insights",
    tools=[dspy.WebSearch(), dspy.DataAnalyzer()]
)

financial_analyst = dspy.ReAct(
    "financial_analysis_task -> financial_report",
    tools=[dspy.Calculator(), dspy.ExcelTool()]
)

# Manager orchestrates workers
manager = dspy.ReAct(
    ManagerAgent,
    tools=[
        Tool(market_researcher),
        Tool(financial_analyst)
    ],
    max_steps=15
)
```

## Tool Best Practices

### 1. Error Handling
```python
class RobustTool(Tool):
    """Tool with comprehensive error handling."""

    def forward(self, **kwargs):
        try:
            result = self.execute(**kwargs)
            return result
        except Exception as e:
            # Log error
            self.log_error(e)
            # Return helpful error message
            return f"Error in {self.name}: {str(e)}. Please check your inputs and try again."

    def log_error(self, error):
        """Log errors for debugging."""
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        print(f"[{timestamp}] {self.name} Error: {error}")
```

### 2. Input Validation
```python
class ValidatedTool(Tool):
    """Tool with input validation."""

    def validate_inputs(self, **kwargs):
        """Validate inputs before execution."""
        # Implement validation logic
        pass

    def forward(self, **kwargs):
        # Validate first
        if not self.validate_inputs(**kwargs):
            return "Invalid inputs provided"

        # Execute if valid
        return self.execute(**kwargs)
```

### 3. Caching Results
```python
class CachedTool(Tool):
    """Tool with caching capability."""

    def __init__(self, cache_ttl=3600):
        super().__init__()
        self.cache = {}
        self.cache_ttl = cache_ttl

    def forward(self, **kwargs):
        # Generate cache key
        cache_key = self.generate_cache_key(**kwargs)

        # Check cache
        if cache_key in self.cache:
            cached_result, timestamp = self.cache[cache_key]
            if self.is_cache_valid(timestamp):
                return cached_result

        # Execute and cache result
        result = self.execute(**kwargs)
        self.cache[cache_key] = (result, time.time())

        return result
```

## Performance Optimization

### 1. Efficient Tool Selection
```python
# Choose tools wisely
efficient_agent = dspy.ReAct(
    "query -> reasoning, answer",
    tools=[
        dspy.WebSearch(max_results=3),  # Limit results
        dspy.Calculator(),
        # Don't include unnecessary tools
    ],
    max_steps=5,  # Limit reasoning steps
    temperature=0.1  # More predictable
)
```

### 2. Parallel Tool Execution
```python
class ParallelReAct(dspy.ReAct):
    """ReAct that can execute tools in parallel when possible."""

    def plan_parallel_actions(self, reasoning):
        """Identify actions that can be executed in parallel."""
        # Analyze reasoning to find parallelizable actions
        pass

    def execute_parallel(self, actions):
        """Execute multiple tools simultaneously."""
        # Use threading or asyncio
        pass
```

### 3. Smart Caching
```python
# Cache at multiple levels
smart_agent = dspy.ReAct(
    "query -> reasoning, answer",
    tools=[
        dspy.CachedWebSearch(cache_file="search_cache.db"),
        dspy.CachedCalculator(cache_file="calc_cache.db")
    ],
    cache=True  # Enable module-level caching
)
```

## Common ReAct Patterns

### 1. Research → Synthesize
```python
# Pattern: Gather information, then synthesize
research_pattern = dspy.ReAct(
    "research_question -> reasoning, synthesis",
    tools=[dspy.WebSearch(max_results=5)],
    instructions="1. Search for information\n2. Analyze findings\n3. Synthesize into coherent answer"
)
```

### 2. Calculate → Interpret
```python
# Pattern: Perform calculations, then interpret
calculation_pattern = dspy.ReAct(
    "calculation_problem -> reasoning, interpretation",
    tools=[dspy.Calculator()],
    instructions="1. Identify calculations needed\n2. Perform calculations\n3. Interpret results in context"
)
```

### 3. Verify → Conclude
```python
# Pattern: Verify information, then conclude
verification_pattern = dspy.ReAct(
    "claim -> reasoning, verified_conclusion",
    tools=[dspy.WebSearch()],
    instructions="1. Break down claim into verifiable parts\n2. Search for evidence\n3. Verify each part\n4. Draw conclusion"
)
```

## Troubleshooting ReAct

### 1. Agent Gets Stuck
```python
# Add timeouts and step limits
bounded_agent = dspy.ReAct(
    "task -> reasoning, result",
    tools=[dspy.WebSearch()],
    max_steps=7,  # Limit reasoning steps
    timeout=30    # Time limit per step
)
```

### 2. Tool Not Used Correctly
```python
# Add clear tool instructions
guided_agent = dspy.ReAct(
    "task -> reasoning, result",
    tools=[dspy.Calculator()],
    instructions="When you need to calculate something, always use the calculator tool. "
                 "Show your calculation steps clearly."
)
```

### 3. Inconsistent Results
```python
# Add deterministic mode
deterministic_agent = dspy.ReAct(
    "task -> reasoning, result",
    tools=[dspy.WebSearch()],
    temperature=0.1,  # Lower temperature
    seed=42           # Fixed seed
)
```

## When to Use ReAct

### Use ReAct when:

1. **External information needed** - Requires web search, APIs
2. **Complex calculations** - Needs computational tools
3. **Multi-step tasks** - Tasks requiring multiple actions
4. **Real-time data** - Needs current information
5. **Interactive tasks** - Requires tool interaction

### Consider Predict or ChainOfThought when:

1. **Static knowledge** - Information already in the model
2. **Simple reasoning** - No external tools needed
3. **Fast response required** - ReAct adds latency
4. **Reliable knowledge** - Model's knowledge is sufficient

## Integration with Assertions

Combine ReAct agents with assertions for reliable tool usage and validated outputs:

### 1. Tool Usage Validation

Ensure agents use tools appropriately and effectively:

```python
import dspy

class ValidatedResearchAgent(dspy.Module):
    """Research agent with validated tool usage."""

    def __init__(self):
        super().__init__()
        self.react = dspy.ReAct("query -> research_findings")

    def forward(self, query):
        # Validate tool usage
        def validate_tool_usage(example, pred, trace=None):
            # Check if tools were actually used
            if not trace or 'tool_calls' not in str(trace):
                raise AssertionError("Must use search tools for research")

            # Check for sufficient tool interactions
            tool_calls = str(trace).count('Action:')
            if tool_calls < 2:
                raise AssertionError("Make multiple searches for comprehensive research")

            # Verify findings incorporate tool results
            if len(pred.research_findings) < 200:
                raise AssertionError("Research findings too brief - use more sources")

            return True

        # Apply assertion
        validated_react = dspy.Assert(
            self.react,
            validation_fn=validate_tool_usage,
            max_attempts=3,
            recovery_hint="Use search tools to gather information from multiple sources"
        )

        return validated_react(query=query)

# Use validated research agent
researcher = ValidatedResearchAgent()
result = researcher(query="Impact of AI on job markets in 2024")
```

### 2. Output Source Verification

Ensure agent outputs properly cite sources from tool usage:

```python
class SourceAwareAgent(dspy.Module):
    """Agent that must cite sources from tools."""

    def __init__(self):
        super().__init__()
        self.react = dspy.ReAct("question -> answer_with_sources")

    def forward(self, question):
        def validate_source_citation(example, pred, trace=None):
            answer = pred.answer_with_sources

            # Check for citations
            citation_patterns = ['Source:', '[', 'According to', 'Based on']
            has_citations = any(pattern in answer for pattern in citation_patterns)

            if not has_citations:
                raise AssertionError(
                    "Answer must include sources. Use patterns like 'Source: [URL]'"
                )

            # Extract citations and verify they match tool results
            if trace:
                # This would parse trace to match URLs with tool calls
                tool_urls = extract_tool_urls(trace)
                answer_urls = extract_citation_urls(answer)

                if not answer_urls:
                    raise AssertionError("No valid source citations found in answer")

                # Ensure at least one citation matches tool usage
                if not any(url in str(tool_urls) for url in answer_urls):
                    raise AssertionError("Citations must match tool search results")

            return True

        # Apply source validation
        with_sources = dspy.Assert(
            self.react,
            validation_fn=validate_source_citation,
            max_attempts=3
        )

        return with_sources(question=question)

def extract_tool_urls(trace):
    """Extract URLs from tool trace."""
    import re
    urls = []
    trace_str = str(trace)
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls.extend(re.findall(url_pattern, trace_str))
    return urls

def extract_citation_urls(text):
    """Extract URLs from citations in answer."""
    import re
    urls = []
    # Find URLs in brackets or after "Source:"
    bracket_pattern = r'\[(https?://[^\]]+)\]'
    source_pattern = r'Source:\s*(https?://[^\s]+)'

    urls.extend(re.findall(bracket_pattern, text))
    urls.extend(re.findall(source_pattern, text))
    return urls
```

### 3. Step-by-Step Action Validation

Validate the agent's reasoning and action sequence:

```python
class StepValidatedAgent(dspy.Module):
    """Agent with validated reasoning steps."""

    def __init__(self):
        super().__init__()
        self.react = dspy.ReAct("task -> solution")

    def forward(self, task):
        def validate_action_sequence(example, pred, trace=None):
            if not trace:
                return True  # No trace to validate

            # Parse the thought-action-observation sequence
            steps = parse_trace_steps(trace)

            # Check for minimum steps
            if len(steps) < 3:
                raise AssertionError("Need more reasoning steps - show your work")

            # Verify thought precedes each action
            for i, step in enumerate(steps):
                if 'Action:' in step and i > 0:
                    prev_step = steps[i-1]
                    if 'Thought:' not in prev_step:
                        raise AssertionError(
                            "Explain your reasoning (Thought:) before taking action"
                        )

            # Check if observations are used in subsequent thoughts
            for i, step in enumerate(steps):
                if 'Observation:' in step and i < len(steps) - 1:
                    next_step = steps[i+1]
                    # Simple check - in practice, this would be more sophisticated
                    if 'Thought:' in next_step and len(next_step) < 50:
                        raise AssertionError(
                            "Reflect on observations before proceeding"
                        )

            return True

        def parse_trace_steps(trace):
            """Parse trace into individual steps."""
            import re
            # Simple parsing - split by Thought/Action/Observation markers
            pattern = r'(Thought:|Action:|Observation:)'
            parts = re.split(pattern, str(trace))

            steps = []
            for i in range(1, len(parts), 2):
                if i < len(parts):
                    steps.append(parts[i] + parts[i+1])
            return steps

        # Apply step validation
        step_validated = dspy.Assert(
            self.react,
            validation_fn=validate_action_sequence,
            max_attempts=2,
            recovery_hint="Show clear Thought: Action: Observation: sequence"
        )

        return step_validated(task=task)
```

### 4. Error Recovery and Retry Logic

Build agents that recover from failures gracefully:

```python
class ResilientAgent(dspy.Module):
    """Agent with error recovery capabilities."""

    def __init__(self):
        super().__init__()
        self.react = dspy.ReAct("goal -> result")

    def forward(self, goal):
        def validate_completion(example, pred, trace=None):
            # Check if goal was actually achieved
            if not assess_goal_achievement(goal, pred.result):
                raise AssertionError(
                    "Goal not fully achieved. Review your approach and try alternative actions."
                )

            # Check for proper error handling in trace
            if trace and 'error' in str(trace).lower():
                # Should see recovery attempts after errors
                if 'Thought:' not in str(trace).split('error')[-1]:
                    raise AssertionError(
                        "After an error, show recovery reasoning before continuing"
                    )

            return True

        def assess_goal_achievement(goal, result):
            """Assess if the agent achieved its goal."""
            # Simple heuristic - could be more sophisticated
            goal_words = set(goal.lower().split())
            result_words = set(result.lower().split())

            # Check if key goal terms appear in result
            overlap = len(goal_words.intersection(result_words))
            return overlap > len(goal_words) * 0.3

        # Custom error handler for assertion failures
        def custom_error_handler(assertion_type, error_msg, attempt):
            """Provide specific recovery hints based on error type."""
            if "Goal not fully achieved" in error_msg:
                return """
                Review the original goal and your current result.
                Identify what's missing and plan specific actions to address gaps.
                Consider: What specific information or actions are still needed?
                """
            elif "error" in error_msg.lower():
                return """
                You encountered an error. Analyze what went wrong and choose:
                1. Try the same action with different parameters
                2. Use an alternative tool or approach
                3. Modify your strategy based on the error
                """
            else:
                return "Review your actions and ensure they address the goal."

        # Apply with custom error handling
        resilient_react = dspy.Assert(
            self.react,
            validation_fn=validate_completion,
            max_attempts=3,
            error_handler=custom_error_handler
        )

        return resilient_react(goal=goal)
```

### 5. Multi-Tool Coordination Validation

Ensure agents coordinate multiple tools effectively:

```python
class MultiToolAgent(dspy.Module):
    """Agent that must use multiple tools in coordination."""

    def __init__(self):
        super().__init__()
        self.react = dspy.ReAct(
            "analysis_request -> comprehensive_analysis",
            tools=[
                dspy.WebSearch(),      # For recent information
                dspy.Calculator(),     # For calculations
                CustomAPITool()        # Custom data source
            ]
        )

    def forward(self, analysis_request):
        def validate_tool_coordination(example, pred, trace=None):
            if not trace:
                return True

            # Check for usage of different tool types
            used_search = 'search' in str(trace).lower()
            used_calc = 'calculator' in str(trace).lower() or 'calculate' in str(trace).lower()
            used_api = 'api' in str(trace).lower()

            tool_count = sum([used_search, used_calc, used_api])

            # Require at least 2 different tools for comprehensive analysis
            if tool_count < 2:
                raise AssertionError(
                    "Use multiple tools (search, calculator, API) for comprehensive analysis"
                )

            # Validate tool use sequence makes sense
            if used_calc and not used_search:
                # If doing calculations, should have data first
                if 'Action:' in str(trace).split('calculator')[0]:
                    raise AssertionError(
                        "Gather data with search before performing calculations"
                    )

            return True

        # Apply multi-tool validation
        coordinated_agent = dspy.Assert(
            self.react,
            validation_fn=validate_tool_coordination,
            max_attempts=3,
            recovery_hint="Coordinate multiple tools: gather data, analyze, calculate"
        )

        return coordinated_agent(analysis_request=analysis_request)

class CustomAPITool:
    """Example custom tool for demonstration."""
    def __call__(self, query):
        # Simulate API call
        return f"API result for: {query}"
```

## Summary

ReAct agents enable:

- **Tool usage** - Connect to external systems and APIs
- **Dynamic reasoning** - Adapt based on tool results
- **Complex problem solving** - Handle multi-step tasks
- **Real-time capabilities** - Access current information
- **Extensibility** - Easy to add new tools
- **Reliability with assertions** - Guaranteed tool usage and output quality

### Key Takeaways

1. **Think-Act-Observe** cycle is the core of ReAct
2. **Choose tools wisely** based on task requirements
3. **Handle errors gracefully** - tools can fail
4. **Limit complexity** - too many tools can confuse the agent
5. **Cache results** - improve performance and reduce costs
6. **Validate with assertions** - Ensure proper tool usage and reliable outputs
7. **Require citations** - Always source information from tools
8. **Check action sequences** - Validate reasoning steps

## Next Steps

- [Custom Modules](./05-custom-modules.md) - Build your own module types
- [Module Composition](./06-composing-modules.md) - Combine modules effectively
- [Practical Examples](../examples/chapter03/) - See ReAct in action
- [Exercises](./07-exercises.md) - Practice building agents

## Further Reading

- [Paper: ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629) - Original ReAct research
- [DSPy Documentation: ReAct](https://dspy-docs.vercel.app/docs/deep-dive/react) - Technical details
- [Tool Integration Guide](../07-advanced-topics.md) - Advanced tool patterns
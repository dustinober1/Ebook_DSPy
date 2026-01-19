# Chapter 3 Exercises

## Prerequisites

- **Chapter 3 Content**: Complete understanding of all module concepts
- **Chapter 2**: Signatures - Mastery of signature design
- **Required Knowledge**: Python programming, basic module usage
- **Difficulty Level**: Intermediate to Advanced
- **Estimated Time**: 3-4 hours

## Exercise Overview

This chapter includes 7 comprehensive exercises to practice working with DSPy modules:
1. **Basic Module Usage** - Master fundamental module operations
2. **Module Composition** - Combine modules effectively
3. **ChainOfThought Applications** - Implement reasoning patterns
4. **ReAct Agent Building** - Create tool-using agents
5. **Custom Module Development** - Build specialized modules
6. **Module Optimization** - Improve performance and reliability
7. **Complete Project** - Build a multi-module application

---

## Exercise 1: Basic Module Usage

### Objective
Master the fundamental operations of DSPy's core modules.

### Tasks

#### Task 1.1: Predict Module Mastery

Create and test a `dspy.Predict` module for text classification:

```python
import dspy

# TODO: Create a text classification signature
# Include: text, categories -> classification, confidence

classification_signature = "________________________________________"

# TODO: Create the Predict module
classifier = dspy.Predict(classification_signature)

# TODO: Test with sample data
test_texts = [
    "I love this product! It works perfectly.",
    "This is terrible. Worst purchase ever.",
    "It's okay, nothing special but does the job."
]

# TODO: Classify each text and print results
```

#### Task 1.2: Module Configuration

Configure modules with different parameters:

```python
# TODO: Create modules with different temperatures
creative_module = dspy.Predict("prompt -> creative_response", temperature=0.8)
precise_module = dspy.Predict("question -> precise_answer", temperature=0.1)

# TODO: Test with the same prompt on both modules
prompt = "Describe a sunset"

# TODO: Compare the outputs and note differences
```

#### Task 1.3: Few-Shot Examples

Add examples to improve module performance:

```python
# TODO: Create examples for math problems
math_examples = [
    dspy.Example(
        problem="What is 15 + 27?",
        answer="42"
    ),
    # TODO: Add 2-3 more examples
]

# TODO: Create a math solver with examples
math_solver = dspy.Predict("math_problem -> answer", demos=math_examples)

# TODO: Test with new problems
test_problems = ["What is 8 × 7?", "What is 144 ÷ 12?"]

# TODO: Run and evaluate results
```

### Validation Questions
- Does your classifier handle different sentiment levels correctly?
- How does temperature affect output consistency?
- Do few-shot examples improve accuracy?

---

## Exercise 2: Module Composition

### Objective
Learn to combine multiple modules to create complex workflows.

### Tasks

#### Task 2.1: Sequential Pipeline

Create a text processing pipeline:

```python
import dspy

# TODO: Create three modules for a pipeline
# 1. Text cleaner
# 2. Sentiment analyzer
# 3. Summary generator

# TODO: Combine into a pipeline class
class TextPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        # TODO: Initialize modules

    def forward(self, text):
        # TODO: Execute pipeline steps
        pass

# TODO: Test the pipeline
sample_text = "   This product is AMAZING! I absolutely LOVE it!    "
pipeline = TextPipeline()
result = pipeline(sample_text)
```

#### Task 2.2: Conditional Routing

Create a router that chooses different modules based on input:

```python
# TODO: Create a router module
class QueryRouter(dspy.Module):
    def __init__(self):
        super().__init__()
        # TODO: Initialize different modules for different query types
        # - Math questions -> calculator
        # - General questions -> general_qa
        # - Creative requests -> creative_writer

    def forward(self, query):
        # TODO: Determine query type
        # TODO: Route to appropriate module
        # TODO: Return result with routing info
        pass

# TODO: Test with different types of queries
queries = [
    "What is 23 × 17?",
    "Who was the first president?",
    "Write a poem about spring"
]
```

#### Task 2.3: Error Handling

Implement error handling in module composition:

```python
# TODO: Create a robust pipeline with error handling
class RobustPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        # TODO: Initialize modules with fallbacks

    def forward(self, text):
        # TODO: Implement try-catch blocks
        # TODO: Use fallback modules when needed
        # TODO: Log errors and continue processing
        pass

# TODO: Test with problematic inputs
problematic_inputs = [
    "",  # Empty string
    None,  # None value
    "x" * 10000  # Very long string
]
```

### Evaluation Criteria
- Pipeline executes all steps correctly
- Router chooses appropriate modules
- System handles errors gracefully
- Performance is acceptable

---

## Exercise 3: ChainOfThought Applications

### Objective
Build complex reasoning systems using Chain of Thought.

### Tasks

#### Task 3.1: Mathematical Reasoning

Create a step-by-step math problem solver:

```python
import dspy

# TODO: Create a detailed math solver signature
math_signature = dspy.Signature(
    # TODO: Include fields for problem, steps, calculations, final answer
)

# TODO: Create ChainOfThought module
math_solver = dspy.ChainOfThought(math_signature)

# TODO: Create examples showing step-by-step solving
math_examples = [
    dspy.Example(
        problem="A box contains 12 red balls and 8 blue balls. What fraction are red?",
        # TODO: Add complete reasoning with steps
    )
]

# TODO: Test with complex problems
complex_problems = [
    "If Sarah earns $3000 per month and saves 20%, how much does she save in a year?",
    "A train travels at 60 mph for 3 hours. How far does it travel?"
]

# TODO: Analyze the reasoning produced
```

#### Task 3.2: Logical Reasoning

Create a logical puzzle solver:

```python
# TODO: Create a logic puzzle solver
class LogicPuzzleSolver(dspy.Module):
    def __init__(self):
        super().__init__()
        # TODO: Initialize ChainOfThought module
        # TODO: Add examples for common logic patterns

    def forward(self, puzzle):
        # TODO: Implement logic puzzle solving
        pass

# TODO: Test with classic logic puzzles
puzzles = [
    "Three friends: Alex, Ben, and Chris. One is a doctor, one is a teacher, and one is an engineer. "
    "Alex is not the doctor. The engineer is not Chris. Ben is not the teacher. "
    "Who is the engineer?"
]

# TODO: Solve and verify logical consistency
```

#### Task 3.3: Analytical Reasoning

Build a data analysis system:

```python
# TODO: Create a data analyzer with ChainOfThought
class DataAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        # TODO: Initialize modules for different analysis types

    def analyze_sales_data(self, data):
        # TODO: Analyze sales data with reasoning
        pass

# TODO: Test with sample data
sales_data = """
Q1: $100k, Q2: $120k, Q3: $110k, Q4: $150k
Products: Electronics 40%, Clothing 30%, Home 20%, Other 10%
Customers: New 30%, Returning 70%
"""

# TODO: Generate insights and recommendations
```

### Expected Output Format
```python
# Your implementations should include
# 1. Clear module definitions
# 2. Example demonstrations
# 3. Result analysis
# 4. Performance considerations

analysis = """
Provide analysis of your implementations:
- Which ChainOfThought applications worked best?
- What improvements could be made?
- How to optimize reasoning quality?
"""
```

---

## Exercise 4: ReAct Agent Building

### Objective
Create sophisticated agents that can use external tools and APIs.

### Tasks

#### Task 4.1: Web Search Agent

Build an agent that searches for and synthesizes information:

```python
import dspy

# TODO: Create a research agent signature
research_signature = dspy.Signature(
    # TODO: Include fields for query, search, synthesis, confidence
)

# TODO: Create ReAct agent with web search
research_agent = dspy.ReAct(research_signature, tools=[dspy.WebSearch()])

# TODO: Test with complex research queries
research_queries = [
    "What are the latest developments in quantum computing?",
    "Compare the pros and cons of remote work for productivity",
    "Find information about sustainable energy trends in 2024"
]

# TODO: Evaluate search quality and synthesis accuracy
```

#### Task 4.2: Calculator Agent

Create an agent that performs complex calculations:

```python
# TODO: Create a calculator agent
class CalculatorAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        # TODO: Initialize ReAct with calculator tool
        # TODO: Add examples for different calculation types

    def solve(self, problem):
        # TODO: Implement problem solving with verification
        pass

# TODO: Test with various calculation problems
calc_problems = [
    "Calculate the monthly payment on a $300k mortgage at 5% for 30 years",
    "What is the probability of drawing 3 red cards from a deck?",
    "Convert 0°C to Fahrenheit and then to Kelvin"
]

# TODO: Verify calculations are correct
```

#### Task 4.3: Custom Tool Creation

Create and integrate custom tools:

```python
# TODO: Create a custom API tool (e.g., weather, stocks, etc.)
class CustomAPI(dspy.predict.react.Tool):
    name = "custom_api"
    description = "Custom API tool demonstration"
    parameters = {"query": "Search query"}

    def forward(self, query):
        # TODO: Implement API call
        pass

# TODO: Create ReAct agent with custom tool
custom_agent = dspy.ReAct(
    "query -> research_result",
    tools=[CustomAPI()]
)

# TODO: Test the custom integration
```

### Advanced Challenge
Create an agent that can:
1. Search for information
2. Perform calculations based on found data
3. Generate reports
4. Verify its own work

---

## Exercise 5: Custom Module Development

### Objective
Build specialized modules for specific use cases.

### Tasks

#### Task 5.1: Text Enhancement Module

```python
import dspy

# TODO: Create a custom module for text enhancement
class TextEnhancer(dspy.Module):
    """Enhance text with style improvements and corrections."""

    def __init__(self):
        super().__init__()
        # TODO: Initialize any internal components

    def forward(self, original_text, enhancement_type="professional"):
        # TODO: Implement text enhancement logic
        # - Grammar correction
        # - Style improvement
        # - Clarity enhancement
        pass

# TODO: Test with different text types
test_texts = [
    "i think this is good but maybe it could be better",
    "The product were awesome when we buyed it",
    "The system processing was completed successfully"
]

# TODO: Evaluate enhancement quality
```

#### Task 5.2: Domain-Specific Module

Choose a domain and create a specialized module:

```python
# TODO: Choose one: Healthcare, Finance, Legal, Education, etc.

# TODO: Create domain-specific signature
domain_signature = dspy.Signature(
    # TODO: Define domain-specific inputs and outputs
)

# TODO: Create custom module with domain logic
class DomainModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # TODO: Initialize domain knowledge base
        # TODO: Load domain-specific rules
        # TODO: Set up validation

    def forward(self, **kwargs):
        # TODO: Implement domain-specific processing
        pass

# TODO: Test with domain-specific examples
```

#### Task 5.3: Multi-Output Module

Create a module that produces multiple related outputs:

```python
# TODO: Create a module with complex multi-output
class MultiOutputModule(dspy.Module):
    """Module that produces multiple related outputs."""

    def __init__(self):
        super().__init__()
        # TODO: Initialize sub-modules or logic

    def forward(self, input_data):
        # TODO: Generate multiple related outputs
        outputs = {}

        # TODO: Implement output generation
        # - Summary
        # - Key points
        # - Sentiment
        # - Tags
        # - Recommendations

        return dspy.Prediction(**outputs)

# TODO: Test with various inputs
```

### Testing Requirements
```python
# TODO: Create unit tests for your custom module
def test_custom_module():
    """Test suite for custom module."""

    # TODO: Test normal cases
    # TODO: Test edge cases
    # TODO: Test error handling
    # TODO: Test performance

    print("All tests passed!")

# TODO: Run your tests
```

---

## Exercise 6: Module Optimization

### Objective
Optimize module performance and reliability.

### Tasks

#### Task 6.1: Performance Optimization

Optimize a module for speed and efficiency:

```python
import time

# TODO: Create an optimized version of a module
class OptimizedModule(dspy.Module):
    """Optimized module with caching and batch processing."""

    def __init__(self):
        super().__init__()
        # TODO: Implement caching mechanism
        self.cache = {}
        # TODO: Optimize LM calls
        # TODO: Batch processing capabilities

    def forward(self, **kwargs):
        # TODO: Check cache first
        # TODO: Implement optimized processing
        # TODO: Cache results
        pass

# TODO: Benchmark vs non-optimized version
def benchmark_modules():
    """Compare performance of optimized vs original module."""

    # TODO: Time both versions
    # TODO: Calculate speed improvement
    # TODO: Report results
```

#### Task 6.2: Reliability Enhancement

Make a module more reliable with error handling and validation:

```python
# TODO: Create a reliable module wrapper
class ReliableWrapper(dspy.Module):
    """Wrapper that adds reliability to any module."""

    def __init__(self, base_module, max_retries=3):
        super().__init__()
        self.base_module = base_module
        self.max_retries = max_retries

    def forward(self, **kwargs):
        # TODO: Implement retry logic
        # TODO: Add input validation
        # TODO: Add output validation
        # TODO: Handle failures gracefully
        pass

# TODO: Test reliability with edge cases
```

#### Task 6.3: Resource Management

Create a module that manages computational resources:

```python
# TODO: Create a resource-aware module
class ResourceManager(dspy.Module):
    """Module that manages tokens and compute resources."""

    def __init__(self, token_limit=1000):
        super().__init__()
        self.token_limit = token_limit
        self.tokens_used = 0

    def forward(self, **kwargs):
        # TODO: Track token usage
        # TODO: Implement token limits
        # TODO: Optimize for efficiency
        pass

# TODO: Test with various input sizes
```

### Metrics to Measure
- Processing time (ms)
- Tokens used per request
- Success rate (%)
- Cache hit rate (%)
- Memory usage (MB)

---

## Exercise 7: Complete Project

### Objective
Build a complete multi-module application for a real-world scenario.

### Choose ONE Project:

#### Option A: Customer Support System
A customer support system that:
- Categorizes incoming tickets
- Analyzes sentiment and urgency
- Generates responses
- Routes to appropriate departments
- Tracks resolution status

#### Option B: Content Analysis Platform
A content analysis platform that:
- Extracts key themes from documents
- Performs sentiment analysis
- Identifies entities and relationships
- Generates summaries
- Provides recommendations

#### Option C: Research Assistant
A research assistant that:
- Searches for information across sources
- Synthesizes findings
- Generates reports
- Identifies knowledge gaps
- Recommends further research

#### Option D: Personal Finance Advisor
A finance advisor that:
- Analyzes financial statements
- Calculates financial ratios
- Provides investment recommendations
- Risk assessment
- Budget optimization suggestions

### Tasks

#### Task 7.1: System Design

Document your system architecture:

```python
# TODO: Create a system design document
system_design = """
Project: [Your Project Title]

Architecture:
1. Module 1: [Description]
2. Module 2: [Description]
3. ...

Data Flow:
[Diagram or description]

Key Components:
- [Component 1]
- [Component 2]
...

Integration Points:
- [How modules interact]
- [External systems needed]
"""

# TODO: Save your design
```

#### Task 7.2: Implementation

Build the complete system:

```python
# TODO: Implement your complete system
class ProjectSystem(dspy.Module):
    """Main system module."""

    def __init__(self):
        super().__init__()
        # TODO: Initialize all modules
        # TODO: Set up connections
        # TODO: Configure defaults

    def process(self, **kwargs):
        # TODO: Process through your pipeline
        # TODO: Return comprehensive results
        pass

# TODO: Test with realistic scenarios
```

#### Task 7.3: Evaluation

Create evaluation criteria:

```python
# TODO: Create evaluation functions
def evaluate_accuracy(system, test_cases):
    """Evaluate system accuracy."""
    # TODO: Implement accuracy testing
    pass

def evaluate_performance(system, test_cases):
    """Evaluate system performance."""
    # TODO: Implement performance testing
    pass

def evaluate_user_satisfaction(system, user_feedback):
    """Evaluate user satisfaction."""
    # TODO: Implement satisfaction analysis
    pass

# TODO: Run comprehensive evaluation
```

### Deliverables

1. **System Architecture Diagram**
2. **Complete Implementation**
3. **Test Suite**
4. **Evaluation Report**
5. **User Documentation**
6. **Future Improvements**

### Success Criteria
- System processes inputs correctly
- Outputs are accurate and useful
- Performance is acceptable
- Error handling is robust
- Code is well-documented

---

## Submission Guidelines

### What to Submit
1. **Code Files**: All Python implementations
2. **Documentation**: Comments and docstrings
3. **Test Results**: Outputs and analyses
4. **Reflection**: What you learned

### How to Submit
1. Create a directory: `exercises/chapter03/your_name/`
2. Organize by exercise (e.g., `exercise1/`, `exercise2/`, etc.)
3. Include all files and documentation
4. Ensure code runs without errors

### Evaluation Criteria

| Criterion | Weight |
|-----------|--------|
| Correctness | 30% |
| Completeness | 25% |
| Code Quality | 20% |
| Documentation | 15% |
| Creativity | 10% |

### Self-Assessment

Before submitting, review:
- [ ] Code follows DSPy best practices
- [ ] All exercises are attempted
- [ ] Code is well-commented
- [ ] Tests demonstrate functionality
- [ ] Documentation is clear

## Solutions and Explanations

Solutions are available in the `solutions/` directory. Each solution includes:

1. **Complete Working Code**: Full implementations
2. **Explanation**: Design choices and rationale
3. **Alternatives**: Other valid approaches
4. **Extensions**: Ideas for improvement

## Further Practice

After completing these exercises:

1. **Extend your solutions** with additional features
2. **Combine exercises** to create more complex systems
3. **Optimize for production** - Consider scalability
4. **Share your work** with the DSPy community
5. **Build real applications** using these patterns

## Summary

These exercises cover:
- Core module usage and configuration
- Advanced composition patterns
- Reasoning with Chain of Thought
- Building tool-using agents
- Creating custom modules
- Performance optimization
- Complete application development

By completing these exercises, you'll have mastered DSPy modules and be ready to build sophisticated LLM applications.

## Next Steps

- Review your solutions against provided answers
- Experiment with optimizations
- Build your own applications
- Proceed to Chapter 4: Evaluation
- Join the DSPy community for support

## Resources

- [Solution Code](../../exercises/chapter03/solutions/) - Complete implementations
- [DSPy Documentation](https://dspy-docs.vercel.app/) - Official docs
- [Community Forum](https://github.com/stanfordnlp/dspy/discussions) - Get help
- [Example Gallery](../examples/chapter03/) - More examples
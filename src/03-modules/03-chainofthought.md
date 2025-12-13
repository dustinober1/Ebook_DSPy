# Chain of Thought Module

## Prerequisites

- **Previous Section**: [The Predict Module](./02-predict-module.md) - Understanding of basic modules
- **Chapter 2**: Signatures - Familiarity with signature design
- **Required Knowledge**: Concept of step-by-step reasoning
- **Difficulty Level**: Intermediate
- **Estimated Reading Time**: 40 minutes

## Learning Objectives

By the end of this section, you will:
- Master the `dspy.ChainOfThought` module for complex reasoning tasks
- Understand how to elicit step-by-step thinking from language models
- Learn to structure reasoning chains for different types of problems
- Discover techniques to improve reasoning quality and reliability
- Know when Chain of Thought outperforms simple prediction

## Introduction to Chain of Thought

Chain of Thought (CoT) is a prompting technique that encourages language models to "think step by step" before providing a final answer. This approach significantly improves performance on complex reasoning tasks that require multiple steps of analysis.

### Why Chain of Thought Works

**Without CoT:**
```
Question: If a rope is 10 meters long and we cut it into 4 equal pieces, then cut each piece in half, how many pieces do we have?
Answer: 4  (Wrong - jumps to conclusion)
```

**With CoT:**
```
Question: If a rope is 10 meters long and we cut it into 4 equal pieces, then cut each piece in half, how many pieces do we have?

Reasoning:
1. Start with 1 rope
2. Cut into 4 equal pieces → now we have 4 pieces
3. Cut each of the 4 pieces in half → each piece becomes 2 pieces
4. Total pieces = 4 pieces × 2 = 8 pieces

Answer: 8  (Correct - shows reasoning)
```

## Basic Usage

### Simple CoT Example
```python
import dspy

# Define a signature that includes reasoning
class MathProblem(dspy.Signature):
    """Solve math problems step by step."""
    problem = dspy.InputField(desc="Math problem to solve", type=str)
    reasoning = dspy.OutputField(desc="Step-by-step reasoning", type=str)
    answer = dspy.OutputField(desc="Final answer", type=str)

# Create ChainOfThought module
math_solver = dspy.ChainOfThought(MathProblem)

# Use it
result = math_solver(
    problem="A baker has 24 cupcakes. If she sells them in boxes of 6, how many boxes does she need?"
)

print("Reasoning:")
print(result.reasoning)
print("\nAnswer:")
print(result.answer)
```

### String Signature with CoT
```python
# Quick CoT with string signature
cot_analyzer = dspy.ChainOfThought(
    "situation -> reasoning, conclusion"
)

result = cot_analyzer(
    situation="The company's revenue increased 20% but expenses increased 30%. Is the company doing better?"
)

print(result.reasoning)
print(result.conclusion)
```

## Structuring Reasoning Chains

### 1. Mathematical Reasoning
```python
class ComplexMathSolver(dspy.Signature):
    """Solve complex mathematical problems with detailed reasoning."""
    problem = dspy.InputField(desc="Complex math problem", type=str)
    givens = dspy.OutputField(desc="Information given in the problem", type=str)
    approach = dspy.OutputField(desc="Mathematical approach to solve", type=str)
    steps = dspy.OutputField(desc="Detailed solution steps", type=str)
    calculations = dspy.OutputField(desc="Show your work", type=str)
    answer = dspy.OutputField(desc="Final numerical answer", type=str)
    verification = dspy.OutputField(desc("Check your answer", type=str))

# Create with examples that show good reasoning
math_examples = [
    dspy.Example(
        problem="A train travels 300 km in 3 hours. What is its speed?",
        givens="Distance = 300 km, Time = 3 hours",
        approach="Use the formula: Speed = Distance / Time",
        steps="1. Identify the formula\n2. Plug in values\n3. Calculate",
        calculations="Speed = 300 km / 3 hours = 100 km/hour",
        answer="100 km/hour",
        verification="Check: 100 km/hour × 3 hours = 300 km ✓"
    )
]

math_solver = dspy.ChainOfThought(ComplexMathSolver, demos=math_examples)

# Solve a complex problem
result = math_solver(
    problem="If John can paint a house in 6 hours and Mary can paint it in 4 hours, "
            "how long will it take them to paint it together?"
)

print(f"Answer: {result.answer}")
```

### 2. Logical Reasoning
```python
class LogicalReasoner(dspy.Signature):
    """Apply logical reasoning to solve problems."""
    scenario = dspy.InputField(desc="Situation requiring logical analysis", type=str)
    facts = dspy.OutputField(desc="Relevant facts from scenario", type=str)
    assumptions = dspy.OutputField(desc("Reasonable assumptions made", type=str))
    logical_steps = dspy.OutputField(desc="Step-by-step logical deduction", type=str)
    conclusion = dspy.OutputField(desc("Logical conclusion", type=str)
    confidence = dspy.OutputField(desc("Confidence in conclusion (1-10)", type=int)

logical_solver = dspy.ChainOfThought(
    LogicalReasoner,
    instructions="Think carefully and logically. Identify all assumptions you make."
)

# Solve a logic puzzle
result = logical_solver(
    scenario="All employees who work in the IT department must know Python. "
            "Sarah works in the IT department but doesn't know Python. "
            "What can we conclude?"
)

print(f"Facts: {result.facts}")
print(f"Conclusion: {result.conclusion}")
print(f"Confidence: {result.confidence}/10")
```

### 3. Analytical Reasoning
```python
class DataAnalyzer(dspy.Signature):
    """Analyze data and provide insights."""
    data = dspy.InputField(desc="Data to analyze", type=str)
    analysis_goal = dspy.InputField(desc="What we want to learn from data", type=str)
    observations = dspy.OutputField(desc("Key observations from data", type=str)
    patterns = dspy.OutputField(desc("Patterns or trends identified", type=str)
    insights = dspy.OutputField(desc("Deep insights from analysis", type=str)
    limitations = dspy.OutputField(desc("Limitations of analysis", type=str)
    recommendations = dspy.OutputField(desc("Actionable recommendations", type=str)

data_analyzer = dspy.ChainOfThought(DataAnalyzer)

# Analyze sales data
result = data_analyzer(
    data="Q1 Sales: $100k, Q2: $120k, Q3: $110k, Q4: $150k. "
         "Marketing spend: Q1: $10k, Q2: $15k, Q3: $12k, Q4: $20k",
    analysis_goal="Understand the effectiveness of marketing spend"
)

print(f"Key Insight: {result.insights}")
print(f"Recommendation: {result.recommendations}")
```

## Advanced CoT Techniques

### 1. Comparative Reasoning
```python
class ComparisonAnalyzer(dspy.Signature):
    """Compare two or more options with detailed reasoning."""
    options = dspy.InputField(desc="Options to compare", type=str)
    criteria = dspy.InputField(desc("Comparison criteria", type=str)
    analysis_per_option = dspy.OutputField(desc("Analysis of each option", type=str)
    comparison_matrix = dspy.OutputField(desc("Detailed comparison", type=str)
    tradeoffs = dspy.OutputField(desc("Trade-offs identified", type=str)
    recommendation = dspy.OutputField(desc("Recommended choice with reasoning", type=str)

comparator = dspy.ChainOfThought(
    ComparisonAnalyzer,
    instructions="Consider all criteria carefully and explain trade-offs clearly."
)

result = comparator(
    options="Option A: Cloud-based system with monthly fees\n"
            "Option B: On-premise system with one-time cost",
    criteria="Cost, maintenance, scalability, security, performance"
)

print(f"Recommendation: {result.recommendation}")
```

### 2. Causal Reasoning
```python
class CausalAnalyzer(dspy.Signature):
    """Analyze cause and effect relationships."""
    situation = dspy.InputField(desc("Situation to analyze", type=str)
    potential_causes = dspy.OutputField(desc("Possible causes to consider", type=str)
    causal_chain = dspy.OutputField(desc("Step-by-step causal analysis", type=str)
    evidence = dspy.OutputField(desc("Evidence supporting conclusions", type=str)
    primary_cause = dspy.OutputField(desc("Most likely primary cause", type=str)
    secondary_factors = dspy.OutputField(desc("Contributing factors", type=str)
    prevention = dspy.OutputField(desc("How to prevent recurrence", type=str)

causal_analyzer = dspy.ChainOfThought(CausalAnalyzer)

result = causal_analyzer(
    situation="Website traffic dropped 50% overnight after a system update"
)

print(f"Primary Cause: {result.primary_cause}")
print(f"Prevention: {result.prevention}")
```

### 3. Creative Problem Solving
```python
class CreativeSolver(dspy.Signature):
    """Generate creative solutions to problems."""
    problem = dspy.InputField(desc("Problem to solve", type=str)
    constraints = dspy.InputField(desc("Constraints and limitations", type=str)
    brainstorming = dspy.OutputField(desc("Initial ideas exploration", type=str)
    solution_development = dspy.OutputField(desc("Develop promising solutions", type=str)
    evaluation = dspy.OutputField(desc("Evaluate solutions against criteria", type=str)
    final_solution = dspy.OutputField(desc("Best solution with implementation plan", type=str)
    alternatives = dspy.OutputField(desc("Backup solutions", type=str)

creative_solver = dspy.ChainOfThought(
    CreativeSolver,
    instructions="Think outside the box while remaining practical."
)

result = creative_solver(
    problem="How to reduce plastic waste in a city of 1 million people?",
    constraints="Limited budget, need public support, implementable within 2 years"
)

print(f"Solution: {result.final_solution}")
```

## Improving CoT Performance

### 1. Use High-Quality Examples
```python
# Examples that demonstrate good reasoning
cooking_examples = [
    dspy.Example(
        recipe="Recipe calls for 2 cups flour but I only have 1 cup",
        reasoning="1. Need to adjust quantities proportionally\n"
                "2. Original ratio: 2 cups flour for full recipe\n"
                "3. Have only 1 cup = 50% of flour\n"
                "4. Must halve all ingredients",
        solution="Halve all ingredient quantities"
    )
]

recipe_adapter = dspy.ChainOfThought(
    "recipe_adaptation_problem -> reasoning, solution",
    demos=cooking_examples
)
```

### 2. Add Specific Instructions
```python
# Guide the reasoning process
diagnostic_module = dspy.ChainOfThought(
    "symptoms -> diagnostic_reasoning, diagnosis",
    instructions="1. List all possible causes\n"
                 "2. Eliminate unlikely causes based on symptoms\n"
                 "3. Consider most probable causes\n"
                 "4. Provide differential diagnosis"
)
```

### 3. Use Structured Prompts
```python
class StructuredReasoning(dspy.Signature):
    """Reason in a highly structured format."""
    problem = dspy.InputField(desc="Problem to solve", type=str)
    step1_identify = dspy.OutputField(desc("Step 1: Identify key information", type=str)
    step2_analyze = dspy.OutputField(desc("Step 2: Analyze relationships", type=str)
    step3_synthesize = dspy.OutputField(desc("Step 3: Synthesize findings", type=str)
    step4_conclude = dspy.OutputField(desc("Step 4: Draw conclusions", type=str)

structured_reasoner = dspy.ChainOfThought(StructuredReasoning)
```

## Real-World Applications

### 1. Medical Diagnosis Assistant
```python
class DiagnosticAssistant(dspy.Signature):
    """Assist in medical diagnosis with systematic reasoning."""
    patient_case = dspy.InputField(desc("Patient symptoms and history", type=str)
    symptom_analysis = dspy.OutputField(desc("Systematic symptom analysis", type=str)
    differential_diagnosis = dspy.OutputField(desc("Possible conditions with reasoning", type=str)
    key_findings = dspy.OutputField(desc("Most important clinical findings", type=str)
    recommended_tests = dspy.OutputField(desc("Diagnostic tests to order", type=str)
    preliminary_diagnosis = dspy.OutputField(desc("Most likely diagnosis", type=str)
    reasoning_confidence = dspy.OutputField(desc("Confidence in diagnosis (1-10)", type=int)

diagnostic_assistant = dspy.ChainOfThought(
    DiagnosticAssistant,
    instructions="Think like a physician. Consider all relevant information systematically. "
                 "Always consider multiple possibilities before concluding."
)

# Note: This is for educational purposes only
result = diagnostic_assistant(
    patient_case="45-year-old male, chest pain that worsens with exertion, "
                 "smoker 20 years, father had heart attack at 55"
)

print(f"Key Findings: {result.key_findings}")
print(f"Recommended Tests: {result.recommended_tests}")
```

### 2. Financial Analysis
```python
class FinancialAnalyzer(dspy.Signature):
    """Analyze financial situations with detailed reasoning."""
    financial_data = dspy.InputField(desc("Financial information", type=str)
    analysis_objective = dspy.InputField(desc("What we need to determine", type=str)
    data_breakdown = dspy.OutputField(desc("Break down the financial data", type=str)
    calculations = dspy.OutputField(desc("Show all calculations", type=str)
    trends = dspy.OutputField(desc("Identify trends and patterns", type=str)
    insights = dspy.OutputField(desc("Financial insights discovered", type=str)
    conclusion = dspy.OutputField(desc("Conclusions with evidence", type=str)
    recommendations = dspy.OutputField(desc("Actionable recommendations", type=str)

financial_analyzer = dspy.ChainOfThought(FinancialAnalyzer)

result = financial_analyzer(
    financial_data="Company Revenue: Year 1: $1M, Y2: $1.3M, Y3: $1.5M. "
                 "Expenses: Y1: $800k, Y2: $1.1M, Y3: $1.4M",
    analysis_objective="Is the company becoming more profitable?"
)

print(f"Conclusion: {result.conclusion}")
print(f"Recommendations: {result.recommendations}")
```

### 3. Legal Reasoning
```python
class LegalReasoner(dspy.Signature):
    """Apply legal reasoning to cases."""
    case_facts = dspy.InputField(desc("Facts of the case", type=str)
    legal_question = dspy.InputField(desc("Legal question to answer", type=str)
    relevant_law = dspy.OutputField(desc("Applicable legal principles", type=str)
    legal_analysis = dspy.OutputField(desc("Step-by-step legal analysis", type=str)
    precedent_cases = dspy.OutputField(desc("Similar case precedents", type=str)
    legal_conclusion = dspy.OutputField(desc("Legal conclusion with reasoning", type=str)
    confidence_level = dspy.OutputField(desc("Confidence in conclusion", type=str)

legal_reasoner = dspy.ChainOfThought(
    LegalReasoner,
    instructions="Apply legal principles systematically. Consider precedents and counterarguments."
)

result = legal_reasoner(
    case_facts="Employee signed non-compete for 1 year within 50 miles. "
               "Company is in California. Employee wants to work for competitor 30 miles away.",
    legal_question="Is the non-compete enforceable?"
)

print(f"Legal Conclusion: {result.legal_conclusion}")
```

## Performance Tips

### 1. Temperature Settings
```python
# Lower temperature for more consistent reasoning
consistent_reasoner = dspy.ChainOfThought(
    "problem -> reasoning, answer",
    temperature=0.1
)

# Higher temperature for creative problem solving
creative_reasoner = dspy.ChainOfThought(
    "problem -> reasoning, solution",
    temperature=0.8
)
```

### 2. Example Quality
```python
# Show the desired reasoning style
good_example = dspy.Example(
    problem="If a store sells items at $10 each and offers a 20% discount, "
            "how much do 5 items cost?",
    reasoning="1. Original price per item = $10\n"
             "2. Discount = 20% of $10 = $2\n"
             "3. Discounted price = $10 - $2 = $8\n"
             "4. Total for 5 items = 5 × $8 = $40",
    answer="$40"
)
```

### 3. Output Constraints
```python
# Specify reasoning length
concise_reasoner = dspy.ChainOfThought(
    "question -> reasoning, answer",
    instructions="Keep reasoning brief but clear (max 3 steps)."
)

detailed_reasoner = dspy.ChainOfThought(
    "question -> reasoning, answer",
    instructions="Provide detailed reasoning showing all work."
)
```

## Common Pitfalls and Solutions

### 1. Circular Reasoning
```python
# Bad: May create circular arguments
circular_risk = dspy.ChainOfThought("x -> reasoning that x is true because x, answer")

# Good: Independent reasoning
valid_reasoning = dspy.ChainOfThought(
    "situation -> evidence_based_reasoning, conclusion",
    instructions="Base reasoning on evidence, not assumptions."
)
```

### 2. Missing Steps
```python
# Add explicit step tracking
class StepTracker(dspy.Signature):
    problem = dspy.InputField(desc="Problem to solve", type=str)
    step_count = dspy.OutputField(desc("Number of reasoning steps", type=int)
    reasoning = dspy.OutputField(desc("Complete reasoning with numbered steps", type=str)

step_tracker = dspy.ChainOfThought(
    StepTracker,
    instructions="Use clear numbered steps. Don't skip steps."
)
```

### 3. Incorrect Calculations
```python
# Add verification step
verified_solver = dspy.ChainOfThought(
    "math_problem -> reasoning, calculations, answer, verification",
    instructions="Always double-check your calculations."
)
```

## When to Use Chain of Thought

### Use CoT when:

1. **Multi-step problems** - Problems requiring multiple reasoning steps
2. **Complex logic** - Tasks with logical dependencies
3. **Mathematical problems** - Any calculation requiring steps
4. **Analysis tasks** - Breaking down complex information
5. **Decision making** - Weighing multiple factors

### Consider Predict when:

1. **Simple transformations** - Direct input-output mapping
2. **Classification tasks** - Simple categorization
3. **Text generation** - Creative writing without analysis
4. **Quick responses** - When speed is critical
5. **High-confidence tasks** - When accuracy is already high

## Summary

Chain of Thought enables:

- **Better reasoning** through step-by-step thinking
- **Improved accuracy** on complex problems
- **Transparent process** - you can see the reasoning
- **Error detection** - steps can be verified
- **Teaching opportunities** - shows how to think

### Key Takeaways

1. **Always show work** - Make reasoning explicit
2. **Use examples** to demonstrate desired reasoning style
3. **Structure reasoning** according to problem type
4. **Verify conclusions** - Include validation steps
5. **Know when to use** - Not all tasks need CoT

## Next Steps

- [ReAct Agents](./04-react-agents.md) - Add tool-using capabilities
- [Module Composition](./06-composing-modules.md) - Combine CoT with other modules
- [Practical Examples](../examples/chapter03/) - See CoT in action
- [Exercises](./07-exercises.md) - Practice CoT techniques

## Further Reading

- [Paper: Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) - Original CoT research
- [DSPy Documentation: ChainOfThought](https://dspy-docs.vercel.app/docs/deep-dive/chain_of_thought)
- [Reasoning Patterns](../05-optimizers.md) - Advanced reasoning techniques
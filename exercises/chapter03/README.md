# Chapter 3: DSPy Modules - Exercises

This directory contains hands-on exercises to practice and master DSPy module concepts. Each exercise builds on concepts from the textbook chapters.

## Exercise Files

### 1. `01_module_basics_exercise.py`
**Module Fundamentals**

Practice basic DSPy module concepts:
- Creating simple Predict modules
- Working with multiple outputs
- Configuring module parameters
- Implementing few-shot learning
- Adding error handling

**Topics Covered:**
- `dspy.Predict` usage
- Signature definition
- Module configuration
- Example-based learning
- Input validation

### 2. `02_chainofthought_exercise.py`
**Chain of Thought Reasoning**

Master step-by-step reasoning with CoT:
- Basic CoT implementation
- Using examples to improve reasoning
- Complex planning tasks
- Error analysis and debugging
- Temperature experimentation
- Complex reasoning patterns

**Topics Covered:**
- `dspy.ChainOfThought` module
- Reasoning step generation
- Few-shot prompt engineering
- Planning and problem decomposition
- Creative vs precise reasoning

### 3. `03_advanced_modules_exercise.py`
**Advanced Module Techniques**

Advanced module patterns and architectures:
- Building ReAct agents with tools
- Creating custom module classes
- Module composition patterns
- Conditional routing
- Stateful modules
- Module validation

**Topics Covered:**
- `dspy.React` agents
- Custom module development
- Module inheritance
- Composition patterns
- State management
- Error handling

### 4. `solutions.py`
**Complete Solutions**

Reference implementations for all exercises. Use these to:
- Check your answers
- Understand alternative approaches
- Learn best practices
- Compare your implementations

## How to Use the Exercises

### Getting Started
1. Read the corresponding chapter sections first
2. Work through exercises in order
3. Fill in TODO sections with your code
4. Run verification functions to check progress

### Exercise Structure
Each exercise follows this pattern:
```python
def exercise_X_Y():
    """Description of the exercise task."""

    # TODO: Your implementation here
    pass

    return {
        "task1": False,  # Set to True when completed
        "task2": False   # Set to True when completed
    }
```

### Verification
Each exercise includes a verification function:
```python
def verify_exercise_X():
    """Check completion status of all exercises."""
    # Run this to see your progress
```

### Hints System
Stuck on an exercise? Use the hint system:
```python
show_hints('X_Y')  # Get hints for specific exercise
```

## Running Exercises

### Individual Exercise
```bash
# Run specific exercise file
python 01_module_basics_exercise.py
```

### Check Progress
```python
# In each exercise file, run:
verify_exercise_1()  # For module basics
verify_exercise_2()  # For chain of thought
verify_exercise_3()  # For advanced modules
```

### View Solutions
```python
# Run solutions file to see complete implementations
python solutions.py
```

## Learning Path

### 1. Start with Module Basics
- Complete `01_module_basics_exercise.py`
- Focus on understanding Predict modules
- Practice with different configurations

### 2. Progress to Chain of Thought
- Complete `02_chainofthought_exercise.py`
- Learn to generate step-by-step reasoning
- Experiment with examples and temperature

### 3. Advanced Techniques
- Complete `03_advanced_modules_exercise.py`
- Build complex module architectures
- Master composition patterns

### 4. Review and Compare
- Check your solutions against `solutions.py`
- Understand different implementation approaches
- Identify areas for improvement

## Common Pitfalls and Tips

### Module Creation
- **Pitfall**: Forgetting to define proper signatures
- **Tip**: Always specify InputField and OutputField with descriptions

### Chain of Thought
- **Pitfall**: Not providing enough context in examples
- **Tip**: Include complete reasoning chains in your examples

### Custom Modules
- **Pitfall**: Not properly calling parent `__init__`
- **Tip**: Always call `super().__init__()` first

### Error Handling
- **Pitfall**: Not handling edge cases
- **Tip**: Test with None, empty strings, and invalid inputs

## Exercise-Specific Tips

### Exercise 1.1 - Basic Predict
```python
# Correct signature definition
class MySignature(dspy.Signature):
    """Description of what this does."""
    input_field = dspy.InputField(desc="Input description", type=str)
    output_field = dspy.OutputField(desc="Output description", type=str)
```

### Exercise 2.1 - Chain of Thought
```python
# Include reasoning in signature
class MathSolverSignature(dspy.Signature):
    problem = dspy.InputField(desc="Math problem", type=str)
    reasoning = dspy.OutputField(desc="Step-by-step solution", type=str)
    answer = dspy.OutputField(desc="Final answer", type=str)
```

### Exercise 3.2 - Custom Module
```python
# Proper custom module structure
class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()  # Always call this first!
        # Initialize sub-modules here

    def forward(self, **kwargs):
        # Implementation here
        return dspy.Prediction(result=output)
```

## Progress Tracking

Each exercise provides:
- ✅ Completion indicators for each task
- Progress percentage calculation
- Specific feedback on what's missing

## Extending the Exercises

After completing the basic exercises:
1. **Add Complexity**: Increase the difficulty of problems
2. **Combine Concepts**: Mix multiple module types
3. **Real Applications**: Apply to your specific use cases
4. **Performance Testing**: Benchmark your implementations

## Additional Resources

- **Chapter Text**: Review `../../src/03-modules/` for detailed explanations
- **Examples**: See `../../examples/chapter03/` for working implementations
- **DSPy Documentation**: https://dspy-docs.vercel.app/

## Getting Help

If you're stuck:
1. Use the hint system: `show_hints('X_Y')`
2. Check the solutions file for reference
3. Review the examples directory
4. Go back to the chapter text

## Success Criteria

You've successfully mastered Chapter 3 when you can:
- ✅ Create and configure basic modules
- ✅ Implement Chain of Thought reasoning
- ✅ Build custom module architectures
- ✅ Compose modules in various patterns
- ✅ Handle errors gracefully
- ✅ Optimize module performance

Good luck with your learning!
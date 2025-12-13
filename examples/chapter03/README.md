# Chapter 3: DSPy Modules - Examples

This directory contains practical examples demonstrating DSPy module concepts and implementations.

## Files Overview

### 1. `01_basic_modules.py`
**Basic Module Usage Examples**

Demonstrates fundamental DSPy module concepts:
- Using `dspy.Predict` for simple tasks
- Module configuration with temperature and other parameters
- Few-shot learning with examples
- Modules with multiple outputs
- Batch processing patterns
- Performance testing and benchmarking
- Error handling strategies
- Module comparison (Predict vs ChainOfThought)

### 2. `02_chain_of_thought.py`
**Chain of Thought Module Examples**

Shows advanced reasoning capabilities:
- Mathematical problem solving with step-by-step reasoning
- Logical puzzle solving and deduction
- Data analysis and pattern recognition
- Scientific method and hypothesis testing
- Ethical reasoning and analysis
- Creative problem solving
- Diagnostic reasoning for technical issues

### 3. `03_react_agents.py`
**ReAct (Reason+Act) Agent Examples**

Demonstrates tool-using agents:
- Tool integration and usage
- Multi-step problem solving with tools
- Information gathering and synthesis
- Multi-tool coordination
- Interactive agent workflows
- Error handling in tool usage
- Custom tool creation and integration

### 4. `04_custom_modules.py`
**Custom Module Development Examples**

Shows how to build custom modules:
- Simple custom modules extending `dspy.Module`
- Advanced pipelines with multiple processing steps
- Adaptive modules with conditional logic
- Stateful modules that maintain information
- Ensemble processing with multiple classifiers
- Validation and error handling patterns
- Metrics tracking and performance monitoring

### 5. `05_module_composition.py`
**Module Composition Patterns**

Demonstrates advanced composition techniques:
- Sequential processing chains
- Parallel processing patterns
- Conditional routing based on input
- Hierarchical module architectures
- Dynamic composition builders
- Pipeline with checkpoints and recovery
- Adaptive ensemble methods
- Multi-modal processing

## How to Use These Examples

### Running Individual Examples
```bash
# Run a specific example
python 01_basic_modules.py
python 02_chain_of_thought.py
python 03_react_agents.py
python 04_custom_modules.py
python 05_module_composition.py
```

### Running All Examples
Each file has a main execution function that runs all demonstrations:
```python
if __name__ == "__main__":
    run_all_examples()
```

### Key Concepts Demonstrated

1. **Module Basics**
   - Module creation and configuration
   - Input/output field definitions
   - Parameter tuning

2. **Reasoning Patterns**
   - Step-by-step problem solving
   - Logical deduction
   - Analytical reasoning

3. **Tool Integration**
   - Tool definition and registration
   - Tool selection and usage
   - Tool composition

4. **Custom Architecture**
   - Module inheritance
   - State management
   - Error handling

5. **Composition Strategies**
   - Sequential processing
   - Parallel execution
   - Conditional routing

## Tips for Learning

1. **Start with Basics**: Begin with `01_basic_modules.py` to understand fundamental concepts.

2. **Follow the Progression**: The examples are ordered by complexity, building on previous concepts.

3. **Modify and Experiment**: Try changing parameters, adding new modules, or combining examples.

4. **Read the Code**: Each example contains detailed comments explaining the concepts.

5. **Connect to Exercises**: These examples directly relate to exercises in the `../../exercises/chapter03/` directory.

## Common Patterns

### Basic Module Pattern
```python
class MySignature(dspy.Signature):
    """Define the input/output contract."""
    input_field = dspy.InputField(desc="Description", type=str)
    output_field = dspy.OutputField(desc="Description", type=str)

module = dspy.Predict(MySignature)
result = module(input_field="some text")
```

### Custom Module Pattern
```python
class MyCustomModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # Initialize sub-modules

    def forward(self, **kwargs):
        # Implement processing logic
        return dspy.Prediction(result=processed_output)
```

### Chain of Thought Pattern
```python
cot_module = dspy.ChainOfThought(MySignature)
result = cot_module(input="problem")  # Includes reasoning field
```

## Error Handling

Examples demonstrate various error handling strategies:
- Input validation
- Graceful degradation
- Retry mechanisms
- Fallback options

## Performance Considerations

- Module caching strategies
- Batch processing techniques
- Memory management
- Processing time optimization

## Next Steps

After exploring these examples:
1. Try the exercises in `../../exercises/chapter03/`
2. Modify the examples for your specific use cases
3. Combine different patterns from multiple examples
4. Create your own custom modules

## Dependencies

These examples require:
- DSPy framework
- Standard Python libraries (math, json, datetime, re, typing)

No external dependencies are needed for most examples.
# Chapter 2 Examples: DSPy Signatures

This directory contains practical examples demonstrating DSPy signatures from basic to advanced concepts.

## Files Overview

### 1. `01_basic_signatures.py`
Demonstrates fundamental DSPy signature concepts:
- Simple string-based signatures (`"input -> output"`)
- Common signature patterns (QA, classification, summarization)
- Basic usage with DSPy modules
- Chaining multiple signatures

**Key Examples:**
- Question Answering with context
- Text classification with confidence
- Document summarization
- Language translation
- Information extraction

### 2. `02_typed_signatures.py`
Shows advanced typed signature implementations:
- Signature classes with `dspy.Signature`
- Typed input and output fields
- Field descriptions and prefixes
- Optional fields and defaults
- Complex data structures

**Real-world Examples:**
- Customer support ticket analysis
- Financial document analysis
- Medical symptom triage
- Educational content generation

### 3. `03_advanced_signatures.py`
Covers sophisticated signature patterns:
- Hierarchical signature composition
- Dynamic signature construction
- Self-referential processing
- Multi-modal content analysis
- Complex workflow orchestration

**Advanced Concepts:**
- Building signatures dynamically at runtime
- Processing hierarchical data structures
- Analyzing content across text, image, audio, video
- Orchestrating multi-stage workflows

### 4. `04_real_world_applications.py`
Production-ready implementations for business domains:
- Customer service intelligence
- Healthcare triage systems
- Financial risk assessment
- Legal compliance checking
- E-commerce personalization

**Enterprise Features:**
- Comprehensive input/output handling
- Domain-specific business logic
- Risk assessment and compliance
- Personalization engines

### 5. `05_signature_composition.py`
Demonstrates composition patterns:
- Sequential processing pipelines
- Parallel independent analysis
- Conditional processing logic
- Hierarchical multi-level systems
- Adaptive workflow selection

**Composition Patterns:**
- Building complex systems from simple components
- Optimizing processing based on input characteristics
- Scaling analysis across multiple documents

## Running the Examples

### Prerequisites

```bash
# Install required packages
pip install dspy-ai
pip install typing-extensions
```

### Running Individual Examples

```bash
# Basic signatures
python 01_basic_signatures.py

# Typed signatures
python 02_typed_signatures.py

# Advanced patterns
python 03_advanced_signatures.py

# Real-world applications
python 04_real_world_applications.py

# Composition examples
python 05_signature_composition.py
```

### Running All Examples

Create a runner script:

```python
# run_all.py
import subprocess
import sys

examples = [
    "01_basic_signatures.py",
    "02_typed_signatures.py",
    "03_advanced_signatures.py",
    "04_real_world_applications.py",
    "05_signature_composition.py"
]

for example in examples:
    print(f"\n{'='*60}")
    print(f"Running {example}")
    print('='*60)
    try:
        subprocess.run([sys.executable, example], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {example}: {e}")
```

Then run:
```bash
python run_all.py
```

## Learning Path

### 1. Start with Basics
- Run `01_basic_signatures.py`
- Understand the `"input -> output"` format
- Try creating your own simple signatures

### 2. Move to Typed Signatures
- Run `02_typed_signatures.py`
- Learn about `dspy.InputField` and `dspy.OutputField`
- Practice adding types and descriptions

### 3. Explore Advanced Concepts
- Run `03_advanced_signatures.py`
- Study dynamic and hierarchical patterns
- Understand multi-modal processing

### 4. Apply to Real Problems
- Run `04_real_world_applications.py`
- Adapt examples to your domain
- Focus on the patterns most relevant to you

### 5. Master Composition
- Run `05_signature_composition.py`
- Learn to build complex systems
- Practice combining different approaches

## Key Concepts Covered

### 1. Signature Fundamentals
- **Basic Syntax**: `"input1, input2 -> output1, output2"`
- **Field Naming**: Descriptive, consistent names
- **Module Usage**: `dspy.Predict(signature)`

### 2. Typed Signatures
- **Signature Classes**: Inherit from `dspy.Signature`
- **Field Types**: `dspy.InputField` and `dspy.OutputField`
- **Metadata**: Descriptions, prefixes, constraints

### 3. Advanced Patterns
- **Dynamic Creation**: Building signatures at runtime
- **Hierarchical**: Nested and composed signatures
- **Self-Reference**: Recursive processing patterns

### 4. Domain Applications
- **Business Logic**: Encoding domain knowledge
- **Compliance**: Regulatory requirement handling
- **Personalization**: User-specific adaptations

### 5. Composition Patterns
- **Sequential**: Step-by-step pipelines
- **Parallel**: Independent analyses
- **Conditional**: Adaptive processing
- **Hierarchical**: Multi-level systems

## Common Patterns

### 1. Document Processing Pipeline
```python
# Extract -> Analyze -> Summarize
TextExtractor -> ContentAnalyzer -> DocumentSummarizer
```

### 2. Customer Service Workflow
```python
# Classify -> Route -> Generate Response
TicketClassifier -> RoutingEngine -> ResponseGenerator
```

### 3. Data Analysis Pipeline
```python
# Clean -> Extract -> Analyze -> Report
DataCleaner -> FeatureExtractor -> PatternAnalyzer -> ReportGenerator
```

## Best Practices

### 1. Signature Design
- Be specific with field names
- Include all necessary context
- Structure outputs for easy consumption
- Use consistent terminology

### 2. Error Handling
- Include error outputs in signatures
- Design for edge cases
- Validate input structures
- Provide fallback options

### 3. Performance
- Design signatures for caching
- Minimize unnecessary processing
- Use appropriate data structures
- Consider batch processing

### 4. Testing
- Create test cases for each signature
- Validate output structures
- Test edge cases and errors
- Use consistent test data

## Extending the Examples

### 1. Add Your Domain
- Create signatures for your specific use case
- Adapt real-world examples to your industry
- Share your patterns with the community

### 2. Performance Optimization
- Add caching mechanisms
- Implement batch processing
- Optimize for specific LLM providers

### 3. Integration
- Connect to databases and APIs
- Build web interfaces
- Create automated workflows

## Troubleshooting

### Common Issues

1. **Signature Not Found**
   - Ensure signature is properly defined
   - Check import statements
   - Verify field naming consistency

2. **Type Errors**
   - Match input types to signature expectations
   - Check for missing required fields
   - Verify data structure formats

3. **Performance Issues**
   - Reduce signature complexity
   - Implement caching
   - Use batch processing for multiple items

4. **Poor Results**
   - Add more context to inputs
   - Improve field descriptions
   - Refine output structure

### Getting Help

- Check the DSPy documentation
- Review these examples for similar patterns
- Experiment with different signature designs
- Join the DSPy community forum

## Next Steps

After mastering these examples:

1. **Chapter 3: Modules** - Learn about DSPy modules that use signatures
2. **Chapter 4: Evaluation** - Measure signature performance
3. **Chapter 5: Optimizers** - Automatically improve signatures
4. **Real Projects** - Build complete applications using signatures

Remember: Signatures are the foundation of DSPy. Master them, and the rest becomes much easier!
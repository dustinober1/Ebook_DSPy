# Chapter 8 Examples - Case Studies

This directory contains the complete demonstration code for Chapter 8's four case studies integrated into a unified AI platform.

## Files

- `complete_case_study_demo.py` - Full integration of all four case studies
- `README.md` - This file

## Running the Demo

### Prerequisites

```bash
pip install dspy-ai
```

### Running the Complete Demo

```bash
python complete_case_study_demo.py
```

The demo provides:

1. **Interactive Demo** - Try queries against all four components
2. **Performance Benchmarks** - Measure response times and accuracy
3. **Component-by-Component Demo** - See each case study individually

## Case Studies Integrated

### 1. Enterprise RAG System
- Knowledge retrieval from documents
- Q&A with source citations
- Context-aware answers

### 2. Customer Support Chatbot
- Natural conversation handling
- Intent recognition
- Multi-turn dialogue

### 3. AI-Powered Code Assistant
- Code generation from natural language
- Code explanations
- Programming help

### 4. Automated Data Analysis Pipeline
- Natural language data querying
- Insight generation
- Statistical analysis

## Example Queries

Try these queries in the interactive demo:

### RAG System
- "What is DSPy?"
- "How do signatures work in DSPy?"
- "Tell me about DSPy modules"

### Chatbot
- "Hello, I need help"
- "Where is my order?"
- "I have a technical issue"

### Code Assistant
- "Write a Python function to sort a list"
- "Create a class for a bank account"
- "How do I read a file in Python?"

### Data Analysis
- "Show me sales trends"
- "Analyze customer satisfaction"
- "What are our key metrics?"

## Architecture

The integrated platform follows this flow:

1. **Request Classification** - Automatically determines the appropriate component
2. **Component Routing** - Routes to the correct case study system
3. **Response Generation** - Generates appropriate response
4. **Analytics Collection** - Tracks performance and usage

## Performance Metrics

The demo tracks:
- Total requests processed
- Response times
- Classification accuracy
- User sessions
- Request distribution

## Extensions

To extend this demo:

1. Replace mock implementations with real components
2. Add authentication and user management
3. Implement persistent storage
4. Add more sophisticated routing logic
5. Include additional AI capabilities

## Learning Outcomes

This demonstration teaches:

1. How to integrate multiple AI systems
2. Request classification and routing
3. Performance monitoring
4. User interaction patterns
5. Real-world DSPy applications

## Next Steps

After running this demo:

1. Review the individual case study implementations
2. Try the exercises in `src/08-case-studies/05-exercises.md`
3. Build your own integrated AI solution
4. Explore advanced DSPy features
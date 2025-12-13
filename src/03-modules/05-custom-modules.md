# Custom Modules

## Prerequisites

- **Previous Sections**: [ReAct Agents](./04-react-agents.md) - Understanding of advanced modules
- **Chapter 2**: Signatures - Mastery of signature design
- **Required Knowledge**: Object-oriented programming in Python
- **Difficulty Level**: Advanced
- **Estimated Reading Time**: 50 minutes

## Learning Objectives

By the end of this section, you will:
- Understand how to build custom DSPy modules from scratch
- Master the module lifecycle and internal architecture
- Learn to implement specialized behaviors for unique use cases
- Discover patterns for extensible and reusable modules
- Build production-ready custom modules

## Why Build Custom Modules?

While DSPy provides powerful built-in modules, custom modules allow you to:

1. **Implement unique behaviors** not covered by standard modules
2. **Optimize for specific domains** or use cases
3. **Integrate proprietary systems** or APIs
4. **Add custom preprocessing** or postprocessing logic
5. **Implement specialized reasoning** patterns
6. **Create reusable components** for your organization

## Module Architecture Deep Dive

### Core Module Components

```python
import dspy
from typing import Any, Dict, List, Optional
import inspect

class CustomModule(dspy.Module):
    """Base class showing all components of a DSPy module."""

    def __init__(self, signature, **kwargs):
        super().__init__()

        # 1. Store the signature
        self.signature = signature

        # 2. Configure language model
        self.lm = kwargs.get('lm', dspy.settings.lm)
        self.temperature = kwargs.get('temperature', 0.7)

        # 3. Setup demos (few-shot examples)
        self.demos = kwargs.get('demos', [])

        # 4. Configure instructions
        self.instructions = kwargs.get('instructions', '')

        # 5. Setup cache
        self.cache_enabled = kwargs.get('cache', False)
        self._cache = {} if self.cache_enabled else None

        # 6. Validation
        self.validate_configuration()

        # 7. Initialize components
        self.initialize_components(**kwargs)

    def forward(self, **kwargs) -> dspy.Prediction:
        """Main execution method - override this in subclasses."""

        # 1. Validate inputs
        self.validate_inputs(**kwargs)

        # 2. Check cache
        cache_key = self.get_cache_key(**kwargs)
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        # 3. Preprocess inputs
        processed_inputs = self.preprocess_inputs(**kwargs)

        # 4. Construct prompt
        prompt = self.construct_prompt(**processed_inputs)

        # 5. Call LLM
        response = self.call_llm(prompt)

        # 6. Parse response
        parsed_output = self.parse_response(response)

        # 7. Postprocess
        final_output = self.postprocess_output(parsed_output, **kwargs)

        # 8. Cache result
        if self.cache_enabled:
            self._cache[cache_key] = final_output

        return final_output

    def validate_configuration(self):
        """Validate module configuration."""
        if not self.signature:
            raise ValueError("Signature is required")

    def initialize_components(self, **kwargs):
        """Initialize module-specific components."""
        pass

    def validate_inputs(self, **kwargs):
        """Validate input parameters."""
        # Check all required signature inputs are present
        required_inputs = self.signature.input_fields
        for input_field in required_inputs:
            if input_field.name not in kwargs:
                raise ValueError(f"Missing required input: {input_field.name}")

    def preprocess_inputs(self, **kwargs):
        """Preprocess inputs before prompt construction."""
        return kwargs

    def construct_prompt(self, **kwargs) -> str:
        """Construct the prompt for the LLM."""
        prompt_parts = []

        # Add instructions
        if self.instructions:
            prompt_parts.append(self.instructions)

        # Add demos
        for demo in self.demos:
            prompt_parts.append(self.format_demo(demo))

        # Add current inputs
        prompt_parts.append(self.format_inputs(**kwargs))

        # Add output format guidance
        prompt_parts.append(self.format_output_guidance())

        return "\n\n".join(prompt_parts)

    def format_demo(self, demo) -> str:
        """Format a few-shot example."""
        # Default implementation
        inputs_str = "\n".join([f"{k}: {v}" for k, v in demo.items() if not k.startswith('output_')])
        outputs_str = "\n".join([f"{k}: {v}" for k, v in demo.items() if k.startswith('output_')])
        return f"Example:\n{inputs_str}\n\n{outputs_str}"

    def format_inputs(self, **kwargs) -> str:
        """Format current inputs."""
        return "\n".join([f"{k}: {v}" for k, v in kwargs.items()])

    def format_output_guidance(self) -> str:
        """Add guidance for output formatting."""
        output_fields = self.signature.output_fields
        return f"Provide the output in this format:\n" + \
               "\n".join([f"{field.name}: <{field.name}>" for field in output_fields])

    def call_llm(self, prompt: str) -> str:
        """Call the language model."""
        return self.lm(prompt, temperature=self.temperature)

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response according to signature."""
        # Default parsing - can be overridden
        output_fields = self.signature.output_fields
        parsed = {}

        # Simple line-by-line parsing
        lines = response.strip().split('\n')
        for line in lines:
            for field in output_fields:
                if line.startswith(f"{field.name}:"):
                    parsed[field.name] = line[len(field.name):].strip()

        # Ensure all outputs are present
        for field in output_fields:
            if field.name not in parsed:
                parsed[field.name] = ""  # Default empty value

        return parsed

    def postprocess_output(self, parsed_output: Dict[str, Any], **kwargs) -> dspy.Prediction:
        """Postprocess parsed output."""
        return dspy.Prediction(**parsed_output)

    def get_cache_key(self, **kwargs) -> str:
        """Generate cache key from inputs."""
        import hashlib
        key_str = str(sorted(kwargs.items())) + str(self.temperature)
        return hashlib.md5(key_str.encode()).hexdigest()
```

## Simple Custom Module Example

### Sentiment Analysis with Confidence
```python
class SentimentAnalyzer(dspy.Module):
    """Custom module for sentiment analysis with confidence scoring."""

    def __init__(self, model="sentiment-analysis-v2"):
        # Define signature
        self.signature = dspy.Signature(
            "text -> sentiment, confidence_score, emotional_indicators"
        )

        # Initialize
        super().__init__()

        # Custom initialization
        self.model = model
        self.sentiment_labels = ["positive", "negative", "neutral"]

        # Preload sentiment lexicon
        self.load_sentiment_lexicon()

    def load_sentiment_lexicon(self):
        """Load or create sentiment word lists."""
        self.positive_words = {
            "excellent", "amazing", "wonderful", "fantastic", "great",
            "good", "love", "perfect", "awesome", "brilliant"
        }

        self.negative_words = {
            "terrible", "awful", "horrible", "bad", "poor",
            "hate", "worst", "disgusting", "disappointing", "useless"
        }

    def preprocess_inputs(self, **kwargs):
        """Add sentiment word counts to inputs."""
        text = kwargs.get("text", "")

        # Count sentiment words
        text_lower = text.lower()
        pos_count = sum(1 for word in self.positive_words if word in text_lower)
        neg_count = sum(1 for word in self.negative_words if word in text_lower)

        kwargs["positive_word_count"] = pos_count
        kwargs["negative_word_count"] = neg_count
        kwargs["sentiment_word_ratio"] = pos_count - neg_count

        return kwargs

    def format_inputs(self, **kwargs):
        """Custom input formatting."""
        text = kwargs.get("text", "")
        pos_count = kwargs.get("positive_word_count", 0)
        neg_count = kwargs.get("negative_word_count", 0)

        return f"Text to analyze: {text}\n" + \
               f"Positive words found: {pos_count}\n" + \
               f"Negative words found: {neg_count}\n" + \
               f"Sentiment word score: {pos_count - neg_count}"

    def construct_prompt(self, **kwargs):
        """Custom prompt construction."""
        text = kwargs.get("text", "")

        prompt = f"""Analyze the sentiment of this text:

Text: {text}

Instructions:
1. Determine if the sentiment is positive, negative, or neutral
2. Provide a confidence score from 0.0 to 1.0
3. List emotional indicators (e.g., joy, anger, surprise, fear)

Output format:
sentiment: <positive/negative/neutral>
confidence_score: <0.0-1.0>
emotional_indicators: <list of emotions>
"""
        return prompt

    def postprocess_output(self, parsed_output, **kwargs):
        """Ensure confidence score is valid and normalized."""
        confidence = parsed_output.get("confidence_score", "0.5")

        # Extract numeric value
        if isinstance(confidence, str):
            import re
            match = re.search(r'[\d.]+', confidence)
            if match:
                confidence = float(match.group())
            else:
                confidence = 0.5

        # Ensure within valid range
        confidence = max(0.0, min(1.0, confidence))

        # Adjust based on sentiment word evidence
        pos_count = kwargs.get("positive_word_count", 0)
        neg_count = kwargs.get("negative_word_count", 0)
        word_confidence = (pos_count + neg_count) / (len(kwargs.get("text", "").split()) + 1)

        # Blend model confidence with word evidence
        final_confidence = 0.7 * confidence + 0.3 * min(1.0, word_confidence)

        parsed_output["confidence_score"] = round(final_confidence, 2)

        return dspy.Prediction(**parsed_output)

# Use the custom module
analyzer = SentimentAnalyzer()
result = analyzer(text="I absolutely love this product! It works perfectly and exceeded all my expectations.")
print(f"Sentiment: {result.sentiment}")
print(f"Confidence: {result.confidence_score}")
print(f"Emotions: {result.emotional_indicators}")
```

## Advanced Custom Module - MultiStepProcessor

### Module with Multiple Processing Steps
```python
class MultiStepProcessor(dspy.Module):
    """Module that processes data through multiple customizable steps."""

    def __init__(self, steps: List[Dict[str, Any]], signature: dspy.Signature):
        """
        Initialize with processing steps.

        Args:
            steps: List of step configurations
            signature: DSPy signature for the module
        """
        self.steps = steps
        self.signature = signature
        self.step_results = {}

        # Validate steps
        self.validate_steps()

        # Initialize components
        super().__init__()

    def validate_steps(self):
        """Validate that steps are properly configured."""
        required_keys = ["name", "type", "prompt"]
        for i, step in enumerate(self.steps):
            for key in required_keys:
                if key not in step:
                    raise ValueError(f"Step {i} missing required key: {key}")

    def forward(self, **kwargs):
        """Execute all processing steps sequentially."""
        # Store initial inputs
        self.step_results["initial"] = kwargs.copy()

        # Process each step
        current_data = kwargs.copy()
        for step in self.steps:
            current_data = self.execute_step(step, current_data)
            self.step_results[step["name"]] = current_data.copy()

        # Final formatting according to signature
        return self.format_final_output(current_data)

    def execute_step(self, step: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single processing step."""
        step_type = step["type"]

        if step_type == "transform":
            return self.execute_transform_step(step, data)
        elif step_type == "analyze":
            return self.execute_analyze_step(step, data)
        elif step_type == "filter":
            return self.execute_filter_step(step, data)
        elif step_type == "aggregate":
            return self.execute_aggregate_step(step, data)
        elif step_type == "enrich":
            return self.execute_enrich_step(step, data)
        else:
            raise ValueError(f"Unknown step type: {step_type}")

    def execute_transform_step(self, step: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a transformation step."""
        field_name = step.get("field")
        transformation = step.get("transformation", "uppercase")

        if field_name and field_name in data:
            original_value = str(data[field_name])

            if transformation == "uppercase":
                data[f"{field_name}_transformed"] = original_value.upper()
            elif transformation == "lowercase":
                data[f"{field_name}_transformed"] = original_value.lower()
            elif transformation == "length":
                data[f"{field_name}_length"] = len(original_value)
            elif transformation == "reverse":
                data[f"{field_name}_reversed"] = original_value[::-1]

        return data

    def execute_analyze_step(self, step: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an analysis step using the LLM."""
        analysis_prompt = step["prompt"].format(**data)

        # Use LM for analysis
        analysis_result = self.lm(analysis_prompt)

        data[f"{step['name']}_analysis"] = analysis_result
        return data

    def execute_filter_step(self, step: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a filtering step."""
        condition = step.get("condition", "all")
        field = step.get("field")

        if condition == "non_empty" and field:
            if field in data and data[field]:
                data[f"{field}_passed_filter"] = True
            else:
                data[f"{field}_passed_filter"] = False

        return data

    def execute_aggregate_step(self, step: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an aggregation step."""
        fields = step.get("fields", [])
        operation = step.get("operation", "combine")

        if operation == "combine" and fields:
            combined = []
            for field in fields:
                if field in data:
                    combined.append(str(data[field]))
            data[f"combined_{'_'.join(fields)}"] = " ".join(combined)

        return data

    def execute_enrich_step(self, step: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an enrichment step (add external information)."""
        field = step.get("field")
        enrich_type = step.get("type", "timestamp")

        if field and field in data:
            if enrich_type == "timestamp":
                from datetime import datetime
                data[f"{field}_enriched_at"] = datetime.now().isoformat()
            elif enrich_type == "length":
                data[f"{field}_length"] = len(str(data[field]))
            elif enrich_type == "hash":
                import hashlib
                content = str(data[field])
                data[f"{field}_hash"] = hashlib.md5(content.encode()).hexdigest()

        return data

    def format_final_output(self, data: Dict[str, Any]) -> dspy.Prediction:
        """Format the final output according to signature."""
        output = {}

        # Extract fields that match signature outputs
        for field in self.signature.output_fields:
            if field.name in data:
                output[field.name] = data[field.name]
            else:
                # Try to find related fields
                related = [k for k in data.keys() if field.name.lower() in k.lower()]
                if related:
                    output[field.name] = str(data[related[0]])
                else:
                    output[field.name] = ""  # Default empty value

        return dspy.Prediction(**output)

# Example usage
steps = [
    {
        "name": "text_cleanup",
        "type": "transform",
        "field": "content",
        "transformation": "lowercase"
    },
    {
        "name": "sentiment_check",
        "type": "analyze",
        "prompt": "Analyze the sentiment of this text: {content_transformed}. Is it positive, negative, or neutral?"
    },
    {
        "name": "timestamp",
        "type": "enrich",
        "field": "content",
        "type": "timestamp"
    }
]

signature = dspy.Signature("content -> cleaned_content, sentiment_analysis, processed_at")
processor = MultiStepProcessor(steps, signature)

result = processor(content="This is an AMAZING product! I love it so much!")
print(f"Cleaned: {result.cleaned_content}")
print(f"Sentiment: {result.sentiment_analysis}")
print(f"Processed at: {result.processed_at}")
```

## Domain-Specific Custom Module

### Financial Document Analyzer
```python
class FinancialDocumentAnalyzer(dspy.Module):
    """Specialized module for analyzing financial documents."""

    def __init__(self):
        self.signature = dspy.Signature(
            "document_text, document_type -> financial_metrics, risk_indicators, recommendations"
        )

        # Financial analysis patterns
        self.metric_patterns = {
            "revenue": r"\$[\d,]+\.?\d*\s*(?:million|billion|thousand)",
            "profit_margin": r"profit\s*margin[:\s]*[\d.]+%",
            "growth": r"growth[:\s]*[\d.]+%"
        }

        # Risk keywords
        self.risk_keywords = [
            "debt", "liability", "risk", "decline", "loss",
            "bankruptcy", "default", "delinquent"
        ]

        # Initialize
        super().__init__()

        # Load financial knowledge base
        self.load_financial_knowledge()

    def load_financial_knowledge(self):
        """Load financial analysis rules."""
        self.financial_rules = {
            "healthy_profit_margin": (15, 50),  # min, max percent
            "debt_to_equity": (0, 2),  # ratio range
            "revenue_growth": (5, 100)  # percent
        }

    def preprocess_inputs(self, **kwargs):
        """Extract initial financial metrics from text."""
        document = kwargs.get("document_text", "")

        # Extract metrics using regex
        extracted_metrics = self.extract_financial_metrics(document)
        kwargs["extracted_metrics"] = extracted_metrics

        # Calculate initial risk score
        risk_score = self.calculate_risk_score(document)
        kwargs["initial_risk_score"] = risk_score

        return kwargs

    def extract_financial_metrics(self, text: str) -> Dict[str, List[str]]:
        """Extract financial metrics from text."""
        import re
        metrics = {}

        for metric, pattern in self.metric_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                metrics[metric] = matches

        return metrics

    def calculate_risk_score(self, text: str) -> float:
        """Calculate initial risk score based on keyword presence."""
        text_lower = text.lower()
        risk_word_count = sum(1 for word in self.risk_keywords if word in text_lower)
        total_words = len(text.split())

        # Normalize by document length
        risk_score = min(1.0, (risk_word_count / total_words) * 100)
        return risk_score

    def construct_prompt(self, **kwargs):
        """Construct specialized financial analysis prompt."""
        document = kwargs.get("document_text", "")
        doc_type = kwargs.get("document_type", "unknown")
        metrics = kwargs.get("extracted_metrics", {})
        risk_score = kwargs.get("initial_risk_score", 0)

        prompt = f"""As a financial analyst, analyze this {doc_type} document:

Document:
{document}

Initial Analysis:
- Extracted Metrics: {metrics}
- Risk Indicators Score: {risk_score:.2f}

Please provide:
1. Key Financial Metrics (with values if found)
2. Risk Indicators (high/medium/low with reasons)
3. Recommendations (actionable insights)

Consider standard financial benchmarks:
- Healthy profit margin: 15-50%
- Debt-to-equity ratio should be < 2
- Revenue growth should be positive

Output format:
financial_metrics: <structured financial metrics>
risk_indicators: <risk assessment with details>
recommendations: <numbered list of recommendations>
"""
        return prompt

    def postprocess_output(self, parsed_output, **kwargs):
        """Enhance output with calculated values."""
        # Add initial metrics to output
        if "extracted_metrics" in kwargs:
            # Convert to string for display
            metrics_str = "\n".join([
                f"{k}: {', '.join(v)}" for k, v in kwargs["extracted_metrics"].items()
            ])

            if "financial_metrics" in parsed_output:
                parsed_output["financial_metrics"] = f"Extracted:\n{metrics_str}\n\nAnalyzed:\n{parsed_output['financial_metrics']}"

        # Add risk scoring context
        initial_score = kwargs.get("initial_risk_score", 0)
        if "risk_indicators" in parsed_output:
            parsed_output["risk_indicators"] = f"Text Analysis Score: {initial_score:.2f}\n{parsed_output['risk_indicators']}"

        return dspy.Prediction(**parsed_output)

# Use the financial analyzer
analyzer = FinancialDocumentAnalyzer()
result = analyzer(
    document_text="Quarterly report shows revenue of $5.2 million with profit margin of 18%. "
                  "Company has $3 million in debt but shows steady growth of 12%.",
    document_type="quarterly_report"
)

print(f"Metrics: {result.financial_metrics}")
print(f"Risks: {result.risk_indicators}")
print(f"Recommendations: {result.recommendations}")
```

## Module Composition Patterns

### Module Wrapper for Existing Functions
```python
def create_module_from_function(func, signature):
    """Create a DSPy module from any Python function."""

    class FunctionModule(dspy.Module):
        def __init__(self):
            self.func = func
            self.signature = signature
            super().__init__()

        def forward(self, **kwargs):
            # Extract signature inputs
            sig_inputs = {field.name: kwargs.get(field.name)
                          for field in self.signature.input_fields
                          if field.name in kwargs}

            # Call the function
            result = self.func(**sig_inputs)

            # Prepare output
            if isinstance(result, dict):
                return dspy.Prediction(**result)
            else:
                # Single output
                output_field = self.signature.output_fields[0]
                return dspy.Prediction(**{output_field.name: result})

    return FunctionModule()

# Example: Wrap a text processing function
def process_text(text: str, operation: str) -> str:
    """Simple text processing function."""
    if operation == "uppercase":
        return text.upper()
    elif operation == "lowercase":
        return text.lower()
    elif operation == "reverse":
        return text[::-1]
    else:
        return text

# Create module from function
text_processor = create_module_from_function(
    process_text,
    dspy.Signature("text, operation -> processed_text")
)

result = text_processor(text="Hello World", operation="uppercase")
print(result.processed_text)  # "HELLO WORLD"
```

## Testing Custom Modules

### Unit Testing Framework
```python
import unittest

class TestSentimentAnalyzer(unittest.TestCase):
    """Test suite for custom SentimentAnalyzer module."""

    def setUp(self):
        self.analyzer = SentimentAnalyzer()

    def test_positive_sentiment(self):
        """Test analysis of positive text."""
        result = self.analyzer(text="This is absolutely wonderful!")

        self.assertEqual(result.sentiment, "positive")
        self.assertGreater(result.confidence_score, 0.5)

    def test_negative_sentiment(self):
        """Test analysis of negative text."""
        result = self.analyzer(text="This is terrible and awful.")

        self.assertEqual(result.sentiment, "negative")
        self.assertGreater(result.confidence_score, 0.5)

    def test_confidence_range(self):
        """Test that confidence is always in valid range."""
        for text in ["Good", "Bad", "Neutral", "Amazing", "Terrible"]:
            result = self.analyzer(text=text)
            self.assertGreaterEqual(result.confidence_score, 0.0)
            self.assertLessEqual(result.confidence_score, 1.0)

    def test_empty_text(self):
        """Test handling of empty text."""
        result = self.analyzer(text="")

        self.assertIn(result.sentiment, ["positive", "negative", "neutral"])
        self.assertIsInstance(result.confidence_score, float)

# Run tests
if __name__ == "__main__":
    unittest.main()
```

## Best Practices for Custom Modules

### 1. Documentation
```python
class WellDocumentedModule(dspy.Module):
    """Example of a well-documented custom module.

    This module processes text and provides multiple analyses. It demonstrates:
    - Clear docstring explaining purpose
    - Type hints for better IDE support
    - Detailed parameter documentation
    - Example usage
    """

    def __init__(self,
                 analysis_types: List[str] = None,
                 confidence_threshold: float = 0.5):
        """
        Initialize the module.

        Args:
            analysis_types: List of analyses to perform
            confidence_threshold: Minimum confidence for outputs

        Example:
            >>> module = WellDocumentedModule(analysis_types=["sentiment", "topic"])
            >>> result = module(text="Sample text")
            >>> print(result.sentiment)
        """
```

### 2. Error Handling
```python
class RobustModule(dspy.Module):
    """Module with comprehensive error handling."""

    def forward(self, **kwargs):
        try:
            # Main processing
            result = self.process(**kwargs)

            # Validate output
            self.validate_output(result)

            return result

        except ValueError as e:
            # Handle expected errors gracefully
            return self.handle_error(e, **kwargs)

        except Exception as e:
            # Log unexpected errors
            self.log_unexpected_error(e)
            return self.get_fallback_output(**kwargs)

    def handle_error(self, error: ValueError, **kwargs) -> dspy.Prediction:
        """Handle expected errors with meaningful fallbacks."""
        return dspy.Prediction(
            error=str(error),
            confidence=0.0,
            status="error"
        )

    def validate_output(self, output: dspy.Prediction):
        """Validate output meets requirements."""
        # Implement validation logic
        pass
```

### 3. Configuration Management
```python
class ConfigurableModule(dspy.Module):
    """Module with flexible configuration."""

    def __init__(self, config: Dict[str, Any] = None):
        # Load default configuration
        self.config = self.load_default_config()

        # Override with provided config
        if config:
            self.config.update(config)

        # Validate configuration
        self.validate_config()

    def load_default_config(self) -> Dict[str, Any]:
        """Load default module configuration."""
        return {
            "temperature": 0.7,
            "max_tokens": 1000,
            "cache_enabled": True,
            "timeout": 30,
            "retry_attempts": 3
        }

    def validate_config(self):
        """Validate module configuration."""
        required_keys = ["temperature"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration: {key}")
```

## Summary

Custom modules enable:

- **Complete control** over module behavior
- **Domain optimization** for specific use cases
- **Integration capabilities** with existing systems
- **Reusability** across projects
- **Testing and validation** of custom logic

### Key Takeaways

1. **Understand the module lifecycle** - Initialize → Validate → Preprocess → Prompt → LLM → Parse → Postprocess
2. **Override carefully** - Only override methods you need to customize
3. **Add validation** - Ensure inputs and outputs are correct
4. **Document thoroughly** - Your modules will be used by others
5. **Test comprehensively** - Unit tests catch bugs early

## Next Steps

- [Module Composition](./06-composing-modules.md) - Combine modules effectively
- [Practical Examples](../examples/chapter03/) - See custom modules in action
- [Exercises](./07-exercises.md) - Build your own custom modules
- [Optimizers](../05-optimizers.md) - Automatically improve custom modules

## Further Reading

- [DSPy Module Source Code](https://github.com/stanfordnlp/dspy/tree/main/dspy) - Learn from the implementation
- [Design Patterns](../07-advanced-topics.md) - Advanced module patterns
- [Testing Strategies](../09-appendices/testing.md) - Comprehensive testing approaches
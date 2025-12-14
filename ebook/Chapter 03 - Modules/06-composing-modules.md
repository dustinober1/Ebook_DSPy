# Composing Modules

## Prerequisites

- **Previous Sections**: [Custom Modules](./05-custom-modules.md) - Understanding of module creation
- **Chapter 2**: Signatures - Mastery of signature design
- **Required Knowledge**: Understanding of software design patterns
- **Difficulty Level**: Advanced
- **Estimated Reading Time**: 40 minutes

## Learning Objectives

By the end of this section, you will:
- Master patterns for composing multiple DSPy modules
- Learn to build complex workflows from simple components
- Understand pipeline, chain, and parallel composition patterns
- Discover how to optimize module composition for performance
- Build sophisticated multi-module systems

## Introduction to Module Composition

Module composition is the art of combining multiple DSPy modules to create powerful, specialized systems. Just as functions can be composed to form complex programs, DSPy modules can be composed to create sophisticated LLM applications.

### Composition Patterns

1. **Sequential/Pipeline Composition** - Pass output of one module to next
2. **Parallel Composition** - Run multiple modules simultaneously
3. **Conditional Composition** - Choose module based on conditions
4. **Hierarchical Composition** - Nested modules for complex logic
5. **Feedback Loops** - Modules that iteratively refine outputs

## Sequential Composition

### Basic Pipeline Pattern
```python
import dspy

# Define individual modules
class TextCleaner(dspy.Module):
    def __init__(self):
        self.signature = dspy.Signature("raw_text -> cleaned_text")

    def forward(self, raw_text):
        # Simple cleaning logic
        cleaned = raw_text.strip().lower()
        return dspy.Prediction(cleaned_text=cleaned)

class TextAnalyzer(dspy.Module):
    def __init__(self):
        self.analyzer = dspy.Predict(
            "cleaned_text -> sentiment, key_topics, entities"
        )

    def forward(self, cleaned_text):
        return self.analyzer(cleaned_text=cleaned_text)

class ReportGenerator(dspy.Module):
    def __init__(self):
        self.generator = dspy.Predict(
            "sentiment, key_topics, entities -> report"
        )

    def forward(self, sentiment, key_topics, entities):
        return self.generator(
            sentiment=sentiment,
            key_topics=key_topics,
            entities=entities
        )

# Compose into a pipeline
class TextAnalysisPipeline(dspy.Module):
    """Pipeline that combines text cleaning, analysis, and reporting."""

    def __init__(self):
        super().__init__()
        self.cleaner = TextCleaner()
        self.analyzer = TextAnalyzer()
        self.generator = ReportGenerator()

    def forward(self, raw_text):
        # Step 1: Clean text
        cleaned_result = self.cleaner(raw_text=raw_text)
        cleaned_text = cleaned_result.cleaned_text

        # Step 2: Analyze text
        analysis_result = self.analyzer(cleaned_text=cleaned_text)

        # Step 3: Generate report
        report_result = self.generator(
            sentiment=analysis_result.sentiment,
            key_topics=analysis_result.key_topics,
            entities=analysis_result.entities
        )

        # Combine all results
        return dspy.Prediction(
            cleaned_text=cleaned_text,
            sentiment=analysis_result.sentiment,
            key_topics=analysis_result.key_topics,
            entities=analysis_result.entities,
            report=report_result.report
        )

# Use the pipeline
pipeline = TextAnalysisPipeline()
result = pipeline(raw_text="  This is an AMAZING product! I love how it works perfectly.  ")

print(f"Report: {result.report}")
```

### Advanced Pipeline with Error Handling
```python
class RobustPipeline(dspy.Module):
    """Pipeline with error handling and fallbacks."""

    def __init__(self, modules: List[dspy.Module], fallbacks: Dict[int, dspy.Module] = None):
        """
        Initialize pipeline with modules and fallbacks.

        Args:
            modules: List of modules in execution order
            fallbacks: Dictionary mapping module index to fallback module
        """
        super().__init__()
        self.modules = modules
        self.fallbacks = fallbacks or {}
        self.module_outputs = {}

    def forward(self, **kwargs):
        """Execute pipeline with error handling."""
        current_input = kwargs.copy()

        for i, module in enumerate(self.modules):
            try:
                # Execute module
                result = module(**current_input)
                self.module_outputs[i] = result

                # Extract outputs for next module
                current_input = self.extract_outputs(result, i)

            except Exception as e:
                print(f"Module {i} failed: {e}")

                # Try fallback if available
                if i in self.fallbacks:
                    print(f"Using fallback for module {i}")
                    fallback_result = self.fallbacks[i](**current_input)
                    self.module_outputs[i] = fallback_result
                    current_input = self.extract_outputs(fallback_result, i)
                else:
                    # Skip this module
                    print(f"No fallback for module {i}, skipping")
                    continue

        return dspy.Prediction(**self.module_outputs)

    def extract_outputs(self, result: dspy.Prediction, module_index: int) -> Dict[str, Any]:
        """Extract outputs from module result for next module."""
        # Get module signature
        if hasattr(self.modules[module_index], 'signature'):
            output_fields = self.modules[module_index].signature.output_fields
            return {field.name: getattr(result, field.name, None)
                    for field in output_fields}
        else:
            # Fallback: return all attributes
            return {k: v for k, v in result.__dict__.items() if not k.startswith('_')}
```

## Parallel Composition

### Parallel Module Execution
```python
class ParallelProcessor(dspy.Module):
    """Execute multiple modules in parallel."""

    def __init__(self, modules: List[dspy.Module], combine_mode: str = "merge"):
        """
        Initialize parallel processor.

        Args:
            modules: List of modules to execute in parallel
            combine_mode: How to combine outputs ("merge", "vote", "select")
        """
        super().__init__()
        self.modules = modules
        self.combine_mode = combine_mode

    def forward(self, **kwargs):
        """Execute all modules in parallel."""
        from concurrent.futures import ThreadPoolExecutor
        import time

        start_time = time.time()

        # Execute modules in parallel
        with ThreadPoolExecutor(max_workers=len(self.modules)) as executor:
            futures = []
            for i, module in enumerate(self.modules):
                future = executor.submit(module, **kwargs)
                futures.append((i, future))

            # Collect results
            results = {}
            for i, future in futures:
                try:
                    result = future.result(timeout=30)
                    results[f"module_{i}"] = result
                except Exception as e:
                    print(f"Module {i} failed: {e}")
                    results[f"module_{i}"] = None

        execution_time = time.time() - start_time

        # Combine results based on mode
        combined = self.combine_results(results)

        # Add metadata
        combined['parallel_metadata'] = {
            'execution_time': execution_time,
            'modules_run': len(self.modules),
            'successful_modules': sum(1 for r in results.values() if r is not None)
        }

        return dspy.Prediction(**combined)

    def combine_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from multiple modules."""
        if self.combine_mode == "merge":
            return self.merge_results(results)
        elif self.combine_mode == "vote":
            return self.vote_results(results)
        elif self.combine_mode == "select":
            return self.select_best_result(results)
        else:
            return {"combined_results": results}

    def merge_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Merge all results into one dictionary."""
        merged = {}
        for name, result in results.items():
            if result:
                for key, value in result.__dict__.items():
                    if not key.startswith('_'):
                        merged[f"{name}_{key}"] = value
        return merged

    def vote_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Vote on categorical outputs."""
        votes = {}
        for name, result in results.items():
            if result and hasattr(result, 'prediction'):
                pred = result.prediction
                if pred not in votes:
                    votes[pred] = []
                votes[pred].append(name)

        # Find most common prediction
        if votes:
            winning_pred = max(votes.keys(), key=lambda k: len(votes[k]))
            return {
                "prediction": winning_pred,
                "vote_counts": {k: len(v) for k, v in votes.items()},
                "confidence": len(votes[winning_pred]) / len(votes)
            }

        return {"prediction": None, "vote_counts": {}, "confidence": 0.0}

    def select_best_result(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best result based on confidence scores."""
        best_result = None
        best_confidence = -1

        for name, result in results.items():
            if result and hasattr(result, 'confidence'):
                if result.confidence > best_confidence:
                    best_result = result
                    best_confidence = result.confidence

        if best_result:
            best_result["selected_from"] = len([r for r in results.values() if r])
            return best_result.__dict__
        else:
            return {"error": "No valid results found"}
```

### Specialized Parallel Patterns

#### Ensemble Classifier
```python
class EnsembleClassifier(dspy.Module):
    """Ensemble of classifiers that vote on predictions."""

    def __init__(self, classifier_configs: List[Dict[str, Any]]):
        """
        Initialize ensemble with multiple classifier configurations.

        Args:
            classifier_configs: List of configs for individual classifiers
        """
        super().__init__()
        self.classifiers = []
        self.weights = []

        for config in classifier_configs:
            # Create classifier
            signature = dspy.Signature(config['signature'])
            classifier = dspy.Predict(signature, **config.get('params', {}))
            self.classifiers.append(classifier)
            self.weights.append(config.get('weight', 1.0))

    def forward(self, text: str) -> dspy.Prediction:
        """Get ensemble prediction."""
        predictions = []
        confidences = []

        # Get predictions from all classifiers
        for classifier in self.classifiers:
            result = classifier(text=text)
            predictions.append(result.prediction)
            confidences.append(getattr(result, 'confidence', 0.5))

        # Weighted voting
        weighted_votes = {}
        for pred, conf, weight in zip(predictions, confidences, self.weights):
            score = conf * weight
            if pred not in weighted_votes:
                weighted_votes[pred] = 0
            weighted_votes[pred] += score

        # Find winner
        winner = max(weighted_votes.keys(), key=weighted_votes.get)
        total_score = sum(weighted_votes.values())
        confidence = weighted_votes[winner] / total_score if total_score > 0 else 0.5

        return dspy.Prediction(
            prediction=winner,
            confidence=confidence,
            all_predictions=predictions,
            vote_breakdown=weighted_votes
        )

# Use ensemble classifier
ensemble = EnsembleClassifier([
    {
        'signature': 'text -> prediction, confidence',
        'params': {'temperature': 0.1},
        'weight': 2.0
    },
    {
        'signature': 'text -> prediction, confidence',
        'params': {'temperature': 0.3},
        'weight': 1.5
    },
    {
        'signature': 'text -> prediction, confidence',
        'params': {'temperature': 0.5},
        'weight': 1.0
    }
])

result = ensemble(text="This product is absolutely fantastic!")
print(f"Ensemble prediction: {result.prediction} (confidence: {result.confidence:.2f})")
```

## Conditional Composition

### Router Module
```python
class Router(dspy.Module):
    """Route inputs to different modules based on conditions."""

    def __init__(self, routes: Dict[str, dspy.Module], default_route: str = None):
        """
        Initialize router.

        Args:
            routes: Dictionary mapping route names to modules
            default_route: Default route if no condition matches
        """
        super().__init__()
        self.routes = routes
        self.default_route = default_route or list(routes.keys())[0]

    def forward(self, **kwargs):
        """Route to appropriate module based on input conditions."""
        # Determine route
        route_name = self.determine_route(**kwargs)

        # Get module
        module = self.routes.get(route_name, self.routes[self.default_route])

        # Execute module
        result = module(**kwargs)

        # Add routing information
        result.route_used = route_name

        return result

    def determine_route(self, **kwargs) -> str:
        """Determine which route to use based on inputs."""
        text = kwargs.get('text', '').lower()

        # Simple routing logic
        if any(word in text for word in ['buy', 'purchase', 'price']):
            return 'commerce'
        elif any(word in text for word in ['help', 'support', 'issue']):
            return 'support'
        elif any(word in text for word in ['what', 'how', 'why']):
            return 'question'
        else:
            return self.default_route

# Create routing system
router = Router(
    routes={
        'commerce': dspy.Predict("text -> category, intent"),
        'support': dspy.Predict("text -> issue_type, priority"),
        'question': dspy.Predict("text -> answer")
    },
    default_route='general'
)

# Test routing
result1 = router(text="I want to buy your product")
print(f"Route: {result1.route_used}, Category: {result1.category}")

result2 = router(text="How does this work?")
print(f"Route: {result2.route_used}, Answer: {result2.answer}")
```

### Adaptive Module
```python
class AdaptiveModule(dspy.Module):
    """Module that adapts its behavior based on input complexity."""

    def __init__(self):
        super().__init__()
        self.simple_module = dspy.Predict("query -> answer")
        self.complex_module = dspy.ChainOfThought("query -> reasoning, answer")

    def forward(self, query: str) -> dspy.Prediction:
        """Choose module based on query complexity."""
        complexity = self.assess_complexity(query)

        if complexity < 0.5:
            # Use simple module for easy queries
            result = self.simple_module(query=query)
            result.processing_type = "simple"
        else:
            # Use reasoning module for complex queries
            result = self.complex_module(query=query)
            result.processing_type = "complex"

        result.complexity_score = complexity
        return result

    def assess_complexity(self, query: str) -> float:
        """Assess query complexity (0-1)."""
        # Simple heuristic
        complexity_indicators = [
            len(query.split()) / 20,  # Word count
            len([c for c in query if c.isupper()]) / len(query),  # Capitals
            len(query.count('?') + query.count('!')) / len(query)  # Punctuation
        ]

        return min(1.0, sum(complexity_indicators) / 3)

# Use adaptive module
adaptive = AdaptiveModule()

simple_result = adaptive(query="What time is it?")
print(f"Type: {simple_result.processing_type}, Answer: {simple_result.answer}")

complex_result = adaptive(query="Explain the economic implications of inflation on small businesses")
print(f"Type: {complex_result.processing_type}, Confidence: {complex_result.complexity_score:.2f}")
```

## Hierarchical Composition

### Multi-Level Analysis System
```python
class DocumentAnalyzer(dspy.Module):
    """Multi-level document analysis system."""

    def __init__(self):
        super().__init__()

        # Level 1: Initial analysis
        self.level1 = dspy.Predict("document -> summary, key_points")

        # Level 2: Deep analysis based on Level 1 results
        self.level2_classifier = Router(
            routes={
                'factual': dspy.Predict("document, summary -> factual_analysis"),
                'opinion': dspy.Predict("document, summary -> opinion_analysis"),
                'mixed': dspy.ChainOfThought("document, summary -> detailed_analysis")
            }
        )

        # Level 3: Specialized analysis
        self.level3_modules = {
            'technical': dspy.Predict("document, detailed_analysis -> technical_insights"),
            'legal': dspy.Predict("document, detailed_analysis -> legal_considerations"),
            'business': dspy.Predict("document, detailed_analysis -> business_impact")
        }

    def forward(self, document: str, document_type: str = None) -> dspy.Prediction:
        """Perform multi-level analysis."""
        # Level 1: Basic analysis
        level1_result = self.level1(document=document)

        # Level 2: Determine document type and analyze
        doc_type = document_type or self.classify_document(document)
        level2_result = self.level2_classifier(
            document=document,
            summary=level1_result.summary
        )

        # Level 3: Specialized analysis if available
        level3_result = None
        if doc_type in self.level3_modules:
            level3_result = self.level3_modules[doc_type](
                document=document,
                detailed_analysis=getattr(level2_result, level2_result.__class__.__name__.lower(), '')
            )

        # Combine all results
        final_result = {
            'summary': level1_result.summary,
            'key_points': level1_result.key_points,
            'document_type': doc_type,
            'level2_analysis': level2_result,
            'level3_analysis': level3_result
        }

        return dspy.Prediction(**final_result)

    def classify_document(self, document: str) -> str:
        """Classify document type."""
        text_lower = document.lower()
        indicators = {
            'technical': ['code', 'algorithm', 'implementation', 'programming'],
            'legal': ['contract', 'agreement', 'liability', 'jurisdiction'],
            'business': ['revenue', 'profit', 'market', 'strategy']
        }

        scores = {doc_type: sum(1 for indicator in indicators if indicator in text_lower)
                 for doc_type, indicators in indicators.items()}

        return max(scores.keys(), key=scores.get) if scores else 'general'

# Use hierarchical analyzer
analyzer = DocumentAnalyzer()
result = analyzer(
    document="The code implements a sorting algorithm using Python. It includes error handling and unit tests. "
            "The implementation is covered by an MIT license.",
    document_type="technical"
)

print(f"Summary: {result.summary}")
print(f"Document Type: {result.document_type}")
```

## Feedback Loop Composition

### Iterative Refinement Module
```python
class IterativeRefiner(dspy.Module):
    """Module that iteratively refines outputs."""

    def __init__(self, base_module, refinement_module, max_iterations: int = 3):
        """
        Initialize iterative refiner.

        Args:
            base_module: Module to generate initial output
            refinement_module: Module to refine outputs
            max_iterations: Maximum number of refinement iterations
        """
        super().__init__()
        self.base_module = base_module
        self.refinement_module = refinement_module
        self.max_iterations = max_iterations

    def forward(self, **kwargs):
        """Generate and iteratively refine output."""
        # Generate initial output
        current_output = self.base_module(**kwargs)

        # Iteratively refine
        for iteration in range(self.max_iterations):
            # Check if refinement is needed
            if self.is_satisfactory(current_output):
                break

            # Refine current output
            refinement_prompt = self.create_refinement_prompt(
                current_output, iteration, **kwargs
            )

            refined = self.refinement_module(
                original=current_output,
                refinement_prompt=refinement_prompt,
                iteration=iteration + 1
            )

            # Update output
            current_output = self.merge_outputs(current_output, refined)

        # Add iteration info
        current_output.iterations = iteration + 1

        return current_output

    def is_satisfactory(self, output: dspy.Prediction) -> bool:
        """Check if output meets quality criteria."""
        # Check confidence if available
        if hasattr(output, 'confidence'):
            return output.confidence >= 0.9
        return True

    def create_refinement_prompt(self, output: dspy.Prediction, iteration: int, **kwargs) -> str:
        """Create prompt for refinement."""
        if iteration == 0:
            return "Please refine this output to be more detailed and comprehensive."
        elif iteration == 1:
            return "Please improve clarity and add more examples."
        else:
            return "Please review and polish the output for final delivery."

    def merge_outputs(self, original: dspy.Prediction, refined: dspy.Prediction) -> dspy.Prediction:
        """Merge original and refined outputs."""
        # Use refined output but keep metadata from original
        merged = refined.__dict__.copy()
        if hasattr(original, 'confidence'):
            merged['original_confidence'] = original.confidence
        return dspy.Prediction(**merged)

# Create iterative refiner
base = dspy.Predict("prompt -> response")
refiner = dspy.ChainOfThought("original, refinement_prompt -> refined_response")

iterative_module = IterativeRefiner(base, refiner)

result = iterative_module(
    prompt="Explain quantum computing"
)
print(f"Final response after {result.iterations} iterations")
```

## Performance Optimization

### Lazy Evaluation
```python
class LazyComposer(dspy.Module):
    """Composer that lazily evaluates modules only when needed."""

    def __init__(self, modules: List[dspy.Module]):
        super().__init__()
        self.modules = modules
        self._results_cache = {}

    def forward(self, required_outputs: List[str], **kwargs):
        """Execute only modules needed for required outputs."""
        # Map outputs to modules
        output_to_module = self.map_outputs_to_modules(required_outputs)

        # Execute required modules
        executed = []
        for module_name in output_to_module.values():
            if module_name not in executed:
                module = getattr(self, module_name)
                result = module(**kwargs)
                self._results_cache[module_name] = result
                executed.append(module_name)

        # Return only required outputs
        return self.extract_required_outputs(required_outputs, **kwargs)

    def map_outputs_to_modules(self, required_outputs: List[str]) -> Dict[str, str]:
        """Map required outputs to module names."""
        mapping = {
            'summary': 'summarizer',
            'sentiment': 'sentiment_analyzer',
            'topics': 'topic_extractor',
            'entities': 'entity_recognizer'
        }

        return {output: mapping.get(output, 'default_module')
                for output in required_outputs
                if output in mapping}
```

### Batch Processing
```python
class BatchProcessor(dspy.Module):
    """Process multiple inputs efficiently in batches."""

    def __init__(self, module: dspy.Module, batch_size: int = 10):
        super().__init__()
        self.module = module
        self.batch_size = batch_size

    def forward(self, inputs: List[Dict[str, Any]]) -> List[dspy.Prediction]:
        """Process inputs in batches."""
        results = []

        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:i + self.batch_size]

            # Process batch
            batch_results = self.process_batch(batch)
            results.extend(batch_results)

        return results

    def process_batch(self, batch: List[Dict[str, Any]]) -> List[dspy.Prediction]:
        """Process a single batch."""
        # This could be optimized to use parallel processing
        return [self.module(**item) for item in batch]
```

## Best Practices

### 1. Design for Testability
```python
class TestableComposer(dspy.Module):
    """Composer designed for easy testing."""

    def __init__(self):
        super().__init__()
        # Use dependency injection
        self.module1 = self.create_module1()
        self.module2 = self.create_module2()

    def create_module1(self):
        """Factory method for module1 (can be overridden in tests)."""
        return dspy.Predict("text -> analysis")

    def create_module2(self):
        """Factory method for module2 (can be overridden in tests)."""
        return dspy.Predict("analysis -> report")

    def forward(self, text: str):
        # Intermediate results can be inspected
        intermediate = self.module1(text=text)
        final = self.module2(analysis=intermediate.analysis)
        return final
```

### 2. Handle Failure Gracefully
```python
class ResilientComposer(dspy.Module):
    """Composer that handles module failures."""

    def __init__(self, modules: List[dspy.Module]):
        super().__init__()
        self.modules = modules
        self.fallback_modules = self.create_fallbacks()

    def forward(self, **kwargs):
        results = {}
        errors = []

        for i, module in enumerate(self.modules):
            try:
                result = module(**{k: v for k, v in kwargs.items()
                                 if self.module_needs_input(module, k)})
                results[f"module_{i}"] = result

            except Exception as e:
                errors.append(f"Module {i}: {e}")
                if i in self.fallback_modules:
                    try:
                        fallback_result = self.fallback_modules[i](**kwargs)
                        results[f"module_{i}_fallback"] = fallback_result
                    except Exception as fallback_error:
                        errors.append(f"Module {i} fallback failed: {fallback_error}")

        return dspy.Prediction(results=results, errors=errors)
```

### 3. Use Type Hints
```python
from typing import Dict, List, Optional, Union
from dspy import Module, Prediction

class TypedComposer(Module):
    """Composer with full type annotations."""

    def __init__(self) -> None:
        super().__init__()
        self.preprocessor: Module = self._create_preprocessor()
        self.analyzer: Module = self._create_analyzer()

    def _create_preprocessor(self) -> Module:
        return dspy.Predict("raw_text -> processed_text")

    def _create_analyzer(self) -> Module:
        return dspy.Predict("processed_text -> analysis, confidence")

    def forward(self, raw_text: str) -> Prediction:
        """Process text with type safety."""
        # Preprocess
        pre_result: Prediction = self.preprocessor(raw_text=raw_text)

        # Analyze
        analysis_result: Prediction = self.analyzer(
            processed_text=pre_result.processed_text
        )

        return Prediction(
            processed_text=pre_result.processed_text,
            analysis=analysis_result.analysis,
            confidence=analysis_result.confidence
        )
```

## Summary

Module composition enables:

- **Complex workflows** from simple modules
- **Flexible architectures** that adapt to needs
- **Optimized execution** through parallel and lazy evaluation
- **Error resilience** with fallbacks and retries
- **Testable and maintainable** code structures

## Advanced Composition Patterns

### Section-by-Section Writing Pattern

The section-by-section writing pattern is essential for generating long-form content like articles, reports, or documentation. This pattern maintains context and coherence while generating content piece by piece.

```python
class SectionBySectionWriter(dspy.Module):
    """Writes long-form content section by section with context management."""

    def __init__(self, section_generator: dspy.Module, max_context_sections: int = 3):
        """
        Initialize section-by-section writer.

        Args:
            section_generator: Module that generates individual sections
            max_context_sections: Number of previous sections to keep in context
        """
        super().__init__()
        self.section_generator = section_generator
        self.max_context_sections = max_context_sections
        self.generated_sections = []
        self.section_context = {}

    def forward(self, outline: List[Dict], topic: str, **kwargs):
        """
        Generate content section by section following an outline.

        Args:
            outline: Structured outline with section information
            topic: Overall topic for context
            **kwargs: Additional parameters for content generation

        Returns:
            Complete generated content with metadata
        """
        self.generated_sections = []

        # Generate each section in order
        for i, section_info in enumerate(outline):
            # Get context from previous sections
            context = self._get_section_context(i)

            # Generate current section
            section_content = self._generate_section(
                section_info=section_info,
                context=context,
                topic=topic,
                **kwargs
            )

            # Store generated section
            self.generated_sections.append({
                'title': section_info.get('title', f'Section {i+1}'),
                'content': section_content,
                'word_count': len(section_content.split()),
                'section_number': i + 1
            })

        # Combine all sections
        full_content = self._combine_sections()

        return dspy.Prediction(
            content=full_content,
            sections=self.generated_sections,
            total_word_count=sum(s['word_count'] for s in self.generated_sections),
            total_sections=len(self.generated_sections)
        )

    def _get_section_context(self, current_section_index: int) -> str:
        """Get context from recent sections."""
        context_sections = []

        # Include previous sections within context window
        start_index = max(0, current_section_index - self.max_context_sections)

        for i in range(start_index, current_section_index):
            if i < len(self.generated_sections):
                section = self.generated_sections[i]
                context_sections.append(
                    f"Section {section['section_number']}: {section['title']}\n"
                    f"Content: {section['content'][:200]}..."  # First 200 chars
                )

        return "\n\n".join(context_sections) if context_sections else ""

    def _generate_section(self,
                         section_info: Dict,
                         context: str,
                         topic: str,
                         **kwargs) -> str:
        """Generate a single section with appropriate context."""
        # Prepare section-specific prompt
        section_prompt = self._create_section_prompt(
            section_info=section_info,
            context=context,
            topic=topic
        )

        # Generate section content
        result = self.section_generator(
            section_prompt=section_prompt,
            word_limit=section_info.get('word_count', 500),
            **kwargs
        )

        return result.content

    def _create_section_prompt(self,
                              section_info: Dict,
                              context: str,
                              topic: str) -> str:
        """Create a comprehensive prompt for section generation."""
        prompt_parts = [
            f"Topic: {topic}",
            f"Section Title: {section_info.get('title', 'Untitled Section')}",
            f"Section Purpose: {section_info.get('purpose', 'Explain this aspect of the topic')}"
        ]

        if context:
            prompt_parts.append(
                f"\nPrevious Sections Context:\n{context}\n"
                "Ensure your section flows naturally from the previous content."
            )

        if section_info.get('keywords'):
            prompt_parts.append(
                f"\nKeywords to include: {', '.join(section_info['keywords'])}"
            )

        if section_info.get('perspective'):
            prompt_parts.append(
                f"\nWrite from this perspective: {section_info['perspective']}"
            )

        return "\n".join(prompt_parts)

    def _combine_sections(self) -> str:
        """Combine all sections into a coherent document."""
        document_parts = []

        for section in self.generated_sections:
            # Add section title
            document_parts.append(f"\n## {section['title']}\n")

            # Add section content
            document_parts.append(section['content'])

            # Add transition
            if section['section_number'] < len(self.generated_sections):
                next_section = self.generated_sections[section['section_number']]
                transition = self._create_transition(
                    current_section=section,
                    next_section=next_section
                )
                if transition:
                    document_parts.append(f"\n{transition}\n")

        return "".join(document_parts)

    def _create_transition(self, current_section: Dict, next_section: Dict) -> str:
        """Create a smooth transition between sections."""
        transition_generator = dspy.Predict(
            "current_section, next_section -> transition_text"
        )

        result = transition_generator(
            current_section=f"{current_section['title']}: {current_section['content'][-100:]}",
            next_section=next_section['title']
        )

        return result.transition_text


# Example: Using the Section-by-Section Writer
class ArticleSectionGenerator(dspy.Module):
    """Specialized module for generating article sections."""

    def __init__(self):
        super().__init__()
        self.generate_section = dspy.ChainOfThought(
            "section_prompt, word_limit -> content"
        )

    def forward(self, section_prompt: str, word_limit: int = 500) -> dspy.Prediction:
        """Generate content for a single section."""
        result = self.generate_section(
            section_prompt=section_prompt,
            word_limit=str(word_limit)
        )

        # Ensure content meets word limit
        content = result.content
        words = content.split()

        if len(words) > word_limit * 1.2:  # Allow 20% overflow
            content = " ".join(words[:word_limit])
            content += "..."  # Indicate truncation
        elif len(words) < word_limit * 0.8:  # Require at least 80%
            # Expand content if too short
            expander = dspy.Predict(
                "content, target_length -> expanded_content"
            )
            expanded = expander(
                content=content,
                target_length=str(word_limit)
            )
            content = expanded.expanded_content

        return dspy.Prediction(content=content)


# Example usage
outline = [
    {
        'title': 'Introduction',
        'purpose': 'Introduce the topic and outline the article',
        'word_count': 300,
        'keywords': ['overview', 'introduction', 'scope']
    },
    {
        'title': 'Background',
        'purpose': 'Provide necessary background information',
        'word_count': 500,
        'keywords': ['history', 'context', 'foundation']
    },
    {
        'title': 'Main Analysis',
        'purpose': 'Present detailed analysis and findings',
        'word_count': 800,
        'perspective': 'analytical'
    },
    {
        'title': 'Conclusion',
        'purpose': 'Summarize key points and provide future outlook',
        'word_count': 300,
        'keywords': ['summary', 'conclusion', 'future']
    }
]

# Create and use the writer
section_gen = ArticleSectionGenerator()
writer = SectionBySectionWriter(section_gen, max_context_sections=2)

result = writer(
    outline=outline,
    topic="The Impact of AI on Education",
    writing_style="academic"
)

print(f"Generated {result.total_sections} sections")
print(f"Total word count: {result.total_word_count}")
print("\nFirst section preview:")
print(result.sections[0]['content'][:200] + "...")
```

This pattern is particularly useful for:
- **Long-form article generation** where maintaining coherence is crucial
- **Documentation writing** with structured sections
- **Report generation** following specific formats
- **Educational content** with progressive concept building
- **Research synthesis** combining findings from multiple sources

### Key Takeaways

1. **Start simple** - Compose basic modules first
2. **Use patterns** - Follow established composition patterns
3. **Handle failures** - Build resilient systems
4. **Optimize wisely** - Use parallel and batch processing
5. **Document composition** - Make architectural decisions clear

## Next Steps

- [Module Exercises](./07-exercises.md) - Practice composition techniques
- [Practical Examples](../examples/chapter03/) - See composition in action
- [Advanced Topics](../07-advanced-topics.md) - Explore advanced patterns
- [Real-World Applications](../06-real-world-applications/) - Apply to real problems

## Further Reading

- [Design Patterns](https://refactoring.guru/) - General design patterns
- [DSPy GitHub](https://github.com/stanfordnlp/dspy) - Module implementation details
- [Performance Guide](../09-appendices/performance.md) - Optimization techniques
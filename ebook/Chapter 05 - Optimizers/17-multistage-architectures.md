# Multi-stage Program Architectures

## Learning Objectives

By the end of this section, you will be able to:
- Design effective multi-stage language model program architectures
- Implement common architectural patterns for complex tasks
- Optimize inter-stage communication and data flow
- Handle error propagation and recovery in multi-stage systems
- Build scalable and maintainable multi-stage programs

## Introduction

Multi-stage program architectures represent a powerful paradigm for tackling complex language model tasks that cannot be effectively solved in a single pass. By breaking down complex problems into sequential stages, we can:

1. **Modularize complexity**: Each stage focuses on a specific subtask
2. **Improve interpretability**: Individual stages can be analyzed and debugged
3. **Enable specialized optimization**: Different stages can use different strategies
4. **Enhance reusability**: Stages can be reused across different programs
5. **Facilitate parallel development**: Teams can work on different stages independently

This section explores architectural patterns, design principles, and implementation strategies for building robust multi-stage programs in DSPy.

## Architectural Patterns

### 1. Sequential Pipeline Architecture

The most common pattern where stages process data in a linear sequence.

```
Input → Stage 1 → Stage 2 → Stage 3 → ... → Stage N → Output
```

#### Implementation

```python
import dspy
from typing import List, Any, Dict, Optional

class SequentialPipeline(dspy.Module):
    """Sequential multi-stage pipeline."""

    def __init__(self, stages: List[dspy.Module]):
        super().__init__()
        self.stages = stages
        self.stage_names = [f"stage_{i}" for i in range(len(stages))]

    def forward(self, **kwargs) -> dspy.Prediction:
        """Forward pass through all stages."""

        current_input = kwargs
        stage_outputs = {}

        for i, stage in enumerate(self.stages):
            # Execute stage
            stage_name = self.stage_names[i]
            output = stage(**current_input)

            # Store output for debugging
            stage_outputs[stage_name] = output

            # Prepare input for next stage
            if hasattr(output, 'predictions'):
                current_input.update(output.predictions)
            else:
                current_input = output

        # Combine all outputs
        return dspy.Prediction(
            output=current_input,
            stage_outputs=stage_outputs,
            trace=stage_outputs
        )

    def add_stage(self, stage: dspy.Module, position: Optional[int] = None):
        """Add a new stage to the pipeline."""

        if position is None:
            self.stages.append(stage)
            self.stage_names.append(f"stage_{len(self.stages)-1}")
        else:
            self.stages.insert(position, stage)
            self.stage_names.insert(position, f"stage_{position}")
            # Rename subsequent stages
            for i in range(position + 1, len(self.stages)):
                self.stage_names[i] = f"stage_{i}"
```

#### Example: Multi-hop Question Answering

```python
# Define signatures for each stage
class QueryDecompositionSignature(dspy.Signature):
    """Decompose complex query into simpler sub-questions."""
    question = dspy.InputField()
    sub_questions = dspy.OutputField(desc="List of simpler sub-questions")

class InformationRetrievalSignature(dspy.Signature):
    """Retrieve relevant information for each sub-question."""
    sub_question = dspy.InputField()
    retrieved_info = dspy.OutputField(desc="Relevant information from knowledge base")

class AnswerSynthesisSignature(dspy.Signature):
    """Synthesize final answer from retrieved information."""
    original_question = dspy.InputField()
    retrieved_facts = dspy.InputField()
    final_answer = dspy.OutputField(desc="Comprehensive answer to original question")

# Create modules for each stage
class QueryDecomposer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.decompose = dspy.ChainOfThought(QueryDecompositionSignature)

    def forward(self, question):
        return self.decompose(question=question)

class InformationRetriever(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.ChainOfThought(InformationRetrievalSignature)
        self.rm = dspy.Retrieve(k=3)

    def forward(self, sub_question):
        # First retrieve from knowledge base
        docs = self.rm(sub_question).passages

        # Then synthesize retrieved information
        prediction = self.retrieve(sub_question=sub_question, context=docs)
        return prediction

class AnswerSynthesizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.synthesize = dspy.ChainOfThought(AnswerSynthesisSignature)

    def forward(self, original_question, retrieved_facts):
        return self.synthesize(
            original_question=original_question,
            retrieved_facts=retrieved_facts
        )

# Build the complete pipeline
def build_multi_hop_qa_pipeline():
    """Build a complete multi-hop QA pipeline."""

    stages = [
        QueryDecomposer(),
        InformationRetriever(),
        AnswerSynthesizer()
    ]

    pipeline = SequentialPipeline(stages)

    # Add custom forward method for multi-hop logic
    def forward(self, question):
        # Stage 1: Decompose query
        decomposition = self.stages[0].forward(question=question)
        sub_questions = decomposition.sub_questions

        # Stage 2: Process each sub-question
        all_facts = []
        for sub_q in sub_questions:
            info = self.stages[1].forward(sub_question=sub_q)
            all_facts.append(info.retrieved_info)

        # Stage 3: Synthesize final answer
        combined_facts = "\n".join(all_facts)
        final_answer = self.stages[2].forward(
            original_question=question,
            retrieved_facts=combined_facts
        )

        return dspy.Prediction(
            question=question,
            sub_questions=sub_questions,
            facts=all_facts,
            answer=final_answer.final_answer
        )

    pipeline.forward = forward.__get__(pipeline, SequentialPipeline)
    return pipeline
```

### 2. Branching Architecture

Different execution paths based on intermediate results.

```
           ┌─→ Stage 2a → Stage 3a ─┐
Input → Stage 1 →                     → Stage N → Output
           └─→ Stage 2b → Stage 3b ─┘
```

#### Implementation

```python
class BranchingPipeline(dspy.Module):
    """Pipeline with conditional branching logic."""

    def __init__(self, router_stage, branches):
        super().__init__()
        self.router = router_stage
        self.branches = branches

    def forward(self, **kwargs):
        """Forward pass with routing."""

        # Route to appropriate branch
        route_decision = self.router(**kwargs)
        branch_name = route_decision.branch

        # Execute selected branch
        if branch_name not in self.branches:
            raise ValueError(f"Unknown branch: {branch_name}")

        branch = self.branches[branch_name]
        result = branch(**kwargs, **route_decision.predictions)

        return dspy.Prediction(
            branch=branch_name,
            route_decision=route_decision,
            result=result
        )

# Example routing module
class TaskRouter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(
            "question -> task_type (multiple_choice / short_answer / essay)"
        )

    def forward(self, question):
        result = self.classify(question=question)
        return dspy.Prediction(
            branch=result.task_type.replace(' ', '_'),
            task_type=result.task_type
        )
```

### 3. Iterative/Loop Architecture

Repeatedly process and refine results.

```
        ┌─────────────────┐
        │                 ▼
Input → Stage 1 → Stage 2 → (condition) → Output
        ▲                 │
        └─────Stage 3─────┘
```

#### Implementation

```python
class IterativePipeline(dspy.Module):
    """Pipeline with iterative refinement."""

    def __init__(self, processing_stages, stopping_condition, max_iterations=5):
        super().__init__()
        self.processing_stages = processing_stages
        self.stopping_condition = stopping_condition
        self.max_iterations = max_iterations

    def forward(self, **kwargs):
        """Forward pass with iteration."""

        current_state = kwargs
        iteration = 0
        iterations_data = []

        while iteration < self.max_iterations:
            # Process through all stages
            for stage in self.processing_stages:
                result = stage(**current_state)
                current_state.update(result.predictions if hasattr(result, 'predictions') else result)

            # Check stopping condition
            should_stop = self.stopping_condition(current_state, iteration)
            iterations_data.append({
                'iteration': iteration,
                'state': current_state.copy(),
                'should_stop': should_stop
            })

            if should_stop:
                break

            iteration += 1

        return dspy.Prediction(
            final_state=current_state,
            iterations=iterations_data,
            converged=should_stop
        )

# Example stopping condition
class QualityBasedStopping:
    def __init__(self, quality_threshold=0.9, patience=2):
        self.quality_threshold = quality_threshold
        self.patience = patience
        self.patience_counter = 0

    def __call__(self, state, iteration):
        # Check quality score in state
        if 'quality_score' in state:
            if state['quality_score'] >= self.quality_threshold:
                return True

        # Check for improvement plateau
        if 'improvement' in state:
            if state['improvement'] < 0.01:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    return True
            else:
                self.patience_counter = 0

        return False
```

### 4. Hierarchical Architecture

Nested multi-stage structures for complex tasks.

```
                    ┌─→ Sub-pipeline A
Input → Stage 1 ──→ ├─→ Sub-pipeline B → Stage N → Output
                    └─→ Sub-pipeline C
```

#### Implementation

```python
class HierarchicalPipeline(dspy.Module):
    """Pipeline with nested sub-pipelines."""

    def __init__(self, structure):
        super().__init__()
        self.structure = self._build_structure(structure)

    def _build_structure(self, structure_def):
        """Build nested structure from definition."""

        structure = {}
        for name, config in structure_def.items():
            if config['type'] == 'module':
                structure[name] = dspy.Module.load(config['module'])
            elif config['type'] == 'pipeline':
                structure[name] = self._build_pipeline(config['stages'])
            elif config['type'] == 'conditional':
                structure[name] = self._build_conditional(config)

        return structure

    def forward(self, stage_name='root', **kwargs):
        """Execute hierarchical structure."""

        if stage_name not in self.structure:
            raise ValueError(f"Unknown stage: {stage_name}")

        stage = self.structure[stage_name]

        if isinstance(stage, dspy.Module):
            return stage(**kwargs)
        elif isinstance(stage, dict):
            # Handle conditional logic
            return self._execute_conditional(stage, kwargs)
```

## Design Principles

### 1. Clear Interface Contracts

Each stage should have well-defined inputs and outputs.

```python
from pydantic import BaseModel, Field
from typing import Optional

class StageInput(BaseModel):
    """Standardized input format for stages."""

    content: str = Field(description="Main content to process")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    context: Optional[str] = Field(default=None, description="Additional context")

class StageOutput(BaseModel):
    """Standardized output format for stages."""

    content: str = Field(description="Processed content")
    confidence: float = Field(description="Confidence score")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Stage-specific metadata")
    error: Optional[str] = Field(default=None, description="Error message if failed")

class StandardStage(dspy.Module):
    """Base class with standardized interfaces."""

    input_schema = StageInput
    output_schema = StageOutput

    def forward(self, **kwargs) -> StageOutput:
        # Validate input
        validated_input = self.input_schema(**kwargs)

        try:
            # Process input
            result = self.process(validated_input)

            # Validate output
            validated_output = self.output_schema(**result)
            return validated_output

        except Exception as e:
            # Return error output
            return StageOutput(
                content="",
                confidence=0.0,
                error=str(e)
            )

    def process(self, input_data: StageInput) -> Dict[str, Any]:
        """Override in subclasses."""
        raise NotImplementedError
```

### 2. Graceful Error Handling

Build robustness through error detection and recovery.

```python
class ErrorHandlingPipeline(dspy.Module):
    """Pipeline with comprehensive error handling."""

    def __init__(self, stages, recovery_strategies=None):
        super().__init__()
        self.stages = stages
        self.recovery_strategies = recovery_strategies or {}
        self.error_log = []

    def forward(self, **kwargs):
        """Forward pass with error recovery."""

        current_state = kwargs

        for i, stage in enumerate(self.stages):
            try:
                # Execute stage
                result = stage(**current_state)

                # Check for stage-level errors
                if hasattr(result, 'error') and result.error:
                    raise StageError(f"Stage {i}: {result.error}")

                current_state = result.predictions if hasattr(result, 'predictions') else result

            except Exception as e:
                # Log error
                error_info = {
                    'stage_index': i,
                    'stage_name': getattr(stage, 'name', f'stage_{i}'),
                    'error': str(e),
                    'input_state': current_state
                }
                self.error_log.append(error_info)

                # Attempt recovery
                if i in self.recovery_strategies:
                    recovery_result = self.recovery_strategies[i](e, current_state)
                    if recovery_result:
                        current_state = recovery_result
                        continue

                # Fallback: skip stage or raise error
                if self._should_continue_on_error(e):
                    continue
                else:
                    raise PipelineError(f"Failed at stage {i}: {str(e)}") from e

        return dspy.Prediction(
            result=current_state,
            error_log=self.error_log
        )

    def _should_continue_on_error(self, error):
        """Determine if pipeline should continue after error."""

        # Non-critical errors can be ignored
        non_critical_errors = ['timeout', 'low_confidence']
        return any(err in str(error).lower() for err in non_critical_errors)

class StageError(Exception):
    """Error occurring within a stage."""

    pass

class PipelineError(Exception):
    """Error affecting the entire pipeline."""

    pass
```

### 3. Caching and Memoization

Optimize performance through intelligent caching.

```python
from functools import lru_cache
import hashlib
import json

class CachedStage(dspy.Module):
    """Stage with caching capabilities."""

    def __init__(self, underlying_stage, cache_size=1000):
        super().__init__()
        self.underlying_stage = underlying_stage
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

    def forward(self, **kwargs):
        """Forward pass with caching."""

        # Generate cache key
        cache_key = self._generate_cache_key(kwargs)

        # Check cache
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]

        # Cache miss - compute result
        self.cache_misses += 1
        result = self.underlying_stage(**kwargs)

        # Store in cache
        self._store_in_cache(cache_key, result)

        return result

    def _generate_cache_key(self, kwargs):
        """Generate unique key for caching."""

        # Serialize input
        serialized = json.dumps(kwargs, sort_keys=True, default=str)

        # Generate hash
        return hashlib.md5(serialized.encode()).hexdigest()

    def _store_in_cache(self, key, value):
        """Store value in cache with LRU eviction."""

        if len(self.cache) >= self.cache_size:
            # Simple LRU: remove first item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[key] = value

    def get_cache_stats(self):
        """Get cache performance statistics."""

        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0

        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }
```

## Advanced Architectural Patterns

### 1. Ensemble Architecture

Multiple processing paths with result aggregation.

```python
class EnsemblePipeline(dspy.Module):
    """Pipeline with ensemble of processing paths."""

    def __init__(self, processors, aggregator):
        super().__init__()
        self.processors = processors
        self.aggregator = aggregator

    def forward(self, **kwargs):
        """Execute all processors and aggregate results."""

        # Run all processors
        results = []
        for i, processor in enumerate(self.processors):
            try:
                result = processor(**kwargs)
                results.append({
                    'processor_id': i,
                    'result': result,
                    'success': True
                })
            except Exception as e:
                results.append({
                    'processor_id': i,
                    'error': str(e),
                    'success': False
                })

        # Aggregate successful results
        successful_results = [r for r in results if r['success']]
        if not successful_results:
            raise PipelineError("All processors failed")

        aggregated = self.aggregator.aggregate(successful_results)

        return dspy.Prediction(
            aggregated_result=aggregated,
            individual_results=results
        )

class ResultAggregator:
    """Aggregate results from multiple processors."""

    def __init__(self, strategy='weighted_voting'):
        self.strategy = strategy

    def aggregate(self, results):
        """Aggregate results based on strategy."""

        if self.strategy == 'weighted_voting':
            return self._weighted_voting(results)
        elif self.strategy == 'best_confidence':
            return self._best_confidence(results)
        elif self.strategy == 'consensus':
            return self._consensus(results)
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.strategy}")

    def _weighted_voting(self, results):
        """Aggregate using weighted voting."""

        # Collect votes
        votes = {}
        for r in results:
            output = r['result']
            if hasattr(output, 'predictions'):
                content = output.predictions.get('content', '')
            else:
                content = str(output)

            confidence = getattr(output, 'confidence', 0.5)
            votes[content] = votes.get(content, 0) + confidence

        # Return highest voted option
        best_content = max(votes, key=votes.get)
        return {'content': best_content, 'confidence': votes[best_content] / sum(votes.values())}
```

### 2. Adaptive Architecture

Dynamically adjust structure based on input characteristics.

```python
class AdaptivePipeline(dspy.Module):
    """Pipeline that adapts its structure dynamically."""

    def __init__(self, component_pool, adapter):
        super().__init__()
        self.component_pool = component_pool
        self.adapter = adapter
        self.current_structure = None

    def forward(self, **kwargs):
        """Adapt structure and execute."""

        # Analyze input characteristics
        input_analysis = self.adapter.analyze_input(kwargs)

        # Select appropriate structure
        optimal_structure = self.adapter.select_structure(
            input_analysis,
            self.component_pool
        )

        # Build dynamic pipeline
        pipeline = self._build_pipeline(optimal_structure)

        # Execute
        result = pipeline(**kwargs)

        return dspy.Prediction(
            result=result,
            used_structure=optimal_structure,
            input_analysis=input_analysis
        )

    def _build_pipeline(self, structure):
        """Build pipeline from structure definition."""

        stages = []
        for component_name in structure:
            if component_name in self.component_pool:
                stages.append(self.component_pool[component_name])

        return SequentialPipeline(stages)

class PipelineAdapter:
    """Adapter for selecting optimal pipeline structure."""

    def __init__(self):
        self.structure_patterns = {
            'simple': ['processor_a'],
            'complex': ['preprocessor', 'processor_a', 'postprocessor'],
            'multi_approach': ['processor_a', 'processor_b', 'aggregator']
        }

    def analyze_input(self, input_data):
        """Analyze input characteristics."""

        # Simple analysis based on input properties
        text = input_data.get('content', '')

        characteristics = {
            'length': len(text.split()),
            'complexity': self._compute_complexity(text),
            'type': self._classify_input_type(text)
        }

        return characteristics

    def select_structure(self, analysis, component_pool):
        """Select optimal structure based on analysis."""

        # Simple rule-based selection
        if analysis['length'] < 50 and analysis['complexity'] < 0.3:
            return self.structure_patterns['simple']
        elif analysis['type'] == 'complex_reasoning':
            return self.structure_patterns['multi_approach']
        else:
            return self.structure_patterns['complex']

    def _compute_complexity(self, text):
        """Simple complexity metric."""

        # Factors: sentence length, punctuation, nested structures
        avg_word_length = np.mean([len(word) for word in text.split()])
        punctuation_ratio = text.count('.') + text.count(',') + text.count(';')
        nested_indicators = text.count('(') + text.count('[')

        complexity = (avg_word_length * 0.2 + punctuation_ratio * 0.3 + nested_indicators * 0.5)
        return min(complexity / 10, 1.0)

    def _classify_input_type(self, text):
        """Classify input type."""

        text_lower = text.lower()

        if any(word in text_lower for word in ['why', 'how', 'explain', 'analyze']):
            return 'complex_reasoning'
        elif '?' in text:
            return 'question'
        else:
            return 'statement'
```

### 3. Parallel Architecture

Execute multiple stages concurrently when possible.

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ParallelPipeline(dspy.Module):
    """Pipeline with parallel execution capabilities."""

    def __init__(self, parallel_groups, merger):
        super().__init__()
        self.parallel_groups = parallel_groups
        self.merger = merger
        self.executor = ThreadPoolExecutor(max_workers=4)

    def forward(self, **kwargs):
        """Execute with parallel stages."""

        current_input = kwargs
        group_results = []

        for group in self.parallel_groups:
            if len(group) == 1:
                # Single stage - execute sequentially
                stage = group[0]
                result = stage(**current_input)
                current_input = result.predictions if hasattr(result, 'predictions') else result
                group_results.append([result])
            else:
                # Multiple stages - execute in parallel
                parallel_results = self._execute_parallel(group, current_input)
                group_results.append(parallel_results)

                # Merge results
                merged = self.merger.merge(parallel_results)
                current_input = merged

        return dspy.Prediction(
            result=current_input,
            group_results=group_results
        )

    def _execute_parallel(self, stages, input_data):
        """Execute multiple stages in parallel."""

        futures = []
        for stage in stages:
            future = self.executor.submit(stage, **input_data)
            futures.append(future)

        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result(timeout=30)
                results.append(result)
            except Exception as e:
                # Handle timeout or other errors
                results.append({'error': str(e)})

        return results

class ResultMerger:
    """Merge results from parallel execution."""

    def merge(self, results):
        """Merge multiple results into single output."""

        successful = [r for r in results if not isinstance(r, dict) or 'error' not in r]

        if not successful:
            raise PipelineError("All parallel stages failed")

        # Simple concatenation strategy
        merged_content = []
        for result in successful:
            if hasattr(result, 'predictions'):
                content = result.predictions.get('content', '')
            else:
                content = str(result)
            merged_content.append(content)

        return {'content': '\n'.join(merged_content)}
```

## Performance Optimization

### 1. Stage Fusion

Combine compatible stages to reduce overhead.

```python
class StageFusion:
    """Fuse compatible stages for optimization."""

    def __init__(self):
        self.fusion_rules = {
            'chain_of_thought_chain': self._fuse_cot_chains,
            'retrieval_processing': self._fuse_retrieval_processing,
            'filter_transform': self._fuse_filter_transform
        }

    def can_fuse(self, stage1, stage2):
        """Check if two stages can be fused."""

        # Check if stages are compatible
        stage1_type = type(stage1).__name__
        stage2_type = type(stage2).__name__

        fusion_key = f"{stage1_type}_{stage2_type}"
        return fusion_key in self.fusion_rules

    def fuse(self, stage1, stage2):
        """Fuse two stages into single optimized stage."""

        stage1_type = type(stage1).__name__
        stage2_type = type(stage2).__name__

        fusion_key = f"{stage1_type}_{stage2_type}"
        if fusion_key not in self.fusion_rules:
            raise ValueError(f"Cannot fuse {stage1_type} and {stage2_type}")

        return self.fusion_rules[fusion_key](stage1, stage2)

    def _fuse_cot_chains(self, cot1, cot2):
        """Fuse two ChainOfThought modules."""

        class FusedCoT(dspy.Module):
            def __init__(self, cot1, cot2):
                super().__init__()
                self.cot1 = cot1
                self.cot2 = cot2
                # Create combined signature
                self.combined = dspy.ChainOfThought(
                    f"{cot1.signature} -> {cot2.signature}"
                )

            def forward(self, **kwargs):
                # Execute both reasoning steps in one call
                return self.combined(**kwargs)

        return FusedCoT(cot1, cot2)
```

### 2. Lazy Evaluation

Defer stage execution until results are needed.

```python
class LazyStage(dspy.Module):
    """Stage with lazy evaluation."""

    def __init__(self, underlying_stage):
        super().__init__()
        self.underlying_stage = underlying_stage
        self._cached_result = None
        self._executed = False

    def forward(self, **kwargs):
        """Return lazy result wrapper."""

        return LazyResult(self.underlying_stage, kwargs)

class LazyResult:
    """Lazy evaluation result."""

    def __init__(self, stage, kwargs):
        self.stage = stage
        self.kwargs = kwargs
        self._result = None

    def get_result(self):
        """Force evaluation and return result."""

        if self._result is None:
            self._result = self.stage(**self.kwargs)

        return self._result

    @property
    def predictions(self):
        """Access predictions lazily."""

        return self.get_result().predictions

    def __getattr__(self, name):
        """Delegate attribute access to actual result."""

        return getattr(self.get_result(), name)
```

## Monitoring and Debugging

### 1. Performance Profiling

```python
import time
from collections import defaultdict

class ProfilingPipeline(dspy.Module):
    """Pipeline with performance profiling."""

    def __init__(self, stages):
        super().__init__()
        self.stages = stages
        self.profile_data = defaultdict(list)

    def forward(self, **kwargs):
        """Forward pass with profiling."""

        current_input = kwargs

        for i, stage in enumerate(self.stages):
            # Profile execution time
            start_time = time.time()

            # Execute stage
            result = stage(**current_input)

            # Record execution time
            execution_time = time.time() - start_time
            self.profile_data[f"stage_{i}"].append(execution_time)

            # Profile memory usage
            if hasattr(result, 'predictions'):
                input_size = len(str(current_input))
                output_size = len(str(result.predictions))
                self.profile_data[f"stage_{i}_sizes"].append((input_size, output_size))

            current_input = result.predictions if hasattr(result, 'predictions') else result

        return dspy.Prediction(
            result=current_input,
            profile_data=dict(self.profile_data)
        )

    def get_performance_summary(self):
        """Get summary of performance data."""

        summary = {}
        for stage_key, times in self.profile_data.items():
            if 'sizes' not in stage_key:
                summary[stage_key] = {
                    'avg_time': np.mean(times),
                    'total_time': sum(times),
                    'num_executions': len(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }

        return summary
```

### 2. Execution Tracing

```python
class TracingPipeline(dspy.Module):
    """Pipeline with detailed execution tracing."""

    def __init__(self, stages):
        super().__init__()
        self.stages = stages
        self.execution_trace = []

    def forward(self, **kwargs):
        """Forward pass with tracing."""

        current_input = kwargs
        trace_entry = {
            'timestamp': time.time(),
            'stage': 'input',
            'input': kwargs.copy(),
            'type': 'input'
        }
        self.execution_trace.append(trace_entry)

        for i, stage in enumerate(self.stages):
            # Trace before execution
            trace_entry = {
                'timestamp': time.time(),
                'stage': f'stage_{i}',
                'stage_name': getattr(stage, 'name', f'Stage {i}'),
                'input': current_input.copy(),
                'type': 'stage_start'
            }
            self.execution_trace.append(trace_entry)

            # Execute stage
            result = stage(**current_input)

            # Trace after execution
            trace_entry = {
                'timestamp': time.time(),
                'stage': f'stage_{i}',
                'stage_name': getattr(stage, 'name', f'Stage {i}'),
                'output': result.predictions if hasattr(result, 'predictions') else result,
                'type': 'stage_end'
            }
            self.execution_trace.append(trace_entry)

            current_input = result.predictions if hasattr(result, 'predictions') else result

        # Trace final output
        trace_entry = {
            'timestamp': time.time(),
            'stage': 'output',
            'output': current_input,
            'type': 'output'
        }
        self.execution_trace.append(trace_entry)

        return dspy.Prediction(
            result=current_input,
            execution_trace=self.execution_trace
        )

    def save_trace(self, filename):
        """Save execution trace to file."""

        import json
        with open(filename, 'w') as f:
            json.dump(self.execution_trace, f, indent=2, default=str)
```

## Best Practices

### 1. Architectural Guidelines

- **Single Responsibility**: Each stage should have one clear purpose
- **Loose Coupling**: Minimize dependencies between stages
- **Clear Interfaces**: Define contracts between stages
- **Error Boundaries**: Isolate failures to prevent cascade
- **Resource Management**: Monitor and limit resource usage

### 2. Performance Considerations

- **Parallel Execution**: Use parallelism when stages are independent
- **Caching**: Cache expensive operations and frequently used results
- **Batch Processing**: Process multiple items together when possible
- **Lazy Evaluation**: Defer computation until needed
- **Resource Pooling**: Reuse resources across stages

### 3. Maintenance Tips

- **Modular Design**: Keep stages independent for easier updates
- **Versioning**: Track versions of stages and pipelines
- **Documentation**: Document stage purposes and interfaces
- **Testing**: Test individual stages and integration
- **Monitoring**: Track performance and error rates

## Summary

Multi-stage program architectures provide powerful patterns for building complex language model applications. Key takeaways:

1. **Architectural Patterns**: Sequential, branching, iterative, and hierarchical designs
2. **Design Principles**: Clear interfaces, error handling, and caching
3. **Advanced Patterns**: Ensemble, adaptive, and parallel architectures
4. **Optimization Techniques**: Stage fusion and lazy evaluation
5. **Monitoring Tools**: Profiling and tracing for debugging

The next section will explore optimization strategies specifically for complex multi-stage pipelines, building on these architectural foundations.
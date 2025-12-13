"""
Module Composition Examples

This file demonstrates advanced DSPy module composition patterns:
- Sequential processing chains
- Parallel processing patterns
- Conditional routing
- Hierarchical architectures
- Dynamic composition
- Complex workflows
"""

import dspy
from typing import List, Dict, Any, Optional, Union, Callable
import json
from datetime import datetime

# Example 1: Sequential Processing Chain
class SequentialChain(dspy.Module):
    """Chain modules for sequential processing."""

    def __init__(self, modules: List[dspy.Module]):
        super().__init__()
        self.modules = modules
        self.module_names = [f"step_{i+1}" for i in range(len(modules))]

    def forward(self, **kwargs):
        """Process input through all modules sequentially."""
        current_output = kwargs
        results = {}

        for i, (module, name) in enumerate(zip(self.modules, self.module_names)):
            # Module might have different expected inputs
            try:
                # Try passing all current outputs
                result = module(**current_output)
                results[name] = result

                # Update current_output for next module
                if hasattr(result, '_asdict'):
                    current_output.update(result._asdict())
                elif hasattr(result, '__dict__'):
                    current_output.update(result.__dict__)
                elif isinstance(result, dict):
                    current_output.update(result)

            except Exception as e:
                # Try with specific input if general fails
                if 'input' in current_output:
                    result = module(input=current_output['input'])
                    results[name] = result
                    current_output = result
                else:
                    raise e

        return dspy.Prediction(
            steps=results,
            final_output=current_output,
            num_modules=len(self.modules)
        )

# Example 2: Parallel Processing
class ParallelProcessor(dspy.Module):
    """Process input through multiple modules in parallel."""

    def __init__(self, modules: Dict[str, dspy.Module]):
        super().__init__()
        self.modules = modules

    def forward(self, **kwargs):
        """Process input through all modules in parallel."""
        results = {}

        for name, module in self.modules.items():
            try:
                result = module(**kwargs)
                results[name] = result
            except Exception as e:
                results[name] = dspy.Prediction(error=str(e))

        return dspy.Prediction(
            parallel_results=results,
            num_modules=len(self.modules)
        )

# Example 3: Conditional Router
class ConditionalRouter(dspy.Module):
    """Route inputs to different modules based on conditions."""

    def __init__(self):
        super().__init__()
        self.classifier_sig = dspy.Signature("input -> route")
        self.classifier = dspy.Predict(self.classifier_sig)
        self.routes = {}

    def add_route(self, route_name: str, condition: str, module: dspy.Module):
        """Add a conditional route."""
        self.routes[route_name] = {
            "condition": condition,
            "module": module
        }

    def forward(self, **kwargs):
        """Route input based on classification."""
        # Classify input to determine route
        input_text = kwargs.get('input', str(kwargs))
        classification = self.classifier(input=input_text)

        # Find matching route
        selected_route = None
        for route_name, route_info in self.routes.items():
            if route_info["condition"].lower() in classification.route.lower():
                selected_route = route_name
                break

        # Default to first route if no match
        if selected_route is None and self.routes:
            selected_route = list(self.routes.keys())[0]

        # Process through selected module
        if selected_route:
            result = self.routes[selected_route]["module"](**kwargs)
        else:
            result = dspy.Prediction(error="No route matched")

        return dspy.Prediction(
            input=input_text,
            classification=classification.route,
            selected_route=selected_route,
            result=result
        )

# Example 4: Hierarchical Architecture
class HierarchicalProcessor(dspy.Module):
    """Hierarchical module composition with multiple levels."""

    def __init__(self, hierarchy: Dict[str, Dict]):
        """
        Initialize with hierarchy specification.
        Example hierarchy:
        {
            "level_1": {"modules": [module1, module2], "combiner": combine_func},
            "level_2": {"modules": [module3], "parent": "level_1"},
            "output": {"finalizer": final_module}
        }
        """
        super().__init__()
        self.hierarchy = hierarchy
        self.processors = {}
        self._build_hierarchy()

    def _build_hierarchy(self):
        """Build the hierarchical processing structure."""
        for level_name, level_config in self.hierarchy.items():
            if "modules" in level_config:
                if "combiner" in level_config:
                    # Parallel processing with combiner
                    self.processors[level_name] = ParallelProcessor(
                        modules=dict(zip(
                            [f"submodule_{i}" for i in range(len(level_config["modules"]))],
                            level_config["modules"]
                        ))
                    )
                else:
                    # Sequential processing
                    self.processors[level_name] = SequentialChain(
                        modules=level_config["modules"]
                    )

    def forward(self, **kwargs):
        """Process through hierarchy."""
        results = {}
        current_input = kwargs

        # Process each level in order
        for level_name in sorted(self.hierarchy.keys()):
            if level_name in self.processors:
                level_result = self.processors[level_name](**current_input)
                results[level_name] = level_result

                # Prepare for next level
                current_input = {"input": level_result}

        # Final output processing
        if "output" in self.hierarchy:
            final_module = self.hierarchy["output"]["finalizer"]
            final_result = final_module(**current_input)
            results["output"] = final_result
        else:
            final_result = results.get(sorted(results.keys())[-1])

        return dspy.Prediction(
            hierarchy_results=results,
            final_output=final_result,
            num_levels=len(self.processors)
        )

# Example 5: Dynamic Composition Builder
class DynamicComposer:
    """Builder for dynamic module composition."""

    def __init__(self):
        self.modules = {}
        self.composition_graph = {}

    def add_module(self, name: str, module: dspy.Module):
        """Add a module to the composer."""
        self.modules[name] = module

    def compose(self, composition_spec: Dict):
        """
        Compose modules based on specification.
        Example spec:
        {
            "type": "chain",
            "modules": ["preprocessor", "analyzer", "postprocessor"]
        }
        """
        comp_type = composition_spec["type"]

        if comp_type == "chain":
            modules = [self.modules[name] for name in composition_spec["modules"]]
            return SequentialChain(modules)

        elif comp_type == "parallel":
            modules = {
                name: self.modules[name]
                for name in composition_spec["modules"]
            }
            return ParallelProcessor(modules)

        elif comp_type == "conditional":
            router = ConditionalRouter()
            for route_spec in composition_spec["routes"]:
                router.add_route(
                    route_spec["name"],
                    route_spec["condition"],
                    self.modules[route_spec["module"]]
                )
            return router

        else:
            raise ValueError(f"Unknown composition type: {comp_type}")

# Example 6: Pipeline with Checkpoints
class CheckpointedPipeline(dspy.Module):
    """Pipeline with intermediate checkpoints and recovery."""

    def __init__(self, modules: List[dspy.Module], checkpoint_interval: int = 1):
        super().__init__()
        self.modules = modules
        self.checkpoint_interval = checkpoint_interval
        self.checkpoints = {}
        self.checkpoint_sigs = {}

        # Create checkpoint signatures
        for i in range(0, len(modules), checkpoint_interval):
            self.checkpoint_sigs[i] = dspy.Signature(
                "step_name, step_output -> checkpoint_data"
            )
            self.checkpoints[i] = dspy.Predict(self.checkpoint_sigs[i])

    def forward(self, **kwargs):
        """Process with checkpointing capability."""
        current_output = kwargs
        checkpoint_data = {}

        for i, module in enumerate(self.modules):
            try:
                # Process through module
                result = module(**current_output)
                current_output = result

                # Create checkpoint at intervals
                if i % self.checkpoint_interval == 0:
                    checkpoint = self.checkpoints[i](
                        step_name=f"module_{i}",
                        step_output=str(result)[:1000]  # Limit size
                    )
                    checkpoint_data[f"checkpoint_{i}"] = checkpoint.checkpoint_data

            except Exception as e:
                # Try to recover from last checkpoint
                print(f"Error at module {i}: {e}")
                last_checkpoint = max(
                    [k for k in checkpoint_data.keys() if int(k.split('_')[1]) < i],
                    default=None
                )

                if last_checkpoint:
                    return dspy.Prediction(
                        error=f"Failed at module {i}, recovered from {last_checkpoint}",
                        checkpoint_data=checkpoint_data,
                        last_checkpoint=last_checkpoint,
                        partial_result=current_output
                    )
                else:
                    raise e

        return dspy.Prediction(
            final_output=current_output,
            checkpoints=checkpoint_data,
            num_modules=len(self.modules),
            successful=True
        )

# Example 7: Adaptive Ensemble
class AdaptiveEnsemble(dspy.Module):
    """Ensemble that adapts based on performance."""

    def __init__(self, modules: List[dspy.Module], weights: Optional[List[float]] = None):
        super().__init__()
        self.modules = modules
        self.initial_weights = weights or [1.0 / len(modules)] * len(modules)
        self.performance_history = []

        self.combiner_sig = dspy.Signature(
            "module_outputs, weights -> combined_output"
        )
        self.combiner = dspy.Predict(self.combiner_sig)

        self.evaluator_sig = dspy.Signature(
            "input, output, reference -> quality_score"
        )
        self.evaluator = dspy.Predict(self.evaluator_sig)

    def forward(self, **kwargs):
        """Process with adaptive ensemble."""
        # Get outputs from all modules
        module_outputs = []
        for module in self.modules:
            try:
                output = module(**kwargs)
                module_outputs.append(str(output))
            except:
                module_outputs.append("Error in module")

        # Combine outputs with current weights
        outputs_str = json.dumps(module_outputs)
        weights_str = json.dumps(self.initial_weights)

        combined = self.combiner(
            module_outputs=outputs_str,
            weights=weights_str
        )

        return dspy.Prediction(
            individual_outputs=module_outputs,
            weights=self.initial_weights,
            combined_output=combined.combined_output,
            num_modules=len(self.modules)
        )

    def update_weights(self, input_data, reference_output, feedback_score):
        """Update ensemble weights based on feedback."""
        # This would typically involve more sophisticated weight updates
        # For demonstration, we'll just track performance
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "score": feedback_score,
            "weights": self.initial_weights.copy()
        })

# Example 8: Multi-Modal Composer
class MultiModalComposer(dspy.Module):
    """Compose modules for processing different data modalities."""

    def __init__(self):
        super().__init__()
        self.modality_detectors = {}
        self.modality_processors = {}

    def add_modality(self, modality: str, detector: dspy.Module, processor: dspy.Module):
        """Add a modality with its detector and processor."""
        self.modality_detectors[modality] = detector
        self.modality_processors[modality] = processor

    def forward(self, **kwargs):
        """Process input based on detected modality."""
        # Detect modality
        detected_modalities = {}
        for modality, detector in self.modality_detectors.items():
            try:
                detection = detector(**kwargs)
                detected_modalities[modality] = detection
            except:
                detected_modalities[modality] = None

        # Process with appropriate processors
        results = {}
        for modality, detection in detected_modalities.items():
            if detection and hasattr(detection, 'detected'):
                if detection.detected:
                    processor = self.modality_processors[modality]
                    result = processor(**kwargs)
                    results[modality] = result

        # If no specific modality detected, use all processors
        if not results:
            for modality, processor in self.modality_processors.items():
                try:
                    result = processor(**kwargs)
                    results[modality] = result
                except:
                    pass

        return dspy.Prediction(
            modalities=detected_modalities,
            processed_results=results,
            input_data=kwargs
        )

# Demonstration Functions
def demonstrate_sequential_chain():
    """Demonstrate sequential processing."""

    print("=" * 60)
    print("Example 1: Sequential Chain Processing")
    print("=" * 60)

    # Create simple modules
    class Preprocessor(dspy.Module):
        def __init__(self):
            super().__init__()
            self.sig = dspy.Signature("input -> preprocessed")
            self.predictor = dspy.Predict(self.sig)

        def forward(self, **kwargs):
            input_text = kwargs.get('input', '')
            result = self.predictor(input=f"Preprocess: {input_text}")
            return dspy.Prediction(input=result.preprocessed)

    class Analyzer(dspy.Module):
        def __init__(self):
            super().__init__()
            self.sig = dspy.Signature("input -> analysis")
            self.predictor = dspy.Predict(self.sig)

        def forward(self, **kwargs):
            input_text = kwargs.get('input', '')
            result = self.predictor(input=f"Analyze: {input_text}")
            return dspy.Prediction(input=result.analysis)

    class Postprocessor(dspy.Module):
        def __init__(self):
            super().__init__()
            self.sig = dspy.Signature("input -> postprocessed")
            self.predictor = dspy.Predict(self.sig)

        def forward(self, **kwargs):
            input_text = kwargs.get('input', '')
            result = self.predictor(input=f"Postprocess: {input_text}")
            return dspy.Prediction(input=result.postprocessed)

    # Create chain
    chain = SequentialChain([Preprocessor(), Analyzer(), Postprocessor()])

    # Test
    result = chain(input="This is a test document about DSPy modules.")
    print(f"\nInput: This is a test document about DSPy modules.")
    print(f"Final Output: {result.final_output}")
    print(f"Number of Modules: {result.num_modules}")

def demonstrate_parallel_processing():
    """Demonstrate parallel processing."""

    print("\n" + "=" * 60)
    print("Example 2: Parallel Processing")
    print("=" * 60)

    # Create modules for different perspectives
    class SentimentAnalyzer(dspy.Module):
        def __init__(self):
            super().__init__()
            self.sig = dspy.Signature("text -> sentiment")
            self.predictor = dspy.Predict(self.sig)

        def forward(self, **kwargs):
            text = kwargs.get('text', '')
            result = self.predictor(input=text)
            return dspy.Prediction(sentiment=result.sentiment)

    class KeywordExtractor(dspy.Module):
        def __init__(self):
            super().__init__()
            self.sig = dspy.Signature("text -> keywords")
            self.predictor = dspy.Predict(self.sig)

        def forward(self, **kwargs):
            text = kwargs.get('text', '')
            result = self.predictor(input=text)
            return dspy.Prediction(keywords=result.keywords)

    class TopicClassifier(dspy.Module):
        def __init__(self):
            super().__init__()
            self.sig = dspy.Signature("text -> topic")
            self.predictor = dspy.Predict(self.sig)

        def forward(self, **kwargs):
            text = kwargs.get('text', '')
            result = self.predictor(input=text)
            return dspy.Prediction(topic=result.topic)

    # Create parallel processor
    parallel = ParallelProcessor({
        "sentiment": SentimentAnalyzer(),
        "keywords": KeywordExtractor(),
        "topic": TopicClassifier()
    })

    # Test
    text = "The new AI technology is revolutionizing healthcare and medical diagnosis."
    result = parallel(text=text)

    print(f"\nText: {text}")
    print(f"Number of Modules: {result.num_modules}")
    for name, output in result.parallel_results.items():
        if hasattr(output, 'sentiment'):
            print(f"{name.title()}: {output.sentiment}")
        elif hasattr(output, 'keywords'):
            print(f"{name.title()}: {output.keywords}")
        elif hasattr(output, 'topic'):
            print(f"{name.title()}: {output.topic}")

def demonstrate_conditional_routing():
    """Demonstrate conditional routing."""

    print("\n" + "=" * 60)
    print("Example 3: Conditional Routing")
    print("=" * 60)

    # Create specialized modules
    class QuestionHandler(dspy.Module):
        def __init__(self):
            super().__init__()
            self.sig = dspy.Signature("question -> answer")
            self.predictor = dspy.Predict(self.sig)

        def forward(self, **kwargs):
            q = kwargs.get('question', '')
            result = self.predictor(input=f"Answer this question: {q}")
            return dspy.Prediction(answer=result.answer)

    class CommandHandler(dspy.Module):
        def __init__(self):
            super().__init__()
            self.sig = dspy.Signature("command -> execution")
            self.predictor = dspy.Predict(self.sig)

        def forward(self, **kwargs):
            cmd = kwargs.get('command', '')
            result = self.predictor(input=f"Execute command: {cmd}")
            return dspy.Prediction(execution=result.execution)

    class StatementHandler(dspy.Module):
        def __init__(self):
            super().__init__()
            self.sig = dspy.Signature("statement -> acknowledgment")
            self.predictor = dspy.Predict(self.sig)

        def forward(self, **kwargs):
            stmt = kwargs.get('statement', '')
            result = self.predictor(input=f"Acknowledge statement: {stmt}")
            return dspy.Prediction(acknowledgment=result.acknowledgment)

    # Create router
    router = ConditionalRouter()
    router.add_route("question", "question", QuestionHandler())
    router.add_route("command", "command", CommandHandler())
    router.add_route("statement", "statement", StatementHandler())

    # Test
    inputs = [
        "What is the capital of France?",
        "Calculate 2 + 2",
        "The weather is nice today."
    ]

    for inp in inputs:
        result = router(input=inp)
        print(f"\nInput: {inp}")
        print(f"Classification: {result.classification}")
        print(f"Route: {result.selected_route}")

def demonstrate_dynamic_composition():
    """Demonstrate dynamic composition builder."""

    print("\n" + "=" * 60)
    print("Example 4: Dynamic Composition")
    print("=" * 60)

    # Create some basic modules
    class Cleaner(dspy.Module):
        def __init__(self):
            super().__init__()
            self.sig = dspy.Signature("text -> clean_text")
            self.predictor = dspy.Predict(self.sig)

        def forward(self, **kwargs):
            text = kwargs.get('text', '')
            result = self.predictor(input=text)
            return dspy.Prediction(text=result.clean_text)

    class Summarizer(dspy.Module):
        def __init__(self):
            super().__init__()
            self.sig = dspy.Signature("text -> summary")
            self.predictor = dspy.Predict(self.sig)

        def forward(self, **kwargs):
            text = kwargs.get('text', '')
            result = self.predictor(input=text)
            return dspy.Prediction(text=result.summary)

    class Translator(dspy.Module):
        def __init__(self):
            super().__init__()
            self.sig = dspy.Signature("text -> translation")
            self.predictor = dspy.Predict(self.sig)

        def forward(self, **kwargs):
            text = kwargs.get('text', '')
            result = self.predictor(input=f"Translate to French: {text}")
            return dspy.Prediction(text=result.translation)

    # Use dynamic composer
    composer = DynamicComposer()
    composer.add_module("cleaner", Cleaner())
    composer.add_module("summarizer", Summarizer())
    composer.add_module("translator", Translator())

    # Compose different pipelines
    chain_spec = {
        "type": "chain",
        "modules": ["cleaner", "summarizer"]
    }

    parallel_spec = {
        "type": "parallel",
        "modules": ["summarizer", "translator"]
    }

    # Create pipelines
    chain_pipeline = composer.compose(chain_spec)
    parallel_pipeline = composer.compose(parallel_spec)

    # Test
    text = "This is a long text that needs to be processed and summarized. It contains multiple sentences and various information that could be useful for different purposes."

    print(f"\nOriginal Text: {text[:60]}...")

    # Sequential pipeline
    chain_result = chain_pipeline(text=text)
    print(f"\nSequential Pipeline Result: {chain_result.final_output}")

    # Parallel pipeline
    parallel_result = parallel_pipeline(text=text)
    print(f"\nParallel Pipeline Results:")
    for name, result in parallel_result.parallel_results.items():
        print(f"  {name}: {result.text[:50]}...")

def demonstrate_hierarchical_processing():
    """Demonstrate hierarchical processing."""

    print("\n" + "=" * 60)
    print("Example 5: Hierarchical Processing")
    print("=" * 60)

    # Simple modules for demonstration
    class ModuleA(dspy.Module):
        def forward(self, **kwargs):
            return dspy.Prediction(output=f"Processed by A: {str(kwargs)[:30]}")

    class ModuleB(dspy.Module):
        def forward(self, **kwargs):
            return dspy.Prediction(output=f"Processed by B: {str(kwargs)[:30]}")

    class ModuleC(dspy.Module):
        def forward(self, **kwargs):
            return dspy.Prediction(output=f"Processed by C: {str(kwargs)[:30]}")

    class FinalModule(dspy.Module):
        def forward(self, **kwargs):
            return dspy.Prediction(output=f"Final result: {str(kwargs)[:50]}")

    # Create hierarchy
    hierarchy = {
        "level_1": {
            "modules": [ModuleA(), ModuleB()],
            "combiner": True
        },
        "level_2": {
            "modules": [ModuleC()],
            "parent": "level_1"
        },
        "output": {
            "finalizer": FinalModule()
        }
    }

    # Create hierarchical processor
    processor = HierarchicalProcessor(hierarchy)

    # Test
    result = processor(input="Test data for hierarchical processing")

    print(f"\nFinal Output: {result.final_output}")
    print(f"Number of Levels: {result.num_levels}")

def demonstrate_checkpointed_pipeline():
    """Demonstrate pipeline with checkpoints."""

    print("\n" + "=" * 60)
    print("Example 6: Checkpointed Pipeline")
    print("=" * 60)

    # Create modules
    class Step1(dspy.Module):
        def forward(self, **kwargs):
            return dspy.Prediction(step="1", data="Step 1 completed")

    class Step2(dspy.Module):
        def forward(self, **kwargs):
            return dspy.Prediction(step="2", data="Step 2 completed")

    class Step3(dspy.Module):
        def forward(self, **kwargs):
            return dspy.Prediction(step="3", data="Step 3 completed")

    # Create checkpointed pipeline
    pipeline = CheckpointedPipeline([Step1(), Step2(), Step3()], checkpoint_interval=1)

    # Test
    result = pipeline(input="Test data")

    print(f"\nFinal Output: {result.final_output}")
    print(f"Successful: {result.successful}")
    print(f"Checkpoints Created: {len(result.checkpoints)}")

# Main execution
def run_all_examples():
    """Run all module composition examples."""

    print("DSPy Module Composition Examples")
    print("These examples demonstrate advanced composition patterns and architectures.")
    print("=" * 60)

    try:
        demonstrate_sequential_chain()
        demonstrate_parallel_processing()
        demonstrate_conditional_routing()
        demonstrate_dynamic_composition()
        demonstrate_hierarchical_processing()
        demonstrate_checkpointed_pipeline()

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_examples()
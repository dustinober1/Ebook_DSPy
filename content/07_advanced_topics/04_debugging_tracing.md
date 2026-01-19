# Debugging and Tracing: Mastering DSPy Application Diagnostics

## Introduction

Debugging complex DSPy applications requires specialized tools and techniques. Unlike traditional software debugging, DSPy applications involve language model interactions, optimization processes, and distributed components that make troubleshooting challenging. This section provides comprehensive strategies for effective debugging and tracing of DSPy applications.

## Understanding DSPy Debugging Challenges

### Unique Challenges

1. **Non-deterministic Outputs**: Language models can produce varying results
2. **Hidden Complexity**: Optimizers and prompt engineering add layers of abstraction
3. **API Dependencies**: External service issues can cause failures
4. **Token Limits**: Context window constraints cause unexpected behavior
5. **Performance Issues**: Latency and cost optimization complexity

### Debugging Categories

```python
from enum import Enum
from typing import Optional, List, Dict, Any
import traceback
import sys

class DebugLevel(Enum):
    NONE = 0
    ERROR = 1
    WARNING = 2
    INFO = 3
    DEBUG = 4
    TRACE = 5

class DSPyDebugger:
    """Comprehensive debugging utility for DSPy applications."""

    def __init__(self, level=DebugLevel.INFO):
        self.level = level
        self.trace_history = []
        self.breakpoints = set()
        self.watch_variables = {}

    def log(self, level, message, data=None):
        """Log message with specified level."""
        if level.value <= self.level.value:
            timestamp = time.time()
            log_entry = {
                "timestamp": timestamp,
                "level": level.name,
                "message": message,
                "data": data
            }
            self.trace_history.append(log_entry)
            self._print_log(log_entry)

    def _print_log(self, entry):
        """Print formatted log entry."""
        color_map = {
            "ERROR": "\033[91m",      # Red
            "WARNING": "\033[93m",    # Yellow
            "INFO": "\033[94m",       # Blue
            "DEBUG": "\033[96m",      # Cyan
            "TRACE": "\033[97m",      # White
        }
        reset = "\033[0m"

        color = color_map.get(entry["level"], "")
        print(f"{color}[{entry['timestamp']:.2f}] {entry['level']}: {entry['message']}{reset}")

        if entry["data"]:
            print(f"  Data: {entry['data']}")

    def trace(self, module_name, func_name, args, kwargs, result=None, error=None):
        """Trace function execution."""
        trace_data = {
            "module": module_name,
            "function": func_name,
            "args": args,
            "kwargs": kwargs,
            "result": result,
            "error": str(error) if error else None,
            "traceback": traceback.format_exc() if error else None
        }

        self.log(DebugLevel.TRACE, f"Executing {module_name}.{func_name}", trace_data)

        # Check breakpoints
        breakpoint_key = f"{module_name}.{func_name}"
        if breakpoint_key in self.breakpoints:
            self._handle_breakpoint(trace_data)

    def _handle_breakpoint(self, trace_data):
        """Handle breakpoint trigger."""
        print(f"\n*** BREAKPOINT: {trace_data['module']}.{trace_data['function']} ***")
        print(f"Args: {trace_data['args']}")
        print(f"Kwargs: {trace_data['kwargs']}")
        if trace_data['result']:
            print(f"Result: {trace_data['result']}")
        if trace_data['error']:
            print(f"Error: {trace_data['error']}")

        # Interactive debugging
        import pdb
        pdb.set_trace()

    def add_breakpoint(self, module_name, func_name):
        """Add breakpoint for debugging."""
        self.breakpoints.add(f"{module_name}.{func_name}")

    def remove_breakpoint(self, module_name, func_name):
        """Remove breakpoint."""
        self.breakpoints.discard(f"{module_name}.{func_name}")

    def watch_variable(self, name, value):
        """Watch variable for changes."""
        if name not in self.watch_variables:
            self.watch_variables[name] = None

        if self.watch_variables[name] != value:
            self.log(DebugLevel.DEBUG, f"Variable {name} changed", {
                "old": self.watch_variables[name],
                "new": value
            })
            self.watch_variables[name] = value

# Global debugger instance
debugger = DSPyDebugger()
```

## Module Tracing

### 1. Function Tracing Decorator

```python
import functools
import inspect

def trace_function(debug_level=DebugLevel.DEBUG):
    """Decorator to trace function execution."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            module_name = func.__module__
            func_name = func.__name__

            # Start trace
            debugger.trace(module_name, func_name, args, kwargs)

            try:
                result = func(*args, **kwargs)
                # Success trace
                debugger.trace(
                    module_name, func_name, args, kwargs,
                    result=result
                )
                return result

            except Exception as e:
                # Error trace
                debugger.trace(
                    module_name, func_name, args, kwargs,
                    error=e
                )
                raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            module_name = func.__module__
            func_name = func.__name__

            # Start trace
            debugger.trace(module_name, func_name, args, kwargs)

            try:
                result = await func(*args, **kwargs)
                # Success trace
                debugger.trace(
                    module_name, func_name, args, kwargs,
                    result=result
                )
                return result

            except Exception as e:
                # Error trace
                debugger.trace(
                    module_name, func_name, args, kwargs,
                    error=e
                )
                raise

        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper

    return decorator

# Usage example
class TracedModule(dspy.Module):
    """Module with automatic tracing."""

    def __init__(self):
        super().__init__()
        self.process = dspy.Predict("input -> output")

    @trace_function()
    def forward(self, input_text):
        debugger.watch_variable("input_length", len(input_text))
        result = self.process(input=input_text)
        debugger.watch_variable("output_length", len(str(result.output)))
        return result
```

### 2. Module State Inspector

```python
class ModuleInspector:
    """Inspect and analyze DSPy module state."""

    def __init__(self):
        self.inspection_history = []

    def inspect_module(self, module: dspy.Module):
        """Inspect module configuration and state."""
        inspection = {
            "module_class": module.__class__.__name__,
            "module_name": module.__class__.__module__,
            "parameters": {},
            "submodules": {},
            "storage": {}
        }

        # Get parameters
        if hasattr(module, 'named_parameters'):
            for name, param in module.named_parameters():
                inspection["parameters"][name] = {
                    "shape": param.shape if hasattr(param, 'shape') else None,
                    "requires_grad": param.requires_grad if hasattr(param, 'requires_grad') else None,
                    "value": param.data if hasattr(param, 'data') else None
                }

        # Get submodules
        if hasattr(module, 'named_modules'):
            for name, submodule in module.named_modules():
                inspection["submodules"][name] = {
                    "type": type(submodule).__name__,
                    "id": id(submodule)
                }

        # Get storage/attributes
        for attr_name in dir(module):
            if not attr_name.startswith('_'):
                attr_value = getattr(module, attr_name)
                if not callable(attr_value) and not isinstance(attr_value, type(module)):
                    inspection["storage"][attr_name] = str(attr_value)[:100]

        self.inspection_history.append(inspection)
        return inspection

    def compare_states(self, before_inspection, after_inspection):
        """Compare module states before and after operation."""
        differences = {
            "parameters_changed": [],
            "storage_changed": [],
            "submodules_changed": []
        }

        # Compare parameters
        for key in before_inspection["parameters"]:
            if key in after_inspection["parameters"]:
                if before_inspection["parameters"][key] != after_inspection["parameters"][key]:
                    differences["parameters_changed"].append(key)

        # Compare storage
        for key in before_inspection["storage"]:
            if key in after_inspection["storage"]:
                if before_inspection["storage"][key] != after_inspection["storage"][key]:
                    differences["storage_changed"].append(key)

        return differences
```

## Performance Tracing

### 1. Performance Profiler

```python
import time
import psutil
import threading
from contextlib import contextmanager

class PerformanceProfiler:
    """Profile DSPy module performance."""

    def __init__(self):
        self.profiles = {}
        self.active_profiles = {}
        self.system_monitor = SystemMonitor()

    @contextmanager
    def profile(self, name):
        """Context manager for profiling."""
        profile_id = f"{name}_{time.time()}"
        profile_data = {
            "name": name,
            "start_time": time.time(),
            "start_memory": self.system_monitor.get_memory_usage(),
            "start_cpu": self.system_monitor.get_cpu_usage()
        }

        self.active_profiles[profile_id] = profile_data

        try:
            yield profile_id
        finally:
            # End profiling
            end_time = time.time()
            end_memory = self.system_monitor.get_memory_usage()
            end_cpu = self.system_monitor.get_cpu_usage()

            profile_data.update({
                "end_time": end_time,
                "duration": end_time - profile_data["start_time"],
                "end_memory": end_memory,
                "end_cpu": end_cpu,
                "memory_delta": end_memory - profile_data["start_memory"],
                "cpu_delta": end_cpu - profile_data["start_cpu"]
            })

            self.profiles[profile_id] = profile_data
            del self.active_profiles[profile_id]

    def get_profile_summary(self, name=None):
        """Get summary of profiles."""
        if name:
            relevant_profiles = [
                p for p in self.profiles.values()
                if p["name"] == name
            ]
        else:
            relevant_profiles = list(self.profiles.values())

        if not relevant_profiles:
            return None

        summary = {
            "total_profiles": len(relevant_profiles),
            "total_duration": sum(p["duration"] for p in relevant_profiles),
            "average_duration": sum(p["duration"] for p in relevant_profiles) / len(relevant_profiles),
            "max_memory": max(p["end_memory"] for p in relevant_profiles),
            "total_memory_delta": sum(p["memory_delta"] for p in relevant_profiles)
        }

        return summary

class SystemMonitor:
    """Monitor system resources."""

    def get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def get_cpu_usage(self):
        """Get current CPU usage percentage."""
        process = psutil.Process()
        return process.cpu_percent()
```

### 2. Token Usage Tracker

```python
import tiktoken

class TokenTracker:
    """Track token usage for cost optimization."""

    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

        self.usage_history = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def count_tokens(self, text):
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def track_api_call(self, prompt, completion, model=None):
        """Track API call token usage."""
        model = model or self.model
        prompt_tokens = self.count_tokens(prompt)
        completion_tokens = self.count_tokens(completion) if completion else 0

        usage_entry = {
            "timestamp": time.time(),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }

        self.usage_history.append(usage_entry)
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        return usage_entry

    def estimate_cost(self):
        """Estimate total cost of API calls."""
        # Pricing (example rates, adjust based on actual pricing)
        pricing = {
            "gpt-3.5-turbo": {"prompt": 0.002, "completion": 0.002},
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03}
        }

        total_cost = 0
        for usage in self.usage_history:
            model_pricing = pricing.get(usage["model"], pricing["gpt-3.5-turbo"])
            prompt_cost = (usage["prompt_tokens"] / 1000) * model_pricing["prompt"]
            completion_cost = (usage["completion_tokens"] / 1000) * model_pricing["completion"]
            total_cost += prompt_cost + completion_cost

        return total_cost

    def get_usage_summary(self):
        """Get summary of token usage."""
        return {
            "total_calls": len(self.usage_history),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "estimated_cost": self.estimate_cost()
        }

# Global token tracker
token_tracker = TokenTracker()
```

## Visual Debugging

### 1. Execution Graph Visualizer

```python
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Any

class ExecutionGraph:
    """Visualize DSPy execution flow."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_data = {}
        self.edge_data = {}

    def add_node(self, node_id, module_name, operation, data=None):
        """Add node to execution graph."""
        self.graph.add_node(node_id)
        self.node_data[node_id] = {
            "module": module_name,
            "operation": operation,
            "data": data or {},
            "timestamp": time.time()
        }

    def add_edge(self, source, target, relationship="data_flow", data=None):
        """Add edge to execution graph."""
        self.graph.add_edge(source, target)
        edge_key = (source, target)
        self.edge_data[edge_key] = {
            "relationship": relationship,
            "data": data or {}
        }

    def visualize(self, save_path=None):
        """Visualize execution graph."""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)

        # Draw nodes
        node_labels = {}
        node_colors = []
        for node_id in self.graph.nodes():
            data = self.node_data[node_id]
            label = f"{data['module']}\n{data['operation']}"
            node_labels[node_id] = label

            # Color based on operation type
            if "generate" in data["operation"].lower():
                node_colors.append("lightblue")
            elif "predict" in data["operation"].lower():
                node_colors.append("lightgreen")
            elif "retrieve" in data["operation"].lower():
                node_colors.append("yellow")
            else:
                node_colors.append("lightgray")

        nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=node_colors,
            node_size=1500,
            alpha=0.7
        )

        nx.draw_networkx_labels(
            self.graph, pos,
            labels=node_labels,
            font_size=8
        )

        # Draw edges
        nx.draw_networkx_edges(
            self.graph, pos,
            edge_color="gray",
            alpha=0.5,
            arrows=True,
            arrowsize=20
        )

        plt.title("DSPy Execution Graph")
        plt.axis("off")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        else:
            plt.show()
```

### 2. Interactive Debugger

```python
import readline
import cmd

class DSPyInteractiveDebugger(cmd.Cmd):
    """Interactive debugger for DSPy applications."""

    intro = "DSPy Interactive Debugger. Type 'help' for commands."
    prompt = "(dspy-debug) "

    def __init__(self, module=None):
        super().__init__()
        self.module = module
        self.execution_history = []
        self.variables = {}

    def do_trace(self, line):
        """Trace module execution."""
        if not self.module:
            print("No module loaded. Use 'load <module>' first.")
            return

        try:
            result = self.module.forward(line)
            print(f"Result: {result}")
            self.execution_history.append((line, result))
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    def do_load(self, line):
        """Load a DSPy module."""
        try:
            # This would load a module from file or import
            # For now, just store the name
            module_name = line.strip()
            print(f"Would load module: {module_name}")
        except Exception as e:
            print(f"Error loading module: {e}")

    def do_history(self, line):
        """Show execution history."""
        if not self.execution_history:
            print("No execution history.")
            return

        for i, (input_data, result) in enumerate(self.execution_history, 1):
            print(f"{i}. Input: {input_data}")
            print(f"   Result: {result}")
            print()

    def do_inspect(self, line):
        """Inspect module state."""
        if not self.module:
            print("No module loaded.")
            return

        inspector = ModuleInspector()
        inspection = inspector.inspect_module(self.module)

        print(f"Module: {inspection['module_class']}")
        print(f"Parameters: {list(inspection['parameters'].keys())}")
        print(f"Submodules: {list(inspection['submodules'].keys())}")
        print(f"Storage keys: {list(inspection['storage'].keys())}")

    def do_profile(self, line):
        """Profile module execution."""
        if not self.module:
            print("No module loaded.")
            return

        profiler = PerformanceProfiler()
        with profiler.profile("debug_profile"):
            try:
                result = self.module.forward(line)
                print(f"Result: {result}")
            except Exception as e:
                print(f"Error: {e}")

        summary = profiler.get_profile_summary()
        print(f"\nProfile Summary:")
        print(f"Duration: {summary['total_duration']:.3f}s")
        print(f"Memory Delta: {summary['total_memory_delta']:.2f} MB")

    def do_tokens(self, line):
        """Show token usage statistics."""
        summary = token_tracker.get_usage_summary()
        print("\nToken Usage Summary:")
        print(f"Total Calls: {summary['total_calls']}")
        print(f"Prompt Tokens: {summary['total_prompt_tokens']}")
        print(f"Completion Tokens: {summary['total_completion_tokens']}")
        print(f"Total Tokens: {summary['total_tokens']}")
        print(f"Estimated Cost: ${summary['estimated_cost']:.4f}")

    def do_clear(self, line):
        """Clear execution history."""
        self.execution_history = []
        print("Execution history cleared.")

    def do_quit(self, line):
        """Exit debugger."""
        print("Goodbye!")
        return True

    def default(self, line):
        """Default command - treat as forward pass."""
        self.do_trace(line)
```

## Advanced Debugging Techniques

### 1. Prompt Engineering Debugger

```python
class PromptDebugger:
    """Debug prompt engineering and LM responses."""

    def __init__(self):
        self.prompt_history = []
        self.response_history = []

    def debug_prompt(self, prompt, response, expected_output=None):
        """Debug prompt-response interaction."""
        debug_info = {
            "timestamp": time.time(),
            "prompt_length": len(prompt),
            "response_length": len(response),
            "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
            "response": response[:500] + "..." if len(response) > 500 else response,
            "expected": expected_output
        }

        # Analyze prompt
        prompt_analysis = self._analyze_prompt(prompt)
        debug_info["prompt_analysis"] = prompt_analysis

        # Analyze response
        response_analysis = self._analyze_response(response)
        debug_info["response_analysis"] = response_analysis

        # Check for issues
        issues = self._detect_issues(debug_info)
        debug_info["issues"] = issues

        self.prompt_history.append(debug_info)
        return debug_info

    def _analyze_prompt(self, prompt):
        """Analyze prompt for potential issues."""
        analysis = {
            "has_examples": "Example:" in prompt,
            "has_instructions": "You are" in prompt or "Please" in prompt,
            "estimated_tokens": token_tracker.count_tokens(prompt),
            "format_specific": "JSON" in prompt or "XML" in prompt,
            "complexity": "high" if len(prompt.split()) > 200 else "medium" if len(prompt.split()) > 50 else "low"
        }
        return analysis

    def _analyze_response(self, response):
        """Analyze response for quality issues."""
        analysis = {
            "is_empty": len(response.strip()) == 0,
            "is_json_like": response.strip().startswith('{') or response.strip().startswith('['),
            "estimated_tokens": token_tracker.count_tokens(response),
            "completeness": "high" if len(response.split()) > 50 else "medium" if len(response.split()) > 10 else "low"
        }
        return analysis

    def _detect_issues(self, debug_info):
        """Detect potential issues in prompt-response pair."""
        issues = []

        # Check for empty response
        if debug_info["response_analysis"]["is_empty"]:
            issues.append("Empty response")

        # Check for very long prompts
        if debug_info["prompt_analysis"]["estimated_tokens"] > 3000:
            issues.append("Very long prompt - may exceed context window")

        # Check for missing instructions
        if not debug_info["prompt_analysis"]["has_instructions"]:
            issues.append("No clear instructions in prompt")

        # Check response completeness
        if debug_info["response_analysis"]["completeness"] == "low":
            issues.append("Very short response - may be incomplete")

        # Check format mismatch
        if debug_info["prompt_analysis"]["format_specific"] == "JSON":
            if not debug_info["response_analysis"]["is_json_like"]:
                issues.append("Expected JSON format but response is not JSON-like")

        return issues

    def get_prompt_suggestions(self, debug_info):
        """Get suggestions for improving prompt."""
        suggestions = []

        if "Empty response" in debug_info["issues"]:
            suggestions.append("Simplify the prompt or provide more context")

        if "No clear instructions" in debug_info["issues"]:
            suggestions.append("Add clear instructions about the desired output format")

        if "Very long prompt" in debug_info["issues"]:
            suggestions.append("Consider reducing prompt length or using summarization")

        if "Expected JSON format" in debug_info["issues"]:
            suggestions.append("Explicitly request JSON output in the prompt")

        return suggestions
```

### 2. Optimization Debugger

```python
class OptimizationDebugger:
    """Debug DSPy optimization processes."""

    def __init__(self):
        self.optimization_history = []

    def debug_optimization(self, optimizer, trainset, metric, compiled_module):
        """Debug optimization process."""
        debug_info = {
            "optimizer_type": type(optimizer).__name__,
            "trainset_size": len(trainset),
            "metric_type": type(metric).__name__,
            "optimization_start": time.time()
        }

        # Analyze optimizer configuration
        config_analysis = self._analyze_optimizer_config(optimizer)
        debug_info["optimizer_config"] = config_analysis

        # Analyze trainset
        trainset_analysis = self._analyze_trainset(trainset)
        debug_info["trainset_analysis"] = trainset_analysis

        self.optimization_history.append(debug_info)
        return debug_info

    def _analyze_optimizer_config(self, optimizer):
        """Analyze optimizer configuration."""
        analysis = {}

        # Check common optimizers
        if hasattr(optimizer, 'max_bootstrapped_demos'):
            analysis["max_bootstrapped_demos"] = optimizer.max_bootstrapped_demos
        if hasattr(optimizer, 'max_labeled_demos'):
            analysis["max_labeled_demos"] = optimizer.max_labeled_demos
        if hasattr(optimizer, 'max_rounds'):
            analysis["max_rounds"] = optimizer.max_rounds

        # Check for potential issues
        issues = []
        if analysis.get("max_bootstrapped_demos", 0) > 20:
            issues.append("Very high max_bootstrapped_demos - may be slow")
        if analysis.get("max_rounds", 0) > 5:
            issues.append("High max_rounds - may overfit")

        analysis["potential_issues"] = issues
        return analysis

    def _analyze_trainset(self, trainset):
        """Analyze training set quality."""
        if not trainset:
            return {"error": "Empty trainset"}

        analysis = {
            "size": len(trainset),
            "has_explanations": False
        }

        # Check for example diversity
        example_lengths = []
        has_explanations = 0

        for example in trainset:
            if hasattr(example, 'explanation') and example.explanation:
                has_explanations += 1

            # Estimate complexity
            text_length = 0
            for attr in dir(example):
                if not attr.startswith('_'):
                    value = getattr(example, attr)
                    if isinstance(value, str):
                        text_length += len(value)
            example_lengths.append(text_length)

        analysis["has_explanations"] = has_explanations > 0
        analysis["explanation_ratio"] = has_explanations / len(trainset)
        analysis["avg_example_length"] = sum(example_lengths) / len(example_lengths)
        analysis["length_variance"] = sum((x - analysis["avg_example_length"])**2 for x in example_lengths) / len(example_lengths)

        # Check for quality issues
        issues = []
        if analysis["explanation_ratio"] < 0.5:
            issues.append("Low explanation ratio - may affect optimization quality")
        if analysis["length_variance"] > analysis["avg_example_length"]:
            issues.append("High variance in example lengths - consider standardizing")

        analysis["quality_issues"] = issues
        return analysis
```

## Best Practices

### 1. Proactive Debugging
- Add logging and tracing from the start
- Use meaningful variable names and comments
- Implement validation checks
- Monitor performance metrics
- Set up error alerts

### 2. Systematic Debugging
- Reproduce issues consistently
- Isolate the problem area
- Use binary search approach
- Document findings
- Test fixes thoroughly

### 3. Production Debugging
- Implement comprehensive logging
- Use distributed tracing
- Monitor key metrics
- Set up alerting
- Keep debug information in production

## Key Takeaways

1. **Structured debugging** is essential for complex DSPy applications
2. **Tracing execution** helps understand flow and identify bottlenecks
3. **Performance monitoring** optimizes resource usage
4. **Interactive debugging** speeds up problem resolution
5. **Visual tools** make complex systems easier to understand
6. **Proactive monitoring** prevents issues in production

## Next Steps

In the next section, we'll explore **Deployment Strategies** to help you take your DSPy applications from development to production, covering various deployment scenarios and best practices.
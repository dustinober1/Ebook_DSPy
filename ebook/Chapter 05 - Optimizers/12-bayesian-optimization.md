# Bayesian Optimization for Prompt Tuning

## Introduction

Bayesian Optimization (BO) is a powerful global optimization technique that excels at optimizing expensive black-box functions with few evaluations. In the context of DSPy and prompt tuning, BO provides a principled approach to navigating the vast space of possible prompt configurations by building a probabilistic model of the performance landscape. This model allows BO to make intelligent decisions about which configurations to evaluate next, effectively balancing exploration (trying uncertain regions) and exploitation (refining promising areas).

### Learning Objectives

By the end of this section, you will:
- Understand Bayesian optimization principles and their application to prompt tuning
- Implement Gaussian Process-based optimization for prompts
- Master acquisition functions for intelligent exploration
- Apply BO to various prompt optimization scenarios
- Evaluate and tune BO hyperparameters for optimal performance

## Bayesian Optimization Fundamentals

### Core Components

Bayesian Optimization consists of four main components:

1. **Search Space**: The domain of possible configurations
2. **Surrogate Model**: A probabilistic model approximating the objective function
3. **Acquisition Function**: Guides the selection of next evaluation points
4. **Optimization Loop**: Iteratively selects and evaluates configurations

```python
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import dspy
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianPromptOptimizer:
    """
    Bayesian Optimization framework for prompt tuning in DSPy.
    """

    def __init__(
        self,
        task_signature,
        trainset,
        valset,
        metric_fn,
        search_space=None,
        surrogate_model="gp",  # Gaussian Process
        acquisition="ei",  # Expected Improvement
        max_iterations=100,
        n_initial_points=10,
        random_state=None
    ):
        self.task_signature = task_signature
        self.trainset = trainset
        self.valset = valset
        self.metric_fn = metric_fn
        self.search_space = search_space or self._define_search_space()
        self.max_iterations = max_iterations
        self.n_initial_points = n_initial_points
        self.random_state = random_state

        # Initialize components
        self.surrogate_model = self._create_surrogate_model(surrogate_model)
        self.acquisition_fn = self._create_acquisition_function(acquisition)

        # Storage for observations
        self.X_observed = []  # Evaluated configurations
        self.y_observed = []  # Corresponding scores

        # Track best solution
        self.best_config = None
        self.best_score = float("-inf")

    def _define_search_space(self):
        """Define the search space for prompt optimization."""
        return {
            "instruction_length": {"type": "discrete", "values": [10, 20, 30, 40, 50]},
            "instruction_style": {
                "type": "categorical",
                "values": ["direct", "polite", "detailed", "concise"]
            },
            "n_examples": {"type": "discrete", "values": [0, 1, 2, 3, 4, 5]},
            "example_complexity": {
                "type": "categorical",
                "values": ["simple", "medium", "complex"]
            },
            "temperature": {"type": "continuous", "bounds": [0.0, 1.0]},
            "top_p": {"type": "continuous", "bounds": [0.8, 1.0]},
            "max_tokens": {"type": "discrete", "values": [50, 100, 150, 200, 250]},
            "format_style": {
                "type": "categorical",
                "values": ["qa", "instruction", "conversation", "template"]
            }
        }

    def optimize(self):
        """Execute Bayesian optimization."""
        print(f"Starting Bayesian optimization for prompt tuning...")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Initial random points: {self.n_initial_points}")

        # Phase 1: Initial random exploration
        print("\n=== Phase 1: Initial Exploration ===")
        for i in range(self.n_initial_points):
            config = self._sample_random_configuration()
            score = self._evaluate_configuration(config)
            self._add_observation(config, score)

        # Phase 2: Bayesian optimization loop
        print("\n=== Phase 2: Bayesian Optimization ===")
        for iteration in range(self.max_iterations - self.n_initial_points):
            print(f"\nIteration {iteration + 1}")

            # Fit surrogate model
            self.surrogate_model.fit(self.X_observed, self.y_observed)

            # Find next point to evaluate
            next_config = self._select_next_configuration()

            # Evaluate selected configuration
            score = self._evaluate_configuration(next_config)
            self._add_observation(next_config, score)

            # Report progress
            print(f"Score: {score:.4f} (Best: {self.best_score:.4f})")

        return self.best_config, self.best_score
```

### Gaussian Process Surrogate Model

```python
class GaussianProcessSurrogate:
    """
    Gaussian Process surrogate model for Bayesian optimization.
    """

    def __init__(
        self,
        kernel="rbf",  # Radial Basis Function
        alpha=1e-6,  # Noise parameter
        length_scale=1.0,
        length_scale_bounds=(1e-1, 10.0)
    ):
        self.kernel = kernel
        self.alpha = alpha
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

        self.X_train = None
        self.y_train = None
        self.L = None  # Cholesky decomposition
        self.alpha_vec = None

    def fit(self, X, y):
        """Fit the Gaussian Process to observed data."""
        # Convert configurations to feature vectors
        X_encoded = self._encode_configurations(X)
        y = np.array(y)

        # Center the target values
        self.y_mean = np.mean(y)
        y_centered = y - self.y_mean

        # Compute kernel matrix
        K = self._compute_kernel_matrix(X_encoded)

        # Add noise for numerical stability
        K += self.alpha * np.eye(K.shape[0])

        # Cholesky decomposition
        self.L = np.linalg.cholesky(K)

        # Solve for alpha vector
        self.alpha_vec = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y_centered))

        # Store training data
        self.X_train = X_encoded

    def predict(self, X, return_std=False):
        """Predict mean and uncertainty for new configurations."""
        X_new = self._encode_configurations(X)

        # Compute kernel between new points and training points
        K_star = self._compute_cross_kernel(X_new, self.X_train)

        # Predict mean
        y_mean = K_star.dot(self.alpha_vec) + self.y_mean

        if return_std:
            # Compute variance
            v = np.linalg.solve(self.L, K_star.T)
            y_var = self._compute_kernel_diagonal(X_new) - np.sum(v ** 2, axis=0)
            y_std = np.sqrt(np.maximum(y_var, 1e-10))

            return y_mean, y_std

        return y_mean

    def _encode_configurations(self, configs):
        """Encode configurations as feature vectors."""
        if not configs:
            return np.array([[]])

        encoded = []
        for config in configs:
            vector = []
            for param_name, param_value in config.items():
                # One-hot encode categorical variables
                if isinstance(param_value, str):
                    # Get all possible values for this parameter
                    all_values = self._get_param_values(param_name)
                    one_hot = [1.0 if v == param_value else 0.0 for v in all_values]
                    vector.extend(one_hot)
                else:
                    # Normalize continuous and discrete values
                    normalized = self._normalize_parameter(param_name, param_value)
                    vector.append(normalized)
            encoded.append(vector)

        return np.array(encoded)

    def _compute_kernel_matrix(self, X):
        """Compute the kernel matrix."""
        if self.kernel == "rbf":
            # RBF kernel
            sq_dist = np.sum(X ** 2, axis=1).reshape(-1, 1) + \
                      np.sum(X ** 2, axis=1) - 2 * np.dot(X, X.T)
            K = np.exp(-0.5 / self.length_scale ** 2 * sq_dist)
        elif self.kernel == "matern":
            # Matérn kernel (ν = 3/2)
            dist = np.sqrt(np.sum(X ** 2, axis=1).reshape(-1, 1) + \
                          np.sum(X ** 2, axis=1) - 2 * np.dot(X, X.T))
            K = (1 + np.sqrt(3) * dist / self.length_scale) * \
                np.exp(-np.sqrt(3) * dist / self.length_scale)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

        return K

    def _compute_cross_kernel(self, X1, X2):
        """Compute kernel between two sets of points."""
        if self.kernel == "rbf":
            sq_dist = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + \
                      np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)
            K = np.exp(-0.5 / self.length_scale ** 2 * sq_dist)
        elif self.kernel == "matern":
            dist = np.sqrt(np.sum(X1 ** 2, axis=1).reshape(-1, 1) + \
                          np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T))
            K = (1 + np.sqrt(3) * dist / self.length_scale) * \
                np.exp(-np.sqrt(3) * dist / self.length_scale)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

        return K
```

### Acquisition Functions

Acquisition functions guide the optimization by balancing exploration and exploitation:

```python
class AcquisitionFunctions:
    """Collection of acquisition functions for Bayesian optimization."""

    @staticmethod
    def expected_improvement(mean, std, best_y, xi=0.01):
        """
        Expected Improvement acquisition function.

        Args:
            mean: Predicted mean values
            std: Predicted standard deviations
            best_y: Best observed value so far
            xi: Exploration-exploitation trade-off
        """
        with np.errstate(divide='warn'):
            imp = mean - best_y - xi
            Z = imp / std
            ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
            ei[std == 0.0] = 0.0

        return ei

    @staticmethod
    def probability_of_improvement(mean, std, best_y, xi=0.01):
        """
        Probability of Improvement acquisition function.
        """
        with np.errstate(divide='warn'):
            Z = (mean - best_y - xi) / std
            pi = norm.cdf(Z)
            pi[std == 0.0] = 0.0

        return pi

    @staticmethod
    def upper_confidence_bound(mean, std, kappa=2.576):
        """
        Upper Confidence Bound acquisition function.

        kappa determines the confidence level (2.576 for 99% confidence)
        """
        return mean + kappa * std

    @staticmethod
    def thompson_sampling(mean, std, n_samples=1000):
        """
        Thompson Sampling acquisition function.
        """
        samples = np.random.normal(mean, std, size=(n_samples, len(mean)))
        return np.mean(samples, axis=0)
```

## Advanced Bayesian Optimization Techniques

### Multi-Objective Bayesian Optimization

```python
class MultiObjectiveBayesianOptimizer:
    """
    Bayesian optimizer for multiple objectives (e.g., accuracy and latency).
    """

    def __init__(
        self,
        objectives,
        task_signature,
        trainset,
        valset,
        metric_fns,
        search_space=None,
        preference_weights=None
    ):
        self.objectives = objectives
        self.task_signature = task_signature
        self.trainset = trainset
        self.valset = valset
        self.metric_fns = metric_fns
        self.search_space = search_space or self._define_search_space()
        self.preference_weights = preference_weights or {obj: 1.0 for obj in objectives}

        # Initialize separate surrogate models for each objective
        self.surrogates = {
            obj: GaussianProcessSurrogate() for obj in objectives
        }

        # Storage
        self.X_observed = []
        self.y_observed = {obj: [] for obj in objectives}
        self.pareto_front = []

    def optimize(self, max_iterations=100):
        """Execute multi-objective optimization."""
        print(f"Starting multi-objective Bayesian optimization...")
        print(f"Objectives: {list(self.objectives)}")

        # Initial random exploration
        for i in range(self.n_initial_points):
            config = self._sample_random_configuration()
            scores = self._evaluate_multi_objective(config)
            self._add_observation(config, scores)

        # Optimization loop
        for iteration in range(max_iterations - self.n_initial_points):
            # Fit all surrogates
            for obj in self.objectives:
                self.surrogates[obj].fit(self.X_observed, self.y_observed[obj])

            # Select next configuration using hypervolume improvement
            next_config = self._select_next_hvi_configuration()

            # Evaluate
            scores = self._evaluate_multi_objective(next_config)
            self._add_observation(next_config, scores)

            # Update Pareto front
            self._update_pareto_front()

        return self.pareto_front

    def _select_next_hvi_configuration(self):
        """Select next configuration using Expected Hypervolume Improvement."""
        # Generate candidate configurations
        candidates = self._generate_candidates(1000)

        # Predict performance for all objectives
        predictions = {}
        uncertainties = {}
        for obj in self.objectives:
            mean, std = self.surrogates[obj].predict(candidates, return_std=True)
            predictions[obj] = mean
            uncertainties[obj] = std

        # Compute hypervolume improvement for each candidate
        hvi_scores = []
        reference_point = self._compute_reference_point()

        for i, candidate in enumerate(candidates):
            # Sample possible outcomes
            n_samples = 100
            samples = []
            for _ in range(n_samples):
                sample_scores = {}
                for obj in self.objectives:
                    sample = np.random.normal(
                        predictions[obj][i],
                        uncertainties[obj][i]
                    )
                    sample_scores[obj] = sample
                samples.append(sample_scores)

            # Compute expected hypervolume improvement
            hvi = self._expected_hypervolume_improvement(
                samples, reference_point
            )
            hvi_scores.append(hvi)

        # Select candidate with highest hypervolume improvement
        best_idx = np.argmax(hvi_scores)
        return candidates[best_idx]

    def _expected_hypervolume_improvement(self, samples, reference_point):
        """Compute expected hypervolume improvement."""
        # Compute hypervolume of current Pareto front
        current_hv = self._compute_hypervolume(self.pareto_front, reference_point)

        # Add samples to Pareto front and compute new hypervolumes
        hvs = []
        for sample in samples:
            temp_front = self.pareto_front + [sample]
            temp_front = self._filter_dominated(temp_front)
            hv = self._compute_hypervolume(temp_front, reference_point)
            hvs.append(hv)

        # Expected improvement
        return np.mean(hvs) - current_hv
```

### Contextual Bayesian Optimization

```python
class ContextualBayesianOptimizer:
    """
    Bayesian optimizer that considers context (e.g., task difficulty, domain).
    """

    def __init__(
        self,
        contexts,
        base_optimizer,
        context_features=None
    ):
        self.contexts = contexts
        self.base_optimizer = base_optimizer
        self.context_features = context_features or self._extract_context_features()

        # Learn context-dependent search spaces
        self.contextual_search_spaces = self._learn_contextual_spaces()

    def optimize_with_context(self, context, max_iterations=50):
        """Optimize for a specific context."""
        print(f"Optimizing for context: {context}")

        # Get context-specific search space
        search_space = self.contextual_search_spaces.get(context, self.base_optimizer.search_space)

        # Create context-aware optimizer
        context_optimizer = BayesianPromptOptimizer(
            task_signature=self.base_optimizer.task_signature,
            trainset=self.base_optimizer.trainset,
            valset=self.base_optimizer.valset,
            metric_fn=self.base_optimizer.metric_fn,
            search_space=search_space,
            max_iterations=max_iterations
        )

        # Warm-start with knowledge from similar contexts
        similar_contexts = self._find_similar_contexts(context)
        if similar_contexts:
            self._warm_start_optimizer(context_optimizer, similar_contexts)

        # Run optimization
        return context_optimizer.optimize()

    def _learn_contextual_spaces(self):
        """Learn context-specific search spaces."""
        contextual_spaces = {}

        for context in self.contexts:
            # Analyze successful configurations for this context
            successful_configs = self._get_successful_configs(context)

            # Infer promising ranges and values
            inferred_space = self._infer_search_space(successful_configs)
            contextual_spaces[context] = inferred_space

        return contextual_spaces
```

## Practical Implementation

### Complete Bayesian Optimization Pipeline

```python
def optimize_dspy_prompts_with_bo(
    task_type="qa",
    trainset_size=100,
    valset_size=50,
    optimization_budget=200
):
    """Complete Bayesian optimization pipeline for DSPy prompts."""

    # 1. Load and prepare data
    print("=== Loading and Preparing Data ===")
    trainset, valset = load_and_prepare_data(
        task_type=task_type,
        train_size=trainset_size,
        val_size=valset_size
    )

    # 2. Define task signature
    if task_type == "qa":
        class QASignature(dspy.Signature):
            """Answer questions based on provided context."""
            context = dspy.InputField(desc="Relevant context")
            question = dspy.InputField(desc="Question to answer")
            answer = dspy.OutputField(desc="Answer")

        task_signature = QASignature

    # 3. Define evaluation metric
    def evaluation_metric(example, pred, trace=None):
        if task_type == "qa":
            return evaluate_qa_performance(example, pred)
        # Add other task types as needed

    # 4. Create Bayesian optimizer
    print("\n=== Initializing Bayesian Optimizer ===")
    optimizer = BayesianPromptOptimizer(
        task_signature=task_signature,
        trainset=trainset,
        valset=valset,
        metric_fn=evaluation_metric,
        surrogate_model="gp",
        acquisition="ei",
        max_iterations=optimization_budget,
        n_initial_points=20
    )

    # 5. Run optimization
    print("\n=== Running Bayesian Optimization ===")
    best_config, best_score = optimizer.optimize()

    # 6. Create optimized prompt module
    print("\n=== Creating Optimized Module ===")
    optimized_module = create_module_from_config(
        task_signature,
        best_config
    )

    # 7. Evaluate on test set
    print("\n=== Final Evaluation ===")
    testset = load_test_data(task_type)
    final_score = evaluate_module(optimized_module, testset, evaluation_metric)

    # 8. Report results
    print(f"\n=== Optimization Results ===")
    print(f"Best validation score: {best_score:.4f}")
    print(f"Test score: {final_score:.4f}")
    print(f"Best configuration:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")

    return {
        "module": optimized_module,
        "config": best_config,
        "val_score": best_score,
        "test_score": final_score,
        "history": optimizer.X_observed,
        "scores": optimizer.y_observed
    }

def create_module_from_config(signature, config):
    """Create a DSPy module from optimized configuration."""
    # Build instruction
    instruction = build_instruction_from_config(config)

    # Create enhanced signature
    class OptimizedSignature(signature):
        instructions = instruction

    # Create module
    if config.get("chain_of_thought", False):
        module = dspy.ChainOfThought(OptimizedSignature)
    else:
        module = dspy.Predict(OptimizedSignature)

    # Configure LM parameters
    module.lm = module.lm.copy(
        temperature=config.get("temperature", 0.7),
        top_p=config.get("top_p", 0.9),
        max_tokens=config.get("max_tokens", 150)
    )

    # Add examples if configured
    if config.get("n_examples", 0) > 0:
        examples = select_examples(config["n_examples"])
        module = module.with_demos(examples)

    return module
```

### Example: Optimizing Chain-of-Thought Prompts

```python
class CoTBayesianOptimizer:
    """
    Specialized Bayesian optimizer for Chain-of-Thought prompts.
    """

    def __init__(self, task_signature, trainset, valset, metric_fn):
        self.task_signature = task_signature
        self.trainset = trainset
        self.valset = valset
        self.metric_fn = metric_fn

        # CoT-specific search space
        self.cot_search_space = {
            "reasoning_instruction": {
                "type": "categorical",
                "values": [
                    "Think step by step.",
                    "Break down the problem.",
                    "Reason through this carefully.",
                    "Work through this methodically."
                ]
            },
            "show_reasoning": {
                "type": "categorical",
                "values": ["before", "after", "integrated"]
            },
            "n_demonstrations": {"type": "discrete", "values": [0, 1, 2, 3]},
            "demo_complexity": {
                "type": "categorical",
                "values": ["minimal", "detailed", "verbose"]
            },
            "final_instruction": {
                "type": "categorical",
                "values": [
                    "Finally, provide the answer.",
                    "Now give the final answer.",
                    "The answer is:",
                    "Therefore:"
                ]
            }
        }

        self.optimizer = BayesianPromptOptimizer(
            task_signature=task_signature,
            trainset=trainset,
            valset=valset,
            metric_fn=metric_fn,
            search_space=self.cot_search_space
        )

    def optimize_cot(self, max_iterations=100):
        """Optimize Chain-of-Thought prompt configuration."""
        print("=== Optimizing Chain-of-Thought Configuration ===")

        # Special evaluation for CoT
        def cot_metric(example, pred, trace=None):
            # Base task performance
            base_score = self.metric_fn(example, pred)

            # Reasoning quality
            reasoning_score = evaluate_reasoning_quality(
                pred.get("rationale", ""),
                example.get("reasoning_steps", [])
            )

            # Combined score
            return 0.7 * base_score + 0.3 * reasoning_score

        # Update optimizer metric
        self.optimizer.metric_fn = cot_metric

        # Run optimization
        return self.optimizer.optimize(max_iterations=max_iterations)

    def create_cot_module(self, config):
        """Create optimized CoT module."""
        # Build CoT instruction
        cot_instruction = build_cot_instruction(config)

        # Create enhanced signature
        class OptimizedCoTSignature(self.task_signature):
            instructions = cot_instruction

        # Create CoT module
        cot_module = dspy.ChainOfThought(OptimizedCoTSignature)

        # Add demonstrations
        if config["n_demonstrations"] > 0:
            demos = create_cot_demonstrations(
                self.trainset[:config["n_demonstrations"]],
                config["demo_complexity"]
            )
            cot_module = cot_module.with_demos(demos)

        return cot_module
```

## Evaluation and Analysis

### Convergence Analysis

```python
def analyze_bo_convergence(optimizer_history, true_optimum=None):
    """Analyze the convergence of Bayesian optimization."""
    iterations = range(len(optimizer_history["scores"]))
    scores = optimizer_history["scores"]
    best_so_far = np.maximum.accumulate(scores)

    # Plot convergence
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(iterations, scores, 'o-', alpha=0.5, label='All evaluations')
    plt.plot(iterations, best_so_far, 'r-', linewidth=2, label='Best so far')
    if true_optimum:
        plt.axhline(y=true_optimum, color='g', linestyle='--', label='True optimum')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title('BO Convergence')
    plt.legend()

    # Plot exploration vs exploitation
    plt.subplot(1, 2, 2)
    uncertainty = optimizer_history.get("uncertainty", [])
    if uncertainty:
        plt.scatter(iterations, uncertainty, c=scores, cmap='viridis', alpha=0.6)
        plt.colorbar(label='Score')
        plt.xlabel('Iteration')
        plt.ylabel('Uncertainty')
        plt.title('Exploration Pattern')

    plt.tight_layout()
    plt.show()

    # Compute convergence metrics
    convergence_metrics = {
        "initial_improvement": best_so_far[10] - scores[0] if len(scores) > 10 else 0,
        "final_improvement": best_so_far[-1] - best_so_far[10] if len(scores) > 10 else best_so_far[-1] - scores[0],
        "iterations_to_90_percent": np.where(best_so_far >= 0.9 * best_so_far[-1])[0][0] if any(best_so_far >= 0.9 * best_so_far[-1]) else len(scores) - 1,
        "regret": (true_optimum - best_so_far[-1]) if true_optimum else None
    }

    print("\n=== Convergence Metrics ===")
    for metric, value in convergence_metrics.items():
        if value is not None:
            print(f"{metric}: {value:.4f}")

    return convergence_metrics
```

### Comparison with Other Methods

```python
def compare_optimization_methods(task_data, budget=200):
    """Compare Bayesian optimization with other optimization methods."""

    methods = {
        "Bayesian Optimization": lambda: run_bayesian_optimization(task_data, budget),
        "Random Search": lambda: run_random_search(task_data, budget),
        "Grid Search": lambda: run_grid_search(task_data, budget//10),  # Grid search is expensive
        "Genetic Algorithm": lambda: run_genetic_algorithm(task_data, budget)
    }

    results = {}

    for method_name, method_fn in methods.items():
        print(f"\n=== Running {method_name} ===")
        start_time = time.time()
        best_config, best_score, history = method_fn()
        end_time = time.time()

        results[method_name] = {
            "best_score": best_score,
            "best_config": best_config,
            "time": end_time - start_time,
            "history": history
        }

        print(f"Best score: {best_score:.4f}")
        print(f"Time: {end_time - start_time:.2f}s")

    # Analysis
    print("\n=== Comparison Summary ===")
    for method, result in results.items():
        efficiency = result["best_score"] / result["time"]
        print(f"{method}:")
        print(f"  Score: {result['best_score']:.4f}")
        print(f"  Time: {result['time']:.2f}s")
        print(f"  Efficiency: {efficiency:.6f}")

    return results
```

## Best Practices

### Bayesian Optimization Configuration

```python
# For high-dimensional spaces
high_dim_config = {
    "surrogate_model": "gp",
    "kernel": "matern",  # Better for high dimensions
    "acquisition": "ei",
    "max_iterations": 500,
    "n_initial_points": 50  # More initial points
}

# For noisy evaluations
noisy_config = {
    "surrogate_model": "gp",
    "alpha": 1e-3,  # Higher noise parameter
    "acquisition": "ucb",  # More exploration
    "kappa": 2.0,
    "max_iterations": 300
}

# For expensive evaluations
expensive_config = {
    "max_iterations": 50,  # Fewer evaluations
    "n_initial_points": 10,
    "acquisition": "ei",  # Good exploitation
    "xi": 0.1  # More exploitation
}
```

### Common Pitfalls and Solutions

1. **Poor Search Space Definition**:
   - Problem: Too large or inappropriate search space
   - Solution: Start with a focused search space and expand if needed

2. **Insufficient Initial Points**:
   - Problem: Poor initial model fitting
   - Solution: Use at least 10-20 initial random points

3. **Local Optima**:
   - Problem: Getting stuck in suboptimal regions
   - Solution: Use exploration-focused acquisition functions

4. **Noisy Evaluations**:
   - Problem: Inconsistent evaluation scores
   - Solution: Increase noise parameter and use multiple evaluations

## Summary

Bayesian optimization provides a principled and efficient approach to prompt tuning in DSPy. By building a probabilistic model of the performance landscape, BO can make intelligent decisions about which configurations to evaluate next, achieving superior performance with fewer evaluations compared to traditional optimization methods.

### Key Takeaways

1. BO balances exploration and exploitation through acquisition functions
2. Gaussian Processes provide effective surrogate models for prompt optimization
3. Multi-objective optimization can handle multiple metrics simultaneously
4. Contextual BO adapts optimization to different task characteristics
5. Proper configuration is crucial for success

## Next Steps

In the next section, we'll explore advanced optimization strategies that combine multiple techniques and discuss how to choose the right optimizer for specific use cases.
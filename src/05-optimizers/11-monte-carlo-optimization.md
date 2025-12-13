# Monte Carlo Optimization in DSPy

## Introduction

Monte Carlo methods provide powerful stochastic optimization techniques that excel in complex, non-convex optimization spaces typical of language model systems. In DSPy, Monte Carlo optimization offers a robust approach to navigate the vast space of possible prompt configurations, model parameters, and program structures. Unlike gradient-based methods that require differentiable objectives, Monte Carlo techniques work with any black-box evaluation function, making them particularly suitable for prompt optimization and discrete parameter search.

### Learning Objectives

By the end of this section, you will:
- Understand Monte Carlo optimization principles in the context of DSPy
- Implement various Monte Carlo optimization strategies
- Apply Monte Carlo methods to prompt and parameter optimization
- Master techniques for efficient exploration and exploitation
- Evaluate and tune Monte Carlo optimizers for different tasks

## Monte Carlo Optimization Fundamentals

### Core Concepts

Monte Carlo optimization relies on random sampling to explore the solution space:

1. **Random Exploration**: Sample points from the search space
2. **Evaluation**: Assess the quality of each sample
3. **Adaptive Sampling**: Focus exploration on promising regions
4. **Convergence**: Gradually converge to optimal solutions

```python
import random
import numpy as np
from typing import List, Dict, Any, Callable
import dspy

class MonteCarloOptimizer:
    """
    Base class for Monte Carlo optimization in DSPy.
    """

    def __init__(
        self,
        evaluation_fn: Callable,
        search_space: Dict[str, Any],
        max_iterations: int = 1000,
        exploration_rate: float = 0.3,
        convergence_threshold: float = 1e-4,
        random_seed: int = None
    ):
        self.evaluation_fn = evaluation_fn
        self.search_space = search_space
        self.max_iterations = max_iterations
        self.exploration_rate = exploration_rate
        self.convergence_threshold = convergence_threshold
        self.random_seed = random_seed

        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # Track optimization history
        self.history = {
            "iterations": [],
            "scores": [],
            "best_scores": [],
            "samples": []
        }

        self.best_solution = None
        self.best_score = float("-inf")

    def optimize(self):
        """Execute Monte Carlo optimization."""
        raise NotImplementedError("Subclasses must implement optimize()")
```

### Random Search Monte Carlo

The simplest Monte Carlo approach:

```python
class RandomSearchMonteCarlo(MonteCarloOptimizer):
    """
    Random search Monte Carlo optimization.
    """

    def optimize(self):
        """Execute random search optimization."""
        print(f"Starting Random Search Monte Carlo optimization...")
        print(f"Max iterations: {self.max_iterations}")

        for iteration in range(self.max_iterations):
            # Sample a random solution
            solution = self._sample_solution()
            score = self.evaluation_fn(solution)

            # Update history
            self.history["iterations"].append(iteration)
            self.history["scores"].append(score)
            self.history["samples"].append(solution)

            # Track best solution
            if score > self.best_score:
                self.best_score = score
                self.best_solution = solution.copy()

            self.history["best_scores"].append(self.best_score)

            # Progress report
            if (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}: Best score = {self.best_score:.4f}")

            # Early stopping check
            if self._check_convergence():
                print(f"Converged at iteration {iteration + 1}")
                break

        return self.best_solution, self.best_score

    def _sample_solution(self):
        """Sample a random solution from search space."""
        solution = {}

        for param_name, param_config in self.search_space.items():
            param_type = param_config["type"]

            if param_type == "categorical":
                solution[param_name] = random.choice(param_config["values"])
            elif param_type == "continuous":
                solution[param_name] = random.uniform(
                    param_config["min"], param_config["max"]
                )
            elif param_type == "integer":
                solution[param_name] = random.randint(
                    param_config["min"], param_config["max"]
                )
            elif param_type == "string_template":
                # For prompt templates
                solution[param_name] = self._sample_string_template(param_config)

        return solution

    def _sample_string_template(self, config):
        """Sample a string template from configuration."""
        if "templates" in config:
            return random.choice(config["templates"])
        elif "components" in config:
            # Build template from components
            template = ""
            for component in config["components"]:
                if random.random() < 0.5:
                    template += component + "\n"
            return template
        else:
            return config.get("default", "")
```

### Simulated Annealing

A more sophisticated Monte Carlo method with temperature-based exploration:

```python
class SimulatedAnnealingMonteCarlo(MonteCarloOptimizer):
    """
    Simulated annealing Monte Carlo optimization.
    """

    def __init__(
        self,
        *args,
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.95,
        min_temperature: float = 0.01,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.temperature = initial_temperature

    def optimize(self):
        """Execute simulated annealing optimization."""
        print(f"Starting Simulated Annealing optimization...")
        print(f"Initial temperature: {self.initial_temperature}")
        print(f"Cooling rate: {self.cooling_rate}")

        # Initialize with random solution
        current_solution = self._sample_solution()
        current_score = self.evaluation_fn(current_solution)

        self.best_solution = current_solution.copy()
        self.best_score = current_score

        for iteration in range(self.max_iterations):
            # Generate neighbor solution
            neighbor_solution = self._generate_neighbor(current_solution)
            neighbor_score = self.evaluation_fn(neighbor_solution)

            # Calculate acceptance probability
            delta_score = neighbor_score - current_score
            if delta_score > 0:
                accept_prob = 1.0
            else:
                accept_prob = np.exp(delta_score / self.temperature)

            # Accept or reject
            if random.random() < accept_prob:
                current_solution = neighbor_solution
                current_score = neighbor_score

                # Update best if improved
                if current_score > self.best_score:
                    self.best_solution = current_solution.copy()
                    self.best_score = current_score

            # Update history
            self.history["iterations"].append(iteration)
            self.history["scores"].append(current_score)
            self.history["best_scores"].append(self.best_score)

            # Cool down
            self.temperature = max(
                self.min_temperature,
                self.temperature * self.cooling_rate
            )

            # Progress report
            if (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}: Score = {current_score:.4f}, "
                      f"Best = {self.best_score:.4f}, Temp = {self.temperature:.4f}")

            # Check convergence
            if self._check_convergence():
                print(f"Converged at iteration {iteration + 1}")
                break

        return self.best_solution, self.best_score

    def _generate_neighbor(self, solution):
        """Generate a neighbor solution by small modifications."""
        neighbor = solution.copy()

        # Randomly select a parameter to modify
        param_name = random.choice(list(self.search_space.keys()))
        param_config = self.search_space[param_name]

        if param_config["type"] == "categorical":
            # Choose a different categorical value
            current_value = solution[param_name]
            available_values = [v for v in param_config["values"] if v != current_value]
            if available_values:
                neighbor[param_name] = random.choice(available_values)

        elif param_config["type"] in ["continuous", "integer"]:
            # Add Gaussian noise
            current_value = solution[param_name]
            noise_scale = (param_config["max"] - param_config["min"]) * 0.1

            if param_config["type"] == "continuous":
                new_value = current_value + np.random.normal(0, noise_scale)
                neighbor[param_name] = np.clip(
                    new_value,
                    param_config["min"],
                    param_config["max"]
                )
            else:  # integer
                new_value = int(current_value + np.random.normal(0, noise_scale / 2))
                neighbor[param_name] = max(
                    param_config["min"],
                    min(param_config["max"], new_value)
                )

        elif param_config["type"] == "string_template":
            # Modify prompt template
            neighbor[param_name] = self._modify_string_template(
                solution[param_name], param_config
            )

        return neighbor
```

## Advanced Monte Carlo Methods

### Cross-Entropy Method

```python
class CrossEntropyMonteCarlo(MonteCarloOptimizer):
    """
    Cross-Entropy Method for optimization.
    """

    def __init__(
        self,
        *args,
        population_size: int = 100,
        elite_fraction: float = 0.1,
        smoothing_factor: float = 0.7,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.elite_size = int(population_size * elite_fraction)
        self.smoothing_factor = smoothing_factor

    def optimize(self):
        """Execute Cross-Entropy optimization."""
        print(f"Starting Cross-Entropy optimization...")
        print(f"Population size: {self.population_size}")
        print(f"Elite fraction: {self.elite_fraction}")

        # Initialize parameter distributions
        distributions = self._initialize_distributions()

        for iteration in range(self.max_iterations):
            # Sample population
            population = self._sample_population(distributions)

            # Evaluate population
            scores = [self.evaluation_fn(ind) for ind in population]

            # Select elite
            elite_indices = np.argsort(scores)[-self.elite_size:]
            elite_population = [population[i] for i in elite_indices]

            # Update distributions based on elite
            distributions = self._update_distributions(
                distributions, elite_population
            )

            # Update best solution
            best_idx = elite_indices[-1]
            if scores[best_idx] > self.best_score:
                self.best_score = scores[best_idx]
                self.best_solution = population[best_idx].copy()

            # Update history
            self.history["iterations"].append(iteration)
            self.history["scores"].append(np.mean(scores))
            self.history["best_scores"].append(self.best_score)

            # Progress report
            if (iteration + 1) % 50 == 0:
                print(f"Iteration {iteration + 1}: "
                      f"Mean score = {np.mean(scores):.4f}, "
                      f"Best = {self.best_score:.4f}")

            # Check convergence
            if self._check_convergence():
                print(f"Converged at iteration {iteration + 1}")
                break

        return self.best_solution, self.best_score

    def _initialize_distributions(self):
        """Initialize probability distributions for parameters."""
        distributions = {}

        for param_name, param_config in self.search_space.items():
            if param_config["type"] == "categorical":
                # Uniform distribution over categorical values
                distributions[param_name] = {
                    "type": "categorical",
                    "probabilities": np.ones(len(param_config["values"])) / len(param_config["values"]),
                    "values": param_config["values"]
                }
            elif param_config["type"] == "continuous":
                # Normal distribution
                distributions[param_name] = {
                    "type": "continuous",
                    "mean": (param_config["min"] + param_config["max"]) / 2,
                    "std": (param_config["max"] - param_config["min"]) / 4
                }
            elif param_config["type"] == "integer":
                # Discrete distribution
                values = list(range(param_config["min"], param_config["max"] + 1))
                distributions[param_name] = {
                    "type": "discrete",
                    "probabilities": np.ones(len(values)) / len(values),
                    "values": values
                }

        return distributions

    def _update_distributions(self, distributions, elite_population):
        """Update distributions based on elite solutions."""
        for param_name in self.search_space.keys():
            dist = distributions[param_name]

            if dist["type"] in ["categorical", "discrete"]:
                # Count occurrences in elite
                counts = {}
                for individual in elite_population:
                    value = individual[param_name]
                    counts[value] = counts.get(value, 0) + 1

                # Update probabilities with smoothing
                new_probs = []
                for value in dist["values"]:
                    count = counts.get(value, 0)
                    old_prob = dist["probabilities"][dist["values"].index(value)]
                    new_prob = (
                        self.smoothing_factor * old_prob +
                        (1 - self.smoothing_factor) * count / len(elite_population)
                    )
                    new_probs.append(new_prob)

                # Normalize
                dist["probabilities"] = np.array(new_probs) / np.sum(new_probs)

            elif dist["type"] == "continuous":
                # Update mean and std
                values = [ind[param_name] for ind in elite_population]
                new_mean = np.mean(values)
                new_std = np.std(values)

                # Smooth update
                dist["mean"] = (
                    self.smoothing_factor * dist["mean"] +
                    (1 - self.smoothing_factor) * new_mean
                )
                dist["std"] = max(
                    0.01,
                    self.smoothing_factor * dist["std"] +
                    (1 - self.smoothing_factor) * new_std
                )

        return distributions
```

### Particle Swarm Optimization

```python
class ParticleSwarmMonteCarlo(MonteCarloOptimizer):
    """
    Particle Swarm Optimization for DSPy.
    """

    def __init__(
        self,
        *args,
        swarm_size: int = 50,
        inertia_weight: float = 0.7,
        cognitive_weight: float = 1.5,
        social_weight: float = 1.5,
        velocity_clamp: float = 0.2,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.swarm_size = swarm_size
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.velocity_clamp = velocity_clamp

    def optimize(self):
        """Execute Particle Swarm optimization."""
        print(f"Starting Particle Swarm optimization...")
        print(f"Swarm size: {self.swarm_size}")

        # Initialize swarm
        particles = self._initialize_swarm()
        velocities = self._initialize_velocities()
        personal_best = particles.copy()
        personal_best_scores = [self.evaluation_fn(p) for p in particles]
        global_best_idx = np.argmax(personal_best_scores)
        global_best = personal_best[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]

        for iteration in range(self.max_iterations):
            for i in range(self.swarm_size):
                # Update velocity
                for param_name in self.search_space.keys():
                    param_config = self.search_space[param_config]

                    if param_config["type"] in ["continuous", "integer"]:
                        # Continuous space update
                        r1, r2 = random.random(), random.random()

                        cognitive_term = (
                            self.cognitive_weight * r1 *
                            (personal_best[i][param_name] - particles[i][param_name])
                        )
                        social_term = (
                            self.social_weight * r2 *
                            (global_best[param_name] - particles[i][param_name])
                        )

                        velocities[i][param_name] = (
                            self.inertia_weight * velocities[i][param_name] +
                            cognitive_term + social_term
                        )

                        # Clamp velocity
                        max_vel = self.velocity_clamp * (
                            param_config["max"] - param_config["min"]
                        )
                        velocities[i][param_name] = np.clip(
                            velocities[i][param_name], -max_vel, max_vel
                        )

                        # Update position
                        particles[i][param_name] += velocities[i][param_name]
                        particles[i][param_name] = np.clip(
                            particles[i][param_name],
                            param_config["min"],
                            param_config["max"]
                        )

                    elif param_config["type"] == "categorical":
                        # Probabilistic update for categorical
                        if random.random() < self.inertia_weight:
                            # Keep current with inertia
                            pass
                        elif random.random() < self.cognitive_weight:
                            # Move toward personal best
                            if random.random() < 0.5:
                                particles[i][param_name] = personal_best[i][param_name]
                        elif random.random() < self.social_weight:
                            # Move toward global best
                            if random.random() < 0.5:
                                particles[i][param_name] = global_best[param_name]

                # Evaluate new position
                score = self.evaluation_fn(particles[i])

                # Update personal best
                if score > personal_best_scores[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_scores[i] = score

                    # Update global best
                    if score > global_best_score:
                        global_best = particles[i].copy()
                        global_best_score = score

            # Update history
            self.history["iterations"].append(iteration)
            self.history["scores"].append(np.mean(personal_best_scores))
            self.history["best_scores"].append(global_best_score)

            # Progress report
            if (iteration + 1) % 50 == 0:
                print(f"Iteration {iteration + 1}: "
                      f"Mean best = {np.mean(personal_best_scores):.4f}, "
                      f"Global best = {global_best_score:.4f}")

            # Check convergence
            if self._check_convergence():
                print(f"Converged at iteration {iteration + 1}")
                break

        self.best_solution = global_best
        self.best_score = global_best_score

        return self.best_solution, self.best_score
```

## Monte Carlo for Prompt Optimization

### Prompt Space Definition

```python
def define_prompt_search_space(task_type="qa"):
    """Define the search space for prompt optimization."""

    if task_type == "qa":
        return {
            "instruction": {
                "type": "string_template",
                "templates": [
                    "Answer the following question based on the given context.",
                    "Using the provided context, please answer the question.",
                    "Given the context, provide a comprehensive answer to the question.",
                    "Based on the information below, respond to the question."
                ],
                "components": [
                    "Be precise and accurate.",
                    "Use only the information provided.",
                    "If the answer is not in the context, say so.",
                    "Provide a detailed explanation."
                ]
            },
            "context_format": {
                "type": "categorical",
                "values": [
                    "Context: {context}\nQuestion: {question}",
                    "{context}\n\nQ: {question}\nA:",
                    "Given this context:\n{context}\n\nAnswer this question: {question}",
                    "Information:\n{context}\n\nQuery: {question}"
                ]
            },
            "max_examples": {
                "type": "integer",
                "min": 0,
                "max": 8
            },
            "example_format": {
                "type": "categorical",
                "values": [
                    "Q: {q}\nA: {a}",
                    "Question: {q}\nAnswer: {a}",
                    "{q} -> {a}",
                    "Example {i}:\nQuestion: {q}\nAnswer: {a}"
                ]
            },
            "temperature": {
                "type": "continuous",
                "min": 0.0,
                "max": 1.0
            }
        }

    elif task_type == "classification":
        return {
            "instruction": {
                "type": "string_template",
                "templates": [
                    "Classify the given text into one of the provided categories.",
                    "Determine which category the following text belongs to.",
                    "Select the appropriate category for this text.",
                    "Categorize this text based on its content."
                ]
            },
            "categories_format": {
                "type": "categorical",
                "values": [
                    "Categories: {categories}",
                    "Choose from: {categories}",
                    "Available categories: {categories}"
                ]
            },
            "text_prefix": {
                "type": "categorical",
                "values": ["", "Text: ", "Input: ", "Given: "]
            },
            "zero_shot": {
                "type": "categorical",
                "values": [True, False]
            },
            "temperature": {
                "type": "continuous",
                "min": 0.0,
                "max": 0.5
            }
        }

    elif task_type == "generation":
        return {
            "instruction": {
                "type": "string_template",
                "templates": [
                    "Generate {type} based on the given prompt.",
                    "Write {type} according to these requirements.",
                    "Create {type} that satisfies the following criteria.",
                    "Produce {type} following the specified guidelines."
                ]
            },
            "length_guidance": {
                "type": "categorical",
                "values": [
                    "Be concise and brief.",
                    "Provide a detailed response.",
                    "Write approximately {length} words.",
                    "Keep it under {length} words."
                ]
            },
            "style_guidance": {
                "type": "categorical",
                "values": [
                    "Use a formal tone.",
                    "Write in a casual style.",
                    "Be professional and clear.",
                    "Use a creative and engaging tone."
                ]
            },
            "temperature": {
                "type": "continuous",
                "min": 0.7,
                "max": 1.0
            },
            "top_p": {
                "type": "continuous",
                "min": 0.8,
                "max": 1.0
            }
        }
```

### Prompt Optimization Implementation

```python
class MonteCarloPromptOptimizer:
    """
    Monte Carlo optimizer specifically for prompt optimization in DSPy.
    """

    def __init__(
        self,
        task_signature,
        trainset,
        valset,
        metric_fn,
        search_space=None,
        optimizer_type="simulated_annealing",
        **optimizer_kwargs
    ):
        self.task_signature = task_signature
        self.trainset = trainset
        self.valset = valset
        self.metric_fn = metric_fn
        self.search_space = search_space or define_prompt_search_space()
        self.optimizer_type = optimizer_type

        # Create optimizer
        self.optimizer = self._create_optimizer(optimizer_kwargs)

    def _create_optimizer(self, optimizer_kwargs):
        """Create the Monte Carlo optimizer."""
        evaluation_fn = lambda config: self._evaluate_prompt_configuration(config)

        if self.optimizer_type == "random_search":
            return RandomSearchMonteCarlo(
                evaluation_fn=evaluation_fn,
                search_space=self.search_space,
                **optimizer_kwargs
            )
        elif self.optimizer_type == "simulated_annealing":
            return SimulatedAnnealingMonteCarlo(
                evaluation_fn=evaluation_fn,
                search_space=self.search_space,
                **optimizer_kwargs
            )
        elif self.optimizer_type == "cross_entropy":
            return CrossEntropyMonteCarlo(
                evaluation_fn=evaluation_fn,
                search_space=self.search_space,
                **optimizer_kwargs
            )
        elif self.optimizer_type == "particle_swarm":
            return ParticleSwarmMonteCarlo(
                evaluation_fn=evaluation_fn,
                search_space=self.search_space,
                **optimizer_kwargs
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

    def _evaluate_prompt_configuration(self, config):
        """Evaluate a prompt configuration."""
        # Create prompt template from configuration
        prompt_template = self._create_prompt_template(config)

        # Create DSPy module with the prompt
        module = self._create_module_with_prompt(prompt_template, config)

        # Evaluate on validation set
        total_score = 0
        for example in self.valset:
            prediction = module(**example.inputs())
            score = self.metric_fn(example, prediction)
            total_score += score

        return total_score / len(self.valset)

    def _create_prompt_template(self, config):
        """Create a prompt template from configuration."""
        template_parts = []

        # Add instruction
        if "instruction" in config:
            template_parts.append(config["instruction"])

        # Add format
        if "context_format" in config:
            format_template = config["context_format"]
        elif "categories_format" in config:
            format_template = config["categories_format"]
        else:
            format_template = ""

        # Add examples if configured
        if config.get("max_examples", 0) > 0:
            examples = self._select_examples(config["max_examples"])
            example_text = self._format_examples(examples, config)
            template_parts.append(example_text)

        # Combine parts
        full_template = "\n\n".join(template_parts)
        if format_template:
            full_template += "\n\n" + format_template

        return full_template

    def optimize(self):
        """Execute prompt optimization."""
        print(f"Starting Monte Carlo prompt optimization...")
        print(f"Optimizer type: {self.optimizer_type}")
        print(f"Validation set size: {len(self.valset)}")

        # Run optimization
        best_config, best_score = self.optimizer.optimize()

        # Create final optimized module
        final_prompt = self._create_prompt_template(best_config)
        final_module = self._create_module_with_prompt(final_prompt, best_config)

        return {
            "module": final_module,
            "config": best_config,
            "score": best_score,
            "history": self.optimizer.history,
            "prompt": final_prompt
        }

    def _create_module_with_prompt(self, prompt_template, config):
        """Create a DSPy module with the optimized prompt."""
        # Create custom signature with the prompt
        class OptimizedSignature(self.task_signature):
            instructions = prompt_template

        # Create module
        if "chain_of_thought" in config and config["chain_of_thought"]:
            module = dspy.ChainOfThought(OptimizedSignature)
        else:
            module = dspy.Predict(OptimizedSignature)

        # Configure LM parameters
        if "temperature" in config:
            module.lm = module.lm.copy(temperature=config["temperature"])

        if "top_p" in config:
            module.lm = module.lm.copy(top_p=config["top_p"])

        return module
```

## Practical Examples

### Example 1: QA System Optimization

```python
def optimize_qa_system():
    """Optimize a QA system using Monte Carlo methods."""

    # Define QA signature
    class QASignature(dspy.Signature):
        """Answer questions based on provided context."""

        context = dspy.InputField(desc="Relevant context for answering")
        question = dspy.InputField(desc="Question to be answered")
        answer = dspy.OutputField(desc="Answer to the question")

    # Load data
    trainset = load_qa_trainset()
    valset = load_qa_valset()

    # Define metric
    def qa_metric(example, pred, trace=None):
        return exact_match_score(example.answer, pred.answer)

    # Create optimizer
    optimizer = MonteCarloPromptOptimizer(
        task_signature=QASignature,
        trainset=trainset,
        valset=valset,
        metric_fn=qa_metric,
        optimizer_type="simulated_annealing",
        max_iterations=500,
        initial_temperature=1.0,
        cooling_rate=0.99
    )

    # Run optimization
    result = optimizer.optimize()

    # Report results
    print("\n=== Optimization Results ===")
    print(f"Best score: {result['score']:.4f}")
    print(f"Best configuration:")
    for key, value in result['config'].items():
        print(f"  {key}: {value}")
    print(f"\nOptimized prompt:\n{result['prompt']}")

    return result
```

### Example 2: Multi-Task Prompt Optimization

```python
class MultiTaskMonteCarloOptimizer:
    """
    Monte Carlo optimizer for multiple related tasks.
    """

    def __init__(self, tasks, shared_search_space=None):
        self.tasks = tasks
        self.shared_search_space = shared_search_space or define_prompt_search_space()
        self.task_optimizers = {}

    def optimize_jointly(self, max_iterations=500):
        """Optimize prompts for all tasks jointly."""
        print(f"Starting joint optimization for {len(self.tasks)} tasks")

        # Initialize optimizers for each task
        for task_name, task_data in self.tasks.items():
            self.task_optimizers[task_name] = MonteCarloPromptOptimizer(
                task_signature=task_data["signature"],
                trainset=task_data["trainset"],
                valset=task_data["valset"],
                metric_fn=task_data["metric"],
                search_space=self.shared_search_space,
                optimizer_type="cross_entropy",
                max_iterations=max_iterations,
                population_size=100
            )

        # Joint optimization loop
        best_configs = {task_name: None for task_name in self.tasks}
        best_scores = {task_name: 0 for task_name in self.tasks}
        shared_config = None

        for iteration in range(max_iterations // 10):  # Outer iterations
            print(f"\nJoint optimization iteration {iteration + 1}")

            # Evaluate each task with current shared config
            if shared_config:
                task_scores = {}
                for task_name in self.tasks:
                    score = self.task_optimizers[task_name]._evaluate_prompt_configuration(
                        shared_config
                    )
                    task_scores[task_name] = score

                avg_score = np.mean(list(task_scores.values()))
                print(f"Average score with shared config: {avg_score:.4f}")

                # Update best if improved
                if avg_score > np.mean(list(best_scores.values())):
                    for task_name in self.tasks:
                        best_configs[task_name] = shared_config.copy()
                        best_scores[task_name] = task_scores[task_name]

            # Optimize each task independently for a few iterations
            for task_name in self.tasks:
                print(f"\nOptimizing task: {task_name}")
                task_result = self.task_optimizers[task_name].optimize()

                # Update shared config if task improved significantly
                if task_result["score"] > best_scores[task_name] * 1.1:
                    shared_config = task_result["config"].copy()

        return {
            "best_configs": best_configs,
            "best_scores": best_scores,
            "shared_config": shared_config
        }

# Usage
tasks = {
    "qa": {
        "signature": QASignature,
        "trainset": load_qa_trainset(),
        "valset": load_qa_valset(),
        "metric": qa_metric
    },
    "summarization": {
        "signature": SummarizationSignature,
        "trainset": load_sum_trainset(),
        "valset": load_sum_valset(),
        "metric": summarization_metric
    }
}

multi_optimizer = MultiTaskMonteCarloOptimizer(tasks)
results = multi_optimizer.optimize_jointly()
```

## Evaluation and Analysis

### Performance Comparison

```python
def compare_monte_carlo_methods(task_data):
    """Compare different Monte Carlo optimization methods."""

    methods = ["random_search", "simulated_annealing", "cross_entropy", "particle_swarm"]
    results = {}

    for method in methods:
        print(f"\n=== Testing {method} ===")

        optimizer = MonteCarloPromptOptimizer(
            task_signature=task_data["signature"],
            trainset=task_data["trainset"],
            valset=task_data["valset"],
            metric_fn=task_data["metric"],
            optimizer_type=method,
            max_iterations=300
        )

        start_time = time.time()
        result = optimizer.optimize()
        end_time = time.time()

        results[method] = {
            "score": result["score"],
            "time": end_time - start_time,
            "config": result["config"],
            "history": result["history"]
        }

    # Analysis
    print("\n=== Performance Comparison ===")
    for method, result in results.items():
        print(f"\n{method}:")
        print(f"  Final score: {result['score']:.4f}")
        print(f"  Time taken: {result['time']:.2f}s")
        print(f"  Convergence iteration: {len(result['history']['iterations'])}")
        print(f"  Efficiency: {result['score'] / result['time']:.6f}")

    return results
```

## Best Practices

### Choosing the Right Monte Carlo Method

1. **Random Search**: Simple problems, initial exploration
2. **Simulated Annealing**: Medium complexity, rugged landscapes
3. **Cross-Entropy**: High-dimensional spaces, categorical variables
4. **Particle Swarm**: Continuous optimization, multiple optima

### Configuration Tips

```python
# For exploration-heavy optimization
exploration_config = {
    "max_iterations": 1000,
    "exploration_rate": 0.5,
    "temperature": 2.0
}

# For exploitation-heavy optimization
exploitation_config = {
    "max_iterations": 500,
    "exploration_rate": 0.1,
    "temperature": 0.5
}

# For balanced optimization
balanced_config = {
    "max_iterations": 750,
    "exploration_rate": 0.3,
    "temperature": 1.0
}
```

### Common Challenges

1. **Curse of Dimensionality**: Search space grows exponentially
2. **Noisy Evaluation**: Model output variability
3. **Computational Cost**: Many evaluations required
4. **Local Optima**: Getting stuck in suboptimal regions

## Summary

Monte Carlo optimization provides a flexible and powerful framework for prompt and parameter optimization in DSPy. By leveraging stochastic search techniques, we can navigate complex, non-convex optimization spaces that are intractable for traditional gradient-based methods. The variety of Monte Carlo methods allows us to choose the most appropriate approach for each specific optimization problem.

### Key Takeaways

1. Monte Carlo methods work with any black-box evaluation function
2. Different methods suit different problem characteristics
3. Proper search space definition is crucial for success
4. Balance between exploration and exploitation is key
5. Multi-task optimization can leverage shared knowledge

## Next Steps

In the next section, we'll explore Bayesian optimization methods, which provide a more principled approach to balancing exploration and exploitation using probabilistic models.
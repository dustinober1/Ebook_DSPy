# Instruction Tuning Frameworks

## Learning Objectives

By the end of this section, you will be able to:
- Understand the principles of instruction tuning for language models
- Implement various instruction tuning methodologies
- Design effective instruction templates and formats
- Evaluate and compare instruction tuning approaches
- Apply best practices for instruction optimization in DSPy

## Introduction

Instruction tuning has emerged as a powerful paradigm for improving language model performance by training models to follow natural language instructions. Unlike traditional fine-tuning that focuses on input-output pairs, instruction tuning emphasizes learning from task descriptions, making models more versatile and better at following complex instructions.

In DSPy, instruction tuning goes beyond model weight optimization to include prompt instruction optimization, where we automatically discover and refine the instructions that guide each module in a multi-stage program.

## Foundations of Instruction Tuning

### What is Instruction Tuning?

Instruction tuning is the process of training language models on datasets where each example includes:
1. **Task instruction**: Natural language description of what to do
2. **Input**: The specific input to process
3. **Output**: The desired output

```
Example Format:
Instruction: "Translate the following English text to French, preserving the original tone and style."
Input: "Hello, how are you today?"
Output: "Bonjour, comment allez-vous aujourd'hui?"
```

### Key Principles

1. **Generalization through Instructions**: Models learn to generalize from instructions rather than memorizing patterns
2. **Zero-shot Capability**: Well-tuned models can perform new tasks without examples
3. **Multi-task Learning**: Simultaneous training on diverse tasks improves overall capabilities
4. **Instruction Following**: Emphasizes understanding and executing natural language commands

### Mathematical Framework

Given a dataset D = {(I_i, x_i, y_i)} where I_i is the instruction, the objective is:

```
θ* = argmax_θ Σ_i log P_θ(y_i | x_i, I_i)
```

Where the model learns to condition its generation on both input and instruction.

## Instruction Tuning Methodologies

### 1. Supervised Instruction Fine-tuning

The most straightforward approach using supervised learning on instruction datasets.

```python
class SupervisedInstructionTuner:
    """Supervised instruction fine-tuning framework."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    def prepare_training_data(self, instruction_dataset):
        """Format data for instruction tuning."""

        formatted_data = []
        for example in instruction_dataset:
            # Format: [INSTRUCTION] Input [OUTPUT] Target
            formatted_text = (
                f"[INSTRUCTION] {example['instruction']}\n"
                f"[INPUT] {example['input']}\n"
                f"[OUTPUT]"
            )

            # Tokenize with labels
            inputs = self.tokenizer(
                formatted_text,
                example['output'],
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            )

            # Set labels for output tokens only
            labels = inputs.input_ids.clone()
            instruction_end = (inputs.input_ids == self.tokenizer.convert_tokens_to_ids("[OUTPUT]")).nonzero(as_tuple=True)[1][0] + 1
            labels[:, :instruction_end] = -100

            formatted_data.append({
                'input_ids': inputs.input_ids,
                'attention_mask': inputs.attention_mask,
                'labels': labels
            })

        return formatted_data

    def train_epoch(self, dataloader):
        """Train for one epoch."""

        total_loss = 0
        self.model.train()

        for batch in dataloader:
            self.optimizer.zero_grad()

            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )

            loss = outputs.loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)
```

### 2. Reinforcement Learning from Human Feedback (RLHF)

Incorporate human preferences to improve instruction following.

```python
class InstructionRLHF:
    """RLHF for instruction tuning."""

    def __init__(self, model, reward_model):
        self.model = model
        self.reward_model = reward_model
        self.ppo_optimizer = PPOOptimizer(model)

    def generate_responses(self, instruction, inputs, num_responses=4):
        """Generate multiple responses for comparison."""

        responses = []
        for input_text in inputs:
            formatted_prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"

            # Sample diverse responses
            for _ in range(num_responses):
                response = self.model.generate(
                    formatted_prompt,
                    max_length=200,
                    temperature=0.7,
                    do_sample=True,
                    num_beams=1
                )
                responses.append(response)

        return responses

    def compute_rewards(self, instruction, inputs, responses):
        """Compute rewards using human feedback model."""

        rewards = []
        for input_text, response in zip(inputs, responses):
            # Use reward model to score instruction following
            reward = self.reward_model.score(
                instruction=instruction,
                input=input_text,
                response=response
            )
            rewards.append(reward)

        return torch.tensor(rewards)

    def optimize_with_ppo(self, instruction, examples):
        """Optimize using PPO with reward feedback."""

        # Generate responses
        responses = self.generate_responses(instruction, examples)

        # Compute rewards
        rewards = self.compute_rewards(instruction, examples, responses)

        # Update policy using PPO
        for input_text, response, reward in zip(examples, responses, rewards):
            self.ppo_optimizer.step(
                state=f"{instruction}\n{input_text}",
                action=response,
                reward=reward
            )
```

### 3. Meta-Learning for Instruction Adaptation

Learn to quickly adapt to new instructions.

```python
class MetaInstructionLearner:
    """Meta-learning framework for rapid instruction adaptation."""

    def __init__(self, model, inner_lr=0.01, outer_lr=1e-4):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)

    def inner_update(self, model, support_set, instruction):
        """Fast adaptation to new instruction."""

        # Create task-specific model copy
        adapted_model = copy.deepcopy(model)
        adapted_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)

        # Few-shot adaptation
        for example in support_set:
            formatted_input = self.format_instruction(
                instruction, example['input']
            )

            outputs = adapted_model(
                formatted_input,
                labels=example['output']
            )

            loss = outputs.loss
            loss.backward()
            adapted_optimizer.step()
            adapted_optimizer.zero_grad()

        return adapted_model

    def meta_update(self, batch_tasks):
        """Meta-optimization across multiple tasks."""

        meta_loss = 0

        for task in batch_tasks:
            # Split into support and query sets
            support_set = task['examples'][:5]
            query_set = task['examples'][5:]

            # Inner adaptation
            adapted_model = self.inner_update(
                self.model, support_set, task['instruction']
            )

            # Compute meta-loss on query set
            for example in query_set:
                formatted_input = self.format_instruction(
                    task['instruction'], example['input']
                )

                outputs = adapted_model(formatted_input)
                loss = F.cross_entropy(outputs.logits, example['output'])
                meta_loss += loss

        # Meta-gradient step
        meta_loss.backward()
        self.meta_optimizer.step()
        self.meta_optimizer.zero_grad()
```

## Instruction Template Design

### Template Components

Effective instruction templates include:

1. **Clear Task Description**: What the model should do
2. **Input Format Specification**: How inputs will be presented
3. **Output Format Specification**: Expected output format
4. **Constraints and Guidelines**: Rules and limitations
5. **Examples**: Few-shot demonstrations (optional)

### Template Examples

#### Basic Template
```
You are a helpful AI assistant. Your task is to {task_description}.

Input format: {input_format}
Output format: {output_format}

Guidelines:
{guidelines}

Example:
{example}

Now, please process the following:
{input}
```

#### Advanced Template with Constraints
```
Role: {role}
Task: {task}
Context: {context}

Input Specifications:
- Type: {input_type}
- Format: {input_format}
- Constraints: {input_constraints}

Output Requirements:
- Format: {output_format}
- Length: {output_length}
- Style: {output_style}
- Must include: {required_elements}

Processing Steps:
1. {step_1}
2. {step_2}
3. {step_3}

Constraints:
- {constraint_1}
- {constraint_2}
- {constraint_3}

Input:
{input}

Output:
```

### Dynamic Template Generation

```python
class InstructionTemplateGenerator:
    """Generate optimized instruction templates."""

    def __init__(self, llm):
        self.llm = llm
        self.template_components = {
            'openings': [
                "You are an expert at...",
                "As a professional...",
                "Your task is to...",
                "Please help me..."
            ],
            'constraints': [
                "Be concise and clear.",
                "Provide detailed explanations.",
                "Use formal language.",
                "Include specific examples."
            ],
            'formats': [
                "Output in JSON format.",
                "Provide a bulleted list.",
                "Write in paragraph form.",
                "Use markdown formatting."
            ]
        }

    def generate_template(self, task_description, examples=None):
        """Generate task-specific instruction template."""

        prompt = f"""
        Generate an effective instruction template for the following task:

        Task: {task_description}

        Examples of desired behavior:
        {examples if examples else "No examples provided"}

        The template should:
        1. Clearly specify the task
        2. Define input/output formats
        3. Include relevant constraints
        4. Guide the model toward desired behavior
        """

        template = self.llm.generate(
            prompt,
            temperature=0.3,
            max_tokens=500
        )

        return self._validate_and_refine_template(template)

    def _validate_and_refine_template(self, template):
        """Validate and refine generated template."""

        # Check for essential components
        required_components = ['task', 'input', 'output']
        missing = [c for c in required_components if c not in template.lower()]

        if missing:
            # Add missing components
            for component in missing:
                template += f"\n\nPlease specify the {component} clearly."

        return template
```

## Automatic Instruction Optimization

### Gradient-based Instruction Optimization

For models that support gradient computation through prompts:

```python
class GradientInstructionOptimizer:
    """Optimize instructions using gradients."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.instruction_embeddings = nn.Embedding(1000, model.config.hidden_size)

    def optimize_instruction(self, initial_instruction, trainset, num_iterations=100):
        """Optimize instruction using gradient descent."""

        # Tokenize initial instruction
        instruction_tokens = self.tokenizer.tokenize(initial_instruction)
        instruction_ids = self.tokenizer.convert_tokens_to_ids(instruction_tokens)

        # Initialize instruction embeddings
        instruction_embeds = self.instruction_embeddings(
            torch.tensor(instruction_ids)
        ).detach().clone()
        instruction_embeds.requires_grad = True

        optimizer = torch.optim.Adam([instruction_embeds], lr=0.01)

        for iteration in range(num_iterations):
            total_loss = 0

            for example in trainset:
                # Combine instruction and input
                input_ids = example['input_ids']
                combined_ids = torch.cat([instruction_ids, input_ids])

                # Forward pass with learnable instruction
                outputs = self.model(
                    input_ids=combined_ids,
                    labels=example['labels']
                )

                loss = outputs.loss
                total_loss += loss

            # Backward pass
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Loss: {total_loss.item()}")

        # Convert optimized embeddings back to text
        optimized_instruction = self._embeddings_to_text(instruction_embeds)
        return optimized_instruction
```

### Evolutionary Instruction Optimization

```python
class EvolutionaryInstructionOptimizer:
    """Evolutionary algorithm for instruction optimization."""

    def __init__(self, llm, population_size=20):
        self.llm = llm
        self.population_size = population_size
        self.mutation_rate = 0.3
        self.crossover_rate = 0.7

    def optimize(self, task_description, examples, generations=50):
        """Evolve optimal instructions."""

        # Initialize population
        population = self._initialize_population(task_description)

        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for instruction in population:
                score = self._evaluate_instruction(
                    instruction, task_description, examples
                )
                fitness_scores.append(score)

            # Select parents
            parents = self._select_parents(population, fitness_scores)

            # Create offspring
            offspring = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    # Crossover
                    if random.random() < self.crossover_rate:
                        child1, child2 = self._crossover(
                            parents[i], parents[i+1]
                        )
                    else:
                        child1, child2 = parents[i], parents[i+1]

                    # Mutation
                    child1 = self._mutate(child1)
                    child2 = self._mutate(child2)

                    offspring.extend([child1, child2])

            # Replace population
            population = self._replace_population(
                population, offspring, fitness_scores
            )

        # Return best instruction
        final_scores = [
            self._evaluate_instruction(i, task_description, examples)
            for i in population
        ]
        best_idx = np.argmax(final_scores)
        return population[best_idx]

    def _evaluate_instruction(self, instruction, task, examples):
        """Evaluate instruction quality."""

        total_score = 0
        for example in examples:
            # Test instruction on example
            prompt = f"{instruction}\n\n{example['input']}"
            response = self.llm.generate(prompt, temperature=0.1)

            # Score response
            score = self._score_response(response, example['output'])
            total_score += score

        return total_score / len(examples)

    def _crossover(self, instruction1, instruction2):
        """Combine two instructions."""

        # Split instructions into sentences
        sentences1 = instruction1.split('. ')
        sentences2 = instruction2.split('. ')

        # Create offspring by mixing sentences
        crossover_point = random.randint(1, min(len(sentences1), len(sentences2)) - 1)

        child1 = '. '.join(sentences1[:crossover_point] + sentences2[crossover_point:])
        child2 = '. '.join(sentences2[:crossover_point] + sentences1[crossover_point:])

        return child1, child2

    def _mutate(self, instruction):
        """Apply mutation to instruction."""

        if random.random() < self.mutation_rate:
            # Prompt LLM to suggest improvements
            mutation_prompt = f"""
            Improve this instruction for better task performance:

            Original instruction: {instruction}

            Keep the core task but improve clarity, add helpful constraints,
            or enhance formatting. Make it more effective.
            """

            mutated = self.llm.generate(mutation_prompt, temperature=0.5)
            return mutated

        return instruction
```

## Evaluation Strategies

### Comprehensive Metrics

1. **Task Performance**: Accuracy, F1, BLEU, etc. on target task
2. **Instruction Following**: How well model follows format and constraints
3. **Generalization**: Performance on unseen instructions
4. **Efficiency**: Inference time and computational cost

### Evaluation Framework

```python
class InstructionEvaluationSuite:
    """Comprehensive instruction evaluation."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.evaluators = {
            'accuracy': AccuracyEvaluator(),
            'instruction_following': InstructionFollowingEvaluator(),
            'fluency': FluencyEvaluator(),
            'consistency': ConsistencyEvaluator()
        }

    def evaluate_instruction(self, instruction, testset):
        """Evaluate instruction across multiple metrics."""

        results = {}

        # Generate responses
        responses = []
        for example in testset:
            prompt = f"{instruction}\n\n{example['input']}"
            response = self.model.generate(prompt)
            responses.append(response)

        # Evaluate each metric
        for metric_name, evaluator in self.evaluators.items():
            scores = []
            for response, example in zip(responses, testset):
                score = evaluator.evaluate(
                    instruction=instruction,
                    input=example['input'],
                    response=response,
                    target=example['output']
                )
                scores.append(score)

            results[metric_name] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            }

        # Compute overall score
        results['overall'] = self._compute_overall_score(results)

        return results

    def compare_instructions(self, instructions, testset):
        """Compare multiple instructions."""

        comparison_results = {}

        for instruction in instructions:
            comparison_results[instruction] = self.evaluate_instruction(
                instruction, testset
            )

        # Statistical significance testing
        comparison_results['significance'] = self._statistical_test(
            comparison_results
        )

        return comparison_results
```

## Best Practices

### Instruction Design Principles

1. **Clarity Over Brevity**: Clear, explicit instructions perform better than concise ones
2. **Specify Format**: Clearly define expected output format
3. **Provide Context**: Include relevant background information
4. **Set Constraints**: Define boundaries and limitations
5. **Include Examples**: Use few-shot examples when helpful

### Common Pitfalls to Avoid

1. **Overly Complex Instructions**: Can confuse the model
2. **Contradictory Requirements**: Leads to inconsistent outputs
3. **Missing Format Specifications**: Results in unpredictable formats
4. **Ambiguous Language**: Causes misinterpretation
5. **Too Many Constraints**: May restrict creativity excessively

### Optimization Tips

1. **Iterative Refinement**: Start simple and add complexity gradually
2. **A/B Testing**: Compare variants systematically
3. **Domain Adaptation**: Tailor instructions to specific domains
4. **Multi-modal Support**: Include visual or structured examples
5. **Version Control**: Track instruction changes and performance

## Integration with DSPy

### DSPy Instruction Tuning Pipeline

```python
import dspy
from dspy.teleprompter import Teleprompter

class DSPyInstructionTuner:
    """DSPy-specific instruction tuning."""

    def __init__(self, model_name="gpt-3.5-turbo"):
        self.lm = dspy.LM(model=model_name)
        dspy.settings.lm = self.lm

    def tune_module_instruction(
        self,
        module_class,
        signature,
        trainset,
        num_candidates=20
    ):
        """Tune instructions for a DSPy module."""

        # Generate instruction candidates
        candidates = self._generate_instruction_candidates(
            module_class, signature, trainset, num_candidates
        )

        # Evaluate candidates
        best_instruction = None
        best_score = -float('inf')

        for instruction in candidates:
            # Create module with instruction
            module = module_class(signature)
            module.set_instruction(instruction)

            # Evaluate on validation set
            score = self._evaluate_module(module, trainset)

            if score > best_score:
                best_score = score
                best_instruction = instruction

        return best_instruction, best_score

    def optimize_multistage_pipeline(self, pipeline, trainset):
        """Optimize instructions for entire pipeline."""

        optimized_instructions = {}

        for stage_name, stage_module in pipeline.stages.items():
            print(f"Optimizing stage: {stage_name}")

            # Get stage-specific training data
            stage_data = self._extract_stage_data(
                stage_name, pipeline, trainset
            )

            # Optimize instruction
            instruction, score = self.tune_module_instruction(
                stage_module.__class__,
                stage_module.signature,
                stage_data
            )

            optimized_instructions[stage_name] = {
                'instruction': instruction,
                'score': score
            }

        return optimized_instructions
```

## Summary

Instruction tuning frameworks provide powerful methods for improving language model performance through better instruction design and optimization. Key takeaways:

1. **Multiple Approaches**: Supervised, RLHF, and meta-learning each offer unique advantages
2. **Template Design**: Well-structured templates significantly impact performance
3. **Automatic Optimization**: Evolutionary and gradient-based methods can discover optimal instructions
4. **Comprehensive Evaluation**: Multi-faceted evaluation ensures robust instruction selection
5. **DSPy Integration**: Seamless integration with DSPy enables end-to-end optimization

The next section will explore demonstration optimization strategies, complementing instruction tuning to create fully optimized multi-stage programs.
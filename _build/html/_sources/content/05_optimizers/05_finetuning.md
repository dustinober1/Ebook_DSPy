# Fine-Tuning Small Language Models in DSPy

## Introduction

While prompt optimization and few-shot learning work well with large language models (LLMs), sometimes you need better performance from smaller models. Fine-tuning adapts small language models to your specific task, achieving competitive performance with lower computational costs.

## When to Use Fine-Tuning

### Ideal Scenarios
- **Domain-Specific Tasks**: Medical, legal, or technical domains
- **High Volume**: Large-scale applications where inference cost matters
- **Latency Critical**: Real-time applications requiring fast responses
- **Privacy Concerns**: On-premises deployment without external APIs
- **Consistent Performance**: Need for stable, reproducible outputs

### Model Size Trade-offs

| Model Size | Parameters | Use Case | Pros | Cons |
|------------|------------|----------|------|------|
| < 1B | < 1B | Simple classification, basic QA | Fast, cheap | Limited capabilities |
| 1-7B | 1-7B | Most tasks, good balance | Capable, efficient | Still needs optimization |
| 7-13B | 7-13B | Complex reasoning | Powerful, smaller | More resources needed |
| > 13B | > 13B | Specialized tasks | High quality | Expensive to fine-tune |

## Setting Up Fine-Tuning

### Prerequisites

```python
# Install required packages
!pip install torch transformers datasets accelerate peft bitsandbytes

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import Dataset
import dspy
```

### Model Selection

```python
# Popular small models for fine-tuning
MODELS = {
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "phi-2": "microsoft/phi-2",
    "qwen-7b": "Qwen/Qwen-7B",
    "gemma-7b": "google/gemma-7b"
}

def load_model(model_name, use_4bit=True):
    """Load a model for fine-tuning."""
    model_id = MODELS[model_name]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=use_4bit,  # QLoRA support
        trust_remote_code=True
    )

    return model, tokenizer
```

## QLoRA: Parameter-Efficient Fine-Tuning

QLoRA (Quantized Low-Rank Adaptation) is a memory-efficient fine-tuning method that works with 4-bit quantized models.

### QLoRA Configuration

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def setup_qlora(model, target_modules=None):
    """Set up QLoRA for parameter-efficient fine-tuning."""
    # Default target modules for common architectures
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",     # MLP
            "lm_head"                                 # Output
        ]

    # LoRA configuration
    lora_config = LoraConfig(
        r=16,                 # Rank
        lora_alpha=32,        # Alpha
        target_modules=target_modules,
        lora_dropout=0.05,    # Dropout
        bias="none",          # No bias adaptation
        task_type="CAUSAL_LM" # Causal language modeling
    )

    # Prepare model for 4-bit training
    model = prepare_model_for_kbit_training(model)

    # Add LoRA adapters
    peft_model = get_peft_model(model, lora_config)

    # Print trainable parameters
    peft_model.print_trainable_parameters()

    return peft_model
```

### Data Preparation

```python
def prepare_training_data(examples, tokenizer, max_length=512):
    """Prepare DSPy examples for fine-tuning."""
    training_data = []

    for example in examples:
        # Format as chat or instruction-following
        if hasattr(example, 'question') and hasattr(example, 'answer'):
            # QA format
            prompt = f"Question: {example.question}\nAnswer: {example.answer}"
        elif hasattr(example, 'context') and hasattr(example, 'response'):
            # Instruction format
            prompt = f"Context: {example.context}\n\nResponse: {example.response}"
        else:
            # Generic format
            prompt = str(example)

        # Tokenize
        tokenized = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

        training_data.append({
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": tokenized["input_ids"].squeeze().clone()  # Labels = input_ids
        })

    return Dataset.from_list(training_data)

# Example: Prepare QA data
qa_examples = [
    dspy.Example(
        question="What is machine learning?",
        answer="Machine learning is a field of AI where computers learn from data."
    ),
    # ... more examples
]

model, tokenizer = load_model("mistral-7b")
training_data = prepare_training_data(qa_examples, tokenizer)
```

## Fine-Tuning Process

### Training Configuration

```python
from transformers import Trainer

def fine_tune_model(model, training_data, val_data=None):
    """Fine-tune the model with QLoRA."""
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        optim="paged_adamw_32bit",  # Memory efficient optimizer
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps" if val_data else "no",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none"  # Disable wandb/tensorboard
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=training_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=lambda data: {
            'input_ids': torch.stack([item['input_ids'] for item in data]),
            'attention_mask': torch.stack([item['attention_mask'] for item in data]),
            'labels': torch.stack([item['labels'] for item in data])
        }
    )

    # Start training
    trainer.train()

    return trainer.model
```

### Integration with DSPy

```python
class FineTunedLLM(dspy.LM):
    """Wrapper for fine-tuned models in DSPy."""

    def __init__(self, model, tokenizer, temperature=0.7):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate(self, prompt, **kwargs):
        """Generate text using the fine-tuned model."""
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # Remove input prompt from output
        if prompt in generated_text:
            generated_text = generated_text.replace(prompt, "").strip()

        return generated_text

    def __call__(self, prompt, **kwargs):
        return [self.generate(prompt, **kwargs)]

# Use in DSPy
fine_tuned_model = FineTunedLLM(model, tokenizer)
dspy.settings.configure(lm=fine_tuned_model)
```

## Task-Specific Fine-Tuning

### Classification Fine-Tuning

```python
def prepare_classification_data(examples, tokenizer, labels):
    """Prepare data for classification tasks."""
    training_data = []

    for example in examples:
        # Format as classification prompt
        prompt = f"""Classify the following text into one of: {', '.join(labels)}

Text: {example.text}

Classification:"""

        # Tokenize
        tokenized = tokenizer(
            prompt + " " + example.label,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        # Create labels
        labels_text = tokenizer.decode(
            tokenized["input_ids"].squeeze(),
            skip_special_tokens=True
        )

        training_data.append({
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": tokenized["input_ids"].squeeze().clone()
        })

    return Dataset.from_list(training_data)

# Example usage
sentiment_examples = [
    dspy.Example(text="I love this!", label="positive"),
    dspy.Example(text="This is bad.", label="negative"),
    # ... more examples
]

sentiment_labels = ["positive", "negative", "neutral"]
sentiment_data = prepare_classification_data(
    sentiment_examples,
    tokenizer,
    sentiment_labels
)
```

### RAG Fine-Tuning

```python
def prepare_rag_data(examples, tokenizer):
    """Prepare data for Retrieval-Augmented Generation."""
    training_data = []

    for example in examples:
        # Format as RAG prompt
        prompt = f"""Context: {example.context}

Question: {example.question}

Answer:"""

        # Tokenize
        tokenized = tokenizer(
            prompt + " " + example.answer,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        training_data.append({
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": tokenized["input_ids"].squeeze().clone()
        })

    return Dataset.from_list(training_data)

class RAGFineTuner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def fine_tune_rag(self, examples):
        """Fine-tune model for RAG tasks."""
        # Prepare data
        training_data = prepare_rag_data(examples, tokenizer)

        # Fine-tune with specific settings for RAG
        training_args = TrainingArguments(
            output_dir="./rag_results",
            num_train_epochs=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=1e-4,  # Lower learning rate for RAG
            warmup_ratio=0.1,
            fp16=True,
            logging_steps=10,
            save_steps=50
        )

        trainer = Trainer(
            model=self.model,
            train_dataset=training_data,
            args=training_args
        )

        trainer.train()
        return trainer.model
```

## Evaluation and Testing

### Fine-Tuned Model Evaluation

```python
def evaluate_fine_tuned_model(model, tokenizer, test_examples):
    """Evaluate fine-tuned model performance."""
    correct = 0
    total = 0
    predictions = []

    model.eval()
    fine_tuned_lm = FineTunedLLM(model, tokenizer, temperature=0)

    for example in test_examples:
        # Generate prediction
        if hasattr(example, 'question'):
            prompt = f"Question: {example.question}\nAnswer:"
        elif hasattr(example, 'text'):
            prompt = f"Text: {example.text}\nClassification:"
        else:
            prompt = str(example)

        with torch.no_grad():
            prediction = fine_tuned_lm.generate(prompt)
            predictions.append((example, prediction))

        # Evaluate (adjust based on task)
        if hasattr(example, 'answer'):
            # QA evaluation
            if example.answer.lower() in prediction.lower():
                correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy, predictions

# Evaluate
accuracy, predictions = evaluate_fine_tuned_model(
    model,
    tokenizer,
    test_examples
)
print(f"Fine-tuned model accuracy: {accuracy:.2%}")
```

### Comparison with Baseline

```python
def compare_models(fine_tuned_model, baseline_lm, test_examples):
    """Compare fine-tuned model with baseline."""
    fine_tuned_lm = FineTunedLLM(fine_tuned_model, tokenizer, temperature=0)

    results = {
        "fine_tuned": [],
        "baseline": []
    }

    for example in test_examples:
        prompt = f"Question: {example.question}\nAnswer:"

        # Fine-tuned prediction
        ft_pred = fine_tuned_lm.generate(prompt)
        results["fine_tuned"].append((example, ft_pred))

        # Baseline prediction
        base_pred = baseline_lm.generate(prompt)[0]
        results["baseline"].append((example, base_pred))

    # Calculate metrics
    ft_correct = sum(1 for ex, pred in results["fine_tuned"]
                    if ex.answer.lower() in pred.lower())
    base_correct = sum(1 for ex, pred in results["baseline"]
                      if ex.answer.lower() in pred.lower())

    ft_acc = ft_correct / len(test_examples)
    base_acc = base_correct / len(test_examples)

    print(f"Fine-tuned accuracy: {ft_acc:.2%}")
    print(f"Baseline accuracy: {base_acc:.2%}")
    print(f"Improvement: {ft_acc - base_acc:.2%}")

    return results
```

## Best Practices

### 1. Data Quality Over Quantity

```python
def filter_high_quality_examples(examples, min_length=10, max_length=500):
    """Filter for high-quality training examples."""
    filtered = []

    for example in examples:
        text = str(example)
        if min_length <= len(text) <= max_length:
            # Additional quality checks
            if not has_repetitions(text) and not has_issues(text):
                filtered.append(example)

    return filtered

def has_repetitions(text):
    """Check for excessive repetitions."""
    words = text.lower().split()
    for i in range(len(words) - 2):
        if words[i] == words[i+1] == words[i+2]:
            return True
    return False
```

### 2. Balanced Training Set

```python
def create_balanced_dataset(examples, field_name):
    """Create a balanced dataset by field."""
    from collections import defaultdict

    # Group by field
    groups = defaultdict(list)
    for example in examples:
        value = getattr(example, field_name, 'unknown')
        groups[value].append(example)

    # Find minimum group size
    min_size = min(len(group) for group in groups.values())

    # Sample from each group
    balanced = []
    for group in groups.values():
        import random
        balanced.extend(random.sample(group, min(min_size, len(group))))

    return balanced
```

### 3. Learning Rate Scheduling

```python
from transformers import get_cosine_schedule_with_warmup

def create_lr_scheduler(optimizer, num_training_steps, warmup_ratio=0.1):
    """Create a learning rate scheduler."""
    warmup_steps = int(num_training_steps * warmup_ratio)
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
```

### 4. Gradient Clipping

```python
training_args = TrainingArguments(
    # ... other args
    max_grad_norm=1.0,  # Prevent gradient explosion
    # ...
)
```

## Common Pitfalls and Solutions

### Pitfall 1: Catastrophic Forgetting

```python
# Problem: Model forgets original capabilities
# Solution: Include diverse examples
def create_mixed_dataset(domain_examples, general_examples, ratio=0.8):
    """Mix domain-specific with general examples."""
    domain_size = int(len(domain_examples) * ratio)
    mixed = domain_examples[:domain_size]
    mixed.extend(general_examples[:len(mixed) - domain_size])
    return mixed
```

### Pitfall 2: Overfitting

```python
# Problem: Model memorizes training data
# Solution: Early stopping and regularization
training_args = TrainingArguments(
    # ... other args
    evaluation_strategy="steps",
    eval_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    weight_decay=0.01,  # L2 regularization
    # ...
)
```

### Pitfall 3: Memory Issues

```python
# Problem: GPU memory overflow
# Solution: Gradient accumulation and mixed precision
training_args = TrainingArguments(
    # ... other args
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # Effective batch size = 16
    fp16=True,  # Mixed precision
    dataloader_pin_memory=False,
    # ...
)
```

## Combined Optimization: Fine-Tuning + Prompt Optimization

One of the most powerful techniques in DSPy is combining fine-tuning with prompt optimization. Research shows that these approaches are complementary, with combined optimization achieving 2-26x improvements over baseline performance.

### Why Fine-Tuning and Prompt Optimization Are Complementary

Fine-tuning and prompt optimization target different aspects of model behavior:

| Aspect | Fine-Tuning | Prompt Optimization |
|--------|-------------|---------------------|
| **Target** | Model weights | Instructions and demonstrations |
| **Effect** | Deep task adaptation | Surface-level guidance |
| **Persistence** | Permanent (model changes) | Runtime (prompt changes) |
| **Flexibility** | Fixed after training | Dynamic per query |

When combined, fine-tuning creates a stronger foundation that prompt optimization can build upon:

```python
# The synergistic effect of combined optimization
# Fine-tuning improvement: +15%
# Prompt optimization improvement: +10%
# Combined improvement: +35% (not just 25%!)

# This synergy occurs because:
# 1. Fine-tuned models follow complex instructions better
# 2. Better instruction following enables more sophisticated prompts
# 3. Optimized prompts unlock capabilities learned during fine-tuning
```

### Optimization Order Effects

**Critical insight**: The order of optimization matters significantly.

```python
# RECOMMENDED: Fine-tuning FIRST, then prompt optimization
def optimal_order_optimization(program, trainset, base_model):
    """
    Fine-tune first, then apply prompt optimization.
    This order consistently outperforms the reverse.
    """
    # Step 1: Fine-tune the base model
    finetuned_model = finetune_model(
        base_model,
        trainset,
        epochs=3
    )

    # Step 2: Configure DSPy with fine-tuned model
    dspy.settings.configure(lm=finetuned_model)

    # Step 3: Apply prompt optimization
    optimizer = BootstrapFewShot(
        metric=accuracy_metric,
        max_bootstrapped_demos=8
    )
    compiled_program = optimizer.compile(program, trainset=trainset)

    return compiled_program

# NOT RECOMMENDED: Prompt optimization first
def suboptimal_order(program, trainset, base_model):
    """
    This order yields lower performance.
    Prompt optimizations don't transfer well to fine-tuned models.
    """
    # Prompts optimized for base model
    dspy.settings.configure(lm=base_model)
    optimizer = BootstrapFewShot(metric=accuracy_metric)
    compiled_program = optimizer.compile(program, trainset=trainset)

    # Fine-tuning doesn't preserve prompt-specific behaviors
    finetuned_model = finetune_model(base_model, trainset)

    return compiled_program  # Prompts may no longer be optimal
```

### Performance Improvements with Combined Optimization

Real-world benchmarks demonstrate the power of combined optimization:

| Task | Baseline | Fine-Tune Only | Prompt Only | Combined | Synergy |
|------|----------|----------------|-------------|----------|---------|
| MultiHopQA | 12% | 28% | 20% | 45% | +9% |
| GSM8K | 11% | 32% | 22% | 55% | +12% |
| Classification | 65% | 82% | 78% | 91% | +4% |

The "synergy" column shows improvement beyond simple addition.

### Code Example: Full Combined Optimization Pipeline

```python
import dspy
from dspy.teleprompter import BootstrapFewShot, MIPRO

def combined_optimization_pipeline(
    program,
    trainset,
    valset,
    base_model_name,
    metric
):
    """
    Complete pipeline for combined fine-tuning and prompt optimization.
    """
    # Phase 1: Prepare fine-tuning data
    print("Phase 1: Preparing fine-tuning data...")
    ft_data = prepare_training_data(trainset, tokenizer)

    # Phase 2: Fine-tune the model
    print("Phase 2: Fine-tuning model...")
    model, tokenizer = load_model(base_model_name)
    peft_model = setup_qlora(model)
    finetuned = fine_tune_model(peft_model, ft_data)

    # Phase 3: Create DSPy LM wrapper
    print("Phase 3: Creating DSPy language model...")
    finetuned_lm = FineTunedLLM(finetuned, tokenizer)
    dspy.settings.configure(lm=finetuned_lm)

    # Phase 4: Apply prompt optimization
    print("Phase 4: Optimizing prompts...")
    # Use BootstrapFewShot for quick optimization
    optimizer = BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=8,
        max_labeled_demos=4
    )

    # Or use MIPRO for maximum performance
    # optimizer = MIPRO(metric=metric, auto="medium")

    compiled_program = optimizer.compile(
        program,
        trainset=trainset,
        valset=valset
    )

    # Phase 5: Evaluate
    print("Phase 5: Evaluating...")
    score = evaluate(compiled_program, valset)
    print(f"Final performance: {score:.2%}")

    return compiled_program, finetuned

# Usage
optimized_program, optimized_model = combined_optimization_pipeline(
    program=MyQASystem(),
    trainset=train_examples,
    valset=val_examples,
    base_model_name="mistralai/Mistral-7B-v0.1",
    metric=exact_match_metric
)
```

### Instruction Complexity Scaling

Fine-tuned models can follow significantly more complex instructions than base models:

```python
# Base model: Limited instruction complexity
simple_instruction = "Answer the question."

# Fine-tuned model: Handles complex multi-step instructions
complex_instruction = """
Analyze the question following this process:
1. Identify the core question and any sub-questions
2. Determine what knowledge domains are relevant
3. Consider potential ambiguities or edge cases
4. Synthesize information from multiple sources
5. Provide a clear, well-structured answer
6. Note any assumptions or limitations
"""

def test_instruction_complexity(model, instructions, test_set):
    """Test model's ability to follow complex instructions."""
    results = {}
    for name, instruction in instructions.items():
        # Configure signature with instruction
        signature = dspy.Signature(
            "question -> answer",
            instruction
        )
        predictor = dspy.Predict(signature)

        scores = []
        for example in test_set:
            try:
                pred = predictor(question=example.question)
                scores.append(accuracy_metric(example, pred))
            except:
                scores.append(0)

        results[name] = np.mean(scores)

    return results

# Fine-tuned models show larger gains with complex instructions
```

### Demonstration Efficiency: Fewer Shots Required

Fine-tuned models achieve equivalent performance with fewer demonstrations:

```python
def compare_demonstration_efficiency(base_lm, finetuned_lm, trainset, testset):
    """
    Compare how many demonstrations each model needs.
    Fine-tuned models typically need 3 demos where base needs 8.
    """
    results = {"base": {}, "finetuned": {}}

    for num_demos in [1, 2, 3, 4, 5, 6, 7, 8]:
        # Test base model
        dspy.settings.configure(lm=base_lm)
        optimizer = BootstrapFewShot(
            metric=accuracy_metric,
            max_bootstrapped_demos=num_demos
        )
        compiled_base = optimizer.compile(program, trainset=trainset)
        results["base"][num_demos] = evaluate(compiled_base, testset)

        # Test fine-tuned model
        dspy.settings.configure(lm=finetuned_lm)
        compiled_ft = optimizer.compile(program, trainset=trainset)
        results["finetuned"][num_demos] = evaluate(compiled_ft, testset)

    # Find efficiency ratio
    base_8shot = results["base"][8]
    for num_demos in [1, 2, 3, 4, 5]:
        if results["finetuned"][num_demos] >= base_8shot:
            print(f"Fine-tuned {num_demos}-shot >= Base 8-shot")
            print(f"Demonstration efficiency: {8/num_demos:.1f}x")
            break

    return results
```

### Integration with COPA

For maximum performance, use the COPA optimizer which systematically combines fine-tuning and prompt optimization:

```python
from copa_optimizer import COPAOptimizer  # See 09-copa-optimizer.md

# COPA handles the optimization order automatically
copa = COPAOptimizer(
    base_model_name="mistralai/Mistral-7B-v0.1",
    metric=accuracy_metric,
    finetune_epochs=3,
    prompt_optimizer="mipro"
)

optimized_program, optimized_model = copa.optimize(
    program=MyQASystem(),
    trainset=train_examples,
    valset=val_examples
)

# COPA achieves 2-26x improvements on complex tasks
```

For more details on COPA and advanced joint optimization techniques, see [COPA: Combined Fine-Tuning and Prompt Optimization](09-copa-optimizer.md).

## Key Takeaways

1. Fine-tuning adapts small models for specific tasks efficiently
2. QLoRA enables memory-efficient fine-tuning with 4-bit models
3. Proper data preparation is crucial for success
4. Balance domain-specific and general examples
5. Monitor for overfitting and catastrophic forgetting
6. Use gradient accumulation for larger effective batch sizes
7. **Combined optimization (fine-tuning + prompts) achieves synergistic improvements**
8. **Always fine-tune first, then apply prompt optimization**
9. **Fine-tuned models require fewer demonstrations (3-shot vs 8-shot)**

## Next Steps

In the next section, we'll explore how to choose the right optimizer for your specific needs and compare different approaches. For advanced joint optimization, see [COPA: Combined Fine-Tuning and Prompt Optimization](09-copa-optimizer.md).
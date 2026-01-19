# Case Study 8: Replit Code Repair with DSPy

## Overview

This case study explores how Replit built an AI-powered code repair system using DSPy for synthetic data generation and model training. The system addresses a critical developer need: fixing bugs identified by Language Server Protocol (LSP) diagnostics, where only 10% of errors had automated fixes available.

## Business Challenge

Replit identified several key pain points in their development environment:

1. **Limited LSP Fix Coverage**: Only 10% of LSP diagnostic messages in Python projects had associated fixes
2. **Manual Debugging Burden**: Developers spent significant time fixing common errors
3. **Scale**: Hundreds of millions of LSP diagnostics generated daily
4. **Real-time Requirements**: Need for instantaneous fixes within the IDE

## Technical Architecture

### Data Pipeline with DSPy

```python
import dspy
from dspy import ChainOfThought, Predict

class CodeRepairPipeline(dspy.Module):
    """DSPy pipeline for synthesizing code fixes from LSP diagnostics"""

    def __init__(self):
        super().__init__()
        self.diagnostic_analyzer = ChainOfThought(
            """code_file, error_line, error_message -> error_analysis
            Analyze the error and identify the fix needed.
            """
        )
        self.fix_synthesizer = ChainOfThought(
            """code_file, error_line, error_analysis, fix_description -> line_diff
            Generate a numbered line diff to fix the error.
            Format: {line_number}{operation}{content}
            """
        )
        self.fix_verifier = Predict(
            """original_code, line_diff -> is_valid, verification_result
            Verify if the line diff correctly fixes the error.
            """
        )

    def forward(self, code_file, error_line, error_message):
        # Step 1: Analyze the error
        analysis = self.diagnostic_analyzer(
            code_file=code_file,
            error_line=error_line,
            error_message=error_message
        )

        # Step 2: Synthesize the fix
        fix_description = f"Fix the {analysis.error_type} at line {error_line}"
        line_diff = self.fix_synthesizer(
            code_file=code_file,
            error_line=error_line,
            error_analysis=analysis.error_analysis,
            fix_description=fix_description
        ).line_diff

        # Step 3: Verify the fix
        verification = self.fix_verifier(
            original_code=code_file,
            line_diff=line_diff
        )

        return dspy.Prediction(
            line_diff=line_diff,
            is_valid=verification.is_valid,
            verification_result=verification.verification_result
        )
```

### Data Format and Schema

```python
class CodeRepairExample:
    """Structured format for code repair training examples"""

    def __init__(self, code_content, diagnostics):
        self.file_path = diagnostics.get("file_path", "main.py")
        self.code_with_line_numbers = self._add_line_numbers(code_content)
        self.error_message = diagnostics["message"]
        self.error_line = diagnostics["range"]["start"]["line"]
        self.error_code = diagnostics["code"]

    def _add_line_numbers(self, code):
        """Add line numbers to code for unambiguous diff application"""
        lines = code.split('\n')
        return '\n'.join(
            f"{i+1:4d} {line}" for i, line in enumerate(lines)
        )

    def to_dspy_format(self):
        """Convert to DSPy training format"""
        return dspy.Example(
            code_file=self.code_with_line_numbers,
            error_line=self.error_line,
            error_message=f"{self.error_code}: {self.error_message}"
        ).with_inputs("code_file", "error_line", "error_message")
```

### Synthetic Data Generation

```python
class SyntheticDataGenerator:
    """Generate training data using DSPy-powered synthetic pipeline"""

    def __init__(self, base_model="gpt-4"):
        self.lm = dspy.OpenAI(model=base_model)
        dspy.settings.configure(lm=self.lm)

        self.pipeline = CodeRepairPipeline()

    def generate_fix(self, buggy_code, error_diagnostic):
        """Generate a synthetic fix for a given error"""
        return self.pipeline(
            code_file=buggy_code,
            error_line=error_diagnostic["line"],
            error_message=error_diagnostic["message"]
        )

    def create_training_dataset(self, real_diagnostics, target_size=100000):
        """Create training dataset from real LSP diagnostics"""
        training_data = []

        for diagnostic in real_diagnostics:
            # Skip if already has a CodeAction fix
            if diagnostic.get("codeAction"):
                continue

            # Skip stylistic errors
            if diagnostic["code"] in ["E501", "I001"]:
                continue

            example = CodeRepairExample(
                code_file=diagnostic["code_content"],
                diagnostics=diagnostic
            )

            # Generate synthetic fix
            result = self.generate_fix(
                example.code_with_line_numbers,
                diagnostic
            )

            if result.is_valid:
                example.synthetic_fix = result.line_diff
                training_data.append(example)

        return training_data[:target_size]
```

### Model Training Integration

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class CodeRepairTrainer:
    """Train specialized model for code repair"""

    def __init__(self, model_name="deepseek-coder"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add special tokens for structured format
        special_tokens = ["<code>", "<error>", "<diff>"]
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

    def format_training_example(self, example):
        """Format example using Replit's sentinel tokens"""
        return f"""<code>
{example.code_with_line_numbers}
<error>
Line {example.error_line}: {example.error_message}
<diff>
{example.synthetic_fix}"""

    def train(self, train_data, val_data, epochs=4):
        """Train the model on synthetic data"""
        # Prepare datasets
        train_texts = [self.format_training_example(ex) for ex in train_data]
        val_texts = [self.format_training_example(ex) for ex in val_data]

        # Tokenize
        train_encodings = self.tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=2048,
            return_tensors="pt"
        )

        # Training configuration
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-5,
            weight_decay=0
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=len(train_encodings["input_ids"]),
            eta_min=1e-7
        )

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in self._create_dataloader(train_encodings):
                optimizer.zero_grad()

                outputs = self.model(**batch)
                loss = outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            # Validation
            val_loss = self._validate(val_encodings)
            print(f"Epoch {epoch+1}: Train Loss={total_loss:.4f}, Val Loss={val_loss:.4f}")
```

## Implementation Results

### Performance Benchmarks

| Model | Replit Repair Eval | LeetCode Repair Eval | Parameters |
|-------|-------------------|---------------------|------------|
| Replit Code Repair 7B | 24.3% | 41.2% | 7B |
| GPT-4 Turbo | 25.1% | 56.7% | - |
| Claude-3 Opus | 22.8% | 53.4% | - |
| DeepSeek-Coder Base | 15.2% | 32.1% | 7B |

### Key Findings

1. **Competitive Performance**: 7B model competitive with models 10x larger
2. **Synthetic Data Quality**: Synthetic fixes less noisy than real user fixes
3. **Data Scaling**: Performance improves with more training examples
4. **Parameter Scaling**: Larger models consistently perform better

### Data Scaling Results

```python
import matplotlib.pyplot as plt
import numpy as np

# Results from Replit's scaling experiments
training_sizes = [10_000, 25_000, 50_000, 75_000]
performances = [18.5, 21.2, 23.8, 24.3]

plt.figure(figsize=(10, 6))
plt.plot(training_sizes, performances, 'bo-')
plt.xlabel('Training Examples')
plt.ylabel('Performance (%)')
plt.title('Code Repair Performance vs Training Data Size')
plt.grid(True)
plt.show()
```

## Production Integration

### IDE Integration

```python
class ReplitCodeFixProvider:
    """Integrate code repair model into Replit IDE"""

    def __init__(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def suggest_fix(self, code_content, diagnostic):
        """Suggest fix for LSP diagnostic"""
        # Format input using sentinel tokens
        input_text = f"""<code>
{self._add_line_numbers(code_content)}
<error>
Line {diagnostic['line']}: {diagnostic['message']}
<diff>"""

        # Generate fix
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=2048
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                top_p=0.95,
                do_sample=True
            )

        # Extract and parse the fix
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return self._parse_line_diff(generated_text)

    def _parse_line_diff(self, diff_text):
        """Parse line diff from generated text"""
        # Implementation for parsing line diff format
        # Returns structured fix that IDE can apply
        pass
```

### Real-time Performance

```python
class CodeFixCache:
    """Cache frequently requested fixes for faster response"""

    def __init__(self, max_size=10000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get_fix(self, code_hash, diagnostic):
        """Get cached fix if available"""
        key = f"{code_hash}:{diagnostic['code']}:{diagnostic['line']}"

        if key in self.cache:
            self.hits += 1
            return self.cache[key]

        self.misses += 1
        return None

    def store_fix(self, code_hash, diagnostic, fix):
        """Store fix in cache"""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        key = f"{code_hash}:{diagnostic['code']}:{diagnostic['line']}"
        self.cache[key] = fix

    def get_stats(self):
        """Get cache performance statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache)
        }
```

## Optimization Techniques

### Few-Shot Prompt Optimization

```python
def optimize_with_examples(train_data, test_data):
    """Optimize model performance with few-shot examples"""

    def select_best_examples(diagnostic_code, k=5):
        """Select best examples for a given error type"""
        examples = [ex for ex in train_data if ex.error_code == diagnostic_code]

        # Score examples based on complexity and uniqueness
        scored = []
        for ex in examples:
            score = (
                len(ex.synthetic_fix.split('\n')) * 0.3 +  # Complexity
                len(set(ex.synthetic_fix)) * 0.7        # Uniqueness
            )
            scored.append((ex, score))

        # Select top-k examples
        scored.sort(key=lambda x: x[1], reverse=True)
        return [ex[0] for ex in scored[:k]]

    # Create optimized prompt templates for each error type
    error_types = set(ex.error_code for ex in train_data)
    optimized_prompts = {}

    for error_type in error_types:
        examples = select_best_examples(error_type)
        optimized_prompts[error_type] = format_examples_as_prompt(examples)

    return optimized_prompts
```

### Post-Training with DPO

```python
class DPOOptimizer:
    """Direct Preference Optimization using user feedback"""

    def __init__(self, model, ref_model):
        self.model = model
        self.ref_model = ref_model

    def collect_preferences(self, feedback_data):
        """Collect user preferences from fix acceptance/rejection"""
        preferences = []

        for item in feedback_data:
            if item["user_action"] == "accepted":
                preferences.append({
                    "chosen": item["generated_fix"],
                    "rejected": item["alternative_fix"],
                    "code": item["code"],
                    "diagnostic": item["diagnostic"]
                })

        return preferences

    def optimize_with_dpo(self, preferences, epochs=1):
        """Optimize model using Direct Preference Optimization"""
        # Implementation of DPO training loop
        # Uses collected preferences to improve fix quality

        for epoch in range(epochs):
            for pref in preferences:
                # Get model scores for chosen and rejected
                chosen_score = self.model.score(pref["chosen"], pref["code"])
                rejected_score = self.model.score(pref["rejected"], pref["code"])

                # Calculate DPO loss
                dpo_loss = self._calculate_dpo_loss(
                    chosen_score, rejected_score
                )

                # Backpropagate and update
                dpo_loss.backward()
                self.model.optimizer.step()
                self.model.optimizer.zero_grad()
```

## Business Impact

### Developer Productivity

- **Time Saved**: Average 5 minutes per bug fix
- **Bug Resolution Rate**: Increased from 10% to 35% automated fixes
- **Developer Satisfaction**: 87% of users find suggestions helpful
- **Learning Impact**: Developers learn from suggested fixes

### Technical Metrics

- **Inference Latency**: <500ms for most fixes
- **Cache Hit Rate**: 72% for common error patterns
- **Daily Fixes Suggested**: ~50,000
- **User Acceptance Rate**: 68% of suggestions accepted

## Lessons Learned

### Technical Insights

1. **Synthetic Data Quality**
   - Better than real user fixes (less noise)
   - Requires careful verification pipeline
   - Line diff format reduces hallucinations

2. **Model Architecture**
   - 7B parameter size provides good balance
   - Base model pretraining crucial for success
   - Sentinel tokens improve consistency

3. **Evaluation Strategy**
   - Academic benchmarks (LeetCode) not reflective of real use
   - Need both functional correctness and exact match metrics
   - Real-world evaluation essential

### Best Practices

1. **Data Curation**
   ```python
   def filter_high_quality_examples(examples):
       """Filter for high-quality training examples"""
       filtered = []

       for ex in examples:
           # Must be syntactically valid
           if not is_valid_python(ex.code_with_line_numbers):
               continue

           # Fix must be applicable
           if not can_apply_diff(ex.code, ex.line_diff):
               continue

           # Fix must actually fix the error
           if not verifies_fix(ex.code, ex.line_diff, ex.error):
               continue

           filtered.append(ex)

       return filtered
   ```

2. **Error Handling**
   ```python
   def safe_fix_generation(code, diagnostic, max_attempts=3):
       """Generate fix with fallback strategies"""

       for attempt in range(max_attempts):
           try:
               result = generate_fix(code, diagnostic)

               if validate_fix(result):
                   return result
               elif attempt < max_attempts - 1:
                   # Try with different temperature
                   adjust_generation_parameters(temperature=0.2 + attempt * 0.2)

           except Exception as e:
               log_error(f"Fix generation failed: {e}")
               continue

       # Return safe fallback
       return create_safe_fallback(diagnostic)
   ```

## Future Directions

Replit is expanding their code repair capabilities:

1. **Multi-Language Support**
   - JavaScript, TypeScript, Go, Rust
   - Cross-language transfer learning
   - Language-specific error patterns

2. **Cross-File Fixes**
   - Multi-file refactoring
   - Import statement fixes
   - Type annotation propagation

3. **Advanced Features**
   - Integration with code completion
   - Proactive error prevention
   - Code improvement suggestions

## Conclusion

Replit's code repair system demonstrates how DSPy can be effectively used for:

- **Synthetic data generation** at scale
- **Fine-tuning specialized models** for specific tasks
- **Production deployment** in developer tools
- **Continuous improvement** through user feedback

The success of this project shows that smaller, specialized models can compete with large general-purpose models when properly trained and optimized for specific tasks. The use of DSPy for data generation and optimization was crucial to achieving these results.

## References

- Replit Blog: "Building LLMs for Code Repair" (April 2024)
- DeepSeek-Coder model and documentation
- DSPy GitHub repository and documentation
- Language Server Protocol specification
- MosaicML training infrastructure documentation
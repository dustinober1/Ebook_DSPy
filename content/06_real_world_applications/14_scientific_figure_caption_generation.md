# Scientific Figure Caption Generation with DSPy

## Overview

Scientific figure captioning requires both technical accuracy and stylistic consistency with the author's writing style. This application demonstrates how DSPy can be used to build a sophisticated two-stage pipeline that generates high-quality scientific figure captions by combining contextual understanding with author-specific stylistic adaptation.

## Key Concepts

### Two-Stage Caption Generation Pipeline

1. **Stage 1 - Context-Aware Generation**
   - Context filtering from related text
   - Category-specific prompt optimization
   - Caption candidate selection

2. **Stage 2 - Stylistic Refinement**
   - Few-shot prompting with profile figures
   - Author-specific style adaptation
   - Final refinement and selection

### DSPy Components Used

- **MIPROv2 Optimizer**: For category-specific prompt optimization
- **SIMBA Optimizer**: For stochastic introspective optimization
- **Retrieval Modules**: For context filtering and similarity matching
- **Chain of Thought**: For structured caption generation

## Implementation

### Basic Setup

```python
import dspy
from dspy.datasets import LaMPCap
from dspy.teleprompters import MIPROv2
from dspy.optimize import SIMBA

# Configure the language model
lm = dspy.OpenAI(model="gpt-4", api_key="your-api-key")
dspy.settings.configure(lm=lm)
```

### Stage 1: Context-Aware Caption Generation

```python
class CaptionContext(dspy.Signature):
    """Generate figure-related context from scientific text."""

    figure_text = dspy.InputField(desc="Text describing the figure")
    paper_context = dspy.InputField(desc="Relevant sections from the paper")
    filtered_context = dspy.OutputField(desc="Most relevant context for caption")

class CaptionGenerator(dspy.Signature):
    """Generate a scientific figure caption given context and category."""

    context = dspy.InputField(desc="Relevant context about the figure")
    figure_category = dspy.InputField(desc="Type of figure (graph, diagram, table, etc.)")
    caption = dspy.OutputField(desc="Scientifically accurate caption")
```

### Category-Specific Optimization

```python
def optimize_for_category(training_data, figure_category):
    """Optimize prompts for specific figure categories."""

    # Filter training data by category
    category_data = [
        example for example in training_data
        if example.figure_category == figure_category
    ]

    # Define the base module
    class CategoryCaptionModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.context_filter = dspy.ChainOfThought(CaptionContext)
            self.caption_gen = dspy.Predict(CaptionGenerator)

        def forward(self, figure_text, paper_context, figure_category):
            # Filter relevant context
            filtered = self.context_filter(
                figure_text=figure_text,
                paper_context=paper_context
            ).filtered_context

            # Generate caption
            caption = self.caption_gen(
                context=filtered,
                figure_category=figure_category
            ).caption

            return dspy.Prediction(
                caption=caption,
                filtered_context=filtered
            )

    # Optimize with MIPROv2
    optimizer = MIPROv2(
        metric=evaluate_caption_quality,
        num_candidates=5,
        init_temperature=0.7
    )

    optimized_module = optimizer.compile(
        CategoryCaptionModule(),
        trainset=category_data
    )

    return optimized_module
```

### Stage 2: Author-Specific Stylistic Refinement

```python
class StyleRefiner(dspy.Signature):
    """Refine caption to match author's writing style."""

    original_caption = dspy.InputField(desc="Generated caption")
    author_examples = dspy.InputField(desc="Examples of author's previous captions")
    refined_caption = dspy.OutputField(desc="Stylistically refined caption")

class AuthorStyleModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.refiner = dspy.ChainOfThought(StyleRefiner)

    def forward(self, caption, author_profile):
        refined = self.refiner(
            original_caption=caption,
            author_examples=author_profile.sample_captions
        ).refined_caption

        return dspy.Prediction(refined_caption=refined)
```

### Complete Pipeline

```python
class ScientificCaptionPipeline(dspy.Module):
    def __init__(self, figure_categories):
        super().__init__()
        self.categories = figure_categories
        self.optimizers = {}

        # Initialize category-specific optimizers
        for category in figure_categories:
            self.optimizers[category] = optimize_for_category(
                training_data, category
            )

        self.style_refiner = AuthorStyleModule()

    def forward(self, example):
        figure_category = example.figure_category

        # Stage 1: Generate context-aware caption
        if figure_category in self.optimizers:
            stage1_result = self.optimizers[figure_category](
                figure_text=example.figure_text,
                paper_context=example.paper_context,
                figure_category=figure_category
            )
            initial_caption = stage1_result.caption
        else:
            # Fallback to general prompt
            initial_caption = generate_general_caption(example)

        # Stage 2: Apply author-specific style
        final_result = self.style_refiner(
            caption=initial_caption,
            author_profile=example.author_profile
        )

        return dspy.Prediction(
            initial_caption=initial_caption,
            final_caption=final_result.refined_caption,
            category=figure_category
        )
```

## Evaluation Metrics

### ROUGE and BLEU Scores

```python
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu

def evaluate_caption_quality(example, prediction, trace=None):
    """Evaluate caption quality using ROUGE and BLEU metrics."""

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Calculate ROUGE scores
    rouge_scores = scorer.score(
        example.reference_caption,
        prediction.final_caption
    )

    # Calculate BLEU score
    bleu_score = corpus_bleu(
        [[example.reference_caption.split()]],
        [prediction.final_caption.split()]
    )

    # Combine metrics (weighted average)
    combined_score = (
        0.4 * rouge_scores['rouge1'].recall +
        0.3 * rouge_scores['rouge2'].recall +
        0.3 * bleu_score
    )

    return combined_score
```

### Category-Specific Performance

```python
def evaluate_by_category(testset, pipeline):
    """Evaluate pipeline performance across different figure categories."""

    results = {}

    for category in set(example.figure_category for example in testset):
        category_examples = [
            example for example in testset
            if example.figure_category == category
        ]

        scores = []
        for example in category_examples:
            prediction = pipeline(example)
            score = evaluate_caption_quality(example, prediction)
            scores.append(score)

        results[category] = {
            'mean_score': sum(scores) / len(scores),
            'count': len(scores),
            'examples': scores
        }

    return results
```

## Performance Results

### Quantitative Improvements

The two-stage approach with DSPy optimization demonstrates significant improvements:

- **ROUGE-1 Recall**: +8.3% improvement
- **Precision Loss**: Limited to only -2.8%
- **BLEU-4 Reduction**: Only -10.9% (much better than baseline)
- **Style Consistency**: 40-48% BLEU score improvement
- **Author Fidelity**: 25-27% ROUGE score improvement

### Category-Specific Benefits

Different figure categories show varying levels of improvement:

```python
# Example performance by category
performance_by_category = {
    'bar_graph': {'improvement': 0.123, 'count': 234},
    'line_plot': {'improvement': 0.098, 'count': 189},
    'diagram': {'improvement': 0.145, 'count': 156},
    'table': {'improvement': 0.087, 'count': 203},
    'heatmap': {'improvement': 0.112, 'count': 98}
}
```

## Advanced Features

### Context Filtering with Embeddings

```python
class SemanticContextFilter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def forward(self, figure_text, paper_context):
        # Generate embeddings
        fig_embedding = self.embedding_model.encode(figure_text)
        context_embeddings = self.embedding_model.encode(paper_context)

        # Calculate similarity scores
        similarities = util.cos_sim(fig_embedding, context_embeddings)

        # Select top-k most relevant contexts
        top_k_indices = np.argsort(similarities[0])[-5:][::-1]
        filtered_context = [paper_context[i] for i in top_k_indices]

        return dspy.Prediction(
            filtered_context=filtered_context,
            similarity_scores=similarities[0][top_k_indices]
        )
```

### Multi-Objective Optimization

```python
def multi_objective_optimization(training_data):
    """Optimize for both accuracy and style preservation."""

    def combined_metric(example, prediction, trace=None):
        accuracy_score = evaluate_caption_quality(example, prediction)
        style_score = evaluate_style_consistency(example, prediction)

        # Weighted combination
        return 0.7 * accuracy_score + 0.3 * style_score

    optimizer = SIMBA(
        metric=combined_metric,
        n_iterations=50,
        temperature_range=(0.1, 1.0)
    )

    return optimizer.compile(CaptionGenerator(), trainset=training_data)
```

## Best Practices

### 1. Data Preparation
- Collect representative examples for each figure category
- Build author profiles with style examples
- Ensure diverse caption styles in training data

### 2. Optimization Strategy
- Use category-specific optimization for better results
- Apply SIMBA for fine-tuning after MIPROv2 optimization
- Balance accuracy and style preservation in metrics

### 3. Evaluation Protocol
- Evaluate both quantitative metrics (ROUGE, BLEU)
- Include qualitative assessment of scientific accuracy
- Test across diverse figure categories and author styles

### 4. Deployment Considerations
- Cache embeddings for faster context filtering
- Implement batch processing for multiple figures
- Add fallback mechanisms for edge cases

## Real-World Applications

This approach has been successfully applied to:

1. **Scientific Publishing**
   - Automated caption generation for journals
   - Consistency across multi-author papers
   - Rapid processing of supplement figures

2. **Research Assistance**
   - Help researchers draft captions
   - Maintain consistency in large studies
   - Style adaptation for target venues

3. **Educational Content**
   - Generate captions for teaching materials
   - Adapt complexity level to audience
   - Ensure accessibility compliance

## Conclusion

The DSPy-based two-stage pipeline for scientific figure caption generation demonstrates how:

- Category-specific optimization significantly improves caption quality
- Author-specific style adaptation maintains consistency
- The MIPROv2 and SIMBA optimizers work effectively for this task
- Retrieval-augmented approaches enhance contextual understanding

This system provides a scalable solution for generating scientifically accurate and stylistically consistent figure captions, addressing a critical need in scientific communication and publishing.

## References

- Original paper: "Leveraging Author-Specific Context for Scientific Figure Caption Generation: 3rd SciCap Challenge" (arXiv:2510.07993)
- LaMP-Cap dataset for scientific caption generation
- DSPy documentation for MIPROv2 and SIMBA optimizers
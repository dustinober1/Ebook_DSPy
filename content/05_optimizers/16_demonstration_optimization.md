# Demonstration Optimization Strategies

## Learning Objectives

By the end of this section, you will be able to:
- Understand the role of demonstrations in language model prompting
- Implement various demonstration selection algorithms
- Design utility functions for evaluating demonstration effectiveness
- Apply diversity metrics for optimal demonstration sets
- Integrate demonstration optimization in multi-stage programs

## Introduction

Demonstrations (few-shot examples) are a critical component of prompt engineering, providing concrete examples that guide language models toward desired behavior. However, the selection and optimization of demonstrations is far from trivial - the quality, diversity, and ordering of examples can dramatically affect model performance.

In multi-stage programs, demonstration optimization becomes even more complex as we must consider:
- Stage-specific demonstration needs
- Cross-stage demonstration dependencies
- Limited context window constraints
- Computational efficiency requirements

This section explores comprehensive strategies for optimizing demonstrations in DSPy programs.

## Theoretical Foundations

### Why Demonstrations Matter

Demonstrations serve multiple purposes:

1. **Task Clarification**: Show what the task actually requires
2. **Format Specification**: Demonstrate expected input/output format
3. **Pattern Recognition**: Reveal underlying patterns in the task
4. **Constraint Illustration**: Show how to handle edge cases
5. **Quality Benchmark**: Set standards for response quality

### Mathematical Framework

Given a demonstration set D = {d₁, d₂, ..., d_k} where each demonstration d_i = (x_i, y_i), the objective is to select or generate demonstrations that maximize expected performance:

```
D* = argmax_{|D|=k} E_{(x,y)~D_test}[f(M(Prompt(I, D, x)), y)]
```

Where:
- I = instruction
- Prompt(I, D, x) = formatted prompt with instruction, demonstrations, and input
- M = language model
- f = evaluation metric

### Demonstration Effectiveness Factors

1. **Relevance**: Similarity to target inputs
2. **Diversity**: Coverage of different patterns and cases
3. **Quality**: Correctness and clarity of examples
4. **Difficulty**: Appropriate complexity level
5. **Consistency**: Alignment with instruction

## Demonstration Selection Algorithms

### 1. Similarity-based Selection

Select demonstrations most similar to the input.

```python
class SimilarityBasedSelector:
    """Select demonstrations based on input similarity."""

    def __init__(self, similarity_metric="cosine", encoder="sentence-transformer"):
        self.similarity_metric = similarity_metric
        self.encoder = SentenceTransformer(encoder) if encoder == "sentence-transformer" else None

    def select(self, query, candidates, k=5):
        """Select top-k most similar demonstrations."""

        if self.encoder:
            # Encode all candidates
            query_emb = self.encoder.encode(query)
            candidate_embs = self.encoder.encode(candidates)

            # Compute similarities
            similarities = np.dot(candidate_embs, query_emb)
        else:
            # Use simple text similarity
            similarities = [
                self._text_similarity(query, candidate)
                for candidate in candidates
            ]

        # Get top-k indices
        top_indices = np.argsort(similarities)[-k:][::-1]

        return [candidates[i] for i in top_indices]

    def _text_similarity(self, text1, text2):
        """Simple text similarity using TF-IDF."""

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer().fit([text1, text2])
        vectors = vectorizer.transform([text1, text2])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

        return similarity
```

### 2. Diversity-aware Selection

Select diverse demonstrations that cover different aspects.

```python
class DiversityAwareSelector:
    """Select demonstrations maximizing diversity."""

    def __init__(self, diversity_weight=0.5):
        self.diversity_weight = diversity_weight

    def select(self, candidates, k=5):
        """Select diverse set of demonstrations."""

        selected = []
        remaining = candidates.copy()

        # Greedy selection maximizing diversity
        while len(selected) < k and remaining:
            if not selected:
                # Select first randomly or by quality
                best = max(remaining, key=lambda x: self._quality_score(x))
            else:
                # Select candidate maximizing diversity-weighted score
                best = self._select_best_for_diversity(selected, remaining)

            selected.append(best)
            remaining.remove(best)

        return selected

    def _select_best_for_diversity(self, selected, candidates):
        """Select best candidate for diversity."""

        best_score = -float('inf')
        best_candidate = None

        for candidate in candidates:
            # Compute diversity score
            diversity = self._compute_diversity(selected + [candidate])

            # Combine with quality
            quality = self._quality_score(candidate)
            score = (
                self.diversity_weight * diversity +
                (1 - self.diversity_weight) * quality
            )

            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate

    def _compute_diversity(self, demonstrations):
        """Compute diversity of demonstration set."""

        if len(demonstrations) <= 1:
            return 0

        # Compute pairwise similarities
        similarities = []
        for i in range(len(demonstrations)):
            for j in range(i + 1, len(demonstrations)):
                sim = self._similarity(demonstrations[i], demonstrations[j])
                similarities.append(sim)

        # Diversity = 1 - average similarity
        return 1 - np.mean(similarities)

    def _quality_score(self, demonstration):
        """Score demonstration quality."""

        # Factors to consider:
        # - Correctness
        # - Clarity
        # - Completeness
        # - Relevance

        # Implementation depends on specific requirements
        return 1.0  # Placeholder
```

### 3. Coverage-based Selection

Ensure demonstrations cover different categories or patterns.

```python
class CoverageBasedSelector:
    """Select demonstrations ensuring coverage of patterns."""

    def __init__(self, pattern_extractor=None):
        self.pattern_extractor = pattern_extractor or self._default_pattern_extractor

    def select(self, candidates, k=5):
        """Select demonstrations covering diverse patterns."""

        # Extract patterns from all candidates
        all_patterns = {}
        for i, demo in enumerate(candidates):
            patterns = self.pattern_extractor(demo)
            for pattern in patterns:
                if pattern not in all_patterns:
                    all_patterns[pattern] = []
                all_patterns[pattern].append(i)

        selected = []
        covered_patterns = set()

        # Greedy selection maximizing pattern coverage
        while len(selected) < k:
            best_candidate = None
            best_new_patterns = set()

            for i, demo in enumerate(candidates):
                if i in selected:
                    continue

                # Find patterns this candidate covers
                demo_patterns = set(self.pattern_extractor(demo))
                new_patterns = demo_patterns - covered_patterns

                if len(new_patterns) > len(best_new_patterns):
                    best_candidate = i
                    best_new_patterns = new_patterns

            if best_candidate is not None:
                selected.append(best_candidate)
                covered_patterns.update(best_new_patterns)
            else:
                # No new patterns, select randomly
                remaining = [i for i in range(len(candidates)) if i not in selected]
                if remaining:
                    selected.append(random.choice(remaining))

        return [candidates[i] for i in selected]

    def _default_pattern_extractor(self, demonstration):
        """Extract basic patterns from demonstration."""

        patterns = []

        # Length pattern
        length = len(demonstration['input'].split())
        if length < 10:
            patterns.append('short')
        elif length < 50:
            patterns.append('medium')
        else:
            patterns.append('long')

        # Format pattern
        if '?' in demonstration['input']:
            patterns.append('question')
        if '.' in demonstration['input'] and demonstration['input'].count('.') > 2:
            patterns.append('complex_sentence')

        # Content pattern
        text = demonstration['input'].lower()
        if any(word in text for word in ['why', 'how', 'when', 'where']):
            patterns.append('wh_question')
        if any(word in text for word in ['list', 'enumerate', 'name']):
            patterns.append('listing_task')

        return patterns
```

### 4. Learning-based Selection

Learn selection policy from training data.

```python
class LearnedSelector:
    """Learn demonstration selection policy from data."""

    def __init__(self, model_architecture="transformer"):
        self.model = self._build_selection_model(model_architecture)
        self.is_trained = False

    def train(self, train_data, val_data):
        """Train selection model on demonstration effectiveness data."""

        # Prepare training data
        # Each example: (query, candidates, labels, performance_scores)
        X_train, y_train = self._prepare_training_data(train_data)

        # Train model
        self.model.fit(X_train, y_train)

        # Validate
        X_val, y_val = self._prepare_training_data(val_data)
        val_score = self.model.evaluate(X_val, y_val)

        self.is_trained = True
        return val_score

    def select(self, query, candidates, k=5):
        """Select demonstrations using learned policy."""

        if not self.is_trained:
            # Fallback to similarity-based selection
            selector = SimilarityBasedSelector()
            return selector.select(query, candidates, k)

        # Score each candidate
        scores = []
        for candidate in candidates:
            features = self._extract_features(query, candidate)
            score = self.model.predict_proba([features])[0, 1]
            scores.append(score)

        # Select top-k
        top_indices = np.argsort(scores)[-k:][::-1]
        return [candidates[i] for i in top_indices]

    def _build_selection_model(self, architecture):
        """Build model architecture."""

        if architecture == "transformer":
            # Simple transformer model for demonstration selection
            model = build_transformer_classifier()
        elif architecture == "gradient_boosting":
            model = GradientBoostingClassifier()
        else:
            model = LogisticRegression()

        return model

    def _extract_features(self, query, candidate):
        """Extract features for selection model."""

        features = {
            'similarity': self._compute_similarity(query, candidate),
            'query_length': len(query.split()),
            'candidate_length': len(candidate['input'].split()),
            'complexity_score': self._compute_complexity(candidate),
            'topic_match': self._compute_topic_match(query, candidate),
        }

        return list(features.values())
```

## Demonstration Generation

### 1. Bootstrap Generation

Generate demonstrations from the model itself.

```python
class BootstrapDemonstrationGenerator:
    """Generate demonstrations using bootstrapping."""

    def __init__(self, model, confidence_threshold=0.8):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.generated_demos = []

    def generate_demonstrations(self, instruction, seed_examples, num_new=20):
        """Generate new demonstrations from model."""

        demonstrations = seed_examples.copy()

        while len(demonstrations) < num_new:
            # Sample from existing demonstrations
            seed = random.choice(demonstrations)

            # Generate new example based on seed
            new_demo = self._generate_variation(seed, instruction)

            # Validate quality
            if self._validate_demonstration(new_demo):
                demonstrations.append(new_demo)
                self.generated_demos.append(new_demo)

        return demonstrations

    def _generate_variation(self, seed_demo, instruction):
        """Generate variation of existing demonstration."""

        # Prompt model to create variation
        prompt = f"""
        Instruction: {instruction}

        Example:
        Input: {seed_demo['input']}
        Output: {seed_demo['output']}

        Create a similar but different example following the same pattern:
        Input:
        """

        response = self.model.generate(prompt, temperature=0.7)
        new_input = self._extract_input_from_response(response)

        # Generate output for new input
        full_prompt = f"""
        Instruction: {instruction}

        Input: {new_input}
        Output:
        """

        new_output = self.model.generate(full_prompt, temperature=0.1)

        return {
            'input': new_input,
            'output': new_output,
            'source': 'generated',
            'parent': seed_demo
        }

    def _validate_demonstration(self, demonstration):
        """Validate quality of generated demonstration."""

        # Check confidence
        confidence = self._compute_confidence(demonstration)

        # Check consistency with pattern
        consistency = self._check_consistency(demonstration)

        # Check uniqueness
        uniqueness = self._check_uniqueness(demonstration)

        return (
            confidence > self.confidence_threshold and
            consistency > 0.8 and
            uniqueness > 0.7
        )
```

### 2. Synthetic Generation

Create demonstrations from structured templates.

```python
class SyntheticDemonstrationGenerator:
    """Generate synthetic demonstrations from templates."""

    def __init__(self, templates=None):
        self.templates = templates or self._default_templates()
        self.attribute_values = self._load_attribute_values()

    def generate_demonstrations(self, instruction, num_demos=50):
        """Generate synthetic demonstrations."""

        demonstrations = []

        for _ in range(num_demos):
            # Sample template
            template = random.choice(self.templates)

            # Sample attribute values
            attributes = self._sample_attributes(template['attributes'])

            # Fill template
            demo = self._fill_template(template, attributes)

            demonstrations.append(demo)

        return demonstrations

    def _fill_template(self, template, attributes):
        """Fill template with sampled attributes."""

        input_text = template['input_template']
        output_text = template['output_template']

        # Replace placeholders
        for attr_name, attr_value in attributes.items():
            input_text = input_text.replace(f'{{{attr_name}}}', str(attr_value))
            output_text = output_text.replace(f'{{{attr_name}}}', str(attr_value))

        return {
            'input': input_text,
            'output': output_text,
            'template': template['name'],
            'attributes': attributes
        }

    def _default_templates(self):
        """Default demonstration templates."""

        return [
            {
                'name': 'math_addition',
                'input_template': 'What is {num1} + {num2}?',
                'output_template': '{num1} + {num2} = {sum}',
                'attributes': ['num1', 'num2']
            },
            {
                'name': 'text_classification',
                'input_template': 'Classify the sentiment: "{text}"',
                'output_template': 'Sentiment: {sentiment}',
                'attributes': ['text', 'sentiment']
            }
        ]
```

## Utility Functions

### 1. Performance-based Utility

Evaluate demonstrations by their impact on model performance.

```python
class PerformanceUtility:
    """Utility based on demonstration performance impact."""

    def __init__(self, model, evaluation_metric):
        self.model = model
        self.evaluation_metric = evaluation_metric

    def compute_utility(self, demonstration_set, validation_examples):
        """Compute utility score for demonstration set."""

        total_score = 0
        total_count = 0

        for example in validation_examples:
            # Create prompt with demonstrations
            prompt = self._format_prompt(demonstration_set, example['input'])

            # Generate response
            response = self.model.generate(prompt)

            # Score response
            score = self.evaluation_metric(response, example['output'])
            total_score += score
            total_count += 1

        return total_score / total_count if total_count > 0 else 0

    def marginal_utility(self, base_set, new_demo, validation_examples):
        """Compute marginal utility of adding new demonstration."""

        # Utility without new demo
        base_utility = self.compute_utility(base_set, validation_examples)

        # Utility with new demo
        extended_set = base_set + [new_demo]
        extended_utility = self.compute_utility(extended_set, validation_examples)

        return extended_utility - base_utility

    def _format_prompt(self, demonstrations, query):
        """Format prompt with demonstrations."""

        prompt = ""
        for i, demo in enumerate(demonstrations):
            prompt += f"Example {i+1}:\n"
            prompt += f"Input: {demo['input']}\n"
            prompt += f"Output: {demo['output']}\n\n"

        prompt += f"Input: {query}\nOutput:"

        return prompt
```

### 2. Information-theoretic Utility

Measure information content of demonstrations.

```python
class InformationUtility:
    """Utility based on information theory metrics."""

    def __init__(self):
        self.vocab_size = 50000  # Approximate vocabulary size

    def compute_entropy(self, demonstration_set):
        """Compute entropy of demonstration set."""

        # Collect all tokens
        all_tokens = []
        for demo in demonstration_set:
            all_tokens.extend(self._tokenize(demo['input']))
            all_tokens.extend(self._tokenize(demo['output']))

        # Compute token frequencies
        token_counts = {}
        for token in all_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        # Compute entropy
        total_tokens = len(all_tokens)
        entropy = 0
        for count in token_counts.values():
            prob = count / total_tokens
            entropy -= prob * np.log2(prob)

        return entropy

    def compute_mutual_information(self, demo1, demo2):
        """Compute mutual information between two demonstrations."""

        # Get tokens
        tokens1 = set(self._tokenize(demo1['input'] + demo1['output']))
        tokens2 = set(self._tokenize(demo2['input'] + demo2['output']))

        # Compute intersection
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        # Jaccard similarity as MI approximation
        mi = intersection / union if union > 0 else 0

        return mi

    def information_gain(self, base_set, new_demo):
        """Compute information gain from adding demonstration."""

        base_entropy = self.compute_entropy(base_set)
        new_entropy = self.compute_entropy(base_set + [new_demo])

        return new_entropy - base_entropy

    def _tokenize(self, text):
        """Simple tokenization."""

        return text.lower().split()
```

### 3. Coverage Utility

Measure how well demonstrations cover the input space.

```python
class CoverageUtility:
    """Utility based on input space coverage."""

    def __init__(self, feature_extractor=None):
        self.feature_extractor = feature_extractor or self._default_feature_extractor
        self.coverage_grid = None

    def compute_coverage(self, demonstration_set, grid_resolution=10):
        """Compute coverage of demonstration set."""

        # Extract features
        features = []
        for demo in demonstration_set:
            features.append(self.feature_extractor(demo['input']))

        features = np.array(features)

        # Normalize features
        features = (features - features.mean(axis=0)) / features.std(axis=0)

        # Create grid
        min_vals = features.min(axis=0)
        max_vals = features.max(axis=0)

        # Count covered grid cells
        grid_cells = set()
        for feature in features:
            cell = self._discretize_feature(feature, min_vals, max_vals, grid_resolution)
            grid_cells.add(cell)

        # Coverage = covered cells / total cells
        total_cells = grid_resolution ** features.shape[1]
        coverage = len(grid_cells) / total_cells

        return coverage

    def coverage_density(self, demonstration_set, query_point):
        """Compute coverage density around query point."""

        query_features = self.feature_extractor(query_point)
        query_features = np.array(query_features).reshape(1, -1)

        demo_features = []
        for demo in demonstration_set:
            features = self.feature_extractor(demo['input'])
            demo_features.append(features)

        demo_features = np.array(demo_features)

        # Compute distances
        distances = np.linalg.norm(demo_features - query_features, axis=1)

        # Density = 1 / average distance
        avg_distance = np.mean(distances)
        density = 1 / (avg_distance + 1e-6)

        return density

    def _discretize_feature(self, feature, min_vals, max_vals, resolution):
        """Discretize continuous feature to grid cell."""

        normalized = (feature - min_vals) / (max_vals - min_vals + 1e-6)
        cell_indices = (normalized * resolution).astype(int)
        cell_indices = np.clip(cell_indices, 0, resolution - 1)

        return tuple(cell_indices)

    def _default_feature_extractor(self, text):
        """Default feature extraction for text."""

        # Simple features
        features = [
            len(text.split()),  # Length
            text.count('?'),  # Number of questions
            text.count(','),  # Number of commas
            sum(1 for c in text if c.isupper()),  # Capital letters
            len(set(text.split())),  # Unique words
        ]

        return features
```

## Diversity Metrics

### 1. Lexical Diversity

Measure vocabulary diversity across demonstrations.

```python
class LexicalDiversity:
    """Measure lexical diversity of demonstrations."""

    def compute_type_token_ratio(self, demonstrations):
        """Compute type-token ratio (TTR)."""

        all_tokens = []
        for demo in demonstrations:
            tokens = self._tokenize(demo['input'] + ' ' + demo['output'])
            all_tokens.extend(tokens)

        num_types = len(set(all_tokens))
        num_tokens = len(all_tokens)

        return num_types / num_tokens if num_tokens > 0 else 0

    def compute_mattr(self, demonstrations, window_size=100):
        """Compute Moving Average Type-Token Ratio (MATTR)."""

        all_tokens = []
        for demo in demonstrations:
            tokens = self._tokenize(demo['input'] + ' ' + demo['output'])
            all_tokens.extend(tokens)

        if len(all_tokens) < window_size:
            return self.compute_type_token_ratio(demonstrations)

        ttrs = []
        for i in range(len(all_tokens) - window_size + 1):
            window = all_tokens[i:i + window_size]
            ttr = len(set(window)) / window_size
            ttrs.append(ttr)

        return np.mean(ttrs)

    def compute_vocab_overlap(self, demo1, demo2):
        """Compute vocabulary overlap between two demonstrations."""

        tokens1 = set(self._tokenize(demo1['input'] + ' ' + demo1['output']))
        tokens2 = set(self._tokenize(demo2['input'] + ' ' + demo2['output']))

        if not tokens1 or not tokens2:
            return 0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union

    def _tokenize(self, text):
        """Simple tokenization."""

        import re
        tokens = re.findall(r'\w+', text.lower())
        return tokens
```

### 2. Structural Diversity

Measure diversity in demonstration structure.

```python
class StructuralDiversity:
    """Measure structural diversity of demonstrations."""

    def compute_pattern_diversity(self, demonstrations):
        """Compute diversity of structural patterns."""

        patterns = []
        for demo in demonstrations:
            pattern = self._extract_pattern(demo['input'])
            patterns.append(pattern)

        unique_patterns = len(set(patterns))
        total_patterns = len(patterns)

        return unique_patterns / total_patterns if total_patterns > 0 else 0

    def compute_length_distribution(self, demonstrations):
        """Analyze length distribution diversity."""

        lengths = [len(demo['input'].split()) for demo in demonstrations]

        # Compute coefficient of variation
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        cv = std_length / mean_length if mean_length > 0 else 0

        return cv

    def compute_complexity_diversity(self, demonstrations):
        """Compute diversity in complexity scores."""

        complexities = []
        for demo in demonstrations:
            complexity = self._compute_complexity(demo['input'])
            complexities.append(complexity)

        # Range of complexities
        complexity_range = max(complexities) - min(complexities)
        max_possible_range = 10  # Normalized range

        return complexity_range / max_possible_range

    def _extract_pattern(self, text):
        """Extract structural pattern from text."""

        # Simplified pattern extraction
        pattern = []

        # Check for question marks
        if '?' in text:
            pattern.append('question')

        # Check for lists
        if ':' in text and (',' in text or ';' in text):
            pattern.append('list')

        # Check for quotes
        if '"' in text or "'" in text:
            pattern.append('quotation')

        # Check for numbers
        if any(c.isdigit() for c in text):
            pattern.append('numeric')

        return tuple(sorted(pattern))

    def _compute_complexity(self, text):
        """Compute text complexity score."""

        # Simple complexity metrics
        avg_word_length = np.mean([len(word) for word in text.split()])
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        avg_sentence_length = len(text.split()) / max(sentence_count, 1)

        # Combine metrics
        complexity = (avg_word_length * 0.3 + avg_sentence_length * 0.7)
        return min(complexity / 20, 1.0)  # Normalize to [0, 1]
```

## Optimization in Multi-stage Programs

### Stage-specific Demonstration Strategies

```python
class MultiStageDemonstrationOptimizer:
    """Optimize demonstrations for multi-stage programs."""

    def __init__(self, stage_configs):
        self.stage_configs = stage_configs
        self.selectors = {}
        self.utilities = {}

        # Initialize selectors for each stage
        for stage_name, config in stage_configs.items():
            self.selectors[stage_name] = self._create_selector(config['selection_strategy'])
            self.utilities[stage_name] = self._create_utility(config['utility_function'])

    def optimize_demonstrations(
        self,
        pipeline,
        trainset,
        demo_budget=50
    ):
        """Optimize demonstrations across all stages."""

        optimized_demos = {}

        # Analyze stage dependencies
        dependencies = self._analyze_dependencies(pipeline)

        # Optimize in dependency order
        for stage_name in self._topological_sort(dependencies):
            print(f"Optimizing demonstrations for stage: {stage_name}")

            # Get stage-specific training data
            stage_data = self._extract_stage_data(
                stage_name, pipeline, trainset
            )

            # Allocate budget for this stage
            stage_budget = self._allocate_budget(
                stage_name, demo_budget, dependencies
            )

            # Optimize demonstrations
            best_demos = self._optimize_stage_demonstrations(
                stage_name,
                stage_data,
                stage_budget
            )

            optimized_demos[stage_name] = best_demos

            # Update pipeline with new demonstrations
            pipeline.stages[stage_name].set_demonstrations(best_demos)

        return optimized_demos

    def _optimize_stage_demonstrations(
        self,
        stage_name,
        stage_data,
        budget
    ):
        """Optimize demonstrations for a specific stage."""

        # Get candidate demonstrations
        candidates = self._get_candidate_demonstrations(stage_data)

        best_set = []
        best_score = -float('inf')

        # Greedy selection with utility evaluation
        for _ in range(min(budget, len(candidates))):
            best_candidate = None
            best_candidate_score = -float('inf')

            for candidate in candidates:
                if candidate in best_set:
                    continue

                # Evaluate utility
                test_set = best_set + [candidate]
                score = self.utilities[stage_name].compute_utility(
                    test_set, stage_data['validation']
                )

                if score > best_candidate_score:
                    best_candidate_score = score
                    best_candidate = candidate

            if best_candidate:
                best_set.append(best_candidate)

        return best_set

    def _analyze_dependencies(self, pipeline):
        """Analyze dependencies between stages."""

        dependencies = {}
        stage_names = list(pipeline.stages.keys())

        for stage_name in stage_names:
            dependencies[stage_name] = []

            # Check if stage uses outputs from other stages
            stage_module = pipeline.stages[stage_name]
            if hasattr(stage_module, 'dependencies'):
                dependencies[stage_name] = stage_module.dependencies

        return dependencies
```

## Practical Considerations

### Context Window Management

```python
class ContextWindowManager:
    """Manage demonstration selection within context limits."""

    def __init__(self, max_tokens=2048, reserve_tokens=500):
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.available_tokens = max_tokens - reserve_tokens

    def select_within_limit(
        self,
        demonstrations,
        query,
        token_estimator=None
    ):
        """Select demonstrations that fit within context limit."""

        if token_estimator is None:
            token_estimator = self._default_token_estimator

        selected = []
        used_tokens = token_estimator(query)

        # Sort by some quality metric (e.g., diversity)
        sorted_demos = self._sort_by_quality(demonstrations)

        for demo in sorted_demos:
            demo_tokens = token_estimator(
                f"Input: {demo['input']}\nOutput: {demo['output']}\n\n"
            )

            if used_tokens + demo_tokens <= self.available_tokens:
                selected.append(demo)
                used_tokens += demo_tokens
            else:
                break

        return selected

    def _default_token_estimator(self, text):
        """Simple token estimation."""

        return len(text.split()) * 1.3  # Rough estimate

    def _sort_by_quality(self, demonstrations):
        """Sort demonstrations by quality."""

        # Simple quality score based on length and complexity
        scored = []
        for demo in demonstrations:
            score = len(demo['input']) + len(demo['output'])
            scored.append((demo, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [demo for demo, _ in scored]
```

### Dynamic Demonstration Updating

```python
class DynamicDemonstrationUpdater:
    """Dynamically update demonstrations based on performance."""

    def __init__(self, update_threshold=0.1, window_size=100):
        self.update_threshold = update_threshold
        self.window_size = window_size
        self.performance_history = []
        self.current_demonstrations = []

    def should_update(self, recent_performance):
        """Determine if demonstrations should be updated."""

        self.performance_history.append(recent_performance)

        if len(self.performance_history) < self.window_size:
            return False

        # Compute performance trend
        recent = self.performance_history[-10:]
        baseline = self.performance_history[-self.window_size:-10]

        avg_recent = np.mean(recent)
        avg_baseline = np.mean(baseline)

        # Update if performance drop exceeds threshold
        performance_drop = avg_baseline - avg_recent
        return performance_drop > self.update_threshold

    def update_demonstrations(
        self,
        instruction,
        examples,
        selector
    ):
        """Update demonstration set."""

        # Use recent examples as candidates
        candidates = examples[-50:]  # Last 50 examples

        # Select new demonstrations
        new_demos = selector.select(
            query=instruction,
            candidates=candidates,
            k=5
        )

        self.current_demonstrations = new_demos
        return new_demos
```

## Best Practices

### 1. Demonstration Quality Guidelines

- **Accuracy**: Ensure demonstrations are correct
- **Clarity**: Make examples easy to understand
- **Relevance**: Choose examples similar to target inputs
- **Diversity**: Cover different patterns and edge cases
- **Consistency**: Align with instruction and task requirements

### 2. Selection Strategy Tips

- Use similarity-based selection for homogeneous tasks
- Employ diversity-aware selection for varied inputs
- Apply coverage-based selection for pattern-rich tasks
- Consider learning-based selection for complex scenarios

### 3. Common Pitfalls

- **Overfitting**: Too similar demonstrations limit generalization
- **Context Overflow**: Exceeding model context limits
- **Poor Quality**: Incorrect examples harm performance
- **Imbalance**: Over-representation of certain patterns

## Summary

Demonstration optimization strategies provide systematic approaches to selecting and improving few-shot examples in language model programs. Key takeaways:

1. **Multiple Selection Algorithms**: Similarity, diversity, coverage, and learning-based approaches
2. **Generation Techniques**: Bootstrap and synthetic generation for creating demonstrations
3. **Utility Functions**: Performance, information-theoretic, and coverage-based utilities
4. **Diversity Metrics**: Lexical and structural diversity measurements
5. **Multi-stage Optimization**: Stage-specific strategies and dependency management

The next section will explore multi-stage program architectures, building on the optimization strategies discussed here to create comprehensive solutions.
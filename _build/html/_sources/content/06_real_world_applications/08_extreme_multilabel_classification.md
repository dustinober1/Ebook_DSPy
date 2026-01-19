# Extreme Multi-Label Classification: Scaling to Millions of Labels

## Introduction

Extreme Multi-Label Classification (XML) represents one of the most challenging frontiers in machine learning and natural language processing. Unlike traditional multi-label classification where you might deal with tens or hundreds of labels, XML tasks involve thousands, millions, or even tens of millions of potential labels. This extreme scale introduces unique computational, statistical, and algorithmic challenges that require specialized approaches.

DSPy, with its modular architecture and optimization capabilities, provides powerful tools for tackling XML problems effectively. In this section, we'll explore the fundamentals of XML, dive into specialized techniques for handling massive label spaces, and learn how to implement scalable XML solutions using DSPy.

## Understanding XML: Fundamentals and Challenges

### What Makes XML "Extreme"?

XML differs from traditional multi-label classification in several key dimensions:

#### Scale Dimensions
1. **Label Cardinality**: Number of labels per instance (typically 1-100)
2. **Label Space Size**: Total number of unique labels (thousands to millions)
3. **Instance Features**: High-dimensional input representations
4. **Data Volume**: Massive training datasets

#### Real-World XML Applications
- **E-commerce**: Product categorization (millions of product categories)
- **Content Tagging**: Wikipedia article tagging (over 2 million categories)
- **Advertising**: Ad targeting with millions of keywords
- **Document Classification**: Legal documents with thousands of topics
- **Bioinformatics**: Gene function annotation with tens of thousands of GO terms

### Core XML Challenges

#### 1. Computational Complexity
```python
# Traditional approach complexity: O(|L|) per instance
# Where |L| is the number of labels (millions!)

class NaiveXMLClassifier:
    def __init__(self, labels):
        self.labels = labels  # Could be millions!
        self.classifiers = {label: BinaryClassifier() for label in labels}

    def predict(self, text):
        # O(|L|) complexity - infeasible for XML
        predictions = []
        for label in self.labels:
            pred = self.classifiers[label].predict(text)
            if pred.confidence > threshold:
                predictions.append(label)
        return predictions
```

#### 2. Data Sparsity
- Most label pairs co-occur rarely
- Long-tail distribution of label frequencies
- Few training examples for rare labels

#### 3. Memory Constraints
- Storing millions of label embeddings
- Maintaining classifier parameters for all labels
- Caching predictions and intermediate results

#### 4. Evaluation Complexity
- Computing precision@k becomes expensive
- Hierarchical evaluation requires tree traversal
- Real-time inference constraints

## Label Space Representation and Indexing

### Efficient Label Representation

#### Label Embeddings
```python
import dspy
import numpy as np
from typing import List, Dict, Tuple

class XMLEmbeddingIndex:
    def __init__(self, labels: List[str], embedding_dim: int = 768):
        """
        Create an efficient embedding index for XML labels.

        Args:
            labels: List of all possible labels
            embedding_dim: Dimension for label embeddings
        """
        self.labels = labels
        self.label_to_id = {label: i for i, label in enumerate(labels)}
        self.embedding_dim = embedding_dim

        # Initialize label embeddings
        self.label_embeddings = np.random.normal(
            0, 0.1, (len(labels), embedding_dim)
        ).astype(np.float32)

        # Build search index
        self._build_search_index()

    def _build_search_index(self):
        """Build efficient search structures for label lookup."""
        import faiss  # Facebook AI Similarity Search

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(self.label_embeddings, axis=1, keepdims=True)
        self.normalized_embeddings = self.label_embeddings / (norms + 1e-8)

        # Build FAISS index for fast similarity search
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(self.normalized_embeddings)

    def search_similar_labels(self, query_embedding: np.ndarray,
                            k: int = 100) -> List[Tuple[str, float]]:
        """
        Find k most similar labels to query embedding.

        Args:
            query_embedding: Query text embedding
            k: Number of similar labels to retrieve

        Returns:
            List of (label, similarity_score) tuples
        """
        # Normalize query embedding
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        query_norm = query_norm.reshape(1, -1).astype(np.float32)

        # Search index
        similarities, indices = self.index.search(query_norm, k)

        # Convert to label-similarity pairs
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx != -1:  # Valid index
                results.append((self.labels[idx], float(sim)))

        return results

    def update_embedding(self, label: str, new_embedding: np.ndarray):
        """Update embedding for a specific label."""
        if label in self.label_to_id:
            label_id = self.label_to_id[label]
            self.label_embeddings[label_id] = new_embedding
            # Rebuild index periodically for efficiency
            if np.random.random() < 0.01:  # 1% chance to rebuild
                self._build_search_index()
```

### Hierarchical Label Organization

#### Tree-based Label Structure
```python
class XMLHierarchy:
    def __init__(self, hierarchy_data: Dict):
        """
        Organize labels in a hierarchical structure.

        Args:
            hierarchy_data: Nested dictionary representing label hierarchy
            Example: {
                "Technology": {
                    "AI": ["Machine Learning", "Deep Learning", "NLP"],
                    "Web": ["Frontend", "Backend", "DevOps"]
                },
                "Science": {
                    "Physics": ["Quantum", "Classical", "Particle"],
                    "Biology": ["Molecular", "Cellular", "Ecological"]
                }
            }
        """
        self.hierarchy = hierarchy_data
        self.flattened_labels = self._flatten_hierarchy()
        self.parent_map = self._build_parent_map()
        self.depth_map = self._build_depth_map()

    def _flatten_hierarchy(self) -> List[str]:
        """Extract all labels from hierarchical structure."""
        labels = []

        def extract_labels(node, path=""):
            if isinstance(node, dict):
                for key, value in node.items():
                    new_path = f"{path}/{key}" if path else key
                    extract_labels(value, new_path)
            elif isinstance(node, list):
                labels.extend(node)
            else:
                labels.append(node)

        extract_labels(self.hierarchy)
        return labels

    def _build_parent_map(self) -> Dict[str, str]:
        """Map each label to its parent category."""
        parent_map = {}

        def build_map(node, parent=None):
            if isinstance(node, dict):
                for key, value in node.items():
                    if isinstance(value, list):
                        for leaf in value:
                            parent_map[leaf] = key
                    else:
                        build_map(value, key)
            elif isinstance(node, list):
                for item in node:
                    if parent:
                        parent_map[item] = parent

        build_map(self.hierarchy)
        return parent_map

    def _build_depth_map(self) -> Dict[str, int]:
        """Calculate depth of each label in hierarchy."""
        depth_map = {}

        def calculate_depth(node, current_depth=0):
            if isinstance(node, dict):
                for key, value in node.items():
                    if isinstance(value, list):
                        for leaf in value:
                            depth_map[leaf] = current_depth + 2
                    else:
                        calculate_depth(value, current_depth + 1)
            elif isinstance(node, list):
                for item in node:
                    depth_map[item] = current_depth + 1

        calculate_depth(self.hierarchy)
        return depth_map

    def get_label_context(self, label: str, context_size: int = 3) -> List[str]:
        """Get contextual labels including parents and siblings."""
        context = []

        # Add parent
        if label in self.parent_map:
            parent = self.parent_map[label]
            context.append(parent)

            # Add siblings (labels with same parent)
            siblings = [
                l for l, p in self.parent_map.items()
                if p == parent and l != label
            ]
            context.extend(siblings[:context_size-1])

        return context

    def get_path_to_root(self, label: str) -> List[str]:
        """Get the complete path from label to root."""
        path = [label]
        current = label

        while current in self.parent_map:
            parent = self.parent_map[current]
            path.append(parent)
            current = parent

        return path[::-1]  # Reverse to get root-to-leaf path
```

### Label Clustering for Scalability

#### Dynamic Label Clustering
```python
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

class XMLLabelClusterer:
    def __init__(self, n_clusters: int = 1000):
        """
        Cluster labels for efficient candidate selection.

        Args:
            n_clusters: Number of label clusters
        """
        self.n_clusters = n_clusters
        self.cluster_model = None
        self.label_clusters = {}
        self.cluster_representatives = {}

    def fit(self, label_embeddings: np.ndarray, labels: List[str]):
        """
        Fit clustering model on label embeddings.

        Args:
            label_embeddings: Embedding matrix for all labels
            labels: List of label names
        """
        # Perform clustering
        self.cluster_model = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_labels = self.cluster_model.fit_predict(label_embeddings)

        # Organize labels by cluster
        for i, cluster_id in enumerate(cluster_labels):
            if cluster_id not in self.label_clusters:
                self.label_clusters[cluster_id] = []
            self.label_clusters[cluster_id].append(labels[i])

        # Find cluster representatives (closest to centroid)
        for cluster_id in range(self.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_embeddings = label_embeddings[cluster_mask]
            cluster_labels_list = np.array(labels)[cluster_mask]

            # Find label closest to centroid
            centroid = self.cluster_model.cluster_centers_[cluster_id]
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            closest_idx = np.argmin(distances)

            self.cluster_representatives[cluster_id] = cluster_labels_list[closest_idx]

    def get_candidate_clusters(self, query_embedding: np.ndarray,
                             top_k: int = 10) -> List[int]:
        """
        Get most relevant clusters for a query.

        Args:
            query_embedding: Query text embedding
            top_k: Number of clusters to retrieve

        Returns:
            List of cluster IDs
        """
        # Find closest cluster centroids
        distances = self.cluster_model.transform(query_embedding.reshape(1, -1))
        top_clusters = np.argsort(distances[0])[:top_k]

        return top_clusters.tolist()

    def get_cluster_labels(self, cluster_id: int) -> List[str]:
        """Get all labels in a specific cluster."""
        return self.label_clusters.get(cluster_id, [])
```

## DSPy XML Implementation

### Core XML Classifier Architecture

#### Scalable XML Classifier with DSPy
```python
class DSPyXMLClassifier(dspy.Module):
    """
    Extreme Multi-Label Classification system using DSPy.

    Features:
    - Efficient candidate label selection
    - Hierarchical prediction
    - Zero-shot and few-shot capabilities
    - Optimized inference pipeline
    """

    def __init__(self,
                 label_index: XMLEmbeddingIndex,
                 hierarchy: XMLHierarchy = None,
                 clusterer: XMLLabelClusterer = None,
                 max_candidates: int = 1000,
                 max_predictions: int = 10):
        """
        Initialize XML Classifier.

        Args:
            label_index: Pre-built label embedding index
            hierarchy: Optional label hierarchy
            clusterer: Optional label clusterer
            max_candidates: Maximum candidate labels to consider
            max_predictions: Maximum predictions to return
        """
        super().__init__()

        self.label_index = label_index
        self.hierarchy = hierarchy
        self.clusterer = clusterer
        self.max_candidates = max_candidates
        self.max_predictions = max_predictions

        # DSPy modules for different prediction stages
        self._initialize_modules()

    def _initialize_modules(self):
        """Initialize DSPy prediction modules."""

        # Module for generating text embedding
        self.text_encoder = dspy.Predict(
            "text -> embedding"
        )

        # Module for candidate selection
        self.candidate_selector = dspy.ChainOfThought(
            "text, candidate_labels -> relevant_candidates, selection_reasoning"
        )

        # Module for final classification
        self.final_classifier = dspy.Predict(
            "text, candidate_labels, context -> predictions, confidence_scores, reasoning"
        )

        # Module for zero-shot classification
        self.zero_shot_classifier = dspy.ChainOfThought(
            "text, label_description, examples -> relevant, confidence"
        )

    def forward(self, text: str,
                candidates: List[str] = None,
                context: Dict = None) -> dspy.Prediction:
        """
        Perform XML classification.

        Args:
            text: Input text to classify
            candidates: Optional pre-selected candidate labels
            context: Additional context information

        Returns:
            Prediction with labels, scores, and metadata
        """
        # Step 1: Generate text embedding
        text_embedding = self._get_text_embedding(text)

        # Step 2: Select candidate labels
        if candidates is None:
            candidates = self._select_candidates(text, text_embedding)

        # Step 3: Get contextual information
        label_context = self._get_label_context(candidates)

        # Step 4: Perform final classification
        result = self._classify_with_candidates(
            text, candidates, label_context
        )

        # Step 5: Post-process and organize results
        predictions = self._post_process_predictions(
            result, text_embedding
        )

        return dspy.Prediction(
            predictions=predictions["labels"],
            confidence_scores=predictions["scores"],
            reasoning=result.reasoning,
            candidates_used=len(candidates),
            context_used=label_context
        )

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for input text."""
        result = self.text_encoder(text=text)
        # Convert string representation to numpy array
        embedding_str = result.embedding
        embedding = self._parse_embedding(embedding_str)
        return embedding

    def _select_candidates(self, text: str,
                         text_embedding: np.ndarray) -> List[str]:
        """Select relevant candidate labels efficiently."""
        candidates = []

        # Method 1: Embedding-based similarity search
        similar_labels = self.label_index.search_similar_labels(
            text_embedding, k=self.max_candidates // 2
        )
        candidates.extend([label for label, _ in similar_labels])

        # Method 2: Cluster-based selection (if available)
        if self.clusterer:
            relevant_clusters = self.clusterer.get_candidate_clusters(
                text_embedding, top_k=20
            )
            for cluster_id in relevant_clusters:
                cluster_labels = self.clusterer.get_cluster_labels(cluster_id)
                candidates.extend(cluster_labels[:50])  # Limit per cluster

        # Method 3: Use DSPy for intelligent selection
        if len(candidates) < self.max_candidates:
            dspy_result = self.candidate_selector(
                text=text,
                candidate_labels="\n".join(candidates)
            )
            # Parse and add selected candidates
            selected = self._parse_candidates(dspy_result.relevant_candidates)
            candidates.extend(selected)

        # Remove duplicates and limit
        candidates = list(set(candidates))[:self.max_candidates]
        return candidates

    def _get_label_context(self, candidates: List[str]) -> str:
        """Generate contextual information for candidates."""
        context_parts = []

        # Hierarchical context
        if self.hierarchy:
            for label in candidates[:10]:  # Limit for efficiency
                path = self.hierarchy.get_path_to_root(label)
                if len(path) > 1:
                    context_parts.append(f"{label}: {' > '.join(path)}")

        # Co-occurrence patterns
        context_parts.append(f"Total candidates: {len(candidates)}")
        context_parts.append(f"Sample candidates: {candidates[:10]}")

        return "\n".join(context_parts)

    def _classify_with_candidates(self, text: str,
                                candidates: List[str],
                                context: str) -> dspy.Prediction:
        """Classify text using pre-selected candidates."""
        result = self.final_classifier(
            text=text,
            candidate_labels="\n".join(candidates),
            context=context
        )

        return result

    def _post_process_predictions(self,
                                 result: dspy.Prediction,
                                 text_embedding: np.ndarray) -> Dict:
        """Post-process and organize predictions."""
        # Parse predictions and scores
        predictions = self._parse_predictions(result.predictions)
        scores = self._parse_scores(result.confidence_scores)

        # Combine predictions with scores
        labeled_scores = list(zip(predictions, scores))

        # Sort by confidence and limit
        labeled_scores.sort(key=lambda x: x[1], reverse=True)
        labeled_scores = labeled_scores[:self.max_predictions]

        # Add hierarchical boost if available
        if self.hierarchy:
            labeled_scores = self._apply_hierarchical_boost(
                labeled_scores, text_embedding
            )

        return {
            "labels": [label for label, _ in labeled_scores],
            "scores": [score for _, score in labeled_scores]
        }

    def _parse_embedding(self, embedding_str: str) -> np.ndarray:
        """Parse embedding from string representation."""
        # Implementation depends on how embeddings are encoded
        import json
        try:
            return np.array(json.loads(embedding_str))
        except:
            # Fallback: generate embedding using external method
            return self._generate_fallback_embedding(embedding_str)

    def _parse_candidates(self, candidates_str: str) -> List[str]:
        """Parse candidate labels from string."""
        # Simple parsing - can be made more sophisticated
        return [c.strip() for c in candidates_str.split(",") if c.strip()]

    def _parse_predictions(self, predictions_str: str) -> List[str]:
        """Parse predictions from string."""
        return [p.strip() for p in predictions_str.split(",") if p.strip()]

    def _parse_scores(self, scores_str: str) -> List[float]:
        """Parse confidence scores from string."""
        scores = []
        for s in scores_str.split(","):
            try:
                scores.append(float(s.strip()))
            except:
                scores.append(0.0)
        return scores

    def _apply_hierarchical_boost(self,
                                 labeled_scores: List[Tuple[str, float]],
                                 text_embedding: np.ndarray) -> List[Tuple[str, float]]:
        """Apply hierarchical boosting to scores."""
        boosted_scores = []

        for label, score in labeled_scores:
            boosted_score = score

            # Boost if similar to parent/children labels
            if self.hierarchy and label in self.hierarchy.parent_map:
                parent = self.hierarchy.parent_map[label]
                # Check if parent is also predicted
                for other_label, other_score in labeled_scores:
                    if other_label == parent:
                        boosted_score *= 1.2  # Boost parent-child combinations
                        break

            boosted_scores.append((label, boosted_score))

        return boosted_scores

    def _generate_fallback_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using fallback method."""
        # This would typically use a sentence transformer or similar
        # For now, return a random embedding
        return np.random.normal(0, 0.1, self.label_index.embedding_dim)
```

### Zero-Shot XML Capabilities

#### Zero-Shot Label Prediction
```python
class ZeroShotXML(dspy.Module):
    """
    Zero-shot XML classification for new/unseen labels.
    """

    def __init__(self, label_descriptions: Dict[str, str]):
        """
        Initialize with label descriptions.

        Args:
            label_descriptions: Mapping of label names to descriptions
            Example: {
                "Machine Learning": "Algorithms that learn patterns from data",
                "Quantum Computing": "Computing using quantum mechanical phenomena"
            }
        """
        super().__init__()
        self.label_descriptions = label_descriptions

        # Initialize zero-shot modules
        self.description_matcher = dspy.ChainOfThought(
            "text, label_description -> relevance_score, explanation"
        )

        self.few_shot_learner = dspy.Predict(
            "text, examples, new_label -> is_relevant, confidence"
        )

    def predict_new_label(self,
                         text: str,
                         label_name: str,
                         label_description: str = None,
                         examples: List[dspy.Example] = None) -> dspy.Prediction:
        """
        Predict relevance for a new/unseen label.

        Args:
            text: Input text
            label_name: Name of the new label
            label_description: Optional description of the label
            examples: Optional few-shot examples

        Returns:
            Prediction with relevance score
        """
        description = label_description or self.label_descriptions.get(label_name, "")

        # Method 1: Description-based matching
        if description:
            desc_result = self.description_matcher(
                text=text,
                label_description=description
            )
            desc_score = float(desc_result.relevance_score)
        else:
            desc_score = 0.0

        # Method 2: Few-shot learning (if examples provided)
        if examples:
            examples_text = "\n".join([
                f"Example {i+1}: {ex.text}\nRelevant to {ex.label}: {ex.relevant}"
                for i, ex in enumerate(examples[:5])
            ])

            fs_result = self.few_shot_learner(
                text=text,
                examples=examples_text,
                new_label=label_name
            )
            fs_score = float(fs_result.confidence) if fs_result.is_relevant.lower() == "true" else 0.0
        else:
            fs_score = 0.0

        # Combine scores
        final_score = max(desc_score, fs_score)

        return dspy.Prediction(
            label=label_name,
            relevance_score=final_score,
            description_based_score=desc_score,
            few_shot_score=fs_score,
            explanation=desc_result.explanation if description else "No description provided"
        )

    def batch_predict_new_labels(self,
                               text: str,
                               new_labels: List[Tuple[str, str]]) -> List[dspy.Prediction]:
        """
        Batch predict multiple new labels.

        Args:
            text: Input text
            new_labels: List of (label_name, description) tuples

        Returns:
            List of predictions for each new label
        """
        predictions = []

        for label_name, description in new_labels:
            pred = self.predict_new_label(text, label_name, description)
            predictions.append(pred)

        # Sort by relevance score
        predictions.sort(key=lambda x: x.relevance_score, reverse=True)

        return predictions
```

## XML Evaluation Metrics and Methodologies

### Specialized XML Metrics

#### Implementation of XML Evaluation Metrics
```python
from typing import List, Set, Dict, Tuple
import numpy as np
from collections import defaultdict

class XMLEvaluator:
    """
    Comprehensive evaluation suite for Extreme Multi-Label Classification.
    """

    def __init__(self, k_values: List[int] = [1, 3, 5, 10]):
        """
        Initialize evaluator.

        Args:
            k_values: Values of k for precision@k and nDCG@k
        """
        self.k_values = k_values

    def precision_at_k(self,
                      true_labels: Set[str],
                      predicted_labels: List[str],
                      k: int) -> float:
        """
        Calculate Precision@k.

        Args:
            true_labels: Set of ground truth labels
            predicted_labels: Ordered list of predicted labels
            k: Cut-off position

        Returns:
            Precision@k score
        """
        if k <= 0:
            return 0.0

        top_k_predictions = predicted_labels[:k]
        relevant_in_top_k = sum(1 for label in top_k_predictions if label in true_labels)

        return relevant_in_top_k / k

    def recall_at_k(self,
                   true_labels: Set[str],
                   predicted_labels: List[str],
                   k: int) -> float:
        """
        Calculate Recall@k.

        Args:
            true_labels: Set of ground truth labels
            predicted_labels: Ordered list of predicted labels
            k: Cut-off position

        Returns:
            Recall@k score
        """
        if len(true_labels) == 0:
            return 0.0

        top_k_predictions = predicted_labels[:k]
        relevant_in_top_k = sum(1 for label in top_k_predictions if label in true_labels)

        return relevant_in_top_k / len(true_labels)

    def f1_at_k(self,
               true_labels: Set[str],
               predicted_labels: List[str],
               k: int) -> float:
        """Calculate F1@k score."""
        precision = self.precision_at_k(true_labels, predicted_labels, k)
        recall = self.recall_at_k(true_labels, predicted_labels, k)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def ndcg_at_k(self,
                 true_labels: Set[str],
                 predicted_labels: List[str],
                 k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@k.

        For binary relevance (relevant = 1, irrelevant = 0).
        """
        def dcg_at_k(relevances: List[int], k: int) -> float:
            """Calculate DCG@k."""
            dcg = 0.0
            for i, rel in enumerate(relevances[:k]):
                dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0
            return dcg

        # Calculate actual DCG
        actual_relevances = [1 if label in true_labels else 0
                           for label in predicted_labels]
        actual_dcg = dcg_at_k(actual_relevances, k)

        # Calculate ideal DCG (perfect ordering)
        ideal_relevances = [1] * min(len(true_labels), k) + \
                          [0] * max(0, k - len(true_labels))
        ideal_dcg = dcg_at_k(ideal_relevances, k)

        if ideal_dcg == 0:
            return 0.0

        return actual_dcg / ideal_dcg

    def ps_at_k(self,
               true_labels: Set[str],
               predicted_labels: List[str],
               k: int) -> float:
        """
        Calculate Propensity Scored Precision@k.

        Accounts for label frequency bias in evaluation.
        """
        # This would require label frequency statistics
        # Simplified implementation shown here
        return self.precision_at_k(true_labels, predicted_labels, k)

    def evaluate_instance(self,
                         true_labels: Set[str],
                         predicted_labels: List[str]) -> Dict[str, float]:
        """
        Evaluate a single instance with all metrics.

        Args:
            true_labels: Ground truth labels
            predicted_labels: Predicted labels (ordered by confidence)

        Returns:
            Dictionary of metric scores
        """
        metrics = {}

        for k in self.k_values:
            metrics[f'precision@{k}'] = self.precision_at_k(
                true_labels, predicted_labels, k
            )
            metrics[f'recall@{k}'] = self.recall_at_k(
                true_labels, predicted_labels, k
            )
            metrics[f'f1@{k}'] = self.f1_at_k(
                true_labels, predicted_labels, k
            )
            metrics[f'ndcg@{k}'] = self.ndcg_at_k(
                true_labels, predicted_labels, k
            )
            metrics[f'ps@{k}'] = self.ps_at_k(
                true_labels, predicted_labels, k
            )

        # Add macro and micro averaged metrics for multiple instances
        return metrics

    def evaluate_dataset(self,
                        test_data: List[Dict]) -> Dict[str, float]:
        """
        Evaluate entire dataset.

        Args:
            test_data: List of instances with 'true_labels' and 'predicted_labels'

        Returns:
            Dictionary of average metric scores
        """
        all_metrics = []

        for instance in test_data:
            metrics = self.evaluate_instance(
                instance['true_labels'],
                instance['predicted_labels']
            )
            all_metrics.append(metrics)

        # Calculate averages
        avg_metrics = {}
        for metric_name in all_metrics[0].keys():
            values = [m[metric_name] for m in all_metrics]
            avg_metrics[f'avg_{metric_name}'] = np.mean(values)
            avg_metrics[f'std_{metric_name}'] = np.std(values)

        return avg_metrics
```

### Propensity Scored Evaluation

#### Handling Label Imbalance in Evaluation
```python
class PropensityScoredEvaluator(XMLEvaluator):
    """
    Evaluator with propensity scoring for imbalanced XML datasets.
    """

    def __init__(self,
                 label_frequencies: Dict[str, int],
                 k_values: List[int] = [1, 3, 5, 10]):
        """
        Initialize with label frequency information.

        Args:
            label_frequencies: Frequency of each label in training data
            k_values: Values of k for evaluation
        """
        super().__init__(k_values)
        self.label_frequencies = label_frequencies
        self.propensity_scores = self._calculate_propensity_scores()

    def _calculate_propensity_scores(self) -> Dict[str, float]:
        """
        Calculate propensity scores for each label.

        Propensity score ~ (frequency + 1)^(-0.55) as per Jain et al. 2016
        """
        max_freq = max(self.label_frequencies.values())
        propensity_scores = {}

        for label, freq in self.label_frequencies.items():
            # Normalize frequency
            norm_freq = freq / max_freq
            # Calculate propensity
            propensity = (norm_freq + 1) ** (-0.55)
            propensity_scores[label] = propensity

        return propensity_scores

    def inv_psr_at_k(self,
                    true_labels: Set[str],
                    predicted_labels: List[str],
                    k: int) -> float:
        """
        Calculate Inverse Propensity Scored Precision@k.

        Gives more weight to rare labels.
        """
        if k <= 0:
            return 0.0

        top_k_predictions = predicted_labels[:k]
        weighted_relevant = 0.0
        total_weight = 0.0

        for i, label in enumerate(top_k_predictions):
            if label in true_labels:
                # Get inverse propensity score
                inv_propensity = 1.0 / self.propensity_scores.get(label, 1.0)
                weighted_relevant += inv_propensity

            total_weight += 1.0 / self.propensity_scores.get(label, 1.0)

        if total_weight == 0:
            return 0.0

        return weighted_relevant / total_weight
```

## Advanced XML Techniques

### In-Context Learning for XML

#### Dynamic In-Context Example Selection
```python
class XMLInContextLearner(dspy.Module):
    """
    In-context learning system specifically designed for XML tasks.
    """

    def __init__(self,
                 example_database: List[dspy.Example],
                 max_examples: int = 5,
                 similarity_threshold: float = 0.7):
        """
        Initialize with example database.

        Args:
            example_database: Collection of labeled examples
            max_examples: Maximum examples to include in context
            similarity_threshold: Minimum similarity for example selection
        """
        super().__init__()
        self.example_database = example_database
        self.max_examples = max_examples
        self.similarity_threshold = similarity_threshold

        # Initialize modules
        self.example_selector = dspy.Predict(
            "query_text, examples -> selected_examples, selection_scores"
        )

        self.context_learner = dspy.ChainOfThought(
            "text, examples, label_space -> predictions, confidence"
        )

        # Pre-compute example embeddings for efficient retrieval
        self.example_embeddings = self._precompute_embeddings()

    def _precompute_embeddings(self) -> Dict[str, np.ndarray]:
        """Pre-compute embeddings for all examples."""
        embeddings = {}
        # In practice, use an efficient embedding model
        for ex in self.example_database:
            # Simplified - would use actual embedding model
            embeddings[ex.text] = np.random.random(768)
        return embeddings

    def select_relevant_examples(self, query_text: str) -> List[dspy.Example]:
        """
        Select most relevant examples for the query.

        Uses multiple strategies:
        1. Label overlap
        2. Text similarity
        3. Label co-occurrence patterns
        """
        selected = []

        # Strategy 1: Exact label matches
        query_embedding = self._get_embedding(query_text)

        for ex in self.example_database:
            if len(selected) >= self.max_examples:
                break

            # Calculate similarity
            ex_embedding = self.example_embeddings.get(ex.text)
            if ex_embedding is not None:
                similarity = np.dot(query_embedding, ex_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(ex_embedding)
                )

                if similarity > self.similarity_threshold:
                    selected.append((ex, similarity))

        # Sort by similarity and select top examples
        selected.sort(key=lambda x: x[1], reverse=True)
        return [ex for ex, _ in selected[:self.max_examples]]

    def forward(self,
                text: str,
                label_space: List[str]) -> dspy.Prediction:
        """
        Perform in-context learning for XML.

        Args:
            text: Input text to classify
            label_space: Available labels for this instance

        Returns:
            Prediction with labels and confidence
        """
        # Select relevant examples
        selected_examples = self.select_relevant_examples(text)

        # Format examples for prompt
        formatted_examples = self._format_examples(selected_examples)

        # Perform in-context learning
        result = self.context_learner(
            text=text,
            examples=formatted_examples,
            label_space=", ".join(label_space)
        )

        # Parse and filter predictions
        predictions = self._parse_predictions(result.predictions, label_space)

        return dspy.Prediction(
            predictions=predictions["labels"],
            confidence=predictions["confidence"],
            examples_used=len(selected_examples),
            reasoning=result.rationale
        )

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        # Simplified - would use actual embedding model
        return np.random.random(768)

    def _format_examples(self, examples: List[dspy.Example]) -> str:
        """Format examples for the prompt."""
        formatted = []
        for i, ex in enumerate(examples, 1):
            formatted.append(
                f"Example {i}:\nText: {ex.text}\nLabels: {ex.labels}\n"
            )
        return "\n".join(formatted)

    def _parse_predictions(self,
                         predictions_str: str,
                         label_space: List[str]) -> Dict:
        """Parse and filter predictions to valid labels."""
        # Parse predictions
        all_predictions = [p.strip() for p in predictions_str.split(",")]

        # Filter to valid labels
        valid_predictions = []
        for pred in all_predictions:
            # Find closest match in label space
            closest = self._find_closest_label(pred, label_space)
            if closest and closest not in valid_predictions:
                valid_predictions.append(closest)

        return {
            "labels": valid_predictions,
            "confidence": 1.0 / (1.0 + len(valid_predictions))  # Simple confidence
        }

    def _find_closest_label(self,
                          prediction: str,
                          label_space: List[str]) -> str:
        """Find closest matching label in label space."""
        prediction = prediction.lower()

        # Exact match
        for label in label_space:
            if label.lower() == prediction:
                return label

        # Partial match
        for label in label_space:
            if prediction in label.lower() or label.lower() in prediction:
                return label

        # Word overlap
        pred_words = set(prediction.split())
        best_match = None
        max_overlap = 0

        for label in label_space:
            label_words = set(label.lower().split())
            overlap = len(pred_words & label_words)
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = label

        return best_match if max_overlap > 0 else None
```

### Meta-Learning for XML

#### Adaptation to New Domains
```python
class XMLMetaLearner(dspy.Module):
    """
    Meta-learning system for rapid adaptation to new XML domains.
    """

    def __init__(self,
                 base_classifier: DSPyXMLClassifier,
                 adaptation_steps: int = 5):
        """
        Initialize meta-learner.

        Args:
            base_classifier: Base XML classifier to adapt
            adaptation_steps: Number of adaptation steps
        """
        super().__init__()
        self.base_classifier = base_classifier
        self.adaptation_steps = adaptation_steps

        # Meta-learning modules
        self.task_analyzer = dspy.Predict(
            "support_examples, query_example -> task_characteristics"
        )

        self.adaptation_generator = dspy.ChainOfThought(
            "base_model, task_characteristics -> adapted_configuration"
        )

    def adapt_to_domain(self,
                       support_set: List[dspy.Example],
                       query_example: dspy.Example) -> DSPyXMLClassifier:
        """
        Adapt base classifier to new domain using few examples.

        Args:
            support_set: Small set of examples from new domain
            query_example: Example to classify

        Returns:
            Adapted classifier
        """
        # Analyze task characteristics
        task_chars = self.task_analyzer(
            support_examples=self._format_support_set(support_set),
            query_example=str(query_example)
        )

        # Generate adaptation configuration
        config = self.adaptation_generator(
            base_model=str(self.base_classifier),
            task_characteristics=task_chars.task_characteristics
        )

        # Apply adaptations
        adapted_classifier = self._apply_adaptations(config)

        return adapted_classifier

    def _format_support_set(self, support_set: List[dspy.Example]) -> str:
        """Format support set for analysis."""
        formatted = []
        for i, ex in enumerate(support_set, 1):
            formatted.append(f"Example {i}: {ex.text} -> {ex.labels}")
        return "\n".join(formatted)

    def _apply_adaptations(self, config: dspy.Prediction) -> DSPyXMLClassifier:
        """Apply configuration adaptations to base classifier."""
        # Parse configuration and apply changes
        # This would modify thresholds, weights, etc.
        return self.base_classifier  # Simplified
```

## Optimization Techniques for XML

### Efficient Training Strategies

#### Hierarchical Bootstrap Optimization
```python
class XMLBootstrapOptimizer:
    """
    Specialized optimizer for XML that leverages label hierarchy.
    """

    def __init__(self,
                 hierarchy: XMLHierarchy,
                 base_optimizer: dspy.BootstrapFewShot):
        """
        Initialize with label hierarchy and base optimizer.

        Args:
            hierarchy: Label hierarchy structure
            base_optimizer: Base DSPy optimizer
        """
        self.hierarchy = hierarchy
        self.base_optimizer = base_optimizer

    def hierarchical_optimize(self,
                            module: DSPyXMLClassifier,
                            trainset: List[dspy.Example]) -> DSPyXMLClassifier:
        """
        Perform hierarchical optimization of XML classifier.

        Optimizes in stages:
        1. Root level classifiers
        2. Branch level classifiers
        3. Leaf level classifiers
        """
        # Group examples by hierarchy level
        root_examples = []
        branch_examples = defaultdict(list)
        leaf_examples = defaultdict(list)

        for example in trainset:
            # Determine hierarchy level for each label
            for label in example.labels:
                depth = self.hierarchy.depth_map.get(label, 0)

                if depth == 1:  # Root level
                    root_examples.append(example)
                elif depth == 2:  # Branch level
                    parent = self.hierarchy.parent_map.get(label, "unknown")
                    branch_examples[parent].append(example)
                else:  # Leaf level
                    parent = self.hierarchy.parent_map.get(label, "unknown")
                    leaf_examples[parent].append(example)

        # Optimize root level
        if root_examples:
            module = self.base_optimizer.compile(
                module, trainset=root_examples[:100]
            )

        # Optimize branch levels
        for parent, examples in branch_examples.items():
            if len(examples) > 10:
                # Create specialized module for this branch
                branch_module = self._create_branch_module(parent)
                branch_module = self.base_optimizer.compile(
                    branch_module, trainset=examples[:50]
                )
                module.branch_modules[parent] = branch_module

        # Optimize leaf levels with few-shot
        for parent, examples in leaf_examples.items():
            if len(examples) > 5:
                # Fine-tune with specific examples
                self._fine_tune_leaf(module, parent, examples)

        return module
```

### Memory-Efficient Inference

#### Streaming Label Processing
```python
class StreamingXMLProcessor:
    """
    Process labels in streams to handle massive label spaces efficiently.
    """

    def __init__(self,
                 label_streams: Dict[str, List[str]],
                 batch_size: int = 10000):
        """
        Initialize with label streams.

        Args:
            label_streams: Dictionary mapping stream names to label lists
            batch_size: Number of labels to process in each batch
        """
        self.label_streams = label_streams
        self.batch_size = batch_size

    def stream_classify(self,
                       text: str,
                       classifier: DSPyXMLClassifier) -> Dict[str, List]:
        """
        Perform streaming classification.

        Processes labels in batches to manage memory usage.
        """
        all_predictions = []

        for stream_name, labels in self.label_streams.items():
            stream_predictions = []

            # Process labels in batches
            for i in range(0, len(labels), self.batch_size):
                batch_labels = labels[i:i + self.batch_size]

                # Classify batch
                batch_result = classifier(
                    text=text,
                    candidates=batch_labels
                )

                # Filter predictions by confidence
                for label, score in zip(
                    batch_result.predictions,
                    batch_result.confidence_scores
                ):
                    if score > 0.1:  # Confidence threshold
                        stream_predictions.append({
                            'label': label,
                            'score': score,
                            'stream': stream_name
                        })

            all_predictions.extend(stream_predictions)

        # Sort by score and return top predictions
        all_predictions.sort(key=lambda x: x['score'], reverse=True)

        return {
            'predictions': all_predictions[:100],  # Top 100 predictions
            'streams_processed': list(self.label_streams.keys()),
            'total_labels_evaluated': sum(len(labels)
                                        for labels in self.label_streams.values())
        }
```

## Real-World XML Applications

### Wikipedia Article Tagging System

```python
class WikipediaTagger(DSPyXMLClassifier):
    """
    XML system for automatically tagging Wikipedia articles.
    """

    def __init__(self, category_hierarchy: Dict):
        """
        Initialize with Wikipedia category hierarchy.

        Args:
            category_hierarchy: Nested Wikipedia category structure
        """
        # Build label index from Wikipedia categories
        all_categories = self._extract_categories(category_hierarchy)
        label_index = XMLEmbeddingIndex(all_categories)

        # Build hierarchy
        hierarchy = XMLHierarchy(category_hierarchy)

        # Initialize base classifier
        super().__init__(
            label_index=label_index,
            hierarchy=hierarchy,
            max_candidates=5000,
            max_predictions=20
        )

        # Wikipedia-specific modules
        self.category_validator = dspy.Predict(
            "article_text, category -> is_valid_category, validation_reasoning"
        )

        self.notability_checker = dspy.ChainOfThought(
            "article_text, category -> notability_score, explanation"
        )

    def _extract_categories(self, hierarchy: Dict) -> List[str]:
        """Extract all category names from hierarchy."""
        categories = []

        def extract(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    categories.append(key)
                    extract(value)
            elif isinstance(node, list):
                categories.extend(node)

        extract(hierarchy)
        return list(set(categories))  # Remove duplicates

    def tag_article(self,
                   article_text: str,
                   article_title: str = None,
                   existing_categories: List[str] = None) -> dspy.Prediction:
        """
        Tag a Wikipedia article with appropriate categories.

        Args:
            article_text: Full article text
            article_title: Article title for context
            existing_categories: Already assigned categories

        Returns:
            Predictions with category tags
        """
        # Prepare context
        context = {
            'title': article_title,
            'existing_categories': existing_categories or [],
            'text_length': len(article_text),
            'has_references': '[References]' in article_text
        }

        # Get initial predictions
        predictions = self.forward(article_text, context=context)

        # Validate predictions
        validated_predictions = []
        for category, score in zip(predictions.predictions,
                                 predictions.confidence_scores):
            # Validate category appropriateness
            validation = self.category_validator(
                article_text=article_text[:1000],  # First 1000 chars
                category=category
            )

            if validation.is_valid_category.lower() == "true":
                # Check notability
                notability = self.notability_checker(
                    article_text=article_text[:1000],
                    category=category
                )

                if float(notability.notability_score) > 0.5:
                    validated_predictions.append({
                        'category': category,
                        'score': score,
                        'validation': validation.validation_reasoning,
                        'notability': notability.notability_score
                    })

        # Sort by combined score
        validated_predictions.sort(
            key=lambda x: x['score'] * float(x['notability']),
            reverse=True
        )

        return dspy.Prediction(
            categories=[p['category'] for p in validated_predictions[:10]],
            scores=[p['score'] for p in validated_predictions[:10]],
            validations=[p['validation'] for p in validated_predictions[:10]],
            notability_scores=[p['notability'] for p in validated_predictions[:10]]
        )
```

## Key Takeaways

1. **XML requires specialized approaches** due to massive label spaces and computational challenges
2. **Efficient candidate selection** is crucial for scalable XML inference
3. **Hierarchical organization** of labels significantly improves performance and interpretability
4. **Zero-shot capabilities** enable handling of new and emerging labels
5. **Specialized evaluation metrics** account for label imbalance and XML-specific challenges
6. **In-context learning** provides powerful adaptation capabilities for XML tasks
7. **Meta-learning** enables rapid domain adaptation with few examples
8. **Memory-efficient processing** is essential for production XML systems

## Next Steps

In the next section, we'll explore **Entity Extraction**, demonstrating how to build systems that can identify and extract structured information from unstructured text. We'll see how DSPy's modular approach extends to extraction tasks and learn optimization strategies for high-accuracy entity recognition.
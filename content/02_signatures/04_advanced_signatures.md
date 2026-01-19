# Advanced Signatures

## Prerequisites

- **Previous Section**: [Typed Signatures](./03-typed-signatures.md) - Understanding of typed signatures
- **Required Knowledge**: Advanced programming concepts, data structures, and system design
- **Difficulty Level**: Advanced
- **Estimated Reading Time**: 40 minutes

## Learning Objectives

By the end of this section, you will:
- Master advanced signature patterns for complex applications
- Understand how to create hierarchical and nested signatures
- Learn techniques for dynamic and conditional signatures
- Be able to design signature systems for production applications

## Hierarchical Signatures

Hierarchical signatures allow you to compose complex workflows from smaller, reusable signature components.

### Base Signatures

```python
import dspy
from typing import Dict, List, Optional, Union

class BaseExtractor(dspy.Signature):
    """Base signature for extracting information from text."""

    source_text = dspy.InputField(
        desc="Source text to extract information from",
        type=str
    )

    extraction_confidence = dspy.OutputField(
        desc="Confidence in extraction quality (0-1)",
        type=float
    )

class NamedEntityExtractor(BaseExtractor):
    """Extract named entities from text."""

    entities = dspy.OutputField(
        desc="List of named entities with types and positions",
        type=List[Dict[str, Union[str, int]]],
        prefix="Named Entities:\n"
    )

class RelationshipExtractor(BaseExtractor):
    """Extract relationships between entities."""

    relationships = dspy.OutputField(
        desc="List of relationships with source, target, and type",
        type=List[Dict[str, str]],
        prefix="Relationships:\n"
    )
```

### Composite Signatures

```python
class DocumentAnalyzer(dspy.Signature):
    """Comprehensive document analysis using multiple extraction methods."""

    document_text = dspy.InputField(
        desc="Full document text to analyze",
        type=str
    )

    analysis_scope = dspy.InputField(
        desc="Scope of analysis (e.g., 'entities_only', 'full_analysis')",
        type=str
    )

    # Use nested signatures
    entity_extraction = dspy.OutputField(
        desc="Named entities found in document",
        type=NamedEntityExtractor,
        prefix="Entity Extraction:\n"
    )

    relationship_extraction = dspy.OutputField(
        desc="Relationships between entities",
        type=RelationshipExtractor,
        prefix="Relationship Extraction:\n"
    )

    document_summary = dspy.OutputField(
        desc="High-level document summary",
        type=str,
        prefix="Document Summary:\n"
    )

    metadata = dspy.OutputField(
        desc="Document metadata (length, language, complexity)",
        type=Dict[str, Union[str, int, float]],
        prefix="Document Metadata:\n"
    )
```

## Dynamic Signatures

Dynamic signatures adapt their structure based on input parameters or runtime conditions.

### Template-Based Dynamic Signatures

```python
class DynamicSignatureBuilder:
    """Build signatures dynamically based on requirements."""

    @staticmethod
    def create_analysis_signature(schema: Dict[str, str]) -> dspy.Signature:
        """Create a signature from a schema definition."""

        class DynamicAnalysis(dspy.Signature):
            """Dynamically created analysis signature."""

            input_data = dspy.InputField(
                desc="Data to analyze",
                type=str
            )

            analysis_instructions = dspy.InputField(
                desc="Specific analysis instructions",
                type=str
            )

            # Dynamically create output fields
            pass

        # Add dynamic output fields
        for field_name, field_desc in schema.items():
            setattr(
                DynamicAnalysis,
                field_name,
                dspy.OutputField(
                    desc=field_desc,
                    type=str,
                    prefix=f"{field_name.replace('_', ' ').title()}:\n"
                )
            )

        return DynamicAnalysis

# Usage
schema = {
    "sentiment_score": "Sentiment rating from -1 to 1",
    "key_themes": "Main themes identified",
    "emotional_tone": "Overall emotional tone",
    "recommendation": "Recommended action"
}

dynamic_signature = DynamicSignatureBuilder.create_analysis_signature(schema)
analyzer = dspy.Predict(dynamic_signature)
```

### Conditional Signatures

```python
class ConditionalSignature(dspy.Signature):
    """Signature that changes based on input conditions."""

    input_data = dspy.InputField(
        desc="Data to process",
        type=Union[str, dict, list]
    )

    data_type = dspy.InputField(
        desc="Type of input data ('text', 'json', 'list')",
        type=str
    )

    processing_mode = dspy.InputField(
        desc="Processing mode ('extract', 'transform', 'analyze')",
        type=str
    )

    # Conditional outputs
    extracted_features = dspy.OutputField(
        desc="Extracted features (when mode='extract')",
        type=List[str],
        optional=True
    )

    transformed_data = dspy.OutputField(
        desc="Transformed data (when mode='transform')",
        type=Union[str, dict],
        optional=True
    )

    analysis_results = dspy.OutputField(
        desc="Analysis results (when mode='analyze')",
        type=Dict[str, Union[str, float, int]],
        optional=True
    )

    processing_metadata = dspy.OutputField(
        desc="Metadata about processing performed",
        type=Dict[str, Union[str, int]],
        prefix="Processing Metadata:\n"
    )
```

## Multi-Modal Signatures

Signatures that handle different types of data and media.

### Image and Text Signatures

```python
class MultiModalAnalyzer(dspy.Signature):
    """Analyze content across multiple modalities."""

    text_content = dspy.InputField(
        desc="Text content to analyze",
        type=str,
        optional=True
    )

    image_description = dspy.InputField(
        desc="Description of image content",
        type=str,
        optional=True
    )

    audio_transcript = dspy.InputField(
        desc="Transcript of audio content",
        type=str,
        optional=True
    )

    content_type = dspy.InputField(
        desc="Types of content provided",
        type=List[str]
    )

    unified_analysis = dspy.OutputField(
        desc="Analysis combining all modalities",
        type=str,
        prefix="Unified Analysis:\n"
    )

    cross_modal_insights = dspy.OutputField(
        desc="Insights from combining different modalities",
        type=List[str],
        prefix="Cross-Modal Insights:\n"
    )

    confidence_scores = dspy.OutputField(
        desc="Confidence scores for each modality",
        type=Dict[str, float],
        prefix="Confidence Scores:\n"
    )
```

## Streaming and Batch Signatures

### Batch Processing Signatures

```python
class BatchProcessor(dspy.Signature):
    """Process multiple items in a batch."""

    batch_items = dspy.InputField(
        desc="List of items to process",
        type=List[Union[str, dict]]
    )

    processing_instructions = dspy.InputField(
        desc="Instructions for batch processing",
        type=str
    )

    batch_size = dspy.InputField(
        desc="Number of items in this batch",
        type=int
    )

    processed_items = dspy.OutputField(
        desc="List of processed items",
        type=List[Dict[str, Union[str, int, float]]],
        prefix="Processed Items:\n"
    )

    batch_summary = dspy.OutputField(
        desc="Summary of batch processing results",
        type=Dict[str, Union[int, float, str]],
        prefix="Batch Summary:\n"
    )

    failed_items = dspy.OutputField(
        desc="Items that failed processing with error messages",
        type=List[Dict[str, str]],
        optional=True
    )
```

### Streaming Signatures

```python
class StreamProcessor(dspy.Signature):
    """Process data streams in chunks."""

    chunk_data = dspy.InputField(
        desc="Current chunk of data to process",
        type=str
    )

    chunk_metadata = dspy.InputField(
        desc="Metadata about current chunk (position, size, etc.)",
        type=Dict[str, Union[int, str]]
    )

    is_final_chunk = dspy.InputField(
        desc="Whether this is the last chunk",
        type=bool
    )

    accumulated_context = dspy.InputField(
        desc="Context from previous chunks",
        type=str,
        optional=True
    )

    processed_chunk = dspy.OutputField(
        desc="Processed version of current chunk",
        type=str
    )

    updated_context = dspy.OutputField(
        desc="Updated context for next chunk",
        type=str,
        optional=True
    )

    chunk_summary = dspy.OutputField(
        desc="Summary of this chunk's processing",
        type=str,
        optional=True
    )
```

## Recursive and Self-Referential Signatures

Signatures that can process hierarchical or nested data structures.

### Tree Processing Signatures

```python
class TreeProcessor(dspy.Signature):
    """Process tree-like data structures."""

    node_data = dspy.InputField(
        desc="Data for current node",
        type=Union[str, dict]
    )

    children_data = dspy.InputField(
        desc="Data for child nodes",
        type=List[Union[str, dict]],
        optional=True
    )

    node_path = dspy.InputField(
        desc="Path from root to current node",
        type=str
    )

    depth = dspy.InputField(
        desc="Depth of current node",
        type=int
    )

    # Recursive processing
    node_analysis = dspy.OutputField(
        desc="Analysis of current node",
        type=Dict[str, Union[str, int, float]]
    )

    children_analyses = dspy.OutputField(
        desc="Analyses of child nodes",
        type=List[Dict[str, Union[str, int, float]]],
        optional=True
    )

    aggregated_analysis = dspy.OutputField(
        desc="Analysis aggregated from children",
        type=Dict[str, Union[str, int, float]],
        optional=True
    )
```

## Domain-Specific Advanced Patterns

### Medical Diagnosis Pipeline

```python
class DiagnosticPipeline(dspy.Signature):
    """Multi-stage medical diagnosis pipeline."""

    patient_data = dspy.InputField(
        desc="Comprehensive patient information",
        type=Dict[str, Union[str, int, float, List[str]]]
    )

    chief_complaint = dspy.InputField(
        desc="Primary reason for medical consultation",
        type=str
    )

    # Stage 1: Symptom Analysis
    symptom_analysis = dspy.OutputField(
        desc="Detailed analysis of presented symptoms",
        type=Dict[str, Union[str, List[str], float]],
        prefix="Symptom Analysis:\n"
    )

    # Stage 2: Differential Diagnosis
    differential_diagnosis = dspy.OutputField(
        desc="List of possible diagnoses with probabilities",
        type=List[Dict[str, Union[str, float, List[str]]]],
        prefix="Differential Diagnosis:\n"
    )

    # Stage 3: Recommended Tests
    recommended_tests = dspy.OutputField(
        desc="Medical tests to narrow diagnosis",
        type=Dict[str, Union[str, List[str], float]],
        prefix="Recommended Tests:\n"
    )

    # Stage 4: Initial Treatment Plan
    initial_treatment = dspy.OutputField(
        desc="Initial treatment recommendations",
        type=Dict[str, Union[str, List[str], Dict[str, str]]],
        prefix="Initial Treatment:\n"
    )

    # Stage 5: Urgency Assessment
    urgency_level = dspy.OutputField(
        desc="Medical urgency level (1-5)",
        type=int,
        prefix="Urgency Level: "
    )

    # Meta-information
    confidence_intervals = dspy.OutputField(
        desc="Confidence intervals for all probabilistic outputs",
        type=Dict[str, List[float]],
        prefix="Confidence Intervals:\n"
    )

    contraindications = dspy.OutputField(
        desc="Potential contraindications to consider",
        type=List[str],
        prefix="Contraindications:\n"
    )
```

### Legal Document Analysis

```python
class LegalDocumentAnalyzer(dspy.Signature):
    """Comprehensive legal document analysis."""

    document_text = dspy.InputField(
        desc="Full text of legal document",
        type=str
    )

    document_type = dspy.InputField(
        desc="Type of legal document (contract, patent, brief, etc.)",
        type=str
    )

    jurisdiction = dspy.InputField(
        desc="Legal jurisdiction governing the document",
        type=str
    )

    # Clauses extraction
    key_clauses = dspy.OutputField(
        desc="Important clauses with summaries",
        type=List[Dict[str, Union[str, int]]],
        prefix="Key Clauses:\n"
    )

    # Risk analysis
    legal_risks = dspy.OutputField(
        desc="Potential legal risks and liabilities",
        type=List[Dict[str, Union[str, int, float]]],
        prefix="Legal Risks:\n"
    )

    # Obligations and rights
    obligations = dspy.OutputField(
        desc="Obligations imposed by the document",
        type=List[Dict[str, str]],
        prefix="Obligations:\n"
    )

    rights = dspy.OutputField(
        desc="Rights granted by the document",
        type=List[Dict[str, str]],
        prefix="Rights:\n"
    )

    # Compliance check
    compliance_status = dspy.OutputField(
        desc="Compliance with relevant laws and regulations",
        type=Dict[str, Union[bool, str, List[str]]],
        prefix="Compliance Status:\n"
    )

    # Recommendations
    recommendations = dspy.OutputField(
        desc="Legal recommendations and next steps",
        type=Dict[str, Union[str, List[str], Dict[str, str]]],
        prefix="Recommendations:\n"
    )
```

## Performance-Optimized Signatures

### Caching and Memoization

```python
class CachedProcessor(dspy.Signature):
    """Signature with built-in caching awareness."""

    input_data = dspy.InputField(
        desc="Data to process",
        type=str
    )

    cache_key = dspy.InputField(
        desc="Cache key for this input",
        type=str,
        optional=True
    )

    use_cache = dspy.InputField(
        desc="Whether to use cached results",
        type=bool,
        default=True
    )

    processing_result = dspy.OutputField(
        desc="Result of processing",
        type=Union[str, dict]
    )

    cache_hit = dspy.OutputField(
        desc="Whether result came from cache",
        type=bool,
        prefix="Cache Hit: "
    )

    processing_time = dspy.OutputField(
        desc="Time taken for processing (ms)",
        type=float,
        prefix="Processing Time: "
    )
```

### Resource-Aware Signatures

```python
class ResourceAwareProcessor(dspy.Signature):
    """Signature that adapts based on available resources."""

    input_size = dspy.InputField(
        desc="Size of input data",
        type=int
    )

    available_memory = dspy.InputField(
        desc="Available memory in MB",
        type=int
    )

    time_constraint = dspy.InputField(
        desc="Maximum processing time in seconds",
        type=float,
        optional=True
    )

    quality_requirement = dspy.InputField(
        desc="Required quality level (1-10)",
        type=int,
        default=7
    )

    processing_strategy = dspy.OutputField(
        desc="Chosen processing strategy",
        type=str,
        prefix="Processing Strategy: "
    )

    optimized_result = dspy.OutputField(
        desc="Result optimized for given constraints",
        type=Union[str, dict]
    )

    resource_usage = dspy.OutputField(
        desc="Actual resource usage statistics",
        type=Dict[str, Union[int, float, str]],
        prefix="Resource Usage:\n"
    )

    quality_metrics = dspy.OutputField(
        desc="Quality metrics for the result",
        type=Dict[str, float],
        prefix="Quality Metrics:\n"
    )
```

## Testing and Debugging Advanced Signatures

### Self-Validating Signatures

```python
class ValidatingSignature(dspy.Signature):
    """Signature that includes validation logic."""

    input_data = dspy.InputField(
        desc="Data to validate and process",
        type=Union[str, dict, list]
    )

    validation_schema = dspy.InputField(
        desc="Schema for validation",
        type=dict
    )

    is_valid = dspy.OutputField(
        desc="Whether input data is valid",
        type=bool,
        prefix="Validation Status: "
    )

    validation_errors = dspy.OutputField(
        desc="List of validation errors",
        type=List[str],
        optional=True
    )

    sanitized_data = dspy.OutputField(
        desc="Sanitized version of input data",
        type=Union[str, dict, list],
        optional=True
    )

    processing_result = dspy.OutputField(
        desc="Result of processing valid data",
        type=Union[str, dict],
        optional=True
    )
```

## Best Practices for Advanced Signatures

### 1. Modular Design
Break complex signatures into smaller, reusable components.

### 2. Clear Documentation
Document complex signatures thoroughly with examples.

### 3. Error Handling
Include error outputs and validation in all complex signatures.

### 4. Performance Considerations
Design signatures with resource constraints in mind.

### 5. Testability
Make signatures easy to test with predictable outputs.

## Summary

Advanced signatures enable:
- **Complex workflows** through hierarchical composition
- **Dynamic behavior** based on input parameters
- **Multi-modal processing** for diverse data types
- **Performance optimization** with resource awareness
- **Production readiness** with validation and error handling

These patterns transform DSPy from a simple prompting tool into a comprehensive framework for building sophisticated AI applications.

## Key Takeaways

1. **Compose signatures** like you compose functions
2. **Design for flexibility** with dynamic and conditional structures
3. **Handle complexity** with hierarchical patterns
4. **Optimize for production** with caching and resource awareness
5. **Include validation** to ensure robust operation

## Further Reading

- [Next Section: Practical Examples](./05-practical-examples.md) - See advanced signatures in action
- [Chapter 3: Modules](../03-modules/) - Using advanced signatures with DSPy modules
- [Chapter 5: Optimizers](../05-optimizers/) - Optimizing complex signature chains
"""
Advanced DSPy Signatures Examples

This file demonstrates advanced DSPy signature patterns including:
- Hierarchical signatures
- Dynamic signature construction
- Self-referential patterns
- Multi-modal signatures
- Complex workflow orchestration
"""

import dspy
from typing import List, Dict, Optional, Union, Any, Type

# Example 1: Hierarchical Document Processing System
class BaseDocumentProcessor(dspy.Signature):
    """Base signature for document processing tasks."""

    document_content = dspy.InputField(
        desc="Content of the document to process",
        type=str,
        prefix="📄 Document:\n"
    )

    processing_confidence = dspy.OutputField(
        desc="Confidence in processing quality (0-1)",
        type=float,
        prefix="🎯 Confidence: "
    )

class TextExtractor(BaseDocumentProcessor):
    """Extract structured text from documents."""

    extracted_entities = dspy.OutputField(
        desc="Named entities with positions and types",
        type=List[Dict[str, Union[str, int]]],
        prefix="🏷️ Entities:\n"
    )

    key_phrases = dspy.OutputField(
        desc="Important phrases and their relevance",
        type=List[Dict[str, Union[str, float]]],
        prefix="💬 Key Phrases:\n"
    )

class RelationshipExtractor(BaseDocumentProcessor):
    """Extract relationships between entities."""

    relationships = dspy.OutputField(
        desc="Relationships between entities with types and confidence",
        type=List[Dict[str, Union[str, float]]],
        prefix="🔗 Relationships:\n"
    )

class ComprehensiveDocumentAnalyzer(dspy.Signature):
    """Complete document analysis using multiple extraction methods."""

    document_content = dspy.InputField(
        desc="Document to analyze",
        type=str,
        prefix="📊 Document:\n"
    )

    analysis_type = dspy.InputField(
        desc="Type of analysis to perform (basic, comprehensive, deep)",
        type=str,
        prefix="🔍 Analysis Type: "
    )

    domain_context = dspy.InputField(
        desc="Domain-specific context for analysis",
        type=str,
        optional=True,
        prefix="🏛️ Domain Context:\n"
    )

    # Nested analysis results
    text_analysis = dspy.OutputField(
        desc="Text extraction and analysis results",
        type=TextExtractor,
        prefix="📝 Text Analysis:\n"
    )

    relationship_analysis = dspy.OutputField(
        desc="Relationship extraction results",
        type=RelationshipExtractor,
        prefix="🔗 Relationship Analysis:\n"
    )

    document_summary = dspy.OutputField(
        desc="High-level document summary",
        type=str,
        prefix="📋 Summary:\n"
    )

    insights = dspy.OutputField(
        desc="Key insights and findings",
        type=List[Dict[str, Union[str, float, List[str]]]],
        prefix="💡 Insights:\n"
    )

# Example 2: Dynamic Signature Builder
class SignatureBuilder:
    """Build signatures dynamically based on requirements."""

    @staticmethod
    def create_pipeline_signature(stages: List[Dict[str, Any]]) -> Type[dspy.Signature]:
        """Create a signature from pipeline stage definitions."""

        class DynamicPipeline(dspy.Signature):
            """Dynamically created pipeline signature."""

            # Common input
            initial_data = dspy.InputField(
                desc="Initial data to process through pipeline",
                type=str,
                prefix="📥 Input Data:\n"
            )

            pipeline_config = dspy.InputField(
                desc="Pipeline configuration and parameters",
                type=Dict[str, Any],
                prefix="⚙️ Configuration:\n"
            )

            pass  # Additional fields will be added dynamically

        # Add stage-specific output fields
        for i, stage in enumerate(stages):
            field_name = f"stage_{i+1}_output"
            field_desc = stage.get("description", f"Output of stage {i+1}")
            field_type = stage.get("type", Union[str, dict])

            setattr(
                DynamicPipeline,
                field_name,
                dspy.OutputField(
                    desc=field_desc,
                    type=field_type,
                    prefix=f"🔧 Stage {i+1} Output:\n"
                )
            )

        # Add pipeline summary
        setattr(
            DynamicPipeline,
            "pipeline_summary",
            dspy.OutputField(
                desc="Summary of entire pipeline execution",
                type=Dict[str, Union[str, int, float]],
                prefix="📊 Pipeline Summary:\n"
            )
        )

        return DynamicPipeline

    @staticmethod
    def create_adaptive_signature(input_schema: Dict[str, type],
                                output_schema: Dict[str, type]) -> Type[dspy.Signature]:
        """Create a signature from input/output schemas."""

        class AdaptiveSignature(dspy.Signature):
            """Signature adapted from schemas."""
            pass

        # Add input fields
        for field_name, field_type in input_schema.items():
            setattr(
                AdaptiveSignature,
                field_name,
                dspy.InputField(
                    desc=f"Input field: {field_name}",
                    type=field_type,
                    prefix=f"📥 {field_name.replace('_', ' ').title()}:\n"
                )
            )

        # Add output fields
        for field_name, field_type in output_schema.items():
            setattr(
                AdaptiveSignature,
                field_name,
                dspy.OutputField(
                    desc=f"Output field: {field_name}",
                    type=field_type,
                    prefix=f"📤 {field_name.replace('_', ' ').title()}:\n"
                )
            )

        return AdaptiveSignature

# Example 3: Self-Referential Processing Pattern
class RecursiveDataProcessor(dspy.Signature):
    """Process hierarchical data structures recursively."""

    data_node = dspy.InputField(
        desc="Current data node to process",
        type=Union[str, dict],
        prefix="🔍 Current Node:\n"
    )

    node_type = dspy.InputField(
        desc="Type of data node (text, object, array, mixed)",
        type=str,
        prefix="📋 Node Type: "
    )

    processing_depth = dspy.InputField(
        desc="Current depth in recursive processing",
        type=int,
        default=0,
        prefix="📊 Depth: "
    )

    max_depth = dspy.InputField(
        desc="Maximum recursion depth",
        type=int,
        default=10,
        prefix="⏬ Max Depth: "
    )

    child_nodes = dspy.InputField(
        desc="Child nodes to process (if any)",
        type=List[Union[str, dict]],
        optional=True,
        prefix="👶 Child Nodes:\n"
    )

    # Processing results
    node_analysis = dspy.OutputField(
        desc="Analysis of current node",
        type=Dict[str, Union[str, int, float]],
        prefix="🔧 Node Analysis:\n"
    )

    processed_children = dspy.OutputField(
        desc="Results from processing child nodes",
        type=List[Dict[str, Any]],
        optional=True,
        prefix="👨‍👦‍👦 Processed Children:\n"
    )

    aggregated_insights = dspy.OutputField(
        desc="Insights aggregated from children",
        type=Dict[str, Union[str, List[str], float]],
        optional=True,
        prefix="📈 Aggregated Insights:\n"
    )

    processing_complete = dspy.OutputField(
        desc="Whether all processing is complete",
        type=bool,
        prefix="✅ Complete: "
    )

# Example 4: Multi-Modal Content Analyzer
class MultiModalAnalyzer(dspy.Signature):
    """Analyze content across multiple modalities."""

    text_content = dspy.InputField(
        desc="Text content to analyze",
        type=str,
        optional=True,
        prefix="📝 Text:\n"
    )

    image_description = dspy.InputField(
        desc="Description of image content",
        type=str,
        optional=True,
        prefix="🖼️ Image Description:\n"
    )

    audio_transcript = dspy.InputField(
        desc="Transcript of audio content",
        type=str,
        optional=True,
        prefix="🎵 Audio Transcript:\n"
    )

    video_summary = dspy.InputField(
        desc="Summary of video content",
        type=str,
        optional=True,
        prefix="🎥 Video Summary:\n"
    )

    metadata = dspy.InputField(
        desc="Metadata about all content",
        type=Dict[str, Union[str, int, float]],
        prefix="📋 Metadata:\n"
    )

    available_modalities = dspy.InputField(
        desc="List of available content modalities",
        type=List[str],
        prefix="📡 Available Modalities:\n"
    )

    # Multi-modal analysis outputs
    unified_understanding = dspy.OutputField(
        desc="Unified understanding across all modalities",
        type=str,
        prefix="🧠 Unified Understanding:\n"
    )

    cross_modal_correlations = dspy.OutputField(
        desc="Correlations and connections between modalities",
        type=List[Dict[str, Union[str, float, List[str]]]],
        prefix("🔗 Cross-Modal Correlations:\n")
    )

    modality_contributions = dspy.OutputField(
        desc="How each modality contributes to understanding",
        type=Dict[str, Union[float, str]],
        prefix="⚖️ Modality Contributions:\n"
    )

    synthesized_insights = dspy.OutputField(
        desc="Insights only possible by combining modalities",
        type=List[Dict[str, Union[str, float]]],
        prefix="💡 Synthesized Insights:\n"
    )

    confidence_scores = dspy.OutputField(
        desc="Confidence scores for each modality and overall",
        type=Dict[str, float],
        prefix="🎯 Confidence Scores:\n"
    )

# Example 5: Complex Workflow Orchestration
class WorkflowOrchestrator(dspy.Signature):
    """Orchestrate complex multi-stage workflows."""

    workflow_definition = dspy.InputField(
        desc="Definition of workflow stages and dependencies",
        type=Dict[str, Union[List, Dict, str]],
        prefix="🔄 Workflow Definition:\n"
    )

    input_data = dspy.InputField(
        desc="Initial data for workflow",
        type=Union[str, dict, List],
        prefix="📥 Input Data:\n"
    )

    context = dspy.InputField(
        desc="Additional context and parameters",
        type=Dict[str, Any],
        prefix="🏛️ Context:\n"
    )

    execution_mode = dspy.InputField(
        desc="Execution mode (sequential, parallel, mixed)",
        type=str,
        default="sequential",
        prefix="⚡ Execution Mode: "
    )

    # Workflow execution results
    stage_results = dspy.OutputField(
        desc="Results from each workflow stage",
        type=Dict[str, Union[Dict, List, str]],
        prefix="📊 Stage Results:\n"
    )

    workflow_summary = dspy.OutputField(
        desc="Summary of complete workflow execution",
        type=Dict[str, Union[str, int, float, bool]],
        prefix="📋 Workflow Summary:\n"
    )

    final_output = dspy.OutputField(
        desc="Final output after processing all stages",
        type=Union[str, dict, List],
        prefix="📤 Final Output:\n"
    )

    execution_metadata = dspy.OutputField(
        desc="Metadata about workflow execution",
        type=Dict[str, Union[int, float, List[str], Dict[str, Any]]],
        prefix="📈 Execution Metadata:\n"
    )

    errors_and_retries = dspy.OutputField(
        desc "Errors encountered and retry attempts",
        type=List[Dict[str, Union[str, int, bool]]],
        optional=True,
        prefix="❌ Errors & Retries:\n"
    )

# Example 6: Adaptive Learning System
class AdaptiveLearningSystem(dspy.Signature):
    """System that adapts based on feedback and performance."""

    task_input = dspy.InputField(
        desc="Input for the learning task",
        type=Union[str, dict],
        prefix="📥 Task Input:\n"
    )

    task_type = dspy.InputField(
        desc="Type of task to perform",
        type=str,
        prefix="🏷️ Task Type: "
    )

    performance_history = dspy.InputField(
        desc="History of performance on similar tasks",
        type=List[Dict[str, Union[str, float, int]]],
        optional=True,
        prefix="📈 Performance History:\n"
    )

    user_feedback = dspy.InputField(
        desc="Previous user feedback on similar outputs",
        type=List[Dict[str, Union[str, int]]],
        optional=True,
        prefix="💬 User Feedback:\n"
    )

    adaptation_parameters = dspy.InputField(
        desc="Parameters controlling adaptation behavior",
        type=Dict[str, Union[float, int, str]],
        prefix="⚙️ Adaptation Parameters:\n"
    )

    # Adaptive outputs
    adaptive_strategy = dspy.OutputField(
        desc="Chosen strategy based on context and history",
        type=Dict[str, Union[str, float, Dict[str, Any]]],
        prefix="🎯 Adaptive Strategy:\n"
    )

    task_result = dspy.OutputField(
        desc="Result of performing the task",
        type=Union[str, dict, List],
        prefix="✅ Task Result:\n"
    )

    confidence_adjustment = dspy.OutputField(
        desc="How confidence was adjusted based on history",
        type=Dict[str, Union[float, str, List[str]]],
        prefix="📊 Confidence Adjustment:\n"
    )

    learning_insights = dspy.OutputField(
        desc="Insights about what was learned from this task",
        type=List[Dict[str, Union[str, float]]],
        prefix="💡 Learning Insights:\n"
    )

    improvement_suggestions = dspy.OutputField(
        desc="Suggestions for improving future performance",
        type=List[Dict[str, Union[str, int, float]]],
        prefix="🔧 Improvement Suggestions:\n"
    )

def demonstrate_advanced_signatures():
    """Demonstrate advanced signature patterns."""

    print("Demonstrating Advanced DSPy Signatures...\n")
    print("=" * 60)

    # Example 1: Hierarchical Processing
    print("\n1. Hierarchical Document Analysis")
    print("-" * 40)
    doc_analyzer = dspy.Predict(ComprehensiveDocumentAnalyzer)
    result = doc_analyzer(
        document_content="Apple Inc. reported Q3 2024 earnings...",
        analysis_type="comprehensive",
        domain_context="technology company earnings report"
    )
    print(f"Document analyzed with {len(result.insights)} insights")

    # Example 2: Dynamic Signature Creation
    print("\n2. Dynamic Signature Creation")
    print("-" * 40)
    pipeline_stages = [
        {"description": "Text preprocessing and cleaning", "type": dict},
        {"description": "Entity extraction and identification", "type": dict},
        {"description": "Sentiment analysis and scoring", "type": dict}
    ]
    DynamicPipeline = SignatureBuilder.create_pipeline_signature(pipeline_stages)
    pipeline_processor = dspy.Predict(DynamicPipeline)

    # Create adaptive signature
    input_schema = {"text": str, "language": str}
    output_schema = {"summary": str, "keywords": List[str], "sentiment": float}
    AdaptiveProcessor = SignatureBuilder.create_adaptive_signature(input_schema, output_schema)
    print("Created dynamic and adaptive signatures")

    # Example 3: Recursive Processing
    print("\n3. Recursive Data Processing")
    print("-" * 40)
    recursive_processor = dspy.Predict(RecursiveDataProcessor)
    result = recursive_processor(
        data_node={"title": "Chapter 1", "content": "Introduction...", "subsections": [...]},
        node_type="object",
        processing_depth=0
    )
    print(f"Processed node at depth {result.node_analysis.get('depth', 0)}")

    # Example 4: Multi-Modal Analysis
    print("\n4. Multi-Modal Content Analysis")
    print("-" * 40)
    multimodal_analyzer = dspy.Predict(MultiModalAnalyzer)
    result = multimodal_analyzer(
        text_content="The product launch was successful...",
        image_description="Photo of product launch event with happy customers",
        available_modalities=["text", "image"]
    )
    print(f"Analyzed {len(result.cross_modal_correlations)} cross-modal correlations")

    # Example 5: Workflow Orchestration
    print("\n5. Workflow Orchestration")
    print("-" * 40)
    workflow_def = {
        "stages": ["extract", "analyze", "summarize"],
        "dependencies": {"analyze": ["extract"], "summarize": ["analyze"]}
    }
    orchestrator = dspy.Predict(WorkflowOrchestrator)
    result = orchestrator(
        workflow_definition=workflow_def,
        input_data="Sample document text...",
        execution_mode="sequential"
    )
    print(f"Completed workflow with {len(result.stage_results)} stages")

    # Example 6: Adaptive Learning
    print("\n6. Adaptive Learning System")
    print("-" * 40)
    adaptive_system = dspy.Predict(AdaptiveLearningSystem)
    result = adaptive_system(
        task_input="Generate a summary of this article...",
        task_type="summarization",
        performance_history=[{"task": "summarization", "score": 0.8}],
        adaptation_parameters={"learning_rate": 0.1, "exploration": 0.2}
    )
    print(f"Adapted strategy: {result.adaptive_strategy.get('name', 'unknown')}")

    print("\n" + "=" * 60)
    print("Advanced signature examples completed!")

if __name__ == "__main__":
    demonstrate_advanced_signatures()
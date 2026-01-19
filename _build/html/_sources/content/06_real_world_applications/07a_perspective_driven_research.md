# Perspective-Driven Research for Article Generation

## Prerequisites

- **Chapter 3**: Modules - Understanding of DSPy modules
- **Chapter 6**: RAG Systems - Retrieval-augmented generation concepts
- **Previous Sections**: Information Retrieval, Document Q&A
- **Required Knowledge**: Basic understanding of research methodologies
- **Difficulty Level**: Advanced
- **Estimated Reading Time**: 45 minutes

## Learning Objectives

By the end of this section, you will:
- Understand perspective-guided questioning for comprehensive research
- Learn to simulate the human research process using AI
- Master multi-perspective information gathering strategies
- Build systems that explore topics from multiple viewpoints
- Create research foundations for long-form article generation

## Introduction: The Need for Perspective-Driven Research

When writing comprehensive articles like Wikipedia entries, single-perspective research often leads to biased or incomplete coverage. Perspective-driven research simulates how human researchers approach topics by exploring them from multiple angles, ensuring comprehensive, balanced, and well-rounded information gathering.

### What is Perspective-Driven Research?

Perspective-driven research is a systematic approach to information gathering that:
- **Simulates Multiple Viewpoints**: Considers topics from different angles (technical, historical, social, economic, etc.)
- **Ensures Comprehensive Coverage**: Reduces blind spots in information gathering
- **Promotes Balance**: Helps avoid bias by considering multiple perspectives
- **Mimics Human Research**: Follows the natural curiosity-driven exploration patterns of human researchers

## Core Components of Perspective-Driven Research

### 1. Perspective Definition

First, we define the perspectives from which to explore a topic:

```python
import dspy
from typing import List, Dict, Any

class PerspectiveGenerator(dspy.Module):
    """Generate relevant perspectives for researching a topic."""

    def __init__(self):
        super().__init__()
        self.generate_perspectives = dspy.ChainOfThought(
            "topic -> perspectives, rationale"
        )

    def forward(self, topic: str) -> dspy.Prediction:
        """
        Generate diverse perspectives for researching a topic.

        Args:
            topic: The topic to be researched

        Returns:
            Prediction containing perspectives and rationale
        """
        prediction = self.generate_perspectives(topic=topic)

        return dspy.Prediction(
            perspectives=prediction.perspectives,
            rationale=prediction.rationale
        )

# Example usage
perspective_gen = PerspectiveGenerator()
result = perspective_gen(topic="Artificial Intelligence in Healthcare")

print("Generated Perspectives:")
print(result.perspectives)
print("\nRationale:")
print(result.rationale)
```

### 2. Perspective-Guided Questioning

For each perspective, generate specific questions:

```python
class PerspectiveQuestionGenerator(dspy.Module):
    """Generate questions from specific perspectives."""

    def __init__(self):
        super().__init__()
        self.generate_questions = dspy.ChainOfThought(
            "topic, perspective -> focused_questions"
        )

    def forward(self, topic: str, perspective: str) -> dspy.Prediction:
        """
        Generate focused questions from a specific perspective.

        Args:
            topic: The main topic
            perspective: The perspective from which to view the topic

        Returns:
            Prediction containing focused questions
        """
        prediction = self.generate_questions(
            topic=topic,
            perspective=perspective
        )

        return dspy.Prediction(
            focused_questions=prediction.focused_questions,
            perspective=perspective
        )

# Example: Generate questions from ethical perspective
question_gen = PerspectiveQuestionGenerator()
ethical_questions = question_gen(
    topic="Gene editing",
    perspective="Ethical considerations"
)

print("Ethical Questions about Gene Editing:")
print(ethical_questions.focused_questions)
```

### 3. Multi-Perspective Information Retrieval

Retrieve information for each perspective:

```python
class PerspectiveRetriever(dspy.Module):
    """Retrieve information from multiple perspectives."""

    def __init__(self, k: int = 5):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=k)
        self.filter_by_perspective = dspy.Predict(
            "documents, perspective -> relevant_documents"
        )

    def forward(self, questions: List[str], perspective: str) -> dspy.Prediction:
        """
        Retrieve and filter documents for a specific perspective.

        Args:
            questions: List of questions from the perspective
            perspective: The current perspective

        Returns:
            Filtered relevant documents
        """
        all_documents = []

        # Retrieve for each question
        for question in questions:
            retrieved = self.retrieve(question=question)
            all_documents.extend(retrieved.passages)

        # Filter and rank by perspective relevance
        filtered = self.filter_by_perspective(
            documents="\n\n".join(all_documents),
            perspective=perspective
        )

        return dspy.Prediction(
            relevant_documents=filtered.relevant_documents.split("\n\n"),
            perspective=perspective,
            total_retrieved=len(all_documents)
        )
```

## Building the Complete Perspective-Driven Research System

### Core Research Pipeline

```python
class PerspectiveDrivenResearch(dspy.Module):
    """Complete perspective-driven research system."""

    def __init__(self, perspectives_per_topic: int = 5, questions_per_perspective: int = 3):
        super().__init__()
        self.perspectives_per_topic = perspectives_per_topic
        self.questions_per_perspective = questions_per_perspective

        # Sub-modules
        self.perspective_generator = PerspectiveGenerator()
        self.question_generator = PerspectiveQuestionGenerator()
        self.retriever = PerspectiveRetriever(k=8)

        # Synthesis module
        self.synthesize_perspective = dspy.ChainOfThought(
            "perspective, documents -> perspective_summary"
        )

    def forward(self, topic: str) -> dspy.Prediction:
        """
        Perform comprehensive perspective-driven research.

        Args:
            topic: The topic to research

        Returns:
            Comprehensive research from multiple perspectives
        """
        # Step 1: Generate perspectives
        perspectives_result = self.perspective_generator(topic=topic)
        perspectives = self._parse_perspectives(perspectives_result.perspectives)

        # Step 2: Generate questions and retrieve for each perspective
        research_results = []

        for perspective in perspectives[:self.perspectives_per_topic]:
            # Generate questions
            questions_result = self.question_generator(
                topic=topic,
                perspective=perspective
            )
            questions = self._parse_questions(questions_result.focused_questions)

            # Retrieve information
            retrieval_result = self.retriever(
                questions=questions[:self.questions_per_perspective],
                perspective=perspective
            )

            # Synthesize perspective
            synthesis = self.synthesize_perspective(
                perspective=perspective,
                documents="\n\n".join(retrieval_result.relevant_documents)
            )

            research_results.append({
                "perspective": perspective,
                "questions": questions[:self.questions_per_perspective],
                "documents": retrieval_result.relevant_documents,
                "summary": synthesis.perspective_summary,
                "num_documents": len(retrieval_result.relevant_documents)
            })

        # Generate overall research summary
        overall_summary = self._generate_overall_summary(topic, research_results)

        return dspy.Prediction(
            topic=topic,
            perspectives_researched=[r["perspective"] for r in research_results],
            research_results=research_results,
            overall_summary=overall_summary,
            total_documents=sum(r["num_documents"] for r in research_results)
        )

    def _parse_perspectives(self, perspectives_text: str) -> List[str]:
        """Parse perspectives from generated text."""
        lines = perspectives_text.strip().split('\n')
        perspectives = []
        for line in lines:
            if line.strip() and (line.strip().startswith('-') or '.' in line[:10]):
                perspectives.append(line.strip().lstrip('- ').strip())
        return perspectives[:10]  # Limit to 10 perspectives

    def _parse_questions(self, questions_text: str) -> List[str]:
        """Parse questions from generated text."""
        questions = []
        lines = questions_text.strip().split('\n')
        for line in lines:
            if '?' in line and (line.strip().startswith('-') or line.strip().startswith('•')):
                questions.append(line.strip().lstrip('- •').strip())
        return questions[:10]  # Limit to 10 questions

    def _generate_overall_summary(self, topic: str, research_results: List[Dict]) -> str:
        """Generate an overall summary of all perspectives."""
        summaries = [f"{r['perspective']}: {r['summary']}" for r in research_results]
        combined = "\n\n".join(summaries)

        # Use ChainOfThought for synthesis
        synthesizer = dspy.ChainOfThought("topic, perspective_summaries -> overall_summary")
        result = synthesizer(
            topic=topic,
            perspective_summaries=combined
        )

        return result.overall_summary
```

### Advanced Features

#### 1. Dynamic Perspective Expansion

```python
class DynamicPerspectiveResearch(PerspectiveDrivenResearch):
    """Research system that dynamically adds perspectives."""

    def __init__(self):
        super().__init__()
        self.identify_gaps = dspy.ChainOfThought(
            "topic, current_perspectives -> missing_perspectives"
        )
        self.gap_questions = dspy.Predict(
            "topic, gap_perspective -> priority_questions"
        )

    def forward(self, topic: str, max_iterations: int = 3) -> dspy.Prediction:
        """Research with dynamic perspective addition."""
        current_result = super().forward(topic=topic)

        for iteration in range(max_iterations):
            # Identify gaps
            gap_analysis = self.identify_gaps(
                topic=topic,
                current_perspectives=", ".join(current_result.perspectives_researched)
            )

            missing = self._parse_perspectives(gap_analysis.missing_perspectives)

            if not missing:
                break  # No more gaps identified

            # Research top missing perspective
            new_perspective = missing[0]
            questions_result = self.gap_questions(
                topic=topic,
                gap_perspective=new_perspective
            )

            questions = self._parse_questions(questions_result.priority_questions)
            retrieval_result = self.retriever(questions=questions, perspective=new_perspective)

            # Add to results
            synthesis = self.synthesize_perspective(
                perspective=new_perspective,
                documents="\n\n".join(retrieval_result.relevant_documents)
            )

            current_result.research_results.append({
                "perspective": new_perspective,
                "questions": questions,
                "documents": retrieval_result.relevant_documents,
                "summary": synthesis.perspective_summary,
                "num_documents": len(retrieval_result.relevant_documents)
            })

            current_result.perspectives_researched.append(new_perspective)
            current_result.total_documents += len(retrieval_result.relevant_documents)

        return current_result
```

#### 2. Cross-Perspective Synthesis

```python
class CrossPerspectiveSynthesizer(dspy.Module):
    """Synthesize insights across different perspectives."""

    def __init__(self):
        super().__init__()
        self.find_connections = dspy.ChainOfThought(
            "perspective1, perspective2 -> connections, conflicts"
        )
        self.resolve_conflicts = dspy.Predict(
            "conflicts, supporting_evidence -> resolutions"
        )
        self.create_synthesis = dspy.ChainOfThought(
            "all_connections, resolved_conflicts -> integrated_understanding"
        )

    def forward(self, research_results: List[Dict]) -> dspy.Prediction:
        """Synthesize across all perspectives."""
        all_connections = []
        all_conflicts = []

        # Compare each pair of perspectives
        for i, result1 in enumerate(research_results):
            for result2 in research_results[i+1:]:
                comparison = self.find_connections(
                    perspective1=f"{result1['perspective']}: {result1['summary']}",
                    perspective2=f"{result2['perspective']}: {result2['summary']}"
                )

                all_connections.append({
                    "perspective1": result1['perspective'],
                    "perspective2": result2['perspective'],
                    "connections": comparison.connections,
                    "conflicts": comparison.conflicts
                })

                if comparison.conflicts:
                    all_conflicts.append(comparison.conflicts)

        # Resolve conflicts
        resolved_conflicts = []
        for conflict in all_conflicts:
            resolution = self.resolve_conflicts(
                conflicts=conflict,
                supporting_evidence=self._gather_evidence(research_results, conflict)
            )
            resolved_conflicts.append({
                "conflict": conflict,
                "resolution": resolution.resolutions
            })

        # Create integrated understanding
        synthesis = self.create_synthesis(
            all_connections=str(all_connections),
            resolved_conflicts=str(resolved_conflicts)
        )

        return dspy.Prediction(
            cross_perspective_connections=all_connections,
            resolved_conflicts=resolved_conflicts,
            integrated_understanding=synthesis.integrated_understanding
        )

    def _gather_evidence(self, research_results: List[Dict], conflict: str) -> str:
        """Gather evidence related to a conflict."""
        evidence = []
        for result in research_results:
            relevant_docs = [
                doc for doc in result['documents']
                if any(word in doc.lower() for word in conflict.lower().split()[:5])
            ]
            if relevant_docs:
                evidence.extend(relevant_docs[:2])
        return "\n\n".join(evidence)
```

## Practical Implementation Example

### Complete Research Workflow

```python
# Initialize the research system
research_system = DynamicPerspectiveResearch()

# Perform comprehensive research
topic = "The Impact of Social Media on Mental Health"
research_result = research_system(topic=topic)

# Display results
print(f"\n=== Research Results for: {topic} ===\n")
print(f"Total Perspectives Researched: {len(research_result.perspectives_researched)}")
print(f"Total Documents Retrieved: {research_result.total_documents}\n")

# Show perspective summaries
for result in research_result.research_results:
    print(f"\n--- {result['perspective']} ---")
    print(f"Questions Asked: {len(result['questions'])}")
    print(f"Documents Found: {result['num_documents']}")
    print(f"Summary: {result['summary'][:200]}...\n")

# Show overall summary
print("\n=== Overall Research Summary ===")
print(research_result.overall_summary)

# Cross-perspective synthesis
synthesizer = CrossPerspectiveSynthesizer()
synthesis = synthesizer(research_result.research_results)

print("\n=== Cross-Perspective Insights ===")
print(f"Connections Found: {len(synthesis.cross_perspective_connections)}")
print(f"Conflicts Resolved: {len(synthesis.resolved_conflicts)}")
print("\nIntegrated Understanding:")
print(synthesis.integrated_understanding)
```

## Integration with Article Generation

The research results can be directly fed into article generation systems:

```python
class ResearchToArticleConverter(dspy.Module):
    """Convert research results into article-ready structure."""

    def __init__(self):
        super().__init__()
        self.create_outline = dspy.ChainOfThought(
            "topic, research_summary -> article_outline"
        )
        self.assign_sections = dspy.Predict(
            "outline, perspectives -> section_perspectives"
        )

    def forward(self, research_result: dspy.Prediction) -> dspy.Prediction:
        """Convert research to article structure."""
        # Create outline from research
        outline_result = self.create_outline(
            topic=research_result.topic,
            research_summary=research_result.overall_summary
        )

        # Assign perspectives to sections
        section_assignment = self.assign_sections(
            outline=outline_result.article_outline,
            perspectives=", ".join(research_result.perspectives_researched)
        )

        return dspy.Prediction(
            topic=research_result.topic,
            outline=outline_result.article_outline,
            section_perspectives=section_assignment.section_perspectives,
            research_data=research_result.research_results
        )
```

## Best Practices for Perspective-Driven Research

### 1. Perspective Selection
- Start with broad categories (technical, social, ethical, economic)
- Consider topic-specific relevant perspectives
- Include both supporting and critical viewpoints
- Balance between breadth and depth

### 2. Question Generation
- Ensure questions are specific to each perspective
- Include both factual and analytical questions
- Generate questions at different abstraction levels
- Avoid redundancy across perspectives

### 3. Information Retrieval
- Use appropriate k values based on topic complexity
- Implement diversity in retrieved documents
- Consider temporal aspects (recent vs historical)
- Filter for quality and relevance

### 4. Synthesis Quality
- Maintain perspective integrity during synthesis
- Clearly identify consensus and disagreements
- Provide evidence for all claims
- Preserve nuance in complex topics

## Evaluation Metrics for Perspective-Driven Research

```python
def perspective_coverage_metric(example, pred, trace=None):
    """Evaluate how well different perspectives are covered."""
    expected_perspectives = set(example.perspectives)
    actual_perspectives = set(pred.perspectives_researched)

    coverage = len(expected_perspectives & actual_perspectives) / len(expected_perspectives)

    if trace is not None:
        return coverage >= 0.8

    return coverage

def question_relevance_metric(example, pred, trace=None):
    """Evaluate relevance of generated questions."""
    # In practice, this would use LLM or human evaluation
    # Simplified version checking for perspective alignment
    total_relevance = 0
    total_questions = 0

    for result in pred.research_results:
        perspective = result['perspective']
        for question in result['questions']:
            # Simple check if perspective keywords appear in question
            if any(word in question.lower() for word in perspective.lower().split()):
                total_relevance += 1
            total_questions += 1

    if total_questions == 0:
        return 0.0

    return total_relevance / total_questions

def information_diversity_metric(example, pred, trace=None):
    """Evaluate diversity of retrieved information."""
    all_docs = []
    for result in pred.research_results:
        all_docs.extend(result['documents'])

    # Simplified diversity calculation
    unique_sources = set()
    for doc in all_docs:
        # Extract source indicator (simplified)
        if 'source:' in doc.lower():
            source = doc.split('source:')[1].split()[0]
            unique_sources.add(source)

    # Reward having diverse sources
    return min(1.0, len(unique_sources) / 10.0)
```

## Summary

Perspective-driven research is a powerful approach for comprehensive information gathering that:

1. **Ensures Comprehensive Coverage** by exploring topics from multiple angles
2. **Reduces Bias** through systematic inclusion of diverse viewpoints
3. **Improves Article Quality** by providing balanced, well-rounded research
4. **Mimics Human Research** patterns for natural exploration
5. **Integrates Seamlessly** with article generation workflows

### Key Takeaways

1. **Multiple Perspectives** are essential for comprehensive research
2. **Guided Questioning** ensures systematic exploration
3. **Dynamic Expansion** helps cover unexpected angles
4. **Cross-Perspective Synthesis** reveals connections and conflicts
5. **Quality Evaluation** ensures research effectiveness

## Next Steps

- [Long-form Article Generation](./08-long-form-generation.md) - Use research to generate complete articles
- [STORM Writing Assistant](../08-case-studies/05-storm-writing-assistant.md) - Complete writing system case study
- [Advanced Evaluation](../../04-evaluation/05-best-practices.md) - Sophisticated evaluation techniques

## Further Reading

- [Research Methodology in the Digital Age](https://example.com/digital-research)
- [Multi-perspective Analysis Frameworks](https://example.com/perspective-analysis)
- [Information Synthesis Techniques](https://example.com/synthesis-methods)
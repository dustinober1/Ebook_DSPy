# Case Study 5: STORM - AI-Powered Writing Assistant for Wikipedia-like Articles

## Overview

STORM (Synthesis of Topic Outlines through Retrieval and Multi-perspective questioning) is a sophisticated AI writing assistant that helps users create comprehensive, well-researched articles from scratch. Inspired by the research paper "Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models," this case study demonstrates how DSPy can be used to build a complete writing system that simulates human research and writing processes.

## Problem Definition

### The Challenge

Writing comprehensive, encyclopedic articles requires:
1. **Thorough Research**: Gathering information from multiple perspectives
2. **Structured Organization**: Creating logical outlines from scattered information
3. **Coherent Writing**: Maintaining consistency across thousands of words
4. **Factual Accuracy**: Ensuring all claims are supported by evidence
5. **Citation Management**: Properly attributing sources

### Key Requirements

1. **Two-Stage Process**: Pre-writing (research and outlining) and writing stages
2. **Multi-perspective Research**: Comprehensive coverage from different angles
3. **Iterative Refinement**: Continuous improvement of content quality
4. **Human-AI Collaboration**: Assist rather than replace human writers
5. **Scalability**: Handle topics of varying complexity

## System Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                    STORM Writing System                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   User Input    │───▶│   Topic         │                │
│  │   (Topic)       │    │   Analysis      │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                       │                         │
│           ▼                       ▼                         │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   Perspective   │    │   Question      │                │
│  │   Generator     │───▶│   Generator     │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                       │                         │
│           ▼                       ▼                         │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   Information   │    │   Outline       │                │
│  │   Retrieval     │───▶│   Generator     │                │
│  └─────────────────┘    └─────────────────┘                │
│                                   │                         │
│                                   ▼                         │
│  ┌─────────────────────────────────────────────────┐       │
│  │              Pre-writing Stage                   │       │
│  └─────────────────────────────────────────────────┘       │
│                                   │                         │
│                                   ▼                         │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   Section       │    │   Citation      │                │
│  │   Generator     │───▶│   Manager       │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                       │                         │
│           ▼                       ▼                         │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   Coherence     │    │   Quality       │                │
│  │   Checker       │───▶   Assurance     │                │
│  └─────────────────┘    └─────────────────┘                │
│                                   │                         │
│                                   ▼                         │
│  ┌─────────────────────────────────────────────────┐       │
│  │               Writing Stage                      │       │
│  └─────────────────────────────────────────────────┘       │
│                                   │                         │
│                                   ▼                         │
│  ┌─────────────────────────────────────────────────┐       │
│  │              Final Article                       │       │
│  └─────────────────────────────────────────────────┘       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Implementation with DSPy

### Core System Components

```python
import dspy
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Helper classes for STORM implementation
class ParallelProcessor:
    """Simple parallel processing helper."""
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

    def process_parallel(self, tasks: List, function):
        """Process tasks in parallel."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(function, task) for task in tasks]
            return [future.result() for future in as_completed(futures)]

class BatchRetriever:
    """Batch retrieval helper for efficient document retrieval."""
    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size

    def retrieve_batch(self, queries: List[str]):
        """Retrieve documents for multiple queries."""
        # This would integrate with dspy.Retrieve or similar
        results = []
        for query in queries:
            # Simulate retrieval
            results.append([f"Document for {query}"])
        return results

class RateLimiter:
    """Simple rate limiter for API calls."""
    def __init__(self, requests_per_minute: int = 10):
        self.requests_per_minute = requests_per_minute
        self.requests = []

    async def acquire(self, user_id: str):
        """Acquire rate limit slot."""
        # Simple implementation - in production use proper rate limiting
        await asyncio.sleep(60 / self.requests_per_minute)

@dataclass
class StormConfig:
    """Configuration for STORM system."""
    max_perspectives: int = 5
    questions_per_perspective: int = 4
    retrieval_documents_per_query: int = 8
    max_outline_sections: int = 10
    words_per_section: int = 500
    citation_style: str = "wikipedia"

class StormWritingAssistant(dspy.Module):
    """Complete STORM writing assistant implementation."""

    def __init__(self, config: Optional[StormConfig] = None):
        super().__init__()
        self.config = config or StormConfig()

        # Stage 1: Pre-writing components
        self.perspective_generator = PerspectiveDrivenResearch()
        self.outline_generator = ArticleOutlineGenerator()
        self.research_synthesizer = ResearchSynthesizer()

        # Stage 2: Writing components
        self.section_writer = LongFormArticleGenerator()
        self.citation_manager = CitationManager()
        self.quality_checker = ArticleQA()

        # Human-AI interaction
        self.human_review_prompter = HumanReviewInterface()

    def forward(self,
               topic: str,
               human_feedback: Optional[Dict] = None) -> dspy.Prediction:
        """
        Generate a complete article using STORM methodology.

        Args:
            topic: The article topic
            human_feedback: Optional feedback from human reviewer

        Returns:
            Complete article with metadata
        """
        # Stage 1: Pre-writing Phase
        prewriting_result = self._prewriting_phase(topic)

        # Stage 2: Writing Phase
        writing_result = self._writing_phase(
            topic=topic,
            outline=prewriting_result.outline,
            research_data=prewriting_result.research_synthesis
        )

        # Stage 3: Quality Assurance
        qa_result = self._quality_assurance(
            article=writing_result.article,
            research_data=prewriting_result.research_synthesis
        )

        # Stage 4: Human Review Integration (if feedback provided)
        if human_feedback:
            final_article = self._incorporate_feedback(
                article=writing_result.article,
                feedback=human_feedback
            )
        else:
            final_article = writing_result.article

        return dspy.Prediction(
            topic=topic,
            article=final_article,
            outline=prewriting_result.outline,
            research_perspectives=prewriting_result.perspectives,
            quality_score=qa_result.overall_quality,
            total_citations=writing_result.total_citations,
            word_count=writing_result.total_word_count,
            human_feedback_applied=bool(human_feedback)
        )

    def _prewriting_phase(self, topic: str) -> dspy.Prediction:
        """Execute the pre-writing phase (research and outlining)."""
        print(f"\n=== Pre-writing Phase for: {topic} ===")

        # Step 1: Multi-perspective research
        print("1. Conducting multi-perspective research...")
        research = self.perspective_generator(
            topic=topic,
            max_perspectives=self.config.max_perspectives,
            questions_per_perspective=self.config.questions_per_perspective
        )

        # Step 2: Synthesize research findings
        print("2. Synthesizing research findings...")
        synthesis = self.research_synthesizer(
            topic=topic,
            research_data=research.research_results
        )

        # Step 3: Generate structured outline
        print("3. Generating structured outline...")
        outline = self.outline_generator(
            topic=topic,
            research_data=synthesis.synthesized_data,
            constraints={
                'word_count_target': self.config.words_per_section * self.config.max_outline_sections,
                'intended_audience': 'general',
                'complexity': 'medium'
            }
        )

        return dspy.Prediction(
            perspectives=research.perspectives_researched,
            research_synthesis=synthesis.synthesized_data,
            outline=outline.outline
        )

    def _writing_phase(self,
                      topic: str,
                      outline: List[Dict],
                      research_data: Dict) -> dspy.Prediction:
        """Execute the writing phase."""
        print(f"\n=== Writing Phase ===")

        # Generate the complete article
        article_result = self.section_writer(
            topic=topic,
            outline=outline,
            research_data=research_data
        )

        print(f"Generated article with {article_result.total_word_count} words")
        print(f"Included {article_result.total_citations} citations")

        return article_result

    def _quality_assurance(self,
                          article: str,
                          research_data: Dict) -> Dict:
        """Perform quality assurance on generated article."""
        print("\n=== Quality Assurance ===")

        qa_result = self.quality_checker.validate_article(
            article=article,
            research_data=research_data,
            outline=[]  # Would pass outline if available
        )

        print(f"Quality Score: {qa_result['overall_quality']:.2f}")
        print(f"Factual Claims Verified: {sum(1 for fc in qa_result['fact_check'] if fc['is_factual'])}/{len(qa_result['fact_check'])}")

        return qa_result

    def _incorporate_feedback(self,
                            article: str,
                            feedback: Dict) -> str:
        """Incorporate human feedback into the article."""
        print("\n=== Incorporating Human Feedback ===")

        feedback_processor = dspy.ChainOfThought(
            "article, feedback -> revised_article"
        )

        result = feedback_processor(
            article=article,
            feedback=str(feedback)
        )

        print("Applied human feedback to article")
        return result.revised_article


class ResearchSynthesizer(dspy.Module):
    """Synthesizes research from multiple perspectives."""

    def __init__(self):
        super().__init__()
        self.identify_connections = dspy.ChainOfThought(
            "perspective_research -> connections, contradictions"
        )
        self.resolve_conflicts = dspy.Predict(
            "contradictions, evidence -> resolutions"
        )
        self.create_synthesis = dspy.ChainOfThought(
            "topic, all_perspectives, connections, resolutions -> synthesized_data"
        )

    def forward(self, topic: str, research_data: Dict) -> dspy.Prediction:
        """Synthesize research from multiple perspectives."""
        # Find connections between perspectives
        connections = self.identify_connections(
            perspective_research=str(research_data)
        )

        # Resolve contradictions
        if connections.contradictions:
            resolutions = self.resolve_conflicts(
                contradictions=connections.contradictions,
                evidence=str(research_data)
            )
        else:
            resolutions = dspy.Prediction(resolutions="No contradictions found")

        # Create final synthesis
        synthesis = self.create_synthesis(
            topic=topic,
            all_perspectives=str(research_data),
            connections=connections.connections,
            resolutions=resolutions.resolutions
        )

        return dspy.Prediction(
            synthesized_data=self._parse_synthesis(synthesis.synthesized_data),
            key_connections=connections.connections,
            conflicts_resolved=len(connections.contradictions) > 0
        )

    def _parse_synthesis(self, synthesis_text: str) -> Dict:
        """Parse synthesis into structured format."""
        # Simplified parsing - in practice would be more sophisticated
        return {
            'unified_findings': synthesis_text,
            'consensus_points': [],
            'open_questions': []
        }


class HumanReviewInterface(dspy.Module):
    """Interface for human review and feedback integration."""

    def __init__(self):
        super().__init__()
        self.generate_review_questions = dspy.Predict(
            "article, topic -> review_questions"
        )
        self.summarize_feedback = dspy.Predict(
            "human_responses -> feedback_summary"
        )

    def generate_review_prompts(self, article: str, topic: str) -> Dict:
        """Generate prompts for human review."""
        questions = self.generate_review_questions(
            article=article[:2000],  # First 2000 chars for context
            topic=topic
        )

        return {
            'accuracy_questions': [
                "Are all factual claims accurate?",
                "Are citations appropriate and correctly formatted?",
                "Is the information up-to-date?"
            ],
            'completeness_questions': [
                "Are there any important aspects missing?",
                "Should any sections be expanded?",
                "Is the coverage balanced?"
            ],
            'readability_questions': [
                "Is the article well-structured?",
                "Are transitions between sections smooth?",
                "Is the language clear and appropriate?"
            ],
            'ai_generated_questions': self._parse_questions(questions.review_questions)
        }

    def process_human_feedback(self,
                             human_responses: Dict) -> Dict:
        """Process and structure human feedback."""
        feedback = self.summarize_feedback(
            human_responses=str(human_responses)
        )

        return {
            'feedback_summary': feedback.feedback_summary,
            'priority_issues': self._identify_priorities(human_responses),
            'actionable_items': self._extract_actions(human_responses)
        }

    def _parse_questions(self, questions_text: str) -> List[str]:
        """Parse questions from generated text."""
        return [q.strip() for q in questions_text.split('\n') if q.strip() and '?' in q]

    def _identify_priorities(self, responses: Dict) -> List[str]:
        """Identify high-priority issues from feedback."""
        priorities = []
        for question, response in responses.items():
            if 'no' in response.lower() or 'missing' in response.lower():
                priorities.append(question)
        return priorities

    def _extract_actions(self, responses: Dict) -> List[str]:
        """Extract actionable items from feedback."""
        actions = []
        for question, response in responses.items():
            if any(action in response.lower() for action in ['add', 'remove', 'expand', 'fix']):
                actions.append(f"For '{question}': {response}")
        return actions
```

## Advanced Features

### 1. Adaptive Research Depth

```python
class AdaptiveResearchDepth(dspy.Module):
    """Adjusts research depth based on topic complexity."""

    def __init__(self):
        super().__init__()
        self.assess_complexity = dspy.ChainOfThought(
            "topic, initial_research -> complexity_level, research_depth_needed"
        )
        self.adjust_questions = dspy.Predict(
            "base_questions, complexity_level -> adjusted_questions"
        )

    def forward(self, topic: str) -> Dict:
        """Determine optimal research depth for topic."""
        # Get initial assessment
        complexity = self.assess_complexity(
            topic=topic,
            initial_research=""  # Would include preliminary research
        )

        # Adjust parameters based on complexity
        depth_params = {
            'complexity': complexity.complexity_level,
            'perspectives_needed': min(8, 2 + int(complexity.research_depth_needed) * 2),
            'questions_per_perspective': min(6, 2 + int(complexity.research_depth_needed)),
            'document_limit': min(15, 5 + int(complexity.research_depth_needed) * 3)
        }

        return depth_params
```

### 2. Dynamic Citation Strategy

```python
class DynamicCitationStrategy(dspy.Module):
    """Adapts citation strategy based on content type."""

    def __init__(self):
        super().__init__()
        self.classify_content = dspy.Predict(
            "content -> content_type, citation_density"
        )
        self.select_citation_style = dspy.Predict(
            "content_type, audience -> optimal_citation_style"
        )

    def get_citation_strategy(self, content: str, audience: str = "general") -> Dict:
        """Determine optimal citation strategy."""
        classification = self.classify_content(content=content)
        style = self.select_citation_style(
            content_type=classification.content_type,
            audience=audience
        )

        return {
            'style': style.optimal_citation_style,
            'density': classification.citation_density,
            'placement_rules': self._get_placement_rules(classification.content_type),
            'verification_level': 'high' if 'controversial' in content.lower() else 'medium'
        }

    def _get_placement_rules(self, content_type: str) -> List[str]:
        """Get citation placement rules for content type."""
        rules = {
            'factual': ["cite every statistic", "cite every direct quote"],
            'opinion': ["cite supporting arguments", "cite counterarguments"],
            'historical': ["cite primary sources", "cite scholarly interpretations"],
            'technical': ["cite specifications", "cite research papers"]
        }
        return rules.get(content_type, ["cite as needed"])
```

## Example Implementation

### Complete STORM Workflow

```python
# Initialize STORM with custom configuration
config = StormConfig(
    max_perspectives=6,
    questions_per_perspective=5,
    retrieval_documents_per_query=10,
    max_outline_sections=12,
    words_per_section=600
)

storm = StormWritingAssistant(config)

# Example: Generate an article
topic = "The Impact of Quantum Computing on Cryptography"

print(f"\n{'='*60}")
print(f"STORM Writing Assistant")
print(f"Generating Article: {topic}")
print(f"{'='*60}\n")

# Generate the article
result = storm(topic=topic)

# Display results
print(f"\n{'='*60}")
print(f"ARTICLE GENERATED SUCCESSFULLY")
print(f"{'='*60}")
print(f"\nTopic: {result.topic}")
print(f"Total Word Count: {result.word_count:,}")
print(f"Total Citations: {result.total_citations}")
print(f"Quality Score: {result.quality_score:.2f}")
print(f"\nPerspectives Researched: {', '.join(result.perspectives)}")

# Show outline structure
print(f"\n=== Article Outline ===")
for i, section in enumerate(result.outline, 1):
    print(f"\n{i}. {section['title']}")
    if 'subsections' in section:
        for j, sub in enumerate(section['subsections'], 1):
            print(f"   {i}.{j} {sub['title']}")

# Simulate human review and feedback
print(f"\n{'='*60}")
print(f"HUMAN REVIEW SIMULATION")
print(f"{'='*60}")

review_interface = HumanReviewInterface()
review_prompts = review_interface.generate_review_prompts(
    article=result.article,
    topic=topic
)

print("\nReview Questions Generated:")
for category, questions in review_prompts.items():
    print(f"\n{category.replace('_', ' ').title()}:")
    for q in questions[:2]:  # Show first 2 questions per category
        print(f"  - {q}")

# Simulate human feedback
human_feedback = {
    "Are all factual claims accurate?": "Mostly, but need verification on quantum supremacy claims",
    "Are there any important aspects missing?": "Should add section on post-quantum cryptography",
    "Is the article well-structured?": "Yes, structure is good",
    "Should any sections be expanded?": "The impact on blockchain needs more detail"
}

# Incorporate feedback
print("\nIncorporating Human Feedback...")
final_result = storm(topic=topic, human_feedback=human_feedback)

print(f"\nArticle updated with human feedback!")
print(f"Human feedback applied: {final_result.human_feedback_applied}")
```

## Performance Metrics and Evaluation

### STORM-Specific Metrics

```python
def storm_evaluation_metrics(storm_system, test_topics: List[str]) -> Dict:
    """Comprehensive evaluation of STORM system performance."""
    results = {
        'research_coverage': [],
        'outline_quality': [],
        'article_quality': [],
        'generation_time': [],
        'citation_accuracy': []
    }

    for topic in test_topics:
        import time
        start_time = time.time()

        # Generate article
        result = storm_system(topic=topic)

        generation_time = time.time() - start_time

        # Evaluate research coverage
        research_score = evaluate_research_coverage(result, topic)
        results['research_coverage'].append(research_score)

        # Evaluate outline quality
        outline_score = evaluate_outline_quality(result.outline, topic)
        results['outline_quality'].append(outline_score)

        # Evaluate article quality
        article_score = result.quality_score
        results['article_quality'].append(article_score)

        # Record generation time
        results['generation_time'].append(generation_time)

        # Evaluate citation accuracy (simplified)
        citation_score = min(1.0, result.total_citations / (result.word_count / 100))
        results['citation_accuracy'].append(citation_score)

    # Calculate averages
    return {
        'avg_research_coverage': sum(results['research_coverage']) / len(results['research_coverage']),
        'avg_outline_quality': sum(results['outline_quality']) / len(results['outline_quality']),
        'avg_article_quality': sum(results['article_quality']) / len(results['article_quality']),
        'avg_generation_time': sum(results['generation_time']) / len(results['generation_time']),
        'avg_citation_accuracy': sum(results['citation_accuracy']) / len(results['citation_accuracy'])
    }

def evaluate_research_coverage(result: dspy.Prediction, topic: str) -> float:
    """Evaluate how well research covers the topic."""
    # Check for multiple perspectives
    perspective_score = min(1.0, len(result.perspectives) / 5.0)

    # Check for comprehensive outline
    section_score = min(1.0, len(result.outline) / 8.0)

    return (perspective_score + section_score) / 2

def evaluate_outline_quality(outline: List[Dict], topic: str) -> float:
    """Evaluate outline structure quality."""
    # Check for logical structure
    has_intro = any('introduction' in s['title'].lower() for s in outline)
    has_conclusion = any('conclusion' in s['title'].lower() for s in outline)

    structure_score = 1.0 if has_intro and has_conclusion else 0.5

    # Check for balance
    if outline:
        word_counts = [s.get('word_count', 500) for s in outline]
        avg = sum(word_counts) / len(word_counts)
        variance = sum((w - avg) ** 2 for w in word_counts) / len(word_counts)
        balance_score = max(0, 1 - variance / (avg ** 2))
    else:
        balance_score = 0

    return (structure_score + balance_score) / 2
```

## Real-World Deployment Considerations

### 1. Scalability Optimizations

```python
class ScalableSTORM(dspy.Module):
    """Optimized STORM for large-scale deployment."""

    def __init__(self):
        super().__init__()
        self.cache = {}  # Simple cache for research results
        self.parallel_processor = ParallelProcessor()
        self.batch_retriever = BatchRetriever()

    def forward(self, topics: List[str]) -> List[dspy.Prediction]:
        """Process multiple topics in parallel."""
        # Batch research for similar topics
        research_batches = self._batch_similar_topics(topics)

        # Process in parallel
        results = []
        for batch in research_batches:
            batch_results = self._process_batch(batch)
            results.extend(batch_results)

        return results

    def _batch_similar_topics(self, topics: List[str]) -> List[List[str]]:
        """Group similar topics for batch processing."""
        # Simplified batching - in practice would use similarity metrics
        return [[topic] for topic in topics]  # Process individually for now

    def _process_batch(self, batch: List[str]) -> List[dspy.Prediction]:
        """Process a batch of topics."""
        # Implementation would use parallel processing
        storm = StormWritingAssistant()
        return [storm(topic=topic) for topic in batch]
```

### 2. Integration with External Systems

```python
class STORMAPI:
    """API wrapper for STORM system."""

    def __init__(self):
        self.storm = StormWritingAssistant()
        self.rate_limiter = RateLimiter(requests_per_minute=10)

    async def generate_article(self,
                             topic: str,
                             user_id: str,
                             options: Optional[Dict] = None) -> Dict:
        """Async API endpoint for article generation."""
        # Rate limiting
        await self.rate_limiter.acquire(user_id)

        # Generate article
        result = self.storm(topic=topic)

        # Format for API response
        return {
            'article_id': self._generate_id(),
            'topic': result.topic,
            'article': result.article,
            'metadata': {
                'word_count': result.word_count,
                'citations': result.total_citations,
                'quality_score': result.quality_score,
                'perspectives': result.perspectives,
                'generation_time': datetime.now().isoformat()
            },
            'status': 'completed'
        }

    def _generate_id(self) -> str:
        """Generate unique article ID."""
        import uuid
        return str(uuid.uuid4())
```

## Summary

The STORM writing assistant demonstrates how DSPy can be used to build sophisticated AI systems that:

1. **Simulate Human Research Processes** through multi-perspective investigation
2. **Generate Comprehensive Outlines** that organize information logically
3. **Produce High-Quality Articles** with proper citations and structure
4. **Incorporate Human Feedback** for collaborative writing
5. **Scale to Production** with proper optimization and APIs

### Key Achievements

- **Two-Stage Architecture**: Clear separation of research and writing phases
- **Quality Assurance**: Comprehensive validation of generated content
- **Human-AI Collaboration**: Seamless integration of human feedback
- **Modular Design**: Components can be customized and extended
- **Production Ready**: Scalable and API-accessible implementation

### Lessons Learned

1. **Research Quality Directly Impacts Article Quality**
2. **Outline Generation is Critical for Coherence**
3. **Citation Management Requires Sophisticated Logic**
4. **Human Feedback Enhances, Not Replaces, AI Writing**
5. **System Architecture Must Support Iterative Improvement**

## Next Steps

- [Multi-hop Search](../06-real-world-applications/02-multi-hop-search.md) - Advanced retrieval techniques
- [Evaluation Best Practices](../../04-evaluation/05-best-practices.md) - System evaluation frameworks
- [Production Deployment](../09-appendices/02-production-deployment.md) - Deploying DSPy applications

## Further Reading

- [Original STORM Paper](https://arxiv.org/abs/2401.05454)
- [Human-AI Collaboration in Writing](https://example.com/human-ai-writing)
- [Scalable AI System Architecture](https://example.com/scalable-ai)
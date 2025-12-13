# Outline Generation for Structured Article Writing

## Prerequisites

- **Chapter 3**: Modules - Understanding of DSPy modules
- **Chapter 6**: RAG Systems - Information retrieval concepts
- **Previous Sections**: Perspective-Driven Research
- **Required Knowledge**: Understanding of document structure and organization
- **Difficulty Level**: Intermediate-Advanced
- **Estimated Reading Time**: 35 minutes

## Learning Objectives

By the end of this section, you will:
- Generate structured article outlines from research data
- Organize information logically using hierarchy and flow principles
- Create outlines that balance comprehensiveness and readability
- Implement outline evaluation and refinement techniques
- Build systems that adapt outline structure to content requirements

## Introduction: The Importance of Good Outlines

A well-structured outline is the backbone of any comprehensive article. It:
- Provides logical flow and progression of ideas
- Ensures comprehensive coverage of the topic
- Helps maintain focus and avoid digression
- Guides the writing process section by section
- Ensures balance between different aspects of the topic

In the context of AI-assisted writing, outline generation is a critical pre-writing step that transforms scattered research findings into a coherent structure.

## Understanding Article Outline Structure

### Typical Article Hierarchy

```
Title
├── Introduction
│   ├── Hook/Opening
│   ├── Background Context
│   └── Thesis/Overview
├── Main Body
│   ├── Section 1
│   │   ├── Subsection 1.1
│   │   └── Subsection 1.2
│   ├── Section 2
│   │   ├── Subsection 2.1
│   │   └── Subsection 2.2
│   └── ...
└── Conclusion
    ├── Summary
    ├── Implications
    └── Future Directions
```

## Building the Outline Generation System

### 1. Research Analysis and Topic Segmentation

```python
import dspy
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re

@dataclass
class ResearchCluster:
    """Represents a cluster of related research findings."""
    theme: str
    key_points: List[str]
    sources: List[str]
    importance: float
    relationships: List[str]

class ResearchAnalyzer(dspy.Module):
    """Analyzes research data to identify key themes and clusters."""

    def __init__(self):
        super().__init__()
        self.identify_themes = dspy.ChainOfThought(
            "research_data, topic -> main_themes, sub_themes"
        )
        self.cluster_findings = dspy.Predict(
            "findings, themes -> clusters"
        )
        self.assess_importance = dspy.ChainOfThought(
            "theme, research_coverage, topic_relevance -> importance_score"
        )

    def forward(self, research_data: Dict, topic: str) -> dspy.Prediction:
        """
        Analyze research data and identify thematic clusters.

        Args:
            research_data: Research findings organized by perspective
            topic: Main topic of the article

        Returns:
            Analyzed research clusters and themes
        """
        # Extract all findings
        all_findings = self._extract_findings(research_data)

        # Identify main themes
        themes_result = self.identify_themes(
            research_data=str(all_findings),
            topic=topic
        )

        # Cluster findings by theme
        clusters_result = self.cluster_findings(
            findings=str(all_findings),
            themes=themes_result.main_themes + themes_result.sub_themes
        )

        # Assess importance of each cluster
        clusters = self._parse_clusters(clusters_result.clusters)
        for cluster in clusters:
            importance = self.assess_importance(
                theme=cluster.theme,
                research_coverage=len(cluster.key_points),
                topic_relevance=topic
            )
            cluster.importance = float(importance.importance_score)

        return dspy.Prediction(
            main_themes=themes_result.main_themes,
            sub_themes=themes_result.sub_themes,
            clusters=sorted(clusters, key=lambda x: x.importance, reverse=True),
            total_findings=len(all_findings)
        )

    def _extract_findings(self, research_data: Dict) -> List[str]:
        """Extract all research findings from structured data."""
        findings = []
        for perspective, data in research_data.items():
            documents = data.get('documents', [])
            summaries = data.get('summaries', [])
            key_points = data.get('key_points', [])

            findings.extend(documents)
            findings.extend(summaries)
            findings.extend(key_points)

        return findings[:50]  # Limit to top 50 findings

    def _parse_clusters(self, clusters_text: str) -> List[ResearchCluster]:
        """Parse cluster information from text."""
        clusters = []
        current_cluster = None

        lines = clusters_text.strip().split('\n')
        for line in lines:
            if line.strip().startswith('Theme:'):
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = ResearchCluster(
                    theme=line.strip().replace('Theme:', '').strip(),
                    key_points=[],
                    sources=[],
                    importance=0.0,
                    relationships=[]
                )
            elif line.strip().startswith('-') and current_cluster:
                point = line.strip().lstrip('- ').strip()
                current_cluster.key_points.append(point)
            elif line.strip().startswith('Sources:') and current_cluster:
                sources = line.strip().replace('Sources:', '').strip()
                current_cluster.sources = [s.strip() for s in sources.split(',')]

        if current_cluster:
            clusters.append(current_cluster)

        return clusters
```

### 2. Outline Structure Planner

```python
class OutlinePlanner(dspy.Module):
    """Plans the structure of article outlines."""

    def __init__(self):
        super().__init__()
        self.determine_structure = dspy.ChainOfThought(
            "topic, complexity, intended_audience -> structure_type, depth, sections_needed"
        )
        self.create_hierarchy = dspy.Predict(
            "main_sections, clusters, word_count_target -> hierarchical_outline"
        )
        self.balance_sections = dspy.ChainOfThought(
            "outline_draft, total_word_count -> balanced_outline, section_word_counts"
        )

    def forward(self,
               topic: str,
               clusters: List[ResearchCluster],
               constraints: Optional[Dict] = None) -> dspy.Prediction:
        """
        Create a structured outline from research clusters.

        Args:
            topic: Article topic
            clusters: Research clusters from analysis
            constraints: Optional constraints (word count, audience, etc.)

        Returns:
            Structured outline with hierarchy
        """
        if constraints is None:
            constraints = {
                'word_count_target': 2000,
                'intended_audience': 'general',
                'complexity': 'medium'
            }

        # Determine appropriate structure
        structure = self.determine_structure(
            topic=topic,
            complexity=constraints['complexity'],
            intended_audience=constraints['intended_audience']
        )

        # Create main sections from top clusters
        main_sections = [cluster.theme for cluster in clusters[:structure.sections_needed]]

        # Build hierarchical outline
        hierarchy = self.create_hierarchy(
            main_sections=main_sections,
            clusters=clusters,
            word_count_target=constraints['word_count_target']
        )

        # Balance section sizes
        balanced = self.balance_sections(
            outline_draft=hierarchy.hierarchical_outline,
            total_word_count=constraints['word_count_target']
        )

        # Parse and structure the final outline
        final_outline = self._parse_outline(balanced.balanced_outline)
        word_counts = self._parse_word_counts(balanced.section_word_counts)

        return dspy.Prediction(
            outline=final_outline,
            section_word_counts=word_counts,
            structure_type=structure.structure_type,
            total_sections=len(final_outline)
        )

    def _parse_outline(self, outline_text: str) -> List[Dict]:
        """Parse outline text into structured format."""
        outline = []
        current_section = None
        current_subsection = None

        lines = outline_text.strip().split('\n')
        for line in lines:
            if line.strip().startswith('I.') or line.strip().startswith('1.'):
                # Main section
                if current_section:
                    outline.append(current_section)
                current_section = {
                    'level': 1,
                    'title': line.strip().split(' ', 1)[1] if ' ' in line.strip() else line.strip(),
                    'subsections': []
                }
                current_subsection = None
            elif line.strip().startswith('   A.') or line.strip().startswith('   1.'):
                # Subsection
                if current_section:
                    current_subsection = {
                        'level': 2,
                        'title': line.strip().split(' ', 1)[1] if ' ' in line.strip() else line.strip(),
                        'subsections': []
                    }
                    current_section['subsections'].append(current_subsection)
            elif line.strip().startswith('      i.') or line.strip().startswith('      1.'):
                # Sub-subsection
                if current_subsection:
                    sub_subsection = {
                        'level': 3,
                        'title': line.strip().split(' ', 1)[1] if ' ' in line.strip() else line.strip(),
                        'subsections': []
                    }
                    current_subsection['subsections'].append(sub_subsection)

        if current_section:
            outline.append(current_section)

        return outline

    def _parse_word_counts(self, word_counts_text: str) -> Dict[str, int]:
        """Parse word count allocations."""
        word_counts = {}
        lines = word_counts_text.strip().split('\n')
        for line in lines:
            if ':' in line:
                section, count = line.strip().split(':', 1)
                word_counts[section.strip()] = int(count.strip())
        return word_counts
```

### 3. Outline Refinement and Optimization

```python
class OutlineOptimizer(dspy.Module):
    """Refines and optimizes article outlines."""

    def __init__(self):
        super().__init__()
        self.check_flow = dspy.ChainOfThought(
            "outline -> flow_score, flow_issues"
        )
        self.check_completeness = dspy.Predict(
            "outline, research_clusters, topic -> missing_elements, redundant_elements"
        )
        self.optimize_structure = dspy.Predict(
            "outline, issues, suggestions -> improved_outline"
        )

    def forward(self,
               outline: List[Dict],
               research_clusters: List[ResearchCluster],
               topic: str) -> dspy.Prediction:
        """
        Optimize outline for better flow and completeness.

        Args:
            outline: Initial outline structure
            research_clusters: Available research content
            topic: Article topic

        Returns:
            Optimized outline
        """
        # Check logical flow
        flow_check = self.check_flow(outline=str(outline))

        # Check completeness against research
        completeness = self.check_completeness(
            outline=str(outline),
            research_clusters=str(research_clusters),
            topic=topic
        )

        # Collect issues and suggestions
        issues = []
        suggestions = []

        if flow_check.flow_score < 0.8:
            issues.append(f"Flow issues: {flow_check.flow_issues}")
            suggestions.append("Reorder sections for better logical progression")

        if completeness.missing_elements:
            issues.append(f"Missing elements: {completeness.missing_elements}")
            suggestions.append("Add sections covering missing aspects")

        if completeness.redundant_elements:
            issues.append(f"Redundant elements: {completeness.redundant_elements}")
            suggestions.append("Combine or remove redundant sections")

        # Optimize if issues found
        if issues:
            optimized = self.optimize_structure(
                outline=str(outline),
                issues="; ".join(issues),
                suggestions="; ".join(suggestions)
            )
            final_outline = self._parse_outline(optimized.improved_outline)
        else:
            final_outline = outline

        return dspy.Prediction(
            optimized_outline=final_outline,
            original_outline=outline,
            issues_identified=len(issues),
            improvements_made=len(issues)
        )

    def _parse_outline(self, outline_text: str) -> List[Dict]:
        """Parse outline text (same as in OutlinePlanner)."""
        # Implementation identical to OutlinePlanner._parse_outline
        outline = []
        current_section = None

        lines = outline_text.strip().split('\n')
        for line in lines:
            if re.match(r'^[IVX]+\.|^\d+\.', line.strip()):
                if current_section:
                    outline.append(current_section)
                current_section = {
                    'level': 1,
                    'title': line.strip().split(' ', 1)[1] if ' ' in line.strip() else line.strip(),
                    'subsections': []
                }
            elif line.strip().startswith('   ') and (re.match(r'^[A-Z]\.|^\d+\.', line.strip())):
                if current_section:
                    subsection = {
                        'level': 2,
                        'title': line.strip().split(' ', 1)[1] if ' ' in line.strip() else line.strip(),
                        'subsections': []
                    }
                    current_section['subsections'].append(subsection)

        if current_section:
            outline.append(current_section)

        return outline
```

### 4. Complete Outline Generation System

```python
class ArticleOutlineGenerator(dspy.Module):
    """Complete system for generating article outlines."""

    def __init__(self):
        super().__init__()
        self.analyzer = ResearchAnalyzer()
        self.planner = OutlinePlanner()
        self.optimizer = OutlineOptimizer()
        self.enhancer = OutlineEnhancer()

    def forward(self,
               topic: str,
               research_data: Dict,
               constraints: Optional[Dict] = None) -> dspy.Prediction:
        """
        Generate a complete, optimized article outline.

        Args:
            topic: Article topic
            research_data: Research findings from multiple perspectives
            constraints: Optional constraints and requirements

        Returns:
            Complete outline with metadata
        """
        # Step 1: Analyze research data
        analysis = self.analyzer(research_data=research_data, topic=topic)

        # Step 2: Plan outline structure
        plan = self.planner(
            topic=topic,
            clusters=analysis.clusters,
            constraints=constraints
        )

        # Step 3: Optimize outline
        optimized = self.optimizer(
            outline=plan.outline,
            research_clusters=analysis.clusters,
            topic=topic
        )

        # Step 4: Enhance with additional details
        enhanced = self.enhancer(
            outline=optimized.optimized_outline,
            research_data=research_data,
            section_word_counts=plan.section_word_counts
        )

        # Generate outline summary
        summary = self._generate_summary(
            topic=topic,
            outline=enhanced.enhanced_outline,
            analysis=analysis
        )

        return dspy.Prediction(
            topic=topic,
            outline=enhanced.enhanced_outline,
            outline_summary=summary,
            section_word_counts=plan.section_word_counts,
            total_sections=len(enhanced.enhanced_outline),
            research_themes=analysis.main_themes,
            optimization_improvements=optimized.improvements_made
        )

    def _generate_summary(self,
                         topic: str,
                         outline: List[Dict],
                         analysis: dspy.Prediction) -> str:
        """Generate a summary of the outline structure."""
        summarizer = dspy.Predict("topic, outline_structure, themes -> summary")

        return summarizer(
            topic=topic,
            outline_structure=str(outline),
            themes=", ".join(analysis.main_themes[:3])
        ).summary


class OutlineEnhancer(dspy.Module):
    """Enhances outlines with additional details and metadata."""

    def __init__(self):
        super().__init__()
        self.add_purposes = dspy.Predict(
            "section_title, article_topic -> section_purpose"
        )
        self.suggest_keywords = dspy.Predict(
            "section_title, section_content_suggestions -> keywords"
        )
        self.assign_perspectives = dspy.Predict(
            "section_title, available_perspectives -> primary_perspective"
        )

    def forward(self,
               outline: List[Dict],
               research_data: Dict,
               section_word_counts: Dict) -> dspy.Prediction:
        """
        Enhance outline with additional metadata.

        Args:
            outline: Basic outline structure
            research_data: Available research content
            section_word_counts: Word count allocations

        Returns:
            Enhanced outline with metadata
        """
        enhanced_outline = []
        available_perspectives = list(research_data.keys())

        for section in outline:
            # Add purpose
            purpose = self.add_purposes(
                section_title=section['title'],
                article_topic=""  # Would be passed from main system
            )

            # Add keywords
            keywords = self.suggest_keywords(
                section_title=section['title'],
                section_content_suggestions=""
            )

            # Assign primary perspective
            perspective = self.assign_perspectives(
                section_title=section['title'],
                available_perspectives=", ".join(available_perspectives)
            )

            enhanced_section = {
                'title': section['title'],
                'level': section['level'],
                'purpose': purpose.section_purpose,
                'keywords': self._parse_keywords(keywords.keywords),
                'perspective': perspective.primary_perspective,
                'word_count': section_word_counts.get(section['title'], 500),
                'subsections': []
            }

            # Process subsections
            for subsection in section.get('subsections', []):
                sub_purpose = self.add_purposes(
                    section_title=subsection['title'],
                    article_topic=""
                )

                enhanced_subsection = {
                    'title': subsection['title'],
                    'level': subsection['level'],
                    'purpose': sub_purpose.section_purpose,
                    'keywords': [],
                    'word_count': section_word_counts.get(subsection['title'], 300),
                    'subsections': []
                }

                enhanced_section['subsections'].append(enhanced_subsection)

            enhanced_outline.append(enhanced_section)

        return dspy.Prediction(enhanced_outline=enhanced_outline)

    def _parse_keywords(self, keywords_text: str) -> List[str]:
        """Parse keywords from generated text."""
        keywords = []
        if ',' in keywords_text:
            keywords = [k.strip() for k in keywords_text.split(',')]
        else:
            keywords = keywords_text.split()
        return keywords[:10]  # Limit to 10 keywords
```

## Advanced Features

### 1. Adaptive Outline Generation

```python
class AdaptiveOutlineGenerator(dspy.Module):
    """Generates outlines that adapt to content constraints."""

    def __init__(self):
        super().__init__()
        self.assess_feasibility = dspy.ChainOfThought(
            "outline, available_research, word_limit -> feasible, adjustments_needed"
        )
        self.adapt_structure = dspy.Predict(
            "outline, constraints -> adapted_outline"
        )

    def generate_adaptive_outline(self,
                                 topic: str,
                                 research_data: Dict,
                                 max_word_count: int,
                                 min_sections: int = 3) -> dspy.Prediction:
        """Generate outline adapted to specific constraints."""
        # Generate initial outline
        generator = ArticleOutlineGenerator()
        initial = generator(topic=topic, research_data=research_data)

        # Assess feasibility
        feasibility = self.assess_feasibility(
            outline=str(initial.outline),
            available_research=str(research_data),
            word_limit=max_word_count
        )

        # Adapt if needed
        if not feasibility.feasible:
            adapted = self.adapt_structure(
                outline=str(initial.outline),
                constraints=f"Max words: {max_word_count}, Min sections: {min_sections}"
            )
            final_outline = self._parse_outline(adapted.adapted_outline)
        else:
            final_outline = initial.outline

        return dspy.Prediction(
            outline=final_outline,
            adaptations_needed=feasibility.adjustments_needed,
            fits_constraints=feasibility.feasible
        )
```

### 2. Multi-Format Outline Support

```python
class OutlineFormatter(dspy.Module):
    """Formats outlines in various styles."""

    def __init__(self):
        super().__init__()
        self.format_outline = dspy.Predict(
            "outline, format_style -> formatted_outline"
        )

    def format_for_purpose(self,
                          outline: List[Dict],
                          format_style: str = "academic") -> str:
        """Format outline for specific purposes."""
        format_request = self.format_outline(
            outline=str(outline),
            format_style=format_style
        )

        return format_request.formatted_outline

# Usage examples:
# academic_format = formatter.format_for_purpose(outline, "academic")
# blog_format = formatter.format_for_purpose(outline, "blog")
# technical_format = formatter.format_for_purpose(outline, "technical")
```

## Example Usage

### Complete Outline Generation Workflow

```python
# Initialize the system
outline_generator = ArticleOutlineGenerator()

# Example research data
research_data = {
    'scientific': {
        'documents': [
            "Recent studies show renewable energy reduces carbon emissions by 40%",
            "Solar panel efficiency has increased to 22% in 2023",
            "Wind energy costs have decreased by 70% in the last decade"
        ],
        'key_points': [
            "Renewable energy is key to climate goals",
            "Technology improvements drive adoption",
            "Cost reduction enables widespread use"
        ]
    },
    'economic': {
        'documents': [
            "Renewable energy creates 3 times more jobs than fossil fuels",
            "Initial investment costs are offset by long-term savings",
            "Market growth projected at 8% annually"
        ],
        'key_points': [
            "Economic benefits exceed costs",
            "Job creation potential",
            "Market expansion opportunities"
        ]
    },
    'social': {
        'documents': [
            "Public acceptance of renewable energy is growing",
            "Community solar projects increase local engagement",
            "Energy independence improves quality of life"
        ],
        'key_points': [
            "Social acceptance increasing",
            "Community benefits",
            "Energy democratization"
        ]
    }
}

# Generate outline
result = outline_generator(
    topic="The Future of Renewable Energy",
    research_data=research_data,
    constraints={
        'word_count_target': 2500,
        'intended_audience': 'educated general',
        'complexity': 'medium'
    }
)

# Display results
print(f"\n=== Outline for: {result.topic} ===\n")
print(f"Total Sections: {result.total_sections}")
print(f"Main Themes: {', '.join(result.research_themes)}\n")

print("\nOutline Structure:")
for section in result.outline:
    print(f"\n{section['title']}")
    print(f"  Purpose: {section['purpose']}")
    print(f"  Word Count: {section['word_count']}")
    print(f"  Perspective: {section['perspective']}")
    if section['keywords']:
        print(f"  Keywords: {', '.join(section['keywords'][:5])}")

    for subsection in section.get('subsections', []):
        print(f"  ├─ {subsection['title']}")
        print(f"    Purpose: {subsection['purpose']}")

print(f"\nOptimization Improvements: {result.optimization_improvements}")
print(f"\nOutline Summary:")
print(result.outline_summary)
```

## Best Practices for Outline Generation

### 1. Research Integration
- Ensure all major research themes are represented
- Balance perspectives across sections
- Allocate space proportional to evidence availability
- Cross-reference related concepts

### 2. Logical Structure
- Start with broad context, narrow to specifics
- Group related concepts together
- Ensure smooth transitions between sections
- Follow natural progression of ideas

### 3. Readability Considerations
- Limit hierarchy depth (max 3-4 levels)
- Balance section lengths
- Use clear, descriptive titles
- Include variety in section types

### 4. Flexibility
- Allow for dynamic adjustment
- Support different article formats
- Accommodate varying word counts
- Enable customization for audiences

## Evaluation Metrics

```python
def outline_quality_metric(example, pred, trace=None):
    """Evaluate outline quality."""
    # Structure completeness
    has_intro = any('introduction' in s['title'].lower() for s in pred.outline)
    has_conclusion = any('conclusion' in s['title'].lower() for s in pred.outline)
    structure_score = 1.0 if has_intro and has_conclusion else 0.5

    # Section balance
    word_counts = [s['word_count'] for s in pred.outline]
    avg_count = sum(word_counts) / len(word_counts)
    balance_score = 1.0 - max(abs(w - avg_count) / avg_count for w in word_counts)

    # Research coverage
    covered_themes = set(s.get('perspective', '') for s in pred.outline)
    total_themes = set(example.research_perspectives.keys())
    coverage_score = len(covered_themes & total_themes) / len(total_themes)

    # Overall score
    overall = (
        0.3 * structure_score +
        0.3 * balance_score +
        0.4 * coverage_score
    )

    if trace is not None:
        return overall >= 0.7

    return overall
```

## Summary

Outline generation is a crucial step in article creation that:

1. **Transforms Research into Structure** by organizing scattered findings
2. **Ensures Logical Flow** through careful section ordering
3. **Balances Content Coverage** across different aspects of the topic
4. **Adapts to Constraints** while maintaining quality
5. **Guides the Writing Process** with clear direction

### Key Takeaways

1. **Research Analysis** is the foundation of good outlines
2. **Hierarchical Structure** helps organize complex topics
3. **Flow Optimization** ensures readability
4. **Constraint Adaptation** enables practical use
5. **Quality Metrics** guide iterative improvement

## Next Steps

- [Long-form Article Generation](./08-long-form-generation.md) - Use outlines to generate complete articles
- [STORM Writing Assistant](../08-case-studies/05-storm-writing-assistant.md) - Complete system integration
- [Advanced Composition](../../03-modules/06-composing-modules.md) - Complex module patterns

## Further Reading

- [Information Architecture for Writers](https://example.com/info-architecture)
- [Academic Writing Structure Guidelines](https://example.com/academic-structure)
- [Content Organization Best Practices](https://example.com/content-organization)
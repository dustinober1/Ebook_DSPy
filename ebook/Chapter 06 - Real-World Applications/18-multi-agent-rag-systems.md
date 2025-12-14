# Multi-Agent RAG Systems with DSPy

## Overview

Multi-Agent RAG systems represent a powerful architecture where multiple specialized agents collaborate to solve complex information retrieval and question-answering tasks. Each agent can have its own expertise, knowledge base, and retrieval tools, while a lead agent orchestrates their interactions. This approach excels in domains requiring deep specialized knowledge across multiple subdomains.

## Architecture Overview

### Core Components

```python
# Architecture Overview
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Agent RAG System                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────────────┐    ┌────────────────┐    ┌──────────────┐  │
│  │   Lead Agent   │───▶│ Expert Agent 1 │    │Expert Agent N│  │
│  │  (Orchestrator)│    │  (e.g., Diabetes)│    │  (e.g., COPD) │  │
│  │                │    │                │    │             │  │
│  └────────────────┘    └────────────────┘    └──────────────┘  │
│           │                      │                      │      │
│           ▼                      ▼                      ▼      │
│  ┌────────────────┐    ┌────────────────┐    ┌──────────────┐  │
│  │  Query Router  │    │ Vector Store 1 │    │Vector Store N│  │
│  │                │    │                │    │             │  │
│  └────────────────┘    └────────────────┘    └──────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              GEPA Optimization Layer                    │  │
│  │  - Student: Agent being optimized                        │  │
│  │  - Judge: Evaluates performance                         │  │
│  │  - Teacher: Suggests improvements                       │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Patterns

1. **Specialized Expert Agents**: Each agent focuses on a specific domain
2. **Dedicated Knowledge Bases**: Separate vector stores for each domain
3. **Hierarchical Orchestration**: Lead agent coordinates expert agents
4. **Tool-Based Communication**: Agents interact through well-defined tool APIs
5. **Independent Optimization**: Each agent can be optimized separately

## Implementation: Medical Multi-Agent System

Let's build a complete multi-agent RAG system for medical questions about diabetes and COPD.

### Step 1: Setting Up Knowledge Bases

```python
import dspy
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Configure language models
lm = dspy.LM(
    "openrouter/openai/gpt-4o-mini",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.3,
    max_tokens=64000
)
dspy.settings.configure(lm=lm)

# Create specialized vector stores
def create_specialized_vectorstore(pdf_paths, save_dir):
    """Create a vector store for a specific medical domain."""
    documents = []

    # Load PDFs
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        documents.extend(docs)

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_dir)

    return vectorstore

# Create domain-specific stores
diabetes_store = create_specialized_vectorstore(
    ["diabetes_guidelines.pdf", "diabetes_research.pdf"],
    "vector_stores/diabetes"
)

copd_store = create_specialized_vectorstore(
    ["copd_guidelines.pdf", "copd_research.pdf"],
    "vector_stores/copd"
)
```

### Step 2: Building Expert Agents

```python
# Define retrieval tools for each domain
def diabetes_search_tool(query: str, k: int = 3) -> str:
    """Retrieve diabetes-related documents."""
    results = diabetes_store.similarity_search_with_score(query, k=k)
    context = "\n".join([
        f"[PASSAGE {i+1}, score={score:.4f}]\n{doc.page_content}"
        for i, (doc, score) in enumerate(results)
    ])
    return context

def copd_search_tool(query: str, k: int = 3) -> str:
    """Retrieve COPD-related documents."""
    results = copd_store.similarity_search_with_score(query, k=k)
    context = "\n".join([
        f"[PASSAGE {i+1}, score={score:.4f}]\n{doc.page_content}"
        for i, (doc, score) in enumerate(results)
    ])
    return context

# Define agent signatures
class MedicalAgentSignature(dspy.Signature):
    """Base signature for medical question answering."""
    question: str = dspy.InputField(desc="Medical question to answer")
    answer: str = dspy.OutputField(desc="Medical answer based on retrieved evidence")

# Create expert agents
class DiabetesExpertAgent(dspy.Module):
    """Specialized agent for diabetes-related questions."""

    def __init__(self):
        super().__init__()
        self.expert = dspy.ReAct(
            MedicalAgentSignature,
            tools=[diabetes_search_tool]
        )

    def forward(self, question):
        return self.expert(question=question)

class COPDExpertAgent(dspy.Module):
    """Specialized agent for COPD-related questions."""

    def __init__(self):
        super().__init__()
        self.expert = dspy.ReAct(
            MedicalAgentSignature,
            tools=[copd_search_tool]
        )

    def forward(self, question):
        return self.expert(question=question)
```

### Step 3: Creating the Lead Orchestrator Agent

```python
# Tools for the lead agent to consult experts
def consult_diabetes_expert(question: str) -> str:
    """Consult the diabetes expert agent."""
    agent = DiabetesExpertAgent()
    result = agent(question=question)
    return result.answer

def consult_copd_expert(question: str) -> str:
    """Consult the COPD expert agent."""
    agent = COPDExpertAgent()
    result = agent(question=question)
    return result.answer

# Lead agent that coordinates experts
class MultiAgentMedicalSystem(dspy.Module):
    """Lead agent that orchestrates multiple medical expert agents."""

    def __init__(self):
        super().__init__()
        self.lead_agent = dspy.ReAct(
            MedicalAgentSignature,
            tools=[consult_diabetes_expert, consult_copd_expert]
        )

    def forward(self, question):
        return self.lead_agent(question=question)

# Initialize the multi-agent system
multi_agent_system = MultiAgentMedicalSystem()
```

## GEPA Optimization for Multi-Agent Systems

### Understanding GEPA's Three-LLM Architecture

```python
from dspy.teleprompt import GEPA

# GEPA uses three different LLMs:
# 1. Student: The agent being optimized
# 2. Judge: Evaluates performance and provides feedback
# 3. Teacher: Suggests improvements based on feedback

class MedicalEvaluationMetric:
    """Custom metric for medical Q&A evaluation."""

    def __init__(self, teacher_lm):
        self.judge = dspy.ChainOfThought(
            """Evaluate medical answer quality.

            Consider:
            - Factual accuracy based on medical guidelines
            - Completeness of information
            - Clinical appropriateness
            - Evidence-based reasoning

            Question: {question}
            Gold Answer: {gold_answer}
            Predicted Answer: {predicted_answer}

            Score (0-1):""",
            lm=teacher_lm
        )

    def __call__(self, example, pred, trace=None):
        """Evaluate with LLM judge for medical accuracy."""
        score = self.judge(
            question=example.question,
            gold_answer=example.answer,
            predicted_answer=pred.answer
        )
        return float(score.score)
```

### Optimizing Individual Expert Agents

```python
# Prepare datasets for each domain
diabetes_trainset = [
    dspy.Example(
        question="What are the diagnostic criteria for gestational diabetes?",
        answer="GDM is diagnosed when..."
    ).with_inputs("question")
    # ... more examples
]

copd_trainset = [
    dspy.Example(
        question="What is the GOLD classification for COPD severity?",
        answer="The GOLD classification..."
    ).with_inputs("question")
    # ... more examples
]

# Configure teacher LLM for GEPA
teacher_lm = dspy.LM(
    "openrouter/openai/gpt-4",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.3
)

# Optimize diabetes expert
def optimize_diabetes_agent():
    """Optimize the diabetes expert agent using GEPA."""

    # Initialize base agent
    base_agent = DiabetesExpertAgent()

    # Create metric
    metric = MedicalEvaluationMetric(teacher_lm)

    # Configure GEPA
    teleprompter = GEPA(
        metric=metric,
        max_full_evals=5,
        num_threads=32,
        track_stats=True,
        reflection_lm=teacher_lm,
        add_format_failure_as_feedback=True
    )

    # Compile (optimize) the agent
    optimized_agent = teleprompter.compile(
        student=base_agent,
        trainset=diabetes_trainset[:20],
        valset=diabetes_trainset[20:30]
    )

    return optimized_agent

# Optimize COPD expert (similar process)
optimized_diabetes = optimize_diabetes_agent()
optimized_copd = optimize_copd_agent()
```

### Optimizing the Lead Orchestrator

```python
def optimize_lead_agent():
    """Optimize the lead orchestrator agent."""

    # Create mixed dataset requiring both expertise
    mixed_trainset = [
        dspy.Example(
            question="Compare management strategies for diabetes and COPD in elderly patients",
            answer="For elderly patients with both conditions..."
        ).with_inputs("question")
        # ... more mixed examples
    ]

    # Initialize lead agent with optimized experts
    class OptimizedMultiAgentSystem(dspy.Module):
        def __init__(self):
            super().__init__()
            self.lead_agent = dspy.ReAct(
                MedicalAgentSignature,
                tools=[consult_diabetes_expert, consult_copd_expert]
            )

        def forward(self, question):
            return self.lead_agent(question=question)

    base_lead = OptimizedMultiAgentSystem()

    # Optimize with GEPA
    metric = MedicalEvaluationMetric(teacher_lm)

    teleprompter = GEPA(
        metric=metric,
        max_full_evals=3,
        num_threads=32,
        track_stats=True,
        reflection_lm=teacher_lm
    )

    optimized_lead = teleprompter.compile(
        student=base_lead,
        trainset=mixed_trainset[:15],
        valset=mixed_trainset[15:20]
    )

    return optimized_lead

optimized_lead = optimize_lead_agent()
```

## Performance Results

### Before and After Optimization

| Agent | Baseline Score | Optimized Score | Improvement |
|-------|----------------|-----------------|-------------|
| Diabetes Expert | 90.72% | 98.90% | +8.18% |
| COPD Expert | 89.44% | 94.22% | +4.78% |
| Lead Orchestrator | 88.79% | 92.42% | +3.63% |

### Key Performance Insights

1. **Expert agents benefit most**: Domain-specific optimization yields highest gains
2. **Lead agent improvements**: Better orchestration and tool selection
3. **Synergistic effects**: Optimized experts improve lead agent performance
4. **Consistent gains**: All agents show measurable improvements

## Advanced Patterns

### 1. Cross-Agent Learning

```python
class CrossAgentLearningSystem(dspy.Module):
    """System where agents learn from each other's responses."""

    def __init__(self):
        super().__init__()
        self.diabetes_agent = DiabetesExpertAgent()
        self.copd_agent = COPDExpertAgent()
        self.synthesizer = dspy.ChainOfThought(
            """Synthesize insights from multiple expert agents.

            Diabetes Expert Response: {diabetes_response}
            COPD Expert Response: {copd_response}
            Original Question: {question}

            Provide a comprehensive answer that integrates both perspectives."""
        )

    def forward(self, question):
        # Get responses from both agents
        diabetes_response = self.diabetes_agent(question)
        copd_response = self.copd_agent(question)

        # Synthesize comprehensive answer
        synthesis = self.synthesizer(
            diabetes_response=diabetes_response.answer,
            copd_response=copd_response.answer,
            question=question
        )

        return dspy.Prediction(answer=synthesis.answer)
```

### 2. Hierarchical Expert Networks

```python
class HierarchicalExpertNetwork(dspy.Module):
    """Multi-level hierarchy of expert agents."""

    def __init__(self):
        super().__init__()

        # Level 1: Domain experts
        self.diabetes_expert = DiabetesExpertAgent()
        self.copd_expert = COPDExpertAgent()

        # Level 2: Sub-specialists
        self.diabetes_sub_specialists = {
            "gestational": DiabetesSubExpert("gestational"),
            "type1": DiabetesSubExpert("type1"),
            "type2": DiabetesSubExpert("type2")
        }

        # Level 3: Lead coordinator
        self.coordinator = dspy.ReAct(
            signature=MedicalAgentSignature,
            tools=list(self.diabetes_sub_specialists.values()) +
                   [self.copd_expert]
        )

    def route_to_appropriate_expert(self, question):
        """Route question to the most appropriate expert."""
        # Use LLM to determine best expert
        router = dspy.Predict(
            """Route medical question to appropriate expert.

            Question: {question}
            Available experts: {experts}

            Selected expert:"""
        )

        expert_choice = router(
            question=question,
            experts=", ".join(self.experts.keys())
        )

        return self.experts[expert_choice.selected_expert]
```

### 3. Dynamic Tool Selection

```python
class AdaptiveMultiAgentSystem(dspy.Module):
    """System that dynamically selects which agents to consult."""

    def __init__(self):
        super().__init__()
        self.experts = {
            "diabetes": DiabetesExpertAgent(),
            "copd": COPDExpertAgent(),
            "cardiology": CardiologyExpertAgent(),
            "nephrology": NephrologyExpertAgent()
        }

        self.planner = dspy.ChainOfThought(
            """Plan which experts to consult for a medical question.

            Question: {question}
            Available experts: {experts}

            Plan which experts to consult and in what order:"""
        )

    def forward(self, question):
        # Plan expert consultation
        plan = self.planner(
            question=question,
            experts=", ".join(self.experts.keys())
        )

        # Execute consultation plan
        expert_responses = []
        for expert_name in plan.plan.split(","):
            if expert_name.strip() in self.experts:
                response = self.experts[expert_name.strip()](question)
                expert_responses.append(f"{expert_name}: {response.answer}")

        # Synthesize final answer
        synthesizer = dspy.ChainOfThought(
            """Synthesize responses from multiple experts.

            Question: {question}
            Expert Responses: {responses}

            Comprehensive answer:"""
        )

        final_answer = synthesizer(
            question=question,
            responses="\n".join(expert_responses)
        )

        return dspy.Prediction(answer=final_answer.comprehensive_answer)
```

## Best Practices

### 1. Agent Design Principles

```python
# Good: Clear, focused expert agents
class DiabetesExpertAgent(dspy.Module):
    """Focused exclusively on diabetes-related queries."""

    def __init__(self):
        super().__init__()
        self.expertise = "diabetes"
        self.tools = [diabetes_search_tool, diabetes_guideline_tool]
        self.expert = dspy.ReAct(
            signature=DiabetesSignature,
            tools=self.tools
        )

# Bad: Overly general agent
class MedicalExpertAgent(dspy.Module):
    """Too broad - should be split into specialists."""
    pass
```

### 2. Tool Interface Design

```python
# Good: Consistent, well-documented tool interfaces
def consult_expert(
    question: str,
    context: Optional[str] = None,
    priority: str = "normal"
) -> str:
    """Standardized expert consultation interface.

    Args:
        question: The medical question to answer
        context: Optional additional context
        priority: "urgent", "normal", or "routine"

    Returns:
        Expert response with citations
    """
    pass

# Bad: Inconsistent interfaces across agents
def ask_diabetes(q): pass
def query_copd(question, extra_info): pass
```

### 3. Error Handling and Fallbacks

```python
class RobustMultiAgentSystem(dspy.Module):
    """System with comprehensive error handling."""

    def __init__(self):
        super().__init__()
        self.primary_experts = [...]
        self.backup_experts = [...]

    def forward(self, question):
        try:
            # Try primary experts
            result = self.consult_primary_experts(question)
            if self.validate_result(result):
                return result
        except Exception as e:
            self.log_error(e)

        # Fallback to backup experts
        return self.consult_backup_experts(question)

    def validate_result(self, result):
        """Validate expert response quality."""
        checks = [
            len(result.answer) > 50,
            "I don't know" not in result.answer,
            result.confidence > 0.7
        ]
        return all(checks)
```

## Evaluation and Monitoring

### Comprehensive Metrics

```python
class MultiAgentEvaluator:
    """Evaluate multi-agent system performance."""

    def __init__(self):
        self.metrics = {
            "accuracy": self.calculate_accuracy,
            "expert_utilization": self.track_expert_usage,
            "response_time": self.measure_latency,
            "coordination_quality": self.evaluate_coordination
        }

    def evaluate(self, system, testset):
        results = {}
        for metric_name, metric_func in self.metrics.items():
            results[metric_name] = metric_func(system, testset)
        return results

    def track_expert_usage(self, system, testset):
        """Track which experts are consulted and how often."""
        expert_usage = defaultdict(int)

        for example in testset:
            # Monitor agent traces
            result = system(example.question, trace=True)

            # Parse trace to identify consulted experts
            for step in result.trace:
                if "diabetes" in step.tool_name:
                    expert_usage["diabetes"] += 1
                elif "copd" in step.tool_name:
                    expert_usage["copd"] += 1

        return dict(expert_usage)
```

## Exercises

1. **Build a Three-Agent System**: Create a multi-agent system for legal advice with experts in contract law, intellectual property, and corporate law.

2. **Implement Cross-Domain Queries**: Design agents that can handle questions requiring knowledge from multiple domains (e.g., "How does diabetes affect COPD treatment?").

3. **Optimize with Different Metrics**: Experiment with various evaluation metrics for GEPA optimization beyond accuracy (e.g., response time, expert efficiency).

4. **Create a Dynamic Expert Network**: Build a system that can add new experts dynamically based on query patterns.

5. **Implement Agent Collaboration**: Design a pattern where agents can directly consult each other without going through the lead agent.

## Conclusion

Multi-agent RAG systems with DSPy provide a powerful architecture for complex question-answering tasks requiring specialized knowledge. By combining:

1. **Specialized Expert Agents**: Domain-specific knowledge and retrieval
2. **Intelligent Orchestration**: Lead agents coordinate expert interactions
3. **GEPA Optimization**: Systematic improvement through feedback loops
4. **Modular Design**: Easy to extend and maintain

You can build sophisticated systems that outperform single-agent approaches on complex, multi-domain tasks. The key is to design clear interfaces, optimize each component systematically, and continuously monitor performance.

---

**References:**
- AIMultiple. (2025). RAG Frameworks: LangChain vs LangGraph vs LlamaIndex vs Haystack vs DSPy
- ArXiv. (2024). A Comparative Study of DSPy Teleprompter Algorithms
- Kargar, I. (2025). Building and Optimizing Multi-Agent RAG Systems with DSPy and GEPA
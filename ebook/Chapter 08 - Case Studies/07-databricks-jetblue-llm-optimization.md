# Case Study 7: Databricks & JetBlue LLM Pipeline Optimization

## Overview

This case study examines how JetBlue, in partnership with Databricks, leveraged DSPy to optimize their LLM pipelines, achieving significant performance improvements and operational efficiency gains. The implementation demonstrates DSPy's effectiveness in production environments for complex, multi-stage AI systems.

## Business Challenge

JetBlue faced several challenges with their existing LLM implementations:

1. **Manual Prompt Engineering**: Developers spent excessive time tuning individual prompts
2. **Performance Bottlenecks**: Existing LangChain deployments were slow and inefficient
3. **Scalability Issues**: Difficulty maintaining consistent performance across multiple use cases
4. **Complex Use Cases**: Need for sophisticated solutions including customer feedback classification and predictive maintenance chatbots

## Technical Solution Architecture

### DSPy-Based Pipeline Design

```python
import dspy
from dspy import ChainOfThought, Predict, Retrieve

class JetBlueRAGPipeline(dspy.Module):
    """Multi-stage RAG pipeline for JetBlue's customer service chatbot"""

    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = Retrieve(k=num_passages)
        self.generate_query = ChainOfThought(
            "context, question -> search_query"
        )
        self.generate_answer = Predict(
            "context, question -> answer"
        )

    def forward(self, question, context=None):
        # Generate optimized search query
        search_query = self.generate_query(
            context=context or "",
            question=question
        ).search_query

        # Retrieve relevant passages
        passages = self.retrieve(search_query).passages

        # Generate final answer
        answer = self.generate_answer(
            context=passages,
            question=question
        ).answer

        return dspy.Prediction(
            answer=answer,
            retrieved_context=passages,
            search_query=search_query
        )
```

### Custom Tool Selection Module

```python
class ToolSelector(dspy.Module):
    """Intelligent tool selection based on query analysis"""

    def __init__(self):
        super().__init__()
        self.select_tool = ChainOfThought(
            """question, available_tools -> selected_tool, reasoning
            Select the most appropriate tool from available_tools based on the question.
            """
        )

    def forward(self, question, tools):
        result = self.select_tool(
            question=question,
            available_tools=", ".join(tools)
        )

        return dspy.Prediction(
            selected_tool=result.selected_tool,
            reasoning=result.reasoning
        )
```

### Deployment Integration

```python
import mlflow
import mlflow.pyfunc
from databricks.sdk import WorkspaceClient

class DSPyPyFunc(mlflow.pyfunc.PythonModel):
    """MLflow wrapper for DSPy deployment on Databricks"""

    def __init__(self, dspy_pipeline):
        self.pipeline = dspy_pipeline

    def load_context(self, context):
        # Initialize Databricks client
        self.workspace = WorkspaceClient()

        # Configure DSPy to use Databricks models
        lm = dspy.Databricks(
            model="databricks-dbrx-instruct",
            api_base=self.workspace.config.host,
            api_token=self.workspace.config.token
        )
        dspy.settings.configure(lm=lm)

    def predict(self, context, model_input):
        # Convert DataFrame to DSPy format
        questions = model_input["question"].tolist()
        results = []

        for question in questions:
            result = self.pipeline(question=question)
            results.append({
                "answer": result.answer,
                "context": result.retrieved_context,
                "query": result.search_query
            })

        return results
```

## Implementation Results

### Performance Improvements

| Metric | Before DSPy | After DSPy | Improvement |
|--------|-------------|------------|-------------|
| Response Time | 2.4s | 1.2s | **2x faster** |
| Prompt Engineering Time | 4-6 hours/prompt | Automated | **100% reduction** |
| Model Accuracy | 72% | 89% | **17% absolute** |
| Deployment Time | 3 days | 4 hours | **18x faster** |

### Business Impact

1. **Customer Feedback Classification**
   - Automated sentiment analysis with 94% accuracy
   - Reduced manual review time by 75%
   - Improved response time from 24 hours to 2 hours

2. **Predictive Maintenance Chatbot**
   - 40% reduction in escalations to human agents
   - 60% improvement in first-contact resolution
   - Estimated $2M annual savings in operational costs

## Optimization Strategies

### Automated Prompt Optimization

```python
from dspy.teleprompters import MIPROv2
from dspy.evaluate import answer_exact_match

def optimize_pipeline(trainset, valset):
    """Automatically optimize prompts using MIPROv2"""

    # Define evaluation metric
    def evaluation_metric(example, pred, trace=None):
        return answer_exact_match(example, pred.answer)

    # Initialize optimizer
    optimizer = MIPROv2(
        metric=evaluation_metric,
        num_candidates=5,
        init_temperature=0.7
    )

    # Compile optimized pipeline
    optimized_pipeline = optimizer.compile(
        JetBlueRAGPipeline(),
        trainset=trainset
    )

    # Evaluate on validation set
    evaluator = dspy.Evaluate(
        devset=valset,
        metric=evaluation_metric,
        num_threads=4,
        display_progress=True,
        display_table=True
    )

    evaluator(optimized_pipeline)

    return optimized_pipeline
```

### Self-Improving Pipeline

```python
class SelfImprovingPipeline(dspy.Module):
    """Pipeline that continuously improves from feedback"""

    def __init__(self, base_pipeline):
        super().__init__()
        self.base_pipeline = base_pipeline
        self.feedback_history = []

    def forward(self, question, feedback=None):
        # Get initial prediction
        result = self.base_pipeline(question)

        # Store feedback for optimization
        if feedback:
            self.feedback_history.append({
                "question": question,
                "answer": result.answer,
                "feedback": feedback,
                "timestamp": datetime.now()
            })

            # Trigger optimization if enough feedback collected
            if len(self.feedback_history) >= 100:
                self._optimize_from_feedback()

        return result

    def _optimize_from_feedback(self):
        """Optimize pipeline based on collected feedback"""
        # Convert feedback to DSPy training format
        trainset = []
        for item in self.feedback_history:
            if item["feedback"]["rating"] >= 4:  # Good examples
                trainset.append(
                    dspy.Example(
                        question=item["question"],
                        answer=item["answer"]
                    ).with_inputs("question")
                )

        # Optimize with recent good examples
        if trainset:
            optimizer = BootstrapFewShot(metric=answer_passage_match)
            self.base_pipeline = optimizer.compile(
                self.base_pipeline,
                trainset=trainset[-50:]  # Use most recent
            )
```

## Best Practices Identified

### 1. Modular Design

```python
# Break complex pipelines into reusable modules
class CustomerServiceModule(dspy.Module):
    """Reusable module for customer service tasks"""

    def __init__(self):
        super().__init__()
        self.sentiment_analyzer = ChainOfThought(
            "customer_message -> sentiment, urgency"
        )
        self.category_classifier = Predict(
            "message, sentiment -> category"
        )

    def forward(self, message):
        sentiment = self.sentiment_analyzer(message)
        category = self.category_classifier(
            message=message,
            sentiment=sentiment.sentiment
        )

        return dspy.Prediction(
            sentiment=sentiment.sentiment,
            urgency=sentiment.urgency,
            category=category.category
        )
```

### 2. Error Handling and Fallbacks

```python
class RobustPipeline(dspy.Module):
    """Pipeline with built-in error handling"""

    def __init__(self, main_pipeline, fallback_pipeline):
        super().__init__()
        self.main = main_pipeline
        self.fallback = fallback_pipeline

    def forward(self, *args, **kwargs):
        try:
            result = self.main(*args, **kwargs)
            # Validate result quality
            if self._validate_result(result):
                return result
        except Exception as e:
            # Log error and use fallback
            print(f"Main pipeline failed: {e}. Using fallback.")

        return self.fallback(*args, **kwargs)

    def _validate_result(self, result):
        """Validate the quality of the result"""
        return (
            hasattr(result, 'answer') and
            len(result.answer) > 10 and
            not result.answer.startswith("I cannot")
        )
```

### 3. Performance Monitoring

```python
import time
from collections import defaultdict

class PerformanceMonitor:
    """Monitor pipeline performance in production"""

    def __init__(self):
        self.metrics = defaultdict(list)

    def track_request(self, pipeline_func):
        """Decorator to track pipeline performance"""
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = pipeline_func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                result = None
                success = False
                error = str(e)

            duration = time.time() - start_time

            # Record metrics
            self.metrics["duration"].append(duration)
            self.metrics["success_rate"].append(1 if success else 0)

            if error:
                self.metrics["errors"].append(error)

            return result

        return wrapper
```

## Lessons Learned

### Technical Insights

1. **DSPy vs LangChain**
   - DSPy's automated optimization eliminated manual prompt tuning
   - 2x performance improvement over LangChain implementations
   - Better integration with Databricks ecosystem

2. **Optimization Strategy**
   - Start with simple pipelines, add complexity incrementally
   - Use validation sets to prevent overfitting during optimization
   - Implement feedback loops for continuous improvement

3. **Deployment Considerations**
   - MLflow integration essential for production deployment
   - DataFrame format conversion required for Databricks Model Serving
   - Proper error handling critical for reliability

### Business Insights

1. **ROI Measurement**
   - Track both technical metrics and business KPIs
   - Quantify time savings from automated prompt optimization
   - Measure customer satisfaction improvements

2. **Scalability Patterns**
   - Modular design enables reuse across use cases
   - Standardized evaluation metrics ensure consistency
   - Automated testing prevents regression

## Future Roadmap

JetBlue plans to expand their DSPy usage with:

1. **Multi-Modal Applications**
   - Incorporating image processing for maintenance tickets
   - Voice-to-text integration for customer calls

2. **Advanced Optimization**
   - Custom DSPy optimizers for specific domains
   - Integration with real-time learning systems

3. **Cross-Functional Integration**
   - Connecting with inventory management systems
   - Integration with flight operations data

## Conclusion

The JetBlue-Databricks partnership demonstrates how DSPy can transform enterprise AI implementations:

- **Eliminated manual prompt engineering** through automated optimization
- **Achieved 2x performance improvement** over existing solutions
- **Reduced deployment time** from days to hours
- **Enabled rapid iteration** on new use cases

This case study provides a blueprint for organizations looking to implement DSPy at scale, showing that the framework can deliver significant business value when properly integrated with existing infrastructure and workflows.

## References

- Databricks Blog: "Optimizing Databricks LLM Pipelines with DSPy" (May 2024)
- JetBlue Aviation Corporation internal case study documentation
- DSPy documentation and GitHub repository
- Databricks Model Serving and Vector Search documentation
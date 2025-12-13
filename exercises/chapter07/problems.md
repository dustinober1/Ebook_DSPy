# Chapter 7: Advanced Topics - Exercises

## Overview

These exercises cover advanced DSPy techniques including caching, async operations, debugging, and deployment strategies. You'll work with performance optimization, real-time processing, and production deployment scenarios.

**Difficulty**: Advanced
**Time Estimate**: 3-4 hours total
**Prerequisites**: Completion of Chapters 1-6

---

## Exercise 1: Implement Efficient Caching

**Objective**: Build a caching layer to improve DSPy program performance and reduce API calls.

**Requirements**:
- Implement disk-based caching for LM calls
- Measure cache hit rates
- Compare performance with/without caching
- Implement cache invalidation strategy

**Starter Code**:
```python
import dspy
import json
import hashlib
from pathlib import Path
from typing import Any, Dict

class CachedDSPy(dspy.Module):
    def __init__(self, cache_dir: str = ".dspy_cache"):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.hits = 0
        self.misses = 0

    def get_cache_key(self, inputs: Dict[str, Any]) -> str:
        # TODO: Create hash of inputs for cache key
        input_str = json.dumps(inputs, sort_keys=True)
        return hashlib.md5(input_str.encode()).hexdigest()

    def get_cached(self, key: str) -> Any:
        # TODO: Retrieve from cache if exists
        pass

    def cache(self, key: str, result: Any) -> None:
        # TODO: Store result in cache
        pass

    def forward(self, **kwargs):
        # TODO: Check cache before calling LM
        pass

# Benchmark with/without caching
program = CachedDSPy()

import time
test_inputs = [
    {"query": "What is DSPy?"},
    {"query": "How do I optimize?"},
    {"query": "What is DSPy?"},  # Duplicate - should hit cache
]

start = time.time()
for inp in test_inputs:
    result = program(**inp)
elapsed = time.time() - start

print(f"Time: {elapsed:.2f}s")
print(f"Cache hits: {program.hits}, misses: {program.misses}")
print(f"Hit rate: {program.hits / (program.hits + program.misses):.1%}")
```

**Expected Output**:
```
Time: 2.45s
Cache hits: 1, misses: 2
Hit rate: 33.3%
```

**Hints**:
- Use JSON serialization for cache keys
- Consider TTL (time-to-live) for cache entries
- Hash function should be deterministic

**Advanced Challenge**: Implement partial matching (cache similar queries)

---

## Exercise 2: Build Async DSPy Operations

**Objective**: Create asynchronous DSPy programs for handling concurrent requests.

**Requirements**:
- Use async/await with DSPy modules
- Handle multiple requests concurrently
- Implement rate limiting
- Measure throughput improvements

**Starter Code**:
```python
import asyncio
import dspy
from typing import List

class AsyncPredictor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought("question -> answer")
        self.semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests

    async def predict(self, question: str) -> str:
        async with self.semaphore:
            # TODO: Run prediction asynchronously
            # Note: DSPy might be blocking, use executor if needed
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.predictor(question=question).answer
            )
            return result

async def main():
    predictor = AsyncPredictor()

    questions = [
        "What is DSPy?",
        "How do modules work?",
        "What is optimization?",
        # TODO: Add more questions
    ]

    # TODO: Run predictions concurrently
    results = await asyncio.gather(*[
        predictor.predict(q) for q in questions
    ])

    return results

# Run async program
if __name__ == "__main__":
    results = asyncio.run(main())
    for q, r in zip(questions, results):
        print(f"Q: {q}")
        print(f"A: {r}\n")
```

**Expected Output**: Multiple answers processed concurrently

**Hints**:
- Use `asyncio.gather()` for concurrent execution
- Implement semaphore to limit concurrent requests
- Consider threading vs async for blocking operations

**Advanced Challenge**: Stream responses as they arrive (async generators)

---

## Exercise 3: Implement Comprehensive Debugging

**Objective**: Build debugging and tracing infrastructure for complex DSPy programs.

**Requirements**:
- Implement step-by-step execution trace
- Log all LM calls with inputs/outputs
- Visualize execution flow
- Identify bottlenecks and failures

**Starter Code**:
```python
import dspy
import json
import logging
from typing import Any, Dict, List
from datetime import datetime

class DebugModule(dspy.Module):
    def __init__(self, enable_trace: bool = True):
        super().__init__()
        self.enable_trace = enable_trace
        self.trace: List[Dict[str, Any]] = []
        self.setup_logging()

    def setup_logging(self):
        # TODO: Configure logging with file output
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler("dspy_debug.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)

    def log_step(self, step_name: str, inputs: Dict, outputs: Any):
        # TODO: Log step execution
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step_name,
            "inputs": inputs,
            "outputs": outputs,
        }
        self.trace.append(entry)
        self.logger.info(f"Step {step_name}: {json.dumps(entry, indent=2)}")

    def forward(self, **kwargs):
        # TODO: Execute with debugging
        pass

    def print_trace(self):
        # TODO: Pretty-print execution trace
        for i, entry in enumerate(self.trace, 1):
            print(f"\n{'='*60}")
            print(f"Step {i}: {entry['step']}")
            print(f"Inputs: {json.dumps(entry['inputs'], indent=2)}")
            print(f"Outputs: {json.dumps(entry['outputs'], indent=2)}")

# Test with complex program
program = DebugModule()
# TODO: Run program and inspect trace
program.print_trace()
```

**Expected Output**: Detailed trace file and formatted output showing execution flow

**Hints**:
- Include timestamps for performance analysis
- Log both successes and failures
- Consider sensitive data redaction

**Advanced Challenge**: Implement interactive debugging (breakpoints, step through)

---

## Exercise 4: Performance Optimization

**Objective**: Profile and optimize a DSPy application for production use.

**Requirements**:
- Profile memory usage and execution time
- Identify bottlenecks
- Implement optimizations (batching, quantization, etc.)
- Measure improvement

**Starter Code**:
```python
import dspy
import time
import tracemalloc
from typing import List

class PerformanceProfiler:
    def __init__(self):
        self.timings = {}
        self.memory_usage = {}

    def profile_function(self, name: str, func, *args, **kwargs):
        # TODO: Profile function execution
        tracemalloc.start()
        start_time = time.perf_counter()

        result = func(*args, **kwargs)

        elapsed = time.perf_counter() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.timings[name] = elapsed
        self.memory_usage[name] = peak / 1024 / 1024  # MB

        return result

    def report(self):
        # TODO: Print performance report
        print("\nPerformance Report")
        print("=" * 50)
        print(f"{'Function':<25} {'Time (s)':<12} {'Memory (MB)':<12}")
        print("-" * 50)

        for name in self.timings:
            print(f"{name:<25} {self.timings[name]:<12.3f} {self.memory_usage[name]:<12.1f}")

# Benchmark different approaches
profiler = PerformanceProfiler()

class NaiveApproach(dspy.Module):
    # TODO: Simple but slow implementation
    pass

class OptimizedApproach(dspy.Module):
    # TODO: Faster implementation with caching/batching
    pass

# Profile both
naive = NaiveApproach()
optimized = OptimizedApproach()

# TODO: Run benchmarks
profiler.profile_function("Naive", lambda: naive())
profiler.profile_function("Optimized", lambda: optimized())

profiler.report()
```

**Expected Output**: Performance comparison showing improvements

**Hints**:
- Use `timeit` for accurate timing measurements
- Profile memory with `tracemalloc`
- Consider hardware differences

**Advanced Challenge**: Implement model quantization for smaller models

---

## Exercise 5: Deployment Strategy

**Objective**: Package and deploy a DSPy application as a production service.

**Requirements**:
- Create a REST API wrapper around DSPy program
- Implement error handling and logging
- Add authentication and rate limiting
- Container support (Docker)

**Starter Code**:
```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
import dspy
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize DSPy program
dspy.configure(model="gpt-4")
program = dspy.ChainOfThought("question -> answer")

# Define API models
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    question: str
    answer: str
    confidence: float

# Create FastAPI app
app = FastAPI(title="DSPy QA API", version="1.0.0")

@app.post("/predict", response_model=QuestionResponse)
async def predict(request: QuestionRequest):
    try:
        # TODO: Validate input
        if not request.question:
            raise HTTPException(status_code=400, detail="Question required")

        # TODO: Run prediction
        result = program(question=request.question)

        # TODO: Add confidence score
        return QuestionResponse(
            question=request.question,
            answer=result.answer,
            confidence=0.95
        )

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/health")
async def health():
    # TODO: Health check
    return {"status": "healthy"}

# TODO: Add rate limiting middleware
# TODO: Add authentication

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Expected Output**: Running API server accessible at `localhost:8000`

**Hints**:
- Use FastAPI for async support
- Implement proper error handling
- Add request validation

**Advanced Challenge**: Deploy to cloud (AWS Lambda, Google Cloud Functions)

---

## Exercise 6: Monitoring and Observability

**Objective**: Implement monitoring for a deployed DSPy system.

**Requirements**:
- Track prediction latency and quality metrics
- Monitor system health (memory, CPU)
- Alert on anomalies
- Create dashboards

**Starter Code**:
```python
import dspy
from datetime import datetime
from collections import defaultdict
import json

class MetricsCollector:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = datetime.now()

    def record_prediction(self, latency: float, accuracy: float, input_length: int):
        # TODO: Record prediction metrics
        self.metrics["latency"].append(latency)
        self.metrics["accuracy"].append(accuracy)
        self.metrics["input_length"].append(input_length)

    def get_statistics(self):
        # TODO: Calculate aggregate statistics
        import statistics

        stats = {}
        for metric_name, values in self.metrics.items():
            if values:
                stats[metric_name] = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

        return stats

    def alert_on_anomaly(self, latency: float, threshold: float = 5.0):
        # TODO: Check for anomalies
        if latency > threshold:
            logger.warning(f"Slow prediction detected: {latency:.2f}s")
            return True
        return False

# Use in production
collector = MetricsCollector()

# Simulate predictions
for i in range(100):
    # TODO: Run prediction and record metrics
    latency = 0.5 + (i % 10) * 0.1  # Varying latency
    accuracy = 0.85 + (i % 5) * 0.02
    input_length = 50 + (i % 100)

    collector.record_prediction(latency, accuracy, input_length)

    if collector.alert_on_anomaly(latency):
        # TODO: Send alert
        pass

# Generate report
stats = collector.get_statistics()
print(json.dumps(stats, indent=2))
```

**Expected Output**: Metrics report with statistics

**Hints**:
- Use time-series database for long-term metrics
- Set reasonable alert thresholds
- Track both technical and business metrics

**Advanced Challenge**: Integration with Prometheus/Grafana

---

## Answer Key

Solutions are available in the `solutions/` directory:

- `exercise1_caching.py` - Caching implementation
- `exercise2_async.py` - Async operations
- `exercise3_debugging.py` - Debugging infrastructure
- `exercise4_optimization.py` - Performance optimization
- `exercise5_deployment.py` - API deployment
- `exercise6_monitoring.py` - Metrics and monitoring

## Review Checklist

- [ ] Code runs without errors
- [ ] All requirements met
- [ ] Performance improvements measured
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Documentation included
- [ ] Advanced challenge attempted (optional)

## Additional Resources

- [Chapter 7: Advanced Topics](../src/07-advanced-topics/)
- [Chapter 9: Troubleshooting](../src/09-appendices/02-troubleshooting.md)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)

---

**Difficulty Level**: Advanced
**Estimated Time**: 3-4 hours
**Next Steps**: Move to Chapter 8 (Case Studies) or practice with real data

# Caching and Performance: Building High-Performance DSPy Applications

## Introduction

Performance is crucial for production DSPy applications. Language model calls can be expensive and slow, making caching and optimization techniques essential for building responsive, cost-effective systems. This chapter explores comprehensive strategies for optimizing DSPy applications through intelligent caching, batching, and performance monitoring.

## Understanding Performance Bottlenecks

### Common Performance Issues

1. **Language Model Latency**: Each API call takes time (500ms-5s)
2. **API Rate Limits**: Providers limit request frequency
3. **Token Costs**: Large prompts and frequent calls increase costs
4. **Memory Usage**: Storing contexts and intermediate results
5. **I/O Operations**: Database queries, file reads, network calls

### Performance Metrics

```python
import time
from functools import wraps
from collections import defaultdict, deque

class PerformanceMonitor:
    """Monitor and track DSPy performance metrics."""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_times = {}

    def time_function(self, func_name):
        """Decorator to time function execution."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                end = time.time()
                self.metrics[f"{func_name}_duration"].append(end - start)
                return result
            return wrapper
        return decorator

    def record_metric(self, metric_name, value):
        """Record a custom metric."""
        self.metrics[metric_name].append(value)

    def get_statistics(self, metric_name, window=100):
        """Get statistics for a metric."""
        values = self.metrics[metric_name][-window:]
        if not values:
            return None

        return {
            "count": len(values),
            "average": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "latest": values[-1]
        }

# Global performance monitor
perf_monitor = PerformanceMonitor()
```

## Caching Strategies

### 1. Result Caching

```python
import hashlib
import pickle
from typing import Any, Optional
import redis
import json

class ResultCache:
    """Cache for DSPy module results."""

    def __init__(self, backend="memory", **kwargs):
        self.backend = backend
        self.setup_cache(backend, **kwargs)

    def setup_cache(self, backend, **kwargs):
        """Setup cache backend."""
        if backend == "memory":
            self.cache = {}
        elif backend == "redis":
            self.cache = redis.Redis(
                host=kwargs.get("host", "localhost"),
                port=kwargs.get("port", 6379),
                db=kwargs.get("db", 0)
            )
        elif backend == "file":
            self.cache_dir = kwargs.get("cache_dir", "./cache")
            import os
            os.makedirs(self.cache_dir, exist_ok=True)
        else:
            raise ValueError(f"Unsupported cache backend: {backend}")

    def _generate_key(self, module_name, args, kwargs):
        """Generate cache key from inputs."""
        # Create a deterministic key from function inputs
        key_data = {
            "module": module_name,
            "args": args,
            "kwargs": kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, module_name, args, kwargs) -> Optional[Any]:
        """Get cached result."""
        key = self._generate_key(module_name, args, kwargs)

        if self.backend == "memory":
            return self.cache.get(key)
        elif self.backend == "redis":
            cached = self.cache.get(key)
            if cached:
                return pickle.loads(cached)
        elif self.backend == "file":
            import os
            cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

        return None

    def set(self, module_name, args, kwargs, result):
        """Cache result."""
        key = self._generate_key(module_name, args, kwargs)

        if self.backend == "memory":
            self.cache[key] = result
        elif self.backend == "redis":
            self.cache.set(key, pickle.dumps(result))
        elif self.backend == "file":
            cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)

    def clear(self):
        """Clear cache."""
        if self.backend == "memory":
            self.cache.clear()
        elif self.backend == "redis":
            self.cache.flushdb()
        elif self.backend == "file":
            import shutil
            shutil.rmtree(self.cache_dir)
            import os
            os.makedirs(self.cache_dir, exist_ok=True)
```

### 2. Semantic Caching

```python
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticCache:
    """Cache that uses semantic similarity for matching."""

    def __init__(self, similarity_threshold=0.9, model_name="all-MiniLM-L6-v2"):
        self.similarity_threshold = similarity_threshold
        self.model = SentenceTransformer(model_name)
        self.cache = []  # List of (embedding, key, value) tuples

    def _get_embedding(self, text):
        """Get text embedding."""
        return self.model.encode(text)

    def _find_similar(self, query_embedding):
        """Find similar cached items."""
        similarities = []
        for cached_embedding, _, _ in self.cache:
            similarity = np.dot(query_embedding, cached_embedding)
            similarities.append(similarity)

        if similarities and max(similarities) >= self.similarity_threshold:
            best_match_idx = np.argmax(similarities)
            return self.cache[best_match_idx][2]  # Return value
        return None

    def get(self, query_text):
        """Get semantically similar cached result."""
        query_embedding = self._get_embedding(query_text)
        return self._find_similar(query_embedding)

    def set(self, text, result):
        """Cache result with semantic indexing."""
        embedding = self._get_embedding(text)
        self.cache.append((embedding, text, result))

        # Limit cache size
        if len(self.cache) > 1000:
            self.cache = self.cache[-1000:]

    def clear(self):
        """Clear semantic cache."""
        self.cache = []
```

### 3. Hierarchical Caching

```python
class HierarchicalCache:
    """Multi-level cache for optimal performance."""

    def __init__(self):
        # L1: In-memory cache (fastest)
        self.l1_cache = ResultCache("memory")
        # L2: Redis cache (fast)
        self.l2_cache = ResultCache("redis", host="localhost", port=6379)
        # L3: File cache (persistent)
        self.l3_cache = ResultCache("file", cache_dir="./cache")

    def get(self, module_name, args, kwargs):
        """Get from cache, checking levels in order."""
        # L1 Cache
        result = self.l1_cache.get(module_name, args, kwargs)
        if result is not None:
            return result

        # L2 Cache
        result = self.l2_cache.get(module_name, args, kwargs)
        if result is not None:
            # Promote to L1
            self.l1_cache.set(module_name, args, kwargs, result)
            return result

        # L3 Cache
        result = self.l3_cache.get(module_name, args, kwargs)
        if result is not None:
            # Promote to L2 and L1
            self.l2_cache.set(module_name, args, kwargs, result)
            self.l1_cache.set(module_name, args, kwargs, result)
            return result

        return None

    def set(self, module_name, args, kwargs, result):
        """Set in all cache levels."""
        self.l1_cache.set(module_name, args, kwargs, result)
        self.l2_cache.set(module_name, args, kwargs, result)
        self.l3_cache.set(module_name, args, kwargs, result)
```

## Batching and Bulk Processing

### 1. Batch Processing Module

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any

class BatchProcessor:
    """Process multiple items in batches for efficiency."""

    def __init__(self, batch_size=10, max_workers=4):
        self.batch_size = batch_size
        self.max_workers = max_workers

    def process_batch(self, items, process_func):
        """Process items in batches."""
        results = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = self._process_single_batch(batch, process_func)
            results.extend(batch_results)
        return results

    def _process_single_batch(self, batch, process_func):
        """Process a single batch."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_func, item) for item in batch]
            return [future.result() for future in futures]

    async def process_batch_async(self, items, process_func):
        """Process items in batches asynchronously."""
        results = []
        semaphore = asyncio.Semaphore(self.max_workers)

        async def process_with_semaphore(item):
            async with semaphore:
                return await process_func(item)

        tasks = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_tasks = [process_with_semaphore(item) for item in batch]
            tasks.extend(batch_tasks)

        results = await asyncio.gather(*tasks)
        return results
```

### 2. Optimized Module with Caching

```python
class OptimizedModule(dspy.Module):
    """DSPy module with built-in caching and batching."""

    def __init__(self, cache=None, batch_size=5):
        super().__init__()
        self.cache = cache or HierarchicalCache()
        self.batch_size = batch_size
        self.batch_processor = BatchProcessor(batch_size=batch_size)
        self.pending_requests = []

    @perf_monitor.time_function("cached_forward")
    def forward(self, *args, **kwargs):
        """Forward pass with caching."""
        # Check cache first
        cache_key = f"{self.__class__.__name__}"
        cached_result = self.cache.get(cache_key, args, kwargs)

        if cached_result is not None:
            perf_monitor.record_metric("cache_hit", 1)
            return cached_result

        # Cache miss - process normally
        perf_monitor.record_metric("cache_miss", 1)
        result = self._forward_impl(*args, **kwargs)

        # Cache result
        self.cache.set(cache_key, args, kwargs, result)

        return result

    def _forward_impl(self, *args, **kwargs):
        """Implement actual forward logic."""
        # Override in subclasses
        raise NotImplementedError

    def batch_forward(self, batch_args, batch_kwargs=None):
        """Process multiple forward passes in batch."""
        if batch_kwargs is None:
            batch_kwargs = [{}] * len(batch_args)

        # Combine args and kwargs for cache lookup
        requests = list(zip(batch_args, batch_kwargs))

        # Check cache for each request
        uncached_requests = []
        uncached_indices = []
        cached_results = [None] * len(requests)

        for i, (args, kwargs) in enumerate(requests):
            cache_key = f"{self.__class__.__name__}"
            result = self.cache.get(cache_key, args, kwargs)
            if result is not None:
                cached_results[i] = result
            else:
                uncached_requests.append((args, kwargs))
                uncached_indices.append(i)

        # Process uncached requests in batch
        if uncached_requests:
            batch_results = self._batch_forward_impl(uncached_requests)

            # Update cache and results
            for i, (args, kwargs) in enumerate(uncached_requests):
                result = batch_results[i]
                cache_key = f"{self.__class__.__name__}"
                self.cache.set(cache_key, args, kwargs, result)
                cached_results[uncached_indices[i]] = result

        return cached_results

    def _batch_forward_impl(self, requests):
        """Implement batch processing logic."""
        # Override in subclasses for batch optimization
        results = []
        for args, kwargs in requests:
            result = self._forward_impl(*args, **kwargs)
            results.append(result)
        return results
```

## Memory Optimization

### 1. Memory Pool

```python
import weakref
from collections import OrderedDict

class MemoryPool:
    """Memory pool for reusing objects and managing memory."""

    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.pool = OrderedDict()
        self.references = weakref.WeakSet()

    def get(self, obj_type):
        """Get object from pool or create new."""
        key = obj_type
        if key in self.pool:
            obj = self.pool.pop(key)
            # Move to end (most recently used)
            self.pool[key] = obj
            return obj
        else:
            return obj_type()

    def release(self, obj, obj_type=None):
        """Release object back to pool."""
        if obj_type is None:
            obj_type = type(obj)

        if len(self.pool) < self.max_size and obj_type not in self.pool:
            self.pool[obj_type] = obj

    def clear(self):
        """Clear memory pool."""
        self.pool.clear()
```

### 2. Context Manager for Large Objects

```python
class ContextWindow:
    """Manage context window size to prevent memory issues."""

    def __init__(self, max_tokens=4096, tokenizer=None):
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.contexts = []

    def add_context(self, text):
        """Add context while managing window size."""
        # Estimate tokens (rough approximation)
        estimated_tokens = len(text.split()) * 1.3

        # Remove old contexts if window is full
        while self._total_tokens() + estimated_tokens > self.max_tokens and self.contexts:
            self.contexts.pop(0)

        self.contexts.append(text)

    def _total_tokens(self):
        """Estimate total tokens in contexts."""
        return sum(len(ctx.split()) * 1.3 for ctx in self.contexts)

    def get_context(self):
        """Get current context."""
        return "\n".join(self.contexts)

    def clear(self):
        """Clear all contexts."""
        self.contexts = []
```

## Performance Monitoring and Analytics

### 1. Performance Profiler

```python
import cProfile
import pstats
import io
from contextlib import contextmanager

class PerformanceProfiler:
    """Profile DSPy application performance."""

    def __init__(self):
        self.profiler = None

    @contextmanager
    def profile(self):
        """Context manager for profiling."""
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        try:
            yield
        finally:
            self.profiler.disable()

    def get_stats(self, sort_by='cumulative'):
        """Get profiling statistics."""
        if not self.profiler:
            return None

        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats(sort_by)
        ps.print_stats()
        return s.getvalue()

    def get_hotspots(self, top_n=10):
        """Get performance hotspots."""
        if not self.profiler:
            return []

        stats = pstats.Stats(self.profiler)
        return stats.get_stats_profile().func_profiles[:top_n]
```

### 2. Real-time Performance Dashboard

```python
import threading
import time
from collections import deque

class PerformanceDashboard:
    """Real-time performance monitoring dashboard."""

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.metrics = {
            'latency': deque(maxlen=window_size),
            'throughput': deque(maxlen=window_size),
            'error_rate': deque(maxlen=window_size),
            'cache_hit_rate': deque(maxlen=window_size),
            'memory_usage': deque(maxlen=window_size)
        }
        self.running = False
        self.thread = None

    def start_monitoring(self, update_interval=1):
        """Start real-time monitoring."""
        self.running = True
        self.thread = threading.Thread(
            target=self._monitor_loop,
            args=(update_interval,),
            daemon=True
        )
        self.thread.start()

    def stop_monitoring(self):
        """Stop monitoring."""
        self.running = False
        if self.thread:
            self.thread.join()

    def _monitor_loop(self, update_interval):
        """Monitoring loop."""
        while self.running:
            self._collect_metrics()
            time.sleep(update_interval)

    def _collect_metrics(self):
        """Collect current metrics."""
        import psutil
        process = psutil.Process()

        # Memory usage
        self.metrics['memory_usage'].append(process.memory_info().rss / 1024 / 1024)  # MB

    def record_latency(self, latency):
        """Record request latency."""
        self.metrics['latency'].append(latency)

    def record_throughput(self, requests_per_second):
        """Record throughput."""
        self.metrics['throughput'].append(requests_per_second)

    def record_error(self):
        """Record an error."""
        self.metrics['error_rate'].append(1)
        # Also record a 0 for successful requests to maintain ratio
        # In practice, you'd track both successes and errors separately

    def record_cache_hit(self):
        """Record cache hit."""
        self.metrics['cache_hit_rate'].append(1)

    def record_cache_miss(self):
        """Record cache miss."""
        self.metrics['cache_hit_rate'].append(0)

    def get_summary(self):
        """Get performance summary."""
        summary = {}
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    'current': values[-1],
                    'average': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values)
                }
        return summary
```

## Optimization Techniques

### 1. Prompt Optimization

```python
class PromptOptimizer:
    """Optimize prompts for better performance and cost efficiency."""

    def __init__(self):
        self.optimization_history = []

    def optimize_prompt_length(self, prompt, target_length=1000):
        """Optimize prompt to reduce length while maintaining effectiveness."""
        if len(prompt) <= target_length:
            return prompt

        # Remove redundant whitespace
        optimized = re.sub(r'\s+', ' ', prompt)

        # Remove examples if too long
        if len(optimized) > target_length:
            lines = optimized.split('\n')
            # Keep only essential parts
            essential_lines = [
                line for line in lines
                if not line.strip().startswith('# Example')
            ]
            optimized = '\n'.join(essential_lines[:len(essential_lines)//2])

        return optimized

    def optimize_examples(self, examples, max_examples=3):
        """Select most diverse examples."""
        if len(examples) <= max_examples:
            return examples

        # Simple diversity selection (could be more sophisticated)
        selected = []
        for i, example in enumerate(examples):
            if i % max(len(examples) // max_examples, 1) == 0:
                selected.append(example)
            if len(selected) >= max_examples:
                break

        return selected
```

### 2. Model Selection Optimization

```python
class ModelOptimizer:
    """Optimize model selection based on task complexity."""

    def __init__(self):
        self.model_costs = {
            "gpt-3.5-turbo": 0.002,  # per 1K tokens
            "gpt-4": 0.03,
            "gpt-4-turbo": 0.01
        }
        self.model_speeds = {
            "gpt-3.5-turbo": 1.0,  # relative speed
            "gpt-4": 0.2,
            "gpt-4-turbo": 0.5
        }

    def select_model(self, task_complexity, speed_priority=False):
        """Select optimal model based on task complexity."""
        if task_complexity < 0.3:
            return "gpt-3.5-turbo"
        elif task_complexity < 0.7:
            return "gpt-4-turbo" if not speed_priority else "gpt-3.5-turbo"
        else:
            return "gpt-4"

    def estimate_cost(self, model, prompt_tokens, completion_tokens):
        """Estimate API cost."""
        cost_per_1k = self.model_costs[model]
        total_tokens = prompt_tokens + completion_tokens
        return (total_tokens / 1000) * cost_per_1k
```

## Putting It All Together

### High-Performance RAG System

```python
class HighPerformanceRAG(dspy.Module):
    """Optimized RAG system with all performance enhancements."""

    def __init__(self, cache=None, batch_size=5):
        super().__init__()
        self.cache = cache or HierarchicalCache()
        self.batch_processor = BatchProcessor(batch_size=batch_size)
        self.context_window = ContextWindow(max_tokens=3000)
        self.prompt_optimizer = PromptOptimizer()
        self.model_optimizer = ModelOptimizer()
        self.dashboard = PerformanceDashboard()

        # Components
        self.retrieve = dspy.Retrieve(k=5)
        self.rank = dspy.Predict("query, documents -> ranked_documents")
        self.generate = dspy.Predict("context, query -> answer")

    @perf_monitor.time_function("rag_forward")
    def forward(self, query):
        """Optimized forward pass."""
        start_time = time.time()

        # Check cache
        cached = self.cache.get("rag", (query,), {})
        if cached:
            self.dashboard.record_cache_hit()
            return cached

        self.dashboard.record_cache_miss()

        # Retrieve documents
        retrieved = self.retrieve(query=query)
        documents = retrieved.passages

        # Rank documents (can be batched)
        ranked_result = self.rank(query=query, documents="\n".join(documents))
        ranked_docs = ranked_result.ranked_documents.split('\n')

        # Optimize context window
        self.context_window.clear()
        for doc in ranked_docs:
            self.context_window.add_context(doc)

        # Generate answer with optimized prompt
        context = self.context_window.get_context()
        optimized_prompt = self.prompt_optimizer.optimize_prompt_length(
            f"Context: {context}\nQuery: {query}"
        )

        result = self.generate(context=context, query=query)

        # Cache result
        final_result = dspy.Prediction(
            answer=result.answer,
            context=ranked_docs
        )
        self.cache.set("rag", (query,), {}, final_result)

        # Record metrics
        latency = time.time() - start_time
        self.dashboard.record_latency(latency)

        return final_result

    def batch_forward(self, queries):
        """Process multiple queries efficiently."""
        start_time = time.time()

        # Check cache for all queries
        uncached_queries = []
        uncached_indices = []
        cached_results = [None] * len(queries)

        for i, query in enumerate(queries):
            cached = self.cache.get("rag", (query,), {})
            if cached:
                cached_results[i] = cached
                self.dashboard.record_cache_hit()
            else:
                uncached_queries.append(query)
                uncached_indices.append(i)
                self.dashboard.record_cache_miss()

        # Process uncached queries in batch
        if uncached_queries:
            batch_results = self._batch_process_queries(uncached_queries)

            # Cache results and fill return array
            for i, result in enumerate(batch_results):
                query = uncached_queries[i]
                idx = uncached_indices[i]
                self.cache.set("rag", (query,), {}, result)
                cached_results[idx] = result

        # Record throughput
        throughput = len(queries) / (time.time() - start_time)
        self.dashboard.record_throughput(throughput)

        return cached_results

    def _batch_process_queries(self, queries):
        """Process multiple queries in batch."""
        # Retrieve all documents
        all_documents = []
        for query in queries:
            retrieved = self.retrieve(query=query)
            all_documents.append(retrieved.passages)

        # Rank documents in parallel
        def rank_query(args):
            query, docs = args
            rank_result = self.rank(query=query, documents="\n".join(docs))
            return rank_result.ranked_documents.split('\n')

        ranked_results = self.batch_processor.process_batch(
            list(zip(queries, all_documents)),
            rank_query
        )

        # Generate answers
        def generate_answer(args):
            query, docs = args
            context = "\n".join(docs)
            result = self.generate(context=context, query=query)
            return dspy.Prediction(answer=result.answer, context=docs)

        answers = self.batch_processor.process_batch(
            list(zip(queries, ranked_results)),
            generate_answer
        )

        return answers
```

## Best Practices

### 1. Cache Strategy
- Use hierarchical caching for optimal hit rates
- Implement cache warming for frequently accessed data
- Set appropriate TTL values based on data volatility
- Monitor cache hit rates and adjust strategies

### 2. Batching Strategy
- Batch requests when possible to reduce overhead
- Balance batch size against latency requirements
- Use async processing for independent operations
- Implement backpressure for high-throughput systems

### 3. Memory Management
- Use context windows to limit memory usage
- Implement memory pools for object reuse
- Monitor memory usage and implement limits
- Use generators for large datasets

### 4. Performance Monitoring
- Track key metrics: latency, throughput, error rate
- Set up alerts for performance degradation
- Use profiling to identify bottlenecks
- Continuously optimize based on metrics

## Key Takeaways

1. **Caching dramatically reduces** API costs and latency
2. **Batch processing improves** throughput efficiency
3. **Memory optimization prevents** system overload
4. **Performance monitoring** is essential for optimization
5. **Hierarchical strategies** provide best results
6. **Context management** balances quality and performance

## Next Steps

In the next section, we'll explore **Async and Streaming** techniques for building real-time DSPy applications that can handle continuous data flows and concurrent operations.
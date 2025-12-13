"""
Chapter 7: Caching and Performance Optimization Examples

This example demonstrates advanced caching strategies and performance
optimization techniques in DSPy applications.
"""

import os
import time
import json
import hashlib
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from functools import wraps, lru_cache
from pathlib import Path

import dspy
from dspy import Module, Predict, Signature, InputField, OutputField
from dspy.teleprompter import BootstrapFewShot

# Configure LM
turbo = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=1000)
dspy.settings.configure(lm=turbo)


class AdvancedCache:
    """Multi-level cache with memory, disk, and Redis support."""

    def __init__(self,
                 memory_size: int = 1000,
                 disk_cache_dir: str = ".dspy_cache",
                 redis_url: Optional[str] = None):
        self.memory_cache = {}
        self.memory_access = {}
        self.memory_size = memory_size

        # Disk cache
        self.cache_dir = Path(disk_cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Redis cache (if available)
        self.redis_client = None
        if redis_url:
            try:
                import redis
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
            except (ImportError, Exception):
                print("Redis not available, using memory and disk cache only")

    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate cache key from data."""
        if isinstance(data, (str, int, float, bool)):
            content = str(data)
        elif isinstance(data, (dict, list)):
            content = json.dumps(data, sort_keys=True)
        else:
            content = str(data.__dict__)

        return f"{prefix}:{hashlib.md5(content.encode()).hexdigest()}"

    def get(self, prefix: str, data: Any) -> Optional[Any]:
        """Get value from cache (memory -> disk -> Redis)."""
        key = self._generate_key(prefix, data)

        # Check memory cache
        if key in self.memory_cache:
            self.memory_access[key] = time.time()
            return self.memory_cache[key]

        # Check disk cache
        disk_path = self.cache_dir / f"{key}.json"
        if disk_path.exists():
            try:
                with open(disk_path, 'r') as f:
                    value = json.load(f)
                # Promote to memory
                self.set(prefix, data, value, memory_only=True)
                return value
            except Exception:
                pass

        # Check Redis cache
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value:
                    value = json.loads(value)
                    # Promote to memory and disk
                    self.set(prefix, data, value, memory_only=True)
                    self._save_to_disk(key, value)
                    return value
            except Exception:
                pass

        return None

    def set(self, prefix: str, data: Any, value: Any, memory_only: bool = False):
        """Set value in cache."""
        key = self._generate_key(prefix, data)

        # Update memory cache
        if len(self.memory_cache) >= self.memory_size:
            # Evict LRU
            lru_key = min(self.memory_access.keys(),
                         key=self.memory_access.get)
            del self.memory_cache[lru_key]
            del self.memory_access[lru_key]

        self.memory_cache[key] = value
        self.memory_access[key] = time.time()

        if not memory_only:
            # Save to disk
            self._save_to_disk(key, value)

            # Save to Redis
            if self.redis_client:
                try:
                    self.redis_client.setex(
                        key,
                        timedelta(hours=24),
                        json.dumps(value)
                    )
                except Exception:
                    pass

    def _save_to_disk(self, key: str, value: Any):
        """Save value to disk cache."""
        try:
            with open(self.cache_dir / f"{key}.json", 'w') as f:
                json.dump(value, f)
        except Exception:
            pass

    def clear(self, level: str = "all"):
        """Clear cache at specified level."""
        if level in ("all", "memory"):
            self.memory_cache.clear()
            self.memory_access.clear()

        if level in ("all", "disk"):
            for file in self.cache_dir.glob("*.json"):
                file.unlink()

        if level in ("all", "redis") and self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception:
                pass


class PerformanceMonitor:
    """Monitor and track DSPy operation performance."""

    def __init__(self):
        self.metrics = {
            "operations": [],
            "total_tokens": 0,
            "total_cost": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }

    def start_operation(self, operation_type: str, details: Dict = None):
        """Start tracking an operation."""
        return {
            "type": operation_type,
            "start_time": time.time(),
            "details": details or {}
        }

    def end_operation(self, operation: Dict, result: Any = None):
        """End tracking an operation."""
        end_time = time.time()
        duration = end_time - operation["start_time"]

        metric = {
            "type": operation["type"],
            "duration": duration,
            "timestamp": end_time,
            "details": operation["details"],
            "success": result is not None
        }

        self.metrics["operations"].append(metric)
        return metric

    def get_summary(self) -> Dict:
        """Get performance summary."""
        ops = self.metrics["operations"]
        if not ops:
            return {}

        summary = {
            "total_operations": len(ops),
            "avg_duration": sum(op["duration"] for op in ops) / len(ops),
            "cache_hit_rate": (
                self.metrics["cache_hits"] /
                (self.metrics["cache_hits"] + self.metrics["cache_misses"])
                if (self.metrics["cache_hits"] + self.metrics["cache_misses"]) > 0
                else 0
            ),
            "total_cost": self.metrics["total_cost"],
            "total_tokens": self.metrics["total_tokens"]
        }

        # Group by operation type
        summary["by_type"] = {}
        for op in ops:
            op_type = op["type"]
            if op_type not in summary["by_type"]:
                summary["by_type"][op_type] = {
                    "count": 0,
                    "total_duration": 0,
                    "avg_duration": 0
                }

            summary["by_type"][op_type]["count"] += 1
            summary["by_type"][op_type]["total_duration"] += op["duration"]
            summary["by_type"][op_type]["avg_duration"] = (
                summary["by_type"][op_type]["total_duration"] /
                summary["by_type"][op_type]["count"]
            )

        return summary


class CachedPredict:
    """Predict wrapper with advanced caching capabilities."""

    def __init__(self,
                 signature: Signature,
                 cache: AdvancedCache = None,
                 monitor: PerformanceMonitor = None):
        self.predict = Predict(signature)
        self.cache = cache or AdvancedCache()
        self.monitor = monitor or PerformanceMonitor()

    def __call__(self, **kwargs):
        """Execute with caching and monitoring."""
        # Check cache first
        cached_result = self.cache.get("predict", kwargs)
        if cached_result:
            self.monitor.metrics["cache_hits"] += 1
            return cached_result

        self.monitor.metrics["cache_misses"] += 1

        # Monitor the operation
        op = self.monitor.start_operation("predict", {"signature": self.predict.signature})

        try:
            # Execute the prediction
            result = self.predict(**kwargs)

            # Cache the result
            self.cache.set("predict", kwargs, result)

            # Track metrics
            self.monitor.end_operation(op, result)

            return result

        except Exception as e:
            self.monitor.end_operation(op, None)
            raise


@dataclass
class BatchRequest:
    """Request for batch processing."""
    inputs: List[Dict[str, Any]]
    batch_size: int = 10
    max_retries: int = 3
    parallel: bool = True


class BatchProcessor:
    """Process multiple DSPy requests efficiently."""

    def __init__(self, predict_func, max_workers: int = 5):
        self.predict_func = predict_func
        self.max_workers = max_workers

    async def process_batch(self, request: BatchRequest) -> List[Any]:
        """Process a batch of requests."""
        if request.parallel:
            return await self._process_parallel(request)
        else:
            return await self._process_sequential(request)

    async def _process_sequential(self, request: BatchRequest) -> List[Any]:
        """Process batch sequentially."""
        results = []

        for i in range(0, len(request.inputs), request.batch_size):
            batch = request.inputs[i:i + request.batch_size]
            batch_results = await self._process_batch_with_retries(
                batch, request.max_retries
            )
            results.extend(batch_results)

        return results

    async def _process_parallel(self, request: BatchRequest) -> List[Any]:
        """Process batch in parallel."""
        semaphore = asyncio.Semaphore(self.max_workers)

        async def process_with_semaphore(item):
            async with semaphore:
                return await self._process_item_with_retries(
                    item, request.max_retries
                )

        tasks = [process_with_semaphore(item) for item in request.inputs]
        return await asyncio.gather(*tasks)

    async def _process_batch_with_retries(self,
                                         batch: List[Dict],
                                         max_retries: int) -> List[Any]:
        """Process a single batch with retries."""
        for attempt in range(max_retries + 1):
            try:
                # Simulate batch processing
                results = []
                for item in batch:
                    result = self.predict_func(**item)
                    results.append(result)
                return results
            except Exception as e:
                if attempt == max_retries:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    async def _process_item_with_retries(self,
                                        item: Dict,
                                        max_retries: int) -> Any:
        """Process a single item with retries."""
        for attempt in range(max_retries + 1):
            try:
                return self.predict_func(**item)
            except Exception as e:
                if attempt == max_retries:
                    raise
                await asyncio.sleep(2 ** attempt)


class ResourceOptimizer:
    """Optimize resource usage in DSPy applications."""

    def __init__(self):
        self.token_counts = {}
        self.cost_per_token = {
            "gpt-3.5-turbo": 0.000002,
            "gpt-4": 0.00003,
            "gpt-4-turbo": 0.00001
        }

    def count_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """Count tokens in text."""
        # Simple estimation (in practice, use tiktoken)
        return len(text.split()) * 1.3  # Rough estimate

    def estimate_cost(self, text: str, model: str = "gpt-3.5-turbo") -> float:
        """Estimate cost for processing text."""
        tokens = self.count_tokens(text, model)
        return tokens * self.cost_per_token.get(model, 0.000002)

    def optimize_prompt(self, prompt: str, target_length: int = 1000) -> str:
        """Optimize prompt length while preserving meaning."""
        words = prompt.split()
        if len(words) <= target_length:
            return prompt

        # Simple truncation (in practice, use more sophisticated methods)
        return " ".join(words[:target_length])

    def suggest_model(self, complexity: str, budget: float = 0.01) -> str:
        """Suggest best model based on complexity and budget."""
        models = {
            "simple": ("gpt-3.5-turbo", 0.002),
            "moderate": ("gpt-4-turbo", 0.01),
            "complex": ("gpt-4", 0.03)
        }

        for level, (model, cost_per_1k) in models.items():
            if complexity == level or budget >= cost_per_1k:
                return model

        return "gpt-3.5-turbo"  # Default


# Example Usage
def main():
    print("DSPy Caching and Performance Optimization Examples")
    print("=" * 60)

    # Initialize components
    cache = AdvancedCache()
    monitor = PerformanceMonitor()

    # Create cached predict
    class QuestionAnswering(Signature):
        """Answer questions based on the given context."""
        question = InputField(desc="The question to answer")
        context = InputField(desc="Relevant context for answering")
        answer = OutputField(desc="The answer to the question")

    qa_predict = CachedPredict(QuestionAnswering, cache, monitor)

    # Example 1: Basic caching
    print("\n1. Basic Caching Demo")
    print("-" * 40)

    question = "What is DSPy?"
    context = "DSPy is a framework for programming language models."

    # First call (cache miss)
    start_time = time.time()
    result1 = qa_predict(question=question, context=context)
    duration1 = time.time() - start_time

    # Second call (cache hit)
    start_time = time.time()
    result2 = qa_predict(question=question, context=context)
    duration2 = time.time() - start_time

    print(f"First call (cache miss): {duration1:.3f}s")
    print(f"Second call (cache hit): {duration2:.3f}s")
    print(f"Speedup: {duration1/duration2:.1f}x")

    # Example 2: Performance monitoring
    print("\n2. Performance Monitoring")
    print("-" * 40)

    # Make multiple predictions
    questions = [
        ("What is machine learning?", "ML is a subset of AI."),
        ("What is Python?", "Python is a programming language."),
        ("What is caching?", "Caching stores computed results for reuse.")
    ]

    for q, ctx in questions:
        qa_predict(question=q, context=ctx)

    # Get performance summary
    summary = monitor.get_summary()
    print(f"Total operations: {summary['total_operations']}")
    print(f"Average duration: {summary['avg_duration']:.3f}s")
    print(f"Cache hit rate: {summary['cache_hit_rate']:.1%}")

    # Example 3: Batch processing
    print("\n3. Batch Processing Demo")
    print("-" * 40)

    async def demo_batch_processing():
        batch_predict = CachedPredict(QuestionAnswering)
        processor = BatchProcessor(batch_predict, max_workers=3)

        # Create batch request
        batch_request = BatchRequest(
            inputs=[
                {"question": q, "context": ctx}
                for q, ctx in questions
            ],
            batch_size=2,
            parallel=True
        )

        start_time = time.time()
        results = await processor.process_batch(batch_request)
        duration = time.time() - start_time

        print(f"Processed {len(results)} items in {duration:.3f}s")
        print(f"Throughput: {len(results)/duration:.1f} items/second")

    asyncio.run(demo_batch_processing())

    # Example 4: Resource optimization
    print("\n4. Resource Optimization")
    print("-" * 40)

    optimizer = ResourceOptimizer()

    # Analyze different prompts
    prompts = [
        "Answer this question briefly",
        "Please provide a comprehensive answer to the following question, considering all aspects and providing detailed explanations",
        "What is the answer?"
    ]

    for i, prompt in enumerate(prompts, 1):
        cost = optimizer.estimate_cost(prompt)
        optimized = optimizer.optimize_prompt(prompt, 20)

        print(f"\nPrompt {i}:")
        print(f"  Original length: {len(prompt)} chars")
        print(f"  Estimated cost: ${cost:.6f}")
        print(f"  Optimized: {optimized}...")

    # Example 5: Model selection
    print("\n5. Model Selection")
    print("-" * 40)

    complexities = ["simple", "moderate", "complex"]
    budgets = [0.001, 0.01, 0.05]

    for complexity in complexities:
        for budget in budgets:
            model = optimizer.suggest_model(complexity, budget)
            print(f"  {complexity:10} task, ${budget:.3f} budget → {model}")

    # Example 6: Cache management
    print("\n6. Cache Management")
    print("-" * 40)

    print("Cache levels:")
    print(f"  Memory: {len(cache.memory_cache)} items")
    print(f"  Disk: {len(list(cache.cache_dir.glob('*.json')))} files")
    print(f"  Redis: {'Available' if cache.redis_client else 'Not available'}")

    # Clear specific cache levels
    cache.clear(level="memory")
    print("\nCleared memory cache")
    print(f"  Memory: {len(cache.memory_cache)} items")

    print("\n7. Advanced Features")
    print("-" * 40)

    # Demonstrate cache persistence
    print("\nCache persistence demo:")
    test_data = {"test": "value", "number": 42}
    cache.set("demo", test_data, {"metadata": "test"})

    # Retrieve from different cache levels
    memory_result = cache.get("demo", test_data)
    print(f"  Retrieved from memory: {memory_result is not None}")

    # Clear and retrieve from disk
    cache.clear(level="memory")
    disk_result = cache.get("demo", test_data)
    print(f"  Retrieved from disk: {disk_result is not None}")

    print("\n✓ All caching and performance optimization examples completed!")


if __name__ == "__main__":
    main()
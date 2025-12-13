"""
Exercise 6: Performance Optimization
Solution for Exercise 6 from Chapter 3

Task: Optimize a module for performance
- Implement caching
- Add batch processing
- Measure and compare performance
"""

import dspy
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

class OptimizedTextProcessor(dspy.Module):
    """Performance-optimized text processor with caching and batch processing."""

    def __init__(self, cache_size: int = 1000, max_workers: int = 4):
        super().__init__()

        # Cache configuration
        self.cache_size = cache_size
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Parallel processing
        self.max_workers = max_workers

        # Internal modules
        self.fast_classifier = dspy.Predict(
            "text -> category",
            temperature=0.1,  # Lower temperature for consistency
            max_tokens=20
        )

        self.fast_analyzer = dspy.Predict(
            "text -> sentiment complexity"
        )

        self.batch_summarizer = dspy.Predict(
            "batch_texts -> batch_summaries"
        )

    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for text."""
        # Use hash for fixed-length keys
        return hashlib.md5(text.encode()).hexdigest()[:16]

    def process_single(self, text: str, use_cache: bool = True) -> Dict[str, Any]:
        """Process a single text with optional caching."""

        start_time = time.time()

        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                self.cache_hits += 1
                cached_result = self.cache[cache_key].copy()
                cached_result["from_cache"] = True
                cached_result["processing_time"] = time.time() - start_time
                return cached_result

        # Cache miss - process the text
        self.cache_misses += 1

        try:
            # Classification
            category_result = self.fast_classifier(text=text)

            # Analysis
            analysis_result = self.fast_analyzer(text=text)

            # Compile result
            result = {
                "text": text,
                "category": category_result.category,
                "sentiment": analysis_result.sentiment,
                "complexity": analysis_result.complexity,
                "from_cache": False,
                "timestamp": datetime.now().isoformat()
            }

            # Store in cache
            if use_cache and len(self.cache) < self.cache_size:
                self.cache[cache_key] = {k: v for k, v in result.items() if k != "from_cache"}

        except Exception as e:
            result = {
                "text": text,
                "error": str(e),
                "from_cache": False,
                "timestamp": datetime.now().isoformat()
            }

        result["processing_time"] = time.time() - start_time
        return result

    def process_batch(self, texts: List[str], use_cache: bool = True, parallel: bool = True) -> Dict[str, Any]:
        """Process multiple texts efficiently."""

        start_time = time.time()
        results = []
        uncached_texts = []
        uncached_indices = []

        # Step 1: Check cache for all texts
        if use_cache:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self.cache:
                    cached_result = self.cache[cache_key].copy()
                    cached_result["from_cache"] = True
                    cached_result["index"] = i
                    results.append(cached_result)
                    self.cache_hits += 1
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    self.cache_misses += 1
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
            self.cache_misses += len(texts)

        # Step 2: Process uncached texts
        if uncached_texts:
            if parallel and len(uncached_texts) > 1:
                # Parallel processing
                processed_uncached = self._process_parallel(uncached_texts)
            else:
                # Sequential processing
                processed_uncached = [
                    self.process_single(text, use_cache=False)
                    for text in uncached_texts
                ]

            # Update cache with new results
            if use_cache:
                for text, result in zip(uncached_texts, processed_uncached):
                    if "error" not in result:
                        cache_key = self._get_cache_key(text)
                        if len(self.cache) < self.cache_size:
                            self.cache[cache_key] = {k: v for k, v in result.items() if k not in ["from_cache", "processing_time"]}

            # Add indices and combine with cached results
            for i, (text, result) in enumerate(zip(uncached_texts, processed_uncached)):
                result["index"] = uncached_indices[i]
                results.append(result)

        # Sort results by original index
        results.sort(key=lambda x: x["index"])
        for r in results:
            r.pop("index", None)

        total_time = time.time() - start_time

        return {
            "results": results,
            "total_texts": len(texts),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            "total_processing_time": total_time,
            "avg_time_per_text": total_time / len(texts),
            "parallel_used": parallel
        }

    def _process_parallel(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process texts in parallel."""

        results = [None] * len(texts)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.process_single, text, use_cache=False): i
                for i, text in enumerate(texts)
            }

            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    results[index] = {
                        "text": texts[index],
                        "error": str(e),
                        "from_cache": False
                    }

        return results

    def clear_cache(self):
        """Clear the cache."""
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.cache_hits + self.cache_misses
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / total if total > 0 else 0,
            "cache_utilization": len(self.cache) / self.cache_size
        }

def performance_comparison():
    """Compare performance of different optimization strategies."""

    processor = OptimizedTextProcessor()

    # Generate test data with duplicates for cache testing
    base_texts = [
        "This is a positive review about a great product.",
        "Negative experience with poor customer service.",
        "Neutral comment about average quality.",
        "Excellent service and fast delivery!",
        "Disappointing quality for the price paid."
    ]

    # Create test set with duplicates
    test_texts = base_texts * 10  # 50 texts with 5 unique ones
    test_texts.extend(base_texts[:2])  # Add more duplicates

    print("=" * 60)
    print("Performance Comparison")
    print("=" * 60)

    # Test 1: Sequential without cache
    print("\n1. Sequential Processing (No Cache):")
    start = time.time()
    processor.clear_cache()
    results1 = processor.process_batch(test_texts, use_cache=False, parallel=False)
    time1 = time.time() - start
    print(f"   Time: {time1:.3f}s")
    print(f"   Avg per text: {time1/len(test_texts):.4f}s")

    # Test 2: Sequential with cache
    print("\n2. Sequential Processing (With Cache):")
    start = time.time()
    processor.clear_cache()
    results2 = processor.process_batch(test_texts, use_cache=True, parallel=False)
    time2 = time.time() - start
    print(f"   Time: {time2:.3f}s")
    print(f"   Avg per text: {time2/len(test_texts):.4f}s")
    print(f"   Cache hit rate: {results2['cache_hit_rate']:.2%}")

    # Test 3: Parallel without cache
    print("\n3. Parallel Processing (No Cache):")
    start = time.time()
    processor.clear_cache()
    results3 = processor.process_batch(test_texts, use_cache=False, parallel=True)
    time3 = time.time() - start
    print(f"   Time: {time3:.3f}s")
    print(f"   Avg per text: {time3/len(test_texts):.4f}s")

    # Test 4: Parallel with cache
    print("\n4. Parallel Processing (With Cache):")
    start = time.time()
    processor.clear_cache()
    results4 = processor.process_batch(test_texts, use_cache=True, parallel=True)
    time4 = time.time() - start
    print(f"   Time: {time4:.3f}s")
    print(f"   Avg per text: {time4/len(test_texts):.4f}s")
    print(f"   Cache hit rate: {results4['cache_hit_rate']:.2%}")

    # Summary
    print("\n" + "=" * 60)
    print("Performance Summary:")
    print("-" * 60)
    print(f"Sequential (No Cache):    {time1:8.3f}s (100%)")
    print(f"Sequential (With Cache):   {time2:8.3f}s ({time2/time1*100:5.1f}%) - {((time1-time2)/time1*100):5.1f}% faster")
    print(f"Parallel (No Cache):       {time3:8.3f}s ({time3/time1*100:5.1f}%) - {((time1-time3)/time1*100):5.1f}% faster")
    print(f"Parallel (With Cache):     {time4:8.3f}s ({time4/time1*100:5.1f}%) - {((time1-time4)/time1*100):5.1f}% faster")

    # Best configuration
    best_time = min(time1, time2, time3, time4)
    best_config = ["Sequential (No Cache)", "Sequential (With Cache)", "Parallel (No Cache)", "Parallel (With Cache)"][[time1, time2, time3, time4].index(best_time)]
    print(f"\nBest Configuration: {best_config} ({best_time:.3f}s)")

    # Cache statistics
    cache_stats = processor.get_cache_stats()
    print(f"\nCache Statistics:")
    print(f"  Final Cache Size: {cache_stats['cache_size']}")
    print(f"  Cache Utilization: {cache_stats['cache_utilization']:.2%}")

def test_scalability():
    """Test scalability with different batch sizes."""

    processor = OptimizedTextProcessor()

    # Test different batch sizes
    batch_sizes = [10, 50, 100, 500, 1000]
    base_text = "This is a test text for scalability testing."

    print("\n" + "=" * 60)
    print("Scalability Test")
    print("=" * 60)

    results = []

    for size in batch_sizes:
        test_batch = [f"{base_text} #{i}" for i in range(size)]

        # Test parallel with cache
        processor.clear_cache()
        start = time.time()
        result = processor.process_batch(test_batch, use_cache=True, parallel=True)
        total_time = time.time() - start

        print(f"\nBatch Size: {size:4d} | Time: {total_time:6.3f}s | Avg: {total_time/size*1000:6.2f}ms/text")
        print(f"  Cache Hit Rate: {result['cache_hit_rate']:.2%}")

        results.append({
            "batch_size": size,
            "total_time": total_time,
            "avg_time_per_item": total_time / size
        })

    # Calculate throughput
    print("\n" + "=" * 60)
    print("Throughput Analysis:")
    print("-" * 60)

    for result in results:
        throughput = result["batch_size"] / result["total_time"]
        print(f"Size {result['batch_size']:4d}: {throughput:6.1f} texts/second")

def demonstrate_memory_efficiency():
    """Demonstrate memory-efficient processing."""

    processor = OptimizedTextProcessor(cache_size=100)  # Smaller cache for demo

    print("\n" + "=" * 60)
    print("Memory Efficiency Demonstration")
    print("=" * 60)

    # Process a large number of texts with limited cache
    unique_texts = [f"Unique text #{i} for memory testing." for i in range(200)]
    repeated_texts = unique_texts[:50] * 4  # 200 texts with only 50 unique

    print(f"\nProcessing {len(repeated_texts)} texts with {len(unique_texts)} unique values")
    print(f"Cache size limit: {processor.cache_size}")

    start = time.time()
    result = processor.process_batch(repeated_texts, use_cache=True, parallel=True)
    total_time = time.time() - start

    print(f"\nResults:")
    print(f"  Processing Time: {total_time:.3f}s")
    print(f"  Cache Hit Rate: {result['cache_hit_rate']:.2%}")
    print(f"  Cache Utilization: {len(processor.cache)}/{processor.cache_size}")

    # Show cache eviction behavior
    cache_stats = processor.get_cache_stats()
    print(f"\nCache Statistics:")
    print(f"  Items in Cache: {cache_stats['cache_size']}")
    print(f"  Total Cache Hits: {cache_stats['cache_hits']}")
    print(f"  Total Cache Misses: {cache_stats['cache_misses']}")
    print(f"  Memory Saved by Cache: ~{(cache_stats['cache_hits'] * 100) / len(repeated_texts):.1f}%")

def main():
    """Main function to run Exercise 6."""

    print("\n" + "=" * 60)
    print("Exercise 6: Performance Optimization")
    print("Optimizing DSPy modules with caching and batch processing")
    print("=" * 60)

    # Run performance comparisons
    performance_comparison()

    # Test scalability
    test_scalability()

    # Demonstrate memory efficiency
    demonstrate_memory_efficiency()

    print("\n" + "=" * 60)
    print("Exercise 6 Completed Successfully!")
    print("Key optimizations implemented:")
    print("  - Result caching for duplicate inputs")
    print("  - Parallel processing for batch operations")
    print("  - Efficient memory usage with cache limits")
    print("=" * 60)

if __name__ == "__main__":
    main()
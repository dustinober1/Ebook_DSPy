# Chapter 7 Exercises: Advanced Topics Mastery

## Overview

These exercises challenge you to apply advanced DSPy concepts to solve complex, real-world problems. You'll work with adapters, performance optimization, async programming, debugging, and deployment strategies to build production-ready applications.

## Exercise 1: Build a Custom Database Adapter

### Objective
Create a comprehensive database adapter for DSPy that can work with PostgreSQL and provide caching functionality.

### Problem
You need to build a database adapter that can store and retrieve DSPy predictions, handle connections efficiently, and provide automatic caching for frequently accessed data.

### Starter Code
```python
import dspy
from typing import Any, Dict, List, Optional
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
import json
import time

class DatabaseAdapter(dspy.Adapter):
    """TODO: Implement database adapter for DSPy."""

    def __init__(self, connection_string, pool_size=5):
        super().__init__()
        self.connection_string = connection_string
        self.pool_size = pool_size
        # TODO: Initialize connection pool and cache

    def store_prediction(self, prediction_id: str, prediction: dspy.Prediction):
        """TODO: Store prediction in database."""
        pass

    def get_prediction(self, prediction_id: str) -> Optional[dspy.Prediction]:
        """TODO: Retrieve prediction from database."""
        pass

    def search_predictions(self, query: Dict[str, Any], limit: int = 10) -> List[dspy.Prediction]:
        """TODO: Search predictions based on criteria."""
        pass

    def close(self):
        """TODO: Close all database connections."""
        pass

# TODO: Implement this function
def create_database_adapter(config: Dict[str, Any]) -> DatabaseAdapter:
    """Create and configure database adapter."""
    pass
```

### Tasks
1. Implement connection pooling with ThreadedConnectionPool
2. Add automatic caching with a configurable TTL
3. Implement table schema for storing DSPy predictions
4. Add search functionality with flexible query support
5. Implement connection health checks and auto-reconnection
6. Add metrics for monitoring performance

### Hints
- Use context managers for safe database operations
- Implement connection retry logic with exponential backoff
- Store predictions as JSON in the database
- Consider adding indexes for better query performance

### Expected Output
```
Database Adapter Statistics:
- Connections: 5/5
- Cache size: 45 entries
- Cache hit rate: 78%
- Average query time: 12ms
- Stored predictions: 156
```

---

## Exercise 2: Implement a High-Performance Caching System

### Objective
Create a multi-level caching system with L1 (memory), L2 (Redis), and L3 (disk) layers, with intelligent cache promotion and eviction policies.

### Problem
You need to build a caching system that can handle high-throughput DSPy operations with minimal latency and maximum cache hit rates.

### Starter Code
```python
import dspy
import redis
import pickle
import os
from typing import Any, Optional, Dict
import hashlib
import time

class AdvancedCacheSystem:
    """TODO: Implement multi-level caching system."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        # L1: Memory cache
        self.l1_cache = {}  # TODO: Implement LRU cache
        self.l1_size = config.get("l1_size", 1000)
        self.l1_ttl = config.get("l1_ttl", 300)

        # L2: Redis cache
        # TODO: Initialize Redis client
        self.l2_ttl = config.get("l2_ttl", 3600)

        # L3: Disk cache
        self.l3_path = config.get("l3_path", "./cache")
        self.l3_ttl = config.get("l3_ttl", 86400)

        # TODO: Initialize all cache layers

    def get(self, key: str) -> Optional[Any]:
        """TODO: Get value from cache (check L1, then L2, then L3)."""
        pass

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """TODO: Set value in all cache layers."""
        pass

    def promote_to_l1(self, key: str, value: Any):
        """TODO: Promote value to L1 cache with eviction."""
        pass

    def evict_from_l1(self):
        """TODO: Evict least recently used item from L1."""
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """TODO: Return cache statistics."""
        pass

# TODO: Implement this function
def benchmark_cache_system(cache: AdvancedCacheSystem):
    """Benchmark cache system performance."""
    pass
```

### Tasks
1. Implement LRU cache for L1 memory layer
2. Set up Redis connection for L2 cache with connection pooling
3. Implement disk-based L3 cache with size limits
4. Add intelligent promotion policies between layers
5. Implement cache warming strategies
6. Add comprehensive statistics and monitoring
7. Create performance benchmark suite

### Hints
- Use OrderedDict for LRU implementation
- Implement serialization/deserialization carefully
- Add cache warming for frequently accessed data
- Consider memory pressure when managing L1 cache

### Expected Output
```
Cache Performance Report:
===================
Total requests: 10,000
L1 hits: 7,234 (72.34%)
L2 hits: 2,456 (24.56%)
L3 hits: 310 (3.10%)
Cache misses: 0 (0.00%)
Average latency: 2.3ms
Total cache size: 1.2GB
```

---

## Exercise 3: Build an Async Streaming RAG System

### Objective
Create an asynchronous RAG system that can handle multiple concurrent queries, process document streams, and provide real-time responses.

### Problem
You need to build a streaming RAG system that can ingest documents in real-time, handle concurrent user queries, and provide low-latency responses with proper backpressure handling.

### Starter Code
```python
import dspy
import asyncio
import aiohttp
from typing import AsyncGenerator, List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import time

class AsyncStreamingRAG(dspy.Module):
    """TODO: Implement async streaming RAG system."""

    def __init__(self, concurrent_limit=10, stream_buffer_size=100):
        super().__init__()
        # TODO: Initialize async components
        self.concurrent_limit = concurrent_limit
        self.stream_buffer_size = stream_buffer_size

    async def ingest_document_stream(self, document_stream: AsyncGenerator[str, None]):
        """TODO: Ingest documents from stream."""
        pass

    async def query_stream(self, query_stream: AsyncGenerator[str, None]) -> AsyncGenerator[Dict[str, Any], None]:
        """TODO: Process queries from stream."""
        pass

    async def batch_query(self, queries: List[str]) -> List[Dict[str, Any]]:
        """TODO: Process multiple queries concurrently."""
        pass

    async def get_stats(self) -> Dict[str, Any]:
        """TODO: Return system statistics."""
        pass

# TODO: Implement this function
async def simulate_real_time_usage(rag_system: AsyncStreamingRAG):
    """Simulate real-time RAG system usage."""
    pass
```

### Tasks
1. Implement document stream ingestion with buffering
2. Create concurrent query processing with semaphore control
3. Add backpressure handling for high throughput
4. Implement streaming response generation
5. Add real-time statistics and monitoring
6. Create graceful degradation under load
7. Test with high-concurrency scenarios

### Hints
- Use asyncio.Semaphore for concurrency control
- Implement circular buffers for stream processing
- Add connection pooling for external APIs
- Consider using asyncio.gather for concurrent operations

### Expected Output
```
Streaming RAG System Stats:
========================
Active streams: 3
Queries processed: 1,247
Average query time: 145ms
Concurrency utilization: 75%
Buffer utilization: 60%
Error rate: 0.1%
Throughput: 43 queries/second
```

---

## Exercise 4: Build a Comprehensive Debugging Toolkit

### Objective
Create a debugging toolkit that can trace DSPy execution, profile performance, identify bottlenecks, and provide insights for optimization.

### Problem
You need to build a comprehensive debugging system that can trace DSPy module execution, profile performance metrics, visualize execution graphs, and provide actionable optimization recommendations.

### Starter Code
```python
import dspy
import time
import cProfile
import pstats
import io
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Callable
from functools import wraps
import inspect

class DSPyDebugToolkit:
    """TODO: Implement comprehensive debugging toolkit."""

    def __init__(self):
        super().__init__()
        # TODO: Initialize debugging components
        self.trace_history = []
        self.profiler = None
        self.execution_graph = nx.DiGraph()
        self.performance_metrics = {}

    def trace_function(self, level: str = "info"):
        """TODO: Return decorator for function tracing."""
        pass

    def start_profiling(self, name: str):
        """TODO: Start performance profiling."""
        pass

    def stop_profiling(self) -> Dict[str, Any]:
        """TODO: Stop profiling and return results."""
        pass

    def add_execution_node(self, module_name: str, operation: str, data: Dict[str, Any]):
        """TODO: Add node to execution graph."""
        pass

    def add_execution_edge(self, source: str, target: str, relationship: str):
        """TODO: Add edge to execution graph."""
        pass

    def visualize_execution(self, save_path: Optional[str] = None):
        """TODO: Visualize execution graph."""
        pass

    def analyze_performance(self) -> Dict[str, Any]:
        """TODO: Analyze performance and provide insights."""
        pass

    def generate_report(self) -> str:
        """TODO: Generate comprehensive debugging report."""
        pass

# TODO: Implement this function
def debug_dspy_module(module: dspy.Module, test_inputs: List[Any]) -> Dict[str, Any]:
    """Debug a DSPy module comprehensively."""
    pass
```

### Tasks
1. Implement function tracing decorator with different levels
2. Add performance profiling with cProfile
3. Create execution graph visualization with NetworkX
4. Implement automatic bottleneck detection
5. Add memory usage tracking
6. Create optimization recommendations engine
7. Generate comprehensive debugging reports

### Hints
- Use functools.wraps for decorator implementation
- Implement hierarchical trace levels
- Use NetworkX for graph visualization
- Consider using memory_profiler for memory tracking

### Expected Output
```
DSPy Debugging Report
====================
Execution traced: 1,234 function calls
Total execution time: 2.45s
Peak memory usage: 512MB

Top 5 slowest functions:
1. retrieve_documents: 1.2s (49% of total)
2. generate_answer: 0.8s (33% of total)
3. process_context: 0.3s (12% of total)

Bottlenecks identified:
- Retrieval system needs caching
- Generate function can be optimized
- Consider async processing for I/O

Recommendations:
1. Implement LRU cache for document retrieval
2. Use batch processing for multiple queries
3. Add connection pooling for API calls
```

---

## Exercise 5: Deploy DSPy Application to Kubernetes

### Objective
Create a complete Kubernetes deployment configuration for a DSPy application with proper resource management, autoscaling, and monitoring.

### Problem
You need to deploy a DSPy application to Kubernetes with production-grade configurations including horizontal pod autoscaling, resource limits, health checks, and monitoring setup.

### Starter Code
```yaml
# TODO: Complete this Kubernetes manifest
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dspy-app
  namespace: dspy-prod
spec:
  replicas: 3  # TODO: Configure autoscaling
  selector:
    matchLabels:
      app: dspy-app
  template:
    metadata:
      labels:
        app: dspy-app
    spec:
      containers:
      - name: dspy-app
        image: dspy-app:latest  # TODO: Configure image
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: dspy-secrets
              key: openai-api-key
        # TODO: Add resource limits and requests
        # TODO: Add health checks

# TODO: Add Service, HPA, ConfigMap, Secret manifests
```

### Tasks
1. Create complete deployment manifest with resource limits
2. Configure Horizontal Pod Autoscaler (HPA) with appropriate metrics
3. Add liveness and readiness probes
4. Create service with load balancer configuration
5. Set up ConfigMap for configuration management
6. Create Secret for sensitive data
7. Add monitoring with ServiceMonitor
8. Create network policies for security
9. Configure persistent volumes if needed

### Hints
- Use appropriate resource requests and limits
- Set meaningful health check endpoints
- Configure HPA with custom metrics if needed
- Use namespace isolation for different environments
- Implement pod disruption budgets

### Expected Output
```
Kubernetes Deployment Status:
=========================
Namespace: dspy-prod
Deployment: dspy-app
Status: Running
Replicas: 3 (desired: 3, ready: 3, unavailable: 0)

Autoscaling:
- Min replicas: 3
- Max replicas: 20
- Current CPU: 45% (target: 70%)
- Current Memory: 60% (target: 80%)

Resources per Pod:
- CPU: 100m / 500m (20% utilization)
- Memory: 512Mi / 2Gi (25% utilization)

Health Status:
- Liveness probe: Passing
- Readiness probe: Passing
- Startup probe: Passing

Services:
- dspy-service: LoadBalancer (External IP: 203.0.113.42)
- Health endpoint: /health
```

---

## Exercise 6: Implement Production Monitoring and Alerting

### Objective
Create a comprehensive monitoring and alerting system for a DSPy application using Prometheus, Grafana, and AlertManager.

### Problem
You need to set up monitoring that tracks key DSPy application metrics, creates meaningful dashboards, and provides proactive alerting for issues.

### Starter Code
```python
# monitoring.py
import time
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import logging
from typing import Dict, Any
import asyncio

class DSPyMonitoring:
    """TODO: Implement comprehensive monitoring."""

    def __init__(self):
        super().__init__()
        # TODO: Define Prometheus metrics
        self.metrics = {}

    def setup_metrics(self):
        """TODO: Setup Prometheus metrics."""
        pass

    def record_request(self, endpoint: str, duration: float, status: str):
        """TODO: Record request metrics."""
        pass

    def record_cache_hit(self, backend: str):
        """TODO: Record cache hit."""
        pass

    def record_token_usage(self, prompt_tokens: int, completion_tokens: int):
        """TODO: Record token usage."""
        pass

    def record_error(self, error_type: str, component: str):
        """TODO: Record error occurrence."""
        pass

# TODO: Implement this function
def setup_grafana_dashboards():
    """Setup Grafana dashboards for DSPy monitoring."""
    pass

# TODO: Implement this function
def setup_alertmanager_rules():
    """Setup AlertManager rules for proactive alerting."""
    pass
```

### Tasks
1. Define comprehensive Prometheus metrics for DSPy operations
2. Create Grafana dashboard JSON configurations
3. Set up AlertManager with meaningful alert rules
4. Implement distributed tracing with Jaeger
5. Create structured logging with ELK stack
6. Add custom health checks and metrics
7. Set up log aggregation and analysis

### Hints
- Use appropriate metric types (Counter, Histogram, Gauge)
- Create dashboard panels for different aspects (performance, errors, usage)
- Set up multi-level alerting (warning, critical)
- Consider using OpenTelemetry for distributed tracing
- Implement structured logging with correlation IDs

### Expected Output
```
Monitoring Setup Complete:
======================
Prometheus server: http://prometheus:9090
Grafana dashboards:
  - DSPy Overview: http://grafana:3000/d/dspy-overview
  - Performance Metrics: http://grafana:3000/d/performance
  - Error Analysis: http://grafana:3000/d/errors

Metrics Exported:
- dspy_requests_total
- dspy_request_duration_seconds
- dspy_cache_hits_total
- dspy_token_usage
- dspy_errors_total
- dspy_active_connections

AlertManager Rules:
- High error rate (>5% in 5 minutes)
- High latency (P95 > 2s)
- High memory usage (>80%)
- API rate limiting
```

---

## Exercise 7: Complete Production-Grade DSPy System

### Objective
Integrate all advanced concepts into a complete production-grade DSPy system with adapters, caching, async processing, debugging, and deployment.

### Problem
You need to build a complete DSPy application that demonstrates mastery of all advanced topics covered in this chapter, suitable for production deployment.

### Starter Code
```python
import dspy
import asyncio
import logging
from typing import AsyncGenerator, Dict, Any, List, Optional

class ProductionDSPySystem:
    """TODO: Implement complete production DSPy system."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        # TODO: Initialize all components
        # - Database adapter
        # - Cache system
        # - Monitoring
        # - Debugging toolkit
        # - Performance optimization

    async def initialize(self):
        """TODO: Initialize all system components."""
        pass

    async def process_streaming_query(self, query_stream: AsyncGenerator[str, None]) -> AsyncGenerator[Dict[str, Any], None]:
        """TODO: Process streaming queries with all optimizations."""
        pass

    async def batch_process(self, queries: List[str], batch_size: int = 10) -> List[Dict[str, Any]]:
        """TODO: Process batch of queries efficiently."""
        pass

    async def get_system_status(self) -> Dict[str, Any]:
        """TODO: Get comprehensive system status."""
        pass

    async def shutdown(self):
        """TODO: Graceful shutdown of all components."""
        pass

# TODO: Implement this function
async def test_production_system():
    """Test production system comprehensively."""
    pass
```

### Tasks
1. Integrate all previously built components
2. Implement proper error handling and recovery
3. Add comprehensive logging and monitoring
4. Create performance benchmarks
5. Implement health checks and self-healing
6. Add configuration management
7. Create deployment scripts and documentation
8. Test scalability and reliability

### Hints
- Use dependency injection for component management
- Implement proper async context managers
- Add comprehensive error handling with retry logic
- Use structured logging with correlation IDs
- Implement circuit breakers for external services
- Create comprehensive health checks

### Expected Output
```
Production DSPy System Status:
=============================
System Health: HEALTHY
Uptime: 99.99%
Last restart: 7 days ago

Component Status:
- Database Adapter: Connected (Pool: 8/10)
- Cache System: Active (L1: 89%, L2: 76%, L3: 92% hit rates)
- Monitoring: Online
- Debugging: Enabled (Verbose level)
- Performance: Optimized

Recent Metrics:
- Queries processed: 1,234,567
- Average response time: 145ms
- Error rate: 0.02%
- Throughput: 8,507 queries/hour
- Cache efficiency: 87%

Resource Usage:
- CPU: 35%
- Memory: 4.2GB
- Network: 125 Mbps
- Storage: 23.5GB

Alerts:
- None active
- Last alert: 3 days ago (High memory usage - resolved)
```

---

## Exercise Solutions Approach

After completing these exercises, you'll have:

1. **Custom Adapters**: Database integration with caching
2. **High-Performance Caching**: Multi-level caching systems
3. **Async Streaming**: Real-time data processing
4. **Debugging Tools**: Comprehensive debugging capabilities
5. **Kubernetes Deployment**: Production deployment configuration
6. **Monitoring Systems**: Complete observability stack
7. **Production System**: Fully integrated, production-ready application

### Key Learning Achievements

- **System Integration**: Combining multiple advanced techniques
- **Performance Engineering**: Optimizing for speed and efficiency
- **Production Readiness**: Understanding deployment requirements
- **Observability**: Comprehensive monitoring and debugging
- **Scalability**: Building systems that handle growth
- **Reliability**: Implementing robust error handling

### Production Readiness Checklist

- [ ] Application handles high concurrent load
- [ ] Comprehensive error handling and recovery
- [ ] Monitoring and alerting configured
- [ ] Security best practices implemented
- [ ] Documentation complete
- [ ] Backup and disaster recovery planned
- [ ] Load testing completed
- [ ] Performance benchmarks established

Good luck mastering advanced DSPy concepts! These exercises will prepare you for building enterprise-grade applications that can handle real-world challenges.
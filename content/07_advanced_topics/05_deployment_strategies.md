# Deployment Strategies: Taking DSPy Applications to Production

## Introduction

Deploying DSPy applications to production requires careful consideration of scalability, reliability, security, and maintainability. This section explores comprehensive deployment strategies that ensure your DSPy applications can handle real-world workloads efficiently and effectively.

## Deployment Architecture Overview

### Production Deployment Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer                             │
├─────────────────────────────────────────────────────────────┤
│                API Gateway / Ingress                          │
│  - Rate Limiting    - Authentication    - SSL Termination     │
├─────────────────────────────────────────────────────────────┤
│                Application Layer                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Web App   │  │   API App   │  │   Background Worker  │  │
│  │ (FastAPI)   │  │ (Flask)     │  │    (Celery)         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    Service Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   DSPy      │  │   Cache     │  │     Message Queue   │  │
│  │  Modules    │  │  (Redis)    │  │     (RabbitMQ)      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    Data Layer                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ PostgreSQL  │  │   Vector DB │  │    Object Storage   │  │
│  │ (Metadata)  │  │ (Pinecone)  │  │       (S3)          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                  Monitoring & Observability                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Metrics    │  │   Logging   │  │     Tracing        │  │
│  │(Prometheus) │  │ (ELK Stack) │  │    (Jaeger)        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Containerization with Docker

### 1. Dockerfile for DSPy Application

```dockerfile
# Multi-stage build for production DSPy application
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Create non-root user
RUN useradd --create-home --shell /bin/bash dspy
USER dspy
WORKDIR /home/dspy/app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/dspy/.local
ENV PATH=/home/dspy/.local/bin:$PATH

# Copy application code
COPY --chown=dspy:dspy . .

# Set environment variables
ENV PYTHONPATH=/home/dspy/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "main.py"]
```

### 2. Docker Compose for Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  dspy-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:5432/dspydb
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - redis
      - postgres
    volumes:
      - ./logs:/home/dspy/app/logs
      - ./data:/home/dspy/app/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=dspydb
      - POSTGRES_USER=dspy
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  monitoring:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  redis_data:
  postgres_data:
  grafana_data:
```

## FastAPI Web Service

### 1. Production-Ready FastAPI Application

```python
# main.py
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

import dspy
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global DSPy configuration
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting DSPy application...")

    # Initialize LM
    lm = dspy.LM(
        model="gpt-3.5-turbo",
        api_base=os.getenv("OPENAI_API_BASE"),
        api_key=os.getenv("OPENAI_API_KEY")
    )
    dspy.settings.configure(lm=lm)

    # Initialize cache
    from dspy.performance import CacheManager
    cache = CacheManager(
        backend=os.getenv("CACHE_BACKEND", "memory"),
        redis_url=os.getenv("REDIS_URL")
    )

    app.state.cache = cache
    app.state.lm = lm

    yield

    # Shutdown
    logger.info("Shutting down DSPy application...")
    if hasattr(app.state.cache, 'close'):
        await app.state.cache.close()

# Create FastAPI app
app = FastAPI(
    title="DSPy Production API",
    description="Production-ready DSPy application",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    context: str = None
    max_tokens: int = 1000
    temperature: float = 0.7

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    response_time: float
    tokens_used: int

# Initialize DSPy module
class ProductionRAG(dspy.Module):
    """Production RAG module with caching."""

    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=5)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question, context=None):
        # Use provided context or retrieve
        if context:
            final_context = context
        else:
            retrieved = self.retrieve(question=question)
            final_context = "\n".join(retrieved.passages)

        prediction = self.generate(context=final_context, question=question)

        return dspy.Prediction(
            answer=prediction.answer,
            context=final_context,
            reasoning=prediction.rationale
        )

# Initialize module
rag_module = ProductionRAG()

# API endpoints
@app.get("/")
async def root():
    return {"message": "DSPy Production API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }

@app.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
async def query_endpoint(
    request: QueryRequest,
    background_tasks: BackgroundTasks
):
    """Main query endpoint with caching and rate limiting."""
    start_time = time.time()

    try:
        # Check cache first
        cache_key = f"query:{hash(request.query)}"
        if app.state.cache:
            cached_result = await app.state.cache.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for query: {request.query[:50]}...")
                return QueryResponse(
                    answer=cached_result["answer"],
                    confidence=cached_result["confidence"],
                    response_time=time.time() - start_time,
                    tokens_used=0
                )

        # Process query
        logger.info(f"Processing query: {request.query[:50]}...")
        result = rag_module.forward(
            question=request.query,
            context=request.context
        )

        # Calculate confidence (simplified)
        confidence = 0.8  # Could be calculated based on various factors

        response = QueryResponse(
            answer=result.answer,
            confidence=confidence,
            response_time=time.time() - start_time,
            tokens_used=len(result.answer.split()) + len(request.query.split())
        )

        # Cache result asynchronously
        if app.state.cache:
            background_tasks.add_task(
                app.state.cache.set,
                cache_key,
                {
                    "answer": result.answer,
                    "confidence": confidence
                }
            )

        return response

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-query")
@limiter.limit("5/minute")
async def batch_query_endpoint(requests: list[QueryRequest]):
    """Batch query endpoint for higher throughput."""
    start_time = time.time()

    try:
        # Process in parallel
        tasks = [
            process_single_query(req) for req in requests
        ]
        results = await asyncio.gather(*tasks)

        return {
            "results": results,
            "total_time": time.time() - start_time,
            "processed": len(results)
        }

    except Exception as e:
        logger.error(f"Error in batch query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_single_query(request: QueryRequest) -> Dict[str, Any]:
    """Process single query (helper for batch)."""
    result = rag_module.forward(
        question=request.query,
        context=request.context
    )

    return {
        "query": request.query,
        "answer": result.answer,
        "confidence": 0.8
    }

# Configuration endpoints
@app.get("/config")
async def get_config():
    """Get current configuration."""
    return {
        "model": "gpt-3.5-turbo",
        "cache_enabled": app.state.cache is not None,
        "rate_limits": {
            "query": "10/minute",
            "batch": "5/minute"
        }
    }

@app.post("/config/reload")
async def reload_config():
    """Reload configuration."""
    # Implement config reloading logic
    return {"message": "Configuration reloaded"}

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get application metrics."""
    return {
        "queries_processed": get_query_count(),
        "cache_hits": get_cache_hits(),
        "average_response_time": get_avg_response_time(),
        "error_rate": get_error_rate()
    }

# Helper functions for metrics
def get_query_count() -> int:
    """Get total query count."""
    # Implement metric collection
    return 0

def get_cache_hits() -> int:
    """Get cache hit count."""
    # Implement metric collection
    return 0

def get_avg_response_time() -> float:
    """Get average response time."""
    # Implement metric collection
    return 0.0

def get_error_rate() -> float:
    """Get error rate."""
    # Implement metric collection
    return 0.0

# Run server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info",
        access_log=True
    )
```

### 2. Configuration Management

```python
# config.py
import os
from typing import Optional
from pydantic import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings with validation."""

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    workers: int = 4
    reload: bool = False

    # DSPy Configuration
    model_name: str = "gpt-3.5-turbo"
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 30

    # Cache Configuration
    cache_backend: str = "memory"  # memory, redis, file
    redis_url: Optional[str] = None
    cache_ttl: int = 3600

    # Database Configuration
    database_url: Optional[str] = None
    database_pool_size: int = 10

    # Rate Limiting
    rate_limit_query: str = "10/minute"
    rate_limit_batch: str = "5/minute"

    # Security
    secret_key: str = "your-secret-key-here"
    allowed_origins: str = "*"

    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"

    # External Services
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_api_base: Optional[str] = os.getenv("OPENAI_API_BASE")

    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

# Configuration validation
def validate_settings(settings: Settings) -> bool:
    """Validate application settings."""

    # Validate required fields
    if settings.cache_backend == "redis" and not settings.redis_url:
        raise ValueError("Redis URL required when using Redis backend")

    # Validate rate limits
    if not settings.rate_limit_query or "/" not in settings.rate_limit_query:
        raise ValueError("Invalid rate limit format")

    # Validate model configuration
    if not settings.model_name:
        raise ValueError("Model name is required")

    return True
```

## Kubernetes Deployment

### 1. Kubernetes Manifests

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: dspy-prod
  labels:
    name: dspy-prod
---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dspy-app
  namespace: dspy-prod
spec:
  replicas: 3
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
        image: dspy-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: dspy-secrets
              key: openai-api-key
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: dspy-secrets
              key: database-url
        resources:
          requests:
            cpu: 100m
            memory: 512Mi
          limits:
            cpu: 500m
            memory: 2Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: dspy-service
  namespace: dspy-prod
spec:
  selector:
    app: dspy-app
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
---
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: dspy-hpa
  namespace: dspy-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: dspy-app
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 2. Helm Chart

```yaml
# Chart.yaml
apiVersion: v2
name: dspy-app
description: A Helm chart for DSPy production application
type: application
version: 1.0.0
appVersion: "1.0.0"

# values.yaml
replicaCount: 3

image:
  repository: dspy-app
  pullPolicy: IfNotPresent
  tag: "latest"

service:
  type: LoadBalancer
  port: 80

ingress:
  enabled: true
  className: "nginx"
  annotations: {}
  hosts:
    - host: dspy.example.com
      paths:
        - path: /
          pathType: Prefix

resources:
  limits:
    cpu: 500m
    memory: 2Gi
  requests:
    cpu: 100m
    memory: 512Mi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilization: 70

config:
  model: "gpt-3.5-turbo"
  maxTokens: 4096
  temperature: 0.7
  cacheBackend: "redis"

redis:
  enabled: true
  auth:
    enabled: false
  master:
    persistence:
      enabled: false

monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
```

## Monitoring and Observability

### 1. Prometheus Metrics

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import functools

# Define metrics
REQUEST_COUNT = Counter('dspy_requests_total', 'Total DSPy requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('dspy_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('dspy_active_connections', 'Active connections')
CACHE_HITS = Counter('dspy_cache_hits_total', 'Cache hits', ['backend'])
API_ERRORS = Counter('dspy_api_errors_total', 'API errors', ['error_type'])
TOKEN_USAGE = Histogram('dspy_token_usage', 'Token usage per request', ['type'])

class DSPyMetrics:
    """Metrics collection for DSPy applications."""

    def __init__(self):
        # Start metrics server
        start_http_server(8001)

    def time_request(self):
        """Decorator to time requests."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    REQUEST_COUNT.labels(method=func.__name__, endpoint="/").inc()
                    result = func(*args, **kwargs)
                    REQUEST_DURATION.observe(time.time() - start_time)
                    return result
                except Exception as e:
                    API_ERRORS.labels(error_type=type(e).__name__).inc()
                    raise
            return wrapper
        return decorator

    def record_cache_hit(self, backend):
        """Record cache hit."""
        CACHE_HITS.labels(backend=backend).inc()

    def record_token_usage(self, prompt_tokens, completion_tokens):
        """Record token usage."""
        TOKEN_USAGE.labels(type="prompt").observe(prompt_tokens)
        TOKEN_USAGE.labels(type="completion").observe(completion_tokens)

# Global metrics instance
metrics = DSPyMetrics()
```

### 2. Logging Configuration

```python
# logging_config.py
import logging
import sys
from pythonjsonlogger import jsonlogger

def setup_logging(log_level="INFO"):
    """Setup structured logging."""

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler("/var/log/dspy/app.log")

    # Create formatters
    formatter = jsonlogger.JsonFormatter(
        fmt='%(asctime)s %(name)s %(levelname)s %(message)s'
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Set log levels for external libraries
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
```

## CI/CD Pipeline

### 1. GitHub Actions Workflow

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-asyncio

    - name: Run tests
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        REDIS_URL: redis://localhost:6379
      run: |
        pytest --cov=./ --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  security:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Run Bandit Security Linter
      run: |
        pip install bandit[toml]
        bandit -r . -f json -o bandit-report.json

    - name: Upload security report
      uses: actions/upload-artifact@v3
      with:
        name: security-report
        path: bandit-report.json

  build-and-deploy:
    needs: [test, security]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Log in to Container Registry
      uses: docker/login@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}

    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}

    - name: Deploy to Kubernetes
      run: |
        echo "${{ secrets.KUBECONFIG }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
        kubectl set image deployment/dspy-app dspy-app=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
        kubectl rollout status deployment/dspy-app
```

### 2. Dockerfile Multi-stage Build

```dockerfile
# Dockerfile.production
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create app directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Production stage
FROM base as production

# Create non-root user
RUN useradd --create-home --shell /bin/bash dspy

# Copy application code
COPY --chown=dspy:dspy . /home/dspy/app/
WORKDIR /home/dspy/app

# Create necessary directories
RUN mkdir -p /home/dspy/app/logs /home/dspy/app/data
RUN chown -R dspy:dspy /home/dspy/app

# Switch to non-root user
USER dspy

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Deployment Best Practices

### 1. Environment Configuration

```python
# environment.py
import os
from enum import Enum

class Environment(Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

def get_environment():
    """Get current environment."""
    return Environment(os.getenv("ENVIRONMENT", "development").lower())

def is_production():
    """Check if running in production."""
    return get_environment() == Environment.PRODUCTION

def is_development():
    """Check if running in development."""
    return get_environment() == Environment.DEVELOPMENT
```

### 2. Error Handling and Graceful Shutdown

```python
# error_handling.py
import signal
import logging
import asyncio
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class GracefulShutdown:
    """Handle graceful shutdown of DSPy application."""

    def __init__(self):
        self.shutdown = False
        self.tasks = []

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown = True

    @asynccontextmanager
    async def lifespan_manager(self):
        """Manage application lifespan."""
        self.setup_signal_handlers()

        try:
            yield
        finally:
            await self.graceful_shutdown()

    async def graceful_shutdown(self):
        """Perform graceful shutdown."""
        if self.shutdown:
            logger.info("Graceful shutdown already in progress")
            return

        self.shutdown = True

        # Cancel all running tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(f"Task {task} cancelled")

        logger.info("Graceful shutdown completed")
```

### 3. Security Configuration

```python
# security.py
import secrets
from typing import List
import hashlib
import hmac

class SecurityManager:
    """Manage security configurations."""

    def __init__(self):
        self.secret_key = self._generate_secret_key()
        self.allowed_origins = self._get_allowed_origins()

    def _generate_secret_key(self) -> str:
        """Generate secure secret key."""
        return secrets.token_urlsafe(32)

    def _get_allowed_origins(self) -> List[str]:
        """Get allowed origins from environment."""
        origins = os.getenv("ALLOWED_ORIGINS", "*")
        return [origin.strip() for origin in origins.split(",")]

    def verify_api_key(self, api_key: str, provided_signature: str) -> bool:
        """Verify API key signature."""
        expected_signature = hmac.new(
            self.secret_key.encode(),
            api_key.encode(),
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(expected_signature, provided_signature)

    def hash_password(self, password: str) -> str:
        """Hash password securely."""
        import bcrypt
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        import bcrypt
        return bcrypt.checkpw(password.encode(), hashed.encode())
```

## Key Takeaways

1. **Containerization** ensures consistent deployment environments
2. **Orchestration** (Kubernetes) enables scalable deployments
3. **Monitoring** is essential for production reliability
4. **CI/CD pipelines** automate deployment and ensure quality
5. **Security** must be considered at every layer
6. **Graceful shutdown** prevents data corruption during updates

## Chapter Summary

This chapter has covered comprehensive deployment strategies for DSPy applications:

- **Containerization** with Docker and multi-stage builds
- **Orchestration** with Kubernetes and Helm charts
- **Web service** implementation with FastAPI
- **Monitoring** with Prometheus and structured logging
- **CI/CD** pipelines with automated testing and deployment
- **Security** configurations and best practices

These strategies ensure your DSPy applications are production-ready, scalable, and maintainable. Always remember that deployment is not just about running code—it's about running it reliably, securely, and efficiently.

## Next Steps

In the final section of this chapter, we'll provide **comprehensive exercises** that test all the advanced concepts covered, from adapters and tools through deployment strategies, helping you master production-ready DSPy application development.
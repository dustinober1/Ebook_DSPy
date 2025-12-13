"""
Chapter 7: Debugging and Deployment Examples

This example demonstrates debugging tools, logging, tracing,
and deployment strategies for DSPy applications.
"""

import os
import json
import time
import logging
import traceback
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

import dspy
from dspy import Module, Predict, Signature, InputField, OutputField

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DebugInfo:
    """Debug information for DSPy operations."""
    timestamp: str
    operation: str
    input_data: Dict
    output_data: Any
    duration: float
    error: Optional[str] = None
    stack_trace: Optional[str] = None
    metadata: Optional[Dict] = None


class DebugPredict:
    """Predict wrapper with comprehensive debugging capabilities."""

    def __init__(self, signature: Signature, debug_mode: bool = True):
        self.predict = Predict(signature)
        self.debug_mode = debug_mode
        self.debug_history = []

    def __call__(self, **kwargs):
        """Execute with full debugging."""
        if not self.debug_mode:
            return self.predict(**kwargs)

        start_time = time.time()
        debug_info = DebugInfo(
            timestamp=datetime.now().isoformat(),
            operation=str(self.predict.signature),
            input_data=kwargs.copy(),
            output_data=None,
            duration=0
        )

        try:
            # Execute prediction
            result = self.predict(**kwargs)
            debug_info.output_data = result
            debug_info.duration = time.time() - start_time

            # Log success
            logger.info(f"Prediction completed in {debug_info.duration:.3f}s")

            self.debug_history.append(debug_info)
            return result

        except Exception as e:
            debug_info.error = str(e)
            debug_info.stack_trace = traceback.format_exc()
            debug_info.duration = time.time() - start_time

            # Log error
            logger.error(f"Prediction failed: {debug_info.error}")

            self.debug_history.append(debug_info)
            raise

    def get_debug_history(self) -> List[DebugInfo]:
        """Get all debug information."""
        return self.debug_history.copy()

    def save_debug_log(self, filepath: str):
        """Save debug history to file."""
        log_data = [asdict(info) for info in self.debug_history]
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)

    def clear_history(self):
        """Clear debug history."""
        self.debug_history.clear()


class Tracer:
    """Trace execution flow through DSPy modules."""

    def __init__(self):
        self.traces = []
        self.active = False

    @contextmanager
    def trace(self, module_name: str, operation: str):
        """Context manager for tracing operations."""
        if not self.active:
            yield
            return

        trace_id = f"{module_name}_{operation}_{int(time.time() * 1000)}"
        start_time = time.time()

        trace_entry = {
            "trace_id": trace_id,
            "module": module_name,
            "operation": operation,
            "start_time": start_time,
            "children": []
        }

        self.traces.append(trace_entry)

        try:
            yield trace_id
        finally:
            trace_entry["end_time"] = time.time()
            trace_entry["duration"] = trace_entry["end_time"] - start_time

    def get_trace_tree(self) -> Dict:
        """Get execution trace as a tree."""
        return {"traces": self.traces}

    def save_trace(self, filepath: str):
        """Save trace to file."""
        with open(filepath, 'w') as f:
            json.dump(self.get_trace_tree(), f, indent=2)


class Profiler:
    """Profile DSPy operations for performance analysis."""

    def __init__(self):
        self.profiles = []
        self.current_profile = None

    @contextmanager
    def profile(self, name: str):
        """Profile a block of code."""
        profile_start = time.time()
        profile = {
            "name": name,
            "start_time": profile_start,
            "memory_start": self._get_memory_usage()
        }

        self.current_profile = profile

        try:
            yield
        finally:
            profile["end_time"] = time.time()
            profile["duration"] = profile["end_time"] - profile["start_time"]
            profile["memory_end"] = self._get_memory_usage()
            profile["memory_delta"] = (
                profile["memory_end"] - profile["memory_start"]
            )

            self.profiles.append(profile)
            self.current_profile = None

    def _get_memory_usage(self) -> int:
        """Get current memory usage in MB."""
        try:
            import psutil
            return psutil.Process().memory_info().rss // 1024 // 1024
        except ImportError:
            return 0

    def get_summary(self) -> Dict:
        """Get profiling summary."""
        if not self.profiles:
            return {}

        total_time = sum(p["duration"] for p in self.profiles)
        total_memory = sum(p.get("memory_delta", 0) for p in self.profiles)

        return {
            "total_profiles": len(self.profiles),
            "total_time": total_time,
            "avg_time": total_time / len(self.profiles),
            "max_time": max(p["duration"] for p in self.profiles),
            "total_memory_delta": total_memory,
            "profiles": self.profiles
        }


class DeploymentConfig:
    """Configuration for deploying DSPy applications."""

    def __init__(self,
                 model_name: str = "gpt-3.5-turbo",
                 max_tokens: int = 1000,
                 temperature: float = 0.7,
                 cache_enabled: bool = True,
                 debug_mode: bool = False):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache_enabled = cache_enabled
        self.debug_mode = debug_mode
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict):
        """Create configuration from dictionary."""
        return cls(**data)

    def save(self, filepath: str):
        """Save configuration to file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str):
        """Load configuration from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class HealthChecker:
    """Health checking for DSPy deployments."""

    def __init__(self, predict_func):
        self.predict_func = predict_func
        self.health_history = []

    async def check_health(self) -> Dict:
        """Perform health check."""
        start_time = time.time()

        try:
            # Test prediction
            test_result = self.predict_func(
                question="Health check test",
                context="Testing system health"
            )

            health_status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "response_time": time.time() - start_time,
                "test_passed": True,
                "details": {
                    "model_working": True,
                    "response_format_valid": bool(test_result)
                }
            }

        except Exception as e:
            health_status = {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "response_time": time.time() - start_time,
                "test_passed": False,
                "error": str(e),
                "details": {
                    "model_working": False
                }
            }

        self.health_history.append(health_status)
        return health_status

    def get_health_summary(self) -> Dict:
        """Get health check summary."""
        if not self.health_history:
            return {"status": "unknown", "message": "No health checks performed"}

        recent_checks = self.health_history[-10:]  # Last 10 checks
        healthy_count = sum(1 for c in recent_checks if c["status"] == "healthy")
        avg_response_time = sum(c.get("response_time", 0) for c in recent_checks) / len(recent_checks)

        return {
            "overall_status": "healthy" if healthy_count >= 7 else "degraded",
            "recent_health_ratio": healthy_count / len(recent_checks),
            "average_response_time": avg_response_time,
            "total_checks": len(self.health_history),
            "latest_check": self.health_history[-1]
        }


# Dockerfile template generator
class DockerGenerator:
    """Generate Docker configurations for DSPy applications."""

    @staticmethod
    def generate_dockerfile(app_name: str = "dspy_app") -> str:
        """Generate Dockerfile for DSPy application."""
        return f"""# DSPy Application Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 dspyuser && chown -R dspyuser:dspyuser /app
USER dspyuser

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

    @staticmethod
    def generate_docker_compose(app_name: str = "dspy_app") -> str:
        """Generate docker-compose.yml for DSPy application."""
        return f"""version: '3.8'

services:
  {app_name}:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DSPY_DEBUG_MODE=false
    volumes:
      - ./logs:/app/logs
      - ./cache:/app/.dspy_cache
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - {app_name}
    restart: unless-stopped

volumes:
  redis_data:
"""

    @staticmethod
    def generate_requirements() -> str:
        """Generate requirements.txt for DSPy application."""
        return """# DSPy core
dspy-ai>=2.0.0
openai>=1.0.0

# Web framework
fastapi>=0.100.0
uvicorn[standard]>=0.23.0

# Monitoring and logging
prometheus-client>=0.17.0
structlog>=23.0.0

# Caching
redis>=4.5.0
diskcache>=5.6.0

# Async support
aiofiles>=23.0.0
asyncio-mqtt>=0.13.0

# Deployment
gunicorn>=21.0.0
python-multipart>=0.0.6

# Development
pytest>=7.0.0
pytest-asyncio>=0.21.0
black>=23.0.0
mypy>=1.0.0
"""


# Example Usage
def main():
    print("DSPy Debugging and Deployment Examples")
    print("=" * 60)

    # Example 1: DebugPredict
    print("\n1. DebugPredict Demo")
    print("-" * 40)

    class QuestionAnswering(Signature):
        question = InputField(desc="The question to answer")
        context = InputField(desc="Context for the question")
        answer = OutputField(desc="The answer")

    debug_predict = DebugPredict(QuestionAnswering, debug_mode=True)

    # Test with valid input
    result1 = debug_predict(
        question="What is debugging?",
        context="Debugging is the process of finding and fixing errors."
    )
    print(f"Result: {result1}")

    # Test with potential error
    try:
        result2 = debug_predict(
            question="What is deployment?",
            context=""  # Empty context might cause issues
        )
    except Exception as e:
        print(f"Caught error: {e}")

    # View debug history
    history = debug_predict.get_debug_history()
    print(f"\nDebug history contains {len(history)} entries")
    for i, entry in enumerate(history):
        print(f"  Entry {i + 1}: {entry.operation} - {entry.duration:.3f}s")
        if entry.error:
            print(f"    Error: {entry.error}")

    # Example 2: Tracer
    print("\n2. Execution Tracing")
    print("-" * 40)

    tracer = Tracer()
    tracer.active = True

    def process_question(question: str):
        with tracer.trace("processor", "analyze_question"):
            # Simulate processing
            time.sleep(0.1)
            with tracer.trace("analyzer", "extract_entities"):
                time.sleep(0.05)
            return f"Analyzed: {question}"

    # Process questions
    questions = ["What is AI?", "How does it work?", "Why use DSPy?"]
    for q in questions:
        result = process_question(q)
        print(f"  {result}")

    # Show trace tree
    trace_tree = tracer.get_trace_tree()
    print(f"\nTotal traces: {len(trace_tree['traces'])}")

    # Example 3: Profiler
    print("\n3. Performance Profiling")
    print("-" * 40)

    profiler = Profiler()

    # Profile different operations
    with profiler.profile("text_generation"):
        time.sleep(0.2)

    with profiler.profile("caching"):
        time.sleep(0.05)

    with profiler.profile("validation"):
        time.sleep(0.1)

    # Get profiling summary
    summary = profiler.get_summary()
    print(f"Profiled {summary['total_profiles']} operations")
    print(f"Total time: {summary['total_time']:.3f}s")
    print(f"Average time: {summary['avg_time']:.3f}s")
    print(f"Max time: {summary['max_time']:.3f}s")

    # Example 4: Deployment Configuration
    print("\n4. Deployment Configuration")
    print("-" * 40)

    # Create and save configuration
    config = DeploymentConfig(
        model_name="gpt-4",
        max_tokens=2000,
        temperature=0.5,
        cache_enabled=True,
        debug_mode=False
    )

    print(f"Configuration: {config.to_dict()}")

    # Save to file
    config.save("dspy_config.json")
    print("Configuration saved to dspy_config.json")

    # Load from file
    loaded_config = DeploymentConfig.load("dspy_config.json")
    print(f"Loaded config model: {loaded_config.model_name}")

    # Example 5: Health Checking
    print("\n5. Health Checking")
    print("-" * 40)

    def mock_predict(**kwargs):
        time.sleep(0.1)  # Simulate processing
        return f"Response to {kwargs.get('question', '')}"

    health_checker = HealthChecker(mock_predict)

    import asyncio

    async def run_health_checks():
        # Run multiple health checks
        for i in range(5):
            health = await health_checker.check_health()
            print(f"  Check {i + 1}: {health['status']} - {health['response_time']:.3f}s")
            await asyncio.sleep(0.2)

        # Get health summary
        summary = health_checker.get_health_summary()
        print(f"\nHealth Summary:")
        print(f"  Overall: {summary['overall_status']}")
        print(f"  Recent ratio: {summary['recent_health_ratio']:.1%}")
        print(f"  Avg response: {summary['average_response_time']:.3f}s")

    asyncio.run(run_health_checks())

    # Example 6: Docker Generation
    print("\n6. Docker Configuration Generation")
    print("-" * 40)

    # Generate Docker files
    dockerfile = DockerGenerator.generate_dockerfile("my_dspy_app")
    docker_compose = DockerGenerator.generate_docker_compose("my_dspy_app")
    requirements = DockerGenerator.generate_requirements()

    print("Generated files:")
    print("  - Dockerfile")
    print("  - docker-compose.yml")
    print("  - requirements.txt")

    # Save generated files
    Path("deployment").mkdir(exist_ok=True)
    with open("deployment/Dockerfile", "w") as f:
        f.write(dockerfile)
    with open("deployment/docker-compose.yml", "w") as f:
        f.write(docker_compose)
    with open("deployment/requirements.txt", "w") as f:
        f.write(requirements)

    print("\nFiles saved to deployment/ directory")

    # Example 7: Debugging Best Practices
    print("\n7. Debugging Best Practices")
    print("-" * 40)

    print("""
    1. Always use DebugPredict in development:
       debug_predict = DebugPredict(YourSignature, debug_mode=True)

    2. Enable tracing for complex workflows:
       tracer = Tracer()
       tracer.active = True

    3. Profile performance bottlenecks:
       with profiler.profile("operation_name"):
           # Your code here

    4. Monitor health in production:
       health_checker = HealthChecker(predict_func)
       await health_checker.check_health()

    5. Log extensively:
       logger.info("Operation completed", extra={"duration": 0.5})

    6. Save debug logs for analysis:
       debug_predict.save_debug_log("debug.json")
    """)

    # Clean up
    import os
    for file in ["dspy_config.json"]:
        if os.path.exists(file):
            os.remove(file)
    if os.path.exists("deployment"):
        import shutil
        shutil.rmtree("deployment")

    print("\n✓ All debugging and deployment examples completed!")


if __name__ == "__main__":
    main()
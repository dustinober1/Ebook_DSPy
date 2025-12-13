"""
Chapter 7: Async and Streaming Examples

This example demonstrates asynchronous programming and streaming
capabilities in DSPy applications.
"""

import asyncio
import json
import time
from typing import AsyncGenerator, Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import dspy
from dspy import Module, Predict, Signature, InputField, OutputField
from dspy.teleprompter import BootstrapFewShot

# Configure LM
turbo = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=1000)
dspy.settings.configure(lm=turbo)


class StreamingPredict:
    """Predict wrapper that supports streaming responses."""

    def __init__(self, signature: Signature, chunk_size: int = 50):
        self.predict = Predict(signature)
        self.chunk_size = chunk_size

    async def stream(self, **kwargs) -> AsyncGenerator[str, None]:
        """Stream prediction response in chunks."""
        # Get full response
        response = self.predict(**kwargs)
        full_text = str(response)

        # Stream in chunks
        for i in range(0, len(full_text), self.chunk_size):
            chunk = full_text[i:i + self.chunk_size]
            yield chunk
            await asyncio.sleep(0.1)  # Simulate streaming delay

    def stream_sync(self, **kwargs) -> List[str]:
        """Synchronous version of streaming."""
        chunks = []
        response = self.predict(**kwargs)
        full_text = str(response)

        for i in range(0, len(full_text), self.chunk_size):
            chunk = full_text[i:i + self.chunk_size]
            chunks.append(chunk)

        return chunks


class AsyncPipeline(Module):
    """Asynchronous pipeline for processing multiple stages."""

    def __init__(self):
        super().__init__()
        self.stages = []

    def add_stage(self, module: Module, name: str = None):
        """Add a processing stage to the pipeline."""
        self.stages.append({
            "module": module,
            "name": name or f"stage_{len(self.stages)}"
        })

    async def process_async(self, input_data: Dict) -> Dict:
        """Process input through all stages asynchronously."""
        result = input_data

        for stage in self.stages:
            # Run stage in background
            result = await self._run_stage_async(stage["module"], result)
            result["stage"] = stage["name"]

        return result

    async def _run_stage_async(self, module: Module, input_data: Dict) -> Dict:
        """Run a single module asynchronously."""
        # Simulate async execution
        await asyncio.sleep(0.1)
        return module(**input_data)

    def process_batch_async(self, inputs: List[Dict]) -> List[Dict]:
        """Process multiple inputs in parallel."""
        async def process_all():
            tasks = [self.process_async(inp) for inp in inputs]
            return await asyncio.gather(*tasks)

        return asyncio.run(process_all())


@dataclass
class StreamingRequest:
    """Request configuration for streaming operations."""
    prompt: str
    max_tokens: int = 1000
    temperature: float = 0.7
    stream: bool = True
    on_chunk: Optional[Callable] = None


class StreamingChat:
    """Chat interface with streaming capabilities."""

    def __init__(self, history_size: int = 10):
        self.history = []
        self.history_size = history_size

    async def chat_stream(self, request: StreamingRequest) -> AsyncGenerator[str, None]:
        """Stream chat response."""
        # Add to history
        self.history.append({
            "role": "user",
            "content": request.prompt,
            "timestamp": datetime.now().isoformat()
        })

        # Generate response
        response_chunks = []
        async for chunk in self._generate_stream(request):
            response_chunks.append(chunk)
            yield chunk

            # Call callback if provided
            if request.on_chunk:
                request.on_chunk(chunk)

        # Add response to history
        self.history.append({
            "role": "assistant",
            "content": "".join(response_chunks),
            "timestamp": datetime.now().isoformat()
        })

        # Trim history if needed
        if len(self.history) > self.history_size * 2:  # User + assistant pairs
            self.history = self.history[-self.history_size * 2:]

    async def _generate_stream(self, request: StreamingRequest) -> AsyncGenerator[str, None]:
        """Generate streaming response."""
        # Simple simulation (in practice, use actual LM streaming)
        words = request.prompt.split()
        response = f"I understand you said: {' '.join(words[:10])}..."

        for i in range(0, len(response), request.max_tokens // 20):
            chunk = response[i:i + request.max_tokens // 20]
            yield chunk
            await asyncio.sleep(0.05)

    def get_history(self) -> List[Dict]:
        """Get conversation history."""
        return self.history.copy()


class ConcurrentProcessor:
    """Process multiple DSPy operations concurrently."""

    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)

    async def process_concurrent(self,
                               tasks: List[Dict],
                               predict_func: Callable) -> List[Any]:
        """Process multiple tasks concurrently with rate limiting."""
        async def process_with_limit(task):
            async with self.semaphore:
                return await self._process_task(task, predict_func)

        # Create all tasks
        coroutines = [process_with_limit(task) for task in tasks]

        # Execute all concurrently
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        return results

    async def _process_task(self, task: Dict, predict_func: Callable) -> Any:
        """Process a single task."""
        try:
            # Simulate processing time
            await asyncio.sleep(0.1)
            return predict_func(**task)
        except Exception as e:
            return {"error": str(e), "task": task}


class RealTimeProcessor:
    """Process streaming data in real-time."""

    def __init__(self, buffer_size: int = 100):
        self.buffer = []
        self.buffer_size = buffer_size
        self.processing = False
        self.callbacks = []

    def add_callback(self, callback: Callable):
        """Add callback for processed data."""
        self.callbacks.append(callback)

    async def add_data(self, data: Dict):
        """Add data to buffer for processing."""
        self.buffer.append({
            "data": data,
            "timestamp": datetime.now()
        })

        # Trigger processing if buffer is full
        if len(self.buffer) >= self.buffer_size and not self.processing:
            await self._process_buffer()

    async def _process_buffer(self):
        """Process all data in buffer."""
        if not self.buffer:
            return

        self.processing = True
        batch = self.buffer.copy()
        self.buffer.clear()

        try:
            # Process batch
            processed = await self._analyze_batch(batch)

            # Call callbacks
            for callback in self.callbacks:
                await callback(processed)

        finally:
            self.processing = False

    async def _analyze_batch(self, batch: List[Dict]) -> Dict:
        """Analyze a batch of data."""
        await asyncio.sleep(0.1)  # Simulate processing

        return {
            "count": len(batch),
            "time_range": {
                "start": batch[0]["timestamp"].isoformat(),
                "end": batch[-1]["timestamp"].isoformat()
            },
            "summary": f"Processed {len(batch)} items"
        }


class WebSocketHandler:
    """WebSocket handler for real-time DSPy interactions."""

    def __init__(self, predict_func: Callable):
        self.predict_func = predict_func
        self.connections = {}

    async def connect(self, websocket_id: str):
        """Handle new WebSocket connection."""
        self.connections[websocket_id] = {
            "connected": True,
            "messages": []
        }
        print(f"WebSocket {websocket_id} connected")

    async def disconnect(self, websocket_id: str):
        """Handle WebSocket disconnection."""
        if websocket_id in self.connections:
            self.connections[websocket_id]["connected"] = False
            print(f"WebSocket {websocket_id} disconnected")

    async def handle_message(self, websocket_id: str, message: Dict):
        """Handle incoming WebSocket message."""
        if websocket_id not in self.connections:
            await self.connect(websocket_id)

        # Process message
        start_time = time.time()
        result = self.predict_func(**message.get("data", {}))
        duration = time.time() - start_time

        response = {
            "id": message.get("id"),
            "result": result,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        }

        # Store in connection
        self.connections[websocket_id]["messages"].append(response)

        return response

    async def stream_response(self,
                            websocket_id: str,
                            message: Dict) -> AsyncGenerator[Dict, None]:
        """Stream response for WebSocket message."""
        if websocket_id not in self.connections:
            await self.connect(websocket_id)

        # Simulate streaming response
        for i in range(5):
            chunk = {
                "id": message.get("id"),
                "chunk": i,
                "data": f"Processing step {i + 1}",
                "timestamp": datetime.now().isoformat()
            }
            yield chunk
            await asyncio.sleep(0.2)

        # Final result
        final_result = self.predict_func(**message.get("data", {}))
        yield {
            "id": message.get("id"),
            "complete": True,
            "result": final_result,
            "timestamp": datetime.now().isoformat()
        }


# Example Usage
async def main():
    print("DSPy Async and Streaming Examples")
    print("=" * 60)

    # Example 1: Basic streaming
    print("\n1. Basic Streaming Demo")
    print("-" * 40)

    class SimpleQA(Signature):
        question = InputField(desc="The question to answer")
        answer = OutputField(desc="The answer")

    streaming_predict = StreamingPredict(SimpleQA)

    print("Streaming response:")
    async for chunk in streaming_predict.stream(
        question="What is asynchronous programming?"
    ):
        print(chunk, end="", flush=True)
    print("\n")

    # Example 2: Async pipeline
    print("\n2. Async Pipeline Demo")
    print("-" * 40)

    class TextProcessor(Module):
        def forward(self, text: str) -> str:
            return f"Processed: {text.lower()}"

    class SentimentAnalyzer(Module):
        def forward(self, text: str) -> str:
            return f"Sentiment: positive" if "good" in text else "Sentiment: neutral"

    pipeline = AsyncPipeline()
    pipeline.add_stage(TextProcessor(), "processor")
    pipeline.add_stage(SentimentAnalyzer(), "sentiment")

    # Process single input
    result = await pipeline.process_async({"text": "This is good text"})
    print(f"Pipeline result: {result}")

    # Process batch
    inputs = [
        {"text": "Good morning"},
        {"text": "Bad weather"},
        {"text": "Excellent work"}
    ]
    batch_results = pipeline.process_batch_async(inputs)
    print(f"Batch processed {len(batch_results)} items")

    # Example 3: Streaming chat
    print("\n3. Streaming Chat Demo")
    print("-" * 40)

    chat = StreamingChat()

    # Collect chunks
    chunks = []
    def on_chunk(chunk):
        chunks.append(chunk)

    request = StreamingRequest(
        prompt="Tell me about async programming",
        on_chunk=on_chunk
    )

    print("Streaming chat response:")
    async for chunk in await chat.chat_stream(request).__anext__():
        pass  # Already collected by callback

    print(f"Received {len(chunks)} chunks")
    print(f"Full response: {''.join(chunks[:5])}...")

    # Show history
    history = chat.get_history()
    print(f"Chat history: {len(history)} messages")

    # Example 4: Concurrent processing
    print("\n4. Concurrent Processing Demo")
    print("-" * 40)

    processor = ConcurrentProcessor(max_workers=3)

    tasks = [
        {"question": f"What is {i}?"}
        for i in range(5)
    ]

    results = await processor.process_concurrent(
        tasks,
        lambda **kwargs: f"Answer to {kwargs.get('question', '')}"
    )

    print(f"Processed {len(results)} tasks concurrently")
    for i, result in enumerate(results[:3]):
        print(f"  Task {i}: {result}")

    # Example 5: Real-time processing
    print("\n5. Real-time Processing Demo")
    print("-" * 40)

    async def on_data(processed):
        print(f"  Processed batch: {processed['summary']}")

    rt_processor = RealTimeProcessor(buffer_size=3)
    rt_processor.add_callback(on_data)

    # Add data gradually
    for i in range(7):
        await rt_processor.add_data({"id": i, "value": f"data_{i}"})
        await asyncio.sleep(0.1)

    # Give time for final processing
    await asyncio.sleep(0.5)

    # Example 6: WebSocket simulation
    print("\n6. WebSocket Handler Demo")
    print("-" * 40)

    def mock_predict(**kwargs):
        return f"Predicted: {kwargs.get('question', '')}"

    ws_handler = WebSocketHandler(mock_predict)
    ws_id = "client_001"

    # Connect
    await ws_handler.connect(ws_id)

    # Send message
    message = {
        "id": "msg_001",
        "data": {"question": "What is streaming?"}
    }

    response = await ws_handler.handle_message(ws_id, message)
    print(f"WebSocket response: {response['result']}")

    # Stream response
    print("Streaming WebSocket response:")
    async for chunk in ws_handler.stream_response(ws_id, message):
        print(f"  {chunk}")

    # Disconnect
    await ws_handler.disconnect(ws_id)

    # Example 7: Performance comparison
    print("\n7. Performance Comparison")
    print("-" * 40)

    # Synchronous processing
    start_time = time.time()
    sync_results = []
    for task in tasks:
        result = mock_predict(**task)
        sync_results.append(result)
    sync_duration = time.time() - start_time

    # Asynchronous processing
    start_time = time.time()
    async_results = await processor.process_concurrent(tasks, mock_predict)
    async_duration = time.time() - start_time

    print(f"Synchronous: {sync_duration:.3f}s for {len(sync_results)} tasks")
    print(f"Asynchronous: {async_duration:.3f}s for {len(async_results)} tasks")
    print(f"Speedup: {sync_duration/async_duration:.1f}x")

    print("\n✓ All async and streaming examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
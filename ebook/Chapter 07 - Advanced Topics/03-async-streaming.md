# Async and Streaming: Building Real-Time DSPy Applications

## Introduction

As applications scale and user expectations grow, the ability to handle real-time data streams and concurrent operations becomes essential. This section explores how to build asynchronous and streaming-capable DSPy applications that can process data as it arrives, handle multiple requests simultaneously, and provide responsive user experiences.

## Understanding Asynchronous DSPy

### Why Async Matters

1. **Non-blocking Operations**: Don't wait for slow API calls
2. **Concurrent Processing**: Handle multiple requests simultaneously
3. **Real-time Responses**: Process streaming data as it arrives
4. **Resource Efficiency**: Better utilization of system resources
5. **Improved Throughput**: Process more requests per second

### Async DSPy Patterns

```python
import asyncio
import aiohttp
from typing import AsyncGenerator, List, Dict, Any
import time
from concurrent.futures import ThreadPoolExecutor

class AsyncDSPyModule(dspy.Module):
    """Base class for asynchronous DSPy modules."""

    def __init__(self):
        super().__init__()
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def aforward(self, *args, **kwargs):
        """Async version of forward method."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.forward,
            *args,
            **kwargs
        )

    async def batch_aforward(self, batch_args, batch_kwargs=None):
        """Async batch processing."""
        if batch_kwargs is None:
            batch_kwargs = [{}] * len(batch_args)

        tasks = [
            self.aforward(*args, **kwargs)
            for args, kwargs in zip(batch_args, batch_kwargs)
        ]

        return await asyncio.gather(*tasks)
```

## Streaming Data Processing

### 1. Stream Processor

```python
from collections import deque
import queue

class StreamProcessor:
    """Process streaming data with DSPy modules."""

    def __init__(self, buffer_size=100, batch_size=5):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.is_running = False
        self.results_queue = queue.Queue()

    async def add_item(self, item):
        """Add item to stream buffer."""
        self.buffer.append(item)
        if len(self.buffer) >= self.batch_size:
            await self._process_batch()

    async def _process_batch(self):
        """Process current buffer as batch."""
        if not self.buffer:
            return

        batch = list(self.buffer)
        self.buffer.clear()

        # Process batch
        results = await self._process_batch_items(batch)

        # Put results in queue
        for result in results:
            self.results_queue.put(result)

    async def _process_batch_items(self, batch):
        """Override in subclasses for specific processing."""
        return batch  # Default: return items as-is

    def get_results(self):
        """Get processed results."""
        results = []
        while not self.results_queue.empty():
            results.append(self.results_queue.get())
        return results

    async def flush(self):
        """Process remaining items in buffer."""
        if self.buffer:
            await self._process_batch()

    def start(self):
        """Start stream processing."""
        self.is_running = True

    def stop(self):
        """Stop stream processing."""
        self.is_running = False
```

### 2. Real-time Text Analyzer

```python
class RealTimeTextAnalyzer(AsyncDSPyModule):
    """Analyze text streams in real-time."""

    def __init__(self):
        super().__init__()
        self.analyze = dspy.Predict("text -> sentiment, topics, entities")
        self.stream_processor = StreamProcessor()

    async def analyze_stream(self, text_stream: AsyncGenerator[str, None]):
        """Analyze text as it streams in."""
        self.stream_processor.start()

        try:
            async for text in text_stream:
                await self.stream_processor.add_item(text)

                # Get and process results
                results = self.stream_processor.get_results()
                for result in results:
                    analysis = await self._analyze_text(result)
                    yield analysis

        finally:
            await self.stream_processor.flush()
            self.stream_processor.stop()

    async def _analyze_text(self, text):
        """Analyze individual text item."""
        loop = asyncio.get_event_loop()
        analysis = await loop.run_in_executor(
            self.executor,
            self.analyze,
            text=text
        )
        return {
            "text": text,
            "sentiment": analysis.sentiment,
            "topics": analysis.topics,
            "entities": analysis.entities,
            "timestamp": time.time()
        }

    async def _process_batch_items(self, batch):
        """Process batch of texts."""
        loop = asyncio.get_event_loop()
        tasks = [self._analyze_text(text) for text in batch]
        return await asyncio.gather(*tasks)
```

## Concurrent Request Handling

### 1. Concurrent Request Manager

```python
import asyncio
from typing import Dict, Callable, Any
from dataclasses import dataclass

@dataclass
class Request:
    id: str
    data: Any
    callback: Callable = None
    timeout: float = 30.0

class ConcurrentRequestManager:
    """Manage concurrent DSPy requests efficiently."""

    def __init__(self, max_concurrent=10, request_timeout=30):
        self.max_concurrent = max_concurrent
        self.request_timeout = request_timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_requests: Dict[str, asyncio.Task] = {}
        self.request_queue = asyncio.Queue()

    async def submit_request(self, request: Request) -> Any:
        """Submit request for processing."""
        # Acquire semaphore to limit concurrency
        async with self.semaphore:
            task = asyncio.create_task(
                self._process_request(request)
            )
            self.active_requests[request.id] = task

            try:
                result = await asyncio.wait_for(
                    task,
                    timeout=request.timeout
                )
                return result
            except asyncio.TimeoutError:
                task.cancel()
                raise TimeoutError(f"Request {request.id} timed out")
            finally:
                self.active_requests.pop(request.id, None)

    async def _process_request(self, request: Request):
        """Process individual request."""
        # Override in subclasses or provide processor function
        if request.callback:
            return await request.callback(request.data)
        else:
            return request.data

    async def submit_batch(self, requests: List[Request]) -> List[Any]:
        """Submit multiple requests concurrently."""
        tasks = [
            self.submit_request(request)
            for request in requests
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def cancel_request(self, request_id: str):
        """Cancel a specific request."""
        if request_id in self.active_requests:
            self.active_requests[request_id].cancel()
            del self.active_requests[request_id]

    def get_active_requests(self) -> List[str]:
        """Get list of active request IDs."""
        return list(self.active_requests.keys())

    async def shutdown(self):
        """Shutdown request manager."""
        # Cancel all active requests
        for task in self.active_requests.values():
            task.cancel()

        # Wait for all tasks to complete
        if self.active_requests:
            await asyncio.gather(
                *self.active_requests.values(),
                return_exceptions=True
            )
```

### 2. Async RAG System

```python
class AsyncRAG(AsyncDSPyModule):
    """Asynchronous RAG system for real-time Q&A."""

    def __init__(self, concurrent_limit=5):
        super().__init__()
        self.concurrent_limit = concurrent_limit
        self.request_manager = ConcurrentRequestManager(
            max_concurrent=concurrent_limit
        )

        # Components
        self.retrieve = dspy.Retrieve(k=5)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    async def aquery(self, question: str) -> dspy.Prediction:
        """Asynchronous query processing."""
        request = Request(
            id=str(time.time()),
            data=question,
            callback=self._process_query,
            timeout=10.0
        )
        return await self.request_manager.submit_request(request)

    async def batch_aquery(self, questions: List[str]) -> List[dspy.Prediction]:
        """Process multiple queries concurrently."""
        requests = [
            Request(
                id=f"query_{i}",
                data=question,
                callback=self._process_query,
                timeout=10.0
            )
            for i, question in enumerate(questions)
        ]
        return await self.request_manager.submit_batch(requests)

    async def _process_query(self, question: str) -> dspy.Prediction:
        """Process individual query."""
        loop = asyncio.get_event_loop()

        # Retrieve documents asynchronously
        retrieved = await loop.run_in_executor(
            self.executor,
            self.retrieve,
            question=question
        )

        # Generate answer
        prediction = await loop.run_in_executor(
            self.executor,
            self.generate,
            context="\n".join(retrieved.passages),
            question=question
        )

        return dspy.Prediction(
            question=question,
            answer=prediction.answer,
            context=retrieved.passages
        )

    async def stream_query(self, questions_stream: AsyncGenerator[str, None]):
        """Process streaming queries."""
        async for question in questions_stream:
            try:
                result = await self.aquery(question)
                yield result
            except Exception as e:
                yield dspy.Prediction(
                    question=question,
                    error=str(e)
                )
```

## WebSocket Integration

### 1. WebSocket Handler

```python
import websockets
import json
from typing import Set

class DSPyWebSocketHandler:
    """WebSocket handler for real-time DSPy interactions."""

    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.dspy_module = None

    def set_module(self, module):
        """Set the DSPy module to use."""
        self.dspy_module = module

    async def register_client(self, websocket, path):
        """Register new WebSocket client."""
        self.clients.add(websocket)
        print(f"Client connected. Total: {len(self.clients)}")

        try:
            await self.handle_client(websocket)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            print(f"Client disconnected. Total: {len(self.clients)}")

    async def handle_client(self, websocket):
        """Handle client messages."""
        async for message in websocket:
            try:
                data = json.loads(message)
                response = await self.process_message(data)
                await websocket.send(json.dumps(response))
            except Exception as e:
                error_response = {
                    "type": "error",
                    "message": str(e)
                }
                await websocket.send(json.dumps(error_response))

    async def process_message(self, data):
        """Process incoming message."""
        message_type = data.get("type", "query")

        if message_type == "query":
            if self.dspy_module:
                if hasattr(self.dspy_module, 'aforward'):
                    result = await self.dspy_module.aforward(data.get("query"))
                else:
                    # Fallback to sync processing
                    result = self.dspy_module.forward(data.get("query"))

                return {
                    "type": "response",
                    "result": result.__dict__
                }
            else:
                return {"type": "error", "message": "No module configured"}

        elif message_type == "stream":
            if hasattr(self.dspy_module, 'stream'):
                # Handle streaming response
                responses = []
                async for response in self.dspy_module.stream(data.get("input")):
                    responses.append(response)
                    await self.broadcast({
                        "type": "stream_update",
                        "partial": response.__dict__
                    })

                return {
                    "type": "stream_complete",
                    "responses": [r.__dict__ for r in responses]
                }

        else:
            return {"type": "error", "message": "Unknown message type"}

    async def broadcast(self, message):
        """Broadcast message to all clients."""
        if self.clients:
            await asyncio.gather(
                *[client.send(json.dumps(message)) for client in self.clients],
                return_exceptions=True
            )

    async def start_server(self):
        """Start WebSocket server."""
        print(f"Starting WebSocket server on {self.host}:{self.port}")
        async with websockets.serve(self.register_client, self.host, self.port):
            await asyncio.Future()  # Run forever

    def run(self):
        """Run the WebSocket server."""
        asyncio.run(self.start_server())
```

### 2. Real-time Chat Bot

```python
class RealTimeChatBot(AsyncDSPyModule):
    """Real-time chat bot with WebSocket integration."""

    def __init__(self):
        super().__init__()
        self.conversation_history = {}
        self.respond = dspy.ChainOfThought("history, message -> response")
        self.websocket_handler = DSPyWebSocketHandler()

    async def start_chat_server(self):
        """Start chat server."""
        self.websocket_handler.set_module(self)
        await self.websocket_handler.start_server()

    async def process_message(self, session_id: str, message: str):
        """Process chat message."""
        # Get conversation history
        history = self.conversation_history.get(session_id, [])

        # Generate response
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self.executor,
            self.respond,
            history=str(history[-5:]),  # Last 5 messages
            message=message
        )

        # Update history
        history.append({"user": message, "bot": response.response})
        self.conversation_history[session_id] = history

        return {
            "session_id": session_id,
            "response": response.response,
            "history_length": len(history)
        }

    async def _process_query(self, message):
        """Process message for WebSocket handler."""
        # Extract session_id from message if available
        session_id = message.get("session_id", "default")
        user_message = message.get("message", "")

        return await self.process_message(session_id, user_message)

    async def stream_response(self, session_id: str, message: str):
        """Stream response generation."""
        words = message.split()
        partial_response = ""

        for word in words:
            partial_response += word + " "
            yield {
                "session_id": session_id,
                "partial": partial_response,
                "complete": False
            }
            await asyncio.sleep(0.1)  # Simulate typing delay

        yield {
            "session_id": session_id,
            "final": partial_response.strip(),
            "complete": True
        }
```

## Event-Driven Architecture

### 1. Event System

```python
from collections import defaultdict
from typing import Dict, List, Callable
import asyncio

class EventSystem:
    """Event system for decoupled DSPy components."""

    def __init__(self):
        self.listeners: Dict[str, List[Callable]] = defaultdict(list)
        self.event_queue = asyncio.Queue()
        self.running = False

    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to event type."""
        self.listeners[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from event type."""
        if callback in self.listeners[event_type]:
            self.listeners[event_type].remove(callback)

    async def publish(self, event_type: str, data: Any):
        """Publish event to all listeners."""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": time.time()
        }
        await self.event_queue.put(event)

    async def start_processing(self):
        """Start event processing loop."""
        self.running = True
        while self.running:
            try:
                event = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=1.0
                )
                await self._handle_event(event)
            except asyncio.TimeoutError:
                continue

    async def _handle_event(self, event):
        """Handle single event."""
        event_type = event["type"]
        if event_type in self.listeners:
            tasks = [
                listener(event) if asyncio.iscoroutinefunction(listener)
                else asyncio.create_task(self._run_sync(listener, event))
                for listener in self.listeners[event_type]
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_sync(self, func, event):
        """Run synchronous listener in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, event)

    def stop(self):
        """Stop event processing."""
        self.running = False
```

### 2. Event-Driven RAG System

```python
class EventDrivenRAG(AsyncDSPyModule):
    """Event-driven RAG system."""

    def __init__(self):
        super().__init__()
        self.event_system = EventSystem()
        self.setup_event_handlers()

        # Components
        self.retrieve = dspy.Retrieve(k=5)
        self.generate = dspy.Predict("context, question -> answer")
        self.cache = {}

        # Start event processing
        asyncio.create_task(self.event_system.start_processing())

    def setup_event_handlers(self):
        """Setup event handlers."""
        self.event_system.subscribe("query_received", self._handle_query)
        self.event_system.subscribe("documents_retrieved", self._handle_documents)
        self.event_system.subscribe("answer_generated", self._handle_answer)

    async def query(self, question: str):
        """Submit query for processing."""
        await self.event_system.publish("query_received", {
            "question": question,
            "timestamp": time.time()
        })

    async def _handle_query(self, event):
        """Handle query received event."""
        data = event["data"]
        question = data["question"]

        # Check cache
        if question in self.cache:
            await self.event_system.publish("answer_generated", {
                "question": question,
                "answer": self.cache[question],
                "from_cache": True
            })
            return

        # Retrieve documents
        loop = asyncio.get_event_loop()
        retrieved = await loop.run_in_executor(
            self.executor,
            self.retrieve,
            question=question
        )

        await self.event_system.publish("documents_retrieved", {
            "question": question,
            "documents": retrieved.passages
        })

    async def _handle_documents(self, event):
        """Handle documents retrieved event."""
        data = event["data"]
        question = data["question"]
        documents = data["documents"]

        # Generate answer
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(
            self.executor,
            self.generate,
            context="\n".join(documents),
            question=question
        )

        # Cache result
        self.cache[question] = answer.answer

        await self.event_system.publish("answer_generated", {
            "question": question,
            "answer": answer.answer,
            "from_cache": False
        })

    async def _handle_answer(self, event):
        """Handle answer generated event."""
        data = event["data"]
        print(f"Answer for '{data['question']}': {data['answer']}")
        if data.get("from_cache"):
            print("(from cache)")

    def stop(self):
        """Stop the event-driven system."""
        self.event_system.stop()
```

## Performance Optimization

### 1. Connection Pooling

```python
import aiohttp
from aiohttp import ClientSession, TCPConnector

class ConnectionPoolManager:
    """Manage HTTP connection pools for API calls."""

    def __init__(self, pool_size=100, pool_timeout=30):
        self.connector = TCPConnector(
            limit=pool_size,
            limit_per_host=pool_size,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        self.session = None

    async def get_session(self):
        """Get or create HTTP session."""
        if self.session is None or self.session.closed:
            self.session = ClientSession(
                connector=self.connector,
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self.session

    async def close(self):
        """Close connection pool."""
        if self.session:
            await self.session.close()

# Global connection pool
connection_pool = ConnectionPoolManager()
```

### 2. Request Coalescing

```python
class RequestCoalescer:
    """Coalesce similar requests to reduce API calls."""

    def __init__(self, window_ms=100):
        self.window_ms = window_ms
        self.pending_requests = {}
        self.lock = asyncio.Lock()

    async def submit(self, key, request_func, *args, **kwargs):
        """Submit request with coalescing."""
        async with self.lock:
            if key in self.pending_requests:
                # Add to existing request
                future = self.pending_requests[key]
                if not hasattr(future, 'args_list'):
                    future.args_list = []
                    future.kwargs_list = []
                future.args_list.append(args)
                future.kwargs_list.append(kwargs)
                return await future
            else:
                # Create new request
                future = asyncio.Future()
                future.args_list = [args]
                future.kwargs_list = [kwargs]
                self.pending_requests[key] = future

                # Schedule execution after window
                asyncio.create_task(
                    self._execute_after_window(key, request_func, future)
                )

                return await future

    async def _execute_after_window(self, key, request_func, future):
        """Execute request after coalescing window."""
        await asyncio.sleep(self.window_ms / 1000)

        async with self.lock:
            # Remove from pending
            self.pending_requests.pop(key, None)

        # Execute with coalesced arguments
        try:
            if len(future.args_list) == 1:
                result = await request_func(
                    *future.args_list[0],
                    **future.kwargs_list[0]
                )
            else:
                # Handle multiple similar requests
                result = await self._handle_multiple_requests(
                    request_func,
                    future.args_list,
                    future.kwargs_list
                )

            if not future.done():
                future.set_result(result)

        except Exception as e:
            if not future.done():
                future.set_exception(e)

    async def _handle_multiple_requests(self, request_func, args_list, kwargs_list):
        """Handle multiple similar requests."""
        # For now, execute first and return to all
        # Override in subclasses for specific coalescing logic
        return await request_func(*args_list[0], **kwargs_list[0])
```

## Best Practices

### 1. Async/Await Patterns
- Use async/await consistently throughout the application
- Avoid mixing sync and async code without proper bridges
- Use asyncio.gather() for concurrent independent operations
- Implement proper error handling with try/except blocks

### 2. Resource Management
- Use connection pooling for HTTP requests
- Implement backpressure for high-throughput systems
- Set appropriate timeouts for all operations
- Clean up resources properly

### 3. Performance Considerations
- Batch requests when possible
- Use semaphores to limit concurrency
- Implement caching for expensive operations
- Monitor and tune performance metrics

### 4. Error Handling
- Implement retry logic with exponential backoff
- Use circuit breakers for failing services
- Log errors appropriately
- Provide fallback mechanisms

## Key Takeaways

1. **Async programming** enables non-blocking operations
2. **Streaming processing** handles real-time data efficiently
3. **WebSocket integration** provides real-time communication
4. **Event-driven architecture** decouples components
5. **Connection pooling** optimizes resource usage
6. **Request coalescing** reduces redundant API calls

## Next Steps

In the next section, we'll explore **Debugging and Tracing** techniques to help you effectively debug and monitor complex DSPy applications in production environments.
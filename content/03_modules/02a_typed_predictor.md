# TypedPredictor: Type-Safe Language Model Wrappers

## Prerequisites

- **Previous Section**: [Predict Module](./02-predict-module.md) - Understanding of basic prediction
- **Chapter 2**: Typed Signatures - Familiarity with typed signature design
- **Required Knowledge**: Python type hints, Pydantic models
- **Difficulty Level**: Intermediate to Advanced
- **Estimated Reading Time**: 40 minutes

## Learning Objectives

By the end of this section, you will:
- Understand how TypedPredictor differs from regular Predict
- Master type-safe prediction with structured outputs
- Learn to implement TypedPredictor as an LM wrapper for signatures
- Use Pydantic models for complex output validation
- Apply TypedPredictor in production scenarios requiring guaranteed output formats

## Introduction to TypedPredictor

TypedPredictor is a specialized module that wraps language models to implement signatures with strict type guarantees. While `dspy.Predict` handles basic input-output mapping, TypedPredictor adds a critical layer: **runtime type validation and automatic parsing** of model outputs into structured Python objects.

### The Core Concept

As described in the foundational DSPy paper "Compiling Declarative Language Model Calls into Self-Improving Pipelines," TypedPredictor serves as the bridge between declarative signatures and the underlying language model:

```
Signature (What) -> TypedPredictor (LM Wrapper) -> LM (How) -> Validated Output
```

TypedPredictor ensures that:
1. The LM receives properly formatted prompts based on your signature
2. The LM output is parsed and validated against your type definitions
3. Invalid outputs are caught and can trigger retry mechanisms

## How TypedPredictor Differs from Predict

### Regular Predict: Flexible but Unvalidated

```python
import dspy

# Standard Predict - outputs are strings
class BasicQA(dspy.Signature):
    """Answer questions accurately."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

basic_qa = dspy.Predict(BasicQA)
result = basic_qa(question="What is 2+2?")

# result.answer could be "4", "four", "The answer is 4", etc.
# No guarantee of format or structure
print(type(result.answer))  # <class 'str'>
```

### TypedPredictor: Strict Type Enforcement

```python
import dspy
from pydantic import BaseModel, Field
from typing import List, Optional

class StructuredAnswer(BaseModel):
    """Structured answer with metadata."""
    answer: str = Field(description="The direct answer")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence 0-1")
    sources: List[str] = Field(default_factory=list, description="Supporting sources")

class TypedQA(dspy.Signature):
    """Answer questions with structured output."""
    question: str = dspy.InputField()
    response: StructuredAnswer = dspy.OutputField()

# TypedPredictor enforces the StructuredAnswer schema
typed_qa = dspy.TypedPredictor(TypedQA)
result = typed_qa(question="What is 2+2?")

# result.response is guaranteed to be a StructuredAnswer object
print(type(result.response))  # <class 'StructuredAnswer'>
print(result.response.answer)      # "4"
print(result.response.confidence)  # 0.99
print(result.response.sources)     # ["mathematical axioms"]
```

## TypedPredictor as an LM Wrapper

The key insight from the DSPy paper is that TypedPredictor acts as an **LM wrapper that implements signatures**. This means:

### 1. Automatic Prompt Construction

TypedPredictor translates your signature into appropriate prompts for the underlying LM:

```python
class DataExtractor(dspy.Signature):
    """Extract structured data from text."""
    text: str = dspy.InputField(desc="Raw text to analyze")
    entities: List[dict] = dspy.OutputField(desc="Extracted entities with types")
    relationships: List[str] = dspy.OutputField(desc="Relationships between entities")

# TypedPredictor generates prompts that instruct the LM to:
# 1. Return data in a parseable format (JSON)
# 2. Follow the schema defined by your output types
# 3. Include all required fields

extractor = dspy.TypedPredictor(DataExtractor)
```

### 2. Output Parsing and Validation

TypedPredictor automatically parses LM outputs into your defined types:

```python
from pydantic import BaseModel, validator
from typing import Literal

class SentimentResult(BaseModel):
    """Validated sentiment analysis result."""
    sentiment: Literal["positive", "negative", "neutral"]
    score: float
    key_phrases: List[str]

    @validator('score')
    def validate_score(cls, v):
        if not -1.0 <= v <= 1.0:
            raise ValueError('Score must be between -1 and 1')
        return v

class SentimentAnalysis(dspy.Signature):
    """Analyze text sentiment with validation."""
    text: str = dspy.InputField()
    analysis: SentimentResult = dspy.OutputField()

analyzer = dspy.TypedPredictor(SentimentAnalysis)
result = analyzer(text="I absolutely love this product!")

# The output is validated:
# - sentiment must be one of the allowed values
# - score must be in valid range
# - key_phrases must be a list
print(result.analysis.sentiment)    # "positive"
print(result.analysis.score)        # 0.92
print(result.analysis.key_phrases)  # ["love", "absolutely"]
```

### 3. Retry on Validation Failure

When validation fails, TypedPredictor can automatically retry:

```python
class StrictOutput(BaseModel):
    """Output with strict validation rules."""
    category: Literal["A", "B", "C"]
    reasoning: str = Field(min_length=50)  # Must be at least 50 chars
    tags: List[str] = Field(min_items=2, max_items=5)

class Classifier(dspy.Signature):
    """Classify with strict output requirements."""
    input_text: str = dspy.InputField()
    classification: StrictOutput = dspy.OutputField()

# Configure with retries for validation failures
classifier = dspy.TypedPredictor(
    Classifier,
    max_retries=3,  # Retry up to 3 times on validation failure
    explain_errors=True  # Include validation errors in retry prompts
)

# If the LM returns invalid output (e.g., category="D"),
# TypedPredictor will retry with the error message included
result = classifier(input_text="Sample text for classification")
```

## Practical Applications

### 1. API Response Generation

```python
from pydantic import BaseModel
from typing import Union, Optional
from enum import Enum

class StatusCode(int, Enum):
    OK = 200
    CREATED = 201
    BAD_REQUEST = 400
    NOT_FOUND = 404
    SERVER_ERROR = 500

class ErrorResponse(BaseModel):
    code: StatusCode
    message: str
    details: Optional[dict] = None

class SuccessResponse(BaseModel):
    code: StatusCode
    data: dict
    metadata: Optional[dict] = None

class APIResponseSignature(dspy.Signature):
    """Generate structured API responses."""
    request_type: str = dspy.InputField(desc="Type of API request")
    request_data: dict = dspy.InputField(desc="Request payload")
    response: Union[SuccessResponse, ErrorResponse] = dspy.OutputField()

api_generator = dspy.TypedPredictor(APIResponseSignature)

# Generate API response
result = api_generator(
    request_type="GET /users/123",
    request_data={"include": ["profile", "settings"]}
)

# Response is guaranteed to match one of the schemas
print(result.response.code)  # StatusCode.OK (200)
```

### 2. Document Analysis Pipeline

```python
from pydantic import BaseModel
from typing import List, Optional
from datetime import date

class Entity(BaseModel):
    name: str
    entity_type: Literal["PERSON", "ORG", "LOCATION", "DATE", "OTHER"]
    confidence: float
    start_pos: int
    end_pos: int

class DocumentSection(BaseModel):
    title: str
    content: str
    importance: Literal["high", "medium", "low"]

class DocumentAnalysis(BaseModel):
    title: str
    author: Optional[str]
    date: Optional[str]
    summary: str = Field(min_length=100, max_length=500)
    entities: List[Entity]
    sections: List[DocumentSection]
    keywords: List[str] = Field(min_items=3, max_items=10)

class AnalyzeDocument(dspy.Signature):
    """Perform comprehensive document analysis."""
    document_text: str = dspy.InputField(desc="Full document text")
    analysis: DocumentAnalysis = dspy.OutputField()

doc_analyzer = dspy.TypedPredictor(AnalyzeDocument)

# Analyze a document
result = doc_analyzer(document_text=long_document)

# All fields are properly typed and validated
for entity in result.analysis.entities:
    print(f"{entity.name} ({entity.entity_type}): {entity.confidence:.2f}")
```

### 3. Code Generation with Validation

```python
from pydantic import BaseModel, validator
import ast

class CodeOutput(BaseModel):
    """Generated code with validation."""
    code: str
    language: Literal["python", "javascript", "typescript"]
    imports: List[str]
    description: str

    @validator('code')
    def validate_python_syntax(cls, v, values):
        if values.get('language') == 'python':
            try:
                ast.parse(v)
            except SyntaxError as e:
                raise ValueError(f"Invalid Python syntax: {e}")
        return v

class GenerateCode(dspy.Signature):
    """Generate validated code."""
    task_description: str = dspy.InputField()
    requirements: List[str] = dspy.InputField()
    generated_code: CodeOutput = dspy.OutputField()

code_generator = dspy.TypedPredictor(GenerateCode, max_retries=3)

result = code_generator(
    task_description="Create a function to calculate fibonacci numbers",
    requirements=["Use recursion", "Add memoization", "Include type hints"]
)

# Code is guaranteed to have valid Python syntax
print(result.generated_code.code)
```

## Advanced TypedPredictor Patterns

### Nested Structures

```python
class Address(BaseModel):
    street: str
    city: str
    country: str
    postal_code: str

class Company(BaseModel):
    name: str
    industry: str
    headquarters: Address
    founded_year: int

class Person(BaseModel):
    name: str
    role: str
    company: Optional[Company]
    contact: Optional[dict]

class EntityExtraction(dspy.Signature):
    """Extract nested entity structures."""
    text: str = dspy.InputField()
    people: List[Person] = dspy.OutputField()

# TypedPredictor handles arbitrarily nested structures
extractor = dspy.TypedPredictor(EntityExtraction)
```

### Union Types for Flexible Outputs

```python
from typing import Union

class TextResponse(BaseModel):
    type: Literal["text"] = "text"
    content: str

class TableResponse(BaseModel):
    type: Literal["table"] = "table"
    headers: List[str]
    rows: List[List[str]]

class ChartData(BaseModel):
    type: Literal["chart"] = "chart"
    chart_type: Literal["bar", "line", "pie"]
    labels: List[str]
    values: List[float]

class FlexibleResponse(dspy.Signature):
    """Generate flexible response formats."""
    query: str = dspy.InputField()
    data_type: str = dspy.InputField(desc="Preferred output format")
    response: Union[TextResponse, TableResponse, ChartData] = dspy.OutputField()

responder = dspy.TypedPredictor(FlexibleResponse)
```

### Custom Validation Logic

```python
from pydantic import BaseModel, root_validator

class ConsistentAnalysis(BaseModel):
    """Analysis with cross-field validation."""
    sentiment: Literal["positive", "negative", "neutral"]
    sentiment_score: float
    recommendation: str

    @root_validator
    def validate_consistency(cls, values):
        sentiment = values.get('sentiment')
        score = values.get('sentiment_score', 0)

        # Ensure score matches sentiment
        if sentiment == "positive" and score < 0:
            raise ValueError("Positive sentiment requires non-negative score")
        if sentiment == "negative" and score > 0:
            raise ValueError("Negative sentiment requires non-positive score")

        return values

class ReviewAnalysis(dspy.Signature):
    """Analyze review with consistency checks."""
    review_text: str = dspy.InputField()
    analysis: ConsistentAnalysis = dspy.OutputField()

analyzer = dspy.TypedPredictor(ReviewAnalysis, max_retries=3)
```

## TypedPredictor in the Compilation Process

TypedPredictor integrates seamlessly with DSPy's compilation (optimization) process:

```python
from dspy.teleprompt import BootstrapFewShot, MIPRO

class StructuredQA(dspy.Signature):
    """QA with structured output."""
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: StructuredAnswer = dspy.OutputField()

# Create TypedPredictor module
typed_qa = dspy.TypedPredictor(StructuredQA)

# Define metric that works with structured output
def qa_metric(example, pred, trace=None):
    if not hasattr(pred, 'answer'):
        return 0.0

    # Access structured fields directly
    correctness = example.expected_answer.lower() in pred.answer.answer.lower()
    confidence_bonus = pred.answer.confidence if correctness else 0

    return 0.7 * float(correctness) + 0.3 * confidence_bonus

# Compile with BootstrapFewShot
optimizer = BootstrapFewShot(metric=qa_metric, max_bootstrapped_demos=4)
compiled_qa = optimizer.compile(typed_qa, trainset=training_examples)

# Or use MIPRO for more advanced optimization
mipro = MIPRO(metric=qa_metric, auto="medium")
optimized_qa = mipro.compile(typed_qa, trainset=training_examples)
```

## Advanced TypedPredictor Implementation Patterns

### 1. Schema Composition and Inheritance

Create complex, reusable schemas with inheritance:

```python
from pydantic import BaseModel
from typing import Generic, TypeVar

T = TypeVar('T')

class BaseResponse(BaseModel, Generic[T]):
    """Base response schema with common fields."""
    success: bool = True
    message: str = "Operation completed"
    data: T

    class Config:
        # Enable JSON schema for complex types
        schema_extra = {
            "example": {
                "success": True,
                "message": "Data retrieved successfully",
                "data": {}
            }
        }

class ValidationMetadata(BaseModel):
    """Metadata for validation results."""
    validation_version: str = "1.0"
    schema_hash: str
    validation_timestamp: str

    def compute_hash(self, schema_dict: dict) -> str:
        """Compute hash for schema validation."""
        import hashlib
        import json
        schema_str = json.dumps(schema_dict, sort_keys=True)
        return hashlib.sha256(schema_str.encode()).hexdigest()[:16]

# Compose complex schemas
class AnalysisResult(BaseModel):
    """Complex analysis result with nested structures."""
    summary: str = Field(..., min_length=10, description="Executive summary")
    details: dict = Field(default_factory=dict, description="Detailed findings")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    sources: List[str] = Field(default_factory=list, description="Reference sources")
    metadata: ValidationMetadata = Field(..., description="Validation metadata")

# Use in TypedPredictor
class AnalyzeDataSignature(dspy.Signature):
    """Analyze data with comprehensive output validation."""
    input_data: dict = dspy.InputField(desc="Data to analyze")
    context: str = dspy.InputField(desc="Analysis context and requirements")
    analysis: BaseResponse[AnalysisResult] = dspy.OutputField()

analyzer = dspy.TypedPredictor(AnalyzeDataSignature)
```

### 2. Dynamic Schema Generation

Generate schemas based on runtime requirements:

```python
from pydantic import create_model, Field
from typing import Dict, Any

class DynamicSchemaPredictor:
    """TypedPredictor with dynamic schema generation."""

    def __init__(self, base_fields: Dict[str, Any]):
        self.base_fields = base_fields
        self.model_cache = {}

    def create_dynamic_model(self, name: str, additional_fields: Dict[str, Any] = None):
        """Create a Pydantic model dynamically."""
        # Combine base and additional fields
        all_fields = {**self.base_fields, **(additional_fields or {})}

        # Create model name
        model_name = f"Dynamic{name.replace(' ', '')}Model"

        # Check cache first
        if model_name in self.model_cache:
            return self.model_cache[model_name]

        # Create dynamic model
        DynamicModel = create_model(
            model_name,
            **all_fields
        )

        # Cache the model
        self.model_cache[model_name] = DynamicModel
        return DynamicModel

    def create_typed_predictor(self, schema_name: str, signature_fields: Dict[str, Any]):
        """Create TypedPredictor with dynamic schema."""
        # Generate dynamic model
        OutputModel = self.create_dynamic_model(schema_name)

        # Create signature class
        signature_dict = {
            '__annotations__': {}
        }

        for field_name, field_config in signature_fields.items():
            if field_config.get('input'):
                signature_dict['__annotations__'][field_name] = str
                signature_dict[field_name] = dspy.InputField(
                    desc=field_config.get('description', '')
                )
            else:
                signature_dict['__annotations__'][field_name] = OutputModel
                signature_dict[field_name] = dspy.OutputField(
                    desc=field_config.get('description', '')
                )

        # Create signature dynamically
        DynamicSignature = type(f"{schema_name}Signature", (dspy.Signature,), signature_dict)

        # Return TypedPredictor
        return dspy.TypedPredictor(DynamicSignature)

# Usage example
base_fields = {
    'id': int,
    'timestamp': str,
    'status': str,
    'result': Dict[str, Any]
}

dynamic_predictor = DynamicSchemaPredictor(base_fields)

# Create a data processor with dynamic schema
processor = dynamic_predictor.create_typed_predictor(
    schema_name="DataProcessor",
    signature_fields={
        'raw_data': {'input': True, 'description': 'Raw input data'},
        'processed': {'input': False, 'description': 'Processed output with validation'}
    }
)
```

### 3. Streaming TypedPredictor

Handle real-time data validation:

```python
from typing import AsyncGenerator
import asyncio

class StreamingTypedPredictor:
    """TypedPredictor for streaming data validation."""

    def __init__(self, signature, batch_size: int = 10, timeout: float = 5.0):
        self.predictor = dspy.TypedPredictor(signature)
        self.batch_size = batch_size
        self.timeout = timeout
        self.buffer = []

    async def process_stream(self, data_stream: AsyncGenerator[dict, None]) -> AsyncGenerator[dict, None]:
        """Process streaming data with validation."""
        async for data_item in data_stream:
            self.buffer.append(data_item)

            # Process batch when full or timeout
            if len(self.buffer) >= self.batch_size:
                async for validated_item in self._process_batch():
                    yield validated_item
                self.buffer.clear()

        # Process remaining items
        if self.buffer:
            async for validated_item in self._process_batch():
                yield validated_item

    async def _process_batch(self) -> AsyncGenerator[dict, None]:
        """Process a batch of items."""
        # Create batch processing prompt
        batch_prompt = self._create_batch_prompt(self.buffer)

        # Validate entire batch
        try:
            result = await asyncio.wait_for(
                self._predict_async(batch_prompt),
                timeout=self.timeout
            )

            # Extract validated items
            if hasattr(result, 'validated_data'):
                for item in result.validated_data:
                    yield item

        except asyncio.TimeoutError:
            # Fallback to individual processing
            for item in self.buffer:
                yield item  # Return original if validation fails

    def _create_batch_prompt(self, items: List[dict]) -> str:
        """Create prompt for batch processing."""
        return f"""
        Validate and process the following batch of data items:
        {json.dumps(items, indent=2)}

        Ensure all items conform to the required schema.
        Return validated items in order.
        """

    async def _predict_async(self, prompt: str):
        """Async prediction wrapper."""
        # In a real implementation, this would use an async-compatible LM
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predictor, input_text=prompt)

# Example usage
class StreamingDataSignature(dspy.Signature):
    """Process streaming data with validation."""
    input_data: List[dict] = dspy.InputField(desc="Batch of data items")
    validated_data: List[dict] = dspy.OutputField(desc="Validated data items")
    validation_errors: List[str] = dspy.OutputField(desc="Any validation errors found")

stream_validator = StreamingTypedPredictor(StreamingDataSignature)
```

### 4. Conditional TypedPredictor

Different validation rules based on conditions:

```python
from enum import Enum
from typing import Union, Optional

class DataClass(str, Enum):
    TEXT = "text"
    NUMERIC = "numeric"
    STRUCTURED = "structured"
    MIXED = "mixed"

class ConditionalTypedPredictor:
    """TypedPredictor with conditional validation logic."""

    def __init__(self):
        self.classifier = dspy.Predict("text -> data_class")
        self.predictors = {
            DataClass.TEXT: dspy.TypedPredictor(TextValidationSignature),
            DataClass.NUMERIC: dspy.TypedPredictor(NumericValidationSignature),
            DataClass.STRUCTURED: dspy.TypedPredictor(StructuredValidationSignature),
            DataClass.MIXED: dspy.TypedPredictor(MixedValidationSignature)
        }

    def forward(self, input_data: str) -> dspy.Prediction:
        """Route to appropriate validator based on data class."""
        # First classify the input
        classification = self.classifier(text=input_data)
        data_class = DataClass(classification.data_class.lower())

        # Route to appropriate validator
        predictor = self.predictors[data_class]
        result = predictor(input_data=input_data)

        # Add metadata
        result.data_class = data_class
        result.validator_type = type(predictor).__name__

        return result

# Define different validation schemas
class TextValidationResult(BaseModel):
    """Validation result for text data."""
    is_valid: bool
    language: str
    sentiment: str
    entities: List[str]
    quality_score: float

class NumericValidationResult(BaseModel):
    """Validation result for numeric data."""
    is_valid: bool
    data_type: str  # int, float, decimal
    range_check: bool
    outliers: List[float]
    statistics: dict

class StructuredValidationResult(BaseModel):
    """Validation result for structured data."""
    is_valid: bool
    schema_version: str
    missing_fields: List[str]
    extra_fields: List[str]
    field_types: dict
    nested_objects: int

# Signature definitions
class TextValidationSignature(dspy.Signature):
    """Validate text data."""
    input_data: str = dspy.InputField()
    validation: TextValidationResult = dspy.OutputField()

class NumericValidationSignature(dspy.Signature):
    """Validate numeric data."""
    input_data: str = dspy.InputField()
    validation: NumericValidationResult = dspy.OutputField()

class StructuredValidationSignature(dspy.Signature):
    """Validate structured data."""
    input_data: str = dspy.InputField()
    validation: StructuredValidationResult = dspy.OutputField()

class MixedValidationSignature(dspy.Signature):
    """Validate mixed-type data."""
    input_data: str = dspy.InputField()
    validation: dict = dspy.OutputField(desc="Mixed validation results")

# Usage
conditional_validator = ConditionalTypedPredictor()
result = conditional_validator(input_data="Sample text data...")
```

### 5. TypedPredictor with Versioning

Maintain backward compatibility with schema evolution:

```python
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import json

class VersionedTypedPredictor:
    """TypedPredictor with schema versioning support."""

    def __init__(self):
        self.versions = {}
        self.migration_handlers = {}
        self.current_version = "1.0.0"

    def register_version(self, version: str, model_class: type, migration_handler=None):
        """Register a schema version."""
        self.versions[version] = model_class
        if migration_handler:
            self.migration_handlers[version] = migration_handler

    def migrate_data(self, data: dict, from_version: str, to_version: str) -> dict:
        """Migrate data between schema versions."""
        current_data = data

        # Apply migrations in order
        versions = sorted(self.versions.keys())
        start_idx = versions.index(from_version)
        end_idx = versions.index(to_version)

        for i in range(start_idx, end_idx):
            current_version = versions[i]
            next_version = versions[i + 1]

            if next_version in self.migration_handlers:
                current_data = self.migration_handlers[next_version](current_data)

        return current_data

    def create_predictor(self, version: str = None):
        """Create TypedPredictor for specific version."""
        target_version = version or self.current_version

        if target_version not in self.versions:
            raise ValueError(f"Version {target_version} not registered")

        model_class = self.versions[target_version]

        # Create dynamic signature
        class VersionedSignature(dspy.Signature):
            """Versioned data processing signature."""
            input_data: dict = dspy.InputField(desc="Input data")
            version: str = dspy.InputField(desc="Target schema version")
            output: model_class = dspy.OutputField(desc="Validated output")

        return dspy.TypedPredictor(VersionedSignature)

    def process_with_versioning(self, input_data: dict, input_version: str,
                              target_version: str = None) -> dspy.Prediction:
        """Process data with automatic version migration."""
        target_version = target_version or self.current_version

        # Migrate data if needed
        if input_version != target_version:
            migrated_data = self.migrate_data(input_data, input_version, target_version)
        else:
            migrated_data = input_data

        # Process with target version schema
        predictor = self.create_predictor(target_version)
        return predictor(input_data=migrated_data, version=target_version)

# Example schema versions
class UserProfileV1(BaseModel):
    """User profile schema v1.0."""
    name: str
    email: str
    age: Optional[int] = None

class UserProfileV2(BaseModel):
    """User profile schema v2.0 - added fields."""
    name: str = Field(..., min_length=2)
    email: str = Field(..., regex=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    age: Optional[int] = Field(None, ge=0, le=150)
    phone: Optional[str] = None
    preferences: dict = Field(default_factory=dict)

# Migration handler from v1 to v2
def migrate_v1_to_v2(data: dict) -> dict:
    """Migrate user profile from v1 to v2."""
    migrated = data.copy()
    migrated['preferences'] = {}  # Add new field
    migrated['phone'] = None  # Add new field
    return migrated

# Register versions
versioned_predictor = VersionedTypedPredictor()
versioned_predictor.register_version("1.0.0", UserProfileV1)
versioned_predictor.register_version("2.0.0", UserProfileV2, migrate_v1_to_v2)

# Usage
old_data = {"name": "John Doe", "email": "john@example.com"}
result = versioned_predictor.process_with_versioning(
    input_data=old_data,
    input_version="1.0.0",
    target_version="2.0.0"
)
```

### 6. TypedPredictor with Performance Optimization

Optimize for high-throughput scenarios:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import multiprocessing
from typing import List, Tuple

class OptimizedTypedPredictor:
    """High-performance TypedPredictor with optimization strategies."""

    def __init__(self, signature, batch_size: int = 32, max_workers: int = None):
        self.predictor = dspy.TypedPredictor(signature)
        self.batch_size = batch_size
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Validation cache
        self._validation_cache = {}
        self._cache_size_limit = 10000

    @lru_cache(maxsize=1000)
    def _cached_schema_validation(self, data_hash: str, schema_hash: str) -> bool:
        """Cached schema validation."""
        # In practice, this would validate against cached schema
        return True

    def process_batch_parallel(self, batch_data: List[dict]) -> List[dspy.Prediction]:
        """Process a batch of items in parallel."""
        # Split batch into chunks
        chunks = [batch_data[i:i + self.batch_size]
                 for i in range(0, len(batch_data), self.batch_size)]

        # Process chunks in parallel
        futures = []
        for chunk in chunks:
            future = self.executor.submit(self._process_chunk, chunk)
            futures.append(future)

        # Collect results
        results = []
        for future in as_completed(futures):
            chunk_results = future.result()
            results.extend(chunk_results)

        return results

    def _process_chunk(self, chunk: List[dict]) -> List[dspy.Prediction]:
        """Process a chunk of items."""
        results = []

        # Create batch prompt for chunk
        batch_prompt = self._create_optimized_batch_prompt(chunk)

        try:
            # Process entire chunk
            chunk_result = self.predictor(batch_input=batch_prompt)

            # Parse individual results
            if hasattr(chunk_result, 'outputs'):
                results = chunk_result.outputs
            else:
                # Fallback to individual processing
                for item in chunk:
                    result = self.predictor(**item)
                    results.append(result)

        except Exception as e:
            # Handle errors gracefully
            for item in chunk:
                error_result = dspy.Prediction(
                    error=str(e),
                    original_input=item
                )
                results.append(error_result)

        return results

    def _create_optimized_batch_prompt(self, chunk: List[dict]) -> str:
        """Create optimized batch processing prompt."""
        # Pre-validate cached items
        uncached_items = []
        for item in chunk:
            item_hash = self._compute_item_hash(item)
            if item_hash not in self._validation_cache:
                uncached_items.append(item)

        # Create efficient prompt
        prompt = f"""
        Process this batch of {len(chunk)} items efficiently.

        Items to process:
        {json.dumps(uncached_items, indent=2)}

        Apply schema validation and return structured results.
        Use batch processing for efficiency.
        """

        return prompt

    def _compute_item_hash(self, item: dict) -> str:
        """Compute hash for item caching."""
        import hashlib
        import json
        item_str = json.dumps(item, sort_keys=True)
        return hashlib.md5(item_str.encode()).hexdigest()[:16]

    def optimize_schema(self, sample_data: List[dict]) -> dict:
        """Optimize schema based on sample data."""
        # Analyze common patterns
        field_frequencies = {}
        field_types = {}

        for item in sample_data:
            for field, value in item.items():
                field_frequencies[field] = field_frequencies.get(field, 0) + 1
                field_types[field] = type(value).__name__

        # Generate optimized schema
        schema = {
            "required_fields": [f for f, freq in field_frequencies.items()
                              if freq > len(sample_data) * 0.8],
            "optional_fields": [f for f, freq in field_frequencies.items()
                              if freq <= len(sample_data) * 0.8],
            "field_types": field_types,
            "optimizations": {
                "use_optional": len(field_frequencies) > 10,
                "batch_size": self.batch_size,
                "cache_enabled": True
            }
        }

        return schema

    def get_performance_metrics(self) -> dict:
        """Get performance metrics."""
        return {
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
            "cache_size": len(self._validation_cache),
            "cache_hit_rate": getattr(self, '_cache_hits', 0) / max(1, getattr(self, '_cache_requests', 1))
        }

# Usage example for high-throughput scenario
signature = dspy.Signature("data -> validated_output")
optimized_predictor = OptimizedTypedPredictor(signature, batch_size=64)

# Process large dataset
large_dataset = [{"data": f"item_{i}"} for i in range(1000)]
results = optimized_predictor.process_batch_parallel(large_dataset)

# Get performance metrics
metrics = optimized_predictor.get_performance_metrics()
```

## Error Handling and Debugging

### Handling Validation Errors

```python
from pydantic import ValidationError

class RobustTypedPredictor:
    """Wrapper with robust error handling."""

    def __init__(self, signature, fallback_fn=None):
        self.predictor = dspy.TypedPredictor(signature, max_retries=3)
        self.fallback_fn = fallback_fn

    def __call__(self, **kwargs):
        try:
            return self.predictor(**kwargs)
        except ValidationError as e:
            print(f"Validation failed after retries: {e}")
            if self.fallback_fn:
                return self.fallback_fn(**kwargs)
            raise
        except Exception as e:
            print(f"Prediction error: {e}")
            raise

# Usage with fallback
def simple_fallback(**kwargs):
    """Fallback to unstructured response."""
    simple = dspy.Predict("question -> answer")
    return simple(**kwargs)

robust_qa = RobustTypedPredictor(TypedQA, fallback_fn=simple_fallback)
```

### Debugging Type Mismatches

```python
import dspy

# Enable detailed tracing
dspy.settings.configure(trace="all")

class DebugSignature(dspy.Signature):
    """Signature for debugging."""
    input_text: str = dspy.InputField()
    output: ComplexStructure = dspy.OutputField()

predictor = dspy.TypedPredictor(DebugSignature)

# Run prediction
result = predictor(input_text="Test input")

# Inspect the raw LM output before parsing
print("Raw LM response:", predictor.lm.last_request_.response)

# Check what was sent to the LM
print("Prompt sent:", predictor.lm.last_request_.prompt)
```

## Best Practices

### 1. Start Simple, Add Complexity Gradually

```python
# Start with simple types
class SimpleOutput(BaseModel):
    result: str
    confidence: float

# Add complexity as needed
class EnhancedOutput(SimpleOutput):
    sources: List[str] = []
    metadata: Optional[dict] = None
```

### 2. Use Descriptive Field Descriptions

```python
class WellDocumentedOutput(BaseModel):
    """Output with clear descriptions for the LM."""
    category: str = Field(
        description="One of: technology, business, science, other"
    )
    summary: str = Field(
        description="A 2-3 sentence summary of the main points"
    )
    key_facts: List[str] = Field(
        description="List of 3-5 key factual statements from the text"
    )
```

### 3. Set Appropriate Retry Limits

```python
# For simple outputs - fewer retries
simple_predictor = dspy.TypedPredictor(SimpleSignature, max_retries=2)

# For complex outputs - more retries
complex_predictor = dspy.TypedPredictor(ComplexSignature, max_retries=5)
```

### 4. Combine with Assertions for Additional Validation

```python
class ValidatedPredictor(dspy.Module):
    """TypedPredictor with additional semantic validation."""

    def __init__(self, signature):
        super().__init__()
        self.typed_predict = dspy.TypedPredictor(signature)

    def forward(self, **kwargs):
        result = self.typed_predict(**kwargs)

        # Additional semantic checks beyond type validation
        dspy.Assert(
            len(result.output.summary) >= 50,
            "Summary must be at least 50 characters"
        )

        return result
```

## Summary

TypedPredictor is a powerful module that brings type safety to language model interactions:

- **Type-Safe Outputs**: Guarantees outputs match your defined schemas
- **LM Wrapper Pattern**: Acts as the bridge between signatures and language models
- **Automatic Validation**: Uses Pydantic for runtime validation
- **Retry Mechanisms**: Handles validation failures gracefully
- **Compilation Compatible**: Works seamlessly with DSPy optimizers

### Key Takeaways

1. **Use TypedPredictor** when you need guaranteed output structure
2. **Leverage Pydantic models** for complex validation rules
3. **Configure appropriate retries** for your use case complexity
4. **Combine with assertions** for semantic validation beyond types
5. **Start simple** and add complexity as requirements evolve

## Next Steps

- [ChainOfThought Module](./03-chainofthought.md) - Add reasoning to typed predictions
- [Assertions Module](./08-assertions.md) - Combine type safety with semantic constraints
- [Custom Modules](./05-custom-modules.md) - Build custom typed modules
- [Exercises](./07-exercises.md) - Practice with TypedPredictor patterns

## Further Reading

- [DSPy Paper: Compiling Declarative Language Model Calls](https://arxiv.org/abs/2310.03714) - Section on LM wrappers
- [Pydantic Documentation](https://docs.pydantic.dev/) - Advanced validation patterns
- [DSPy Documentation: TypedPredictor](https://dspy-docs.vercel.app/docs/deep-dive/typed-predictor)

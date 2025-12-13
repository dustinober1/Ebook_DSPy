# Adapters and Tools: Extending DSPy Capabilities

## Introduction

Adapters and tools are the building blocks that allow DSPy to integrate with external systems, handle specialized tasks, and extend its core functionality. Understanding how to create and use adapters is crucial for building production-ready applications that need to work with databases, APIs, file systems, and other external resources.

## Understanding DSPy Adapters

### What is an Adapter?

An adapter is a component that bridges DSPy with external systems or provides specialized functionality. Adapters follow the interface principle, allowing seamless integration while maintaining consistency within the DSPy ecosystem.

### Types of Adapters

1. **Data Adapters**: Connect to databases, file systems, APIs
2. **Tool Adapters**: Provide specialized functionality (calculators, validators)
3. **Integration Adapters**: Connect with external services (cloud providers, monitoring)
4. **Custom Adapters**: Domain-specific adapters for specialized use cases

## Built-in DSPy Adapters

### Database Adapter

```python
import dspy
from dspy.adapters import DatabaseAdapter

class PostgreSQLAdapter(DatabaseAdapter):
    """PostgreSQL database adapter for DSPy."""

    def __init__(self, connection_string, table_name="dspy_data"):
        super().__init__()
        self.connection_string = connection_string
        self.table_name = table_name
        self._connection = None

    def connect(self):
        """Establish database connection."""
        import psycopg2
        self._connection = psycopg2.connect(self.connection_string)
        return self._connection

    def query(self, sql_query, params=None):
        """Execute SQL query and return results."""
        if not self._connection:
            self.connect()

        cursor = self._connection.cursor()
        cursor.execute(sql_query, params or ())
        results = cursor.fetchall()
        cursor.close()
        return results

    def insert(self, data):
        """Insert data into database."""
        columns = list(data.keys())
        values = list(data.values())
        placeholders = ", ".join(["%s"] * len(values))

        sql = f"""
        INSERT INTO {self.table_name} ({", ".join(columns)})
        VALUES ({placeholders})
        """

        if not self._connection:
            self.connect()

        cursor = self._connection.cursor()
        cursor.execute(sql, values)
        self._connection.commit()
        cursor.close()

    def close(self):
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
```

### API Adapter

```python
class APIAdapter(dspy.Adapter):
    """Generic API adapter for DSPy integration."""

    def __init__(self, base_url, headers=None, auth=None):
        super().__init__()
        self.base_url = base_url
        self.headers = headers or {}
        self.auth = auth
        self.session = None

    def _get_session(self):
        """Initialize HTTP session."""
        import requests
        if not self.session:
            self.session = requests.Session()
            self.session.headers.update(self.headers)
            if self.auth:
                self.session.auth = self.auth
        return self.session

    def get(self, endpoint, params=None):
        """Make GET request to API."""
        session = self._get_session()
        url = f"{self.base_url}/{endpoint}"
        response = session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def post(self, endpoint, data=None):
        """Make POST request to API."""
        session = self._get_session()
        url = f"{self.base_url}/{endpoint}"
        response = session.post(url, json=data)
        response.raise_for_status()
        return response.json()

    def put(self, endpoint, data=None):
        """Make PUT request to API."""
        session = self._get_session()
        url = f"{self.base_url}/{endpoint}"
        response = session.put(url, json=data)
        response.raise_for_status()
        return response.json()
```

## Creating Custom Adapters

### File System Adapter

```python
import os
import json
import pickle
from pathlib import Path

class FileSystemAdapter(dspy.Adapter):
    """Adapter for file system operations."""

    def __init__(self, base_path="."):
        super().__init__()
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def read_file(self, filename, encoding='utf-8'):
        """Read file content."""
        file_path = self.base_path / filename
        return file_path.read_text(encoding=encoding)

    def write_file(self, filename, content, encoding='utf-8'):
        """Write content to file."""
        file_path = self.base_path / filename
        file_path.write_text(content, encoding=encoding)

    def read_json(self, filename):
        """Read JSON file."""
        file_path = self.base_path / filename
        return json.loads(file_path.read_text())

    def write_json(self, filename, data, indent=2):
        """Write data to JSON file."""
        file_path = self.base_path / filename
        file_path.write_text(json.dumps(data, indent=indent))

    def save_pickle(self, filename, obj):
        """Save object as pickle."""
        file_path = self.base_path / filename
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)

    def load_pickle(self, filename):
        """Load object from pickle."""
        file_path = self.base_path / filename
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def list_files(self, pattern="*"):
        """List files matching pattern."""
        return list(self.base_path.glob(pattern))

    def delete_file(self, filename):
        """Delete file."""
        file_path = self.base_path / filename
        if file_path.exists():
            file_path.unlink()
```

### Cache Adapter

```python
import time
from typing import Any, Optional

class CacheAdapter(dspy.Adapter):
    """Generic caching adapter."""

    def __init__(self, ttl=3600):
        super().__init__()
        self.cache = {}
        self.ttl = ttl  # Time to live in seconds

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]  # Expired
        return None

    def set(self, key: str, value: Any):
        """Set value in cache."""
        self.cache[key] = (value, time.time())

    def delete(self, key: str):
        """Delete value from cache."""
        if key in self.cache:
            del self.cache[key]

    def clear(self):
        """Clear all cache."""
        self.cache.clear()

    def size(self):
        """Get cache size."""
        return len(self.cache)

    def cleanup_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if current_time - timestamp >= self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]
```

## Specialized Tools

### Calculator Tool

```python
import operator
import math

class CalculatorTool(dspy.Tool):
    """Mathematical calculator tool."""

    def __init__(self):
        super().__init__()
        self.operations = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '^': operator.pow,
            '%': operator.mod,
        }

    def calculate(self, expression: str) -> float:
        """Evaluate mathematical expression."""
        # Simple expression parser
        tokens = expression.split()
        if len(tokens) != 3:
            raise ValueError("Expression must be in format: num operator num")

        try:
            num1 = float(tokens[0])
            op = tokens[1]
            num2 = float(tokens[2])
        except ValueError:
            raise ValueError("Invalid numbers in expression")

        if op not in self.operations:
            raise ValueError(f"Unsupported operator: {op}")

        return self.operations[op](num1, num2)

    def advanced_calculate(self, expression: str) -> float:
        """Evaluate more complex expressions."""
        # For complex expressions, use eval with safety checks
        allowed_names = {
            "sqrt": math.sqrt,
            "log": math.log,
            "exp": math.exp,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e
        }

        # Safety check: only allowed functions
        for name in expression:
            if name.isalpha() and name not in allowed_names:
                raise ValueError(f"Function {name} not allowed")

        return eval(expression, {"__builtins__": {}}, allowed_names)
```

### Text Processing Tool

```python
import re
from typing import List, Dict

class TextProcessingTool(dspy.Tool):
    """Advanced text processing tool."""

    def __init__(self):
        super().__init__()

    def extract_emails(self, text: str) -> List[str]:
        """Extract email addresses from text."""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(pattern, text)

    def extract_phone_numbers(self, text: str) -> List[str]:
        """Extract phone numbers from text."""
        patterns = [
            r'\b\d{3}-\d{3}-\d{4}\b',  # 123-456-7890
            r'\b\(\d{3}\) \d{3}-\d{4}\b',  # (123) 456-7890
            r'\b\d{10}\b'  # 1234567890
        ]
        numbers = []
        for pattern in patterns:
            numbers.extend(re.findall(pattern, text))
        return numbers

    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text."""
        pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(pattern, text)

    def clean_text(self, text: str, remove_special_chars=True, lowercase=True) -> str:
        """Clean and normalize text."""
        if lowercase:
            text = text.lower()

        if remove_special_chars:
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return re.findall(r'\b\w+\b', text.lower())

    def get_word_frequency(self, text: str) -> Dict[str, int]:
        """Get word frequency dictionary."""
        tokens = self.tokenize(text)
        frequency = {}
        for token in tokens:
            frequency[token] = frequency.get(token, 0) + 1
        return frequency

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract top keywords from text."""
        frequency = self.get_word_frequency(text)
        # Sort by frequency and return top N
        sorted_words = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:top_n]]
```

### Validation Tool

```python
import re
from datetime import datetime

class ValidationTool(dspy.Tool):
    """Data validation tool."""

    def __init__(self):
        super().__init__()
        self.validators = {
            'email': self.validate_email,
            'phone': self.validate_phone,
            'url': self.validate_url,
            'date': self.validate_date,
            'credit_card': self.validate_credit_card
        }

    def validate_email(self, email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    def validate_phone(self, phone: str) -> bool:
        """Validate phone number format."""
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', phone)
        return len(digits) == 10

    def validate_url(self, url: str) -> bool:
        """Validate URL format."""
        pattern = r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?$'
        return bool(re.match(pattern, url))

    def validate_date(self, date_str: str, format='%Y-%m-%d') -> bool:
        """Validate date format."""
        try:
            datetime.strptime(date_str, format)
            return True
        except ValueError:
            return False

    def validate_credit_card(self, card_number: str) -> bool:
        """Validate credit card number using Luhn algorithm."""
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', card_number)

        if len(digits) not in [13, 14, 15, 16, 19]:
            return False

        # Luhn algorithm
        total = 0
        num_digits = len(digits)
        oddeven = num_digits & 1

        for i, digit in enumerate(digits):
            d = int(digit)
            if ((i & 1) ^ oddeven) == 0:
                d = d * 2
                if d > 9:
                    d -= 9
            total += d

        return total % 10 == 0

    def validate(self, data: Dict[str, Any], rules: Dict[str, str]) -> Dict[str, bool]:
        """Validate data against rules."""
        results = {}
        for field, validation_type in rules.items():
            if field not in data:
                results[field] = False
                continue

            if validation_type not in self.validators:
                raise ValueError(f"Unknown validation type: {validation_type}")

            validator = self.validators[validation_type]
            results[field] = validator(str(data[field]))

        return results
```

## Integration with External Services

### Google Sheets Adapter

```python
import gspread
from google.oauth2.service_account import Credentials

class GoogleSheetsAdapter(dspy.Adapter):
    """Google Sheets adapter for DSPy."""

    def __init__(self, credentials_file=None, scopes=None):
        super().__init__()
        self.credentials_file = credentials_file
        self.scopes = scopes or ['https://www.googleapis.com/auth/spreadsheets']
        self.client = None

    def _get_client(self):
        """Initialize Google Sheets client."""
        if not self.client:
            creds = Credentials.from_service_account_file(
                self.credentials_file,
                scopes=self.scopes
            )
            self.client = gspread.authorize(creds)
        return self.client

    def read_worksheet(self, spreadsheet_id, worksheet_name):
        """Read data from worksheet."""
        client = self._get_client()
        spreadsheet = client.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet(worksheet_name)
        return worksheet.get_all_records()

    def write_worksheet(self, spreadsheet_id, worksheet_name, data):
        """Write data to worksheet."""
        client = self._get_client()
        spreadsheet = client.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet(worksheet_name)

        # Clear existing data
        worksheet.clear()

        # Write headers
        if data:
            headers = list(data[0].keys())
            worksheet.append_row(headers)

            # Write data rows
            for row in data:
                worksheet.append_row([row.get(header, "") for header in headers])

    def append_row(self, spreadsheet_id, worksheet_name, row_data):
        """Append a row to worksheet."""
        client = self._get_client()
        spreadsheet = client.open_by_key(spreadsheet_id)
        worksheet = spreadsheet.worksheet(worksheet_name)
        worksheet.append_row(row_data)
```

### AWS S3 Adapter

```python
import boto3
from botocore.exceptions import ClientError

class S3Adapter(dspy.Adapter):
    """AWS S3 adapter for DSPy."""

    def __init__(self, bucket_name, aws_access_key=None, aws_secret_key=None, region='us-east-1'):
        super().__init__()
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )

    def upload_file(self, file_path, object_name=None):
        """Upload file to S3."""
        if object_name is None:
            object_name = os.path.basename(file_path)

        try:
            self.s3_client.upload_file(file_path, self.bucket_name, object_name)
            return f"s3://{self.bucket_name}/{object_name}"
        except ClientError as e:
            raise Exception(f"Failed to upload file: {e}")

    def download_file(self, object_name, file_path):
        """Download file from S3."""
        try:
            self.s3_client.download_file(self.bucket_name, object_name, file_path)
            return file_path
        except ClientError as e:
            raise Exception(f"Failed to download file: {e}")

    def list_objects(self, prefix=''):
        """List objects in S3 bucket."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            return response.get('Contents', [])
        except ClientError as e:
            raise Exception(f"Failed to list objects: {e}")

    def delete_object(self, object_name):
        """Delete object from S3."""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=object_name)
            return True
        except ClientError as e:
            raise Exception(f"Failed to delete object: {e}")
```

## Using Adapters in DSPy Modules

### RAG System with Database Adapter

```python
class EnhancedRAG(dspy.Module):
    """RAG system with database persistence."""

    def __init__(self, db_adapter):
        super().__init__()
        self.db = db_adapter
        self.retrieve = dspy.Retrieve(k=5)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        # Check cache first
        cache_key = f"rag:{hash(question)}"
        cached_result = self.db.get(cache_key)

        if cached_result:
            return dspy.Prediction(**cached_result)

        # Process normally
        retrieved = self.retrieve(question=question)
        context = retrieved.passages
        prediction = self.generate(context="\n".join(context), question=question)

        # Cache result
        result = {
            "answer": prediction.answer,
            "context": context,
            "reasoning": prediction.rationale
        }
        self.db.set(cache_key, result)

        return dspy.Prediction(**result)
```

### Agent with Tool Integration

```python
class ToolEnabledAgent(dspy.Module):
    """Agent that can use various tools."""

    def __init__(self):
        super().__init__()
        self.tools = {
            'calculator': CalculatorTool(),
            'text_processor': TextProcessingTool(),
            'validator': ValidationTool()
        }
        self.decide_tool = dspy.Predict("task -> tool_name, parameters")
        self.execute_tool = dspy.Predict("tool_name, parameters -> result")

    def forward(self, task):
        # Decide which tool to use
        decision = self.decide_tool(task=task)

        if decision.tool_name in self.tools:
            # Execute tool
            tool = self.tools[decision.tool_name]
            if hasattr(tool, decision.parameters.split('.')[0]):
                result = getattr(tool, decision.parameters.split('.')[0])(
                    task
                )
            else:
                result = tool.calculate(task)  # Default for calculator
        else:
            result = f"Unknown tool: {decision.tool_name}"

        return dspy.Prediction(
            task=task,
            tool_used=decision.tool_name,
            result=result
        )
```

## Best Practices for Adapters

### 1. Error Handling
```python
class ResilientAdapter(dspy.Adapter):
    def __init__(self):
        super().__init__()
        self.max_retries = 3
        self.retry_delay = 1

    def call_with_retry(self, func, *args, **kwargs):
        """Call function with retry logic."""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(self.retry_delay * (2 ** attempt))
```

### 2. Connection Pooling
```python
import threading
from queue import Queue

class ConnectionPool:
    def __init__(self, create_connection, max_size=10):
        self.create_connection = create_connection
        self.max_size = max_size
        self.pool = Queue(maxsize=max_size)
        self.lock = threading.Lock()

    def get_connection(self):
        if not self.pool.empty():
            return self.pool.get()
        else:
            return self.create_connection()

    def return_connection(self, connection):
        if not self.pool.full():
            self.pool.put(connection)
```

### 3. Configuration Management
```python
class ConfigurableAdapter(dspy.Adapter):
    def __init__(self, config_file=None):
        super().__init__()
        self.config = self.load_config(config_file)

    def load_config(self, config_file):
        """Load configuration from file."""
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        return {}

    def get_config(self, key, default=None):
        """Get configuration value."""
        return self.config.get(key, default)
```

## Testing Adapters

### Unit Testing
```python
import unittest
from unittest.mock import Mock, patch

class TestCalculatorTool(unittest.TestCase):
    def setUp(self):
        self.calculator = CalculatorTool()

    def test_basic_addition(self):
        result = self.calculator.calculate("2 + 3")
        self.assertEqual(result, 5)

    def test_invalid_expression(self):
        with self.assertRaises(ValueError):
            self.calculator.calculate("2 + three")

    def test_division_by_zero(self):
        with self.assertRaises(ZeroDivisionError):
            self.calculator.calculate("5 / 0")
```

### Integration Testing
```python
class TestAPIAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = APIAdapter("https://api.example.com")
        self.session_mock = Mock()

    @patch('requests.Session')
    def test_get_request(self, mock_session):
        mock_session.return_value.get.return_value.json.return_value = {"status": "ok"}
        result = self.adapter.get("test")
        self.assertEqual(result, {"status": "ok"})
```

## Key Takeaways

1. **Adapters bridge** DSPy with external systems
2. **Custom adapters** enable domain-specific integrations
3. **Tools provide** specialized functionality
4. **Error handling** is essential for robust adapters
5. **Testing ensures** adapter reliability
6. **Configuration** makes adapters flexible

## Next Steps

In the next section, we'll explore **Caching and Performance** techniques to build high-performance DSPy applications that can scale effectively.
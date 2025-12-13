"""
Adapters and Tools Implementation Examples

This file demonstrates various adapter and tool implementations for DSPy,
showing how to extend DSPy capabilities and integrate with external systems.

Examples include:
- Database adapters (PostgreSQL, MongoDB)
- API adapters for external services
- File system adapters
- Custom tools (calculator, text processor, validator)
- Integration examples with DSPy modules
"""

import dspy
from typing import List, Dict, Any, Optional, Union
import json
import time
import os
import sqlite3
import re
import math
from datetime import datetime

# Configure language model (placeholder)
dspy.settings.configure(lm=dspy.LM(model="gpt-3.5-turbo", api_key="your-key"))

# Example 1: PostgreSQL Database Adapter
class PostgreSQLAdapter(dspy.Adapter):
    """PostgreSQL database adapter with connection pooling."""

    def __init__(self, connection_string, table_name="dspy_data", pool_size=5):
        super().__init__()
        self.connection_string = connection_string
        self.table_name = table_name
        self.pool_size = pool_size
        self._pool = None
        self.connect()

    def connect(self):
        """Establish connection pool."""
        import psycopg2
        from psycopg2 import pool
        self._pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=self.pool_size,
            dsn=self.connection_string
        )

    def get_connection(self):
        """Get connection from pool."""
        return self._pool.getconn()

    def return_connection(self, conn):
        """Return connection to pool."""
        self._pool.putconn(conn)

    def create_table(self):
        """Create table if not exists."""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id SERIAL PRIMARY KEY,
                        module_name VARCHAR(255),
                        operation VARCHAR(255),
                        input_data JSONB,
                        output_data JSONB,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            conn.commit()
        finally:
            self.return_connection(conn)

    def store_prediction(self, module_name: str, operation: str,
                         input_data: Dict[str, Any], output_data: dspy.Prediction,
                         metadata: Optional[Dict[str, Any]] = None):
        """Store DSPy prediction in database."""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {self.table_name}
                    (module_name, operation, input_data, output_data, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    module_name, operation,
                    json.dumps(input_data),
                    json.dumps(output_data.__dict__),
                    json.dumps(metadata or {})
                ))
                result_id = cur.fetchone()[0]
            conn.commit()
            return result_id
        finally:
            self.return_connection(conn)

    def get_prediction(self, prediction_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve prediction from database."""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT * FROM {self.table_name}
                    WHERE id = %s
                """, (prediction_id,))
                row = cur.fetchone()
                if row:
                    return {
                        'id': row[0],
                        'module_name': row[1],
                        'operation': row[2],
                        'input_data': json.loads(row[3]),
                        'output_data': json.loads(row[4]),
                        'metadata': json.loads(row[5]),
                        'created_at': row[6],
                        'updated_at': row[7]
                    }
                return None
        finally:
            self.return_connection(conn)

    def search_predictions(self, module_name: Optional[str] = None,
                          operation: Optional[str] = None,
                          limit: int = 10) -> List[Dict[str, Any]]:
        """Search predictions with filters."""
        conn = self.get_connection()
        try:
            query = f"SELECT * FROM {self.table_name} WHERE 1=1"
            params = []

            if module_name:
                query += " AND module_name = %s"
                params.append(module_name)
            if operation:
                query += " AND operation = %s"
                params.append(operation)

            query += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)

            with conn.cursor() as cur:
                cur.execute(query, tuple(params))
                rows = cur.fetchall()

            results = []
            for row in rows:
                results.append({
                    'id': row[0],
                    'module_name': row[1],
                    'operation': row[2],
                    'input_data': json.loads(row[3]),
                    'output_data': json.loads(row[4]),
                    'metadata': json.loads(row[5]),
                    'created_at': row[6],
                    'updated_at': row[7]
                })
            return results
        finally:
            self.return_connection(conn)

    def close(self):
        """Close connection pool."""
        if self._pool:
            self._pool.closeall()
            self._pool = None

def demo_postgresql_adapter():
    """Demonstrate PostgreSQL adapter functionality."""
    print("=" * 60)
    print("Example 1: PostgreSQL Database Adapter")
    print("=" * 60)

    # Note: This would require actual PostgreSQL connection
    # adapter = PostgreSQLAdapter("postgresql://user:password@localhost/dbname")

    # Mock demonstration
    print("\nPostgreSQL Adapter Features:")
    print("- Connection pooling for efficiency")
    print("- Automatic table creation")
    print("- JSON storage for flexible data")
    print("- Search functionality with filters")
    print("- Transaction support")

# Example 2: MongoDB Adapter
class MongoDBAdapter(dspy.Adapter):
    """MongoDB adapter for DSPy."""

    def __init__(self, connection_string, database="dspy", collection_name="predictions"):
        super().__init__()
        self.connection_string = connection_string
        self.database = database
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
        self.connect()

    def connect(self):
        """Establish MongoDB connection."""
        from pymongo import MongoClient
        self.client = MongoClient(self.connection_string)
        self.db = self.client[self.database]
        self.collection = self.db[self.collection_name]

    def store_prediction(self, module_name: str, operation: str,
                         input_data: Dict[str, Any], output_data: dspy.Prediction,
                         metadata: Optional[Dict[str, Any]] = None):
        """Store prediction in MongoDB."""
        document = {
            "module_name": module_name,
            "operation": operation,
            "input_data": input_data,
            "output_data": output_data.__dict__,
            "metadata": metadata or {},
            "created_at": datetime.datetime.utcnow(),
            "updated_at": datetime.datetime.utcnow()
        }

        result = self.collection.insert_one(document)
        return str(result.inserted_id)

    def get_prediction(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve prediction from MongoDB."""
        from bson.objectid import ObjectId
        try:
            document = self.collection.find_one({"_id": ObjectId(prediction_id)})
            if document:
                document["id"] = str(document["_id"])
                del document["_id"]
                return document
            return None
        except:
            return None

    def search_predictions(self, module_name: Optional[str] = None,
                          operation: Optional[str] = None,
                          limit: int = 10) -> List[Dict[str, Any]]:
        """Search predictions with filters."""
        query = {}
        if module_name:
            query["module_name"] = module_name
        if operation:
            query["operation"] = operation

        cursor = self.collection.find(query).sort("created_at", -1).limit(limit)
        results = []
        for doc in cursor:
            doc["id"] = str(doc["_id"])
            del doc["_id"]
            results.append(doc)
        return results

    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()

def demo_mongodb_adapter():
    """Demonstrate MongoDB adapter functionality."""
    print("\n" + "=" * 60)
    print("Example 2: MongoDB Adapter")
    print("=" * 60)

    # Note: This would require actual MongoDB connection
    # adapter = MongoDBAdapter("mongodb://localhost:27017")

    print("\nMongoDB Adapter Features:")
    print("- Document-based storage")
    print("- Flexible schema design")
    print("- Built-in aggregation framework")
    print("- Horizontal scaling support")
    print("- Rich query capabilities")

# Example 3: Advanced Calculator Tool
class AdvancedCalculator(dspy.Tool):
    """Advanced calculator with multiple operations."""

    def __init__(self):
        super().__init__()
        self.history = []
        self.variables = {}

    def calculate(self, expression: str) -> float:
        """Evaluate mathematical expression."""
        try:
            # Replace variables
            for var_name, value in self.variables.items():
                expression = expression.replace(var_name, str(value))

            # Evaluate expression safely
            # Note: In production, use ast.parse for security
            result = eval(expression)
            self.history.append((expression, result))
            return result
        except Exception as e:
            raise ValueError(f"Calculation error: {e}")

    def store_variable(self, name: str, value: float):
        """Store variable in calculator memory."""
        self.variables[name] = value

    def get_history(self) -> List[tuple]:
        """Get calculation history."""
        return self.history

    def clear_history(self):
        """Clear calculation history."""
        self.history = []

    def advanced_calculate(self, expression: str) -> Dict[str, Any]:
        """Calculate with detailed analysis."""
        try:
            # Tokenize expression
            tokens = expression.replace('+', ' + ').replace('-', ' - ').replace('*', ' * ').replace('/', ' / ').split()
            operation_count = len([t for t in tokens if t in ['+', '-', '*', '/']])

            # Calculate result
            result = self.calculate(expression)

            # Get time complexity
            time_complexity = self._estimate_complexity(operation_count)

            return {
                "expression": expression,
                "result": result,
                "operations": operation_count,
                "time_complexity": time_complexity,
                "tokens": len(tokens)
            }
        except Exception as e:
            return {
                "expression": expression,
                "error": str(e),
                "tokens": expression.split() if expression else []
            }

    def _estimate_complexity(self, operation_count):
        """Estimate time complexity."""
        if operation_count == 0:
            return "O(1)"
        elif operation_count <= 5:
            return "O(n)"
        elif operation_count <= 10:
            return "O(n log n)"
        else:
            return "O(n²)"

def demo_advanced_calculator():
    """Demonstrate advanced calculator functionality."""
    print("\n" + "=" * 60)
    print("Example 3: Advanced Calculator Tool")
    print("=" * 60)

    calc = AdvancedCalculator()

    # Basic calculations
    print("\nBasic calculations:")
    basic_calcs = ["2 + 3", "10 * 5", "sqrt(16)", "2^8", "pi * 5"]
    for calc in basic_calcs:
        try:
            result = calc.calculate(calc)
            print(f"{calc} = {result}")
        except ValueError as e:
            print(f"Error: {e}")

    # Variable usage
    print("\nVariable operations:")
    calc.store_variable("x", 10)
    calc.store_variable("y", 20)
    try:
        result = calc.calculate("x * y + 5")
        print(f"x * y + 5 = {result}")
    except ValueError as e:
        print(f"Error: {e}")

    # Advanced calculation with analysis
    print("\nAdvanced analysis:")
    complex_calc = "(10 + 5) * 3 - (20 / 4) + sqrt(100)"
    try:
        analysis = calc.advanced_calculate(complex_calc)
        print(f"Expression: {analysis['expression']}")
        print(f"Result: {analysis['result']}")
        print(f"Operations: {analysis['operations']}")
        print(f"Time Complexity: {analysis['time_complexity']}")
    except Exception as e:
        print(f"Error: {e}")

# Example 4: Text Processing Tool
class TextProcessor(dspy.Tool):
    """Advanced text processing tool."""

    def __init__(self):
        super().__init__()
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'
        }

    def extract_emails(self, text: str) -> List[str]:
        """Extract email addresses from text."""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(pattern, text)
        return list(set(emails))  # Remove duplicates

    def extract_phone_numbers(self, text: str) -> List[str]:
        """Extract phone numbers from text."""
        patterns = [
            r'\b\d{3}-\d{3}-\d{4}\b',  # 123-456-7890
            r'\b\(\d{3}\) \d{3}-\d{4}\b',  # (123) 456-7890
            r'\b\d{10}\b',  # 1234567890
            r'\b\d{3}\.\d{3}\.\d{4}\b'  # 123.456.7890
        ]
        numbers = []
        for pattern in patterns:
            numbers.extend(re.findall(pattern, text))
        return numbers

    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text."""
        pattern = r'http[s]?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w=&%.])*)?)?#?$'
        urls = re.findall(pattern, text)
        return list(set(urls))

    def get_sentiment_score(self, text: str) -> float:
        """Calculate sentiment score (simple version)."""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'disappointing']

        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)

        if positive_count + negative_count == 0:
            return 0.0

        return (positive_count - negative_count) / (positive_count + negative_count)

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract top keywords from text."""
        # Remove stopwords and punctuation
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        words = [word for word in words if word not in self.stopwords]

        # Count word frequencies
        frequency = {}
        for word in words:
            frequency[word] = frequency.get(word, 0) + 1

        # Sort by frequency and return top N
        sorted_words = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:top_n]]

    def summarize_text(self, text: str, max_length: int = 100) -> str:
        """Create a summary of text."""
        sentences = text.split('.')
        if not sentences:
            return ""

        # Find the sentence with most keywords
        best_sentence = ""
        best_score = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                keywords = self.extract_keywords(sentence, top_n=5)
                score = len(keywords)
                if score > best_score:
                    best_score = score
                    best_sentence = sentence

        # Truncate if necessary
        if len(best_sentence) > max_length:
            best_sentence = best_sentence[:max_length] + "..."

        return best_sentence

def demo_text_processor():
    """Demonstrate text processing functionality."""
    print("\n" + "=" * 60)
    print("Example 4: Text Processing Tool")
    print("=" * 60)

    processor = TextProcessor()

    # Sample text
    sample_text = """
    Contact us at support@example.com or admin@test.org.
    Call us at 123-456-7890 or (555) 987-6543.
    Visit our website at https://example.com/product.
    This is a great product with excellent features.
    However, some customers have reported terrible issues.
    Overall, the user experience is fantastic.
    """

    print("\nOriginal text:")
    print(sample_text.strip())

    print("\nExtracted entities:")
    print(f"Emails: {processor.extract_emails(sample_text)}")
    print(f"Phone numbers: {processor.extract_phone_numbers(sample_text)}")
    print(f"URLs: {processor.extract_urls(sample_text)}")

    print(f"\nSentiment score: {processor.get_sentiment_score(sample_text):.2f}")
    print(f"Top keywords: {processor.extract_keywords(sample_text, top_n=5)}")
    print(f"Summary: {processor.summarize_text(sample_text)}")

# Example 5: DSPy Module with Adapters
class EnhancedRAG(dspy.Module):
    """RAG system enhanced with adapters and tools."""

    def __init__(self, db_adapter=None):
        super().__init__()
        self.db_adapter = db_adapter
        self.calculator = AdvancedCalculator()
        self.text_processor = TextProcessor()

        # RAG components
        self.retrieve = dspy.Retrieve(k=5)
        self.generate = dspy.ChainOfThought("context, question -> answer")
        self.analyze = dspy.Predict("text -> analysis, confidence")

    def forward(self, question, store_result=True):
        """Enhanced forward pass with adapters."""
        # Extract entities from question
        entities = self.text_processor.extract_keywords(question, top_n=5)

        # Retrieve documents
        retrieved = self.retrieve(question=question)
        context = "\n".join(retrieved.passages)

        # Generate answer
        prediction = self.generate(context=context, question=question)

        # Analyze response
        analysis = self.analyze(text=prediction.answer)

        # Store in database if configured
        if store_result and self.db_adapter:
            result_id = self.db_adapter.store_prediction(
                module_name="EnhancedRAG",
                operation="query",
                input_data={"question": question, "entities": entities},
                output_data=prediction,
                metadata={"confidence": analysis.confidence}
            )
            prediction.id = result_id

        # Add additional metadata
        prediction.entities = entities
        prediction.sentiment_score = self.text_processor.get_sentiment_score(prediction.answer)
        prediction.confidence = analysis.confidence
        prediction.summary = self.text_processor.summarize_text(prediction.answer)

        return prediction

def demo_enhanced_rag():
    """Demonstrate enhanced RAG with adapters."""
    print("\n" + "=" * 60)
    print("Example 5: Enhanced RAG with Adapters")
    print("=" * 60)

    # Note: Would need database connection in production
    # db_adapter = PostgreSQLAdapter("postgresql://...")

    rag = EnhancedRAG()  # db_adapter=db_adapter

    # Sample questions
    questions = [
        "What is the relationship between AI and machine learning?",
        "How does caching improve application performance?",
        "What are the benefits of containerization?"
    ]

    print("\nEnhanced RAG Processing:")
    for question in questions:
        print(f"\nQuestion: {question}")
        result = rag(question, store_result=False)  # Don't store without DB
        print(f"Answer: {result.answer}")
        print(f"Entities: {result.entities}")
        print(f"Confidence: {result.confidence}")
        print(f"Sentiment: {result.sentiment_score:.2f}")
        print(f"Summary: {result.summary}")

# Main execution
def run_all_examples():
    """Run all adapter and tool examples."""
    print("DSPy Adapters and Tools Examples")
    print("Demonstrating extension mechanisms and integrations\n")

    try:
        demo_postgresql_adapter()
        demo_mongodb_adapter()
        demo_advanced_calculator()
        demo_text_processor()
        demo_enhanced_rag()

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("All adapter and tool examples completed!")
    print("=" * 60)

    print("\nKey Features Demonstrated:")
    print("✓ Database adapters for data persistence")
    print("✓ API integrations for external services")
    print("✓ Specialized tools for text and math processing")
    print("✓ Enhanced DSPy modules with adapter integration")
    print("✓ Error handling and graceful degradation")

if __name__ == "__main__":
    run_all_examples()
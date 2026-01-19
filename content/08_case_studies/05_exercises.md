# Chapter 8 Exercises

These exercises provide hands-on practice implementing real-world DSPy applications based on the case studies presented in this chapter.

## Exercise 1: Building a Mini-RAG System

### Objective
Create a simplified version of the enterprise RAG system for a personal knowledge base.

### Requirements
1. Document ingestion from PDF files
2. Vector storage using ChromaDB
3. Retrieval and answer generation
4. Basic citation support

### Steps

#### Step 1: Setup and Document Processing
```python
# Implement document chunking
def chunk_document(text: str, chunk_size: int = 1000) -> List[str]:
    """
    Split document into overlapping chunks.

    Args:
        text: Document text
        chunk_size: Size of each chunk

    Returns:
        List of text chunks
    """
    # Your code here
    pass

# Test with sample text
sample_text = """
DSPy is a framework for programming language models.
It allows you to write structured programs that leverage the power of LMs.
With DSPy, you can define signatures, modules, and optimizers.
The framework provides tools for RAG, classification, and many other tasks.
"""

chunks = chunk_document(sample_text, chunk_size=100)
print(f"Created {len(chunks)} chunks")
```

#### Step 2: Vector Storage
```python
import chromadb
from sentence_transformers import SentenceTransformer

class SimpleRAG:
    def __init__(self):
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection("documents")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def add_documents(self, chunks: List[str], metadata: List[Dict]):
        """Add document chunks to vector store."""
        # Generate embeddings
        embeddings = self.embedder.encode(chunks).tolist()

        # Add to collection
        # Your code here
        pass

    def query(self, query: str, n_results: int = 3) -> Dict:
        """Query the RAG system."""
        # Generate query embedding
        query_embedding = self.embedder.encode([query]).tolist()

        # Search collection
        # Your code here
        pass
```

#### Step 3: Answer Generation with DSPy
```python
import dspy

class RAGAnswerSignature(dspy.Signature):
    """Generate answer from retrieved context."""
    context = dspy.InputField(desc="Retrieved document chunks")
    question = dspy.InputField(desc="User question")
    answer = dspy.OutputField(desc="Answer based on context")
    sources = dspy.OutputField(desc="Source information")

class RAGAnswerer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(RAGAnswerSignature)

    def forward(self, question: str, retrieved_docs: List[Dict]):
        context = "\n\n".join([doc['content'] for doc in retrieved_docs])

        result = self.generate(
            context=context,
            question=question
        )

        return result

# Test your implementation
rag = SimpleRAG()
rag.add_documents(chunks, [{"source": "sample.txt"} for _ in chunks])

answerer = RAGAnswerer()
retrieved = rag.query("What is DSPy?")
response = answerer.forward("What is DSPy?", retrieved)

print(f"Answer: {response.answer}")
```

### Challenge Extensions
1. Add support for multiple document formats
2. Implement re-ranking for better retrieval
3. Add conversation history support
4. Implement a simple cache for frequent queries

---

## Exercise 2: Customer Support Chatbot Enhancement

### Objective
Enhance the customer support chatbot with additional features.

### Requirements
1. Add sentiment analysis
2. Implement multi-language support
3. Add escalation logic
4. Create a simple web interface

### Steps

#### Step 1: Sentiment Analysis
```python
from textblob import TextBlob

class EnhancedIntentClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        # Your existing classifier code

    def analyze_sentiment(self, message: str) -> Dict:
        """
        Analyze sentiment of the message.

        Returns:
            Dict with sentiment score and category
        """
        # Your code here using TextBlob or DSPy
        pass

    def should_escalate(self, sentiment: Dict, intent: str) -> bool:
        """
        Determine if the conversation should be escalated.

        Args:
            sentiment: Sentiment analysis result
            intent: Classified intent

        Returns:
            True if escalation is needed
        """
        # Your logic here
        pass
```

#### Step 2: Multi-language Support
```python
from langdetect import detect
from deep_translator import GoogleTranslator

class MultiLanguageSupport:
    def __init__(self):
        self.translator = GoogleTranslator(source='auto', target='en')

    def detect_and_translate(self, text: str) -> Dict:
        """
        Detect language and translate to English if needed.

        Returns:
            Dict with original_text, detected_lang, and translated_text
        """
        # Your code here
        pass

    def translate_response(self, response: str, target_lang: str) -> str:
        """Translate response back to user's language."""
        # Your code here
        pass
```

#### Step 3: Web Interface
```python
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
chatbot = CustomerSupportChatbot(config)

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json['message']
    session_id = request.json.get('session_id', 'default')

    # Process message
    response = chatbot.process_message(session_id, message)

    return jsonify(response)

# Create a simple HTML template for the chat interface
# Your code here
```

### Challenge Extensions
1. Add voice input/output support
2. Implement proactive suggestions
3. Add customer authentication
4. Create analytics dashboard

---

## Exercise 3: Code Assistant Features

### Objective
Add new features to the AI code assistant.

### Requirements
1. Code explanation feature
2. Code review suggestions
3. Refactoring recommendations
4. Code similarity detection

### Steps

#### Step 1: Code Explanation
```python
class CodeExplainer(dspy.Module):
    def __init__(self):
        super().__init__()
        # Define signature for code explanation
        class ExplainSignature(dspy.Signature):
            code = dspy.InputField(desc="Code to explain")
            language = dspy.InputField(desc="Programming language")
            explanation = dspy.OutputField(desc="Detailed explanation")

        self.explain = dspy.ChainOfThought(ExplainSignature)

    def explain_code(self, code: str, language: str) -> str:
        """Generate detailed explanation of the code."""
        result = self.explain(code=code, language=language)
        return result.explanation

# Test implementation
explainer = CodeExplainer()
code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

explanation = explainer.explain_code(code, "python")
print(f"Explanation: {explanation}")
```

#### Step 2: Code Review
```python
class CodeReviewer(dspy.Module):
    def __init__(self):
        super().__init__()
        # Define signature for code review
        class ReviewSignature(dspy.Signature):
            code = dspy.InputField(desc="Code to review")
            standards = dspy.InputField(desc="Coding standards to check")
            issues = dspy.OutputField(desc="Identified issues")
            suggestions = dspy.OutputField(desc="Improvement suggestions")

        self.review = dspy.ChainOfThought(ReviewSignature)

    def review_code(self, code: str, standards: str = "PEP8") -> Dict:
        """Review code for issues and suggest improvements."""
        result = self.review(code=code, standards=standards)
        return {
            "issues": result.issues.split('\n'),
            "suggestions": result.suggestions.split('\n')
        }
```

#### Step 3: Refactoring Suggestions
```python
class RefactoringAssistant(dspy.Module):
    def __init__(self):
        super().__init__()
        # Your implementation here

    def suggest_refactoring(self, code: str, focus: str = "performance") -> Dict:
        """
        Suggest refactoring improvements.

        Args:
            code: Code to analyze
            focus: Focus area (performance, readability, maintainability)

        Returns:
            Refactoring suggestions
        """
        # Your code here
        pass
```

### Challenge Extensions
1. Add support for more programming languages
2. Implement code completion
3. Add automated testing suggestions
4. Create IDE plugin

---

## Exercise 4: Data Analysis Dashboard

### Objective
Create a dashboard for the automated data analysis pipeline.

### Requirements
1. Interactive query interface
2. Real-time visualization
3. Alert management
4. Report scheduling

### Steps

#### Step 1: Interactive Query Interface
```python
import streamlit as st

class DataAnalysisDashboard:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.setup_ui()

    def setup_ui(self):
        st.title("Data Analysis Dashboard")

        # Sidebar for query input
        st.sidebar.header("Query Data")
        query = st.sidebar.text_input("Enter your question:")

        # Data source selection
        sources = st.sidebar.multiselect(
            "Select data sources:",
            ["sales", "customers", "products", "inventory"]
        )

        if st.sidebar.button("Analyze"):
            if query and sources:
                with st.spinner("Analyzing..."):
                    # Process query
                    trigger = {
                        "type": "query",
                        "query": query,
                        "sources": sources
                    }
                    results = self.pipeline.run_pipeline(trigger)

                    # Display results
                    self.display_results(results)

    def display_results(self, results):
        """Display analysis results."""
        st.header("Analysis Results")

        # Display insights
        if "insights" in results:
            st.subheader("Key Insights")
            for insight in results["insights"]:
                st.write(f"• {insight}")

        # Display statistics
        if "statistics" in results:
            st.subheader("Statistics")
            st.json(results["statistics"])

        # Display visualizations
        if "visualizations" in results:
            st.subheader("Visualizations")
            # Your code to render visualizations
            pass

# Initialize dashboard
if __name__ == "__main__":
    pipeline = AutomatedDataPipeline(config)
    dashboard = DataAnalysisDashboard(pipeline)
```

#### Step 2: Real-time Monitoring
```python
import plotly.graph_objects as go
from datetime import datetime, timedelta

class RealTimeMonitor:
    def __init__(self):
        self.metrics_history = []

    def update_metrics(self, pipeline):
        """Update pipeline metrics."""
        metrics = {
            "timestamp": datetime.now(),
            "queries_processed": pipeline.metrics.get("queries", 0),
            "avg_response_time": pipeline.metrics.get("avg_time", 0),
            "errors": pipeline.metrics.get("errors", 0)
        }

        self.metrics_history.append(metrics)

        # Keep only last 100 entries
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]

    def create_dashboard(self):
        """Create real-time monitoring dashboard."""
        fig = go.Figure()

        # Add metrics traces
        if self.metrics_history:
            timestamps = [m["timestamp"] for m in self.metrics_history]
            response_times = [m["avg_response_time"] for m in self.metrics_history]

            fig.add_trace(go.Scatter(
                x=timestamps,
                y=response_times,
                name="Response Time",
                mode="lines+markers"
            ))

        fig.update_layout(
            title="Pipeline Performance",
            xaxis_title="Time",
            yaxis_title="Response Time (s)"
        )

        return fig
```

### Challenge Extensions
1. Add user authentication
2. Implement report sharing
3. Add custom alert rules
4. Create mobile app version

---

## Exercise 5: STORM Writing Assistant Implementation

### Objective
Build a simplified version of the STORM writing assistant for generating articles.

### Requirements
1. Multi-perspective research simulation
2. Outline generation from research
3. Section-by-section content generation
4. Basic citation integration

### Steps

#### Step 1: Perspective-Based Research
```python
import dspy

class SimplePerspectiveResearch(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_perspectives = dspy.Predict(
            "topic -> perspectives"
        )
        self.generate_questions = dspy.Predict(
            "topic, perspective -> questions"
        )

    def research_topic(self, topic: str, num_perspectives: int = 3) -> Dict:
        """Simulate multi-perspective research."""
        # Generate perspectives
        perspectives_result = self.generate_perspectives(topic=topic)
        perspectives = perspectives_result.perspectives.split('\n')[:num_perspectives]

        research_data = {}
        for perspective in perspectives:
            # Generate questions for each perspective
            questions_result = self.generate_questions(
                topic=topic,
                perspective=perspective
            )
            questions = questions_result.questions.split('\n')[:3]

            # Simulate research findings
            research_data[perspective] = {
                'questions': questions,
                'findings': self._simulate_findings(perspective, questions)
            }

        return research_data

    def _simulate_findings(self, perspective: str, questions: List[str]) -> List[str]:
        """Simulate research findings for questions."""
        findings = []
        for question in questions:
            # In a real implementation, this would retrieve from sources
            finding = f"From {perspective} perspective: {question} leads to important insights"
            findings.append(finding)
        return findings

# Test the research module
researcher = SimplePerspectiveResearch()
research_data = researcher.research_topic("The Impact of Renewable Energy")
print(f"Researched {len(research_data)} perspectives")
```

#### Step 2: Outline Generation
```python
class SimpleOutlineGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.create_outline = dspy.Predict(
            "topic, research_findings -> outline"
        )

    def generate_outline(self, topic: str, research_data: Dict) -> List[Dict]:
        """Generate article outline from research."""
        # Compile research findings
        all_findings = []
        for perspective, data in research_data.items():
            for finding in data['findings']:
                all_findings.append(f"({perspective}) {finding}")

        findings_text = "\n".join(all_findings)

        # Generate outline
        outline_result = self.create_outline(
            topic=topic,
            research_findings=findings_text
        )

        # Parse outline into structured format
        sections = []
        lines = outline_result.outline.split('\n')
        current_section = None

        for line in lines:
            if line.strip().startswith('I.') or line.strip().startswith('1.'):
                if current_section:
                    sections.append(current_section)
                current_section = {
                    'title': line.strip().split(' ', 1)[1],
                    'subsections': []
                }
            elif line.strip().startswith('   A.') and current_section:
                current_section['subsections'].append(
                    line.strip().split(' ', 1)[1]
                )

        if current_section:
            sections.append(current_section)

        return sections

# Test outline generation
outliner = SimpleOutlineGenerator()
outline = outliner.generate_outline("The Impact of Renewable Energy", research_data)
print(f"Generated outline with {len(outline)} main sections")
```

#### Step 3: Content Generation with Citations
```python
class ContentGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_content = dspy.Predict(
            "section_title, research_data, word_count -> content"
        )
        self.add_citations = dspy.Predict(
            "content, research_data -> cited_content"
        )

    def generate_section(self,
                        section_title: str,
                        research_data: Dict,
                        word_count: int = 300) -> Dict:
        """Generate content for a section with citations."""
        # Convert research data to text
        research_text = ""
        for perspective, data in research_data.items():
            research_text += f"\n{perspective}:\n"
            research_text += "\n".join(data['findings'])

        # Generate content
        content_result = self.generate_content(
            section_title=section_title,
            research_data=research_text,
            word_count=str(word_count)
        )

        # Add citations
        cited_result = self.add_citations(
            content=content_result.content,
            research_data=research_text
        )

        return {
            'title': section_title,
            'content': cited_result.cited_content,
            'word_count': len(cited_result.cited_content.split())
        }

# Test content generation
generator = ContentGenerator()
if outline:
    section = generator.generate_section(
        outline[0]['title'],
        research_data
    )
    print(f"Generated section: {section['title']}")
    print(f"Word count: {section['word_count']}")
```

#### Step 4: Assemble Complete Article
```python
class ArticleAssembler:
    def __init__(self, content_generator: ContentGenerator):
        self.content_generator = content_generator

    def create_article(self,
                      topic: str,
                      outline: List[Dict],
                      research_data: Dict) -> Dict:
        """Assemble complete article from outline and research."""
        article_parts = []

        # Add title
        article_parts.append(f"# {topic}\n")

        # Generate content for each section
        for section in outline:
            # Generate section content
            section_content = self.content_generator.generate_section(
                section['title'],
                research_data,
                word_count=400
            )

            # Add to article
            article_parts.append(f"\n## {section_content['title']}\n")
            article_parts.append(section_content['content'])

            # Generate subsections if any
            for subsection in section.get('subsections', []):
                sub_content = self.content_generator.generate_section(
                    subsection,
                    research_data,
                    word_count=200
                )

                article_parts.append(f"\n### {sub_content['title']}\n")
                article_parts.append(sub_content['content'])

        # Combine all parts
        full_article = '\n'.join(article_parts)

        return {
            'title': topic,
            'content': full_article,
            'sections': len(outline),
            'word_count': len(full_article.split())
        }

# Create complete article
assembler = ArticleAssembler(generator)
article = assembler.create_article(
    "The Impact of Renewable Energy",
    outline,
    research_data
)

print(f"Article generated!")
print(f"Total sections: {article['sections']}")
print(f"Total words: {article['word_count']}")
```

### Challenge Extensions
1. Add quality assessment using FactScore
2. Implement verifiability checking
3. Add human review simulation
4. Create different article formats (blog post, academic paper, etc.)

---

## Exercise 6: Integration Challenge

### Objective
Integrate multiple case studies into a unified platform.

### Requirements
1. Combine RAG and chatbot for Q&A
2. Add code generation to chatbot
3. Include data analysis in support system
4. Create unified monitoring

### Steps

#### Step 1: Unified Platform Architecture
```python
class UnifiedAIPlatform:
    def __init__(self, config):
        # Initialize all components
        self.rag_system = EnterpriseRAGSystem(config["rag"])
        self.chatbot = CustomerSupportChatbot(config["chatbot"])
        self.code_assistant = AICodeAssistant(config["code"])
        self.data_pipeline = AutomatedDataPipeline(config["data"])

        # Create unified routing
        self.router = QueryRouter()

    def process_request(self, request: Dict) -> Dict:
        """
        Route and process requests to appropriate component.

        Args:
            request: Unified request format

        Returns:
            Response from appropriate component
        """
        # Determine request type
        request_type = self.router.classify(request)

        # Route to appropriate component
        if request_type == "qa":
            return self.rag_system.query(**request)
        elif request_type == "chat":
            return self.chatbot.process_message(**request)
        elif request_type == "code":
            return self.code_assistant.process_code_request(**request)
        elif request_type == "data":
            return self.data_pipeline.run_pipeline(request)
        else:
            return {"error": "Unknown request type"}
```

#### Step 2: Intelligent Routing
```python
class QueryRouter(dspy.Module):
    def __init__(self):
        super().__init__()
        # Define routing signature
        class RouteSignature(dspy.Signature):
            query = dspy.InputField(desc="User query or request")
            context = dspy.InputField(desc="Available context")
            route = dspy.OutputField(desc="Target component (qa, chat, code, data)")
            confidence = dspy.OutputField(desc="Routing confidence")

        self.route = dspy.Predict(RouteSignature)

    def classify(self, request: Dict) -> str:
        """Classify request to appropriate component."""
        query = request.get("query", request.get("message", ""))
        context = str(request.get("context", {}))

        result = self.route(query=query, context=context)

        # Use confidence threshold
        if float(result.confidence) > 0.7:
            return result.route
        else:
            return "chat"  # Default to chatbot
```

### Deliverables
1. **Documentation**: Explain your integration approach
2. **Demo**: Show examples of cross-component interactions
3. **Performance Analysis**: Measure integrated system performance
4. **Future Enhancements**: Propose additional features

---

## Project-Based Assignment

### Final Project: Build a Comprehensive AI Assistant

Choose one of the following projects:

#### Option 1: Educational AI Tutor
- Combine RAG for knowledge retrieval
- Add chatbot for student interaction
- Include code examples and explanations
- Generate practice problems and solutions

#### Option 2: Business Intelligence Assistant
- Integrate data analysis pipeline
- Add natural language querying
- Generate automated reports
- Include alerting for anomalies

#### Option 3: Developer Productivity Tool
- Combine code assistant with documentation
- Add project analysis
- Include automated testing suggestions
- Generate project documentation

#### Option 4: Wikipedia-like Article Generator
- Implement STORM-based research and writing
- Add multi-perspective analysis
- Include citation and fact-checking
- Generate articles on complex topics

### Requirements for Final Project:
1. **Complete Implementation**: Working code for all features
2. **Documentation**: User guide and developer documentation
3. **Testing**: Unit tests for key components
4. **Demo**: Video or interactive demo
5. **Reflection**: Lessons learned and improvements

### Evaluation Criteria:
- **Functionality**: 40% - Does it work as expected?
- **Code Quality**: 20% - Is it well-structured and maintainable?
- **Innovation**: 20% - Does it demonstrate creative use of DSPy?
- **Documentation**: 10% - Is it well-documented?
- **Presentation**: 10% - Is the demo clear and professional?

---

## Solutions

### Solution Hints (Not Complete Answers)

#### Exercise 1 - Mini-RAG System
```python
def chunk_document(text: str, chunk_size: int = 1000) -> List[str]:
    """Split document into overlapping chunks."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # Overlap of 200 characters
        start = end - 200

    return chunks

# For add_documents:
self.collection.add(
    embeddings=embeddings,
    documents=chunks,
    metadatas=metadata,
    ids=[f"doc_{i}" for i in range(len(chunks))]
)
```

### Additional Resources
1. [DSPy Documentation](https://dspy-docs.vercel.app/)
2. [Vector Databases Comparison](https://python.langchain.com/docs/modules/data_connection/vectorstores/)
3. [Streamlit Documentation](https://docs.streamlit.io/)
4. [FastAPI Documentation](https://fastapi.tiangolo.com/)
5. [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)

Remember to share your solutions and learn from others in the community!
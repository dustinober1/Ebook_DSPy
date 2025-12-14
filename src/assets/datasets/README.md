# Sample Datasets for DSPy Ebook

This directory contains sample datasets for use in DSPy examples and exercises. All datasets are synthetic and created for educational purposes.

## Dataset Files

### Core Datasets

#### `qa_pairs.json`
Question-answer pairs with context for retrieval-augmented generation and QA tasks.

- **10 QA pairs** covering DSPy concepts
- Fields: question, context, answer
- Use case: Training RAG systems, testing QA modules
- Format: JSON

#### `classification_data.csv`
Text classification dataset with multiple labels and sentiment analysis.

- **26 examples** of product reviews and feedback
- Fields: text, label (positive/negative), sentiment, category
- Categories: product_review, customer_service, logistics, technical_support, usability, pricing, documentation, quality, performance, reliability
- Use case: Text classification tasks, sentiment analysis
- Format: CSV

#### `entity_examples.json`
Named Entity Recognition (NER) examples with annotated entities.

- **8 examples** with entity annotations
- Entity types: PERSON, ORG, LOC, DATE, PRODUCT, DURATION, STRUCTURE, MEASUREMENT, EVENT, DEGREE, FIELD, CONCEPT, PERCENT
- Fields: text, entities (with text, label, start, end positions)
- Use case: Entity extraction tasks, NER training
- Format: JSON

### Domain-Specific Datasets

#### `domain_specific/healthcare_clinical_notes.json`
Clinical notes dataset simulating healthcare documents.

- **5 patient clinical notes** with medical information
- Fields: patient_id, date, note text, conditions, procedures, medications
- Topics: chest pain, respiratory infections, orthopedic injuries, migraines, diabetes management
- Use case: Healthcare information extraction, clinical decision support
- Format: JSON

#### `domain_specific/finance_documents.json`
Financial documents covering multiple financial analysis scenarios.

- **5 financial documents** including earnings reports, loan applications, stock analysis, risk assessments, market analysis
- Fields: type, date, document excerpt, key metrics
- Use case: Financial document analysis, risk assessment, data extraction
- Format: JSON

#### `domain_specific/legal_contracts.json`
Legal contract samples covering common business agreements.

- **5 different contract types**: Service Agreement, NDA, Employment Contract, Lease Agreement, Purchase Agreement
- Fields: type, parties, date, excerpt, key_terms
- Use case: Contract analysis, clause extraction, legal document processing
- Format: JSON

### Research & Business Documents

#### `research_papers/abstracts.json`
Research paper abstracts from major AI/NLP conferences.

- **5 landmark papers** in AI, NLP, and information retrieval
- Fields: title, authors, year, venue, abstract
- Papers: DSPy, RAG, Transformer, Chain-of-Thought, Dense Passage Retrieval
- Use case: Literature review automation, research metadata extraction
- Format: JSON

#### `business_docs/sample_documents.json`
Business communication and planning documents.

- **5 different document types**: Email, Meeting Notes, Proposal, Status Report, Press Release
- Fields: type, date, title, content (specific fields vary by type)
- Use case: Business document classification, information extraction, meeting summarization
- Format: JSON

## Data Structure & Usage

### Using JSON Datasets

```python
import json

# Load JSON dataset
with open('qa_pairs.json') as f:
    data = json.load(f)

for item in data['qa_pairs']:
    print(item['question'])
    print(item['answer'])
```

### Using CSV Datasets

```python
import csv

# Load CSV dataset
with open('classification_data.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"Text: {row['text']}")
        print(f"Label: {row['label']}")
```

### Converting to DSPy Examples

```python
import dspy
import json

# Load data
with open('qa_pairs.json') as f:
    data = json.load(f)

# Convert to DSPy examples
examples = []
for item in data['qa_pairs']:
    example = dspy.Example(
        question=item['question'],
        context=item['context'],
        answer=item['answer']
    ).with_inputs('question', 'context')
    examples.append(example)

# Split into train/test
train_set = examples[:8]
test_set = examples[8:]
```

## Data Statistics

| Dataset | Type | Size | Records | Use Case |
|---------|------|------|---------|----------|
| qa_pairs | JSON | Small | 10 | QA/RAG |
| classification_data | CSV | Small | 26 | Classification |
| entity_examples | JSON | Small | 8 | NER |
| healthcare_notes | JSON | Small | 5 | Medical NLP |
| finance_documents | JSON | Small | 5 | Financial Analysis |
| legal_contracts | JSON | Small | 5 | Legal Analysis |
| research_abstracts | JSON | Small | 5 | Literature Review |
| business_documents | JSON | Small | 5 | Business NLP |

## Important Notes

1. **Synthetic Data**: All datasets are synthetic examples created for educational purposes only
2. **Small Scale**: These are minimal datasets for learning and testing; production systems need larger, curated datasets
3. **License**: These datasets are provided under the same license as the DSPy ebook (CC BY-SA 4.0 for content)
4. **Extension**: Feel free to create similar datasets for your own DSPy projects

## Using These Datasets in Exercises

Each dataset type corresponds to example chapters:

- **qa_pairs.json** → Chapter 6 (RAG Systems)
- **classification_data.csv** → Chapter 6 (Classification Tasks)
- **entity_examples.json** → Chapter 6 (Entity Extraction)
- **healthcare_notes.json** → Chapter 8 (Healthcare Case Study)
- **finance_documents.json** → Chapter 8 (Finance Case Study)
- **legal_contracts.json** → Chapter 8 (Legal Case Study)
- **research_abstracts.json** → Chapter 8 (Research Case Study)
- **business_documents.json** → Chapter 8 (Business Case Study)

## Tips for Working with These Datasets

1. Start small - test with a few examples before scaling up
2. Check data format before processing - validate JSON/CSV structure
3. Handle missing fields gracefully in your code
4. Use these as templates to create domain-specific datasets
5. Consider augmenting with real data for production systems

## Contributing New Datasets

If you want to add new datasets to this collection:

1. Create a new JSON or CSV file following the existing structure
2. Include a clear, descriptive header/schema
3. Add 5-10 representative examples
4. Update this README with the new dataset information
5. Submit a pull request with your contribution

---

**Last Updated**: December 2024
**Version**: 1.0

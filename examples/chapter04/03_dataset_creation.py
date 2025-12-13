"""
Dataset Creation for DSPy Evaluation
====================================

This example demonstrates how to create, manage, and validate
evaluation datasets for DSPy modules.

Requirements:
    - dspy-ai
    - python-dotenv

Usage:
    python 03_dataset_creation.py
"""

import dspy
import json
import csv
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any
from datetime import datetime


# =============================================================================
# 1. Basic Dataset Creation
# =============================================================================

def create_basic_dataset() -> List[dspy.Example]:
    """Create a basic QA dataset manually."""
    examples = [
        # Geography
        dspy.Example(
            question="What is the capital of France?",
            answer="Paris",
            category="geography"
        ).with_inputs("question"),

        dspy.Example(
            question="What is the largest ocean?",
            answer="Pacific Ocean",
            category="geography"
        ).with_inputs("question"),

        # Science
        dspy.Example(
            question="What is the chemical symbol for water?",
            answer="H2O",
            category="science"
        ).with_inputs("question"),

        dspy.Example(
            question="What planet is closest to the sun?",
            answer="Mercury",
            category="science"
        ).with_inputs("question"),

        # History
        dspy.Example(
            question="Who was the first US President?",
            answer="George Washington",
            category="history"
        ).with_inputs("question"),

        dspy.Example(
            question="In what year did World War I begin?",
            answer="1914",
            category="history"
        ).with_inputs("question"),
    ]

    return examples


# =============================================================================
# 2. Dataset from JSON
# =============================================================================

def create_sample_json_data(filepath: Path):
    """Create sample JSON data file."""
    data = [
        {"question": "What is 2+2?", "answer": "4", "difficulty": "easy"},
        {"question": "What is the square root of 144?", "answer": "12", "difficulty": "medium"},
        {"question": "What is the derivative of x^2?", "answer": "2x", "difficulty": "hard"},
        {"question": "What is 10% of 200?", "answer": "20", "difficulty": "easy"},
        {"question": "What is the value of pi to 2 decimal places?", "answer": "3.14", "difficulty": "medium"},
    ]

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    return data


def load_dataset_from_json(filepath: Path) -> List[dspy.Example]:
    """Load dataset from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    examples = [
        dspy.Example(**item).with_inputs("question")
        for item in data
    ]

    return examples


# =============================================================================
# 3. Dataset from CSV
# =============================================================================

def create_sample_csv_data(filepath: Path):
    """Create sample CSV data file."""
    data = [
        {"text": "This product is amazing!", "sentiment": "positive"},
        {"text": "Terrible experience, would not recommend.", "sentiment": "negative"},
        {"text": "It's okay, nothing special.", "sentiment": "neutral"},
        {"text": "Best purchase I've ever made!", "sentiment": "positive"},
        {"text": "Complete waste of money.", "sentiment": "negative"},
        {"text": "Does what it's supposed to do.", "sentiment": "neutral"},
    ]

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["text", "sentiment"])
        writer.writeheader()
        writer.writerows(data)

    return data


def load_dataset_from_csv(filepath: Path) -> List[dspy.Example]:
    """Load dataset from CSV file."""
    examples = []

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            example = dspy.Example(**row).with_inputs("text")
            examples.append(example)

    return examples


# =============================================================================
# 4. Train/Dev/Test Splitting
# =============================================================================

def split_dataset(
    data: List[dspy.Example],
    train_ratio: float = 0.6,
    dev_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
    """
    Split dataset into train/dev/test sets.

    Args:
        data: List of examples
        train_ratio: Fraction for training (default 0.6)
        dev_ratio: Fraction for development (default 0.2)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (trainset, devset, testset)
    """
    # Make a copy to avoid modifying original
    data = list(data)

    # Shuffle with fixed seed
    random.Random(seed).shuffle(data)

    # Calculate split points
    n = len(data)
    train_end = int(n * train_ratio)
    dev_end = int(n * (train_ratio + dev_ratio))

    # Split
    trainset = data[:train_end]
    devset = data[train_end:dev_end]
    testset = data[dev_end:]

    return trainset, devset, testset


def stratified_split(
    data: List[dspy.Example],
    label_field: str,
    train_ratio: float = 0.6,
    dev_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[dspy.Example], List[dspy.Example], List[dspy.Example]]:
    """
    Stratified split maintaining class balance.

    Args:
        data: List of examples
        label_field: Field name containing the label
        train_ratio: Fraction for training
        dev_ratio: Fraction for development
        seed: Random seed

    Returns:
        Tuple of (trainset, devset, testset)
    """
    # Group by label
    by_label = defaultdict(list)
    for example in data:
        label = getattr(example, label_field, 'unknown')
        by_label[label].append(example)

    trainset, devset, testset = [], [], []
    rng = random.Random(seed)

    for label, examples in by_label.items():
        rng.shuffle(examples)
        n = len(examples)

        train_end = int(n * train_ratio)
        dev_end = int(n * (train_ratio + dev_ratio))

        trainset.extend(examples[:train_end])
        devset.extend(examples[train_end:dev_end])
        testset.extend(examples[dev_end:])

    # Shuffle each set
    rng.shuffle(trainset)
    rng.shuffle(devset)
    rng.shuffle(testset)

    return trainset, devset, testset


# =============================================================================
# 5. Dataset Validation
# =============================================================================

def validate_dataset(
    dataset: List[dspy.Example],
    required_fields: List[str],
    name: str = "dataset"
) -> Dict[str, Any]:
    """
    Validate dataset quality.

    Args:
        dataset: List of examples to validate
        required_fields: Fields that must be present
        name: Name for reporting

    Returns:
        Dict with validation results
    """
    results = {
        'name': name,
        'total': len(dataset),
        'valid': 0,
        'issues': []
    }

    for i, example in enumerate(dataset):
        valid = True

        # Check required fields
        for field in required_fields:
            if not hasattr(example, field):
                results['issues'].append(f"Example {i}: Missing field '{field}'")
                valid = False
            elif getattr(example, field) is None:
                results['issues'].append(f"Example {i}: Field '{field}' is None")
                valid = False

        # Check inputs are marked
        if not example.inputs():
            results['issues'].append(f"Example {i}: No inputs marked (use with_inputs())")
            valid = False

        # Check for empty strings
        for field in required_fields:
            value = getattr(example, field, "")
            if isinstance(value, str) and len(value.strip()) == 0:
                results['issues'].append(f"Example {i}: Field '{field}' is empty")
                valid = False

        if valid:
            results['valid'] += 1

    results['valid_ratio'] = results['valid'] / results['total'] if results['total'] > 0 else 0

    return results


def check_dataset_balance(
    dataset: List[dspy.Example],
    label_field: str
) -> Dict[str, int]:
    """
    Check class balance in dataset.

    Returns dict of label -> count.
    """
    labels = [getattr(ex, label_field, 'unknown') for ex in dataset]
    return dict(Counter(labels))


def remove_duplicates(
    dataset: List[dspy.Example],
    key_field: str
) -> List[dspy.Example]:
    """
    Remove duplicate examples based on a field.

    Returns deduplicated list.
    """
    seen = set()
    unique = []

    for example in dataset:
        key = getattr(example, key_field, None)
        if key and key not in seen:
            seen.add(key)
            unique.append(example)

    return unique


# =============================================================================
# 6. Dataset Versioning
# =============================================================================

def save_dataset_versioned(
    dataset: List[dspy.Example],
    filepath: Path,
    version: str,
    description: str = ""
) -> Dict[str, Any]:
    """
    Save dataset with version metadata.

    Args:
        dataset: List of examples
        filepath: Path to save JSON
        version: Version string (e.g., "1.0.0")
        description: Description of changes

    Returns:
        Metadata dict
    """
    # Create metadata
    metadata = {
        'version': version,
        'created': datetime.now().isoformat(),
        'size': len(dataset),
        'description': description,
        'fields': list(dataset[0].toDict().keys()) if dataset else []
    }

    # Convert examples to dicts
    data = [ex.toDict() for ex in dataset]

    # Save
    output = {
        'metadata': metadata,
        'data': data
    }

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved dataset v{version} with {len(dataset)} examples to {filepath}")

    return metadata


def load_dataset_versioned(filepath: Path) -> Tuple[List[dspy.Example], Dict[str, Any]]:
    """
    Load versioned dataset.

    Returns:
        Tuple of (examples, metadata)
    """
    with open(filepath, 'r') as f:
        content = json.load(f)

    metadata = content.get('metadata', {})
    data = content.get('data', [])

    # Determine input fields (assume first field is input)
    input_fields = metadata.get('input_fields', [])
    if not input_fields and data:
        input_fields = [list(data[0].keys())[0]]

    examples = [
        dspy.Example(**item).with_inputs(*input_fields)
        for item in data
    ]

    print(f"Loaded dataset v{metadata.get('version', 'unknown')} with {len(examples)} examples")

    return examples, metadata


# =============================================================================
# Demonstration
# =============================================================================

def demo_dataset_operations():
    """Demonstrate all dataset operations."""
    print("\n" + "="*70)
    print("DATASET CREATION AND MANAGEMENT DEMO")
    print("="*70)

    # 1. Basic dataset creation
    print("\n1. BASIC DATASET CREATION")
    print("-" * 40)
    basic_dataset = create_basic_dataset()
    print(f"Created {len(basic_dataset)} examples")
    print(f"Sample: {basic_dataset[0].question} -> {basic_dataset[0].answer}")

    # 2. JSON dataset
    print("\n2. JSON DATASET")
    print("-" * 40)
    json_path = Path("temp_data/math_qa.json")
    create_sample_json_data(json_path)
    json_dataset = load_dataset_from_json(json_path)
    print(f"Loaded {len(json_dataset)} examples from JSON")

    # 3. CSV dataset
    print("\n3. CSV DATASET")
    print("-" * 40)
    csv_path = Path("temp_data/sentiment.csv")
    create_sample_csv_data(csv_path)
    csv_dataset = load_dataset_from_csv(csv_path)
    print(f"Loaded {len(csv_dataset)} examples from CSV")

    # 4. Dataset splitting
    print("\n4. DATASET SPLITTING")
    print("-" * 40)

    # Create larger dataset for splitting demo
    large_dataset = basic_dataset * 5  # 30 examples

    trainset, devset, testset = split_dataset(large_dataset, seed=42)
    print(f"Random split: Train={len(trainset)}, Dev={len(devset)}, Test={len(testset)}")

    # Stratified split
    stratified_train, stratified_dev, stratified_test = stratified_split(
        large_dataset, 'category', seed=42
    )
    print(f"Stratified split: Train={len(stratified_train)}, Dev={len(stratified_dev)}, Test={len(stratified_test)}")

    # Check balance
    print("\nCategory distribution in train set:")
    balance = check_dataset_balance(stratified_train, 'category')
    for cat, count in balance.items():
        print(f"  {cat}: {count}")

    # 5. Dataset validation
    print("\n5. DATASET VALIDATION")
    print("-" * 40)
    validation_result = validate_dataset(
        basic_dataset,
        required_fields=['question', 'answer'],
        name='basic_qa'
    )
    print(f"Valid examples: {validation_result['valid']}/{validation_result['total']}")
    print(f"Valid ratio: {validation_result['valid_ratio']:.1%}")

    if validation_result['issues']:
        print("Issues found:")
        for issue in validation_result['issues'][:3]:
            print(f"  - {issue}")

    # 6. Deduplication
    print("\n6. DEDUPLICATION")
    print("-" * 40)
    duplicated = basic_dataset + basic_dataset[:2]  # Add duplicates
    print(f"Before dedup: {len(duplicated)} examples")

    deduped = remove_duplicates(duplicated, 'question')
    print(f"After dedup: {len(deduped)} examples")

    # 7. Versioning
    print("\n7. DATASET VERSIONING")
    print("-" * 40)
    versioned_path = Path("temp_data/qa_dataset_v1.json")
    metadata = save_dataset_versioned(
        basic_dataset,
        versioned_path,
        version="1.0.0",
        description="Initial QA dataset with geography, science, and history questions"
    )
    print(f"Metadata: {metadata}")

    # Load it back
    loaded_dataset, loaded_metadata = load_dataset_versioned(versioned_path)
    print(f"Loaded back: {len(loaded_dataset)} examples, version {loaded_metadata['version']}")

    # Cleanup
    print("\n8. CLEANUP")
    print("-" * 40)
    import shutil
    if Path("temp_data").exists():
        shutil.rmtree("temp_data")
        print("Cleaned up temporary files")

    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)


if __name__ == "__main__":
    demo_dataset_operations()

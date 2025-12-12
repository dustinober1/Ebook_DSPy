"""
Title: Descriptive Title for This Example
Chapter: XX - Chapter Name
Topic: Specific DSPy concept being demonstrated

Description:
    More detailed explanation of what this example teaches and demonstrates.
    This should be 2-4 sentences explaining the purpose and learning objectives.

    Use this section to provide context about why this example is useful
    and what real-world scenario it relates to.

Learning Objectives:
    - Specific skill you'll learn from this example
    - Another concept demonstrated here
    - Additional takeaway from running this code

Requirements:
    - dspy-ai>=2.5.0
    - openai>=1.0.0 (or other LM provider)
    - python-dotenv>=1.0.0

Setup:
    1. Install requirements: pip install -r requirements.txt
    2. Set up your API key in .env file:
       OPENAI_API_KEY=your-key-here
    3. Run this script: python XX_example_name.py

Usage:
    python XX_example_name.py

    Optional arguments (if applicable):
    python XX_example_name.py --arg1 value1 --arg2 value2

Author: Dustin Ober
Date: 2025-01-XX
"""

import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

import dspy

# Load environment variables from .env file
load_load_dotenv()


# ============================================================================
# Configuration
# ============================================================================

def configure_lm() -> dspy.LM:
    """
    Configure and return the language model.

    Returns:
        Configured language model instance

    Raises:
        ValueError: If API key is not set
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment. "
            "Please set it in your .env file."
        )

    # Configure the language model
    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        api_key=api_key,
        temperature=0.7,
        max_tokens=500
    )

    return lm


# ============================================================================
# DSPy Components
# ============================================================================

class ExampleSignature(dspy.Signature):
    """
    Brief description of what this signature does.

    This signature takes [input description] and produces [output description].
    """

    # Input fields
    input_field: str = dspy.InputField(
        desc="Description of what this input represents"
    )

    # Output fields
    output_field: str = dspy.OutputField(
        desc="Description of what this output should contain"
    )


class ExampleModule(dspy.Module):
    """
    Brief description of what this module does.

    This module demonstrates [concept being taught].
    """

    def __init__(self):
        """Initialize the module with necessary predictors."""
        super().__init__()

        # Initialize predictor(s)
        self.predictor = dspy.Predict(ExampleSignature)

    def forward(self, input_field: str) -> dspy.Prediction:
        """
        Process the input and generate output.

        Args:
            input_field: Description of the input parameter

        Returns:
            Prediction containing the output_field
        """
        # Make prediction using the signature
        prediction = self.predictor(input_field=input_field)

        return prediction


# ============================================================================
# Helper Functions
# ============================================================================

def validate_input(input_value: str) -> bool:
    """
    Validate the input before processing.

    Args:
        input_value: The input to validate

    Returns:
        True if valid, False otherwise
    """
    # Add validation logic here
    return bool(input_value and len(input_value) > 0)


def format_output(prediction: dspy.Prediction) -> str:
    """
    Format the prediction output for display.

    Args:
        prediction: The prediction object from DSPy

    Returns:
        Formatted string for display
    """
    # Format the output nicely
    output = f"Result: {prediction.output_field}"
    return output


# ============================================================================
# Main Demonstration
# ============================================================================

def run_example():
    """
    Main function demonstrating the example.

    This function shows how to:
    1. Configure the language model
    2. Create and use the module
    3. Process results
    """
    print("=" * 70)
    print("Example: [Descriptive Title]")
    print("=" * 70)
    print()

    # Step 1: Configure the language model
    print("Step 1: Configuring language model...")
    lm = configure_lm()
    dspy.configure(lm=lm)
    print("✓ Language model configured")
    print()

    # Step 2: Create the module
    print("Step 2: Creating DSPy module...")
    module = ExampleModule()
    print("✓ Module created")
    print()

    # Step 3: Prepare example inputs
    print("Step 3: Running examples...")
    examples = [
        "Example input 1",
        "Example input 2",
        "Example input 3",
    ]

    # Step 4: Process each example
    for i, example_input in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"  Input: {example_input}")

        # Validate input
        if not validate_input(example_input):
            print("  ✗ Invalid input, skipping...")
            continue

        # Run the module
        try:
            prediction = module(input_field=example_input)
            output = format_output(prediction)
            print(f"  {output}")
            print("  ✓ Success")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print()
    print("=" * 70)
    print("Example completed!")
    print("=" * 70)


def main():
    """Entry point for the script."""
    try:
        run_example()
    except KeyboardInterrupt:
        print("\n\nExample interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        raise


# ============================================================================
# Sample Output (as comments for reference)
# ============================================================================

"""
Expected Output:
===============

======================================================================
Example: [Descriptive Title]
======================================================================

Step 1: Configuring language model...
✓ Language model configured

Step 2: Creating DSPy module...
✓ Module created

Step 3: Running examples...

Example 1:
  Input: Example input 1
  Result: [Generated output based on input 1]
  ✓ Success

Example 2:
  Input: Example input 2
  Result: [Generated output based on input 2]
  ✓ Success

Example 3:
  Input: Example input 3
  Result: [Generated output based on input 3]
  ✓ Success

======================================================================
Example completed!
======================================================================

Notes:
------
- Key insight from this example
- Important observation about the output
- Lesson learned or best practice demonstrated
"""


# ============================================================================
# Script Execution
# ============================================================================

if __name__ == "__main__":
    main()

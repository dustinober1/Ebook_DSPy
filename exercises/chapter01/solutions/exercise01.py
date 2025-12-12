"""
Exercise 1: Verify DSPy Installation
=====================================
Solution for verifying DSPy installation and basic functionality.
"""

import os
import sys
from dotenv import load_dotenv
import dspy


def main():
    """Verify DSPy installation and configuration."""

    print("DSPy Installation Check")
    print("=======================")

    # Load environment variables
    load_dotenv()

    # Check DSPy version
    try:
        version = dspy.__version__
        print(f"✓ DSPy version: {version}")
    except AttributeError:
        print("⚠ Could not determine DSPy version")
        version = "unknown"

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("✓ API key found")
    else:
        print("✗ No API key found in environment variables")
        print("  Please set OPENAI_API_KEY in your .env file")
        sys.exit(1)

    # Configure language model
    try:
        lm = dspy.OpenAI(model="gpt-4o-mini")
        dspy.configure(lm=lm)
        print("✓ Language model configured")
    except Exception as e:
        print(f"✗ Failed to configure language model: {e}")
        sys.exit(1)

    # Run a test prediction
    try:
        # Define a simple signature
        class SimpleQA(dspy.Signature):
            """Answer basic questions."""
            question = dspy.InputField(desc="A simple question")
            answer = dspy.OutputField(desc="A concise answer")

        # Create and run predictor
        predict = dspy.Predict(SimpleQA)
        result = predict(question="What is 2+2?")

        print("✓ Test prediction successful")
        print(f"\nTest question: What is 2+2?")
        print(f"Test answer: {result.answer}")

    except Exception as e:
        print(f"✗ Test prediction failed: {e}")
        sys.exit(1)

    print("\nAll checks passed!")


if __name__ == "__main__":
    main()
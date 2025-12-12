"""
Exercise 3: Configure Multiple Language Models
===============================================
Solution for testing and comparing different language models in DSPy.
"""

import os
import time
from dotenv import load_dotenv
import dspy

load_dotenv()


def test_model(lm, model_name, question):
    """Test a model and return response and time taken."""

    result = {
        "model": model_name,
        "time": 0,
        "response": "",
        "error": None
    }

    try:
        # Use context manager to temporarily switch models
        with dspy.context(lm=lm):
            # Create a simple Q&A predictor
            class SimpleQA(dspy.Signature):
                """Answer questions briefly."""
                question = dspy.InputField(desc="A question to answer")
                answer = dspy.OutputField(desc="A brief answer to the question")

            predict = dspy.Predict(SimpleQA)

            # Measure response time
            start_time = time.time()
            response = predict(question=question)
            end_time = time.time()

            result["time"] = round(end_time - start_time, 2)
            result["response"] = response.answer

    except Exception as e:
        result["error"] = str(e)
        result["time"] = 0
        result["response"] = f"Error: {e}"

    return result


def check_ollama():
    """Check if Ollama is available and has models."""
    try:
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            if lines and any(line.strip() for line in lines):
                # Return first available model
                first_line = next(line for line in lines if line.strip())
                model_name = first_line.split()[0]
                return model_name
    except:
        pass
    return None


def main():
    """Test and compare multiple language models."""

    print("Testing Multiple Language Models")
    print("=================================\n")

    # Define test question
    question = "Explain quantum computing in simple terms"
    print(f"Question: {question}\n")

    # Configure different models
    models = []

    # Fast, cheap model
    try:
        fast_lm = dspy.OpenAI(model="gpt-4o-mini")
        models.append(("gpt-4o-mini", fast_lm))
    except:
        print("Warning: Could not configure gpt-4o-mini")

    # Powerful model
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            smart_lm = dspy.OpenAI(model="gpt-4o")
            models.append(("gpt-4o", smart_lm))
    except:
        print("Warning: Could not configure gpt-4o")

    # Alternative provider (Anthropic Claude)
    try:
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            claude_lm = dspy.Anthropic(model="claude-3-haiku-20240307")
            models.append(("claude-3-haiku", claude_lm))
    except:
        print("Warning: Could not configure Claude")

    # Local model (Ollama) - if available
    ollama_model = check_ollama()
    if ollama_model:
        try:
            local_lm = dspy.Ollama(model=ollama_model)
            models.append((f"ollama/{ollama_model}", local_lm))
        except:
            print(f"Warning: Could not configure Ollama model {ollama_model}")

    # Test each model
    results = []
    for model_name, lm in models:
        print(f"Testing {model_name}...")
        result = test_model(lm, model_name, question)
        results.append(result)

    # Display results
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50 + "\n")

    for result in results:
        print(f"Model: {result['model']}")
        print(f"Time: {result['time']}s")
        if result['error']:
            print(f"Error: {result['error']}")
        else:
            # Truncate very long responses
            response = result['response']
            if len(response) > 200:
                response = response[:200] + "..."
            print(f"Response: {response}")
        print("-" * 50)

    # Create comparison table
    print("\nComparison Table:")
    print("-" * 60)
    print(f"{'Model':<20} {'Time (s)':<10} {'Response Length':<15} {'Status'}")
    print("-" * 60)

    for result in results:
        status = "✓ Success" if not result['error'] else "✗ Failed"
        response_len = len(result['response']) if not result['error'] else "N/A"
        print(f"{result['model']:<20} {result['time']:<10} {response_len:<15} {status}")

    # Find fastest and most detailed
    successful_results = [r for r in results if not r['error']]
    if successful_results:
        fastest = min(successful_results, key=lambda x: x['time'])
        most_detailed = max(successful_results, key=lambda x: len(x['response']))

        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"Fastest: {fastest['model']} ({fastest['time']}s)")
        print(f"Most detailed: {most_detailed['model']} ({len(most_detailed['response'])} chars)")

    print("\nNote: Response times include network latency and processing time.")
    print("For accurate benchmarking, run multiple iterations and average the results.")


if __name__ == "__main__":
    main()
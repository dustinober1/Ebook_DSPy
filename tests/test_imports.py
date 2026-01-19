"""Test that all DSPy imports work correctly."""

def test_dspy_import():
    import dspy
    assert dspy is not None

def test_dspy_predict():
    from dspy import Predict
    assert Predict is not None

def test_dspy_chainofthought():
    from dspy import ChainOfThought
    assert ChainOfThought is not None

def test_dspy_module():
    from dspy import Module
    assert Module is not None

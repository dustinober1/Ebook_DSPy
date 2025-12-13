# Contributing to DSPy Ebook

First off, thank you for considering contributing to this DSPy ebook! It's people like you that make this such a great resource.

## How Can I Contribute?

### Reporting Bugs

This section guides you through submitting a bug report. Following these guidelines helps us understand your report, reproduce the behavior, and find related reports.

**Before submitting a bug report:**
- Check the [issues list](https://github.com/dustinober1/Ebook_DSPy/issues) - your issue might already exist
- Check the [Troubleshooting Guide](src/09-appendices/02-troubleshooting.md) - it may already cover the problem

**When submitting a bug report, include:**
- Clear, descriptive title
- Exact steps to reproduce
- Specific examples to demonstrate the steps
- Description of the observed behavior
- Explanation of the expected behavior
- Screenshots if applicable
- Your environment (Python version, OS, DSPy version)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- Use a **clear, descriptive title**
- Provide a **detailed description** of the suggested enhancement
- Provide **specific examples** to demonstrate the suggestion
- Explain **why this enhancement would be useful**
- List some **alternative approaches** you've considered

### Pull Requests

- Fill in the required template
- Follow the Python/Markdown style guides
- Document new code with docstrings
- Add tests for new functionality
- Keep commits atomic and well-described

## Development Setup

### Prerequisites

- Python 3.9+
- Git
- mdBook (for building documentation)

### Setting Up Your Development Environment

1. **Fork the repository**
   ```bash
   # Go to https://github.com/dustinober1/Ebook_DSPy and click "Fork"
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR-USERNAME/Ebook_DSPy.git
   cd Ebook_DSPy
   ```

3. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Set up virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

5. **Install build tools**
   ```bash
   brew install mdbook  # Or: cargo install mdbook
   ```

## Code Style Guidelines

### Python Code

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use meaningful variable names
- Include docstrings for all functions and classes
- Maximum line length: 100 characters
- Use type hints where appropriate

Example:
```python
def process_data(input_file: str) -> Dict[str, Any]:
    """
    Process input data from file.

    Args:
        input_file: Path to input file

    Returns:
        Dictionary containing processed data
    """
    # implementation
```

### Markdown Content

- Use clear, concise language
- Break content into logical sections with headers
- Use lists and tables for organizing information
- Include code examples where helpful
- Link to related sections using relative paths
- Use consistent formatting:
  - Headers: `#` for H1, `##` for H2, etc.
  - Bold: `**text**`
  - Italics: `*text*`
  - Code: `` `code` `` for inline, triple backticks for blocks

### Commit Messages

Follow conventional commit format:
```
type(scope): subject

body

footer
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(chapter02): Add advanced signature examples

Added three new examples demonstrating:
- Multi-field signatures
- Complex type hints
- Signature composition

Fixes #123
```

## Content Guidelines

### Writing Style

- Use active voice and present tense
- Be concise and avoid jargon where possible
- Explain acronyms on first use
- Use inclusive language
- Link to related resources

### Code Examples

All code examples must:
- Have valid Python 3.9+ syntax
- Be properly formatted and commented
- Include expected output in docstrings
- Be self-contained and runnable
- Use meaningful variable names
- Demonstrate best practices

### Exercises

Exercise problems should:
- Start with a clear objective
- Include realistic constraints
- Provide hints if the problem is difficult
- Have complete, well-commented solutions
- Include expected output examples

## Testing

### Validate Code Examples

```bash
# Test syntax of all Python examples
python3 << 'EOF'
import os
import py_compile

for root, dirs, files in os.walk("examples"):
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            py_compile.compile(filepath, doraise=True)
            print(f"✓ {filepath}")
EOF
```

### Validate Links

Check that all internal links are correct:
```bash
# Verify all .md files exist
grep -r '\[' src/ | grep '](.*\.md)' | awk -F']' '{print $2}' | sort -u
```

### Build Documentation

```bash
mdbook build
# Check output in build/html/
```

## Review Process

1. **Code Review**
   - Maintainers will review your PR
   - Changes may be requested
   - Tests must pass
   - Follow our style guides

2. **Quality Standards**
   - All code must pass syntax validation
   - Documentation must be clear and complete
   - Examples must be tested
   - Commit history should be clean

3. **Merge**
   - PR will be merged after approval
   - You'll be credited as a contributor
   - Your contribution will appear in the next release

## Contributor License Agreement

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (MIT for code, CC BY-SA 4.0 for content).

## Community

- **Discord**: [Stanford NLP Discord](https://discord.gg/stanfordnlp)
- **Issues**: [GitHub Issues](https://github.com/dustinober1/Ebook_DSPy/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dustinober1/Ebook_DSPy/discussions)

## Additional Notes

### Project Structure

- `src/` - Book content (Markdown)
- `examples/` - Code examples organized by chapter
- `exercises/` - Practice problems and solutions
- `assets/` - Images, datasets, templates
- `book.toml` - mdBook configuration

### Useful Resources

- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [mdBook Documentation](https://rust-lang.github.io/mdBook/)
- [Markdown Guide](https://www.markdownguide.org/)
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)

## Recognition

All contributors will be recognized in:
- The `CONTRIBUTORS.md` file
- The GitHub contributors page
- Release notes

Thank you for helping improve this resource! 🎉

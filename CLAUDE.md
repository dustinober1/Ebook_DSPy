# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an mdBook-based ebook project that teaches DSPy (a framework for programming LM-based applications). The project is structured as a comprehensive learning guide with progressive chapters, code examples, exercises, and real-world case studies.

## Build Commands

### Core Development Commands
```bash
# Build the ebook (validates code examples first)
./scripts/build.sh

# Build HTML version only
./scripts/build.sh --html-only

# Skip code validation (faster but less safe)
./scripts/build.sh --skip-validation

# Local development server with live reload
./scripts/serve.sh
# Default: http://localhost:3000

# Custom port for development server
./scripts/serve.sh --port 8080

# Direct mdBook commands
mdbook build              # Build to build/html/
mdbook serve              # Dev server at http://localhost:3000
```

### Code Validation
```bash
# Validate all Python code examples
python scripts/validate_code.py

# Validate with import checking
python scripts/validate_code.py --check-imports

# Verbose validation output
python scripts/validate_code.py --verbose
```

## Project Architecture

### Content Organization
- **src/**: Book content organized by chapters (00-09 prefix for ordering)
- **examples/**: Python code examples organized by chapter
- **exercises/**: Practice problems with solutions
- **assets/**: Supporting materials (images, datasets, templates)
- **scripts/**: Build and utility scripts

### Chapter Structure Pattern
```
src/XX-chapter-name/
├── 00-chapter-intro.md      # Chapter overview
├── 01-topic-name.md         # Main content files
├── 02-another-topic.md
└── 06-exercises.md          # Chapter exercises (standard naming)
```

### Code Example Organization
```
examples/chapterXX/
├── 01_descriptive_name.py
├── 02_another_example.py
└── README.md                # Optional: Example descriptions
```

### Key Configuration Files
- **book.toml**: mdBook configuration (HTML/PDF/EPUB outputs)
- **SUMMARY.md**: Complete table of contents defining book structure
- **requirements.txt**: Python dependencies for code examples

## Development Workflow

### When Adding New Content
1. Follow naming convention: `XX-descriptive-name.md` (XX = chapter number)
2. Use templates in `assets/templates/` for consistency
3. Add corresponding code examples in `examples/chapterXX/`
4. Update SUMMARY.md to include new content
5. Validate code examples: `python scripts/validate_code.py`
6. Build to verify: `./scripts/build.sh`

### Content Templates Available
- `assets/templates/chapter_template.md`: Standard chapter structure
- `assets/templates/exercise_template.md`: Exercise format
- `assets/templates/example_template.py`: Code example format

### Progressive Learning Structure
- **Chapters 1**: Fundamentals (beginner)
- **Chapters 2-3**: Core Concepts (intermediate)
- **Chapters 4-5**: Evaluation & Optimization (intermediate-advanced)
- **Chapters 6-7**: Real-world applications (advanced)
- **Chapter 8**: Case studies (expert)
- **Chapter 9**: Appendices (reference)

## Important Constraints

### File Naming Conventions
- Content files: `XX-descriptive-name.md` (01-09 for chapters, 00+ for sections)
- Examples: `XX_descriptive_name.py` (matching chapter numbers)
- Always use two-digit numbers for proper ordering

### Code Example Requirements
- All examples must be syntactically valid Python
- Use imports compatible with requirements.txt
- Examples should be runnable independently
- Follow PEP 8 standards
- Include docstrings explaining the example

### Build System Integration
- The build script automatically validates all code examples
- Failed validation prevents the build (unless --skip-validation)
- Multiple output formats: HTML (default), PDF (optional), EPUB (optional)

## Dependencies and Setup

### Required Tools
- **mdBook**: `cargo install mdbook`
- **Python 3.9+**: For code examples
- **Optional**: mdbook-pdf, mdbook-epub for additional formats

### Python Dependencies
- Core: `dspy-ai>=2.5.0`
- LLM providers: `openai>=1.0.0`, `anthropic>=0.8.0`
- Development: `python-dotenv`, `pytest`

## Testing and Quality

### Validation Script Features
- Syntax checking for all Python files
- Import resolution verification
- PEP 8 compliance checking
- Colored terminal output for clarity

### Common Issues to Check
- Broken internal links (mdBook will report)
- Missing code examples referenced in content
- Incorrect file numbering in SUMMARY.md
- Python syntax errors in examples
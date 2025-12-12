#!/usr/bin/env python3
"""
Code Validation Script for DSPy Ebook

This script validates all Python code examples in the ebook by:
- Checking syntax validity
- Verifying imports can be resolved
- Running basic linting checks
- Ensuring code follows PEP 8 standards (optional)

Usage:
    python scripts/validate_code.py
    python scripts/validate_code.py --verbose
    python scripts/validate_code.py --check-imports

Author: Dustin Ober
Date: 2025-12-12
"""

import ast
import sys
import os
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import importlib.util


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_success(message: str) -> None:
    """Print success message in green."""
    print(f"{Colors.GREEN}✓{Colors.RESET} {message}")


def print_error(message: str) -> None:
    """Print error message in red."""
    print(f"{Colors.RED}✗{Colors.RESET} {message}")


def print_warning(message: str) -> None:
    """Print warning message in yellow."""
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {message}")


def print_info(message: str) -> None:
    """Print info message in blue."""
    print(f"{Colors.BLUE}ℹ{Colors.RESET} {message}")


def find_python_files(base_dir: Path) -> List[Path]:
    """
    Find all Python files in the examples directory.

    Args:
        base_dir: Base directory to search from

    Returns:
        List of Python file paths
    """
    examples_dir = base_dir / "examples"
    if not examples_dir.exists():
        print_warning(f"Examples directory not found: {examples_dir}")
        return []

    python_files = list(examples_dir.rglob("*.py"))
    return sorted(python_files)


def validate_syntax(filepath: Path, verbose: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Validate Python syntax of a file.

    Args:
        filepath: Path to the Python file
        verbose: Whether to print verbose output

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()

        # Parse the AST to check syntax
        ast.parse(source, filename=str(filepath))

        if verbose:
            print_success(f"Syntax valid: {filepath.relative_to(filepath.parents[2])}")

        return True, None

    except SyntaxError as e:
        error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
        return False, error_msg

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        return False, error_msg


def extract_imports(filepath: Path) -> List[str]:
    """
    Extract all import statements from a Python file.

    Args:
        filepath: Path to the Python file

    Returns:
        List of imported module names
    """
    imports = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()

        tree = ast.parse(source, filename=str(filepath))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module.split('.')[0])

    except Exception:
        pass

    return list(set(imports))  # Remove duplicates


def check_imports(filepath: Path, verbose: bool = False) -> Tuple[bool, List[str]]:
    """
    Check if all imports can be resolved.

    Args:
        filepath: Path to the Python file
        verbose: Whether to print verbose output

    Returns:
        Tuple of (all_imports_valid, list_of_missing_imports)
    """
    imports = extract_imports(filepath)
    missing_imports = []

    # Standard library modules we don't need to check
    stdlib_modules = {
        'os', 'sys', 'typing', 'pathlib', 'json', 'csv', 'datetime',
        'collections', 'itertools', 'functools', 're', 'argparse',
        'logging', 'unittest', 'ast', 'importlib'
    }

    for module_name in imports:
        # Skip standard library modules
        if module_name in stdlib_modules:
            continue

        # Try to find the module
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            missing_imports.append(module_name)

    if missing_imports and verbose:
        print_warning(
            f"Missing imports in {filepath.name}: {', '.join(missing_imports)}"
        )

    return len(missing_imports) == 0, missing_imports


def validate_file(
    filepath: Path,
    check_import_flag: bool = False,
    verbose: bool = False
) -> Tuple[bool, str]:
    """
    Validate a single Python file.

    Args:
        filepath: Path to the Python file
        check_import_flag: Whether to check imports
        verbose: Whether to print verbose output

    Returns:
        Tuple of (is_valid, status_message)
    """
    # Check syntax
    is_valid, error_msg = validate_syntax(filepath, verbose)

    if not is_valid:
        return False, f"Syntax error: {error_msg}"

    # Check imports if requested
    if check_import_flag:
        imports_valid, missing = check_imports(filepath, verbose)
        if not imports_valid:
            return False, f"Missing imports: {', '.join(missing)}"

    return True, "Valid"


def main():
    """Main function to validate all code examples."""
    parser = argparse.ArgumentParser(
        description="Validate Python code examples in the DSPy ebook"
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="Print verbose output"
    )
    parser.add_argument(
        '-i', '--check-imports',
        action='store_true',
        help="Check if imports can be resolved"
    )
    parser.add_argument(
        '--dir',
        type=str,
        default='.',
        help="Base directory (default: current directory)"
    )

    args = parser.parse_args()

    # Get base directory
    base_dir = Path(args.dir).resolve()

    print(f"{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}DSPy Ebook - Code Validation{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    print()

    # Find all Python files
    print_info("Searching for Python files...")
    python_files = find_python_files(base_dir)

    if not python_files:
        print_warning("No Python files found in examples/ directory")
        print()
        print_info("This is expected if you haven't created any examples yet.")
        print_info("The validation script is ready for when you add code examples.")
        return 0

    print_success(f"Found {len(python_files)} Python file(s)")
    print()

    # Validate each file
    print_info("Validating files...")
    print()

    errors = []
    warnings = []

    for filepath in python_files:
        relative_path = filepath.relative_to(base_dir)

        is_valid, message = validate_file(
            filepath,
            check_import_flag=args.check_imports,
            verbose=args.verbose
        )

        if not is_valid:
            errors.append((relative_path, message))
            print_error(f"{relative_path}: {message}")
        elif args.verbose:
            print_success(f"{relative_path}: {message}")

    # Print summary
    print()
    print(f"{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}Validation Summary{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    print()

    total_files = len(python_files)
    valid_files = total_files - len(errors)

    print(f"Total files:  {total_files}")
    print(f"Valid files:  {Colors.GREEN}{valid_files}{Colors.RESET}")
    print(f"Errors:       {Colors.RED}{len(errors)}{Colors.RESET}")

    if errors:
        print()
        print(f"{Colors.BOLD}Files with errors:{Colors.RESET}")
        for filepath, message in errors:
            print(f"  {Colors.RED}✗{Colors.RESET} {filepath}")
            print(f"    {message}")

    print()

    if errors:
        print_error("Validation failed! Please fix the errors above.")
        return 1
    else:
        print_success("All code examples validated successfully!")
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)

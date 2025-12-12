#!/bin/bash

# ==============================================================================
# DSPy Ebook Build Script
# ==============================================================================
#
# This script builds the DSPy ebook in multiple formats:
# - HTML (via mdBook)
# - PDF (via mdbook-pdf, if installed)
# - EPUB (via mdbook-epub, if installed)
#
# Before building, it validates all Python code examples to ensure they are
# syntactically correct.
#
# Usage:
#   ./scripts/build.sh              # Build all formats
#   ./scripts/build.sh --skip-validation  # Skip code validation
#   ./scripts/build.sh --html-only  # Build HTML only
#
# Author: Dustin Ober
# Date: 2025-12-12
# ==============================================================================

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Parse command line arguments
SKIP_VALIDATION=false
HTML_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --html-only)
            HTML_ONLY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-validation   Skip Python code validation"
            echo "  --html-only         Build HTML version only"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# ==============================================================================
# Helper Functions
# ==============================================================================

print_header() {
    echo -e "\n${BOLD}======================================================================${NC}"
    echo -e "${BOLD}$1${NC}"
    echo -e "${BOLD}======================================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# ==============================================================================
# Pre-build Checks
# ==============================================================================

print_header "DSPy Ebook - Build Script"

# Check if mdBook is installed
if ! check_command mdbook; then
    print_error "mdBook is not installed"
    echo ""
    print_info "To install mdBook, run:"
    echo "  cargo install mdbook"
    echo ""
    print_info "Or visit: https://rust-lang.github.io/mdBook/guide/installation.html"
    exit 1
fi

print_success "mdBook is installed ($(mdbook --version))"

# Check for optional tools
if check_command mdbook-pdf; then
    print_success "mdbook-pdf is installed"
    HAS_PDF=true
else
    print_warning "mdbook-pdf is not installed (PDF generation will be skipped)"
    print_info "To install: cargo install mdbook-pdf"
    HAS_PDF=false
fi

if check_command mdbook-epub; then
    print_success "mdbook-epub is installed"
    HAS_EPUB=true
else
    print_warning "mdbook-epub is not installed (EPUB generation will be skipped)"
    print_info "To install: cargo install mdbook-epub"
    HAS_EPUB=false
fi

# ==============================================================================
# Code Validation
# ==============================================================================

if [ "$SKIP_VALIDATION" = false ]; then
    print_header "Step 1: Validating Python Code Examples"

    if [ -f "scripts/validate_code.py" ]; then
        if python3 scripts/validate_code.py; then
            print_success "All code examples validated successfully"
        else
            print_error "Code validation failed"
            echo ""
            print_info "Fix the errors above or use --skip-validation to build anyway"
            exit 1
        fi
    else
        print_warning "Validation script not found, skipping validation"
    fi
else
    print_warning "Skipping code validation (--skip-validation flag used)"
fi

# ==============================================================================
# Build HTML Version
# ==============================================================================

print_header "Step 2: Building HTML Version"

if mdbook build; then
    print_success "HTML version built successfully"
    print_info "Output: build/html/"
else
    print_error "HTML build failed"
    exit 1
fi

# Exit here if HTML-only build requested
if [ "$HTML_ONLY" = true ]; then
    print_header "Build Complete"
    print_success "HTML version is ready in build/html/"
    exit 0
fi

# ==============================================================================
# Build PDF Version (if available)
# ==============================================================================

if [ "$HAS_PDF" = true ]; then
    print_header "Step 3: Building PDF Version"

    # Check if PDF output is enabled in book.toml
    if grep -q '^\[output\.pdf\]' book.toml; then
        if mdbook-pdf build; then
            print_success "PDF version built successfully"
            print_info "Output: build/pdf/"
        else
            print_warning "PDF build failed (continuing anyway)"
        fi
    else
        print_info "PDF output not configured in book.toml (skipping)"
        print_info "To enable PDF, uncomment [output.pdf] section in book.toml"
    fi
else
    print_info "Skipping PDF generation (mdbook-pdf not installed)"
fi

# ==============================================================================
# Build EPUB Version (if available)
# ==============================================================================

if [ "$HAS_EPUB" = true ]; then
    print_header "Step 4: Building EPUB Version"

    # Check if EPUB output is enabled in book.toml
    if grep -q '^\[output\.epub\]' book.toml; then
        if mdbook-epub build; then
            print_success "EPUB version built successfully"
            print_info "Output: build/epub/"
        else
            print_warning "EPUB build failed (continuing anyway)"
        fi
    else
        print_info "EPUB output not configured in book.toml (skipping)"
        print_info "To enable EPUB, uncomment [output.epub] section in book.toml"
    fi
else
    print_info "Skipping EPUB generation (mdbook-epub not installed)"
fi

# ==============================================================================
# Build Summary
# ==============================================================================

print_header "Build Complete"

echo "Build artifacts:"
echo ""
if [ -d "build/html" ]; then
    print_success "HTML:  build/html/index.html"
fi
if [ -d "build/pdf" ] && [ -n "$(ls -A build/pdf 2>/dev/null)" ]; then
    print_success "PDF:   build/pdf/"
fi
if [ -d "build/epub" ] && [ -n "$(ls -A build/epub 2>/dev/null)" ]; then
    print_success "EPUB:  build/epub/"
fi

echo ""
print_info "To view the HTML version locally, run: ./scripts/serve.sh"
echo ""

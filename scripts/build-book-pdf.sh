#!/bin/bash

# ==============================================================================
# DSPy Ebook - Build Book-Only PDF Script
# ==============================================================================
#
# This script creates a clean PDF version of the ebook with just the
# book content, no navigation or web elements.
#
# Usage:
#   ./scripts/build-book-pdf.sh
#
# Author: Dustin Ober
# Date: 2025-12-13
# ==============================================================================

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

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

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_header "Building Book-Only PDF"

# Check dependencies
if ! command -v mdbook &> /dev/null; then
    print_error "mdBook is not installed"
    exit 1
fi

if ! command -v cargo &> /dev/null; then
    print_error "Cargo is not installed"
    exit 1
fi

# Check if mdbook-pdf is installed
if ! cargo list --installed | grep -q mdbook-pdf; then
    print_info "Installing mdbook-pdf..."
    cargo install mdbook-pdf
    print_success "mdbook-pdf installed"
fi

# Backup original book.toml
if [ -f "book.toml" ]; then
    print_info "Backing up original book.toml..."
    cp book.toml book.toml.backup
fi

# Use print configuration
print_info "Using print-only configuration..."
cp book-print.toml book.toml

# Clean previous builds
print_info "Cleaning previous builds..."
rm -rf build/print/

# Build the print version
print_info "Building book with print configuration..."
if mdbook build; then
    print_success "Build completed successfully"
else
    print_error "Build failed"
    # Restore original book.toml
    if [ -f "book.toml.backup" ]; then
        mv book.toml.backup book.toml
    fi
    exit 1
fi

# Find the generated PDF
PDF_PATH="build/print/pdf/output.pdf"
BOOK_PDF="DSPy_A_Practical_Guide_Book.pdf"

if [ -f "$PDF_PATH" ]; then
    # Copy to project root
    cp "$PDF_PATH" "$BOOK_PDF"

    # Get file size
    SIZE=$(du -h "$BOOK_PDF" | cut -f1)

    print_success "Book PDF created successfully!"
    print_info "File: $BOOK_PDF"
    print_info "Size: $SIZE"

    # Open the PDF for review
    if command -v open &> /dev/null; then
        print_info "Opening PDF for review..."
        open "$BOOK_PDF"
    fi
else
    print_error "PDF not found at expected location: $PDF_PATH"

    # List what was generated
    echo ""
    print_info "Files generated in build/print/:"
    find build/print -type f -name "*.pdf" 2>/dev/null || echo "No PDF files found"
fi

# Restore original book.toml
print_info "Restoring original configuration..."
if [ -f "book.toml.backup" ]; then
    mv book.toml.backup book.toml
fi

print_header "Build Complete"

print_info "Summary:"
echo "  - Clean book PDF: $BOOK_PDF"
echo "  - Navigation elements: Removed"
echo "  - Web elements: Hidden"
echo "  - Print optimized: Yes"
echo ""
print_info "The PDF is ready for review!"
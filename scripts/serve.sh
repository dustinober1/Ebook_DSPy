#!/bin/bash

# ==============================================================================
# DSPy Ebook Development Server
# ==============================================================================
#
# This script starts a local development server for the DSPy ebook.
# The server automatically rebuilds the book when files change and provides
# live reload in the browser.
#
# Usage:
#   ./scripts/serve.sh              # Start server (default: localhost:3000)
#   ./scripts/serve.sh --port 8080  # Start on custom port
#   ./scripts/serve.sh --open       # Start and open in browser
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

# Default values
PORT=3000
OPEN_BROWSER=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -o|--open)
            OPEN_BROWSER=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Start a local development server for the DSPy ebook"
            echo ""
            echo "Options:"
            echo "  -p, --port PORT     Port to serve on (default: 3000)"
            echo "  -o, --open          Open browser automatically"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "The server will watch for file changes and automatically rebuild."
            echo "Press Ctrl+C to stop the server."
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ==============================================================================
# Helper Functions
# ==============================================================================

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_header() {
    echo -e "\n${BOLD}======================================================================${NC}"
    echo -e "${BOLD}$1${NC}"
    echo -e "${BOLD}======================================================================${NC}\n"
}

# ==============================================================================
# Pre-flight Checks
# ==============================================================================

print_header "DSPy Ebook - Development Server"

# Check if mdBook is installed
if ! command -v mdbook &> /dev/null; then
    print_error "mdBook is not installed"
    echo ""
    print_info "To install mdBook, run:"
    echo "  cargo install mdbook"
    echo ""
    print_info "Or visit: https://rust-lang.github.io/mdBook/guide/installation.html"
    exit 1
fi

print_success "mdBook is installed ($(mdbook --version))"

# Check if SUMMARY.md exists
if [ ! -f "SUMMARY.md" ]; then
    print_error "SUMMARY.md not found"
    print_info "Make sure you're running this script from the ebook root directory"
    exit 1
fi

print_success "Found SUMMARY.md"

# Check if book.toml exists
if [ ! -f "book.toml" ]; then
    print_error "book.toml not found"
    print_info "Make sure you're running this script from the ebook root directory"
    exit 1
fi

print_success "Found book.toml"

# ==============================================================================
# Start Development Server
# ==============================================================================

print_header "Starting Development Server"

print_info "Server will be available at: http://localhost:${PORT}"
print_info "Watching for file changes..."
print_info "Press Ctrl+C to stop the server"
echo ""

# Build the serve command
SERVE_CMD="mdbook serve --port ${PORT}"

if [ "$OPEN_BROWSER" = true ]; then
    SERVE_CMD="${SERVE_CMD} --open"
    print_info "Browser will open automatically"
    echo ""
fi

# Start the server
echo -e "${BOLD}Starting mdBook server...${NC}"
echo ""

# Trap Ctrl+C to provide clean exit
trap 'echo -e "\n\n${YELLOW}Server stopped by user${NC}\n"; exit 0' INT

# Run mdbook serve
$SERVE_CMD

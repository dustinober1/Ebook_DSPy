#!/bin/bash

# ==============================================================================
# DSPy Ebook Deployment Script
# ==============================================================================
#
# This script helps deploy the DSPy ebook to various environments.
# Supports local testing, staging, and production deployment.
#
# Usage:
#   ./scripts/deploy.sh                 # Build and test locally
#   ./scripts/deploy.sh --staging     # Deploy to staging
#   ./scripts/deploy.sh --production  # Deploy to production
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

# Parse command line arguments
ENVIRONMENT="local"
SKIP_BUILD=false
DEPLOY_SUBDIR=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --staging)
            ENVIRONMENT="staging"
            shift
            ;;
        --production)
            ENVIRONMENT="production"
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --subdir)
            DEPLOY_SUBDIR=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --staging      Deploy to staging environment"
            echo "  --production   Deploy to production environment"
            echo "  --skip-build   Skip build step, use existing files"
            echo "  --subdir       Deploy to subdirectory"
            echo "  -h, --help     Show this help message"
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
# Main Deployment Logic
# ==============================================================================

print_header "DSPy Ebook Deployment Script"
print_info "Environment: $ENVIRONMENT"

# Check dependencies
print_info "Checking dependencies..."

if ! check_command mdbook; then
    print_error "mdBook is not installed"
    echo ""
    print_info "To install mdBook, run:"
    echo "  cargo install mdbook"
    exit 1
fi

if ! check_command python3; then
    print_error "Python 3 is not installed"
    exit 1
fi

print_success "Dependencies verified"

# Set environment-specific variables
case $ENVIRONMENT in
    "staging")
        DEPLOY_DIR="build/staging"
        SITE_URL="/staging/"
        ;;
    "production")
        DEPLOY_DIR="build/production"
        SITE_URL="/"
        ;;
    *)
        DEPLOY_DIR="build/local"
        SITE_URL="/Ebook_DSPy/"
        ;;
esac

# Build the ebook
if [ "$SKIP_BUILD" = false ]; then
    print_header "Step 1: Building Ebook"

    # Validate Python code
    print_info "Validating Python code examples..."
    if python3 scripts/validate_code.py; then
        print_success "Code validation passed"
    else
        print_error "Code validation failed"
        exit 1
    fi

    # Build with mdBook
    print_info "Building with mdBook..."
    if mdbook build; then
        print_success "Build completed successfully"
    else
        print_error "Build failed"
        exit 1
    fi
else
    print_warning "Skipping build step (--skip-build flag used)"
fi

# Prepare deployment files
print_header "Step 2: Preparing Deployment Files"

# Clean previous deployment
rm -rf "$DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR"

# Copy HTML files
print_info "Copying HTML files..."
cp -r build/html/html/* "$DEPLOY_DIR/"

# Create PDF directory and copy PDF
print_info "Preparing PDF files..."
mkdir -p "$DEPLOY_DIR/pdf"
PDF_FILE="build/html/pdf/output.pdf"
if [ -f "$PDF_FILE" ]; then
    # Get version from git
    VERSION=$(git describe --tags --abbrev=0 2>/dev/null || echo "v$(date +%Y.%m.%d)")
    cp "$PDF_FILE" "$DEPLOY_DIR/pdf/DSPy_A_Practical_Guide_$VERSION.pdf"
    ln -sf "DSPy_A_Practical_Guide_$VERSION.pdf" "$DEPLOY_DIR/pdf/DSPy_A_Practical_Guide_latest.pdf"
    print_success "PDF files prepared"
else
    print_warning "PDF file not found at $PDF_FILE"
fi

# Create version info
print_info "Creating version info..."
VERSION_INFO="{
  \"version\": \"$VERSION\",
  \"commit\": \"$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')\",
  \"build_date\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\",
  \"environment\": \"$ENVIRONMENT\"
}"
echo "$VERSION_INFO" > "$DEPLOY_DIR/version.json"

# Create index.html if it doesn't exist
if [ ! -f "$DEPLOY_DIR/index.html" ]; then
    print_info "Creating index.html..."
    cat > "$DEPLOY_DIR/index.html" << EOF
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Redirecting to DSPy Ebook</title>
    <meta http-equiv="refresh" content="0; url=00-frontmatter/00-preface.html">
    <link rel="canonical" href="00-frontmatter/00-preface.html">
</head>
<body>
    <p>Redirecting to <a href="00-frontmatter/00-preface.html">DSPy: A Practical Guide</a>...</p>
</body>
</html>
EOF
fi

print_success "Deployment files prepared"

# Environment-specific deployment
print_header "Step 3: Environment Deployment"

case $ENVIRONMENT in
    "local")
        print_info "Local build complete!"
        print_info "To view the ebook, run:"
        echo "  mdbook serve --port 3000"
        echo ""
        print_info "Or open the built files:"
        echo "  open $DEPLOY_DIR/index.html"
        ;;

    "staging")
        print_info "Staging deployment prepared at: $DEPLOY_DIR"
        print_info "Files ready for manual deployment to staging server"
        ;;

    "production")
        print_warning "Production deployment should be done via GitHub Actions"
        print_info "To trigger production deployment:"
        echo "  1. Push to main branch"
        echo "  2. Go to Actions tab in GitHub"
        echo "  3. Run 'Deploy Ebook' workflow with production flag"
        ;;
esac

# Show deployment summary
print_header "Deployment Summary"

echo "Environment: $ENVIRONMENT"
echo "Deploy directory: $DEPLOY_DIR"
echo "Version: $VERSION"

if [ -f "$DEPLOY_DIR/pdf/DSPy_A_Practical_Guide_latest.pdf" ]; then
    echo "PDF size: $(du -h "$DEPLOY_DIR/pdf/DSPy_A_Practical_Guide_latest.pdf" | cut -f1)"
fi

echo ""
print_success "Deployment preparation complete!"

# Next steps
if [ "$ENVIRONMENT" != "production" ]; then
    echo ""
    print_info "Next steps:"
    echo "1. Review the built files"
    echo "2. Test locally if needed"
    echo "3. Deploy to your target platform"
    echo "4. Update any configuration as needed"
fi
#!/bin/bash
# Installation script for Apple Silicon Macs with MPS support

set -e  # Exit on error

echo "=========================================="
echo "Installing Dependencies for macOS"
echo "=========================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"
echo ""

# Step 1: Install PyTorch first (required for MPS support on Apple Silicon)
echo "Step 1: Installing PyTorch with MPS support..."
echo "This may take a few minutes..."
echo ""

# Install PyTorch - try standard method first
if pip install torch torchvision torchaudio; then
    echo "✓ PyTorch installed successfully"
else
    echo "ERROR: Failed to install PyTorch"
    echo ""
    echo "Please try installing PyTorch manually:"
    echo "  pip install torch torchvision torchaudio"
    echo ""
    echo "Or visit: https://pytorch.org/get-started/locally/"
    exit 1
fi

# Verify PyTorch installation and MPS availability
echo ""
echo "Verifying PyTorch installation..."
python3 -c "
import torch
print(f'✓ PyTorch version: {torch.__version__}')
if hasattr(torch.backends, 'mps'):
    if torch.backends.mps.is_available():
        print('✓ MPS (Metal Performance Shaders) is available!')
        print('  Your Mac will use GPU acceleration for PyTorch operations.')
    else:
        print('⚠ MPS is not available (will use CPU)')
        print('  This may be due to macOS version (requires 12.3+)')
else:
    print('⚠ MPS backend not found')
    print('  PyTorch version may be too old (requires 1.12+)')
print(f'  CUDA available: {torch.cuda.is_available()}')
" || {
    echo "⚠ Could not verify PyTorch installation"
}

# Step 2: Install other requirements
echo ""
echo "Step 2: Installing other dependencies..."
echo ""

# Install packages one by one to avoid conflicts
packages=(
    "numpy>=1.21.0,<2.0.0"
    "pandas>=1.5.0,<3.0.0"
    "scikit-learn>=1.0.0,<2.0.0"
    "transformers>=4.20.0"
    "sentence-transformers>=2.2.0"
    "sentencepiece>=0.1.99"
    "nltk>=3.8.0"
    "jieba>=0.42.1"
    "matplotlib>=3.5.0"
    "seaborn>=0.12.0"
    "plotly>=5.0.0"
    "networkx>=2.8.0"
    "xgboost>=1.7.0,<3.0.0"
    "tqdm>=4.64.0"
)

for package in "${packages[@]}"; do
    echo "Installing $package..."
    pip install "$package" || echo "⚠ Warning: Failed to install $package"
done

# Step 3: Download NLTK data
echo ""
echo "Step 3: Downloading NLTK data..."
python3 -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    print('✓ NLTK data downloaded successfully')
except Exception as e:
    print(f'⚠ NLTK download warning: {e}')
    print('You may need to download NLTK data manually later')
"

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Verify MPS: python3 verify_mps.py"
echo "2. Run pipeline: python3 main.py"
echo ""
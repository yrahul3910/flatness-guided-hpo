#!/bin/bash
# Build script for image-rs

# Set LibTorch path
export LIBTORCH="${LIBTORCH:-$HOME/Downloads/libtorch}"

if [ ! -d "$LIBTORCH" ]; then
    echo "Error: LibTorch not found at $LIBTORCH"
    echo "Please download LibTorch and set LIBTORCH environment variable"
    echo "Example: export LIBTORCH=~/Downloads/libtorch"
    exit 1
fi

echo "Using LibTorch from: $LIBTORCH"

# Build
cargo build --release

echo ""
echo "Build complete! Run with:"
echo "  LIBTORCH=$LIBTORCH cargo run --release"

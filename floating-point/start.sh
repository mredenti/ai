#!/usr/bin/env bash

echo "🚀 Setting up the Floating-Point Explorer..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null
then
    echo "❌ Python 3 could not be found. Please install Python 3.10 or higher."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "⬇️  Installing dependencies (marimo, numpy)..."
pip install -q --upgrade pip
pip install -q marimo numpy

# Run the marimo notebook
echo "✨ Starting Marimo server..."
echo "🌐 The notebook will open in your default web browser shortly."
marimo run floating_point_explorer.py
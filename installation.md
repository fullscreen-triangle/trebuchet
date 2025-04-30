# Trebuchet Installation Guide

This document provides detailed installation instructions for the Trebuchet microservices framework. These instructions are optimized for macOS, specifically for Apple Silicon (M1/M2) Macs running macOS Sequoia (15.4.1).

## Table of Contents

- [System Requirements](#system-requirements)
- [Prerequisites](#prerequisites)
- [Installation Steps](#installation-steps)
- [Project Folder Structure](#project-folder-structure)
- [Development Environment Setup](#development-environment-setup)
- [Configuration](#configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## System Requirements

- **Operating System**: macOS 13 (Ventura) or newer (optimized for macOS Sequoia 15.4.1)
- **CPU**: Apple Silicon (M1/M2/M3) or Intel-based Mac
- **RAM**: Minimum 8GB, 16GB+ recommended
- **Disk Space**: At least 10GB of free space
- **Internet Connection**: Required for downloading dependencies

## Prerequisites

### Install Homebrew

Homebrew is the preferred package manager for installing dependencies on macOS:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

If you're using Apple Silicon (M1/M2/M3), you may need to add Homebrew to your PATH:

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

### Install Rust

Install Rust using rustup, the Rust toolchain installer:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Follow the on-screen instructions, selecting option 1 for the default installation.

Verify the installation:

```bash
source "$HOME/.cargo/env"
rustc --version
cargo --version
```

### Install Python 3.8+

Install Python using Homebrew:

```bash
brew install python@3.10
```

Verify the installation:

```bash
python3 --version
```

Create a virtual environment for Python dependencies:

```bash
pip3 install virtualenv
```

### Install Node.js and npm

Install Node.js (required for WASM development) using Homebrew:

```bash
brew install node@16
```

Verify the installation:

```bash
node --version
npm --version
```

### Install Additional Tools

Install additional tools needed for development:

```bash
# Install Git
brew install git

# Install LLVM (required for some Rust crates with C++ bindings)
brew install llvm

# Install CMake (required for building some dependencies)
brew install cmake

# Install pkg-config (required for linking system libraries)
brew install pkg-config
```

## Installation Steps

### 1. Clone the Repository

```bash
# Create a development directory (optional)
mkdir -p ~/Development
cd ~/Development

# Clone the repository
git clone https://github.com/yourusername/trebuchet.git
cd trebuchet
```

### 2. Set Up the Rust Environment

```bash
# Update Rust to the latest stable version
rustup update stable

# Add WebAssembly target for WASM components
rustup target add wasm32-unknown-unknown

# Install Cargo tools
cargo install cargo-expand
cargo install cargo-watch
cargo install cargo-edit
```

### 3. Build the Core Project

```bash
# Build the project in debug mode
cargo build

# Or build in release mode for better performance
cargo build --release
```

### 4. Set Up Python Environment

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r python-bridge/requirements.txt
```

### 5. Set Up Node.js Environment (for WASM Frontend)

```bash
# Navigate to the WASM frontend directory
cd wasm-frontend

# Install Node.js dependencies
npm install
```

### 6. Install the Command-Line Tool

```bash
# Install the CLI tool globally
cargo install --path trebuchet-cli

# Verify the installation
trebuchet --version
```

## Project Folder Structure

The Trebuchet project follows a modular structure with the following main directories:

```
trebuchet/
├── Cargo.toml                # Root Cargo manifest
├── Cargo.lock                # Locked dependencies
├── .gitignore                # Git ignore file
├── README.md                 # Project README
├── docs/                     # Documentation
│   ├── assets/               # Documentation assets
│   └── ...                   # Other documentation
├── trebuchet-cli/            # CLI application
│   ├── Cargo.toml            # CLI-specific manifest
│   └── src/                  # CLI source code
├── trebuchet-core/           # Core functionality
│   ├── Cargo.toml            # Core-specific manifest
│   └── src/                  # Core source code
│       ├── lib.rs            # Library entry point
│       ├── config/           # Configuration management
│       ├── error/            # Error handling
│       └── models/           # Data models
├── microservices/            # Performance microservices
│   ├── heihachi/             # Audio processing engine
│   ├── gospel/               # Genomics analysis service
│   ├── purpose/              # Data pipeline service
│   ├── combine/              # Model router service
│   └── Cargo.toml            # Workspace manifest
├── python-bridge/            # Python interoperability
│   ├── Cargo.toml            # Bridge-specific manifest
│   ├── src/                  # Rust source for Python bridge
│   ├── py/                   # Python modules
│   └── requirements.txt      # Python dependencies
├── wasm-frontend/            # WASM integration for browsers
│   ├── Cargo.toml            # WASM-specific manifest
│   ├── src/                  # Rust source for WASM
│   ├── js/                   # JavaScript integration
│   ├── package.json          # Node.js package manifest
│   └── webpack.config.js     # Webpack configuration
├── communication/            # Communication infrastructure
│   ├── message-bus/          # Message bus implementation
│   └── api-gateway/          # API Gateway implementation
└── examples/                 # Example projects and workflows
    ├── audio-pipeline/       # Example audio processing
    ├── genomics-workflow/    # Example genomics workflow
    └── model-ensemble/       # Example model ensemble
```

## Development Environment Setup

### Code Editor Setup

#### Visual Studio Code

For the best development experience, we recommend using Visual Studio Code with the following extensions:

1. Install VS Code:

```bash
brew install --cask visual-studio-code
```

2. Install recommended extensions:

```bash
# Launch VS Code from the project directory
code .

# Install extensions (can also be done from the Extensions panel)
code --install-extension rust-lang.rust-analyzer
code --install-extension vadimcn.vscode-lldb
code --install-extension serayuzgur.crates
code --install-extension tamasfe.even-better-toml
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension dbaeumer.vscode-eslint
code --install-extension esbenp.prettier-vscode
```

3. Configure VS Code settings:

Create a `.vscode/settings.json` file in the project root with the following content:

```json
{
  "rust-analyzer.checkOnSave.command": "clippy",
  "rust-analyzer.cargo.allFeatures": true,
  "editor.formatOnSave": true,
  "[rust]": {
    "editor.defaultFormatter": "rust-lang.rust-analyzer"
  },
  "[python]": {
    "editor.defaultFormatter": "ms-python.python"
  },
  "[javascript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true
}
```

### Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Development environment
TREBUCHET_ENV=development

# Log level
RUST_LOG=info

# Python bridge configuration
PYTHON_BRIDGE_ENABLED=true
PYTHON_PATH=/usr/local/bin/python3

# API keys (if needed)
# API_KEY=your_api_key_here
```

## Configuration

### Core Configuration

The main configuration file is located at `config/trebuchet.yaml`. Create this file with the following content:

```yaml
# Trebuchet core configuration
version: "1.0"

# Environment settings
environment: development

# Logging configuration
logging:
  level: info
  format: json
  file: logs/trebuchet.log

# Service configuration
services:
  heihachi:
    enabled: true
    threads: 4
    buffer_size: 8192
  
  gospel:
    enabled: true
    threads: 4
    
  purpose:
    enabled: true
    batch_size: 1024
    
  combine:
    enabled: true
    model_registry: local

# Communication settings
communication:
  message_bus:
    type: tokio-channel
    buffer_size: 1000
    
  api_gateway:
    host: 127.0.0.1
    port: 8080
    cors_enabled: true
```

### Python Bridge Configuration

Create a configuration file at `python-bridge/config.yaml`:

```yaml
# Python bridge configuration
version: "1.0"

# Python environment
python:
  path: .venv/bin/python
  libraries:
    - numpy
    - torch
    - pandas
    - scikit-learn

# Interface settings
interface:
  type: pyo3
  fallback: json-stdio
```

## Verification

To verify that your installation is working correctly:

```bash
# Run the test suite
cargo test

# Start the CLI in interactive mode
trebuchet interactive

# Run a sample workflow
trebuchet run examples/audio-pipeline/workflow.yaml
```

## Troubleshooting

### Common Issues on macOS

#### Library Linking Issues

If you encounter linking errors with system libraries:

```bash
# Install required system libraries
brew install openssl@1.1 libffi

# Set environment variables for linking
export OPENSSL_DIR=$(brew --prefix openssl@1.1)
export LDFLAGS="-L$(brew --prefix libffi)/lib"
export CPPFLAGS="-I$(brew --prefix libffi)/include"
```

#### Python Bridge Issues

If the Python bridge fails to build:

```bash
# Ensure Python development headers are available
brew reinstall python@3.10

# Set Python config path
export PYTHON_CONFIGURE_OPTS="--enable-framework"
```

#### Permission Issues

If you encounter permission issues:

```bash
# Fix permissions in Homebrew directories
sudo chown -R $(whoami) $(brew --prefix)/*

# Fix permissions in Cargo directories
sudo chown -R $(whoami) ~/.cargo/
```

#### "Command not found" After Installation

If the `trebuchet` command is not found after installation:

```bash
# Add Cargo bin directory to PATH
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

## Getting Help

If you encounter any issues not covered in this guide:

1. Check the [GitHub issues](https://github.com/yourusername/trebuchet/issues) for similar problems
2. Join the [Trebuchet Discord community](https://discord.gg/trebuchet)
3. Post on the [Trebuchet discussion forum](https://github.com/yourusername/trebuchet/discussions)

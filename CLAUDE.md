# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Unsloth is a library for 2-5x faster LLM fine-tuning with 70-80% less VRAM usage. It provides optimized implementations of popular language models including Llama, Mistral, Qwen, Gemma, and others with support for various quantization methods, LoRA/QLoRA training, and export to multiple formats (GGUF, Ollama, vLLM, Hugging Face).

## Installation & Setup

This project uses setuptools with pyproject.toml configuration. Install with:
```bash
pip install unsloth
```

For development, the project supports multiple CUDA/PyTorch combinations via optional dependencies defined in pyproject.toml.

## Development Commands

### Testing
- **Run tests**: `python -m pytest tests/` (use pytest for most tests)
- **Run specific test**: `python -m pytest tests/test_model_registry.py`
- **MoE tests**: Special handling required - use test scripts instead of pytest directly:
  - `python -m tests.test_qwen3_moe --permute_x --permute_y --autotune` (NOT pytest)
  - See `unsloth/kernels/moe/tests/run_qwen3_moe_tests.sh` for more details

### CLI Tool
- **Main CLI**: `python unsloth-cli.py --help` - CLI for fine-tuning with extensive configuration options
- **Auto-install script**: `python unsloth/_auto_install.py` - determines optimal pip installation command

### Model validation commands
- Check xformers: `python -m xformers.info`
- Check bitsandbytes: `python -m bitsandbytes`

## Code Architecture

### Core Structure
- **`unsloth/`** - Main package directory
  - **`models/`** - Model implementations (Llama, Mistral, Qwen, Gemma, etc.)
    - `loader.py` - Central model loading logic (FastLanguageModel, FastVisionModel, etc.)
    - `_utils.py` - Utility functions for model operations
    - Individual model files: `llama.py`, `mistral.py`, `qwen2.py`, `qwen3.py`, etc.
  - **`kernels/`** - Optimized Triton kernels for acceleration
    - `fast_lora.py`, `cross_entropy_loss.py`, `rope_embedding.py`, etc.
    - `moe/` - Mixture of Experts implementations with grouped GEMM operations
  - **`registry/`** - Model registration system for supported models
  - **`dataprep/`** - Data preparation utilities
  - `trainer.py` - Training utilities and patches
  - `save.py` - Model saving and export functionality

### Key Concepts
1. **Import Order Critical**: Unsloth MUST be imported before transformers, trl, or peft to apply optimizations
2. **Model Factory Pattern**: Uses FastLanguageModel.from_pretrained() as main entry point
3. **Kernel Optimization**: Custom Triton kernels provide 2x+ speedup for key operations
4. **Multi-GPU**: Currently single GPU only (multi-GPU in beta)
5. **Memory Optimization**: Extensive CUDA memory management and fragmentation reduction

### Model Support Pattern
Each supported model (Llama, Mistral, etc.) follows this pattern:
- Model-specific file in `models/` (e.g., `llama.py`)
- Registry entry in `registry/` 
- Optional custom kernels in `kernels/`
- Comprehensive tests in `tests/`

### Testing Architecture
- **Unit tests**: `tests/` directory with pytest-based tests
- **MoE tests**: Special module-based execution due to Triton/autotuning conflicts
- **Integration tests**: Model saving, merging, and perplexity validation
- **Benchmark tests**: Performance validation in `tests/utils/`

## Important Development Notes

1. **Critical Import Order**: Always import unsloth before other ML libraries
2. **Windows Support**: Requires specific setup including Visual Studio C++, CUDA toolkit, and platform-specific wheels
3. **CUDA Memory**: Project includes extensive CUDA memory optimization settings
4. **MoE Testing**: Use test scripts, not pytest, for MoE-related tests due to Triton autotuning issues
5. **Multi-GPU**: Currently single GPU only - multi-GPU support is in beta

## Environment Variables
- `UNSLOTH_USE_MODELSCOPE=1` - Use ModelScope for model/dataset downloads
- `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` - Required for compatibility
- `PYTORCH_CUDA_ALLOC_CONF` - Automatically set for memory optimization

## Model Export Support
The project supports exporting to multiple formats:
- GGUF (llama.cpp compatible)
- Ollama
- vLLM
- Hugging Face Hub
- Various quantization formats (4-bit, 8-bit, 16-bit)
# resonance-lab
Python project for Resonance Stream, which generate output for translator model, based from Qwen 3 1.7B

## Development
- Rules, gates and layout: [`CLAUDE.md`](CLAUDE.md); current state: [`MEMORY.md`](MEMORY.md)
- `pip install -r requirements-dev.txt` then `just check` (lint + data-stage tests, CPU only)
- `python run_pipeline.py --pipeline unsloth` runs a training pipeline (`pipelines.py` lists them)

## Prerequisites
- Python 3.13
- Windows OS (Linux not tested yet)
- CUDA Toolkit 2.6

## Convert to Model (F16 -> GGUF -> q4_k_m)
- Make sure model_f16_clean config.json "architectures": "Qwen3ForCausalLM"
- git clone --recursive https://github.com/ggerganov/llama.cpp
- cmake -B build
- cmake --build build --config Release -j
- python llama.cpp/convert_hf_to_gguf.py model_f16_clean --outfile model.f16.gguf
- llama.cpp/build/bin/Release/llama-quantize.exe .\model.f16.gguf model_q4_k_m.gguf q4_k_m
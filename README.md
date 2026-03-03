# Resonance Lab
Python project for Resonance Stream, which generate output for translator model, based from Qwen 3 1.7B

hugging-face: https://huggingface.co/enjay27/Qwen3-Blue-Protocol-Translator-JA-KO

## Prerequisites
- Python 3.13
- Windows OS (Linux not tested yet)
- CUDA Toolkit 2.6

## Install
- pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu126
- pip uninstall xformers -y
- pip install --no-deps xformers==0.0.33.post2
- pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu126
- pip install llamafactory

## upload to HF
- pip install sacrebleu
- hf upload enjay27/Qwen3-Blue-Protocol-Translator-JA-KO model_name

## Convert to Model (F16 -> GGUF -> q4_k_m)
- Make sure model_f16_clean config.json "architectures": "Qwen3ForCausalLM"
- git clone --recursive https://github.com/ggerganov/llama.cpp
- cmake -B build
- cmake --build build --config Release -j
- python llama.cpp/convert_hf_to_gguf.py model_merged --outfile model_gguf/bp-qwen3-f16.gguf --outtype f16
- llama.cpp/build/bin/Release/llama-quantize.exe model_gguf/bp-qwen3-f16.gguf model_gguf/bp-qwen3-q4_k_m.gguf q4_k_m

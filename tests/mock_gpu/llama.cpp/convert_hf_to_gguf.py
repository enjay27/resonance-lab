"""Fake llama.cpp convert_hf_to_gguf.py <merged model dir> --outfile <gguf> --outtype f16."""

import argparse
import os
import sys

sys.path.insert(0, os.environ["MOCK_GPU_LIB"])
import contract  # noqa: E402

TOOL = "convert_hf_to_gguf.py"

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--outfile", required=True)
parser.add_argument("--outtype", required=True)
args = parser.parse_args()

contract.require(TOOL, os.path.isdir(args.model), f"model directory {args.model!r} does not exist")
missing = [name for name in ("config.json", "model.safetensors") if not os.path.isfile(os.path.join(args.model, name))]
contract.require(TOOL, not missing, f"{args.model!r} lacks {', '.join(missing)}: it is not a merged Hugging Face model")
contract.require(TOOL, args.outtype == "f16", f"--outtype {args.outtype!r}: the pipeline quantizes from f16")
contract.require(TOOL, os.path.isdir(os.path.dirname(os.path.abspath(args.outfile))), f"output folder of {args.outfile!r} does not exist")
contract.write(args.outfile, contract.GGUF_MAGIC + b" mock f16\n")
print(f"mock convert {args.model} -> {args.outfile}")

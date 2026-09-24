#!/usr/bin/env bash
# Fetch BAAI/bge-small-en-v1.5 in ONNX form (~133 MB).
# ONNX + tokenizers avoids a multi-GB torch install; the model is never
# fine-tuned here, only used for inference, so the ONNX export is sufficient.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
BASE=https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main
for f in tokenizer.json config.json tokenizer_config.json special_tokens_map.json; do
  [ -s "models/$f" ] || curl -fsSL -o "models/$f" "$BASE/$f"
  echo "  models/$f  $(wc -c < "models/$f") bytes"
done
if [ ! -s models/model.onnx ]; then
  echo "  downloading model.onnx (~133 MB) ..."
  curl -fL --progress-bar -o models/model.onnx "$BASE/onnx/model.onnx"
fi
echo "  models/model.onnx  $(wc -c < models/model.onnx) bytes"

"""Embed paragraphs with BAAI/bge-small-en-v1.5 via onnxruntime.

CLS pooling + L2 normalisation, which is what this model was trained for.
Paragraphs longer than the 512-token window are split into overlapping chunks
and averaged, so a long paragraph is not silently truncated.
"""

import json
import os

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

HERE = os.path.dirname(__file__)
MODELS = os.path.join(HERE, "..", "models")
PROC = os.path.join(HERE, "..", "data", "proc")

MAX_LEN = 512
STRIDE = 384


class Embedder:
    def __init__(self):
        self.tok = Tokenizer.from_file(os.path.join(MODELS, "tokenizer.json"))
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            os.path.join(MODELS, "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def _chunks(self, text):
        ids = self.tok.encode(text, add_special_tokens=False).ids
        if not ids:
            ids = [0]
        body = MAX_LEN - 2
        if len(ids) <= body:
            return [[101] + ids + [102]]
        out = []
        for s in range(0, len(ids), STRIDE):
            piece = ids[s:s + body]
            if not piece:
                break
            out.append([101] + piece + [102])
            if s + body >= len(ids):
                break
        return out

    def encode(self, texts, batch_size=16, log_every=0):
        vecs = []
        flat, owner = [], []
        for i, t in enumerate(texts):
            for c in self._chunks(t):
                flat.append(c)
                owner.append(i)

        chunk_vecs = np.zeros((len(flat), 384), dtype=np.float32)
        order = np.argsort([len(c) for c in flat])          # bucket by length
        for bi in range(0, len(order), batch_size):
            idx = order[bi:bi + batch_size]
            batch = [flat[i] for i in idx]
            n = max(len(b) for b in batch)
            ids = np.zeros((len(batch), n), dtype=np.int64)
            mask = np.zeros((len(batch), n), dtype=np.int64)
            for r, b in enumerate(batch):
                ids[r, :len(b)] = b
                mask[r, :len(b)] = 1
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            out = self.sess.run(None, feed)[0]
            chunk_vecs[idx] = out[:, 0, :]                  # CLS token
            if log_every and (bi // batch_size) % log_every == 0:
                print(f"    {bi:>6}/{len(order)} chunks", flush=True)

        owner = np.array(owner)
        for i in range(len(texts)):
            v = chunk_vecs[owner == i].mean(axis=0)
            vecs.append(v)
        V = np.vstack(vecs)
        V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
        return V.astype(np.float32)


def main():
    rows = [json.loads(l) for l in open(os.path.join(PROC, "paragraphs.jsonl"))]
    emb = Embedder()
    print(f"embedding {len(rows):,} paragraphs ...")
    V = emb.encode([r["text"] for r in rows], log_every=40)
    np.save(os.path.join(PROC, "embeddings.npy"), V)
    print("saved", V.shape, "norm check", float(np.linalg.norm(V[0])))


if __name__ == "__main__":
    main()

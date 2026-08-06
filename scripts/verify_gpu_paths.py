#!/usr/bin/env python3
"""Verify three TinyInfer generation paths on GPU — token-level alignment."""

import sys
sys.path.insert(0, "/mnt/d/project/TinyInfer")

from tinyinfer.config import EngineConfig
from tinyinfer.runtime import (
    GenerationRequest,
    generate_greedy,
    generate_with_owned_cache,
    generate_with_paged_cache,
    load_reference_model,
    reference_generate,
)

def main():
    print("Loading model...")
    loaded = load_reference_model(EngineConfig())
    print(f"  Model: {loaded.model_id} @ {loaded.model_revision}")
    print(f"  Device: {loaded.model.device}")

    prompts = [
        GenerationRequest("short_en", "Explain KV cache in one concise paragraph.", 16),
        GenerationRequest("short_zh", "请用一段简洁的话解释大语言模型中的 KV Cache。", 16),
        GenerationRequest("structured", "Return exactly three numbered points explaining prefill, decode, and token latency.", 16),
        GenerationRequest("long_context", "A reference runtime separates prompt prefill from one-token decode. " * 10 + "Summarize the engineering lesson.", 16),
    ]

    all_passed = True
    for req in prompts:
        print(f"\n--- {req.request_id} ---")
        ref = reference_generate(loaded, req)

        r1 = generate_greedy(loaded, req)
        ok1 = r1.output_token_ids == ref
        print(f"  generate_greedy         : {'PASS' if ok1 else 'FAIL'}")

        r2 = generate_with_owned_cache(loaded, req)
        ok2 = r2.output_token_ids == ref
        print(f"  generate_with_owned_cache: {'PASS' if ok2 else 'FAIL'}")

        r3 = generate_with_paged_cache(loaded, req)
        ok3 = r3.output_token_ids == ref
        print(f"  generate_with_paged_cache: {'PASS' if ok3 else 'FAIL'}")

        if not (ok1 and ok2 and ok3):
            print(f"  REF: {ref}")
            if not ok1: print(f"  G1 : {r1.output_token_ids}")
            if not ok2: print(f"  G1.5: {r2.output_token_ids}")
            if not ok3: print(f"  G2: {r3.output_token_ids}")
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("ALL THREE PATHS: TOKEN AGREEMENT = 100%")
        return 0
    else:
        print("TOKEN AGREEMENT FAILED")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
# Multimodal Inferencing Library Development Plan

This document outlines a sequential plan to add advanced features to the multimodal inferencing library. The plan is broken down into phases and discrete tasks, prioritizing foundational changes that enable more complex capabilities later.

## Phase 1: Foundational Refactoring (Gaining Control over Generation)

This phase focuses on moving the token generation loop from C++ to Python, which is the essential prerequisite for most advanced features. This has a potential performance cost due to the Python/C++ round-trip for each token.

*   **Task 1.1: Define a Single-Token Result Struct.** In `generation_helper.h`, create a new C++ struct (`SingleTokenResult`) to hold the output of a single generation step. It should contain the generated `llama_token` and a boolean flag indicating if the end-of-generation token was sampled.
*   **Task 1.2: Create a Single-Token Generation Helper.** In `generation_helper.cpp`, create a new function `generate_one_token_cpp`. This function will perform one step: sample a token, accept it, decode it, update `n_past`, and return the `SingleTokenResult`. The existing `generate_tokens_cpp` function will be kept for performance comparison.
*   **Task 1.3: Manage `llama_batch` from Python.** Modify `cpppy/ccpyy-mtmd.py` to initialize and free the `llama_batch` object directly in Python code, as it will now be passed into the C++ helper at each step.
*   **Task 1.4: Implement the Generation Loop in Python.** In `cpppy/ccpyy-mtmd.py`, create a new generation path that uses a Python `while` loop to call the new `generate_one_token_cpp` repeatedly.
*   **Task 1.5: Performance Evaluation.** Add logic to the Python script to allow switching between the old C++ loop (`generate_tokens_cpp`) and the new Python loop. Benchmark the tokens/second performance of both approaches.
*   **Task 1.6: Retire Old C++ Loop (Conditional).** If the performance degradation of the Python loop is acceptable for most use cases, remove the old `generate_tokens_cpp` function and related code. Otherwise, keep both and document the trade-offs.

## Phase 2: Exposing KV Cache Primitives

This phase exposes the low-level `llama.cpp` KV cache functions needed for caching and batching. These are simple wrappers and should have negligible performance impact.

*   **Task 2.1: Expose KV Cache Sequence Removal.** Create a C++ helper function that wraps `llama_kv_cache_seq_rm` to allow Python to remove a specific sequence from the KV cache.
*   **Task 2.2: Expose KV Cache Sequence Copying.** Create a C++ helper function that wraps `llama_kv_cache_seq_cp` to allow Python to copy the KV state from one sequence ID to another.
*   **Task 2.3: Expose KV Cache Clearing.** Create a C++ helper function that wraps `llama_kv_cache_clear` to allow Python to completely clear the KV cache.

## Phase 3: Implementing State Caching

This phase uses the primitives from Phase 2 to build the caching feature.

*   **Task 3.1: Implement Image Encoding Caching.** In Python, after processing an image, use the sequence copy helper (from Task 2.2) to save the resulting KV cache state to a new, unused sequence ID. Store a mapping of the image identifier to this cache sequence ID in a Python dictionary.
*   **Task 3.2: Implement Cache Restoration.** In Python, before processing an image, check the dictionary from Task 3.1. If the image is found in the cache, use the sequence copy helper to restore its KV state into the active sequence, skipping the expensive image evaluation step.

## Phase 4: Enabling Advanced Logit Control

This phase provides the tools for logit-based generation guidance. This has a high potential for performance impact.

*   **Task 4.1: Create a Logit-Reading Helper.** Create a C++ helper function that provides Python with access (e.g., via a pointer or a copy) to the raw logits array after a `llama_decode` call.
*   **Task 4.2: Implement a Python-Side Logit-Guiding Hook.** In the Python generation loop (from Task 1.4), after decoding but before sampling, call the new logit-reading helper.
*   **Task 4.3: Create a Custom Sampling Helper.** Create a C++ helper that can take a modified logits array from Python and perform the sampling operation, returning the chosen token. This separates sampling from the main generation step.
*   **Task 4.4: Performance Evaluation.** Compare the performance of generation with the Python logit hook against the original Python loop (from Phase 1).
*   **Task 4.5: Create C++-based Logit Processors (Optional).** If performance with Python hooks is insufficient, design an interface to register C++-based logit processor plugins (e.g., for forcing JSON output) that can be called from the C++ side without returning to Python.

## Phase 5: Implementing Multi-Sequence Batching

This final phase combines previous work to enable true batch inferencing.

*   **Task 5.1: Design a Python Sequence Manager.** Create a Python class or data structure to manage the state of multiple, independent generation sequences (e.g., their prompts, `n_past`, sequence IDs, and generated text).
*   **Task 5.2: Implement Multi-Sequence Batch Construction.** Modify the main Python loop to iterate through the active sequences and add one token from each to a single `llama_batch` for a batched `llama_decode` call.
*   **Task 5.3: Implement Per-Sequence Sampling and State Update.** After the batched decode, loop through each sequence in Python, sample its next token individually (using its portion of the logits), and update its state.
*   **Task 5.4: Implement Sequence Completion Handling.** When a sequence finishes, use the KV cache removal helper (from Task 2.1) to free its resources and remove it from the active batch.

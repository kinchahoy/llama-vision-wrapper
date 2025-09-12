# Llama-Vision-Wrapper Usage Guide

This guide explains how to set up and use the Llama-Vision-Wrapper library, which provides Python bindings for `llama.cpp` to perform multimodal inference.

## 1. Setup

The library requires compiling the underlying C++ code from `llama.cpp` and a helper library. A Python script `setup.py` is provided to automate this process.

To set up the project, run the interactive setup script:
```bash
python setup.py
```
You will be prompted to choose a backend for `llama.cpp` (e.g., `cuda`, `metal`, `none`). The script will then perform all necessary steps: patching, configuring, and building the C++ components.

## 2. Multimodal Inference in Python

The core of the Python wrapper is the `ResourceManager` class from the `llama_cpp_wrapper` module. It handles loading models and managing all necessary resources within a context manager.

Here is the typical workflow for performing multimodal inference:

1.  **Initialize ResourceManager**: All operations are performed within a `with ResourceManager() as rm:` block.
2.  **Load Models**: Load the main GGUF model and the multimodal projector file.
    ```python
    model = rm.load_model("path/to/model.gguf", n_gpu_layers=32)
    ctx_mtmd = rm.load_mtmd("path/to/mmproj.gguf", model, use_gpu=True)
    ```
3.  **Create Context & Sampler**: Set up the `llama.cpp` context and a sampler for generation.
    ```python
    ctx = rm.create_context(model, n_ctx=2048, n_batch=512)
    sampler = rm.create_sampler(model, temp=0.8, top_k=40)
    ```
4.  **Load Images**: Load one or more images from file paths.
    ```python
    bitmaps = [rm.load_image(ctx_mtmd, "image1.jpg")[0]]
    ```
5.  **Tokenize Prompt**: Prepare the prompt for the model. Use `<__media__>` as a placeholder for each image. The tokenizer returns a set of "chunks" containing either text tokens or image data.
    ```python
    prompt = "USER: Describe this image.\n<__media__>\nASSISTANT:"
    chunks = rm.tokenize_prompt(ctx_mtmd, prompt, bitmaps)
    ```
6.  **Process Prompt**: Evaluate the chunks to process the prompt and image(s), filling the KV cache.
    ```python
    n_past = 0
    for i in range(gbl.mtmd_input_chunks_size(chunks)):
        chunk = gbl.mtmd_input_chunks_get(chunks, i)
        chunk_type = gbl.mtmd_input_chunk_get_type(chunk)

        if chunk_type == gbl.MTMD_INPUT_CHUNK_TYPE_TEXT:
            n_past = rm.eval_text_chunk(ctx, chunk, n_past, n_batch=512)
        elif chunk_type == gbl.MTMD_INPUT_CHUNK_TYPE_IMAGE:
            rm.encode_image_chunk(ctx_mtmd, chunk)
            n_past = rm.decode_image_chunk(ctx, ctx_mtmd, chunk, n_past, n_batch=512)
    ```
7.  **Generate Response**: Generate new tokens based on the processed prompt.
    ```python
    result, _ = rm.generate(sampler, ctx, model, n_past, n_ctx=2048, max_new_tokens=256)
    print(result.generated_text)
    ```

## 3. Running the Example

An example script is available at `cppyy-src/cppyy-mtmd.py`. It demonstrates the full process, including downloading models from Hugging Face Hub.

You can run it like this:
```bash
python cppyy-src/cppyy-mtmd.py --prompt "USER: Describe this image.\n<__media__>\nASSISTANT:" --image path/to/your/image.jpg --n-gpu-layers 32
```
The script will automatically download the default SmolVLM model and multimodal projector.

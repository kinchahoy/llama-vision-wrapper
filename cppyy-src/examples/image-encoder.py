"""
Example script to encode one or more images into media embeddings using the llama_cpp_wrapper.
"""
import sys
import argparse
import os
from huggingface_hub import hf_hub_download

# Adjust path to import from parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from llama_cpp_wrapper import ResourceManager, gbl, timed_operation
                                                                                                                                                                
def main():                                                                                                                                                     
    """Main execution function."""                                                                                                                              
    parser = argparse.ArgumentParser(
        description="Encode images into media embeddings using cppyy."
    )
    parser.add_argument(
        "images",
        metavar="IMAGE",
        type=str,
        nargs='+',
        help="Path to one or more input images.",
    )
    parser.add_argument(
        "--repo-id",
        "-hf",
        type=str,
        default="ggml-org/SmolVLM2-2.2B-Instruct-GGUF",
        help="Hugging Face repository ID for GGUF models.",
    )                                                                                                                                                           
    parser.add_argument(                                                                                                                                        
        "--model",                                                                                                                                              
        "-m",                                                                                                                                                   
        type=str,                                                                                                                                               
        default="SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",                                                                                                           
        help="Model file name in the repository.",                                                                                                              
    )                                                                                                                                                           
    parser.add_argument(                                                                                                                                        
        "--mmproj",                                                                                                                                             
        type=str,                                                                                                                                               
        default="mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf",                                                                                                      
        help="Multimodal projector file name in the repository.",                                                                                               
    )                                                                                                                                                           
    parser.add_argument(                                                                                                                                        
        "--n-gpu-layers",                                                                                                                                       
        "-ngl",                                                                                                                                                 
        type=int,                                                                                                                                               
        default=0,                                                                                                                                              
        dest="n_gpu_layers",                                                                                                                                    
        help="Number of layers to offload to GPU.",                                                                                                             
    )                                                                                                                                                           
    parser.add_argument(                                                                                                                                        
        "-t",                                                                                                                                                   
        type=int,                                                                                                                                               
        default=8,                                                                                                                                              
        dest="n_threads",                                                                                                                                       
        help="Number of threads to use for computation.",                                                                                                       
    )                                                                                                                                                           
    parser.add_argument(                                                                                                                                        
        "--output-dir",                                                                                                                                         
        "-o",                                                                                                                                                   
        type=str,                                                                                                                                               
        default=".",                                                                                                                                            
        help="Directory to save the output embedding files.",                                                                                                   
    )                                                                                                                                                           
    parser.add_argument(                                                                                                                                        
        "--verbose-cpp",                                                                                                                                        
        action="store_true",                                                                                                                                    
        help="Enable verbose logging from the C++ backend.",                                                                                                    
    )                                                                                                                                                           
    args = parser.parse_args()                                                                                                                                  
                                                                                                                                                                
    try:
        print("--- Initializing ---")
        print(f"Threads set to: {args.n_threads}")
        os.makedirs(args.output_dir, exist_ok=True)

        print("--- Downloading models from Hugging Face Hub ---")
        with timed_operation("Model download"):
            model_path = hf_hub_download(repo_id=args.repo_id, filename=args.model)
        with timed_operation("MMPROJ download"):
            mmproj_path = hf_hub_download(repo_id=args.repo_id, filename=args.mmproj)
        print(f"Model path: {model_path}")
        print(f"MMPROJ path: {mmproj_path}")        # Initialize resources
        with ResourceManager(verbose=args.verbose_cpp) as rm:
            # Load model and multimodal projector
            model = rm.load_model(model_path, args.n_gpu_layers)
            ctx_mtmd = rm.load_mtmd(
                mmproj_path,
                model,
                args.n_gpu_layers > 0,
                args.n_threads,
                args.verbose_cpp,
            )

            # A dummy prompt to trigger media processing
            dummy_prompt = "<__media__>"
                                                                                                                                                                
            for image_path in args.images:
                print(f"\n--- Processing {image_path} ---")
                if not os.path.exists(image_path):
                    print(f"Warning: Image file not found: {image_path}. Skipping.")
                    continue

                # Load image
                bitmap, _ = rm.load_image(ctx_mtmd, image_path)
                print(f"Image loaded: {gbl.mtmd_bitmap_get_nx(bitmap)}x{gbl.mtmd_bitmap_get_ny(bitmap)}")

                # Tokenize to get the media chunk
                chunks = rm.tokenize_prompt(ctx_mtmd, dummy_prompt, bitmap)                # Find the image chunk
                image_chunk = None
                for i in range(gbl.mtmd_input_chunks_size(chunks)):
                    chunk = gbl.mtmd_input_chunks_get(chunks, i)
                    if gbl.mtmd_input_chunk_get_type(chunk) in [
                        gbl.MTMD_INPUT_CHUNK_TYPE_IMAGE,
                        gbl.MTMD_INPUT_CHUNK_TYPE_AUDIO,
                    ]:
                        image_chunk = chunk
                        break

                if not image_chunk:
                    print(f"Warning: Could not create a media chunk for {image_path}. Skipping.")
                    continue
                                                                                                                                                                
                # Encode the media chunk
                print("Encoding media chunk...")
                rm.encode_image_chunk(ctx_mtmd, image_chunk)

                # Save the embedding
                base_name = os.path.basename(image_path)
                file_name, _ = os.path.splitext(base_name)
                output_path = os.path.join(args.output_dir, f"{file_name}.embedding")

                print(f"Saving media embedding to {output_path}...")
                rm.save_encoded_chunk(ctx_mtmd, model, image_chunk, output_path)
                print(f"Successfully saved embedding for {image_path}")

    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"{type(e).__name__}: {e}")
        sys.exit(1)

    print("\n--- All images processed. ---")

if __name__ == "__main__":
    main()
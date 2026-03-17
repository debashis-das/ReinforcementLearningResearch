from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os

def run_cached_model_example():
    """
    Example of running a cached model using Python for text generation
    """
    
    # Option 1: Using pipeline with a specific model (will use cache if available)
    print("=== Option 1: Using Pipeline ===")
    try:
        # This will use cached model if available, otherwise download
        generator = pipeline(
            "text-generation",
            model="Qwen/Qwen3.5-0.8B-Base",  # You can replace with any cached model
            device_map="auto"  # Automatically use GPU if available
        )
        
        # Generate text
        prompt = "The future of artificial intelligence is"
        results = generator(
            prompt,
            max_length=50,
            num_return_sequences=1,
            temperature=0.7
        )
        
        print(f"Prompt: {prompt}")
        print(f"Generated: {results[0]['generated_text']}")
        
    except Exception as e:
        print(f"Pipeline error: {e}")
    
    # Option 2: Load from specific cache directory
    print("\n=== Option 2: Load from Specific Cache ===")
    
    # Set custom cache directory if needed
    # os.environ['TRANSFORMERS_CACHE'] = '/path/to/your/cache'
    # os.environ['HF_HOME'] = '/path/to/your/cache'
    
    # Option 3: Load model and tokenizer separately from cache
    # print("\n=== Option 3: Manual Model Loading ===")
    # try:
    #     model_name = "Qwen/Qwen3.5-0.8B-Base"  # Replace with your cached model
        
    #     # Load tokenizer and model (will use cache)
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_name,
    #         device_map="auto"
    #     )
        
    #     # Generate text manually
    #     prompt = "Machine learning will revolutionize"
    #     inputs = tokenizer(prompt, return_tensors="pt")
        
    #     # Generate
    #     with torch.no_grad():
    #         outputs = model.generate(
    #             **inputs,
    #             max_length=50,
    #             num_return_sequences=1,
    #             temperature=0.7,
    #             do_sample=True,
    #             pad_token_id=tokenizer.eos_token_id
    #         )
        
    #     # Decode and print
    #     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     print(f"Prompt: {prompt}")
    #     print(f"Generated: {generated_text}")
        
    # except Exception as e:
    #     print(f"Manual loading error: {e}")

def list_cached_models():
    """
    List models in your Hugging Face cache
    """
    try:
        from huggingface_hub import list_cached_models
        
        print("\n=== Cached Models ===")
        cached_models = list_cached_models()
        
        for model in cached_models:
            print(f"Model: {model.repo_id}")
            print(f"Size: {model.size_on_disk / 1024**3:.2f} GB")
            print("---")
            
    except ImportError:
        print("huggingface_hub not available for cache inspection")

def check_cache_status():
    """
    Check cache directory and environment variables
    """
    print("\n=== Cache Status ===")
    print(f"HF_HOME: {os.environ.get('HF_HOME', 'Not set')}")
    print(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', 'Not set')}")
    print(f"HF_HUB_CACHE: {os.environ.get('HF_HUB_CACHE', 'Not set')}")
    
    # Default cache location
    default_cache = os.path.expanduser("~/.cache/huggingface")
    print(f"Default cache: {default_cache}")
    
    if os.path.exists(default_cache):
        cache_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(default_cache)
            for filename in filenames
        )
        print(f"Cache size: {cache_size / 1024**3:.2f} GB")

if __name__ == "__main__":
    import torch
    
    # Check cache status first
    check_cache_status()
    
    # List cached models
    list_cached_models()
    
    # Run the generation examples
    run_cached_model_example()
    
    print("\n=== Generation Complete ===")
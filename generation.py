from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

cache_path = "/home/personal/models"
model_name = "Qwen/Qwen3.5-0.8B-Base"

def run_model():
    """
    Example of running a cached model using Python for text generation
    """
    try:

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_path
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model,
            cache_dir=cache_path
        )
        generator = pipeline(
            "text-generation",
            model=model,  # You can replace with any cached model
            tokenizer=tokenizer
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
        print(f"Error in pipeline generation: {e}")

if __name__ == "__main__":
    # Run the generation examples
    run_model()
    
    print("\n=== Generation Complete ===")
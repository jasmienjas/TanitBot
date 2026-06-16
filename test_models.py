#!/usr/bin/env python3
"""
Tunisian Arabic LLM Benchmarking Script for RunPod
Created to evaluate and compare different models on dialectal Tunisian Arabic (Derja).
"""
import os
import gc
import sys
import time
import json
import argparse
import traceback

# Import torch and Hugging Face packages conditionally
try:
    import torch
    import torch.nn as nn
    
    # Monkey-patch nn.Module.set_submodule for compatibility with older PyTorch versions (< 2.5)
    if not hasattr(nn.Module, "set_submodule"):
        def _set_submodule(self, target: str, module: nn.Module) -> None:
            atoms = target.split(".")
            name = atoms.pop(-1)
            mod = self
            for item in atoms:
                mod = getattr(mod, item)
            setattr(mod, name, module)
        nn.Module.set_submodule = _set_submodule
except ImportError:
    print("[!] PyTorch is not installed. Please run: pip install torch")
    sys.exit(1)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
except ImportError:
    print("[!] HuggingFace Transformers is not installed. Please run: pip install transformers accelerate")
    sys.exit(1)

# Default Tunisian Arabic Prompts Suite
DEFAULT_PROMPTS = [
    {
        "id": "conversational_intro",
        "category": "Conversational",
        "description": "Greeting and basic chat in Tunisian Arabic",
        "prompt": "أحكيلي بالتونسي شنوة تنجم تعمل كـ مساعد ذكي وكيفاش تنجم تعاوني؟"
    },
    {
        "id": "conversational_story",
        "category": "Conversational",
        "description": "Short story generation in Tunisian Arabic",
        "prompt": "أحكيلي حكاية قصيرة بالتونسي على راجل مشا للمارشي يلوج على الحوت."
    },
    {
        "id": "translation_en_to_tn",
        "category": "Translation",
        "description": "Translate English to Tunisian Arabic",
        "prompt": "ترجم الجملة هذي للدارجة التونسية:\n'I am very tired today because I worked all night long, and now I just want to drink a cup of coffee and sleep.'"
    },
    {
        "id": "translation_msa_to_tn",
        "category": "Translation",
        "description": "Translate Modern Standard Arabic (MSA) to Tunisian Arabic",
        "prompt": "ترجم الجملة هذي من الفصحى للدارجة التونسية:\n'أين تقع محطة القطار وكيف يمكنني الذهاب إلى هناك؟ وهل التذاكر متوفرة الآن؟'"
    },
    {
        "id": "cultural_idiom",
        "category": "Culture & Idioms",
        "description": "Explaining a Tunisian idiom",
        "prompt": "فسرلي بالباهي المثل التونسي هذا شنوة معناه ووين نستعملوه: 'كان صبت اندبي وكان خلت اندبي'."
    },
    {
        "id": "cultural_geography",
        "category": "Culture & Geography",
        "description": "Tunisian regional and tourist advice",
        "prompt": "شنوة أحسن بلاصة تنصحني نزورها في تونس في الصيف؟ وعلاش هي بالذات؟"
    },
    {
        "id": "sentiment_negative",
        "category": "Sentiment Analysis",
        "description": "Classifying negative Tunisian Arabic statement",
        "prompt": "شنوة الإحساس (إيجابي، سلبي، محايد) متع الجملة هذي بالتونسي: 'الخدمة هذي فدّدتني وكرهت حياتي بسببها والكرهبة ديما معطلة'؟ جاوب في كلمة وحدة وبدون تفاصيل."
    },
    {
        "id": "sentiment_positive",
        "category": "Sentiment Analysis",
        "description": "Classifying positive Tunisian Arabic statement",
        "prompt": "شنوة الإحساس (إيجابي، سلبي، محايد) متع الجملة هذي بالتونسي: 'ملا جو وملا كادو مزيان، يعطيك الصحة وفرحتني برشا!'؟ جاوب في كلمة وحدة وبدون تفاصيل."
    },
    {
        "id": "code_switching",
        "category": "Code-Switching",
        "description": "Tunisian Arabic mixed with French words",
        "prompt": "جاوبني بالتونسي مع استعمال شوية كلمات بالفرنسية (كيفما يحكيو برشا توانسة في الشارع): كيفاش نجم نعمل رز بالفاكهة؟"
    }
]


def load_model_and_tokenizer(model_id, quantization=None, hf_token=None, cache_dir=None):
    """
    Loads tokenizer and model with appropriate configurations for CUDA/CPU and optional quantization.
    """
    print(f"\n[+] Loading model and tokenizer for: {model_id}")
    
    # Configure quantization if specified
    bnb_config = None
    if quantization in [4, "4"]:
        print("    [Config] Loading in 4-bit precision (NF4)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        )
    elif quantization in [8, "8"]:
        print("    [Config] Loading in 8-bit precision...")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Determine automatic device mapping and data type for modern GPUs
    if torch.cuda.is_available():
        device_map = "auto"
        if bnb_config is not None:
            torch_dtype = None  # bitsandbytes manages precision internally
        else:
            # Prefer bfloat16 for modern Ampere/Hopper/Ada GPUs on RunPod, fallback to float16
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            print(f"    [Config] GPU detected. Using half-precision: {torch_dtype}")
    else:
        device_map = None
        torch_dtype = torch.float32
        print("    [Config] No GPU available. Defaulting to float32 on CPU (Slow!).")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        device_map=device_map,
        token=hf_token,
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    
    return model, tokenizer


def clean_gpu_memory(model=None, tokenizer=None):
    """
    Performs rigorous memory cleanup to prevent CUDA Out-Of-Memory issues between models.
    """
    print("[-] Cleaning GPU memory...")
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("[✓] Memory cleaned successfully.")


def run_prompt_evaluation(model, tokenizer, prompt_text, args):
    """
    Constructs the prompt, runs inference, measures speed and latency, and decodes the result.
    """
    messages = [{"role": "user", "content": prompt_text}]
    
    # Try using chat template formatting, otherwise fall back to raw prompt text
    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
    except Exception as e:
        # Fallback format: Standard LLM prompt format
        formatted_prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt")
        
    # Ensure input_ids is converted to a 2D PyTorch tensor of token IDs
    # 1. Handle dictionary or BatchEncoding structures
    if isinstance(input_ids, dict) or hasattr(input_ids, "keys"):
        if "input_ids" in input_ids:
            input_ids = input_ids["input_ids"]

    # 2. Handle single tokenizers.Encoding object
    if hasattr(input_ids, "ids"):
        input_ids = input_ids.ids

    # 3. Handle list or tuple structures (which might contain Encoding objects)
    if isinstance(input_ids, (list, tuple)):
        resolved = []
        for item in input_ids:
            if hasattr(item, "ids"):
                resolved.append(item.ids)
            elif isinstance(item, (list, tuple)):
                resolved.append(list(item))
            else:
                try:
                    resolved.append(int(item))
                except:
                    pass
        input_ids = resolved

    # 4. Convert to PyTorch Tensor
    if not isinstance(input_ids, torch.Tensor):
        if isinstance(input_ids, list) and len(input_ids) > 0 and isinstance(input_ids[0], list):
            input_ids = torch.tensor(input_ids)
        else:
            input_ids = torch.tensor([input_ids])
    else:
        input_ids = input_ids.long()

    # 5. Ensure 2D shape (batch_size, sequence_length)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    # Relocate input tensors to model's active device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(model, "device"):
        input_ids = input_ids.to(model.device)
    else:
        input_ids = input_ids.to(device)

    # Set parameters for generation
    gen_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0.0,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0.0:
        gen_config["temperature"] = args.temperature
        gen_config["top_p"] = args.top_p

    # Perform inference and record latency
    start_time = time.time()
    with torch.no_grad():
        output_tokens = model.generate(input_ids, **gen_config)
    end_time = time.time()
    
    latency = end_time - start_time
    
    # Separate generated tokens from the prompt context
    input_length = input_ids.shape[1]
    new_tokens = output_tokens[0][input_length:]
    
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    num_generated_tokens = len(new_tokens)
    tokens_per_sec = num_generated_tokens / latency if latency > 0 else 0
    
    return {
        "response": response_text,
        "latency_seconds": round(latency, 2),
        "tokens_generated": num_generated_tokens,
        "tokens_per_second": round(tokens_per_sec, 2)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple LLMs on Tunisian Arabic (Derja) on RunPod/GPUs.")
    parser.add_argument(
        "--models",
        type=str,
        default="Qwen/Qwen2-1.5B-Instruct,CohereLabs/c4ai-command-r-v01",
        help="Comma-separated Hugging Face model IDs to test."
    )
    parser.add_argument(
        "--quantize",
        type=int,
        choices=[0, 4, 8],
        default=0,
        help="Quantization level: 4 (4-bit), 8 (8-bit), or 0 (no quantization/16-bit half precision)."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Inference temperature. Use 0 for greedy decoding."
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate per prompt."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results.json",
        help="Filename to save the results in JSON format."
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Path to an optional custom JSON file containing prompts (keys must match DEFAULT_PROMPTS structure)."
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face authorization token. Defaults to HF_TOKEN environment variable."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./hf_cache",
        help="Directory where Hugging Face downloads and caches models. Defaults to './hf_cache' (recommended for RunPod workspace storage)."
    )
    args = parser.parse_args()

    # Determine models list
    models_to_test = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models_to_test:
        print("[!] Error: No models provided to evaluate.")
        sys.exit(1)

    # Login to HF if token is present
    if args.hf_token:
        try:
            from huggingface_hub import login
            login(token=args.hf_token)
            print("[+] Successfully logged in to Hugging Face Hub.")
        except Exception as e:
            print(f"[!] Warning: Failed to log in with token: {e}")
    else:
        print("[!] No Hugging Face token found. Gatekeeper models (e.g. Llama-3) might fail to load.")

    # Load prompts
    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        try:
            with open(args.prompts_file, "r", encoding="utf-8") as f:
                prompts = json.load(f)
            print(f"[+] Loaded custom prompts file: {args.prompts_file} ({len(prompts)} prompts found)")
        except Exception as e:
            print(f"[!] Error loading custom prompts file: {e}. Falling back to default prompts.")

    # Structure to hold all benchmark findings
    results_summary = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "quantization": f"{args.quantize}-bit" if args.quantize > 0 else "16-bit (half)",
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens
        },
        "runs": []
    }

    print("=" * 80)
    print("             TUNISIAN ARABIC (DERJA) LLM BENCHMARK RUN")
    print("=" * 80)
    print(f"Models to test : {models_to_test}")
    print(f"Quantization   : {results_summary['metadata']['quantization']}")
    print(f"Prompts count  : {len(prompts)}")
    print(f"Save Location  : {args.output}")
    print("=" * 80)

    # Main sequential evaluation loop
    for idx, model_id in enumerate(models_to_test):
        print(f"\n[Run {idx+1}/{len(models_to_test)}] Evaluating model: {model_id}")
        
        model = None
        tokenizer = None
        model_run_data = {
            "model_id": model_id,
            "success": False,
            "evaluations": [],
            "error": None
        }

        try:
            # Load
            model, tokenizer = load_model_and_tokenizer(model_id, args.quantize, args.hf_token, args.cache_dir)
            model_run_data["success"] = True
            
            # Execute prompts
            for p_idx, prompt_obj in enumerate(prompts):
                print(f"  [{p_idx+1}/{len(prompts)}] Category: '{prompt_obj['category']}' ({prompt_obj['id']})...")
                
                try:
                    res = run_prompt_evaluation(model, tokenizer, prompt_obj["prompt"], args)
                    print(f"    -> Generated {res['tokens_generated']} tokens in {res['latency_seconds']}s ({res['tokens_per_second']} tok/s)")
                    
                    eval_entry = {
                        "prompt_id": prompt_obj["id"],
                        "category": prompt_obj["category"],
                        "description": prompt_obj["description"],
                        "prompt": prompt_obj["prompt"],
                        "response": res["response"],
                        "latency_seconds": res["latency_seconds"],
                        "tokens_generated": res["tokens_generated"],
                        "tokens_per_second": res["tokens_per_second"],
                        "success": True
                    }
                    model_run_data["evaluations"].append(eval_entry)
                    
                except Exception as eval_err:
                    print(f"    [!] Generation failed for prompt '{prompt_obj['id']}': {repr(eval_err)}")
                    traceback.print_exc()
                    model_run_data["evaluations"].append({
                        "prompt_id": prompt_obj["id"],
                        "category": prompt_obj["category"],
                        "prompt": prompt_obj["prompt"],
                        "error": str(eval_err),
                        "success": False
                    })
                    
        except Exception as load_err:
            print(f"[!] Critical Error loading model {model_id}: {load_err}")
            traceback.print_exc()
            model_run_data["error"] = str(load_err)
            
        finally:
            # Append run results and clear memory immediately
            results_summary["runs"].append(model_run_data)
            clean_gpu_memory(model, tokenizer)

    # Save findings
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results_summary, f, ensure_ascii=False, indent=2)
        print(f"\n[✓] All runs finished. Summary successfully saved to: {args.output}")
        
        # Print a short stdout markdown summary table
        print("\n" + "=" * 80)
        print("                              EVALUATION SUMMARY")
        print("=" * 80)
        print(f"{'Model ID':<45} | {'Tokens/s':<10} | {'Avg Latency (s)':<15}")
        print("-" * 80)
        for run in results_summary["runs"]:
            if not run["success"]:
                print(f"{run['model_id']:<45} | {'FAILED':<10} | {'N/A':<15}")
                continue
            
            valid_tokens_per_s = [e["tokens_per_second"] for e in run["evaluations"] if e.get("success")]
            valid_latency = [e["latency_seconds"] for e in run["evaluations"] if e.get("success")]
            
            avg_tok_s = sum(valid_tokens_per_s) / len(valid_tokens_per_s) if valid_tokens_per_s else 0
            avg_lat = sum(valid_latency) / len(valid_latency) if valid_latency else 0
            
            print(f"{run['model_id']:<45} | {avg_tok_s:<10.2f} | {avg_lat:<15.2f}")
        print("=" * 80 + "\n")

    except Exception as save_err:
        print(f"[!] Failed to save evaluations to {args.output}: {save_err}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
from importlib.machinery import ModuleSpec
from unittest.mock import MagicMock

# Mock module class that returns MagicMock for everything
class MockModule(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

# Custom finder to intercept imports of torchvision and torchaudio and their submodules
class MockFinder:
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("torchvision") or fullname.startswith("torchaudio"):
            return ModuleSpec(fullname, self)
            
    def create_module(self, spec):
        return MockModule()
        
    def exec_module(self, module):
        pass

# Clean up any existing entries in sys.modules and register the finder
for key in list(sys.modules.keys()):
    if key.startswith("torchvision") or key.startswith("torchaudio"):
        del sys.modules[key]

sys.meta_path.insert(0, MockFinder())

# Pre-load nvidia python package shared libraries (like libnvJitLink) to resolve bitsandbytes load errors
import glob
import ctypes
try:
    import os
    import sys
    # Search in all directories in sys.path (which includes site-packages) for nvidia libraries
    for path in sys.path:
        if not path or not os.path.exists(path):
            continue
        for pattern in ["**/libnvJitLink.so*", "**/libcudart.so*", "**/libcublas.so*"]:
            # Search both nested namespace and flat packages
            libs = glob.glob(os.path.join(path, "nvidia", "**", pattern), recursive=True)
            libs += glob.glob(os.path.join(path, "nvidia_*", "**", pattern), recursive=True)
            for lib in libs:
                try:
                    ctypes.CDLL(lib)
                except Exception:
                    pass
except Exception:
    pass

import os
import gc
import sys
import glob
import time
import json
import argparse
import traceback

# Diagnostic check for required packages
MISSING_PACKAGES = []
for pkg in ["torch", "transformers", "faiss", "sentence_transformers", "pypdf"]:
    try:
        if pkg == "faiss":
            import faiss
        elif pkg == "pypdf":
            import pypdf
        else:
            __import__(pkg)
    except ImportError as e:
        print(f"[!] ImportError for package '{pkg}': {e}")
        traceback.print_exc()
        MISSING_PACKAGES.append(pkg)

if MISSING_PACKAGES:
    print("[!] Critical: Missing required packages:")
    for pkg in MISSING_PACKAGES:
        print(f"    - {pkg}")
    print("\nPlease run the following command to install the missing dependencies:")
    print("pip install torch transformers accelerate bitsandbytes faiss-cpu sentence-transformers pypdf")
    sys.exit(1)

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

import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# Try importing langchain text splitter, otherwise fall back to pure-Python implementation
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as e:
        print(f"[!] Warning: LangChain text splitter failed to load ({e}). Using native Python recursive splitter fallback.")
        # Pure-Python implementation of RecursiveCharacterTextSplitter
        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size=1000, chunk_overlap=200, separators=None):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.separators = separators or ["\n\n", "\n", " ", ""]

            def split_text(self, text):
                chunks = []
                def _split(t, seps):
                    if len(t) <= self.chunk_size:
                        return [t]
                    if not seps:
                        return [t[i:i+self.chunk_size] for i in range(0, len(t), self.chunk_size - self.chunk_overlap)]
                    
                    sep = seps[0]
                    next_seps = seps[1:]
                    parts = t.split(sep)
                    current = ""
                    res = []
                    for part in parts:
                        part_with_sep = part + sep if part != parts[-1] else part
                        if len(part_with_sep) > self.chunk_size:
                            if current:
                                res.append(current)
                                current = ""
                            res.extend(_split(part_with_sep, next_seps))
                        else:
                            if len(current) + len(part_with_sep) <= self.chunk_size:
                                current += part_with_sep
                            else:
                                if current:
                                    res.append(current)
                                current = part_with_sep
                    if current:
                        res.append(current)
                    return res
                return _split(text, self.separators)

# Default system instruction engineered for Tunisian Arabic Digital Safety
SYSTEM_INSTRUCTION = """أنت خبير محترف وموثوق في السلامة الرقمية والأمن السيبراني (Digital Safety Expert) لمساعدة المستخدمين التوانسة وحمايتهم من المخاطر الرقمية.
أجب عن سؤال المستخدم بالاعتماد على سياق المعلومات المرفق (Retrieved Context) الذي يحتوي على مستندات أمان رقمي باللغة العربية والانجليزية.

اتبع القواعد التالية بدقة:
1. اللغة: يجب أن تكون الإجابة كاملة بالدارجة التونسية (Derja) بأسلوب دافئ، مبسط، وواضح يفهمه المواطن التونسي العادي. لا تستخدم الفصحى الجافة ولا تنسخ نصوصاً بالإنجليزية.
2. المصطلحات التقنية: يمكنك استخدام المصطلحات التقنية المعروفة (مثل VPN, 2FA, Phishing, Spam, Password Manager, Malware) ولكن يجب شرحها بالدارجة التونسية بطريقة سهلة ومبسطة جداً عند ذكرها لأول مرة.
3. التنسيق والهيكلة: نظّم إجابتك في نقاط واضحة (Bullet points) وخطوات عملية يسهل تطبيقها.
4. قيود السياق:
   - اعتمد أساساً على السياق المرفق للإجابة.
   - إذا لم يحتوي السياق على الإجابة التفصيلية، يمكنك استخدام معلوماتك العامة في السلامة الرقمية للإجابة، ولكن أخبر المستخدم بلطف بالدارجة التونسية أن هذه معلومات عامة لحمايته.
   - إذا كان سؤال المستخدم خارج موضوع السلامة الرقمية والأمن السيبراني (مثلاً أسئلة في الطبخ، السفر، الرياضة)، اعتذر منه بلطف بالدارجة التونسية وقل له أنك مخصص فقط للمساعدة في السلامة الرقمية وحماية الحسابات.
5. النبرة: نبرة ودودة ومطمئنة ولكن عملية ومحترفة في نفس الوقت، لأن قضايا الاختراق والسرقة تسبب القلق للمستخدم."""

USER_PROMPT_TEMPLATE = """السياق المرفق (Retrieved Context):
{context}

سؤال المستخدم: {query}"""


def build_and_save_index(rag_files_dir, index_dir, embed_model):
    """
    Reads PDFs from RAG files directory, chunks them, computes embeddings, and saves index to disk.
    """
    print("=" * 80)
    print("                      BUILDING RAG VECTOR INDEX")
    print("=" * 80)
    
    if not os.path.exists(rag_files_dir):
        print(f"[!] Error: RAG files directory '{rag_files_dir}' does not exist.")
        return None, None
        
    pdf_files = glob.glob(os.path.join(rag_files_dir, "*.pdf"))
    if not pdf_files:
        print(f"[!] Warning: No PDF files found in '{rag_files_dir}'.")
        return None, None
        
    print(f"[+] Found {len(pdf_files)} PDF files to process:")
    for f in pdf_files:
        print(f"    - {os.path.basename(f)}")
        
    documents = []
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"[+] Extracting text from {filename}...")
        try:
            reader = PdfReader(pdf_path)
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    documents.append({
                        "text": text.strip(),
                        "metadata": {
                            "source": filename,
                            "page": page_idx + 1
                        }
                    })
        except Exception as e:
            print(f"    [!] Error reading {filename}: {e}")
            
    if not documents:
        print("[!] Error: No text could be extracted from any PDFs.")
        return None, None
        
    print(f"[+] Total pages extracted: {len(documents)}")
    
    # Chunking
    print("[+] Chunking text using RecursiveCharacterTextSplitter...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = []
    for doc in documents:
        split_texts = text_splitter.split_text(doc["text"])
        for chunk_text in split_texts:
            chunks.append({
                "text": chunk_text,
                "metadata": doc["metadata"]
            })
            
    print(f"[+] Generated {len(chunks)} text chunks.")
    
    # Embedding
    chunk_texts = [c["text"] for c in chunks]
    print("[+] Generating embeddings (this may take a minute)...")
    try:
        embeddings = embed_model.encode(chunk_texts, show_progress_bar=True, normalize_embeddings=True)
    except Exception as e:
        print(f"[!] Warning: Embedding generation failed on default device ({e}). Falling back to CPU...")
        embed_model.to("cpu")
        embeddings = embed_model.encode(chunk_texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings_np = np.array(embeddings).astype('float32')
    
    # FAISS Index
    dimension = embeddings_np.shape[1]
    print(f"[+] Creating FAISS Inner-Product Index (dimension={dimension})...")
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_np)
    
    # Save
    os.makedirs(index_dir, exist_ok=True)
    index_file = os.path.join(index_dir, "faiss_index.bin")
    meta_file = os.path.join(index_dir, "chunks_metadata.json")
    
    print(f"[+] Saving FAISS index to {index_file}...")
    faiss.write_index(index, index_file)
    
    print(f"[+] Saving chunks metadata to {meta_file}...")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
        
    print("[✓] Index building completed successfully.")
    print("=" * 80 + "\n")
    return index, chunks


def load_index(index_dir):
    """
    Loads FAISS index and chunks metadata from disk.
    """
    index_file = os.path.join(index_dir, "faiss_index.bin")
    meta_file = os.path.join(index_dir, "chunks_metadata.json")
    
    if not os.path.exists(index_file) or not os.path.exists(meta_file):
        return None, None
        
    print(f"[+] Loading FAISS index from {index_file}...")
    index = faiss.read_index(index_file)
    
    print(f"[+] Loading chunks metadata from {meta_file}...")
    with open(meta_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    return index, chunks


def retrieve(query, index, chunks, embedding_model, top_k=4):
    """
    Retrieves top_k relevant chunks for the given query.
    """
    try:
        query_vector = embedding_model.encode([query], normalize_embeddings=True).astype('float32')
    except Exception as e:
        print(f"[!] Warning: Query embedding failed ({e}). Falling back to CPU...")
        embedding_model.to("cpu")
        query_vector = embedding_model.encode([query], normalize_embeddings=True).astype('float32')
    distances, indices = index.search(query_vector, top_k)
    
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if 0 <= idx < len(chunks):
            results.append({
                "chunk": chunks[idx],
                "score": float(dist)
            })
    return results


def load_generation_model(model_id, quantization=0, hf_token=None, cache_dir=None):
    """
    Loads Command R model and tokenizer with GPU & quantization configurations.
    """
    print(f"\n[+] Loading model and tokenizer for: {model_id}")
    
    bnb_config = None
    if quantization in [4, "4"]:
        print("    [Config] Loading in 4-bit precision (NF4)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            llm_int8_enable_fp32_cpu_offload=True
        )
    elif quantization in [8, "8"]:
        print("    [Config] Loading in 8-bit precision...")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )

    if torch.cuda.is_available():
        if bnb_config is not None:
            is_large_model = "command-r" in model_id.lower() or "cohere" in model_id.lower() or "c4ai" in model_id.lower()
            if is_large_model:
                print("    [Config] Large model detected. Using custom device_map to offload embed_tokens/lm_head to CPU...")
                device_map = {
                    "model.embed_tokens": "cpu",
                    "model.layers": 0,
                    "model.norm": 0,
                    "model.rotary_emb": 0,
                    "lm_head": "cpu"
                }
            else:
                device_map = {"": 0}
        else:
            device_map = "auto"
        
        if bnb_config is not None:
            torch_dtype = None  # bitsandbytes manages precision internally
        else:
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"    [Config] GPU detected. Using device_map: {device_map}, torch_dtype: {torch_dtype}")
    else:
        device_map = None
        torch_dtype = torch.float32
        print("    [Config] No GPU available. Defaulting to float32 on CPU (Warning: very slow).")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        token=hf_token,
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    
    return model, tokenizer


def clean_gpu_memory(model=None, tokenizer=None):
    """
    Frees up memory.
    """
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def run_rag_query(query, index, chunks, embed_model, model, tokenizer, args):
    """
    Performs retrieval, prompts Command R, and gets the Tunisian Arabic RAG response.
    """
    # 1. Retrieve
    print(f"\n[+] Searching vector index for: '{query}'...")
    retrieved_items = retrieve(query, index, chunks, embed_model, top_k=args.top_k)
    
    print(f"[+] Found {len(retrieved_items)} relevant documents:")
    context_str = ""
    for idx, item in enumerate(retrieved_items):
        source = item["chunk"]["metadata"]["source"]
        page = item["chunk"]["metadata"]["page"]
        score = item["score"]
        print(f"    {idx+1}. [{source}] Page {page} (Similarity Score: {score:.4f})")
        context_str += f"--- Document [{idx+1}]: {source} (Page {page}) ---\n{item['chunk']['text']}\n\n"
        
    # 2. Formulate RAG Prompts
    formatted_user_prompt = USER_PROMPT_TEMPLATE.format(context=context_str, query=query)
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": formatted_user_prompt}
    ]
    
    # Format message inputs using chat template
    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
    except Exception as e:
        print(f"[!] Warning: Tokenizer chat template failed: {e}. Falling back to formatted string.")
        raw_prompt = f"<|im_start|>system\n{SYSTEM_INSTRUCTION}<|im_end|>\n<|im_start|>user\n{formatted_user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer.encode(raw_prompt, return_tensors="pt")

    # Coerce input to proper device-placed PyTorch tensor
    if isinstance(input_ids, dict) or hasattr(input_ids, "keys"):
        if "input_ids" in input_ids:
            input_ids = input_ids["input_ids"]
    if hasattr(input_ids, "ids"):
        input_ids = input_ids.ids
    if not isinstance(input_ids, torch.Tensor):
        if isinstance(input_ids, list) and len(input_ids) > 0 and isinstance(input_ids[0], list):
            input_ids = torch.tensor(input_ids)
        else:
            input_ids = torch.tensor([input_ids])
    else:
        input_ids = input_ids.long()
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_ids = input_ids.to(model.device if hasattr(model, "device") else device)

    # 3. Generate
    gen_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0.0,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0.0:
        gen_config["temperature"] = args.temperature
        gen_config["top_p"] = args.top_p

    print("[+] Clearing GPU cache and running garbage collection...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("[+] Generating response from Command R...")
    start_time = time.time()
    with torch.no_grad():
        output_tokens = model.generate(input_ids, **gen_config)
    latency = time.time() - start_time
    
    input_length = input_ids.shape[1]
    new_tokens = output_tokens[0][input_length:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    print("\n" + "=" * 80)
    print("                              COMMAND R RESPONSE")
    print("=" * 80)
    print(response_text)
    print("=" * 80)
    print(f"Generated {len(new_tokens)} tokens in {latency:.2f} seconds ({len(new_tokens)/latency:.2f} tok/s)\n")


def main():
    parser = argparse.ArgumentParser(description="Tunisian Arabic Digital Safety RAG Assistant.")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="A digital safety query to answer in Tunisian Arabic."
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Force rebuilding the vector index from PDFs in RAG directory."
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run an interactive session where you can input multiple queries."
    )
    parser.add_argument(
        "--rag-files-dir",
        type=str,
        default="RAG files",
        help="Path to directory containing PDF files for context."
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="rag_index",
        help="Directory to save/load the built FAISS index and chunk metadata."
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="HuggingFace model ID for multilingual embeddings."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="CohereLabs/c4ai-command-r-v01",
        help="Cohere Command R model ID on Hugging Face."
    )
    parser.add_argument(
        "--quantize",
        type=int,
        choices=[0, 4, 8],
        default=4,
        help="Quantization level for loading Command R: 4 (4-bit), 8 (8-bit), or 0 (no quantization/16-bit)."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Inference temperature."
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
        default=512,
        help="Maximum generated new tokens."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Number of retrieved text chunks to feed to the model."
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face authorization token. Defaults to HF_TOKEN env variable."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./hf_cache",
        help="HF downloads cache directory."
    )
    args = parser.parse_args()

    # Login to HF if token is present
    if args.hf_token:
        try:
            from huggingface_hub import login
            login(token=args.hf_token)
            print("[+] Successfully logged in to Hugging Face Hub.")
        except Exception as e:
            print(f"[!] Warning: Failed to log in with token: {e}")

    # Step 1: Initialize Embedding Model on CPU (saves GPU VRAM for the 35B generation model)
    print(f"[+] Loading embedding model for RAG (CPU-enforced): {args.embedding_model}")
    embed_model = SentenceTransformer(args.embedding_model, device="cpu")

    # Step 2: Ensure Index is Loaded or Built
    index, chunks = None, None
    if not args.build_index:
        index, chunks = load_index(args.index_dir)
        if index is not None:
            print(f"[✓] Vector index loaded successfully from {args.index_dir}.")
        else:
            print("[!] Vector index files not found. Initiating index build.")
            
    if index is None or args.build_index:
        index, chunks = build_and_save_index(args.rag_files_dir, args.index_dir, embed_model)
        if index is None:
            print("[!] Critical: Failed to load or build vector index. Exiting.")
            sys.exit(1)

    # If building index only, we exit here
    if args.build_index and not args.query and not args.interactive:
        print("[✓] Index built successfully. Exiting since no query/interactive flag was set.")
        sys.exit(0)

    # Step 3: Load Command R Generation Model
    model, tokenizer = None, None
    try:
        model, tokenizer = load_generation_model(
            args.model_id,
            quantization=args.quantize,
            hf_token=args.hf_token,
            cache_dir=args.cache_dir
        )
    except Exception as e:
        print(f"[!] Critical Error loading Command R model: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Step 4: Handle query or interactive mode
    try:
        if args.interactive:
            print("\n" + "=" * 80)
            print("         TUNISIAN ARABIC DIGITAL SAFETY ASSISTANT - INTERACTIVE CHAT")
            print("=" * 80)
            print("Type 'exit' or 'quit' to close the assistant.\n")
            
            while True:
                try:
                    user_query = input("سؤالك (Your question): ").strip()
                    if not user_query:
                        continue
                    if user_query.lower() in ["exit", "quit"]:
                        print("بسلامة! (Goodbye!)")
                        break
                    run_rag_query(user_query, index, chunks, embed_model, model, tokenizer, args)
                except KeyboardInterrupt:
                    print("\nبسلامة! (Goodbye!)")
                    break
                except Exception as e:
                    print(f"[!] Error processing query: {e}")
                    traceback.print_exc()
        elif args.query:
            run_rag_query(args.query, index, chunks, embed_model, model, tokenizer, args)
        else:
            # Fallback to default query if none is provided
            default_query = "كيفاش نحمي تلفوني وحساباتي من الإختراق والتجسس؟"
            print(f"[+] No query provided. Running default query: '{default_query}'")
            run_rag_query(default_query, index, chunks, embed_model, model, tokenizer, args)
            
    finally:
        clean_gpu_memory(model, tokenizer)
        print("[+] Finished execution.")


if __name__ == "__main__":
    main()

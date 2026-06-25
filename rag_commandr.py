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

import glob
import ctypes
try:
    import os
    import sys
    for path in sys.path:
        if not path or not os.path.exists(path):
            continue
        for pattern in ["**/libnvJitLink.so*", "**/libcudart.so*", "**/libcublas.so*"]:
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
from threading import Thread

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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
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
1. قواعد السلامة الرقمية ومكافحة الابتزاز (هام جداً):
   - تحذير حرج: لا تنصح المستخدم أبداً بأفعال قد تعرض سلامته أو خصوصيته للخطر.
   - يمنع منعاً باتاً اقتراح ردود فعل علنية أو تنظيم حملات مضادة على منصات التواصل الاجتماعي، أو محاولة التواصل مع المبتز/المهدد والتفاوض معه.
   - التزم بالخطوات القياسية والآمنة للتعامل مع الابتزاز والعنف الرقمي:
     1. المحافظة على الهدوء وعدم الاستسلام أو التجاوب مع التهديدات (عدم إرسال أموال أو صور إضافية).
     2. جمع وتوثيق الأدلة فوراً وبدقة (أخذ لقطات شاشة واضحة للمحادثات، الحسابات، الروابط، وأرقام الهواتف) دون حذفها، وحفظها في مكان آمن.
     3. قطع الاتصال تماماً بالجهة المبتزة (حظر الحسابات Block) والإبلاغ عنها داخل المنصة الرقمية (Report).
     4. التوجه الفوري إلى السلطات الأمنية المختصة (الإدارة الفرعية للوقاية الاجتماعية أو إدارة الشرطة العدلية بتونس، أو القضاء) وتقديم شكاية رسمية استناداً للقوانين التونسية (مثل القانون الأساسي عدد 58 لسنة 2017 المتعلق بالقضاء على العنف ضد المرأة)، والاتصال بمراكز الدعم أو الجمعيات الحقوقية (مثل جمعية النساء الديمقراطيات أو الكريديف على الرقم الأخضر 1805).

2. اللغة والأسلوب (دارجة تونسية قحّة):
   - يجب أن تكون الإجابة كاملة بالدارجة التونسية (Derja) بأسلوب مبسط، واضح، ومحترف للغاية.
   - تنبيه هام جداً: لا تكتب أبداً باللهجة المصرية أو الخليجية (تجنب تماماً كلمات مثل: "عايز"، "كده"، "دي"، "إيه"، "علشان"، "بتاع"، "شوية"، "هكذا"، "دون"، "نقدر").
   - تجنب تماماً الكلمات المبتذلة أو غير المهنية مثل "صديقي" أو "يا باهي" أو "friend" أو "عزيزي". كن رصيناً ومحترفاً في كلامك.

3. التفاعل والأسئلة التوجيهية (هام جداً):
   - في نهاية إجابتك، اطرح سؤالاً توجيهياً أو سؤال متابعة (Follow-up question) واحد بذكاء لمساعدة المستخدم على توضيح مشكلته أكثر (مثال: هل ما زلت تدخل لحسابك؟ هل فما شكون يبتز فيك؟).
   - كن تفاعلياً ومشاركاً وحاول فهم تفاصيل التهديد لتقديم حل دقيق ومخصص.

4. توثيق المراجع والملفات (هام جداً):
   - يجب عليك توثيق مصدر كل معلومة تذكرها من السياق المرفق.
   - اكتب اسم المصدر ورقم الصفحة في نهاية الجملة مباشرة مستخدماً الصيغة التالية حرفياً: [المصدر: اسم_المصدر، صفحة X]
   - تنبيه هام جداً: يجب عليك كتابة الاسم الكامل للمصدر كما هو مبيّن في حقل "المصدر" في السياق المرفق حرفياً دون أي تغيير أو اختصار (مثال: اكتب الاسم الكامل للمصدر بالكامل، ولا تختصره أبداً إلى كلمة واحدة مثل "منصري" أو "الحاج").
   - خذ اسم المصدر ورقم الصفحة بدقة من البيانات المرافقة لكل مستند في السياق. لا تخترع أسماء ملفات أبداً.

5. المصطلحات التقنية: استخدم المصطلحات التقنية المعروفة (مثل VPN, 2FA, Password Manager) مع شرحها بالدارجة التونسية بطريقة مبسطة جداً عند ذكرها لأول مرة.

6. الهيكلة والتركيز الشديد (منعاً لانقطاع الإجابة):
   - قدّم الإجابة في نقاط واضحة (Bullet points) لخطوات عملية متسلسلة.
   - اقتصر على 3 نقاط فقط كحد أقصى، واجعل كل نقطة قصيرة جداً (لا تتجاوز جملة واحدة بسيطة ومباشرة).
   - لا تكتب مقدمات طويلة أو خاتمة مكررة لضمان سرعة توليد الإجابة ومنع انقطاع الاتصال.

7. النبرة والأداء العالي: نبرة مطمئنة، محترفة وموثوقة، خالية من العاطفة الزائدة، ولكن فيها تعاطف ودعم حقيقي للمستخدم الذي يمر بظرف صعب رقمياً (مثل حالات الابتزاز أو العنف الرقمي). ركّز على النقاط الأساسية وتجنب الشروح الطويلة غير الضرورية لضمان سرعة توليد الإجابة. """

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


def is_index_up_to_date(rag_files_dir, index_dir):
    """
    Checks if index files exist and are fully synchronized with the PDF files
    in the RAG directory. Returns True if up to date, False otherwise.
    """
    index_file = os.path.join(index_dir, "faiss_index.bin")
    meta_file = os.path.join(index_dir, "chunks_metadata.json")
    
    if not os.path.exists(index_file) or not os.path.exists(meta_file):
        print("[!] Index files are missing.")
        return False
        
    if not os.path.exists(rag_files_dir):
        print(f"[!] RAG directory '{rag_files_dir}' does not exist.")
        return True # Nothing to index anyway
        
    pdf_files = glob.glob(os.path.join(rag_files_dir, "*.pdf"))
    pdf_filenames = {os.path.basename(f) for f in pdf_files}
    
    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        indexed_files = {c.get("metadata", {}).get("source") for c in chunks if "metadata" in c}
        indexed_files = {f for f in indexed_files if f}
    except Exception as e:
        print(f"[!] Failed to read index metadata: {e}")
        return False
        
    if pdf_filenames != indexed_files:
        print(f"[!] Index mismatch: PDFs in folder ({pdf_filenames}) vs indexed PDFs ({indexed_files})")
        return False
        
    # Check if any PDF file was modified after index was created
    index_mtime = os.path.getmtime(index_file)
    for pdf_path in pdf_files:
        if os.path.getmtime(pdf_path) > index_mtime:
            print(f"[!] PDF modified since last index build: {os.path.basename(pdf_path)}")
            return False
            
    return True


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

def get_clean_source_name(filename):
    """
    Maps raw PDF filenames to professional, human-readable document titles.
    """
    mapping = {
        "Digital-Safety-Guide-JOSA-FB-Arabic.pdf": "دليل السلامة الرقمية (JOSA & Facebook)",
        "Digital-Safety-Toolkit-Nevada.pdf": "دليل السلامة الرقمية (NCEDSV)",
        "Digital-Security-Arabic.pdf": "تدريب الأمن السيبراني الرقمي (UNPO Academy)",
        "EU-WP2016 2-3 1 Cyber Hygiene.pdf": "مراجعة ممارسات النظافة السيبرانية (ENISA)",
        "MENA-PSS-Manual-English.pdf": "دليل الدعم النفسي والاجتماعي للناجيات من العنف الرقمي (SecDev)",
        "Tunisia-DVAW-2021-legal-dimensions-AR.pdf": "دراسة الكريديف حول واقع النصوص القانونية والعنف الرقمي - أنوار منصري (2021)",
        "Tunisia-DVAW-2021-women-in-journalism-AR.pdf": "دراسة الكريديف حول العنف الرقمي ضد الصحفيات التونسيات - هدى الحاج قاسم (2021)",
        "MENA-Tunisia-DVAW-2022-CREDIF-Review-Issue53-ARFR.pdf": "مجلة الكريديف عدد 53: العنف الرقمي ضد النساء والفتيات (2022)"
    }
    return mapping.get(filename, filename.replace(".pdf", "").replace("-", " ").replace("_", " "))


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
                # Load the entire model on GPU 0 first, and manually offload embed_tokens and lm_head to CPU
                print("    [Config] Large model detected. Loading on GPU 0 and applying manual CPU offloading...")
                device_map = {"": 0}
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

    # Apply manual CPU offload to save VRAM and avoid meta-tensor/offload bugs
    if torch.cuda.is_available() and bnb_config is not None:
        is_large_model = "command-r" in model_id.lower() or "cohere" in model_id.lower() or "c4ai" in model_id.lower()
        if is_large_model:
            if hasattr(model, "model") and hasattr(model.model, "embed_tokens") and hasattr(model, "lm_head"):
                print("    [Config] Moving embed_tokens and lm_head to CPU manually to prevent OOM...")
                # Remove accelerate's automatic device-alignment hooks so they don't interfere with our manual CPU placement
                try:
                    from accelerate.hooks import remove_hook_from_module
                    remove_hook_from_module(model.model.embed_tokens)
                    remove_hook_from_module(model.lm_head)
                    print("    [Config] Successfully removed accelerate hooks from embed_tokens and lm_head.")
                except Exception as hook_err:
                    print(f"    [Warning] Could not remove accelerate hooks: {hook_err}")
                
                target_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
                
                # Move to CPU first, then cast to target_dtype on CPU to avoid GPU allocation OOM
                model.model.embed_tokens = model.model.embed_tokens.cpu().to(target_dtype)
                model.lm_head = model.lm_head.cpu().to(target_dtype)
                
                # Register hooks for device and precision transfer
                model.model.embed_tokens.register_forward_pre_hook(
                    lambda module, inputs: (inputs[0].to("cpu"),)
                )
                model.model.embed_tokens.register_forward_hook(
                    lambda module, inputs, outputs: outputs.to("cuda:0", dtype=target_dtype)
                )
                model.lm_head.register_forward_pre_hook(
                    lambda module, inputs: (inputs[0].to("cpu", dtype=target_dtype),)
                )
                model.lm_head.register_forward_hook(
                    lambda module, inputs, outputs: outputs.to("cuda:0", dtype=target_dtype)
                )
                
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
    
 
    if torch.cuda.is_available():
        try:
            type(model).device = property(lambda self: torch.device("cuda:0"))
        except Exception:
            pass

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
        clean_source = get_clean_source_name(source)
        page = item["chunk"]["metadata"]["page"]
        score = item["score"]
        print(f"    {idx+1}. [{source}] Page {page} (Similarity Score: {score:.4f})")
        context_str += f"المستند [{idx+1}]:\nالمصدر: {clean_source}\nالصفحة: {page}\nالنص:\n{item['chunk']['text']}\n\n"
        
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
        print(f"[!] Warning: Tokenizer chat template failed: {e}. Falling back to Cohere formatted string.")
        raw_prompt = f"<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{SYSTEM_INSTRUCTION}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{formatted_user_prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
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
    # Even if model.device returns CPU (due to CPU offloaded embeddings),
    # the main model inputs must be on the GPU to ensure that internal generation tensors
    # (such as position_ids and attention_mask) are created on the GPU, matching the decoder layers.
    input_ids = input_ids.to("cuda:0" if torch.cuda.is_available() else device)

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


def run_server(args, model, tokenizer, embed_model, index, chunks):
    """
    Runs a FastAPI web server to expose the RAG generation API.
    """
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel
        from typing import List, Dict, Any
        import uvicorn
    except ImportError:
        print("[!] Critical: FastAPI and Uvicorn are required for server mode.")
        print("Please run: pip install fastapi uvicorn")
        sys.exit(1)

    app = FastAPI(title="TanitBot RAG API Server")

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        messages: List[ChatMessage]
        lang: str = "ar"

    @app.post("/api/chat")
    async def chat_endpoint(request: ChatRequest):
        # Find the last user query
        user_messages = [m for m in request.messages if m.role == "user"]
        if not user_messages:
            query = ""
        else:
            query = user_messages[-1].content

        print(f"[Server] Received chat query: '{query}'")

        # 1. Retrieve context
        print(f"[Server] Searching vector index (top_k={args.top_k})...")
        retrieved_items = retrieve(query, index, chunks, embed_model, top_k=args.top_k)
        
        context_str = ""
        for idx, item in enumerate(retrieved_items):
            source = item["chunk"]["metadata"]["source"]
            clean_source = get_clean_source_name(source)
            page = item["chunk"]["metadata"]["page"]
            context_str += f"المستند [{idx+1}]:\nالمصدر: {clean_source}\nالصفحة: {page}\nالنص:\n{item['chunk']['text']}\n\n"

        # 2. Formulate prompts
        formatted_user_prompt = USER_PROMPT_TEMPLATE.format(context=context_str, query=query)
        
        # Build prompt messages array containing recent conversation history
        messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
        
        # Limit history to the last 8 messages (4 turns) to bound context size and prevent GPU memory spikes
        recent_messages = request.messages[-8:] if request.messages else []
        
        # Append previous messages as-is
        for msg in recent_messages[:-1]:
            messages.append({"role": msg.role, "content": msg.content})
            
        # Inject the retrieved context into the last user message
        if recent_messages:
            last_msg = recent_messages[-1]
            if last_msg.role == "user":
                messages.append({"role": "user", "content": formatted_user_prompt})
            else:
                messages.append({"role": last_msg.role, "content": last_msg.content})
        else:
            messages.append({"role": "user", "content": formatted_user_prompt})

        # Format using chat template
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
        except Exception as e:
            print(f"[Server] Tokenizer apply_chat_template failed: {e}. Falling back to Cohere formatted text.")
            # Build conversation history in raw Cohere format
            raw_prompt = f"<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{SYSTEM_INSTRUCTION}<|END_OF_TURN_TOKEN|>"
            for msg in recent_messages[:-1]:
                role_token = "USER_TOKEN" if msg.role == "user" else "CHATBOT_TOKEN"
                raw_prompt += f"<|START_OF_TURN_TOKEN|><|{role_token}|>{msg.content}<|END_OF_TURN_TOKEN|>"
            raw_prompt += f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{formatted_user_prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
            input_ids = tokenizer.encode(raw_prompt, return_tensors="pt")

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
        # Force placement onto GPU 0 for inference if available to match RoPE/model weights
        input_ids = input_ids.to("cuda:0" if torch.cuda.is_available() else device)

        # 3. Stream Response
        def generate_stream():
            nonlocal input_ids
            # Clear cache before generation
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            from transformers import StoppingCriteria, StoppingCriteriaList
            import threading

            class AbortStoppingCriteria(StoppingCriteria):
                def __init__(self, stop_flag):
                    self.stop_flag = stop_flag
                def __call__(self, input_ids, scores, **kwargs):
                    return self.stop_flag.is_set()

            stop_event = threading.Event()
            stopping_criteria = StoppingCriteriaList([AbortStoppingCriteria(stop_event)])

            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            gen_config = {
                "inputs": input_ids,
                "streamer": streamer,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.temperature > 0.0,
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "stopping_criteria": stopping_criteria,
            }
            if args.temperature > 0.0:
                gen_config["temperature"] = args.temperature
                gen_config["top_p"] = args.top_p

            # PyTorch's gradient tracking context is thread-local. Spawning a new thread
            # runs without the main thread's torch.no_grad() context, building a computation
            # graph and leaking GPU memory. We wrap the generation explicitly in torch.no_grad().
            def thread_target():
                with torch.no_grad():
                    try:
                        model.generate(**gen_config)
                    except Exception as thread_err:
                        print(f"[Server Thread Error] Exception during generation: {thread_err}")
                        traceback.print_exc()

            # Start generation thread
            t = Thread(target=thread_target)
            t.start()

            try:
                for text_chunk in streamer:
                    print(text_chunk, end="", flush=True)
                    yield text_chunk
            finally:
                print("\n[Server Stream] Generation finished or connection closed.")
                # Clean up references and clear CUDA memory after generation completes
                # to prevent memory fragmentation and leaks on subsequent requests.
                stop_event.set()
                t.join(timeout=2.0)
                input_ids = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )

    # Add health check
    @app.get("/health")
    def health():
        return {"status": "healthy", "model": args.model_id}

    # Add API status endpoint to handle external proxy/orchestrator health checks
    @app.get("/api/status")
    @app.get("/status")
    def api_status(token: str = None):
        return {"status": "healthy", "model": args.model_id}

    # Add index info diagnostics
    @app.get("/index-info")
    @app.get("/api/index-info")
    def get_index_info():
        file_counts = {}
        for chunk in chunks:
            src = chunk.get("metadata", {}).get("source", "unknown")
            file_counts[src] = file_counts.get(src, 0) + 1
        return {
            "total_chunks": len(chunks),
            "files": file_counts
        }

    print(f"[Server] Starting FastAPI server on {args.host}:{args.port}...")
    uvicorn.run(app, host=args.host, port=args.port)


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
        default=1024,
        help="Maximum generated new tokens."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
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
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run RAG assistant as a FastAPI web server."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host bind address."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port."
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
    should_build = args.build_index
    if not should_build:
        if is_index_up_to_date(args.rag_files_dir, args.index_dir):
            index, chunks = load_index(args.index_dir)
            if index is not None:
                print(f"[✓] Vector index loaded successfully from {args.index_dir}.")
            else:
                should_build = True
        else:
            print("[!] Vector index is missing, out-of-date, or mismatched. Automatically rebuilding index...")
            should_build = True
            
    if should_build:
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

    # Step 4: Handle server, query or interactive mode
    try:
        if args.server:
            run_server(args, model, tokenizer, embed_model, index, chunks)
        elif args.interactive:
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

import os
import time
import requests
from tqdm import tqdm
from llama_cpp import Llama  # ✅ Sửa: dùng llama.cpp thay vì ctransformers

MODEL_URL = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

def download_model():
    os.makedirs("models", exist_ok=True)
    if os.path.exists(MODEL_PATH):
        print(f"📦 Mô hình đã tồn tại: {MODEL_PATH}")
        return

    print(f"⬇️ Đang tải mô hình từ Hugging Face: {MODEL_URL}")
    with requests.get(MODEL_URL, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("Content-Length", 0))

        with open(MODEL_PATH, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc="📦 Tải về"
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    print(f"✅ Đã tải xong mô hình: {MODEL_PATH}")

def load_llm():
    download_model()
    print(f"\n🧠 Đang load mô hình GGUF từ: {MODEL_PATH}")
    start = time.time()

    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=os.cpu_count(),
        n_gpu_layers=-1,       # ✅ Sử dụng full GPU trên Mac (Metal)
        use_mlock=True,
        verbose=False
    )

    elapsed = time.time() - start
    print(f"✅ Đã load mô hình trên GPU (Metal) - Thời gian: {elapsed:.2f} giây")
    print(f"📏 Context length hỗ trợ: {llm.n_ctx} tokens")

    return llm

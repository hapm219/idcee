import os
import time
import sys
import requests
import contextlib
import gc
from tqdm import tqdm
from llama_cpp import Llama  # ✅ Dùng llama.cpp thay vì ctransformers

MODEL_URL = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

def download_model():
    os.makedirs("models", exist_ok=True)
    if os.path.exists(MODEL_PATH):
        # print(f"📦 Mô hình đã tồn tại: {MODEL_PATH}")  # Ẩn log nếu cần
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

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = fnull
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def load_llm():
    download_model()
    print(f"\n🧠 Sẵn sàng tạo mô hình llama.cpp từ: {MODEL_PATH}")

    def safe_infer(prompt, max_tokens=256, **kwargs):  # ✅ Hỗ trợ stream & args khác
        with suppress_output():
            llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=1024,               # ✅ Giới hạn context để tránh lỗi Metal
                n_threads=os.cpu_count(),
                n_gpu_layers=32,         # ✅ Khuyến nghị cho Mac M4
                use_mlock=True,
                verbose=False
            )
        result = llm(prompt, max_tokens=max_tokens, **kwargs)
        del llm
        gc.collect()
        return result

    return safe_infer

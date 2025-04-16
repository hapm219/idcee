import os
import time
import requests
from tqdm import tqdm
from ctransformers import AutoModelForCausalLM

MODEL_URL = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

def download_model():
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print(f"⬇️ Đang tải mô hình từ Hugging Face: {MODEL_URL}")
        r = requests.get(MODEL_URL, stream=True)
        r.raise_for_status()
        total_size = int(r.headers.get("Content-Length", 0))

        with open(MODEL_PATH, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc="📦 Tải về"
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        print(f"✅ Đã tải xong mô hình: {MODEL_PATH}")
    else:
        print(f"📦 Mô hình đã tồn tại: {MODEL_PATH}")

def load_llm():
    download_model()
    print(f"\n🧠 Đang load mô hình GGUF từ: {MODEL_PATH}")
    start = time.time()

    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        model_type="mistral",
        gpu_layers=40,               # ✅ Ưu tiên dùng CPU
        max_new_tokens=256,
        context_length=2048  # ✅ ép sử dụng đúng context từ file
    )

    elapsed = time.time() - start
    device = "GPU" if getattr(llm.config, "gpu_layers", 0) > 0 else "CPU"
    context_len = getattr(llm, "context_length", "Không xác định")

    print(f"✅ Đã load mô hình trên {device} - Thời gian: {elapsed:.2f} giây")
    print(f"📏 Context length hỗ trợ: {context_len} tokens")

    return llm

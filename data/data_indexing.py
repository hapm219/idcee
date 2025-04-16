import os
import faiss
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

REFINED_DIR = Path("refined")
INDEX_DIR = Path("index")
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_LOCAL_PATH = Path("encoder/bge-m3")

processed_files = 0
all_txt_files = 0
written_dirs = set()

def load_or_download_model():
    if not EMBEDDING_LOCAL_PATH.exists():
        print(f"⬇️ Đang tải mô hình embedding: {EMBEDDING_MODEL_NAME}")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        model.save(str(EMBEDDING_LOCAL_PATH))
    else:
        print(f"📥 Đang load mô hình từ thư mục cục bộ: {EMBEDDING_LOCAL_PATH}")
        model = SentenceTransformer(str(EMBEDDING_LOCAL_PATH))
    return model

def process_text_file(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    chunks = text.split("\n\n")
    return [c.strip() for c in chunks if len(c.strip()) > 20]

def build_faiss_index_for_folder(folder: Path, model: SentenceTransformer):
    global processed_files, written_dirs
    all_texts = []
    all_embeddings = []

    for txt_file in folder.glob("*.txt"):
        texts = process_text_file(txt_file)
        if not texts:
            continue

        prompt_texts = [
            f"Represent this sentence for searching relevant passages: {t}"
            for t in texts
        ]
        embeddings = model.encode(prompt_texts, show_progress_bar=False, convert_to_numpy=True)

        all_texts.extend(texts)
        all_embeddings.append(embeddings)
        processed_files += 1

    if not all_embeddings:
        print(f"⚠️  Không có đoạn văn nào trong {folder}")
        return

    all_embeddings = np.vstack(all_embeddings)

    index = faiss.IndexFlatL2(all_embeddings.shape[1])
    index.add(all_embeddings)

    rel_path = folder.relative_to(REFINED_DIR)
    out_dir = INDEX_DIR / rel_path
    out_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out_dir / "index.faiss"))
    with open(out_dir / "mapping.json", "w", encoding="utf-8") as f:
        json.dump(all_texts, f, ensure_ascii=False, indent=2)

    written_dirs.add(str(out_dir))
    print(f"✅ Indexed {len(all_texts)} đoạn văn → {out_dir}/index.faiss")

def scan_and_index_all():
    global all_txt_files
    model = load_or_download_model()

    folders = [f for f in REFINED_DIR.glob("**/") if any(f.glob("*.txt"))]
    all_txt_files = sum(len(list(f.glob("*.txt"))) for f in folders)

    for folder in folders:
        build_faiss_index_for_folder(folder, model)

    print("\n📊 Tổng kết indexing:")
    print(f"🔍 Số file .txt tìm thấy: {all_txt_files}")
    print(f"✍️  Số file đã xử lý: {processed_files}")
    print("📂 Ghi FAISS index vào các thư mục:")
    for d in sorted(written_dirs):
        print(f"   - {d}")

if __name__ == "__main__":
    scan_and_index_all()

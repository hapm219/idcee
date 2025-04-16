import time
from load_model import load_llm
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from pathlib import Path
import glob

TOP_K = 1
INDEX_DIR = Path("data/index")
EMBEDDING_PATH = "data/encoder/bge-m3"

def load_embedding_model():
    print("📥 Đang tải mô hình embedding bge-m3...")
    return SentenceTransformer(EMBEDDING_PATH)

def load_all_indexes():
    all_indexes = []
    all_mappings = []
    for faiss_path in glob.glob(str(INDEX_DIR / "**/index.faiss"), recursive=True):
        try:
            index = faiss.read_index(faiss_path)
            mapping_file = Path(faiss_path).parent / "mapping.json"
            if not mapping_file.exists():
                continue
            with open(mapping_file, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            all_indexes.append(index)
            all_mappings.append(mapping)
            print(f"✅ Loaded index: {faiss_path}")
        except Exception as e:
            print(f"❌ Lỗi khi load {faiss_path}: {e}")
    return all_indexes, all_mappings

def search_similar_chunks(query, model, all_indexes, all_mappings, top_k=TOP_K):
    query_emb = model.encode(
        f"Represent this sentence for searching relevant passages: {query}",
        convert_to_numpy=True
    )

    results = []
    for index, texts in zip(all_indexes, all_mappings):
        D, I = index.search(np.array([query_emb]), top_k)
        for score, idx in zip(D[0], I[0]):
            if 0 <= idx < len(texts):
                results.append((score, texts[idx]))

    results = sorted(results, key=lambda x: x[0])[:top_k]
    return [r[1] for r in results]

def limit_context(chunks, max_chars=800):
    context = ""
    for c in chunks:
        if len(context) + len(c) > max_chars:
            break
        context += c + "\n\n"
    return context.strip()

def main():
    llm = load_llm()
    embed_model = load_embedding_model()
    all_indexes, all_mappings = load_all_indexes()

    print("🤖 IDCee sẵn sàng! Gõ 'exit' để thoát.\n")

    while True:
        user_input = input("🧑 You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break

        retrieved_chunks = search_similar_chunks(user_input, embed_model, all_indexes, all_mappings)
        context = limit_context(retrieved_chunks)

        prompt = f"""Bạn là trợ lý AI IDCee. Trả lời ngắn gọn, bằng tiếng Việt và chỉ sử dụng thông tin từ phần Thông tin nội bộ.

👉 Nếu không có thông tin phù hợp, trả lời duy nhất câu sau:
"Tôi không tìm thấy thông tin trong tài liệu nội bộ để trả lời câu hỏi này."

❗ Không được bịa, suy đoán hoặc thêm nội dung ngoài dữ liệu cung cấp. Chỉ sử dụng nội dung chứa từ khóa: "{query}"

### Câu hỏi:
{query}

### Thông tin nội bộ:
{context}

### Trả lời:
"""

        start = time.time()
        try:
            response = llm(prompt, max_new_tokens=192, stop=["###", "Câu hỏi:"])
            elapsed = time.time() - start

            if not response or str(response).strip() == "":
                print("🤖 IDCee: (Không thể tạo câu trả lời)")
            else:
                print(f"🤖 IDCee: {response.strip()}")
            print(f"⏱️ Thời gian phản hồi: {elapsed:.2f} giây\n")

        except Exception as e:
            print(f"❌ Lỗi infer: {e}")
            continue

if __name__ == "__main__":
    main()

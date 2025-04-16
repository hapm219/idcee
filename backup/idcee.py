import time
from load_model import load_llm

def main():
    llm = load_llm()
    print("🤖 IDCee sẵn sàng! Gõ 'exit' để thoát.\n")

    while True:
        user_input = input("🧑 You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break

        # Prompt template định hướng rõ ràng
        prompt = f"""Bạn là trợ lý AI tên IDCee. Hãy trả lời bằng tiếng Việt, ngắn gọn, chính xác, dễ hiểu.
Nếu câu trả lời có nhiều ý, hãy trình bày bằng danh sách gạch đầu dòng.

Câu hỏi: {user_input}
Trả lời:"""

        start = time.time()
        try:
            response = llm(prompt, max_new_tokens=1024)
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

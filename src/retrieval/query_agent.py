"""
=============================================================================
QUERY AGENT (LLM ROUTER)
=============================================================================
Description:
Acts as the "Brain" before the search engine. It uses Google Gemini to read 
natural language queries, remove noise, split the intent into modalities 
(Visual, OCR, ASR), and dynamically assign weights based on user intent.
=============================================================================
"""

import os
import json
import google.generativeai as genai
from dotenv import load_dotenv 

# Load environment variables
load_dotenv()

class QueryAgent:
    def __init__(self):
        """
        Initializes the Gemini model with a strict System Prompt to enforce 
        JSON formatting and specific query parsing rules.
        """
        print("Initializing LLM Query Agent...")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("[ERROR] GEMINI_API_KEY not found in .env file!")
            
        genai.configure(api_key=api_key)

        # The System Prompt dictates the rules for the AI
        self.system_prompt = """
        Bạn là một chuyên gia phân tích truy vấn (Query Parser) cho hệ thống tìm kiếm Video Đa phương thức.
        Nhiệm vụ của bạn là đọc câu hỏi tự nhiên của người dùng, khử nhiễu (loại bỏ các từ như "tìm cảnh", "cho tôi xem", v.v.) 
        và bóc tách thành 3 mảng: hình ảnh, chữ viết trên màn hình, và lời bình luận viên.
        Đồng thời, bạn phải tự động điều chỉnh Trọng số (weights) tùy theo ý định của người dùng.

        Quy tắc điền thông tin (QUAN TRỌNG VỀ NGÔN NGỮ):
        1. visual_query: Chỉ giữ lại miêu tả hành động, bối cảnh, màu sắc, vật thể. BẮT BUỘC DỊCH PHẦN NÀY SANG TIẾNG ANH (Ví dụ: "hai người đàn ông" -> "two men").
        2. ocr_query: Nếu người dùng nhắc đến "có chữ", "số", "bảng hiệu", "logo" -> điền vào đây. GIỮ NGUYÊN TIẾNG VIỆT.
        3. asr_query: Nếu người dùng nhắc đến "nghe thấy", "bình luận viên nói", "hô to" -> điền vào đây. GIỮ NGUYÊN TIẾNG VIỆT.
        
        Quy tắc đánh Trọng số (weights):
        - Mặc định: image=1.0, caption=1.0, yolo=1.0, ocr=0.5, asr=0.5
        - Nếu có ocr_query, hãy tăng ocr weight lên 1.5 hoặc 2.0.
        - Nếu có asr_query, hãy tăng asr weight lên 1.5 hoặc 2.0.
        - Nếu nhấn mạnh vào hành động cụ thể, hãy tăng yolo/image lên 1.2.

        YÊU CẦU BẮT BUỘC: 
        Chỉ trả về DUY NHẤT một JSON Object, không giải thích gì thêm.
        """

        # Using gemini-1.5-flash for maximum speed with JSON mode enabled
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=self.system_prompt,
            generation_config={"response_mime_type": "application/json"}
        )

    def parse_query(self, user_text: str) -> dict:
        """
        Sends the user text to Gemini and returns a parsed dictionary.
        Includes a fallback mechanism if the API fails.
        """
        # Default fallback dictionary in case of API timeout or failure
        fallback_result = {
            "visual_query": user_text,
            "ocr_query": "",
            "asr_query": "",
            "weights": {
                "image": 1.0, "caption": 1.0, "yolo": 1.0, "ocr": 1.0, "asr": 1.0
            }
        }

        if not user_text or not user_text.strip():
            return fallback_result

        try:
            # Send request to Gemini
            response = self.model.generate_content(user_text)
            
            # Parse the JSON string into a Python Dictionary
            parsed_data = json.loads(response.text)
            
            # Ensure the structure matches our expectations
            if "weights" not in parsed_data:
                parsed_data["weights"] = fallback_result["weights"]
                
            return parsed_data

        except Exception as e:
            print(f"[QueryAgent WARNING] LLM parsing failed: {e}. Using fallback.")
            return fallback_result

# Quick test when running this file directly
if __name__ == "__main__":
    agent = QueryAgent()
    sample_query = "Tìm cảnh vận động viên đang phát cầu, trên áo có số 15, hình như bình luận viên đang nói là 'trận đấu bắt đầu'"
    
    print(f"Câu hỏi gốc: {sample_query}\n")
    result = agent.parse_query(sample_query)
    
    print("Kết quả JSON từ Agent:")
    print(json.dumps(result, ensure_ascii=False, indent=2)) 
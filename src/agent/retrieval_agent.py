"""
=============================================================================
RETRIEVAL AGENT (BRAIN) - POWERED BY GOOGLE GENAI (NEW SDK)
=============================================================================
Description:
The Agentic Brain of the system. Uses Gemini's Tool Calling capabilities via 
the new `google-genai` SDK to route queries, execute atomic searches, and 
evaluate thresholds dynamically.
=============================================================================
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Updated import path to match your actual project structure
from src.retrieval.tools import (
    search_visual_modality,
    search_text_modality,
    search_audio_modality
)

load_dotenv()

class RetrievalAgent:
    def __init__(self):
        # Load API key from environment
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("[ERROR] GOOGLE_API_KEY not found. Please check your .env file!")
        
        # 1. Initialize the new Client
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-3.6-flash"
        
        # 2. Map tool names to actual Python functions
        self.tool_map = {
            "search_visual_modality": search_visual_modality,
            "search_text_modality": search_text_modality,
            "search_audio_modality": search_audio_modality
        }
        
        # 3. Declare tools using the new SDK's Schema format matching exact function arguments
        self.tools = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="search_visual_modality",
                        description="Search video frames using visual descriptions. Translate Vietnamese queries to English before searching.",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "visual_query": types.Schema(
                                    type=types.Type.STRING,
                                    description="The visual description query in English"
                                )
                            },
                            required=["visual_query"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="search_text_modality",
                        description="Search video frames using exact text, OCR, or logos visible in the video.",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "text_query": types.Schema(
                                    type=types.Type.STRING,
                                    description="The text or logo query"
                                )
                            },
                            required=["text_query"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="search_audio_modality",
                        description="Search video frames using spoken words or audio transcripts (ASR).",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "spoken_query": types.Schema(
                                    type=types.Type.STRING,
                                    description="The spoken words query"
                                )
                            },
                            required=["spoken_query"]
                        )
                    )
                ]
            )
        ]
        
        # 4. Define the Agent's system instruction
        self.system_instruction = (
            "You are an expert AI Video Retrieval Agent. Your job is to find the most accurate video keyframes "
            "based on the user's prompt. You have access to three specific tools (Visual, OCR, ASR). "
            "Analyze the user's intent carefully. Break down complex queries into multiple tool calls if necessary. "
            "Always translate Vietnamese queries to English before passing them to the visual tool."
        )

    def run(self, user_query: str, score_threshold: float = 0.25) -> list:
        """
        Executes the ReAct (Reasoning and Acting) loop for a given user query.
        """
        print(f"\n[AGENT START] User Query: '{user_query}'")
        
        # Configure the chat session with tools and instructions
        config = types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            tools=self.tools,
            temperature=0.0, # Temperature = 0 ensures logical and deterministic behavior
        )
        
        # Start a chat session using the new Client
        chat = self.client.chats.create(
            model=self.model_name,
            config=config
        )
        
        final_results = {}
        iteration = 0
        max_iterations = 3
        
        # Send the initial query to the model
        response = chat.send_message(user_query)
        
        while iteration < max_iterations:
            iteration += 1
            
            # Extract function calls using the new SDK syntax
            function_calls = response.function_calls
            
            if not function_calls:
                print("[AGENT] No more tools to call. Finalizing...")
                break
                
            tool_responses = []
            
            for tool_call in function_calls:
                tool_name = tool_call.name
                tool_args = tool_call.args
                
                print(f"  -> [THOUGHT] Agent decided to use: {tool_name}")
                print(f"  -> [ACTION] Executing with args: {tool_args}")
                
                if tool_name in self.tool_map:
                    # Execute the corresponding tool
                    tool_result = self.tool_map[tool_name](**tool_args)
                    
                    # Accumulate results and keep the highest score for each frame
                    for fid, score in tool_result.items():
                        final_results[fid] = max(final_results.get(fid, 0), score)
                        
                    # Format the result to notify Gemini of the tool's success
                    simplified_result = {"status": "success", "top_frames_found": len(tool_result)}
                    
                    # Package the response in the format expected by the new SDK
                    tool_responses.append(
                        types.Part.from_function_response(
                            name=tool_name,
                            response=simplified_result
                        )
                    )
            
            # Dynamic Evaluation (Threshold Check)
            if final_results:
                best_score = max(final_results.values())
                print(f"  -> [OBSERVATION] Best score in this iteration: {best_score:.4f}")
                
                if best_score >= score_threshold:
                    print("  -> [VERDICT] Results meet threshold. Stopping search.")
                    break
                else:
                    print("  -> [VERDICT] Score below threshold. Iterating...")
            
            # Send the tool results back to Gemini to continue the ReAct loop
            if tool_responses:
                response = chat.send_message(tool_responses)
            else:
                break
                
        # Sort results by descending score and return Top 50
        sorted_results = sorted(final_results.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:50] 
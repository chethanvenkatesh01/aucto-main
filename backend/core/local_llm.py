# backend/core/local_llm.py

import requests
import json
import os
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PHYSICS_LAYER")

# POINT THIS TO YOUR BEAST'S LOCAL SERVER
# Docker -> Host machine communication
LOCAL_INFERENCE_URL = os.environ.get("LOCAL_LLM_URL", "http://host.docker.internal:11434/v1/chat/completions")

class LocalInferenceAdapter:
    """
    The Sovereign Interface.
    Routes cognitive tasks to the local silicon (Reflex vs Reflection lanes).
    """

    @staticmethod
    def generate(prompt: str, role: str = "analyst", json_mode: bool = False, system_instruction: str = None) -> str:
        """
        role='analyst'    -> Routes to 8B Model (Fast/Reflex/GPU-0)
        role='strategist' -> Routes to 70B Model (Deep/Reflection/Dual-GPU)
        """
        
        # 1. Asymmetric Scheduling (The "Split Brain")
        if role == "strategist":
            model_id = "llama3:70b"  # The Heavy Lifter (ensure you pulled this: ollama pull llama3:70b)
            # Fallback for dev rig if 70b is not loaded
            # model_id = "llama3" 
            temperature = 0.6
        else:
            model_id = "llama3"      # The Fast Worker (8B)
            temperature = 0.1

        # 2. Construct Payload (OpenAI Compatible API)
        headers = {"Content-Type": "application/json"}
        
        default_system = "You are a sovereign component of the Auctorian Decision Engine. Act with precision."
        sys_msg = system_instruction if system_instruction else default_system

        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "stream": False
        }

        if json_mode:
            payload["format"] = "json"

        # 3. Execute on Local Silicon
        try:
            logger.info(f"⚡ [PHYSICS] Sending thought to {model_id} (Role: {role})...")
            response = requests.post(LOCAL_INFERENCE_URL, headers=headers, json=payload, timeout=300) # Long timeout for 70B
            response.raise_for_status()
            result = response.json()
            content = result['choices'][0]['message']['content']
            return content
        except Exception as e:
            logger.error(f"🔥 [PHYSICS ERROR] GPU Inference Failed: {e}")
            # Fallback response to prevent crash
            return json.dumps({"error": "Sovereign Compute Node Offline", "rationale": "Hardware Failure", "actions": []})

# Singleton Instance
sovereign_brain = LocalInferenceAdapter()
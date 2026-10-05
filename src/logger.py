import os
import json
import datetime

class ChimeraLogger:
    """
    Chimera Centralized Structured Logger.
    Complies with Rule 5: Mandatory Structured Failure Routing.
    Routes logs to strictly segregated JSON endpoints with zero network dependencies.
    """
    
    LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
    
    @classmethod
    def _ensure_logs_dir(cls):
        os.makedirs(cls.LOGS_DIR, exist_ok=True)
        
    @classmethod
    def _append_to_log(cls, filename: str, payload: dict):
        cls._ensure_logs_dir()
        log_path = os.path.join(cls.LOGS_DIR, filename)
        
        # Add timestamp
        payload["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
        
        logs = []
        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            except json.JSONDecodeError:
                pass
                
        logs.append(payload)
        
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)

    @classmethod
    def log_hallucination(cls, prompt_context: str, generated_text: str, confidence_score: float):
        payload = {
            "prompt_context": prompt_context,
            "generated_text": generated_text,
            "confidence_score": confidence_score
        }
        cls._append_to_log("hallucinations.json", payload)

    @classmethod
    def log_failed_math(cls, expression: str, incorrect_output: str, expected_output: str = None):
        payload = {
            "expression": expression,
            "incorrect_output": incorrect_output,
            "expected_output": expected_output
        }
        cls._append_to_log("failed_math.json", payload)

    @classmethod
    def log_failed_code(cls, err_type: str, message: str, traceback_str: str):
        payload = {
            "type": err_type,
            "message": message,
            "traceback": traceback_str
        }
        cls._append_to_log("failed_code.json", payload)

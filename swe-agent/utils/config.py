import os
from typing import Optional, Dict, Any
from pathlib import Path
import json


class Config:
    def __init__(self):
        self.debug_mode: bool = False
        self.max_iterations: int = 8
        self.model_temperature: float = 0.1
        self.max_tokens: int = 4000
        self.tool_timeout: int = 30
        self.history_file: str = str(Path.home() / ".reasoning_cli_history")
        self.log_level: str = "INFO"
        
        # load from environment
        self._load_from_env()
        
    def _load_from_env(self) -> None:
        self.debug_mode = os.getenv("REASONING_CLI_DEBUG", "false").lower() == "true"
        self.max_iterations = int(os.getenv("REASONING_CLI_MAX_ITERATIONS", "8"))
        self.model_temperature = float(os.getenv("REASONING_CLI_TEMPERATURE", "0.1"))
        self.max_tokens = int(os.getenv("REASONING_CLI_MAX_TOKENS", "4000"))
        self.tool_timeout = int(os.getenv("REASONING_CLI_TOOL_TIMEOUT", "30"))
        self.log_level = os.getenv("REASONING_CLI_LOG_LEVEL", "INFO")
        
    @classmethod
    def load_from_file(cls, config_path: str) -> 'Config':
        config = cls()
        
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
                
            for key, value in data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    
        except Exception as e:
            print(f"warning: failed to load config from {config_path}: {e}")
            
        return config
        
    def save_to_file(self, config_path: str) -> bool:
        try:
            config_data = {
                "debug_mode": self.debug_mode,
                "max_iterations": self.max_iterations,
                "model_temperature": self.model_temperature,
                "max_tokens": self.max_tokens,
                "tool_timeout": self.tool_timeout,
                "log_level": self.log_level
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            return True
            
        except Exception:
            return False

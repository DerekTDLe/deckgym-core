import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

class DiagnosticLogger:
    """
    Handles logging of detailed game states when errors or unusual events occur.
    Saves logs to a 'diagnostics' folder to help developers debug.
    """
    
    def __init__(self, log_dir: str = "diagnostics"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._action_histories = {} # env_idx -> List[int]
    
    def record_action(self, env_idx: int, action: int):
        """Record an action taken in an environment."""
        if env_idx not in self._action_histories:
            self._action_histories[env_idx] = []
        self._action_histories[env_idx].append(action)
        
        # Limit history size to 1000 last actions
        if len(self._action_histories[env_idx]) > 1000:
            self._action_histories[env_idx].pop(0)
            
    def clear_history(self, env_idx: int):
        """Clear action history for an environment (e.g. on reset)."""
        if env_idx in self._action_histories:
            self._action_histories[env_idx] = []

    def log_error(self, 
                  reason: str, 
                  env_idx: int, 
                  game_state: Any, 
                  extra_info: Optional[Dict[str, Any]] = None):
        """
        Log a detailed error report.
        
        Args:
            reason: Short string describing why the log was triggered (e.g. 'panic', 'max_turns')
            env_idx: Index of the environment
            game_state: The PyState object from Rust
            extra_info: Any additional context
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{reason}_env{env_idx}_{timestamp}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        try:
            # Get full state as JSON from Rust
            state_json = json.loads(game_state.to_json())
            
            report = {
                "reason": reason,
                "env_idx": env_idx,
                "timestamp": timestamp,
                "turn_count": game_state.turn_count,
                "current_player": game_state.current_player,
                "points": list(game_state.points),
                "action_history": self._action_histories.get(env_idx, []),
                "game_state": state_json,
                "extra_info": extra_info or {}
            }
            
            with open(filepath, "w") as f:
                json.dump(report, f, indent=2)
                
            print(f"[DIAGNOSTIC] Logged incident to {filepath}")
        except Exception as e:
            print(f"[DIAGNOSTIC] Failed to log incident: {e}")

# Global instance
_logger = DiagnosticLogger()

def get_logger():
    return _logger

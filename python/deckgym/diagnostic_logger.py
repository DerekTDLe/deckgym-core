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
        self._action_histories = {}  # env_idx -> List[int]

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

    def log_error(
        self,
        reason: str,
        env_idx: int,
        game_state: Any,
        extra_info: Optional[Dict[str, Any]] = None,
    ):
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
            # Handle cases where game_state might be None or invalid due to crash
            if game_state is None:
                state_json = {"error": "Game state is None"}
                turn_count = 0
                current_player = 0
                points = [0, 0]
            else:
                try:
                    state_json = json.loads(game_state.to_json())
                    turn_count = game_state.turn_count
                    current_player = game_state.current_player
                    points = list(game_state.points)
                except Exception as e:
                    state_json = {"error": f"Failed to serialize state: {e}"}
                    turn_count = getattr(game_state, "turn_count", 0)
                    current_player = getattr(game_state, "current_player", 0)
                    points = list(getattr(game_state, "points", [0, 0]))

            report = {
                "reason": reason,
                "env_idx": env_idx,
                "timestamp": timestamp,
                "turn_count": turn_count,
                "current_player": current_player,
                "points": points,
                "action_history": self._action_histories.get(env_idx, []),
                "game_state": state_json,
                "extra_info": extra_info or {},
            }

            with open(filepath, "w") as f:
                json.dump(report, f, indent=2)

            print(f"\n{'!' * 100}")
            print(f" [DIAGNOSTIC CRITICAL] TRIPPED: {reason}")
            print(f" [FILEPATH] {filepath}")
            print(f"{'!' * 100}\n")
        except Exception as e:
            print(f"[DIAGNOSTIC] Failed to log incident: {e}")

    def capture_panic(self, env_provider=None):
        """
        Context manager to capture panics during training.
        
        Args:
            env_provider: Optional callable that returns the current environment(s)
                         to dump state from on crash.
        """
        from contextlib import contextmanager

        @contextmanager
        def _capture():
            try:
                yield
            except BaseException as e:
                # Capture the panic message
                panic_msg = str(e)
                print(f"\n{'!' * 80}")
                print(f"CRITICAL ERROR CAPTURED: {panic_msg}")
                print(f"{'!' * 80}")
                
                if env_provider:
                    try:
                        env = env_provider()
                        if env:
                            self.dump_all_states(env, reason="crash_dump", extra_info={"error": panic_msg})
                    except Exception as dump_err:
                        print(f"[DIAGNOSTIC] Could not dump states during crash: {dump_err}")
                
                raise e

        return _capture()

    def dump_all_states(self, env, reason: str = "manual_dump", extra_info: Optional[Dict] = None):
        """Dump states of all environments in a VecEnv."""
        from deckgym.batched_env import BatchedDeckGymEnv
        
        if isinstance(env, BatchedDeckGymEnv):
            for i in range(env.n_envs):
                try:
                    state = env.vec_game.get_state(i)
                    self.log_error(reason, i, state, extra_info)
                except Exception as e:
                    print(f"[DIAGNOSTIC] Failed to dump env {i}: {e}")
        elif hasattr(env, "envs"): # DummyVecEnv
            for i, sub_env in enumerate(env.envs):
                try:
                    # Access underlying game if possible
                    inner = sub_env
                    while hasattr(inner, "env"): inner = inner.env
                    if hasattr(inner, "game"):
                        self.log_error(reason, i, inner.game.get_state(), extra_info)
                except Exception as e:
                    print(f"[DIAGNOSTIC] Failed to dump sub-env {i}: {e}")

    def setup_excepthook(self):
        """Install a global exception hook to capture unhandled panics."""
        import sys
        import traceback

        def hook(type, value, tb):
            msg = "".join(traceback.format_exception(type, value, tb))
            print(f"\n[DIAGNOSTIC] Unhandled Exception Hook Triggered:\n{msg}")
            # We can't easily dump env state here without a reference to the env
            # but we've logged the traceback.
            sys.__excepthook__(type, value, tb)

        sys.excepthook = hook


# Global instance
_logger = DiagnosticLogger()


def get_logger():
    return _logger

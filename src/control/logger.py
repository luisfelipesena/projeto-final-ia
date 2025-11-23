"""
Logging Configuration for Fuzzy Control System

Provides structured JSON logging for state transitions and rule activations.

Logs are written to:
- logs/fuzzy_control/state_transitions.json
- logs/fuzzy_control/rule_activations.json
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Create logs directory
LOGS_DIR = Path("logs/fuzzy_control")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Log file paths
STATE_TRANSITIONS_LOG = LOGS_DIR / "state_transitions.json"
RULE_ACTIVATIONS_LOG = LOGS_DIR / "rule_activations.json"


class JSONLogger:
    """Structured JSON logger for fuzzy control events"""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.touch(exist_ok=True)
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """
        Log an event as JSON line
        
        Args:
            event_type: Type of event (e.g., 'state_transition', 'rule_activation')
            data: Event data dictionary
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            **data
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(event) + '\n')


# Global loggers
state_logger = JSONLogger(STATE_TRANSITIONS_LOG)
rule_logger = JSONLogger(RULE_ACTIVATIONS_LOG)


def log_state_transition(from_state: str, to_state: str, conditions: Optional[Dict] = None):
    """
    Log a state machine transition
    
    Args:
        from_state: Previous state name
        to_state: New state name
        conditions: Optional sensor conditions that triggered transition
    """
    state_logger.log_event('state_transition', {
        'from_state': from_state,
        'to_state': to_state,
        'conditions': conditions or {}
    })


def log_rule_activation(rule_id: str, activation_level: float, inputs: Dict, outputs: Dict):
    """
    Log a fuzzy rule activation
    
    Args:
        rule_id: Rule identifier
        activation_level: Degree of activation (0.0-1.0)
        inputs: Input values that triggered rule
        outputs: Output values produced by rule
    """
    rule_logger.log_event('rule_activation', {
        'rule_id': rule_id,
        'activation_level': activation_level,
        'inputs': inputs,
        'outputs': outputs
    })


# Standard Python logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'fuzzy_control.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('fuzzy_control')

"""
Training Run Logger

Structured JSON logging for training runs, capturing hyperparameters, metrics,
hardware profiles, and audit trail for reproducibility.

Based on: data-model.md TrainingRun entity
"""

import json
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from uuid import uuid4


class TrainingRunLogger:
    """
    Logger for training runs with structured JSON output
    
    Captures:
    - run_id (UUID)
    - model_type (lidar | camera)
    - dataset_hash
    - hyperparameters
    - metrics
    - hardware_profile
    - artifacts
    - notes
    """
    
    def __init__(self, log_dir: Path, model_type: str, dataset_hash: str):
        """
        Initialize training run logger
        
        Args:
            log_dir: Directory to write log files
            model_type: 'lidar' or 'camera'
            dataset_hash: Hash of dataset manifest
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_id = str(uuid4())
        self.model_type = model_type
        self.dataset_hash = dataset_hash
        self.start_time = datetime.now().isoformat()
        
        self.hyperparameters: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.artifacts: list[str] = []
        self.notes: str = ""
        
        # Capture hardware profile
        self.hardware_profile = self._capture_hardware_profile()
        
        # Capture git commit if available
        self._capture_git_info()
    
    def _capture_hardware_profile(self) -> Dict[str, Any]:
        """Capture hardware profile information"""
        profile = {
            'os': platform.system(),
            'os_version': platform.version(),
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
        }
        
        # Try to get CPU info
        try:
            import psutil
            profile['cpu_count'] = psutil.cpu_count()
            profile['cpu_freq'] = psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            profile['memory_total_gb'] = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            pass
        
        # Try to detect GPU
        try:
            import torch
            if torch.cuda.is_available():
                profile['gpu'] = {
                    'available': True,
                    'device_count': torch.cuda.device_count(),
                    'device_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
                }
            else:
                profile['gpu'] = {'available': False}
        except ImportError:
            profile['gpu'] = {'available': False, 'note': 'PyTorch not available'}
        
        return profile
    
    def _capture_git_info(self):
        """Capture git commit hash and branch"""
        try:
            git_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            git_branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            self.hyperparameters['git_commit'] = git_commit
            self.hyperparameters['git_branch'] = git_branch
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Not a git repo or git not available
            pass
    
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """
        Log training hyperparameters
        
        Args:
            hyperparameters: Dict with optimizer, learning_rate, batch_size, epochs, etc.
        """
        self.hyperparameters.update(hyperparameters)
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """
        Log training metrics
        
        Args:
            metrics: Dict with accuracy, recall_per_class, loss_curves, latency_ms, etc.
        """
        self.metrics.update(metrics)
    
    def add_artifact(self, artifact_id: str):
        """
        Add reference to model artifact
        
        Args:
            artifact_id: UUID of ModelArtifact
        """
        self.artifacts.append(artifact_id)
    
    def add_note(self, note: str):
        """
        Add note (citations, observations)
        
        Args:
            note: Markdown string with notes
        """
        if self.notes:
            self.notes += "\n\n" + note
        else:
            self.notes = note
    
    def save(self) -> Path:
        """
        Save training run log to JSON file
        
        Returns:
            Path to saved log file
        """
        end_time = datetime.now().isoformat()
        
        run_data = {
            'run_id': self.run_id,
            'model_type': self.model_type,
            'dataset_hash': self.dataset_hash,
            'start_time': self.start_time,
            'end_time': end_time,
            'hyperparameters': self.hyperparameters,
            'metrics': self.metrics,
            'hardware_profile': self.hardware_profile,
            'artifacts': self.artifacts,
            'notes': self.notes,
        }
        
        log_file = self.log_dir / f'training_run_{self.run_id}.json'
        with open(log_file, 'w') as f:
            json.dump(run_data, f, indent=2)
        
        # Also save a metrics-only file for quick access
        metrics_file = self.log_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        return log_file
    
    def get_run_id(self) -> str:
        """Get the run ID"""
        return self.run_id


def create_logger(log_dir: str, model_type: str, dataset_hash: str) -> TrainingRunLogger:
    """
    Convenience function to create a TrainingRunLogger
    
    Args:
        log_dir: Directory path for logs
        model_type: 'lidar' or 'camera'
        dataset_hash: Hash of dataset manifest
    
    Returns:
        TrainingRunLogger instance
    """
    return TrainingRunLogger(Path(log_dir), model_type, dataset_hash)



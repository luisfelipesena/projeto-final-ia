"""
Model Artifact Metadata Generator

Generates ModelArtifact JSON metadata with checksums, preprocessing specs,
calibration constants, and performance metrics.

Based on: data-model.md ModelArtifact entity
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from uuid import uuid4


def compute_file_checksum(file_path: Path) -> str:
    """
    Compute SHA256 checksum of a file
    
    Args:
        file_path: Path to file
    
    Returns:
        SHA256 hex digest
    """
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def generate_artifact_metadata(
    model_file: Path,
    model_type: str,
    format: str,
    metrics_snapshot: Dict[str, Any],
    preprocessing: Dict[str, Any],
    calibration: Dict[str, Any],
    run_id: Optional[str] = None,
    spec_version: Optional[str] = None,
    decision_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate ModelArtifact metadata JSON
    
    Args:
        model_file: Path to model artifact file (must exist)
        model_type: 'lidar' or 'camera'
        format: 'torchscript', 'onnx', or 'metrics-json'
        metrics_snapshot: Key performance metrics at export time
        preprocessing: Normalization params, input resolution, etc.
        calibration: Focal length, principal point (camera) or sector bounds (LIDAR)
        run_id: Optional training run ID that produced this artifact
        spec_version: Optional spec version reference
        decision_id: Optional DECISIONS.md entry ID
    
    Returns:
        ModelArtifact dict ready for JSON serialization
    """
    if not model_file.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_file}")
    
    artifact_id = str(uuid4())
    checksum = compute_file_checksum(model_file)
    created_at = datetime.now().isoformat()
    
    # Ensure preprocessing includes required fields
    required_preprocessing_fields = ['input_channels', 'resolution', 'mean', 'std', 'normalization_space']
    for field in required_preprocessing_fields:
        if field not in preprocessing:
            raise ValueError(f"preprocessing missing required field: {field}")
    
    metadata = {
        'artifact_id': artifact_id,
        'model_type': model_type,
        'format': format,
        'file_path': str(model_file.relative_to(Path.cwd())) if model_file.is_relative_to(Path.cwd()) else str(model_file),
        'checksum': checksum,
        'metrics_snapshot': metrics_snapshot,
        'preprocessing': preprocessing,
        'calibration': calibration,
        'created_at': created_at,
    }
    
    # Optional fields
    if run_id:
        metadata['run_id'] = run_id
    if spec_version:
        metadata['spec_version'] = spec_version
    if decision_id:
        metadata['decision_id'] = decision_id
    
    return metadata


def save_artifact_metadata(metadata: Dict[str, Any], output_path: Path):
    """
    Save artifact metadata to JSON file
    
    Args:
        metadata: ModelArtifact dict from generate_artifact_metadata()
        output_path: Path to save metadata JSON
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_artifact_metadata(metadata_path: Path) -> Dict[str, Any]:
    """
    Load artifact metadata from JSON file
    
    Args:
        metadata_path: Path to metadata JSON
    
    Returns:
        ModelArtifact dict
    """
    with open(metadata_path, 'r') as f:
        return json.load(f)


def verify_artifact_integrity(model_file: Path, metadata_path: Path) -> bool:
    """
    Verify artifact file matches checksum in metadata
    
    Args:
        model_file: Path to model artifact
        metadata_path: Path to metadata JSON
    
    Returns:
        True if checksum matches, False otherwise
    """
    metadata = load_artifact_metadata(metadata_path)
    expected_checksum = metadata.get('checksum')
    
    if not expected_checksum:
        return False
    
    actual_checksum = compute_file_checksum(model_file)
    return actual_checksum == expected_checksum





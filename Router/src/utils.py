"""
Utility functions for Router
"""

import numpy as np
import yaml
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors
    
    Args:
        v1: First vector (d,)
        v2: Second vector (d,)
    
    Returns:
        Cosine similarity in [-1, 1]
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return np.dot(v1, v2) / (norm1 * norm2)


def softmax_with_temperature(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Compute softmax with temperature, with numerical stability
    
    Args:
        scores: Score vector (n,)
        temperature: Temperature parameter (lower = sharper)
    
    Returns:
        Probability distribution (n,)
    """
    # Numerical stability: subtract max
    scores_temp = scores / temperature
    scores_temp = scores_temp - np.max(scores_temp)
    
    exp_scores = np.exp(scores_temp)
    probs = exp_scores / np.sum(exp_scores)
    
    return probs


def normalized_entropy(probs: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute normalized entropy (in [0, 1])
    
    Args:
        probs: Probability distribution (n,)
        epsilon: Small value for numerical stability
    
    Returns:
        Normalized entropy in [0, 1]
    """
    n = len(probs)
    
    if n <= 1:
        return 0.0
    
    # Ensure epsilon is float
    epsilon = float(epsilon)
    
    # Clip probabilities to avoid log(0)
    probs = np.clip(probs, epsilon, 1.0)
    
    # Compute entropy: -sum(p * log(p))
    entropy = -np.sum(probs * np.log(probs))
    
    # Normalize by log(n)
    max_entropy = np.log(n)
    
    if max_entropy == 0:
        return 0.0
    
    normalized = entropy / max_entropy
    
    return np.clip(normalized, 0.0, 1.0)


def sigmoid(x: float) -> float:
    """Sigmoid function"""
    return 1.0 / (1.0 + np.exp(-x))


def adaptive_coverage_threshold(
    uncertainty: float,
    rho_min: float = 0.60,
    rho_max: float = 0.95,
    tau: float = 0.5,
    beta: float = 8.0
) -> float:
    """
    Compute adaptive coverage threshold based on uncertainty
    
    ρ_B(ũ_B) = ρ_min + (ρ_max - ρ_min) * σ(β(ũ_B - τ))
    
    Args:
        uncertainty: Normalized uncertainty in [0, 1]
        rho_min: Minimum coverage (when certain)
        rho_max: Maximum coverage (when uncertain)
        tau: Threshold where adaptation starts
        beta: Steepness of transition
    
    Returns:
        Coverage threshold in [rho_min, rho_max]
    """
    # Sigmoid transition
    sig_value = sigmoid(beta * (uncertainty - tau))
    
    # Linear interpolation
    rho = rho_min + (rho_max - rho_min) * sig_value
    
    return rho


def cumulative_selection(
    probs: np.ndarray,
    ids: List[str],
    coverage_threshold: float
) -> Tuple[List[str], List[float], float]:
    """
    Select items based on cumulative probability mass (Top-K forbidden!)
    
    Select items until cumulative probability >= coverage_threshold
    
    Args:
        probs: Probability distribution (n,)
        ids: Item IDs (n,)
        coverage_threshold: Cumulative probability threshold
    
    Returns:
        selected_ids: Selected item IDs
        selected_probs: Corresponding probabilities
        actual_coverage: Actual cumulative probability covered
    """
    # Sort by probability (descending)
    sorted_indices = np.argsort(probs)[::-1]
    
    selected_ids = []
    selected_probs = []
    cumulative = 0.0
    
    for idx in sorted_indices:
        cumulative += probs[idx]
        selected_ids.append(ids[idx])
        selected_probs.append(float(probs[idx]))
        
        # Stop when coverage threshold is reached
        if cumulative >= coverage_threshold:
            break
    
    return selected_ids, selected_probs, cumulative


def save_debug_info(data: Dict[str, Any], output_path: str):
    """Save debug information to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert numpy types to Python native types
    def convert_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        else:
            return obj
    
    data = convert_types(data)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Debug info saved to {output_path}")


def load_or_create_prototypes(
    config: Dict[str, Any],
    embedding_model
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load or create prototype embeddings
    
    Args:
        config: Configuration dict
        embedding_model: Embedding model instance
    
    Returns:
        block_prototypes: Dict[block_id, embedding]
        role_prototypes: Dict[role_id, embedding]
    """
    block_proto_file = config['prototypes']['block_prototype_file']
    role_proto_file = config['prototypes']['role_prototype_file']
    
    # Try to load existing prototypes
    if os.path.exists(block_proto_file) and os.path.exists(role_proto_file):
        logger.info("Loading existing prototypes...")
        block_data = np.load(block_proto_file, allow_pickle=True).item()
        role_data = np.load(role_proto_file, allow_pickle=True).item()
        return block_data, role_data
    
    # Generate new prototypes
    logger.info("Generating new prototypes from descriptions...")
    
    block_descriptions = config['prototypes']['block_descriptions']
    role_descriptions = config['prototypes']['role_descriptions']
    
    # Generate block prototypes
    block_prototypes = {}
    for block_id, description in block_descriptions.items():
        embedding = embedding_model.encode(description)
        block_prototypes[block_id] = embedding
        logger.info(f"Generated prototype for block: {block_id}")
    
    # Generate role prototypes
    role_prototypes = {}
    for role_id, description in role_descriptions.items():
        embedding = embedding_model.encode(description)
        role_prototypes[role_id] = embedding
        logger.info(f"Generated prototype for role: {role_id}")
    
    # Save prototypes
    os.makedirs(os.path.dirname(block_proto_file), exist_ok=True)
    np.save(block_proto_file, block_prototypes)
    np.save(role_proto_file, role_prototypes)
    logger.info("Prototypes saved.")
    
    return block_prototypes, role_prototypes

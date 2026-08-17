"""
Stage-2: Adaptive Role Routing
Determines "who participates and how many" (fine-grained participation)
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple

from .utils import (
    cosine_similarity,
    softmax_with_temperature,
    normalized_entropy,
    adaptive_coverage_threshold,
    cumulative_selection
)

logger = logging.getLogger(__name__)


class Stage2RoleRouter:
    """
    Stage-2: Adaptive Role Routing
    
    Within the scope selected by Stage-1 (blocks), determines:
    1. Which roles should participate
    2. How many agents are needed
    
    Based on:
    1. Role relevance scores (cosine similarity)
    2. Role probability distribution (softmax with temperature)
    3. Uncertainty-aware adaptive coverage threshold
    4. Cumulative probability mass selection (NO Top-K!)
    """
    
    def __init__(
        self,
        role_prototypes: Dict[str, np.ndarray],
        block_role_mapping: Dict[str, List[str]],
        config: Dict[str, Any]
    ):
        """
        Args:
            role_prototypes: Dict[role_id, prototype_embedding]
            block_role_mapping: Dict[block_id, list of role_ids]
            config: Configuration dict
        """
        self.role_prototypes = role_prototypes
        self.block_role_mapping = block_role_mapping
        self.config = config
        
        # Stage-2 hyperparameters (with fallback to Stage-1)
        if 'stage2' in config:
            stage2_config = config['stage2']
            self.temperature = stage2_config.get('temperature_role', config['stage1']['temperature_block'])
            coverage_config = stage2_config['coverage']
            self.rho_min = coverage_config.get('rho_min_role', config['stage1']['coverage']['rho_min'])
            self.rho_max = coverage_config.get('rho_max_role', config['stage1']['coverage']['rho_max'])
            self.tau = coverage_config.get('tau_role', config['stage1']['coverage']['tau'])
            self.beta = coverage_config.get('beta_role', config['stage1']['coverage']['beta'])
            self.epsilon = stage2_config.get('epsilon', config['stage1']['epsilon'])
        else:
            # Fallback to stage1 config if stage2 not specified
            logger.warning("Stage-2 config not found, using Stage-1 config as fallback")
            stage1_config = config['stage1']
            self.temperature = stage1_config['temperature_block']
            coverage_config = stage1_config['coverage']
            self.rho_min = coverage_config['rho_min']
            self.rho_max = coverage_config['rho_max']
            self.tau = coverage_config['tau']
            self.beta = coverage_config['beta']
            self.epsilon = stage1_config['epsilon']
        
        logger.info(f"Stage2 initialized with {len(self.role_prototypes)} roles")
        logger.info(f"Temperature: {self.temperature}")
        logger.info(f"Coverage: ρ_min={self.rho_min}, ρ_max={self.rho_max}, τ={self.tau}, β={self.beta}")
    
    def get_candidate_roles(self, selected_blocks: List[str]) -> List[str]:
        """
        Construct candidate role set from selected blocks
        
        R_q = ∪_{B ∈ B_q} R_B
        
        Args:
            selected_blocks: List of selected block IDs from Stage-1
        
        Returns:
            List of candidate role IDs (duplicates removed)
        """
        candidate_roles = []
        
        for block_id in selected_blocks:
            if block_id in self.block_role_mapping:
                block_roles = self.block_role_mapping[block_id]
                candidate_roles.extend(block_roles)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_roles = []
        for role in candidate_roles:
            if role not in seen:
                seen.add(role)
                unique_roles.append(role)
        
        logger.info(f"Candidate roles from {len(selected_blocks)} blocks: {len(unique_roles)} roles")
        logger.info(f"Candidates: {unique_roles}")
        
        return unique_roles
    
    def compute_role_scores(
        self,
        v: np.ndarray,
        candidate_roles: List[str]
    ) -> Dict[str, float]:
        """
        Compute role relevance scores using cosine similarity
        
        s_r = cos(v, p_r)
        
        Args:
            v: Query representation vector (d,)
            candidate_roles: List of candidate role IDs
        
        Returns:
            Dict[role_id, score]
        """
        scores = {}
        
        for role_id in candidate_roles:
            if role_id not in self.role_prototypes:
                logger.warning(f"Role prototype not found: {role_id}, skipping")
                continue
            
            prototype = self.role_prototypes[role_id]
            score = cosine_similarity(v, prototype)
            scores[role_id] = score
        
        logger.info(f"Role scores computed: {len(scores)} roles")
        
        return scores
    
    def compute_role_probabilities(
        self,
        scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute role probability distribution using softmax with temperature
        
        p_r = exp(s_r / T) / Σ exp(s_k / T)
        
        Args:
            scores: Dict[role_id, score]
        
        Returns:
            Dict[role_id, probability]
        """
        role_ids = list(scores.keys())
        score_array = np.array([scores[role_id] for role_id in role_ids])
        
        # Apply softmax with temperature
        prob_array = softmax_with_temperature(score_array, self.temperature)
        
        # Convert back to dict
        probs = {
            role_id: prob_array[i]
            for i, role_id in enumerate(role_ids)
        }
        
        logger.info(f"Role probabilities computed (temp={self.temperature})")
        
        return probs
    
    def compute_uncertainty(self, probs: Dict[str, float]) -> float:
        """
        Compute normalized uncertainty using entropy
        
        ũ = -Σ p_r log(p_r) / log(|R_q|)
        
        Args:
            probs: Dict[role_id, probability]
        
        Returns:
            Normalized uncertainty in [0, 1]
        """
        prob_array = np.array(list(probs.values()))
        
        uncertainty = normalized_entropy(prob_array, self.epsilon)
        
        logger.info(f"Role uncertainty computed: ũ = {uncertainty:.4f}")
        
        return uncertainty
    
    def compute_adaptive_threshold(self, uncertainty: float) -> float:
        """
        Compute adaptive coverage threshold based on uncertainty
        
        ρ(ũ) = ρ_min + (ρ_max - ρ_min) * σ(β(ũ - τ))
        
        When certain → low coverage (fewer agents)
        When uncertain → high coverage (more agents)
        
        Args:
            uncertainty: Normalized uncertainty
        
        Returns:
            Coverage threshold in [rho_min, rho_max]
        """
        threshold = adaptive_coverage_threshold(
            uncertainty,
            self.rho_min,
            self.rho_max,
            self.tau,
            self.beta
        )
        
        logger.info(f"Adaptive threshold: ρ = {threshold:.4f} (for ũ = {uncertainty:.4f})")
        
        return threshold
    
    def select_roles(
        self,
        probs: Dict[str, float],
        coverage_threshold: float
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Select roles based on cumulative probability mass
        
        NO Top-K! Select until cumulative probability >= threshold
        
        Σ p_(i) >= ρ(ũ) → A_0 = {r_1, ..., r_K}
        
        Args:
            probs: Dict[role_id, probability]
            coverage_threshold: Coverage threshold
        
        Returns:
            selected_roles: List of selected role IDs (A_0)
            selection_info: Selection debug information
        """
        role_ids = list(probs.keys())
        prob_array = np.array([probs[role_id] for role_id in role_ids])
        
        # Cumulative selection
        selected_ids, selected_probs, actual_coverage = cumulative_selection(
            prob_array,
            role_ids,
            coverage_threshold
        )
        
        logger.info(f"Selected {len(selected_ids)} roles (coverage: {actual_coverage:.4f})")
        logger.info(f"Selected roles (A_0): {selected_ids}")
        
        # Selection info
        selection_info = {
            "selected_roles": selected_ids,
            "selected_probs": {
                role_id: prob
                for role_id, prob in zip(selected_ids, selected_probs)
            },
            "num_selected": len(selected_ids),
            "coverage_threshold": coverage_threshold,
            "actual_coverage": actual_coverage
        }
        
        return selected_ids, selection_info
    
    def route(
        self,
        v: np.ndarray,
        selected_blocks: List[str]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Main routing function for Stage-2
        
        Args:
            v: Query representation vector (d,) from Stage-0
            selected_blocks: Selected blocks from Stage-1
        
        Returns:
            selected_roles: List of selected role IDs (A_0)
            debug_info: Debug information
        """
        logger.info("=" * 60)
        logger.info("STAGE-2: Adaptive Role Routing")
        logger.info("=" * 60)
        
        # Step 1: Construct candidate role set
        candidate_roles = self.get_candidate_roles(selected_blocks)
        
        # Edge case: no candidate roles
        if not candidate_roles:
            logger.warning("No candidate roles found!")
            return [], {
                "candidate_roles": [],
                "role_scores": {},
                "role_probs": {},
                "uncertainty": 0.0,
                "coverage_threshold": 0.0,
                "selection_info": {}
            }
        
        # Edge case: single candidate role
        if len(candidate_roles) == 1:
            logger.info(f"Single candidate role: {candidate_roles[0]} (auto-selected)")
            return candidate_roles, {
                "candidate_roles": candidate_roles,
                "role_scores": {candidate_roles[0]: 1.0},
                "role_probs": {candidate_roles[0]: 1.0},
                "uncertainty": 0.0,
                "coverage_threshold": self.rho_min,
                "selection_info": {
                    "selected_roles": candidate_roles,
                    "selected_probs": {candidate_roles[0]: 1.0},
                    "num_selected": 1,
                    "coverage_threshold": self.rho_min,
                    "actual_coverage": 1.0
                }
            }
        
        # Step 2: Compute role relevance scores
        scores = self.compute_role_scores(v, candidate_roles)
        
        # Step 3: Compute role probability distribution
        probs = self.compute_role_probabilities(scores)
        
        # Step 4: Compute uncertainty
        uncertainty = self.compute_uncertainty(probs)
        
        # Step 5: Compute adaptive coverage threshold
        coverage_threshold = self.compute_adaptive_threshold(uncertainty)
        
        # Step 6: Select roles based on cumulative probability
        selected_roles, selection_info = self.select_roles(probs, coverage_threshold)
        
        # Debug info
        debug_info = {
            "candidate_roles": candidate_roles,
            "role_scores": scores,
            "role_probs": probs,
            "uncertainty": uncertainty,
            "coverage_threshold": coverage_threshold,
            "selection_info": selection_info,
            "temperature": self.temperature
        }
        
        logger.info("=" * 60)
        logger.info(f"Stage-2 Complete: {len(selected_roles)} roles selected (A_0)")
        logger.info("=" * 60)
        
        return selected_roles, debug_info

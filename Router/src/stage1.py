"""
Stage-1: Adaptive Soft Block Routing
Determines "where to look" (coarse scope)
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


class Stage1BlockRouter:
    """
    Stage-1: Adaptive Soft Block Routing
    
    Determines which blocks to activate based on:
    1. Block relevance scores (cosine similarity)
    2. Block probability distribution (softmax with temperature)
    3. Uncertainty-aware adaptive coverage threshold
    4. Cumulative probability mass selection (NO Top-K!)
    """
    
    def __init__(
        self,
        block_prototypes: Dict[str, np.ndarray],
        config: Dict[str, Any]
    ):
        """
        Args:
            block_prototypes: Dict[block_id, prototype_embedding]
            config: Configuration dict
        """
        self.block_prototypes = block_prototypes
        self.config = config
        
        # Extract block configuration
        self.blocks = config['blocks']
        self.block_ids = list(self.block_prototypes.keys())
        
        # Stage-1 hyperparameters
        stage1_config = config['stage1']
        self.temperature = stage1_config['temperature_block']
        
        # Domain Prior parameters
        self.lambda_prior = stage1_config.get('lambda_prior', 0.0)
        self.domain_priors = stage1_config.get('domain_priors', {})
        
        # Adaptive coverage parameters
        coverage_config = stage1_config['coverage']
        self.rho_min = coverage_config['rho_min']
        self.rho_max = coverage_config['rho_max']
        self.tau = coverage_config['tau']
        self.beta = coverage_config['beta']
        
        self.epsilon = stage1_config['epsilon']
        
        logger.info(f"Stage1 initialized with {len(self.block_ids)} blocks")
        logger.info(f"Temperature: {self.temperature}")
        logger.info(f"Domain Prior: λ={self.lambda_prior} ({'ENABLED' if self.lambda_prior > 0 else 'DISABLED'})")
        logger.info(f"Coverage: ρ_min={self.rho_min}, ρ_max={self.rho_max}, τ={self.tau}, β={self.beta}")
    
    def compute_block_prior(self, domain_hint: str) -> Dict[str, float]:
        """
        Compute block prior based on domain hint from Stage-0
        
        π(h, B) = domain_priors[h][B] if exists, else 0
        
        Args:
            domain_hint: Domain hint from LLM summary (e.g., "mathematics", "biology")
        
        Returns:
            Dict[block_id, prior_score]
        """
        # Initialize all priors to 0
        priors = {block_id: 0.0 for block_id in self.block_ids}
        
        if not domain_hint or self.lambda_prior == 0:
            return priors
        
        # Normalize domain hint (lowercase, strip)
        domain_hint_norm = domain_hint.lower().strip()
        
        # Look up prior mapping
        if domain_hint_norm in self.domain_priors:
            prior_mapping = self.domain_priors[domain_hint_norm]
            for block_id, boost in prior_mapping.items():
                if block_id in priors:
                    priors[block_id] = boost
            
            logger.info(f"Domain prior applied: '{domain_hint}' → {prior_mapping}")
        else:
            logger.debug(f"No prior mapping for domain_hint: '{domain_hint}'")
        
        return priors
    
    def compute_block_scores(
        self,
        v: np.ndarray,
        domain_hint: str = None
    ) -> Dict[str, float]:
        """
        Compute block relevance scores with optional domain prior
        
        s_B = cos(v, p_B) + λ * π(h, B)
        
        Args:
            v: Query representation vector (d,)
            domain_hint: Optional domain hint from Stage-0
        
        Returns:
            Dict[block_id, score]
        """
        scores = {}
        
        # Compute cosine similarities
        for block_id in self.block_ids:
            prototype = self.block_prototypes[block_id]
            cos_score = cosine_similarity(v, prototype)
            scores[block_id] = cos_score
        
        # Apply domain prior if enabled
        if self.lambda_prior > 0 and domain_hint:
            priors = self.compute_block_prior(domain_hint)
            
            logger.info(f"Applying domain prior (λ={self.lambda_prior}):")
            for block_id in self.block_ids:
                original_score = scores[block_id]
                prior_boost = self.lambda_prior * priors[block_id]
                scores[block_id] = original_score + prior_boost
                
                if priors[block_id] > 0:
                    logger.info(f"  {block_id}: {original_score:.4f} + {prior_boost:.4f} = {scores[block_id]:.4f}")
        
        logger.info(f"Block scores computed: {len(scores)} blocks")
        
        return scores
    
    def compute_block_probabilities(
        self,
        scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute block probability distribution using softmax with temperature
        
        p_B = exp(s_B / T_B) / Σ exp(s_B' / T_B)
        
        Args:
            scores: Dict[block_id, score]
        
        Returns:
            Dict[block_id, probability]
        """
        # Convert to array
        score_array = np.array([scores[block_id] for block_id in self.block_ids])
        
        # Apply softmax with temperature
        prob_array = softmax_with_temperature(score_array, self.temperature)
        
        # Convert back to dict
        probs = {
            block_id: prob_array[i]
            for i, block_id in enumerate(self.block_ids)
        }
        
        logger.info(f"Block probabilities computed (temp={self.temperature})")
        
        return probs
    
    def compute_uncertainty(self, probs: Dict[str, float]) -> float:
        """
        Compute normalized uncertainty using entropy
        
        ũ_B = -Σ p_B log(p_B) / log(|B|)
        
        Args:
            probs: Dict[block_id, probability]
        
        Returns:
            Normalized uncertainty in [0, 1]
        """
        prob_array = np.array([probs[block_id] for block_id in self.block_ids])
        
        uncertainty = normalized_entropy(prob_array, self.epsilon)
        
        logger.info(f"Uncertainty computed: ũ_B = {uncertainty:.4f}")
        
        return uncertainty
    
    def compute_adaptive_threshold(self, uncertainty: float) -> float:
        """
        Compute adaptive coverage threshold based on uncertainty
        
        ρ_B(ũ_B) = ρ_min + (ρ_max - ρ_min) * σ(β(ũ_B - τ))
        
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
        
        logger.info(f"Adaptive threshold: ρ_B = {threshold:.4f} (for ũ_B = {uncertainty:.4f})")
        
        return threshold
    
    def select_blocks(
        self,
        probs: Dict[str, float],
        coverage_threshold: float
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Select blocks based on cumulative probability mass
        
        NO Top-K! Select until cumulative probability >= threshold
        
        Args:
            probs: Dict[block_id, probability]
            coverage_threshold: Coverage threshold
        
        Returns:
            selected_blocks: List of selected block IDs
            selection_info: Selection debug information
        """
        # Convert to arrays
        prob_array = np.array([probs[block_id] for block_id in self.block_ids])
        
        # Cumulative selection
        selected_ids, selected_probs, actual_coverage = cumulative_selection(
            prob_array,
            self.block_ids,
            coverage_threshold
        )
        
        logger.info(f"Selected {len(selected_ids)} blocks (coverage: {actual_coverage:.4f})")
        logger.info(f"Selected blocks: {selected_ids}")
        
        # Selection info
        selection_info = {
            "selected_blocks": selected_ids,
            "selected_probs": {
                block_id: prob
                for block_id, prob in zip(selected_ids, selected_probs)
            },
            "num_selected": len(selected_ids),
            "coverage_threshold": coverage_threshold,
            "actual_coverage": actual_coverage
        }
        
        return selected_ids, selection_info
    
    def route(
        self,
        v: np.ndarray,
        domain_hint: str = None
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Main routing function for Stage-1
        
        Args:
            v: Query representation vector (d,)
            domain_hint: Optional domain hint from Stage-0 (for prior injection)
        
        Returns:
            selected_blocks: List of selected block IDs
            debug_info: Debug information
        """
        logger.info("=" * 60)
        logger.info("STAGE-1: Adaptive Soft Block Routing")
        logger.info("=" * 60)
        
        # Step 1: Compute block relevance scores (with optional prior)
        scores = self.compute_block_scores(v, domain_hint=domain_hint)
        
        # Step 2: Compute block probability distribution
        probs = self.compute_block_probabilities(scores)
        
        # Step 3: Compute uncertainty
        uncertainty = self.compute_uncertainty(probs)
        
        # Step 4: Compute adaptive coverage threshold
        coverage_threshold = self.compute_adaptive_threshold(uncertainty)
        
        # Step 5: Select blocks based on cumulative probability
        selected_blocks, selection_info = self.select_blocks(probs, coverage_threshold)
        
        # Debug info
        debug_info = {
            "block_scores": scores,
            "block_probs": probs,
            "uncertainty": uncertainty,
            "coverage_threshold": coverage_threshold,
            "selection_info": selection_info,
            "all_blocks": self.block_ids,
            "temperature": self.temperature,
            "domain_hint": domain_hint,
            "lambda_prior": self.lambda_prior
        }
        
        logger.info("=" * 60)
        logger.info(f"Stage-1 Complete: {len(selected_blocks)} blocks selected")
        logger.info("=" * 60)
        
        return selected_blocks, debug_info

"""
Main Router Class
2-Stage Soft-Gated, Uncertainty-Aware Multi-Agent Router
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import os

from .utils import (
    load_config,
    load_or_create_prototypes,
    save_debug_info
)
from .stage0 import Stage0RepresentationGenerator
from .stage1 import Stage1BlockRouter
from .stage2 import Stage2RoleRouter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Router:
    """
    2-Stage Soft-Gated, Uncertainty-Aware Multi-Agent Router
    
    Input: question q (str)
    Output: initial agent set A0 (List[str])
    
    Pipeline:
        Stage-0: Representation Generation (v = α·Embed(q) + (1-α)·Embed(t))
        Stage-1: Adaptive Soft Block Routing (select blocks)
        Stage-2: (To be implemented) Role Selection within blocks
    """
    
    def __init__(
        self,
        config_path: str,
        embedding_model=None,
        llm_client=None
    ):
        """
        Initialize Router
        
        Args:
            config_path: Path to configuration YAML file
            embedding_model: Sentence embedding model (if None, will load from config)
            llm_client: LLM client for structured summary (if None, will create from config)
        """
        logger.info("=" * 80)
        logger.info("Initializing 2-Stage Uncertainty-Aware Multi-Agent Router")
        logger.info("=" * 80)
        
        # Load configuration
        self.config = load_config(config_path)
        logger.info(f"Configuration loaded from {config_path}")
        
        # Initialize embedding model
        if embedding_model is None:
            embedding_model = self._load_embedding_model()
        self.embedding_model = embedding_model
        logger.info(f"Embedding model: {self.config['stage0']['embedding_model']}")
        
        # Initialize LLM client
        if llm_client is None:
            llm_client = self._create_llm_client()
        self.llm_client = llm_client
        logger.info(f"LLM model: {self.config['stage0']['llm_model']}")
        
        # Load or generate prototypes
        self.block_prototypes, self.role_prototypes = load_or_create_prototypes(
            self.config,
            self.embedding_model
        )
        logger.info(f"Prototypes loaded: {len(self.block_prototypes)} blocks, {len(self.role_prototypes)} roles")
        
        # Initialize Stage-0
        self.stage0 = Stage0RepresentationGenerator(
            self.embedding_model,
            self.llm_client,
            self.config
        )
        logger.info("Stage-0 (Representation Generator) initialized")
        
        # Initialize Stage-1
        self.stage1 = Stage1BlockRouter(
            self.block_prototypes,
            self.config
        )
        logger.info("Stage-1 (Block Router) initialized")
        
        # Initialize Stage-2
        block_role_mapping = {
            block_id: self.config['blocks'][block_id]['roles']
            for block_id in self.config['blocks']
        }
        self.stage2 = Stage2RoleRouter(
            self.role_prototypes,
            block_role_mapping,
            self.config
        )
        logger.info("Stage-2 (Role Router) initialized")
        
        # Debug settings
        self.debug = self.config['debug']['verbose']
        self.save_debug = self.config['debug']['save_intermediate_results']
        self.debug_dir = self.config['debug']['output_dir']
        
        if self.save_debug:
            os.makedirs(self.debug_dir, exist_ok=True)
            logger.info(f"Debug output directory: {self.debug_dir}")
        
        logger.info("=" * 80)
        logger.info("Router initialization complete!")
        logger.info("=" * 80)
    
    def _load_embedding_model(self):
        """Load embedding model from config"""
        from sentence_transformers import SentenceTransformer
        
        model_name = self.config['stage0']['embedding_model']
        # Force CPU to avoid GPU OOM when vLLM is running
        model = SentenceTransformer(model_name, device='cpu')
        
        return model
    
    def _create_llm_client(self):
        """Create LLM client from config"""
        from .llm_client import create_llm_client
        
        # Create LLM client (will auto-detect OpenAI API keys)
        client = create_llm_client(self.config)
        
        return client
    
    def get_roles_from_blocks(self, block_ids: List[str]) -> List[str]:
        """
        Get all roles from selected blocks
        
        Args:
            block_ids: List of selected block IDs
        
        Returns:
            List of role IDs
        """
        roles = []
        for block_id in block_ids:
            block_roles = self.config['blocks'][block_id]['roles']
            roles.extend(block_roles)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_roles = []
        for role in roles:
            if role not in seen:
                seen.add(role)
                unique_roles.append(role)
        
        return unique_roles
    
    def route(
        self,
        question: str,
        use_llm_summary: bool = True,
        debug_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main routing function
        
        Args:
            question: Input question text
            use_llm_summary: Whether to use LLM for structured summary
            debug_id: Optional debug ID for saving intermediate results
        
        Returns:
            Dict containing:
                - selected_roles: List[str] (initial agent set A0)
                - selected_blocks: List[str]
                - debug_info: Dict with all intermediate results
        """
        logger.info("\n" + "=" * 80)
        logger.info("ROUTER: Processing Question")
        logger.info("=" * 80)
        logger.info(f"Question: {question[:100]}{'...' if len(question) > 100 else ''}")
        logger.info("=" * 80)
        
        # Stage-0: Generate representation
        v, stage0_debug = self.stage0.generate_representation(
            question,
            use_llm_summary=use_llm_summary
        )
        
        # Extract domain_hint from Stage-0 for prior injection
        domain_hint = None
        if stage0_debug.get('used_llm') and stage0_debug.get('summary'):
            domain_hint = stage0_debug['summary'].get('domain_hint', None)
        
        # Stage-1: Route to blocks (with domain prior)
        selected_blocks, stage1_debug = self.stage1.route(v, domain_hint=domain_hint)
        
        # Stage-2: Route to roles (within selected blocks)
        selected_roles, stage2_debug = self.stage2.route(v, selected_blocks)
        
        logger.info("\n" + "=" * 80)
        logger.info("ROUTER: Result Summary")
        logger.info("=" * 80)
        logger.info(f"Selected Blocks: {len(selected_blocks)} blocks")
        logger.info(f"   {selected_blocks}")
        logger.info(f"   └─ Block Uncertainty: {stage1_debug['uncertainty']:.4f}")
        logger.info(f"   └─ Block Coverage: {stage1_debug['coverage_threshold']:.4f}")
        logger.info(f"Selected Agents (A_0): {len(selected_roles)} agents")
        logger.info(f"   {selected_roles}")
        logger.info(f"   └─ Role Uncertainty: {stage2_debug['uncertainty']:.4f}")
        logger.info(f"   └─ Role Coverage: {stage2_debug['coverage_threshold']:.4f}")
        logger.info("=" * 80)
        logger.info(f"FINAL: {len(selected_roles)} agents will participate")
        logger.info("=" * 80 + "\n")
        
        # Build final agent list with probabilities (sorted by relevance)
        role_probs = stage2_debug['selection_info']['selected_probs']
        final_agents = [
            {
                "agent": role_id,
                "probability": float(role_probs[role_id])
            }
            for role_id in selected_roles
        ]
        # Sort by probability (descending)
        final_agents.sort(key=lambda x: x['probability'], reverse=True)
        
        # Compile result
        result = {
            # === FINAL OUTPUT (Most Important!) ===
            "final_agents": final_agents,  # List[{agent, probability}] - 최종 참여 에이전트
            "num_agents": len(selected_roles),  # Total number of agents
            
            # === Detailed Information ===
            "selected_roles": selected_roles,  # List of role IDs (for backward compatibility)
            "role_probabilities": stage2_debug['selection_info']['selected_probs'],  # Dict: {role_id: p_r}
            "selected_blocks": selected_blocks,  # List of block IDs
            "block_probabilities": stage1_debug['selection_info']['selected_probs'],  # Dict: {block_id: p_B}
            "num_blocks": len(selected_blocks),  # Total number of selected blocks
            
            # === Coverage & Uncertainty ===
            "coverage": {
                "block_coverage": stage1_debug['selection_info']['actual_coverage'],
                "role_coverage": stage2_debug['selection_info']['actual_coverage'],
                "block_threshold": stage1_debug['coverage_threshold'],
                "role_threshold": stage2_debug['coverage_threshold']
            },
            "uncertainty": {
                "block_uncertainty": stage1_debug['uncertainty'],
                "role_uncertainty": stage2_debug['uncertainty']
            },
            
            # === Debug Info ===
            "debug_info": {
                "stage0": stage0_debug,
                "stage1": stage1_debug,
                "stage2": stage2_debug,
                "question": question
            }
        }
        
        # Save debug info if enabled
        if self.save_debug and debug_id:
            debug_path = os.path.join(self.debug_dir, f"{debug_id}.json")
            save_debug_info(result, debug_path)
        
        return result
    
    def batch_route(
        self,
        questions: List[str],
        use_llm_summary: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Batch routing for multiple questions
        
        Args:
            questions: List of questions
            use_llm_summary: Whether to use LLM for structured summary
        
        Returns:
            List of routing results
        """
        results = []
        
        for i, question in enumerate(questions):
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing question {i+1}/{len(questions)}")
            logger.info(f"{'='*80}")
            
            result = self.route(
                question,
                use_llm_summary=use_llm_summary,
                debug_id=f"batch_{i}"
            )
            
            results.append(result)
        
        return results

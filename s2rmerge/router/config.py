"""Router configuration, read from the per-benchmark YAML files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

from s2rmerge.paths import PROTOTYPE_DIR


@dataclass(frozen=True)
class CoverageConfig:
    """Bounds and shape of the adaptive coverage threshold."""

    rho_min: float
    rho_max: float
    tau: float
    kappa: float


@dataclass(frozen=True)
class StageConfig:
    """Softmax temperature and coverage rule for one routing stage."""

    temperature: float
    coverage: CoverageConfig
    epsilon: float = 1e-10


@dataclass(frozen=True)
class Block:
    """A domain block: a description to embed and the roles it contains."""

    id: str
    description: str
    roles: List[str]


@dataclass(frozen=True)
class RouterConfig:
    """Everything the router needs for one benchmark."""

    name: str
    alpha: float
    embedding_model: str
    llm_model: str
    summary_prompt: str
    lambda_prior: float
    domain_priors: Dict[str, Dict[str, float]]
    stage1: StageConfig
    stage2: StageConfig
    blocks: Dict[str, Block]
    role_descriptions: Dict[str, str]
    prototype_prefix: str

    @property
    def block_ids(self) -> List[str]:
        return list(self.blocks)

    @property
    def role_ids(self) -> List[str]:
        seen: List[str] = []
        for block in self.blocks.values():
            for role in block.roles:
                if role not in seen:
                    seen.append(role)
        return seen

    @property
    def block_prototype_path(self) -> Path:
        return PROTOTYPE_DIR / f"{self.prototype_prefix}_block.npy"

    @property
    def role_prototype_path(self) -> Path:
        return PROTOTYPE_DIR / f"{self.prototype_prefix}_role.npy"

    def roles_of(self, block_id: str) -> List[str]:
        return self.blocks[block_id].roles


def _coverage(raw: Dict[str, Any], suffix: str = "") -> CoverageConfig:
    def value(key: str) -> float:
        return float(raw[f"{key}{suffix}"])

    return CoverageConfig(
        rho_min=value("rho_min"),
        rho_max=value("rho_max"),
        tau=value("tau"),
        kappa=value("kappa"),
    )


def load_router_config(path: Path) -> RouterConfig:
    """Parse a router YAML file, failing loudly on a missing key."""
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    stage0 = raw["stage0"]
    stage1 = raw["stage1"]
    stage2 = raw["stage2"]

    blocks = {
        block_id: Block(
            id=block_id,
            description=block["description"],
            roles=list(block["roles"]),
        )
        for block_id, block in raw["blocks"].items()
    }

    return RouterConfig(
        name=raw.get("name", Path(path).stem),
        alpha=float(stage0["alpha"]),
        embedding_model=stage0["embedding_model"],
        llm_model=stage0["llm_model"],
        summary_prompt=stage0["summary_prompt"],
        lambda_prior=float(stage1.get("lambda_prior", 0.0)),
        domain_priors=stage1.get("domain_priors", {}) or {},
        stage1=StageConfig(
            temperature=float(stage1["temperature_block"]),
            coverage=_coverage(stage1["coverage"]),
            epsilon=float(stage1.get("epsilon", 1e-10)),
        ),
        stage2=StageConfig(
            temperature=float(stage2["temperature_role"]),
            coverage=_coverage(stage2["coverage"], suffix="_role"),
            epsilon=float(stage2.get("epsilon", 1e-10)),
        ),
        blocks=blocks,
        role_descriptions=dict(raw["role_descriptions"]),
        prototype_prefix=raw.get("prototype_prefix", Path(path).stem),
    )

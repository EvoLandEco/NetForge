from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

SPECIES = ("pig", "cattle", "poultry", "mixed")
TYPE_FARM = "farm"
TYPE_REGION = "region"
NODE_FEATURE_COLUMNS = [
    "xco",
    "yco",
    "num_farms",
    "total_animals",
    "count_ft_pig",
    "count_ft_cattle",
    "count_ft_poultry",
    "count_ft_mixed",
]
PROFILE_DEFAULTS = {
    "toy": {"n_farms": 120, "n_regions": 8, "farm_blocks": 6, "region_blocks": 2, "days": 60},
    "small": {"n_farms": 600, "n_regions": 18, "farm_blocks": 10, "region_blocks": 3, "days": 140},
    "medium": {"n_farms": 1800, "n_regions": 36, "farm_blocks": 18, "region_blocks": 4, "days": 280},
    "large": {"n_farms": 4500, "n_regions": 60, "farm_blocks": 28, "region_blocks": 6, "days": 365},
}


@dataclass
class SimulationConfig:
    output_root: str
    dataset: str
    seed: int
    profile: str
    n_farms: int
    n_regions: int
    farm_blocks: int
    region_blocks: int
    days: int
    start_date: str
    directed: bool
    weight_name: str
    write_weight_npy: bool
    write_edge_truth: bool
    write_block_activity_truth: bool
    write_block_pair_truth: bool
    geography_width_km: float = 280.0
    geography_height_km: float = 220.0
    farm_dispersion_km: float = 12.0
    distance_scale_km_ff: float = 38.0
    distance_scale_km_fr: float = 55.0
    distance_scale_km_rf: float = 55.0
    distance_scale_km_rr: float = 90.0
    farm_activity_p_init: float = 0.55
    farm_activity_p01: float = 0.22
    farm_activity_p11: float = 0.88
    region_activity_p_init: float = 0.75
    region_activity_p01: float = 0.35
    region_activity_p11: float = 0.94
    activity_concentration: float = 35.0
    base_new_ff: float = 0.018
    base_new_fr: float = 0.010
    base_new_rf: float = 0.013
    base_new_rr: float = 0.004
    base_persist_ff: float = 0.62
    base_persist_fr: float = 0.48
    base_persist_rf: float = 0.52
    base_persist_rr: float = 0.40
    base_reactivate_ff: float = 0.10
    base_reactivate_fr: float = 0.07
    base_reactivate_rf: float = 0.08
    base_reactivate_rr: float = 0.05
    weight_mean_ff: float = 28.0
    weight_mean_fr: float = 40.0
    weight_mean_rf: float = 48.0
    weight_mean_rr: float = 65.0
    weight_alpha_ff: float = 0.45
    weight_alpha_fr: float = 0.33
    weight_alpha_rf: float = 0.30
    weight_alpha_rr: float = 0.26
    season_amplitude: float = 0.18
    weekend_penalty: float = 0.82
    same_region_bonus: float = 0.88
    ft_similarity_weight: float = 0.35
    exact_new_dyad_threshold: int = 5000
    proposal_attempts_per_edge: int = 80


@dataclass
class NodeRecord:
    node_id: int
    node_type: str
    true_block: int
    role_block: int
    region_id: int
    region_code: str
    x_m: float
    y_m: float
    num_farms: float
    total_animals: float
    feature_counts: dict[str, float]
    out_mass: float
    in_mass: float
    specialty: str
    coord_source: str
    priority: str
    cr_code: str
    trade_species: str
    diersoort: str
    diergroep: str
    diergroeplang: str
    btypnl: str
    bedrtype: str
    ubn: str


@dataclass
class BlockActivityParams:
    p_init: float
    p01: float
    p11: float


@dataclass
class BlockPairParams:
    block_u: int
    block_v: int
    source_type: str
    target_type: str
    new_scale: float
    persist_prob: float
    reactivate_prob: float
    weight_mean: float
    weight_alpha: float


@dataclass
class EdgeEvent:
    u: int
    i: int
    ts: int
    idx: int
    weight: int
    category: str
    block_u: int
    block_v: int
    source_type: str
    target_type: str


class EdgeScorer:
    def __init__(self, *, nodes: dict[int, NodeRecord], cfg: SimulationConfig) -> None:
        self.nodes = nodes
        self.cfg = cfg
        self._pair_cache: dict[tuple[int, int], float] = {}

    def canonical(self, u: int, v: int) -> tuple[int, int]:
        if self.cfg.directed:
            return int(u), int(v)
        a, b = int(u), int(v)
        return (a, b) if a <= b else (b, a)

    def channel(self, u: int, v: int) -> tuple[str, str]:
        left = self.nodes[int(u)].node_type
        right = self.nodes[int(v)].node_type
        if self.cfg.directed:
            return left, right
        return (left, right) if left <= right else (right, left)

    def distance_scale_km(self, u: int, v: int) -> float:
        left, right = self.channel(u, v)
        if left == TYPE_FARM and right == TYPE_FARM:
            return float(self.cfg.distance_scale_km_ff)
        if left == TYPE_FARM and right == TYPE_REGION:
            return float(self.cfg.distance_scale_km_fr)
        if left == TYPE_REGION and right == TYPE_FARM:
            return float(self.cfg.distance_scale_km_rf)
        return float(self.cfg.distance_scale_km_rr)

    def pair_modifier(self, u: int, v: int) -> float:
        key = self.canonical(u, v)
        cached = self._pair_cache.get(key)
        if cached is not None:
            return cached
        left = self.nodes[int(u)]
        right = self.nodes[int(v)]
        dx = float(left.x_m - right.x_m)
        dy = float(left.y_m - right.y_m)
        dist_km = math.sqrt(dx * dx + dy * dy) / 1000.0
        geo = math.exp(-dist_km / max(self.distance_scale_km(u, v), 1e-6))

        left_vec = np.asarray([left.feature_counts[name] for name in SPECIES], dtype=float)
        right_vec = np.asarray([right.feature_counts[name] for name in SPECIES], dtype=float)
        left_norm = float(np.linalg.norm(left_vec))
        right_norm = float(np.linalg.norm(right_vec))
        if left_norm > 0.0 and right_norm > 0.0:
            ft_cos = float(np.dot(left_vec, right_vec) / (left_norm * right_norm))
            ft_cos = max(-1.0, min(1.0, ft_cos))
        else:
            ft_cos = 0.0
        ft_factor = 1.0 - float(self.cfg.ft_similarity_weight) + float(self.cfg.ft_similarity_weight) * (0.5 + 0.5 * max(ft_cos, 0.0))
        same_region = 1.0 if int(left.region_id) == int(right.region_id) else 0.0
        region_factor = 1.0 - (1.0 - float(self.cfg.same_region_bonus)) * (1.0 - same_region)
        value = float(max(1e-8, min(1.0, geo * ft_factor * region_factor)))
        self._pair_cache[key] = value
        return value

    def edge_score(self, u: int, v: int) -> float:
        left = self.nodes[int(u)]
        right = self.nodes[int(v)]
        mass = math.sqrt(max(left.out_mass, 1e-6) * max(right.in_mass, 1e-6))
        return float(max(1e-8, mass * self.pair_modifier(u, v)))


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _logit(value: float) -> float:
    clipped = min(max(float(value), 1e-9), 1.0 - 1e-9)
    return math.log(clipped / (1.0 - clipped))


def _expit(value: float) -> float:
    if value >= 0.0:
        z_value = math.exp(-value)
        return 1.0 / (1.0 + z_value)
    z_value = math.exp(value)
    return z_value / (1.0 + z_value)


def _safe_beta(rng: np.random.Generator, mean: float, concentration: float) -> float:
    mean_value = min(max(float(mean), 1e-6), 1.0 - 1e-6)
    conc = max(float(concentration), 1e-3)
    alpha = mean_value * conc
    beta = (1.0 - mean_value) * conc
    return float(rng.beta(alpha, beta))


def _sample_nb2(rng: np.random.Generator, mean: float, alpha: float) -> int:
    mean_value = max(float(mean), 0.0)
    alpha_value = max(float(alpha), 0.0)
    if mean_value <= 1e-12:
        return 0
    if alpha_value <= 1e-10:
        return int(rng.poisson(mean_value))
    shape = 1.0 / alpha_value
    if not np.isfinite(shape) or shape >= 1e8:
        return int(rng.poisson(mean_value))
    latent_rate = float(rng.gamma(shape=shape, scale=mean_value / shape))
    return int(rng.poisson(max(latent_rate, 0.0)))


def _sample_shifted_nb(rng: np.random.Generator, mean: float, alpha: float) -> int:
    shifted_mean = max(float(mean) - 1.0, 1e-6)
    return int(1 + _sample_nb2(rng, shifted_mean, alpha))


def _gumbel_top_k_indices(weights: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    if k <= 0 or len(weights) == 0:
        return np.asarray([], dtype=int)
    safe = np.clip(np.asarray(weights, dtype=float), 1e-12, None)
    scores = np.log(safe) + rng.gumbel(size=len(safe))
    if k >= len(scores):
        return np.argsort(scores)[::-1]
    picked = np.argpartition(scores, -int(k))[-int(k):]
    return picked[np.argsort(scores[picked])[::-1]]


def _ordinal_range(start_date_text: str, days: int) -> list[int]:
    start = date.fromisoformat(str(start_date_text))
    return [int((start + timedelta(days=offset)).toordinal()) for offset in range(int(days))]


def _season_multiplier(ts_ordinal: int, cfg: SimulationConfig) -> float:
    current_date = date.fromordinal(int(ts_ordinal))
    day_index = int(ts_ordinal - date.fromisoformat(cfg.start_date).toordinal())
    annual = math.sin(2.0 * math.pi * day_index / max(int(cfg.days), 1))
    weekly = math.cos(2.0 * math.pi * current_date.weekday() / 7.0)
    value = 1.0 + float(cfg.season_amplitude) * annual + 0.08 * weekly
    if current_date.weekday() >= 5:
        value *= float(cfg.weekend_penalty)
    return float(max(0.15, value))


def _normalise(weights: np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if values.size == 0:
        return values
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(len(values), 1.0 / len(values), dtype=float)
    return values / total


def _grid_points(n_points: int, *, width_km: float, height_km: float, rng: np.random.Generator) -> np.ndarray:
    cols = int(math.ceil(math.sqrt(n_points * width_km / max(height_km, 1e-6))))
    rows = int(math.ceil(n_points / max(cols, 1)))
    xs = np.linspace(0.08 * width_km, 0.92 * width_km, cols)
    ys = np.linspace(0.08 * height_km, 0.92 * height_km, rows)
    points = []
    for y_value in ys:
        for x_value in xs:
            jitter_x = float(rng.normal(scale=0.03 * width_km / max(cols, 1)))
            jitter_y = float(rng.normal(scale=0.03 * height_km / max(rows, 1)))
            points.append((max(0.0, x_value + jitter_x), max(0.0, y_value + jitter_y)))
            if len(points) >= int(n_points):
                break
        if len(points) >= int(n_points):
            break
    return np.asarray(points, dtype=float)


def _species_metadata(species: str) -> dict[str, str]:
    if species == "pig":
        return {
            "trade_species": "pig",
            "diersoort": "varken",
            "diergroep": "monogastric",
            "diergroeplang": "pig holdings",
            "BtypNL": "pig_farm",
            "bedrtype": "breeding",
        }
    if species == "cattle":
        return {
            "trade_species": "cattle",
            "diersoort": "rund",
            "diergroep": "ruminant",
            "diergroeplang": "cattle holdings",
            "BtypNL": "cattle_farm",
            "bedrtype": "dairy",
        }
    if species == "poultry":
        return {
            "trade_species": "poultry",
            "diersoort": "pluimvee",
            "diergroep": "avian",
            "diergroeplang": "poultry holdings",
            "BtypNL": "poultry_farm",
            "bedrtype": "broiler",
        }
    return {
        "trade_species": "mixed",
        "diersoort": "mixed",
        "diergroep": "mixed",
        "diergroeplang": "mixed livestock",
        "BtypNL": "mixed_farm",
        "bedrtype": "mixed",
    }


def _species_dirichlet_alpha(species: str) -> np.ndarray:
    lookup = {
        "pig": np.asarray([7.0, 1.5, 0.9, 1.1], dtype=float),
        "cattle": np.asarray([1.3, 7.5, 0.8, 1.2], dtype=float),
        "poultry": np.asarray([0.8, 1.1, 7.8, 1.2], dtype=float),
        "mixed": np.asarray([2.4, 2.2, 2.0, 4.2], dtype=float),
    }
    return lookup[str(species)].copy()


def _farm_priority(species: str, herd_size: float) -> str:
    if herd_size >= 2200:
        return "high"
    if herd_size >= 900:
        return "medium"
    if species == "mixed":
        return "medium"
    return "low"


def _herd_size_log_mean(species: str) -> float:
    lookup = {
        "pig": math.log(1400.0),
        "cattle": math.log(240.0),
        "poultry": math.log(4200.0),
        "mixed": math.log(650.0),
    }
    return float(lookup[str(species)])


def _sample_species(rng: np.random.Generator, probs: np.ndarray) -> str:
    index = int(rng.choice(np.arange(len(SPECIES)), p=np.asarray(probs, dtype=float)))
    return str(SPECIES[index])


def _canonical_edge(u: int, v: int, *, directed: bool) -> tuple[int, int]:
    if directed:
        return int(u), int(v)
    a, b = int(u), int(v)
    return (a, b) if a <= b else (b, a)


def _canonical_block_pair(block_u: int, block_v: int, *, directed: bool) -> tuple[int, int]:
    if directed:
        return int(block_u), int(block_v)
    a, b = int(block_u), int(block_v)
    return (a, b) if a <= b else (b, a)


def _channel_base(cfg: SimulationConfig, left_type: str, right_type: str, name: str) -> float:
    if left_type == TYPE_FARM and right_type == TYPE_FARM:
        return float(getattr(cfg, f"{name}_ff"))
    if left_type == TYPE_FARM and right_type == TYPE_REGION:
        return float(getattr(cfg, f"{name}_fr"))
    if left_type == TYPE_REGION and right_type == TYPE_FARM:
        return float(getattr(cfg, f"{name}_rf"))
    return float(getattr(cfg, f"{name}_rr"))


def _build_region_nodes(cfg: SimulationConfig, rng: np.random.Generator) -> tuple[list[NodeRecord], dict[int, tuple[float, float]], dict[int, int], np.ndarray]:
    points = _grid_points(cfg.n_regions, width_km=cfg.geography_width_km, height_km=cfg.geography_height_km, rng=rng)
    region_nodes: list[NodeRecord] = []
    region_centroids: dict[int, tuple[float, float]] = {}
    region_block_map: dict[int, int] = {}
    region_size_weights = _normalise(rng.gamma(shape=2.2, scale=1.0, size=int(cfg.n_regions)))
    for region_id in range(int(cfg.n_regions)):
        x_km, y_km = points[region_id]
        region_code = f"CR_{region_id:03d}"
        block_id = int(cfg.farm_blocks + (region_id % max(int(cfg.region_blocks), 1)))
        region_block_map[region_id] = block_id
        region_centroids[region_id] = (1000.0 * float(x_km), 1000.0 * float(y_km))
        region_nodes.append(
            NodeRecord(
                node_id=region_id,
                node_type=TYPE_REGION,
                true_block=block_id,
                role_block=block_id - int(cfg.farm_blocks),
                region_id=region_id,
                region_code=region_code,
                x_m=1000.0 * float(x_km),
                y_m=1000.0 * float(y_km),
                num_farms=0.0,
                total_animals=0.0,
                feature_counts={name: 0.0 for name in SPECIES},
                out_mass=1.0,
                in_mass=1.0,
                specialty="mixed",
                coord_source="simulated",
                priority="high",
                cr_code=region_code,
                trade_species="mixed",
                diersoort="mixed",
                diergroep="regional",
                diergroeplang="regional aggregation",
                btypnl="region_hub",
                bedrtype="hub",
                ubn="",
            )
        )
    return region_nodes, region_centroids, region_block_map, region_size_weights


def _farm_block_preferences(cfg: SimulationConfig, region_block_map: dict[int, int], rng: np.random.Generator) -> np.ndarray:
    region_blocks = max(int(cfg.region_blocks), 1)
    farm_blocks = max(int(cfg.farm_blocks), 1)
    weights = np.zeros((int(cfg.n_regions), farm_blocks), dtype=float)
    centers = np.linspace(0, max(farm_blocks - 1, 0), region_blocks)
    spread = max(1.0, 0.18 * farm_blocks)
    block_positions = np.arange(farm_blocks, dtype=float)
    for region_id in range(int(cfg.n_regions)):
        region_role_block = int(region_block_map[region_id] - int(cfg.farm_blocks))
        center = centers[region_role_block % len(centers)] if len(centers) else 0.0
        base = np.exp(-((block_positions - center) ** 2) / (2.0 * spread * spread))
        noise = rng.gamma(shape=2.0, scale=1.0, size=farm_blocks)
        probs = base * noise + 1e-6
        weights[region_id] = probs / probs.sum()
    return weights


def _farm_block_species_weights(cfg: SimulationConfig, rng: np.random.Generator) -> dict[int, np.ndarray]:
    weights: dict[int, np.ndarray] = {}
    n_species = len(SPECIES)
    for block_id in range(int(cfg.farm_blocks)):
        dominant = block_id % n_species
        alpha = np.full(n_species, 0.8, dtype=float)
        alpha[dominant] = 6.5
        weights[block_id] = rng.dirichlet(alpha)
    return weights


def _build_farm_nodes(
    cfg: SimulationConfig,
    rng: np.random.Generator,
    *,
    region_centroids: dict[int, tuple[float, float]],
    region_block_map: dict[int, int],
    region_size_weights: np.ndarray,
) -> list[NodeRecord]:
    farms: list[NodeRecord] = []
    block_preferences = _farm_block_preferences(cfg, region_block_map, rng)
    species_weights = _farm_block_species_weights(cfg, rng)
    region_probs = _normalise(region_size_weights)
    for farm_offset in range(int(cfg.n_farms)):
        node_id = int(cfg.n_regions + farm_offset)
        region_id = int(rng.choice(np.arange(int(cfg.n_regions)), p=region_probs))
        block_id = int(rng.choice(np.arange(int(cfg.farm_blocks)), p=block_preferences[region_id]))
        species = _sample_species(rng, species_weights[block_id])
        herd_size = float(rng.lognormal(mean=_herd_size_log_mean(species), sigma=0.48))
        trade_volume = float(rng.gamma(shape=4.8, scale=3.4))
        mix = rng.dirichlet(_species_dirichlet_alpha(species))
        feature_counts = {name: float(trade_volume * mix[index]) for index, name in enumerate(SPECIES)}
        centroid_x, centroid_y = region_centroids[region_id]
        radial_scale_m = 1000.0 * float(cfg.farm_dispersion_km)
        dx = float(rng.normal(scale=radial_scale_m * 0.55))
        dy = float(rng.normal(scale=radial_scale_m * 0.55))
        x_value = float(max(0.0, centroid_x + dx))
        y_value = float(max(0.0, centroid_y + dy))
        labels = _species_metadata(species)
        out_mass = float(math.sqrt(max(herd_size, 1.0)) * rng.lognormal(mean=0.0, sigma=0.18))
        in_mass = float(math.sqrt(max(herd_size, 1.0)) * rng.lognormal(mean=0.0, sigma=0.18))
        farms.append(
            NodeRecord(
                node_id=node_id,
                node_type=TYPE_FARM,
                true_block=block_id,
                role_block=block_id,
                region_id=region_id,
                region_code=f"CR_{region_id:03d}",
                x_m=x_value,
                y_m=y_value,
                num_farms=1.0,
                total_animals=herd_size,
                feature_counts=feature_counts,
                out_mass=out_mass,
                in_mass=in_mass,
                specialty=species,
                coord_source="simulated",
                priority=_farm_priority(species, herd_size),
                cr_code=f"CR_{region_id:03d}",
                trade_species=labels["trade_species"],
                diersoort=labels["diersoort"],
                diergroep=labels["diergroep"],
                diergroeplang=labels["diergroeplang"],
                btypnl=labels["BtypNL"],
                bedrtype=labels["bedrtype"],
                ubn=f"{region_id + 1:03d}{farm_offset + 1:06d}",
            )
        )
    return farms


def _finalise_region_nodes(region_nodes: list[NodeRecord], farms: list[NodeRecord]) -> list[NodeRecord]:
    farms_by_region: dict[int, list[NodeRecord]] = defaultdict(list)
    for farm in farms:
        farms_by_region[int(farm.region_id)].append(farm)
    updated_regions: list[NodeRecord] = []
    for region in region_nodes:
        members = farms_by_region.get(int(region.region_id), [])
        if members:
            num_farms = float(len(members))
            total_animals = float(sum(member.total_animals for member in members))
            feature_counts = {name: float(sum(member.feature_counts[name] for member in members)) for name in SPECIES}
            dominant = str(Counter(member.specialty for member in members).most_common(1)[0][0])
            labels = _species_metadata(dominant)
            out_mass = float(math.sqrt(max(total_animals, 1.0)) * 0.90)
            in_mass = float(math.sqrt(max(total_animals, 1.0)) * 1.05)
        else:
            num_farms = 0.0
            total_animals = 0.0
            feature_counts = {name: 0.0 for name in SPECIES}
            dominant = "mixed"
            labels = _species_metadata(dominant)
            out_mass = 1.0
            in_mass = 1.0
        updated_regions.append(
            NodeRecord(
                node_id=region.node_id,
                node_type=region.node_type,
                true_block=region.true_block,
                role_block=region.role_block,
                region_id=region.region_id,
                region_code=region.region_code,
                x_m=region.x_m,
                y_m=region.y_m,
                num_farms=num_farms,
                total_animals=total_animals,
                feature_counts=feature_counts,
                out_mass=out_mass,
                in_mass=in_mass,
                specialty=dominant,
                coord_source=region.coord_source,
                priority=region.priority,
                cr_code=region.cr_code,
                trade_species=labels["trade_species"],
                diersoort=labels["diersoort"],
                diergroep=labels["diergroep"],
                diergroeplang=labels["diergroeplang"],
                btypnl=labels["BtypNL"],
                bedrtype=labels["bedrtype"],
                ubn=region.ubn,
            )
        )
    return updated_regions


def _node_frames(nodes: list[NodeRecord]) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    ordered = sorted(nodes, key=lambda item: item.node_id)
    node_map_rows = []
    feature_rows = []
    for node in ordered:
        node_map_rows.append(
            {
                "node_id": int(node.node_id),
                "type": str(node.node_type),
                "ubn": str(node.ubn),
                "corop": str(node.region_code),
                "coord_source": str(node.coord_source),
                "priority": str(node.priority),
                "CR_code": str(node.cr_code),
                "trade_species": str(node.trade_species),
                "diersoort": str(node.diersoort),
                "diergroep": str(node.diergroep),
                "diergroeplang": str(node.diergroeplang),
                "BtypNL": str(node.btypnl),
                "bedrtype": str(node.bedrtype),
                "specialty": str(node.specialty),
                "true_block": int(node.true_block),
                "true_role_block": int(node.role_block),
                "region_id": int(node.region_id),
            }
        )
        feature_rows.append(
            [
                float(node.x_m),
                float(node.y_m),
                float(node.num_farms),
                float(node.total_animals),
                float(node.feature_counts.get("pig", 0.0)),
                float(node.feature_counts.get("cattle", 0.0)),
                float(node.feature_counts.get("poultry", 0.0)),
                float(node.feature_counts.get("mixed", 0.0)),
            ]
        )
    node_map = pd.DataFrame(node_map_rows).sort_values("node_id").reset_index(drop=True)
    features = np.asarray(feature_rows, dtype=float)
    schema = {"node_feature_columns_in_order": list(NODE_FEATURE_COLUMNS), "node_row_offset": 0}
    return node_map, features, schema


def _block_types(nodes: list[NodeRecord]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for node in nodes:
        mapping[int(node.true_block)] = str(node.node_type)
    return mapping


def _sample_block_activity_params(cfg: SimulationConfig, rng: np.random.Generator, block_type: str) -> BlockActivityParams:
    if block_type == TYPE_REGION:
        p_init = _safe_beta(rng, cfg.region_activity_p_init, cfg.activity_concentration)
        p01 = _safe_beta(rng, cfg.region_activity_p01, cfg.activity_concentration)
        p11 = _safe_beta(rng, cfg.region_activity_p11, cfg.activity_concentration)
    else:
        p_init = _safe_beta(rng, cfg.farm_activity_p_init, cfg.activity_concentration)
        p01 = _safe_beta(rng, cfg.farm_activity_p01, cfg.activity_concentration)
        p11 = _safe_beta(rng, cfg.farm_activity_p11, cfg.activity_concentration)
    if p11 < p01:
        p11 = min(0.995, max(p01 + 0.05, p11))
    return BlockActivityParams(p_init=float(p_init), p01=float(p01), p11=float(p11))


def _build_activity_params(cfg: SimulationConfig, rng: np.random.Generator, block_types: dict[int, str]) -> dict[int, BlockActivityParams]:
    return {int(block_id): _sample_block_activity_params(cfg, rng, block_type) for block_id, block_type in sorted(block_types.items())}


def _sample_block_activity_states(
    cfg: SimulationConfig,
    rng: np.random.Generator,
    *,
    timeline: list[int],
    activity_params: dict[int, BlockActivityParams],
) -> dict[int, dict[int, bool]]:
    block_ids = sorted(activity_params)
    current = {block_id: bool(rng.random() < activity_params[block_id].p_init) for block_id in block_ids}
    out: dict[int, dict[int, bool]] = {}
    for ts_value in timeline:
        out[int(ts_value)] = {int(block_id): bool(current[block_id]) for block_id in block_ids}
        next_state: dict[int, bool] = {}
        for block_id in block_ids:
            params = activity_params[block_id]
            probability = params.p11 if current[block_id] else params.p01
            next_state[block_id] = bool(rng.random() < probability)
        current = next_state
    return out


def _build_block_pair_params(cfg: SimulationConfig, rng: np.random.Generator, block_types: dict[int, str]) -> dict[tuple[int, int], BlockPairParams]:
    params: dict[tuple[int, int], BlockPairParams] = {}
    blocks = sorted(block_types)
    for block_u in blocks:
        for block_v in blocks:
            if not cfg.directed and block_u > block_v:
                continue
            source_type = str(block_types[block_u])
            target_type = str(block_types[block_v])
            same_block_bonus = 1.18 if int(block_u) == int(block_v) else 1.0
            new_scale = _channel_base(cfg, source_type, target_type, "base_new") * same_block_bonus * math.exp(float(rng.normal(scale=0.35)))
            persist_prob = _expit(_logit(_channel_base(cfg, source_type, target_type, "base_persist")) + float(rng.normal(scale=0.32)))
            reactivate_prob = _expit(_logit(_channel_base(cfg, source_type, target_type, "base_reactivate")) + float(rng.normal(scale=0.34)))
            weight_mean = _channel_base(cfg, source_type, target_type, "weight_mean") * math.exp(float(rng.normal(scale=0.30)))
            weight_alpha = max(0.02, _channel_base(cfg, source_type, target_type, "weight_alpha") * math.exp(float(rng.normal(scale=0.15))))
            params[(int(block_u), int(block_v))] = BlockPairParams(
                block_u=int(block_u),
                block_v=int(block_v),
                source_type=source_type,
                target_type=target_type,
                new_scale=float(new_scale),
                persist_prob=float(min(max(persist_prob, 0.01), 0.995)),
                reactivate_prob=float(min(max(reactivate_prob, 0.0), 0.95)),
                weight_mean=float(max(weight_mean, 1.2)),
                weight_alpha=float(weight_alpha),
            )
    return params


def _build_block_node_lookup(nodes: list[NodeRecord]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    grouped: dict[int, list[NodeRecord]] = defaultdict(list)
    for node in nodes:
        grouped[int(node.true_block)].append(node)
    node_ids_by_block: dict[int, np.ndarray] = {}
    out_probs_by_block: dict[int, np.ndarray] = {}
    in_probs_by_block: dict[int, np.ndarray] = {}
    for block_id, members in grouped.items():
        ordered = sorted(members, key=lambda item: item.node_id)
        node_ids = np.asarray([int(node.node_id) for node in ordered], dtype=int)
        out_probs = _normalise(np.asarray([float(node.out_mass) for node in ordered], dtype=float))
        in_probs = _normalise(np.asarray([float(node.in_mass) for node in ordered], dtype=float))
        node_ids_by_block[int(block_id)] = node_ids
        out_probs_by_block[int(block_id)] = out_probs
        in_probs_by_block[int(block_id)] = in_probs
    return node_ids_by_block, out_probs_by_block, in_probs_by_block


def _sample_existing_edges(
    candidates: list[tuple[int, int]],
    *,
    k: int,
    scorer: EdgeScorer,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    if k <= 0 or not candidates:
        return []
    if k >= len(candidates):
        return list(candidates)
    weights = np.asarray([scorer.edge_score(edge[0], edge[1]) for edge in candidates], dtype=float)
    picked = _gumbel_top_k_indices(weights, k, rng)
    return [candidates[int(index)] for index in picked]


def _feasible_pair_capacity(source_ids: np.ndarray, target_ids: np.ndarray, *, same_block: bool, directed: bool) -> int:
    n_source = int(len(source_ids))
    n_target = int(len(target_ids))
    if directed:
        capacity = n_source * n_target
        if same_block:
            capacity -= n_source
        return max(capacity, 0)
    if same_block:
        return max(n_source * max(n_source - 1, 0) // 2, 0)
    return max(n_source * n_target, 0)


def _enumerate_new_dyads(
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    *,
    same_block: bool,
    directed: bool,
    seen_group: set[tuple[int, int]],
    scorer: EdgeScorer,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    candidates: list[tuple[int, int]] = []
    weights: list[float] = []
    if directed:
        for source in source_ids.tolist():
            for target in target_ids.tolist():
                if same_block and int(source) == int(target):
                    continue
                edge = _canonical_edge(int(source), int(target), directed=directed)
                if edge in seen_group:
                    continue
                candidates.append(edge)
                weights.append(float(scorer.edge_score(edge[0], edge[1])))
    else:
        if same_block:
            local = source_ids.tolist()
            for left_index in range(len(local)):
                source = int(local[left_index])
                for right_index in range(left_index + 1, len(local)):
                    target = int(local[right_index])
                    edge = _canonical_edge(source, target, directed=False)
                    if edge in seen_group:
                        continue
                    candidates.append(edge)
                    weights.append(float(scorer.edge_score(edge[0], edge[1])))
        else:
            for source in source_ids.tolist():
                for target in target_ids.tolist():
                    edge = _canonical_edge(int(source), int(target), directed=False)
                    if edge in seen_group:
                        continue
                    candidates.append(edge)
                    weights.append(float(scorer.edge_score(edge[0], edge[1])))
    return candidates, np.asarray(weights, dtype=float)


def _sample_new_edges_for_group(
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    *,
    k: int,
    same_block: bool,
    directed: bool,
    source_probs: np.ndarray,
    target_probs: np.ndarray,
    seen_group: set[tuple[int, int]],
    scorer: EdgeScorer,
    rng: np.random.Generator,
    exact_threshold: int,
    attempts_per_edge: int,
) -> list[tuple[int, int]]:
    if k <= 0 or len(source_ids) == 0 or len(target_ids) == 0:
        return []
    capacity = _feasible_pair_capacity(source_ids, target_ids, same_block=same_block, directed=directed)
    remaining_capacity = max(capacity - len(seen_group), 0)
    if remaining_capacity <= 0:
        return []
    target_k = min(int(k), int(remaining_capacity))
    if remaining_capacity <= int(exact_threshold):
        candidates, weights = _enumerate_new_dyads(
            source_ids,
            target_ids,
            same_block=same_block,
            directed=directed,
            seen_group=seen_group,
            scorer=scorer,
        )
        if not candidates:
            return []
        picked = _gumbel_top_k_indices(weights, min(target_k, len(candidates)), rng)
        return [candidates[int(index)] for index in picked]

    chosen: list[tuple[int, int]] = []
    chosen_set: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = max(200, int(attempts_per_edge) * int(target_k))
    while len(chosen) < target_k and attempts < max_attempts:
        batch_size = min(max(32, 4 * (target_k - len(chosen))), 512)
        source_draws = rng.choice(len(source_ids), size=batch_size, replace=True, p=source_probs)
        target_draws = rng.choice(len(target_ids), size=batch_size, replace=True, p=target_probs)
        for source_index, target_index in zip(source_draws.tolist(), target_draws.tolist()):
            attempts += 1
            source = int(source_ids[int(source_index)])
            target = int(target_ids[int(target_index)])
            if same_block and source == target:
                if attempts >= max_attempts:
                    break
                continue
            edge = _canonical_edge(source, target, directed=directed)
            if edge in seen_group or edge in chosen_set:
                if attempts >= max_attempts:
                    break
                continue
            if rng.random() <= scorer.pair_modifier(edge[0], edge[1]):
                chosen.append(edge)
                chosen_set.add(edge)
                if len(chosen) >= target_k:
                    break
            if attempts >= max_attempts:
                break
    return chosen


def _weight_mean_factor(nodes: dict[int, NodeRecord], u: int, v: int, *, animal_reference: float) -> float:
    left = nodes[int(u)]
    right = nodes[int(v)]
    scale = 0.5 * (math.log1p(max(left.total_animals, 0.0)) + math.log1p(max(right.total_animals, 0.0)))
    if animal_reference <= 1e-9:
        return 1.0
    return float(max(0.45, min(1.65, scale / animal_reference)))


def _sample_edge_weight(
    *,
    edge: tuple[int, int],
    ts_value: int,
    params: BlockPairParams,
    cfg: SimulationConfig,
    scorer: EdgeScorer,
    nodes: dict[int, NodeRecord],
    animal_reference: float,
    rng: np.random.Generator,
) -> int:
    pair_factor = 0.72 + 0.52 * scorer.pair_modifier(edge[0], edge[1])
    size_factor = _weight_mean_factor(nodes, edge[0], edge[1], animal_reference=animal_reference)
    mean_value = float(max(1.0, params.weight_mean * _season_multiplier(ts_value, cfg) * pair_factor * size_factor))
    return _sample_shifted_nb(rng, mean_value, params.weight_alpha)


def _simulate_panel(
    cfg: SimulationConfig,
    rng: np.random.Generator,
    *,
    nodes: list[NodeRecord],
    timeline: list[int],
    block_activity_states: dict[int, dict[int, bool]],
    block_pair_params: dict[tuple[int, int], BlockPairParams],
    activity_params: dict[int, BlockActivityParams],
) -> dict[str, Any]:
    node_by_id = {int(node.node_id): node for node in nodes}
    block_types = _block_types(nodes)
    block_ids = sorted(block_types)
    node_ids_by_block, out_probs_by_block, in_probs_by_block = _build_block_node_lookup(nodes)
    scorer = EdgeScorer(nodes=node_by_id, cfg=cfg)
    animal_reference = float(np.mean([math.log1p(max(node.total_animals, 0.0)) for node in nodes if node.node_type == TYPE_FARM]))

    seen_by_group: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    prev_by_group: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    all_events: list[EdgeEvent] = []
    block_activity_rows: list[dict[str, Any]] = []
    block_pair_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    edge_index = 1

    for ts_value in timeline:
        current_states = block_activity_states[int(ts_value)]
        current_by_group: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
        category_by_group: dict[tuple[int, int], dict[str, list[tuple[int, int]]]] = defaultdict(lambda: {"persist": [], "reactivated": [], "new": []})
        weight_total = 0.0
        active_blocks = {int(block_id) for block_id, active in current_states.items() if active}
        active_nodes_eligible = set()
        active_farms_eligible = set()
        active_regions_eligible = set()
        for block_id in active_blocks:
            ids = node_ids_by_block.get(int(block_id), np.asarray([], dtype=int))
            active_nodes_eligible.update(int(value) for value in ids.tolist())
            if block_types[int(block_id)] == TYPE_FARM:
                active_farms_eligible.update(int(value) for value in ids.tolist())
            else:
                active_regions_eligible.update(int(value) for value in ids.tolist())
            if cfg.write_block_activity_truth:
                block_activity_rows.append(
                    {
                        "ts": int(ts_value),
                        "block_id": int(block_id),
                        "block_type": str(block_types[int(block_id)]),
                        "active": 1,
                        "eligible_node_count": int(len(ids)),
                        "eligible_farm_count": int(sum(node_by_id[int(value)].node_type == TYPE_FARM for value in ids.tolist())),
                        "eligible_region_count": int(sum(node_by_id[int(value)].node_type == TYPE_REGION for value in ids.tolist())),
                    }
                )
        if cfg.write_block_activity_truth:
            inactive_blocks = sorted(set(block_ids) - active_blocks)
            for block_id in inactive_blocks:
                block_activity_rows.append(
                    {
                        "ts": int(ts_value),
                        "block_id": int(block_id),
                        "block_type": str(block_types[int(block_id)]),
                        "active": 0,
                        "eligible_node_count": int(len(node_ids_by_block.get(int(block_id), np.asarray([], dtype=int)))),
                        "eligible_farm_count": int(block_types[int(block_id)] == TYPE_FARM) * int(len(node_ids_by_block.get(int(block_id), np.asarray([], dtype=int)))),
                        "eligible_region_count": int(block_types[int(block_id)] == TYPE_REGION) * int(len(node_ids_by_block.get(int(block_id), np.asarray([], dtype=int)))),
                    }
                )

        for block_u in active_blocks:
            source_ids = node_ids_by_block.get(int(block_u), np.asarray([], dtype=int))
            if len(source_ids) == 0:
                continue
            for block_v in active_blocks:
                key = _canonical_block_pair(int(block_u), int(block_v), directed=cfg.directed)
                if key not in block_pair_params:
                    continue
                target_ids = node_ids_by_block.get(int(block_v), np.asarray([], dtype=int))
                if len(target_ids) == 0:
                    continue
                same_block = int(block_u) == int(block_v)
                params = block_pair_params[key]
                season = _season_multiplier(int(ts_value), cfg)
                persist_pool = sorted(prev_by_group[key])
                reactivate_pool = sorted(seen_by_group[key] - prev_by_group[key])
                mu_new = float(max(0.0, params.new_scale * season * math.sqrt(max(len(source_ids), 1) * max(len(target_ids), 1))))
                unseen_capacity = max(
                    _feasible_pair_capacity(source_ids, target_ids, same_block=same_block, directed=cfg.directed) - len(seen_by_group[key]),
                    0,
                )
                new_target = min(int(rng.poisson(mu_new)), int(unseen_capacity))
                persist_target = int(rng.binomial(len(persist_pool), min(max(params.persist_prob * (0.92 + 0.08 * season), 0.0), 0.999)))
                reactivate_target = int(rng.binomial(len(reactivate_pool), min(max(params.reactivate_prob * (0.94 + 0.06 * season), 0.0), 0.98)))

                persist_edges = _sample_existing_edges(persist_pool, k=persist_target, scorer=scorer, rng=rng)
                reactivate_edges = _sample_existing_edges(reactivate_pool, k=reactivate_target, scorer=scorer, rng=rng)
                new_edges = _sample_new_edges_for_group(
                    source_ids,
                    target_ids,
                    k=new_target,
                    same_block=same_block,
                    directed=cfg.directed,
                    source_probs=out_probs_by_block[int(block_u)],
                    target_probs=in_probs_by_block[int(block_v)],
                    seen_group=seen_by_group[key],
                    scorer=scorer,
                    rng=rng,
                    exact_threshold=int(cfg.exact_new_dyad_threshold),
                    attempts_per_edge=int(cfg.proposal_attempts_per_edge),
                )

                chosen = [("persist", edge) for edge in persist_edges] + [("reactivated", edge) for edge in reactivate_edges] + [("new", edge) for edge in new_edges]
                for category, edge in chosen:
                    if edge in current_by_group[key]:
                        continue
                    current_by_group[key].add(edge)
                    category_by_group[key][category].append(edge)
                    weight_value = _sample_edge_weight(
                        edge=edge,
                        ts_value=int(ts_value),
                        params=params,
                        cfg=cfg,
                        scorer=scorer,
                        nodes=node_by_id,
                        animal_reference=animal_reference,
                        rng=rng,
                    )
                    weight_total += float(weight_value)
                    source_type = node_by_id[edge[0]].node_type
                    target_type = node_by_id[edge[1]].node_type
                    all_events.append(
                        EdgeEvent(
                            u=int(edge[0]),
                            i=int(edge[1]),
                            ts=int(ts_value),
                            idx=int(edge_index),
                            weight=int(weight_value),
                            category=str(category),
                            block_u=int(node_by_id[edge[0]].true_block),
                            block_v=int(node_by_id[edge[1]].true_block),
                            source_type=str(source_type),
                            target_type=str(target_type),
                        )
                    )
                    edge_index += 1

                if cfg.write_block_pair_truth:
                    block_pair_rows.append(
                        {
                            "ts": int(ts_value),
                            "block_u": int(key[0]),
                            "block_v": int(key[1]),
                            "source_type": str(params.source_type),
                            "target_type": str(params.target_type),
                            "eligible_source_count": int(len(source_ids)),
                            "eligible_target_count": int(len(target_ids)),
                            "persist_target": int(persist_target),
                            "persist_count": int(len(persist_edges)),
                            "reactivate_target": int(reactivate_target),
                            "reactivate_count": int(len(reactivate_edges)),
                            "new_target": int(new_target),
                            "new_count": int(len(new_edges)),
                            "edge_count": int(len(current_by_group[key])),
                            "new_scale": float(params.new_scale),
                            "persist_prob": float(params.persist_prob),
                            "reactivate_prob": float(params.reactivate_prob),
                        }
                    )

        realized_nodes = set()
        realized_farms = set()
        realized_regions = set()
        persist_total = 0
        reactivate_total = 0
        new_total = 0
        for key, edge_set in current_by_group.items():
            for edge in edge_set:
                realized_nodes.add(int(edge[0]))
                realized_nodes.add(int(edge[1]))
                if node_by_id[int(edge[0])].node_type == TYPE_FARM:
                    realized_farms.add(int(edge[0]))
                else:
                    realized_regions.add(int(edge[0]))
                if node_by_id[int(edge[1])].node_type == TYPE_FARM:
                    realized_farms.add(int(edge[1]))
                else:
                    realized_regions.add(int(edge[1]))
            persist_total += int(len(category_by_group[key]["persist"]))
            reactivate_total += int(len(category_by_group[key]["reactivated"]))
            new_total += int(len(category_by_group[key]["new"]))
            seen_by_group[key].update(edge_set)
            prev_by_group[key] = set(edge_set)
        inactive_keys = sorted(set(prev_by_group).difference(current_by_group))
        for key in inactive_keys:
            prev_by_group[key] = set()

        snapshot_rows.append(
            {
                "ts": int(ts_value),
                "date": date.fromordinal(int(ts_value)).isoformat(),
                "edge_count": int(persist_total + reactivate_total + new_total),
                "persist_count": int(persist_total),
                "reactivated_count": int(reactivate_total),
                "new_count": int(new_total),
                "weight_total": float(weight_total),
                "active_block_count": int(len(active_blocks)),
                "active_nodes_eligible": int(len(active_nodes_eligible)),
                "active_farms_eligible": int(len(active_farms_eligible)),
                "active_regions_eligible": int(len(active_regions_eligible)),
                "realized_active_nodes": int(len(realized_nodes)),
                "realized_active_farms": int(len(realized_farms)),
                "realized_active_regions": int(len(realized_regions)),
            }
        )

    edges_frame = pd.DataFrame([asdict(event) for event in all_events])
    if edges_frame.empty:
        edges_frame = pd.DataFrame(columns=["u", "i", "ts", "idx", "weight", "category", "block_u", "block_v", "source_type", "target_type"])
    snapshot_frame = pd.DataFrame(snapshot_rows)
    block_activity_frame = pd.DataFrame(block_activity_rows)
    block_pair_frame = pd.DataFrame(block_pair_rows)
    return {
        "edges": edges_frame,
        "snapshot_truth": snapshot_frame,
        "block_activity_truth": block_activity_frame,
        "block_pair_snapshot_truth": block_pair_frame,
        "seen_edge_count": int(len({(int(row.u), int(row.i), int(row.ts)) for row in edges_frame.itertuples(index=False)})) if len(edges_frame) else 0,
    }


def _write_dataset(
    cfg: SimulationConfig,
    *,
    dataset_dir: Path,
    node_map: pd.DataFrame,
    node_features: np.ndarray,
    node_schema: dict[str, Any],
    edges: pd.DataFrame,
    snapshot_truth: pd.DataFrame,
    block_activity_truth: pd.DataFrame,
    block_pair_snapshot_truth: pd.DataFrame,
    block_params: pd.DataFrame,
    block_pair_params: pd.DataFrame,
    nodes: list[NodeRecord],
) -> dict[str, str]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    truth_dir = dataset_dir / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    node_map_path = dataset_dir / "node_map.csv"
    node_features_path = dataset_dir / "node_features.npy"
    node_schema_path = dataset_dir / "node_schema.json"
    edges_path = dataset_dir / "edges.csv"

    node_map.sort_values("node_id").to_csv(node_map_path, index=False)
    np.save(node_features_path, node_features.astype(float))
    _save_json(node_schema_path, node_schema)
    paths["node_map_csv"] = str(node_map_path)
    paths["node_features_npy"] = str(node_features_path)
    paths["node_schema_json"] = str(node_schema_path)

    if cfg.write_weight_npy:
        if len(edges):
            weights = np.zeros(int(edges["idx"].max()) + 1, dtype=float)
            weights[edges["idx"].to_numpy(dtype=int)] = edges["weight"].to_numpy(dtype=float)
        else:
            weights = np.zeros(1, dtype=float)
        weight_path = dataset_dir / f"ml_{cfg.dataset}_{cfg.weight_name}.npy"
        np.save(weight_path, weights)
        edges[["u", "i", "ts", "idx"]].to_csv(edges_path, index=False)
        paths["weight_npy"] = str(weight_path)
        paths["weight_input_mode"] = "external_npy"
    else:
        edges[["u", "i", "ts", "weight"]].rename(columns={"weight": cfg.weight_name}).to_csv(edges_path, index=False)
        paths["weight_input_mode"] = "csv_column"
    paths["edges_csv"] = str(edges_path)

    node_truth = pd.DataFrame(
        [
            {
                "node_id": int(node.node_id),
                "type": str(node.node_type),
                "true_block": int(node.true_block),
                "true_role_block": int(node.role_block),
                "region_id": int(node.region_id),
                "region_code": str(node.region_code),
                "x_m": float(node.x_m),
                "y_m": float(node.y_m),
                "num_farms": float(node.num_farms),
                "total_animals": float(node.total_animals),
                "out_mass": float(node.out_mass),
                "in_mass": float(node.in_mass),
                "specialty": str(node.specialty),
            }
            for node in nodes
        ]
    )
    node_truth_path = truth_dir / "node_truth.csv"
    node_truth.to_csv(node_truth_path, index=False)
    paths["node_truth_csv"] = str(node_truth_path)

    snapshot_truth_path = truth_dir / "snapshot_truth.csv"
    snapshot_truth.to_csv(snapshot_truth_path, index=False)
    paths["snapshot_truth_csv"] = str(snapshot_truth_path)

    block_params_path = truth_dir / "block_params.csv"
    block_params.to_csv(block_params_path, index=False)
    paths["block_params_csv"] = str(block_params_path)

    block_pair_params_path = truth_dir / "block_pair_params.csv"
    block_pair_params.to_csv(block_pair_params_path, index=False)
    paths["block_pair_params_csv"] = str(block_pair_params_path)

    if cfg.write_block_activity_truth and not block_activity_truth.empty:
        block_activity_path = truth_dir / "block_activity_truth.csv"
        block_activity_truth.to_csv(block_activity_path, index=False)
        paths["block_activity_truth_csv"] = str(block_activity_path)
    if cfg.write_block_pair_truth and not block_pair_snapshot_truth.empty:
        block_pair_snapshot_path = truth_dir / "block_pair_snapshot_truth.csv.gz"
        block_pair_snapshot_truth.to_csv(block_pair_snapshot_path, index=False, compression="gzip")
        paths["block_pair_snapshot_truth_csv"] = str(block_pair_snapshot_path)
    if cfg.write_edge_truth and len(edges):
        edge_truth_path = truth_dir / "edge_truth.csv.gz"
        edges.sort_values(["ts", "idx"]).to_csv(edge_truth_path, index=False, compression="gzip")
        paths["edge_truth_csv"] = str(edge_truth_path)

    return paths


def _metadata_field_summary(node_map: pd.DataFrame) -> dict[str, list[str]]:
    categories: dict[str, list[str]] = {}
    for column in node_map.columns:
        if column == "node_id":
            continue
        values = node_map[column].dropna().astype(str).unique().tolist()
        if len(values) <= 12:
            categories[column] = sorted(values)
    return categories


def create_dataset(cfg: SimulationConfig) -> dict[str, Any]:
    rng = np.random.default_rng(int(cfg.seed))
    dataset_dir = Path(cfg.output_root).expanduser().resolve() / cfg.dataset
    timeline = _ordinal_range(cfg.start_date, cfg.days)

    region_nodes, region_centroids, region_block_map, region_size_weights = _build_region_nodes(cfg, rng)
    farm_nodes = _build_farm_nodes(
        cfg,
        rng,
        region_centroids=region_centroids,
        region_block_map=region_block_map,
        region_size_weights=region_size_weights,
    )
    region_nodes = _finalise_region_nodes(region_nodes, farm_nodes)
    nodes = sorted(region_nodes + farm_nodes, key=lambda item: item.node_id)
    node_map, node_features, node_schema = _node_frames(nodes)

    block_types = _block_types(nodes)
    activity_params = _build_activity_params(cfg, rng, block_types)
    block_activity_states = _sample_block_activity_states(cfg, rng, timeline=timeline, activity_params=activity_params)
    block_pair_params = _build_block_pair_params(cfg, rng, block_types)

    panel = _simulate_panel(
        cfg,
        rng,
        nodes=nodes,
        timeline=timeline,
        block_activity_states=block_activity_states,
        block_pair_params=block_pair_params,
        activity_params=activity_params,
    )

    block_params_frame = pd.DataFrame(
        [
            {
                "block_id": int(block_id),
                "block_type": str(block_types[int(block_id)]),
                "p_init": float(params.p_init),
                "p01": float(params.p01),
                "p11": float(params.p11),
                "node_count": int(sum(node.true_block == int(block_id) for node in nodes)),
            }
            for block_id, params in sorted(activity_params.items())
        ]
    )
    block_pair_params_frame = pd.DataFrame(
        [
            {
                "block_u": int(key[0]),
                "block_v": int(key[1]),
                "source_type": str(params.source_type),
                "target_type": str(params.target_type),
                "new_scale": float(params.new_scale),
                "persist_prob": float(params.persist_prob),
                "reactivate_prob": float(params.reactivate_prob),
                "weight_mean": float(params.weight_mean),
                "weight_alpha": float(params.weight_alpha),
            }
            for key, params in sorted(block_pair_params.items())
        ]
    )

    paths = _write_dataset(
        cfg,
        dataset_dir=dataset_dir,
        node_map=node_map,
        node_features=node_features,
        node_schema=node_schema,
        edges=panel["edges"],
        snapshot_truth=panel["snapshot_truth"],
        block_activity_truth=panel["block_activity_truth"],
        block_pair_snapshot_truth=panel["block_pair_snapshot_truth"],
        block_params=block_params_frame,
        block_pair_params=block_pair_params_frame,
        nodes=nodes,
    )

    manifest = {
        "creator": {
            "name": "creator.py",
            "version": 1,
            "seed": int(cfg.seed),
        },
        "dataset": cfg.dataset,
        "output_root": str(Path(cfg.output_root).expanduser().resolve()),
        "dataset_dir": str(dataset_dir),
        "directed": bool(cfg.directed),
        "config": asdict(cfg),
        "files": paths,
        "pipeline_inputs": {
            "data_root": str(Path(cfg.output_root).expanduser().resolve()),
            "dataset": cfg.dataset,
            "edges_csv": paths["edges_csv"],
            "node_features_npy": paths["node_features_npy"],
            "node_schema_json": paths["node_schema_json"],
            "node_map_csv": paths["node_map_csv"],
            "weight_npy": paths.get("weight_npy"),
            "weight_col": cfg.weight_name,
            "ts_col": "ts",
            "src_col": "u",
            "dst_col": "i",
            "ts_format": "ordinal",
            "ts_unit": "D",
            "holiday_country": "NL",
        },
        "summary": {
            "node_count": int(len(nodes)),
            "farm_count": int(sum(node.node_type == TYPE_FARM for node in nodes)),
            "region_count": int(sum(node.node_type == TYPE_REGION for node in nodes)),
            "edge_count": int(len(panel["edges"])),
            "snapshot_count": int(len(panel["snapshot_truth"])),
            "observed_days_with_edges": int((panel["snapshot_truth"]["edge_count"] > 0).sum()) if len(panel["snapshot_truth"]) else 0,
            "weight_total": float(panel["edges"]["weight"].sum()) if len(panel["edges"]) else 0.0,
            "mean_edges_per_day": float(panel["snapshot_truth"]["edge_count"].mean()) if len(panel["snapshot_truth"]) else 0.0,
            "mean_realized_active_farms": float(panel["snapshot_truth"]["realized_active_farms"].mean()) if len(panel["snapshot_truth"]) else 0.0,
            "mean_realized_active_regions": float(panel["snapshot_truth"]["realized_active_regions"].mean()) if len(panel["snapshot_truth"]) else 0.0,
        },
        "metadata_value_samples": _metadata_field_summary(node_map),
    }
    manifest_path = dataset_dir / "manifest.json"
    _save_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a simulated trade dataset that matches the temporal-generator assumptions.")
    parser.add_argument("--output-root", default="./simulated_data", help="Root directory for generated datasets.")
    parser.add_argument("--dataset", default="ideal_trade_sim", help="Dataset name under the output root.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--profile", choices=sorted(PROFILE_DEFAULTS), default="medium", help="Preset scale.")
    parser.add_argument("--n-farms", type=int, default=None, help="Override the farm count.")
    parser.add_argument("--n-regions", type=int, default=None, help="Override the region-node count.")
    parser.add_argument("--farm-blocks", type=int, default=None, help="Override the farm block count.")
    parser.add_argument("--region-blocks", type=int, default=None, help="Override the region block count.")
    parser.add_argument("--days", type=int, default=None, help="Override the number of days.")
    parser.add_argument("--start-date", default="2018-01-01", help="First calendar date in ISO format.")
    parser.add_argument("--undirected", action="store_true", help="Generate an undirected panel.")
    parser.add_argument("--weight-name", default="headcount", help="Weight column name for pipeline use.")
    parser.add_argument("--skip-weight-npy", action="store_true", help="Write weights into the edge CSV instead of a sidecar NPY file.")
    parser.add_argument("--skip-edge-truth", action="store_true", help="Skip the per-edge truth export.")
    parser.add_argument("--skip-block-activity-truth", action="store_true", help="Skip the per-block activity truth export.")
    parser.add_argument("--skip-block-pair-truth", action="store_true", help="Skip the per-block-pair truth export.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> SimulationConfig:
    defaults = PROFILE_DEFAULTS[str(args.profile)]
    return SimulationConfig(
        output_root=str(args.output_root),
        dataset=str(args.dataset),
        seed=int(args.seed),
        profile=str(args.profile),
        n_farms=int(args.n_farms if args.n_farms is not None else defaults["n_farms"]),
        n_regions=int(args.n_regions if args.n_regions is not None else defaults["n_regions"]),
        farm_blocks=int(args.farm_blocks if args.farm_blocks is not None else defaults["farm_blocks"]),
        region_blocks=int(args.region_blocks if args.region_blocks is not None else defaults["region_blocks"]),
        days=int(args.days if args.days is not None else defaults["days"]),
        start_date=str(args.start_date),
        directed=not bool(args.undirected),
        weight_name=str(args.weight_name),
        write_weight_npy=not bool(args.skip_weight_npy),
        write_edge_truth=not bool(args.skip_edge_truth),
        write_block_activity_truth=not bool(args.skip_block_activity_truth),
        write_block_pair_truth=not bool(args.skip_block_pair_truth),
    )


def main() -> None:
    args = _parse_args()
    _configure_logging(bool(args.verbose))
    cfg = _config_from_args(args)
    LOGGER.info(
        "Creating simulated dataset | dataset=%s | profile=%s | farms=%s | regions=%s | days=%s | directed=%s",
        cfg.dataset,
        cfg.profile,
        cfg.n_farms,
        cfg.n_regions,
        cfg.days,
        cfg.directed,
    )
    manifest = create_dataset(cfg)
    LOGGER.info(
        "Finished dataset creation | dataset_dir=%s | edges=%s | nodes=%s | manifest=%s",
        manifest["dataset_dir"],
        manifest["summary"]["edge_count"],
        manifest["summary"]["node_count"],
        manifest["manifest_path"],
    )


if __name__ == "__main__":
    main()

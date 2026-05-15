"""
Shared infrastructure for XGBoost-merger experiments.

Everything here assumes the attribute cache (`output/cache/cache_attributes_<seq>.pkl`)
already exists, so the expensive parts of the pipeline (detection, tracking,
ReID, jersey, team) never re-run. Each experiment only touches:

    cached attributes -> splitter -> merger -> MOT file -> HOTA eval

For a single experiment script, the splitter is applied once per sequence and
its output is cached in memory; each config deep-copies that output before
feeding it to a merger (mergers mutate tracklets in place).

Split policy
------------
All tuning/sweep experiments MUST run on the *validation* split. Only the
single chosen winner of each experiment is re-evaluated on *test*. This is
enforced by defaulting `EvalSplit` to "valid"; passing "test" requires an
explicit call and is logged as such.

The runner temporarily overrides `settings.EVAL_SPLIT` / `settings.DATA_ROOT`
for the duration of a run so you can sweep without editing settings.py.
"""

from __future__ import annotations

import copy
import csv
import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from tqdm import tqdm

import settings
from tracklets.split_tracklets import split_tracklets
from tracklets.xgboost_merger import XGBoostMerger
from utils.build import build_configs
from utils.evaluation_utils import save_mot_file_for_sn_trackeval
from utils.run_evaluation import run_evaluation


EXPERIMENTS_ROOT = settings.PROJECT_ROOT / "experiments"
RESULTS_CSV      = EXPERIMENTS_ROOT / "results.csv"
CACHE_ROOT       = settings.OUTPUT_ROOT / "cache"
SEQMAP_ROOT      = settings.PROJECT_ROOT / "evaluation" / "seqmaps"

DEFAULT_MODEL_DIR = settings.WEIGHTS_ROOT / "xgboost_0_neg_ratio"
TUNED_MODEL_DIR   = settings.WEIGHTS_ROOT / "xgboost_tuned"


# ---------------------------------------------------------------------------
# Split handling
# ---------------------------------------------------------------------------

def set_eval_split(split: str) -> None:
    """
    Temporarily retarget settings so downstream code (save_mot_file_for_sn_trackeval,
    run_evaluation) points at the correct split.
    """
    assert split in ("train", "valid", "test"), f"Unknown split: {split}"
    settings.EVAL_SPLIT = split
    settings.DATA_ROOT  = Path(
        rf"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\data\soccernet\soccernet-player-tracking\{split}"
    )


def load_sequences(split: str) -> List[str]:
    """Read the seqmap file and return the list of sequence names for the split."""
    seqmap = SEQMAP_ROOT / f"SNPT-{split}.txt"
    with open(seqmap) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    # seqmap format: either bare sequence names or a header "name\n<seq>\n..."
    return [ln for ln in lines if ln.startswith("SNPT-")]


def load_attribute_tracklets(sequence: str) -> Optional[dict]:
    path = CACHE_ROOT / f"cache_attributes_{sequence}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# HOTA parsing
# ---------------------------------------------------------------------------

def parse_hota_summary(method_name: str) -> Dict[str, float]:
    """
    Parse evaluation/SNPT/<method>/pedestrian_summary.txt into a dict of
    common metrics.
    """
    path = settings.PROJECT_ROOT / "evaluation" / "SNPT" / method_name / "pedestrian_summary.txt"
    if not path.exists():
        return {}
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 2:
        return {}
    keys   = lines[0].split()
    vals   = lines[1].split()
    out    = {}
    for k, v in zip(keys, vals):
        try:
            out[k] = float(v)
        except ValueError:
            pass
    # Convenience aliases
    picks = ["HOTA", "DetA", "AssA", "IDF1", "MOTA"]
    return {k: out[k] for k in picks if k in out}


# ---------------------------------------------------------------------------
# Results CSV
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "timestamp", "exp_id", "run_name", "split",
    "HOTA", "DetA", "AssA", "IDF1", "MOTA",
    "config",
]


def append_result(row: Dict) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


# ---------------------------------------------------------------------------
# Merger factories
# ---------------------------------------------------------------------------

MergerFactory = Callable[[], "XGBoostMerger"]


def default_xgboost_factory(
    model_dir: Path = DEFAULT_MODEL_DIR,
    merge_threshold: float = 0.5,
    linkage_method: str = "average",
    jersey_entropy_threshold: float = 0.15,
    team_consistency_threshold: float = 0.9,
    team_confidence_threshold: float = 0.6,
    disable_temporal_constraint: bool = False,
    disable_jersey_constraint: bool = False,
    disable_team_constraint: bool = False,
) -> MergerFactory:
    """
    Build a zero-arg factory producing a fresh XGBoostMerger. The factory is
    called once per sequence (XGBoost is stateful with respect to predict
    calls, but re-instantiating costs ~ms and keeps experiments isolated).
    """
    model_path = model_dir / "xgboost_merger.json"
    meta_path  = model_dir / "xgboost_merger_meta.json"

    def _make():
        return XGBoostMerger(
            model_path=model_path,
            meta_path=meta_path,
            merge_threshold=merge_threshold,
            linkage_method=linkage_method,
            jersey_entropy_threshold=jersey_entropy_threshold,
            team_consistency_threshold=team_consistency_threshold,
            team_confidence_threshold=team_confidence_threshold,
            disable_temporal_constraint=disable_temporal_constraint,
            disable_jersey_constraint=disable_jersey_constraint,
            disable_team_constraint=disable_team_constraint,
        )
    return _make


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    run_name: str
    metrics: Dict[str, float]
    config: Dict
    exp_id: str
    split: str


@dataclass
class ExperimentRunner:
    """
    Apply the splitter once per sequence, cache the result, then run any
    number of merger configs against that cached splitter output.

    Typical usage:
        runner = ExperimentRunner(exp_id="EXP1", split="valid")
        runner.prepare()
        runner.run("thresh_0.5", default_xgboost_factory(merge_threshold=0.5),
                   config={"merge_threshold": 0.5})
    """
    exp_id: str
    split: str = "valid"
    sequences: Optional[List[str]] = None

    _splitted_cache: Dict[str, Dict] = field(default_factory=dict, init=False)
    _prepared: bool = field(default=False, init=False)

    # -- setup --

    def prepare(self) -> None:
        """Load cached attributes + run splitter once per sequence."""
        set_eval_split(self.split)
        _, _, _, splitter_cfg, _ = build_configs()

        if self.sequences is None:
            self.sequences = load_sequences(self.split)

        missing = []
        for seq in tqdm(self.sequences, desc=f"[{self.exp_id}] split+cache"):
            tracklets = load_attribute_tracklets(seq)
            if tracklets is None:
                missing.append(seq)
                continue
            self._splitted_cache[seq] = split_tracklets(tracklets, splitter_cfg)
        if missing:
            print(f"[{self.exp_id}] WARNING: missing attribute cache for: {missing}")
        self._prepared = True

    # -- per-run --

    def run(
        self,
        run_name: str,
        merger_factory: MergerFactory,
        config: Optional[Dict] = None,
        reuse_if_exists: bool = True,
    ) -> RunResult:
        """
        Run the pipeline for a single merger config and write the MOT files
        under method_name=`<exp_id>__<run_name>__<split>` so eval folders are
        unambiguous and don't clash with main.py runs.
        """
        if not self._prepared:
            self.prepare()

        method_name = self._method_name(run_name)
        eval_dir    = settings.PROJECT_ROOT / "evaluation" / "SNPT" / method_name
        summary     = eval_dir / "pedestrian_summary.txt"

        if reuse_if_exists and summary.exists():
            print(f"[{self.exp_id}] reusing existing result at {summary}")
            metrics = parse_hota_summary(method_name)
            return self._finalize(run_name, metrics, config or {}, method_name)

        t_start = time.time()
        for seq in tqdm(self.sequences, desc=f"[{self.exp_id}:{run_name}] merge"):
            if seq not in self._splitted_cache:
                continue
            splitted = copy.deepcopy(self._splitted_cache[seq])
            merger   = merger_factory()
            merged   = merger.merge(splitted)
            save_mot_file_for_sn_trackeval(
                tracklets_dict=merged,
                output_path=settings.PROJECT_ROOT / "evaluation",
                sequence_name=seq,
                method_name=method_name,
            )
        dt = time.time() - t_start
        print(f"[{self.exp_id}:{run_name}] merged all sequences in {dt:.1f}s; running HOTA eval...")

        run_evaluation(method_name, self.split)
        metrics = parse_hota_summary(method_name)
        return self._finalize(run_name, metrics, config or {}, method_name)

    # -- internals --

    def _method_name(self, run_name: str) -> str:
        clean = re.sub(r"[^A-Za-z0-9._-]+", "_", run_name)
        return f"{self.exp_id}__{clean}__{self.split}"

    def _finalize(self, run_name: str, metrics: Dict, config: Dict, method_name: str) -> RunResult:
        append_result({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "exp_id":    self.exp_id,
            "run_name":  run_name,
            "split":     self.split,
            "HOTA":      metrics.get("HOTA", ""),
            "DetA":      metrics.get("DetA", ""),
            "AssA":      metrics.get("AssA", ""),
            "IDF1":      metrics.get("IDF1", ""),
            "MOTA":      metrics.get("MOTA", ""),
            "config":    str(config),
        })
        print(f"[{self.exp_id}:{run_name}] {metrics}")
        return RunResult(
            run_name=run_name,
            metrics=metrics,
            config=config,
            exp_id=self.exp_id,
            split=self.split,
        )


# ---------------------------------------------------------------------------
# Summary helper (per-experiment pretty table)
# ---------------------------------------------------------------------------

def summarize_results(results: List[RunResult], sort_by: str = "HOTA") -> None:
    if not results:
        print("No results to summarize.")
        return
    results_sorted = sorted(
        results,
        key=lambda r: r.metrics.get(sort_by, -1),
        reverse=True,
    )
    keys = ["HOTA", "DetA", "AssA", "IDF1", "MOTA"]
    print(f"\n=== Results ({results[0].exp_id}, split={results[0].split}), sorted by {sort_by} ===")
    header = f"{'run_name':<40} " + " ".join(f"{k:>7}" for k in keys)
    print(header)
    print("-" * len(header))
    for r in results_sorted:
        row = f"{r.run_name:<40} " + " ".join(
            f"{r.metrics.get(k, float('nan')):>7.3f}" for k in keys
        )
        print(row)
    print()

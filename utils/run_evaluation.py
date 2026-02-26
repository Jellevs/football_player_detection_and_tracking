import subprocess
import sys
import settings

def run_evaluation(method_name: str, eval_split: str):
    eval_root = settings.PROJECT_ROOT / "evaluation"
    
    cmd = [
        sys.executable,  # <-- instead of "python"
        str(settings.PROJECT_ROOT / "sn-trackeval" / "scripts" / "run_mot_challenge.py"),
        "--BENCHMARK", "SNMOT",
        "--SPLIT_TO_EVAL", f"SNMOT-{eval_split}",
        "--GT_FOLDER", str(settings.DATA_ROOT),
        "--TRACKERS_FOLDER", str(eval_root / "SNPT"),
        "--TRACKERS_TO_EVAL", method_name,
        "--SEQMAP_FILE", str(eval_root / "seqmaps" / f"SNPT-{eval_split}.txt"),
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--DO_PREPROC", "False",
        "--USE_PARALLEL", "False",
        "--TRACKER_SUB_FOLDER", "data",
        "--SKIP_SPLIT_FOL", "True",
    ]
    
    subprocess.run(cmd, check=True)
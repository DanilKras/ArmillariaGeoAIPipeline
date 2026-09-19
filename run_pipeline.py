import subprocess
import sys
import time
from pathlib import Path

PIPELINE_STEPS = [
    "data_extraction.py",
    "data_processing.py",
    "negative_sampling.py",
    "feature_extraction.py",
    "model_training.py",
    "inference.py",
]


def run_step(script_name: str) -> None:
    script_path = Path("src") / script_name
    if not script_path.exists():
        print(f"Skipping {script_name}: file not found.")
        return

    print(f"\n--- Running: {script_name} ---")
    start = time.time()
    result = subprocess.run([sys.executable, "-m", f"src.{script_path.stem}"])

    if result.returncode != 0:
        print(f"Error in {script_name} (exit code {result.returncode}). Aborting.")
        sys.exit(result.returncode)

    print(f"Finished {script_name} in {time.time() - start:.1f}s")


def main() -> None:
    start_total = time.time()
    for step in PIPELINE_STEPS:
        run_step(step)

    elapsed_min = (time.time() - start_total) / 60
    print(f"\nPipeline completed in {elapsed_min:.1f} min.")


if __name__ == "__main__":
    main()

import subprocess
import sys
import time
from pathlib import Path

PIPELINE_MODULES = [
    "src.macro.data_extraction",
    "src.macro.data_processing",
    "src.macro.negative_sampling",
    "src.macro.feature_extraction",
    "src.macro.model_training",
    "src.macro.inference",
]


def run_step(module_name: str) -> None:
    rel_path = Path(*module_name.split(".")).with_suffix(".py")
    if not rel_path.exists():
        print(f"Skipping {module_name}: file {rel_path} not found.")
        return

    print(f"\n--- Running: {module_name} ---")
    start = time.time()
    result = subprocess.run([sys.executable, "-m", module_name])

    if result.returncode != 0:
        print(f"Error in {module_name} (exit code {result.returncode}). Aborting.")
        sys.exit(result.returncode)

    print(f"Finished {module_name} in {time.time() - start:.1f}s")


def main() -> None:
    start_total = time.time()
    for module in PIPELINE_MODULES:
        run_step(module)

    elapsed_min = (time.time() - start_total) / 60
    print(f"\nPipeline successfully completed in {elapsed_min:.1f} min.")


if __name__ == "__main__":
    main()

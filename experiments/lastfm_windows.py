"""Windows I/O adapter for the unchanged original baseline entry point.

Fresh run: python -B -u experiments/lastfm_windows.py
Same-run recovery: add --resume D:\CoLaKGExperiments\lastfm-baseline-...
Only atomic output replacement is adapted; training and identity checks remain
in lastfm_baseline.py. No source-hash check is bypassed on recovery.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ENTRY = HERE / "lastfm_baseline.py"
OUTPUT = Path(r"D:\CoLaKGExperiments")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_with_retry(temporary, destination, retry_log=None):
    """Retry only Windows access/sharing/locking errors, for <=5.55 seconds.

    Sleeps consume no Python/NumPy/Torch random numbers. Other failures are
    immediately fatal. Both paths are already owned by this isolated run.
    """
    for attempt in range(10):
        try:
            os.replace(str(temporary), str(destination))
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 9:
                raise
            if retry_log is not None:
                with retry_log.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(),
                                             "destination": destination.name, "attempt": attempt + 1,
                                             "winerror": error.winerror}) + "\n")
            time.sleep(min(0.05 * 2**attempt, 0.8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    adaptation = None
    if args.resume:
        run = args.resume.resolve()
        if run.parent != OUTPUT.resolve() or not run.name.startswith("lastfm-baseline-"):
            raise SystemExit("Recovery requires the original formal run under D:/CoLaKGExperiments")
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        expected = manifest["identity"]["source_sha256"]["experiments\\lastfm_baseline.py"]
        if digest(ENTRY) != expected:
            raise SystemExit("Original entry changed: no source identity bypass permitted")
        pointer = json.loads((run / "recovery.json").read_text(encoding="utf-8"))
        checkpoint = (run / "checkpoints" / pointer["file"]).resolve()
        if checkpoint.parent != (run / "checkpoints").resolve() or digest(checkpoint) != pointer["sha256"]:
            raise SystemExit("Same-run recovery checksum mismatch")
        identifier = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        adaptation = run / "io-adaptations" / identifier
        adaptation.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(str(checkpoint), str(adaptation / "same-run-recovery-input.pt"))
        if digest(adaptation / "same-run-recovery-input.pt") != pointer["sha256"]:
            raise SystemExit("Recovery input archive checksum mismatch")
        shutil.copyfile(str(run / "status.json"), str(adaptation / "prior-status.json"))
        shutil.copyfile(str(Path(__file__)), str(adaptation / "lastfm_windows.py"))
        metadata = {"reason": "User-approved Windows atomic status-file replacement repair",
                    "run_id": run.name, "resume_completed_epochs": pointer["completed_epochs"],
                    "recovery_input_sha256": pointer["sha256"], "entry_sha256": digest(ENTRY),
                    "adapter_sha256": digest(Path(__file__)), "argv": sys.argv,
                    "changes": ["bounded retry of os.replace for winerror 5/32/33 only"],
                    "unchanged": ["original runner/source identity verification", "model/data/config",
                                  "BPR/sampler/Adam", "model+optimizer+all RNG recovery"]}
        (adaptation / "adaptation.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    spec = importlib.util.spec_from_file_location("lastfm_baseline_runtime", ENTRY)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)

    def retry_log(path):
        # All writer calls are under this run; fresh runs have no adaptation archive yet.
        return (adaptation / "replace-retries.jsonl") if adaptation else path.parent / "replace-retries.jsonl"

    def atomic_json(path, value):
        temporary = path.with_name(path.name + ".tmp")
        baseline.write_json(temporary, value)
        replace_with_retry(temporary, path, retry_log(path))

    def atomic_torch_save(torch, value, path):
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        replace_with_retry(temporary, path, retry_log(path))

    baseline.atomic_json = atomic_json
    baseline.atomic_torch_save = atomic_torch_save
    return baseline.main()


if __name__ == "__main__":
    raise SystemExit(main())

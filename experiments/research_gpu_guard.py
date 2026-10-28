"""Recognize the verified desktop UI process; block every other GPU process."""
import csv
import subprocess


def classify_processes(output):
    ignored, blocking = [], []
    for row in csv.reader(output.splitlines()):
        if not row:
            continue
        if len(row) != 2 or not row[0].strip().isdigit():
            raise RuntimeError("Unrecognized nvidia-smi process output; refuse to guess")
        pid, name = int(row[0]), row[1].strip()
        normalized = name.replace("\\", "/").lower()
        # This desktop process was inspected on this machine at 2026-10-25 06:33.
        # It is the current Codex interface, not a Python/model-training process.
        codex_ui = (normalized.startswith("c:/program files/windowsapps/openai.codex_")
                    and normalized.endswith("/app/chatgpt.exe")
                    and normalized.count("/") == 5)
        (ignored if codex_ui else blocking).append({"pid": pid, "process": name})
    return {"ignored_verified_codex_ui": ignored, "blocking": blocking}


def check_gpu_processes():
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name",
                                      "--format=csv,noheader"], text=True)
    result = classify_processes(output)
    if result["blocking"]:
        raise RuntimeError("Other GPU processes present; no interruption attempted: " + str(result["blocking"]))
    return result

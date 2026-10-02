import argparse
import subprocess
import time
import sys
import os
import platform

import pipelines


def log_diagnostic(stage, status, elapsed=None):
    """Prints a structured diagnostic message for each pipeline stage."""
    timestamp = time.strftime("%H:%M:%S")
    time_str = f" | Time: {elapsed:.2f}s" if elapsed is not None else ""
    color = "\033[92m" if status == "SUCCESS" else "\033[91m"
    reset = "\033[0m"
    print(f"[{timestamp}] {stage:<20} | {color}{status:<8}{reset}{time_str}")

def check_system():
    """Diagnostic info: Checks the environment before starting."""
    print("--- System Diagnostic ---")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    try:
        import torch  # imported here: the data gate has no torch
    except ImportError:
        print("GPU: unknown (torch not installed in this environment)")
    else:
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("GPU: NOT FOUND (Check CUDA drivers)")
    print("-" * 25 + "\n")

def stage_env(model, base=None, fast=False):
    """The environment the stage scripts run in: --model, when given, overrides RESONANCE_LF_PROFILE; --fast makes it that
    model's fast profile (the model being --model, else the variable, else the default)."""
    env = dict(os.environ if base is None else base)
    if fast:
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "llamafactory"))
        from lf_tools import model_name
        env["RESONANCE_LF_PROFILE"] = model_name(model, env, fast=True)
    elif model:
        env["RESONANCE_LF_PROFILE"] = model
    return env

def run_step(name, script_path, args=(), env=None):
    """Executes a single python script and tracks its performance."""
    if not os.path.exists(script_path):
        log_diagnostic(name, "MISSING")
        return False

    start_time = time.perf_counter()
    try:
        # Run the script as a sub-process
        subprocess.run([sys.executable, script_path, *args], check=True, capture_output=False, env=env)
        elapsed = time.perf_counter() - start_time
        log_diagnostic(name, "SUCCESS", elapsed)
        return True
    except subprocess.CalledProcessError:
        elapsed = time.perf_counter() - start_time
        log_diagnostic(name, "FAILED", elapsed)
        return False

def main():
    parser = argparse.ArgumentParser(description="Run a training pipeline, stage by stage.")
    parser.add_argument("--pipeline", choices=sorted(pipelines.PIPELINES), default=pipelines.DEFAULT,
                        help="training backend (default: %(default)s)")
    parser.add_argument("--model", help="llamafactory model profile (default: $RESONANCE_LF_PROFILE, else "
                        "the default profile); the parameter wins over the environment")
    parser.add_argument("--fast", action="store_true", help="llamafactory: run on the model's fast profile (<model>-fast)")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--from", dest="from_stage", metavar="STAGE",
                       help="start at this stage (name or unique prefix, e.g. 'merge') and run the rest")
    which.add_argument("--only", metavar="STAGE", help="run just this stage")
    args = parser.parse_args()
    if args.fast and args.pipeline != "llamafactory":
        parser.error("--fast needs the llamafactory pipeline (the others have no model profiles)")

    check_system()
    print(f"Pipeline: {args.pipeline}\n")
    env = stage_env(args.model, fast=args.fast)
    total_start = time.perf_counter()

    try:
        pipeline = pipelines.select_stages(pipelines.stages(args.pipeline), args.from_stage, args.only)
    except ValueError as e:
        parser.error(str(e))

    for stage in pipeline:
        name = stage.name
        success = run_step(name, stage.path, stage.args, env)
        if not success:
            print(f"\n[!] Pipeline halted at {name}. Check logs.")
            sys.exit(1)

    total_elapsed = time.perf_counter() - total_start
    print("\n" + "="*40)
    print(f"PIPELINE COMPLETE | Total Time: {total_elapsed/60:.2f} minutes")
    print("="*40)

if __name__ == "__main__":
    main()
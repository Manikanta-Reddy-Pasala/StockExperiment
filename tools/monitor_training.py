#!/usr/bin/env python3
"""Monitor ML training progress in real-time"""

import sys
import time
import subprocess
from datetime import datetime

def get_training_process():
    """Find the training process."""
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )
        for line in result.stdout.split('\n'):
            if 'train_ml_models' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) >= 11:
                    pid = parts[1]
                    cpu = parts[2]
                    mem = parts[3]
                    # Get elapsed time
                    etime_result = subprocess.run(
                        ['ps', '-p', pid, '-o', 'etime='],
                        capture_output=True,
                        text=True
                    )
                    elapsed = etime_result.stdout.strip()
                    return {
                        'pid': pid,
                        'cpu': cpu,
                        'mem': mem,
                        'elapsed': elapsed
                    }
    except:
        pass
    return None

def format_elapsed(elapsed_str):
    """Convert elapsed time to readable format."""
    parts = elapsed_str.split(':')
    if len(parts) == 2:  # MM:SS
        return f"{parts[0]}m {parts[1]}s"
    elif len(parts) == 3:  # HH:MM:SS
        return f"{parts[0]}h {parts[1]}m {parts[2]}s"
    return elapsed_str

print("=" * 80)
print("ML TRAINING PROGRESS MONITOR")
print("=" * 80)
print("\nPress Ctrl+C to stop monitoring (training will continue)\n")

try:
    iteration = 0
    while True:
        iteration += 1

        proc = get_training_process()

        if proc:
            elapsed = format_elapsed(proc['elapsed'])
            cpu_pct = float(proc['cpu'])
            cores = int(cpu_pct / 100)

            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Training running for {elapsed} | "
                  f"CPU: {proc['cpu']}% (~{cores} cores) | "
                  f"MEM: {proc['mem']}% | "
                  f"PID: {proc['pid']}", end='', flush=True)

            # Estimate progress (very rough)
            minutes = 0
            if ':' in proc['elapsed']:
                parts = proc['elapsed'].split(':')
                if len(parts) == 2:
                    minutes = int(parts[0])
                elif len(parts) == 3:
                    minutes = int(parts[0]) * 60 + int(parts[1])

            # Training typically takes 10-15 minutes
            if iteration % 10 == 0:  # Every 10 iterations (20 seconds)
                if minutes < 10:
                    print(f"\n   Estimated: ~{10-minutes} to {15-minutes} minutes remaining")
                elif minutes >= 10 and minutes < 15:
                    print(f"\n   Should complete soon (typical: 10-15 minutes total)")
                elif minutes >= 15:
                    print(f"\n   Taking longer than usual... check for issues if over 20 minutes")
        else:
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                  f"✅ Training process not found - likely COMPLETED!", end='')
            print("\n\nRun this to verify:")
            print("  python3 tools/check_ml_status.py")
            break

        time.sleep(2)

except KeyboardInterrupt:
    print("\n\n✋ Monitoring stopped (training continues in background)")
    if proc:
        print(f"\nTraining is still running (PID: {proc['pid']})")
        print("\nTo check status later:")
        print("  python3 tools/monitor_training.py")
        print("\nTo check if complete:")
        print("  python3 tools/check_ml_status.py")

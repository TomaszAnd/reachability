"""Compare before/after benchmark results."""
import json
from pathlib import Path


def compare(before_dir: str, after_seq_dir: str, after_par_dir: str = None):
    before = json.load(open(Path(before_dir) / 'results.json'))
    after_seq = json.load(open(Path(after_seq_dir) / 'results.json'))
    after_par = json.load(open(Path(after_par_dir) / 'results.json')) if after_par_dir else {}

    has_par = bool(after_par)
    header = f"{'Model':<20} | {'Before':>10} | {'After(seq)':>10}"
    if has_par:
        header += f" | {'After(par)':>10} | {'Speedup(par)':>12}"
    else:
        header += f" | {'Speedup(seq)':>12}"

    print("="*70)
    print("BEFORE vs AFTER COMPARISON")
    print("="*70)
    print(header)
    print("-"*70)

    total_before = 0
    total_after_seq = 0
    total_after_par = 0

    for key in before:
        t_before = before[key]['time_seconds']
        t_after_seq = after_seq.get(key, {}).get('time_seconds', float('inf'))
        total_before += t_before
        total_after_seq += t_after_seq

        speedup_seq = t_before / t_after_seq if t_after_seq > 0 else 0
        line = f"{key:<20} | {t_before:>9.1f}s | {t_after_seq:>9.1f}s"

        if has_par:
            t_after_par = after_par.get(key, {}).get('time_seconds', float('inf'))
            total_after_par += t_after_par
            speedup_par = t_before / t_after_par if t_after_par > 0 else 0
            line += f" | {t_after_par:>9.1f}s | {speedup_par:>11.1f}x"
        else:
            line += f" | {speedup_seq:>11.1f}x"

        print(line)

    print("-"*70)
    line = f"{'TOTAL':<20} | {total_before:>9.1f}s | {total_after_seq:>9.1f}s"
    if has_par:
        speedup = total_before / total_after_par if total_after_par > 0 else 0
        line += f" | {total_after_par:>9.1f}s | {speedup:>11.1f}x"
    else:
        speedup = total_before / total_after_seq if total_after_seq > 0 else 0
        line += f" | {speedup:>11.1f}x"
    print(line)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('before', help='Path to before benchmark dir')
    parser.add_argument('after_seq', help='Path to after (sequential) benchmark dir')
    parser.add_argument('--after-par', help='Path to after (parallel) benchmark dir')
    args = parser.parse_args()
    compare(args.before, args.after_seq, args.after_par)

#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
# ]
# ///

import pstats, os, sys

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} old_profile.out new_profile.out")
        sys.exit(1)

    old_file, new_file = sys.argv[1], sys.argv[2]

    old = pstats.Stats(old_file)
    new = pstats.Stats(new_file)

    diffs = []
    for func, new_stats in new.stats.items():
        new_time = new_stats[3]  # cumulative time
        old_time = old.stats.get(func, (0,0,0,0))[3]
        change = new_time - old_time
        if abs(change) > 1e-6:
            file, line, name = func
            short_file = os.path.basename(file) if file != "~" else "~"
            label = f"{short_file}:{line}:{name}"
            diffs.append((change, new_time, old_time, label))

    # Pretty header
    print(f"{'Function':50} | {'Δ (s)':>10} | {'New (s)':>10} | {'Old (s)':>10} | Note")
    print("-" * 100)

    for change, new_time, old_time, label in sorted(diffs, key=lambda x: -abs(x[0]))[:20]:
        note = "SLOWER" if change > 0 else "FASTER"
        print(f"{label:50} | {change:+10.4f} | {new_time:10.4f} | {old_time:10.4f} | {note}")

if __name__ == "__main__":
    main()

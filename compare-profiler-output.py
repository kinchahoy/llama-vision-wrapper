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

    # Calculate overall totals
    old_total = old.total_tt
    new_total = new.total_tt
    total_change = new_total - old_total

    # Print overall comparison
    print(f"Profile Comparison:")
    print(f"  Old file: {old_file}")
    print(f"  New file: {new_file}")
    print(f"  Old total time: {old_total:.4f}s")
    print(f"  New total time: {new_total:.4f}s")
    print(f"  Overall change: {total_change:+.4f}s ({((new_total/old_total - 1) * 100):+.1f}%)")
    print()

    # Thresholds for significance
    MIN_ABS_CHANGE = 0.001  # 1ms minimum absolute change
    MIN_PCT_CHANGE = 5.0    # 5% minimum percentage change
    
    significant_diffs = []
    insignificant_diffs = []
    
    for func, new_stats in new.stats.items():
        new_time = new_stats[3]  # cumulative time
        old_time = old.stats.get(func, (0,0,0,0))[3]
        change = new_time - old_time
        
        if abs(change) > 1e-6:  # Basic filter for any change
            file, line, name = func
            short_file = os.path.basename(file) if file != "~" else "~"
            label = f"{short_file}:{line}:{name}"
            
            # Calculate percentage change
            if old_time > 0:
                pct_change = (change / old_time) * 100
            else:
                pct_change = float('inf') if change > 0 else 0
            
            # Determine if change is significant
            is_significant = (abs(change) >= MIN_ABS_CHANGE and 
                            abs(pct_change) >= MIN_PCT_CHANGE)
            
            entry = (change, new_time, old_time, label, pct_change)
            if is_significant:
                significant_diffs.append(entry)
            else:
                insignificant_diffs.append(entry)

    # Show significant changes first
    if significant_diffs:
        print(f"Significant Function Changes (≥{MIN_ABS_CHANGE:.3f}s and ≥{MIN_PCT_CHANGE}%):")
        print(f"{'Function':50} | {'Δ (s)':>10} | {'Δ (%)':>8} | {'New (s)':>10} | {'Old (s)':>10} | Note")
        print("-" * 110)

        for change, new_time, old_time, label, pct_change in sorted(significant_diffs, key=lambda x: -abs(x[0])):
            note = "SLOWER" if change > 0 else "FASTER"
            pct_str = f"{pct_change:+.1f}" if abs(pct_change) < 999 else "∞" if pct_change > 0 else "-∞"
            print(f"{label:50} | {change:+10.4f} | {pct_str:>8} | {new_time:10.4f} | {old_time:10.4f} | {note}")
        print()

    # Show a few insignificant changes at the bottom
    if insignificant_diffs:
        print(f"Minor Changes (showing top 10 by absolute change):")
        print(f"{'Function':50} | {'Δ (s)':>10} | {'Δ (%)':>8} | {'New (s)':>10} | {'Old (s)':>10} | Note")
        print("-" * 110)
        
        for change, new_time, old_time, label, pct_change in sorted(insignificant_diffs, key=lambda x: -abs(x[0]))[:10]:
            pct_str = f"{pct_change:+.1f}" if abs(pct_change) < 999 else "∞" if pct_change > 0 else "-∞"
            print(f"{label:50} | {change:+10.4f} | {pct_str:>8} | {new_time:10.4f} | {old_time:10.4f} | NOT SIG")
        
        if len(insignificant_diffs) > 10:
            print(f"... and {len(insignificant_diffs) - 10} more minor changes")

if __name__ == "__main__":
    main()

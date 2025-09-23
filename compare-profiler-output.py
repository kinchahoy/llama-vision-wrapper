#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
# ]
# ///

import pstats, os, sys, argparse

def main():
    parser = argparse.ArgumentParser(description='Compare two profiler output files')
    parser.add_argument('old_profile', help='Path to old profile file')
    parser.add_argument('new_profile', help='Path to new profile file')
    parser.add_argument('-n', '--top', type=int, default=10, 
                       help='Number of top results to show in each category (default: 10)')
    
    args = parser.parse_args()
    
    old_file, new_file = args.old_profile, args.new_profile
    top_n = args.top

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

    # Categorize functions
    new_functions = []      # Functions that exist only in new profile
    changed_functions = []  # Functions that exist in both with significant changes
    minor_changes = []      # Functions with insignificant changes
    
    # Thresholds for significance
    MIN_ABS_CHANGE = 0.001  # 1ms minimum absolute change
    MIN_PCT_CHANGE = 5.0    # 5% minimum percentage change
    
    # Process all functions in new profile
    for func, new_stats in new.stats.items():
        new_time = new_stats[3]  # cumulative time
        old_time = old.stats.get(func, (0,0,0,0))[3]
        
        file, line, name = func
        short_file = os.path.basename(file) if file != "~" else "~"
        label = f"{short_file}:{line}:{name}"
        
        if old_time == 0:  # New function
            new_functions.append((new_time, label))
        else:
            change = new_time - old_time
            if abs(change) > 1e-6:  # Has some change
                # Calculate percentage change
                pct_change = (change / old_time) * 100
                
                # Determine if change is significant
                is_significant = (abs(change) >= MIN_ABS_CHANGE and 
                                abs(pct_change) >= MIN_PCT_CHANGE)
                
                entry = (change, new_time, old_time, label, pct_change)
                if is_significant:
                    changed_functions.append(entry)
                else:
                    minor_changes.append(entry)

    # Show new functions
    if new_functions:
        print(f"New Functions (top {min(top_n, len(new_functions))} by time):")
        print(f"{'Function':50} | {'Time (s)':>10}")
        print("-" * 65)
        for new_time, label in sorted(new_functions, key=lambda x: -x[0])[:top_n]:
            print(f"{label:50} | {new_time:10.4f}")
        if len(new_functions) > top_n:
            print(f"... and {len(new_functions) - top_n} more new functions")
        print()

    # Show significant changes
    if changed_functions:
        print(f"Significant Changes (top {min(top_n, len(changed_functions))} by absolute change):")
        print(f"{'Function':50} | {'Δ (s)':>10} | {'Δ (%)':>8} | {'New (s)':>10} | {'Old (s)':>10}")
        print("-" * 105)
        for change, new_time, old_time, label, pct_change in sorted(changed_functions, key=lambda x: -abs(x[0]))[:top_n]:
            pct_str = f"{pct_change:+.1f}" if abs(pct_change) < 999 else "∞" if pct_change > 0 else "-∞"
            print(f"{label:50} | {change:+10.4f} | {pct_str:>8} | {new_time:10.4f} | {old_time:10.4f}")
        if len(changed_functions) > top_n:
            print(f"... and {len(changed_functions) - top_n} more significant changes")
        print()

    # Show minor changes only if requested and there are any
    if minor_changes and top_n > 0:
        print(f"Minor Changes (top {min(top_n, len(minor_changes))} by absolute change):")
        print(f"{'Function':50} | {'Δ (s)':>10} | {'Δ (%)':>8} | {'New (s)':>10} | {'Old (s)':>10}")
        print("-" * 105)
        for change, new_time, old_time, label, pct_change in sorted(minor_changes, key=lambda x: -abs(x[0]))[:top_n]:
            pct_str = f"{pct_change:+.1f}" if abs(pct_change) < 999 else "∞" if pct_change > 0 else "-∞"
            print(f"{label:50} | {change:+10.4f} | {pct_str:>8} | {new_time:10.4f} | {old_time:10.4f}")
        if len(minor_changes) > top_n:
            print(f"... and {len(minor_changes) - top_n} more minor changes")

if __name__ == "__main__":
    main()

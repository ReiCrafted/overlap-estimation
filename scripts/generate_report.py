import argparse
import sys
from pathlib import Path

# Add project root to path so 'overlap_detection' is discoverable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from overlap_detection.reporting import write_summary_report

def parse_args():
    parser = argparse.ArgumentParser(description="Generate summary report from aggregate results.")
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory containing aggregate_results.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save the report and plots.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    csv_path = args.results_dir / "aggregate_results.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return
        
    print(f"Generating report from {csv_path} into {args.output_dir}...")
    write_summary_report(csv_path, args.output_dir)
    print("Done!")

if __name__ == "__main__":
    main()

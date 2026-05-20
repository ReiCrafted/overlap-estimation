import argparse
import sys
from pathlib import Path

# Add project root to path so 'overlap_detection' is discoverable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from overlap_detection.annotation_gui import AnnotationGUI

def parse_args():
    parser = argparse.ArgumentParser(description="Launch ground truth annotation GUI.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory containing image pairs.")
    parser.add_argument("--annotator", type=str, required=True, help="Name of the annotator.")
    return parser.parse_args()

def main():
    args = parse_args()
    gui = AnnotationGUI(
        dataset_dir=args.dataset_dir,
        annotator_name=args.annotator,
    )
    gui.run()

if __name__ == "__main__":
    main()

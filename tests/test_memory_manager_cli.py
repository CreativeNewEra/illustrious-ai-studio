import os
import sys

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from memory_manager import create_parser


def test_memory_manager_args():
    parser = create_parser()
    args = parser.parse_args(["--profile", "balanced", "--threshold", "low:65", "--threshold", "high:90"])
    assert args.profile == "balanced"
    assert args.threshold == ["low:65", "high:90"]

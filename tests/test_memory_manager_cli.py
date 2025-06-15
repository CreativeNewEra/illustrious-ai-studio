from illustrious_ai_studio.memory_manager import create_parser


def test_memory_manager_args():
    parser = create_parser()
    args = parser.parse_args(["--profile", "balanced", "--threshold", "low:65", "--threshold", "high:90"])
    assert args.profile == "balanced"
    assert args.threshold == ["low:65", "high:90"]

import os
import sys

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())


def test_defaults():
    from main import create_parser
    parser = create_parser()
    args = parser.parse_args([])
    assert args.lazy_load is False
    assert args.no_api is False
    assert args.web_port == 7860
    assert args.api_port == 8000
    assert args.share is False
    assert args.log_level == "INFO"
    assert args.memory_profile is None
    assert args.memory_threshold is None


def test_flags_and_ports():
    from main import create_parser
    parser = create_parser()
    args = parser.parse_args(["--lazy-load", "--no-api", "--web-port", "9000", "--api-port", "1234", "--share"])
    assert args.lazy_load is True
    assert args.no_api is True
    assert args.web_port == 9000
    assert args.api_port == 1234
    assert args.share is True

def test_memory_cli_options():
    from main import create_parser
    parser = create_parser()
    args = parser.parse_args([
        "--memory-profile", "aggressive",
        "--memory-threshold", "low:60",
        "--memory-threshold", "high:90",
    ])
    assert args.memory_profile == "aggressive"
    assert args.memory_threshold == ["low:60", "high:90"]

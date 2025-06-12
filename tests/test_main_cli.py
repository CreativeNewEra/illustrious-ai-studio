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
    assert args.log_level == "INFO"


def test_flags_and_ports():
    from main import create_parser
    parser = create_parser()
    args = parser.parse_args(["--lazy-load", "--no-api", "--web-port", "9000", "--api-port", "1234"])
    assert args.lazy_load is True
    assert args.no_api is True
    assert args.web_port == 9000
    assert args.api_port == 1234

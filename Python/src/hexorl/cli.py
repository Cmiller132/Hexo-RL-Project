"""CLI entry point for hexorl."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Hexo-RL training pipeline")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("epoch", help="Run one training epoch")
    subparsers.add_parser("bench", help="Run benchmarks")
    subparsers.add_parser("arena", help="Run evaluation arena")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    print(f"hexorl {args.command} — Phase 1 stub (not yet implemented)")


if __name__ == "__main__":
    main()

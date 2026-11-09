"""Run optimized alpha=0.4 simulation and then render its baseline comparison."""

from .optimized_alpha04_movie import create_movie
from .optimized_alpha04_run import run


def main():
    run()
    create_movie()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

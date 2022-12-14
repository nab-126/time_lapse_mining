"""Utility functions for argument parsing."""

import argparse

def get_args_parser() -> argparse.ArgumentParser:
    """Create argparser for common main function."""
    parser = argparse.ArgumentParser(
        description="Time Lapse Mining", add_help=False
    )

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--keyword", type=str, nargs='+')
    parser.add_argument("--cluster_id", default=0, type=int)
    parser.add_argument("--top_k", default=20, type=int, help="number of time lapses to reconstruct for a dataset") 
    parser.add_argument("--num_iterations", default=2e4, type=int, help="number of iterations for which to run RANSAC")
    parser.add_argument("--fps", default=10, type=int, help="frames per second of the video")
    parser.add_argument("--num_steps", default=500, type=int, help="number of iterations for which to run the optimization algorighm for smoothing the time lapse")
    parser.add_argument("--lmbd", default=10, type=int, help="optimization algorithm lmbd")
    
    return parser

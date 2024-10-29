"""
Convert a diskcache cache to our new cache format.

"""

import argparse
import os

from diskcache import Cache

from benchmark.utils.cache.disk_cache import HDF5DiskCache


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--cache_path", type=str, help="Path to the cache")
    argparser.add_argument("--output_path", type=str, help="Path to the output cache")
    args = argparser.parse_args()

    if not os.path.exists(args.cache_path):
        raise FileNotFoundError(f"Cache path {args.cache_path} does not exist.")

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    cache = Cache(args.cache_path)
    output_cache = HDF5DiskCache(args.output_path)

    cache_size = len(cache)
    for i, key in enumerate(cache):
        if i % 100 == 0:
            print(f"Transfer in progress {i}/{cache_size}...")
        output_cache[key] = cache[key]

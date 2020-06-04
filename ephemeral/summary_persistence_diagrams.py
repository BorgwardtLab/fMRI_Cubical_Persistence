import argparse

import numpy as np

from topology import load_persistence_diagram_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'FILES',
        nargs='+',
        type=str,
        help='Input files')

    args = parser.parse_args()

    counts = []

    for filename in args.FILES:
        dimensions, _, _ = load_persistence_diagram_json(filename)
        unique_dimensions = set(dimensions)

        counts.append(len(dimensions == 2))

        for dimension in unique_dimensions:
            print(f'{dimension}: {len(dimensions == dimension)}')

    print(np.mean(counts))

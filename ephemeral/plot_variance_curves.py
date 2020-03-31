import argparse
import json

import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='Input file')

    args = parser.parse_args()

    with open(args.INPUT) as f:
        data = json.load(f)

    blocked_fields = [
        'input',
        'dimension',
        'statistic',
        'power'
    ]

    n_statistics = len(data.keys()) - len(blocked_fields)
    statistics = sorted([s for s in data.keys() if s not in blocked_fields])

    fig, ax = plt.subplots(n_statistics, 1)

    for index, statistic in enumerate(statistics):
        ax[index].plot(data[statistic], label=statistic)
        ax[index].legend()

    plt.show()

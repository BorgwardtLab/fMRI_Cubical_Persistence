#!/usr/bin/env python
#
# Calculates a persistence image from a persistence diagram. This is an
# auxiliary script for creating additional figures.

import argparse

from persim import PersImage
from topology import load_persistence_diagram_txt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='Input file')
    parser.add_argument('RESOLUTION', type=int, help='Resolution')

    args = parser.parse_args()

    pd = load_persistence_diagram_txt(args.INPUT)

    clf = PersImage()
    X = clf.transform([pd])[0]

    with open(f'PI_{args.RESOLUTION}.txt', 'w') as f:
        for j in range(args.RESOLUTION):
            for i in range(args.RESOLUTION):
                print(f'{i} {j} {X[j, i]}', file=f)
            print('', file=f)

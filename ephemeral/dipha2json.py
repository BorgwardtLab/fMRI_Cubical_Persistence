#!/usr/bin/env python
#
# Converts DIPHA persistence diagrams to JSON files. While JSON files
# require more space, they are easier to query and process. The files
# will follow the original DIPHA format and consist of three arrays:
#
#   - dimensions (key = `dimensions`)
#   - creation values (key = `creation_values`)
#   - destruction values (key = `destruction_values`)
#
# Additional information, if any, will be stored in other keys
# pertaining to the individual subjects.


import argparse
import json
import os
import warnings

from topology import load_persistence_diagram_dipha

from tqdm import tqdm


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', nargs='+', type=str, help='Input file(s)')

    args = parser.parse_args()

    for filename in tqdm(args.FILE, desc='File'):
        dimensions, creation_values, destruction_values = \
            load_persistence_diagram_dipha(filename)

        # Dimensions can be stored as integers; we do not want
        # floating point numbers here because they might cause
        # troubles later on.
        dimensions = dimensions.astype('int64')

        output_filename = os.path.splitext(filename)[0] + '.json'

        # Make sure that we are not overwriting anything.
        if os.path.exists(output_filename):
            warnings.warn(f'{output_filename} already exists. Will '
                          f'ignore it and continue.')

            continue

        with open(output_filename, 'w') as f:
            json.dump({
                    'dimensions': dimensions.tolist(),
                    'creation_values': list(creation_values),
                    'destruction_values': list(destruction_values)
                }, f, indent=4
            )

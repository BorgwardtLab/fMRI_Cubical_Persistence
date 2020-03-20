"""Feature vector creation approaches for persistence diagrams.

This module contains feature vector creation approaches for persistence
diagrams that permit the use in modern machine learning algorithms.
"""


def persistence(x, y):
    '''
    Helper function for calculating the persistence of a tuple in
    a persistence diagram.

    :param x: First coordinate
    :param y: Second coordinate

    :return: Persistence of the feature; this is *always* a non-negative
    number.
    '''

    return abs(x - y)


def featurise_distances(diagram):
    '''
    Creates a feature vector by calculating distances to the diagonal
    for every point in the diagram and returning a sorted vector. The
    representation is *stable* but might not be discriminative.

    :param diagram: Persistence diagram

    :return: Sorted vector of distances to diagonal. The vector is
    sorted in *descending* order such that high persistence points
    precede those of low persistence.
    '''

    distances = [persistence(x, y) for x, y in diagram]
    return sorted(distances, reverse=True)


def featurise_pairwise_distances(diagram):
    '''
    Creates a feature vector by calculating the minimum of pairwise
    distances and distances to the diagonal of each pair of points.
    This representation follows the paper:

        Stable Topological Signatures for Points on 3D Shapes

    The representation is stable, but more costly to compute.

    :param diagram: Persistence diagram

    :return: Sorted vector of distances as described above. The vector
    is sorted in *descending* order.
    '''

    distances = []

    # Auxiliary function for calculating the infinity distance between
    # the two points.
    def distance(a, b, x, y):
        return max(abs(a - x), abs(b - y))

    for i, (a, b) in enumerate(diagram):
        for j, (x, y) in enumerate(diagram[i:]):
            k = i + j  # not required for now

            m = min(
                    distance(a, b, x, y),
                    persistence(a, b),
                    persistence(x, y)
                )

            distances.append(m)

    return sorted(distances, reverse=True)

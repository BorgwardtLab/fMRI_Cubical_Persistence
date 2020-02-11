'''
Classes and functions for representing topological descriptors and
working with them.
'''


import collections
import json
import math
import numbers

import numpy as np
import pandas as pd

from scipy.stats import moment
from sklearn.neighbors import NearestNeighbors


class PersistenceDiagram(collections.Sequence):
    '''
    Represents a persistence diagram, i.e. a pairing of nodes in
    a graph. The purpose of this class is to provide a *simpler*
    interface for storing and accessing this pairing.
    '''

    def __init__(self,
                 dimension=None,
                 creation_values=None,
                 destruction_values=None):
        '''
        Creates a new persistence diagram. Depending on the parameters
        supplied, the diagram is either created empty or with a set of
        pairs. If pairs are supplied, consistency will be checked.

        Parameters
        ----------

            dimension:
                The dimension of the persistence diagram (optional)

            creation_values:
                An optional set of creation values (creation times) for
                the tuples in the persistence diagram. If this is given
                the `destruction_values` argument must also be present.
                Moreover, the two vectors need to have the same length.

            destruction_values:
                An optional set of destruction values (destruction times)
                for the tuples in the persistence diagram. The same rules
                as for the `creation_values` apply.
        '''

        self._pairs = []
        self._dimension = dimension

        if creation_values is not None or destruction_values is not None:
            assert creation_values is not None
            assert destruction_values is not None

            assert len(creation_values) == len(destruction_values)

            for c, d in zip(creation_values, destruction_values):
                self.append(c, d)

    @property
    def dimension(self):
        '''
        Returns the dimension of the persistence diagram. This is
        permitted to be `None`, indicating that *no* dimension is
        specified.
        '''
        return self._dimension

    def __len__(self):
        '''
        :return: The number of pairs in the persistence diagram.
        '''

        return len(self._pairs)

    def __getitem__(self, index):
        '''
        Returns the persistence pair at the given index.
        '''

        return self._pairs[index]

    def add(self, x, y):
        '''
        Appends a new persistence pair to the given diagram. Performs no
        other validity checks.

        :param x: Creation value of the given persistence pair
        :param y: Destruction value of the given persistence pair
        '''

        self._pairs.append((x, y))

    def append(self, x, y):
        '''
        Alias for `add()`. Adds a new persistence pair to the diagram.
        '''

        self.add(x, y)

    def union(self, other):
        '''
        Calculates the union of two persistence diagrams. The current
        persistence diagram is modified in place.

        :param other: Other persistence diagram
        :return: Updated persistence diagram
        '''

        for x, y in other:
            self.add(x, y)

        return self

    def total_persistence(self, p=1):
        '''
        Calculates the total persistence of the current pairing.
        '''

        return sum([abs(x - y)**p for x, y in self._pairs])**(1.0 / p)

    def infinity_norm(self, p=1):
        '''
        Calculates the infinity norm of the current pairing.
        '''

        return max([abs(x - y)**p for x, y in self._pairs])

    def remove_diagonal(self):
        '''
        Removes diagonal elements, i.e. elements for which x and
        y coincide.
        '''

        self._pairs = [(x, y) for x, y in self._pairs if x != y]

    def above_diagonal(self):
        '''
        Returns diagram that consists of all persistence points above
        the diagonal, as well as all diagonal points.
        '''

        pd = PersistenceDiagram()

        for x, y in self:
            if x <= y:
                pd.add(x, y)

        return pd

    def below_diagonal(self):
        '''
        Returns diagram that consists of all persistence points below
        the diagonal, as well as all diagonal points.
        '''

        pd = PersistenceDiagram()

        for x, y in self:
            if x >= y:
                pd.add(x, y)

        return pd

    def persistence(self):
        '''
        Returns a list of all persistence values. This is useful for
        calculating statistics based on their distribution.
        '''

        return [abs(x - y) for x, y in self]

    def persistence_moment(self, order=1):
        '''
        Calculates persistence moments, i.e. moments of the persistence
        values of a persistence diagram.

        :param order: Order of the moment that should be calculated
        :return: Persistence moment of the specified order
        '''

        return moment(self.persistence(), moment=order, axis=None)

    def nearest_neighbours(self, k=1):
        '''
        Calculates the nearest neighbours of each point in the
        persistence diagram and returns them. To evaluate each
        neighbour, the Chebyshev metric is used.

        :param k: Number of nearest neighbours to evaluate. By default,
        only a single nearest neighbour will be returned.

        :return: Tuple of *distances* and *indices* corresponding to the
        nearest neighbour of each point in the diagram.
        '''

        nn = NearestNeighbors(n_neighbors=1, metric='chebyshev')
        nn.fit(self._pairs)

        return nn.kneighbors()

    def nn_distances(self, k=1):
        '''
        Returns a list of all nearest neighbour distances of the
        diagram.
        '''

        distances, _ = self.nearest_neighbours(k)

        return distances.ravel()

    def entropy(self):
        '''
        Calculates a simple persistent entropy of the diagram, i.e. an
        entropy measure that takes into account all persistence values
        and returns an appropriately weighted sum.
        '''

        pers = self.persistence()
        total_pers = np.sum(pers)
        probabilities = np.array([p / total_pers for p in pers])

        return np.sum(-probabilities * np.log(probabilities))

    def spatial_entropy(self):
        '''
        Calculates a simple spatial entropy of the diagram that is based
        on the *relative* distribution of points in the diagram.
        '''

        distances = self.nn_distances()
        areas = 2 * math.pi * distances**2
        total_area = np.sum(areas)
        probabilities = np.array([areas / total_area for area in areas])

        # Ensures that a probability of zero will just result in
        # a logarithm of zero as well. This is required whenever
        # one deals with entropy calculations.
        log_prob = np.log(probabilities,
                          out=np.zeros_like(probabilities),
                          where=(probabilities > 0))

        return np.sum(-probabilities * log_prob)

    def __repr__(self):
        '''
        :return: String-based representation of the diagram
        '''

        return '\n'.join([f'{x} {y}' for x, y in self._pairs])


def load_persistence_diagram_txt(filename, comment='#'):
    '''
    Loads a persistence diagram from a filename and returns it. No
    additional error checking will be performed. The function just
    assumes that the file consists of tuples that can be converted
    to a `float` representation. Empty lines and comment lines are
    skipped.

    :param filename: Input filename
    :param comment: Optional comment character; lines starting with this
    character are skipped

    :return: Persistence diagram
    '''

    pd = PersistenceDiagram()

    with open(filename) as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comment lines
            if not line:
                continue
            elif line.startswith(comment):
                continue

            x, y = map(float, line.split())

            pd.add(x, y)

    return pd

def _create_persistence_diagrams(dimensions, creation, destruction):
    '''
    Internal utility function for creating a set of persistence diagrams
    from same-size lists. This is useful when reading diagrams in either
    DIPHA or in JSON format.

    Parameters
    ----------

        dimensions:
            List of dimensions for each (creation, destruction) tuple

        creation:
            List of creation values for persistence pairs

        destruction:
            List of destruction values for persistence pairs

    Returns
    -------

    Sequence of persistence diagrams, one for each unique dimension in
    the `dimensions` list.
    '''

    # Create a persistence diagram for each unique dimension in the
    # data.
    unique_dimensions = np.unique(dimensions)
    persistence_diagrams = []

    for dimension in unique_dimensions:
        C = creation[dimensions == dimension]
        D = destruction[dimensions == dimension]

        persistence_diagrams.append(
            PersistenceDiagram(dimension=dimension,
                               creation_values=C,
                               destruction_values=D)
        )

    return persistence_diagrams


def load_persistence_diagram_dipha(filename, return_raw=True):
    '''
    Loads a persistence diagram from a file. The file is assumed to be
    in DIPHA format.

    Parameters
    ----------

        filename:
            Filename to load the persistence diagram from. The file
            needs to be in DIPHA format, i.e. a binary format. This
            function checks whether the format is correct.

        return_raw:
            Flag indicating whether the *raw* persistence values shall
            be returned. If set, will return triples:

                - dimension
                - creation values
                - destruction values

            Each of these will be an array indicating the corresponding
            value. The persistence diagram could then be constructed by
            extracting a subset of the values.

            If `return_raw` is False, a sequence of `PersistenceDiagram`
            instances will be returned instead.

    Returns
    -------

    Raw triples (dimension, creation, destruction) or a sequence of
    persistence diagrams, depending on the `return_raw` parameter.
    '''

    def _read_int64(f):
        return np.fromfile(f, dtype=np.int64, count=1)[0]

    def _fromfile(f, dtype, count, skip):
        data = np.zeros((count, ))

        for c in range(count):
            data[c] = np.fromfile(f, dtype=dtype, count=1)[0]
            f.seek(f.tell() + skip)

        return data

    with open(filename, 'rb') as f:

        magic_number = _read_int64(f)
        file_id = _read_int64(f)

        # Ensures that this is DIPHA file containing a persistence
        # diagram, and nothing something else.
        assert magic_number == 8067171840
        assert file_id == 2

        n_pairs = _read_int64(f)

        # FIXME: this does *not* follow the original MATLAB script, but
        # it produces the proper results.
        dimensions = _fromfile(
            f,
            dtype=np.int64,
            count=n_pairs,
            skip=16
        )

        # Go back whence you came!
        f.seek(0, 0)
        f.seek(32)

        creation_values = _fromfile(
            f,
            dtype=np.double,
            count=n_pairs,
            skip=16
        )

        # Go back whence you came!
        f.seek(0, 0)
        f.seek(40)

        destruction_values = _fromfile(
            f,
            dtype=np.double,
            count=n_pairs,
            skip=16
        )

    if return_raw:
        return dimensions, creation_values, destruction_values
    else:
        return _create_persistence_diagrams(dimensions,
                                            creation_values,
                                            destruction_values)


def load_persistence_diagram_json(filename, return_raw=True):
    '''
    Loads a persistence diagram from a file. The file is assumed to be
    in JSON format. Like `load_persistence_diagram_dipha`, this method
    permits loading 'raw' values or persistence diagrams.

    Parameters
    ----------

        filename:
            Filename to load the persistence diagram from. The file
            needs to be in JSON format, with at least three keys in
            the file:

                - `dimensions`
                - `creation_values`
                - `destruction_values`

            The function checks whether the file format is correct.

        return_raw:
            Flag indicating whether the *raw* persistence values shall
            be returned. If set, will return triples:

                - dimension
                - creation values
                - destruction values

            Each of these will be an array indicating the corresponding
            value. The persistence diagram could then be constructed by
            extracting a subset of the values.

            If `return_raw` is False, a sequence of `PersistenceDiagram`
            instances will be returned instead.

    Returns
    -------

    Raw triples (dimension, creation, destruction) or a sequence of
    persistence diagrams, depending on the `return_raw` parameter.
    '''

    with open(filename, 'r') as f:
       data = json.load(f)

    assert 'dimensions' in data.keys()
    assert 'creation_values' in data.keys()
    assert 'destruction_values' in data.keys()

    dimensions = data['dimensions']
    creation_values = data['creation_values']
    destruction_values = data['destruction_values']

    if return_raw:
        return dimensions, creation_values, destruction_values
    else:
        return _create_persistence_diagrams(dimensions,
                                            creation_values,
                                            destruction_values)


def make_betti_curve(diagram, ignore_errors=False):
    '''
    Creates a Betti curve of a persistence diagram, i.e. a curve that
    depicts the number of active intervals according to the threshold
    of the filtration.

    :param diagram: Persistence diagram to convert
    :param ignore_errors: If set, ignores errors instead of raising an
    exception. This is only meant for debugging.

    :return: List of tuples of the form (x, y), where x refers to
    a threshold, and y refers to a function value.
    '''

    event_points = []

    for x, y in diagram:
        event_points.append((x, True))
        event_points.append((y, False))

    event_points = sorted(event_points, key=lambda x: x[0])
    n_active = 0

    output = []

    # Create the 'raw' sequence of event points first. This blindly
    # assumes that all creation and destruction times are different
    # from each other.

    for p, is_generator in event_points:
        if is_generator:
            n_active += 1
        else:
            n_active -= 1

        output.append((p, n_active))

    # Check some edge cases first: if both a generator and a destroyer
    # share the same time, the barcode representation will not exhibit
    # any gaps. This creates undesirable situations when creating each
    # of the curves.

    creation_times = set([t for t, g in event_points if g])
    destruction_times = set([t for t, g in event_points if not g])

    shared_times = creation_times.intersection(destruction_times)
    if shared_times:
        if ignore_errors:
            pass
        else:
            raise RuntimeError('Inconsistent creation/destruction time')

    # If the diagram is empty, skip everything. In the following, I will
    # assume that at least a single point exists.
    if not event_points:
        return None

    prev_p = event_points[0][0]   # Previous time point
    prev_v = event_points[0][1]   # Previous number of active intervals

    output_ = []

    # Functor that is called to simplify the loop processing, which
    # requires one extra pass to handle the last interval properly.
    def process_event_points(p, n_active):

        # Admittedly, not the most elegant solution, but at least I do
        # not have to duplicate the loop body.
        nonlocal prev_p
        nonlocal prev_v
        nonlocal output_

        if prev_p == p:
            prev_v = n_active

        # Time point changed; the monotonically increasing subsequence
        # should now be stored.
        else:

            # Create a transition point if this is not the first output
            if output_:

                # This makes the previous interval half-open by
                # introducing a fake transition point *between*
                # the existing points.
                old_value = output_[-1][1]
                old_point = np.nextafter(prev_p, prev_p - 1)

                # Inserts a fake point to obtain half-open intervals for
                # the whole function.
                output_.append((old_point, old_value))

            output_.append((prev_p, prev_v))

            prev_p = p
            prev_v = n_active

    for p, n_active in output:
        process_event_points(p, n_active)

    # Store the last subsequence if applicable. To this end, we need to
    # check if the last proper output was different from our previously
    # seen value. If so, there's another sequence in the output that we
    # missed so far.
    if prev_p != output_[-1][0]:

        # Note that the two arguments are fake; they are only required
        # to trigger the insertion of another interval.
        process_event_points(prev_p + 1, prev_v + 1)

    output = output_
    return BettiCurve(output)


class BettiCurve:
    '''
    Class representing a Betti curve in some dimension, i.e. a curve
    that contains the number of active topological features at every
    point of a filtration process.

    This class provides some required wrapper functions to simplify,
    and improve, the usage of this concept.
    '''

    def __init__(self, values):
        '''
        Creates a new Betti curve from a sequence of values. The values
        are supposed to be ordered according to their filtration value,
        such that the first dimension represents the filtration axis.

        :param values: Input values
        '''

        if isinstance(values, pd.Series):
            self._data = values

            # It's brute force, but this ensures that the data frames
            # are compatible with each other.
            assert self._data.index.name == 'x'

        else:
            self._data = pd.DataFrame.from_records(
                values, columns=['x', 'y'], index='x'
            )['y']

    def __repr__(self):
        return self._data.__repr__()

    def __add__(self, other):
        '''
        Performs addition of two Betti curves. This necessitates
        re-indexing values accordingly in order to evaluate them
        properly.

        In case `other` is a number, does elementwise addition.

        :param other: Betti curve to add to the current one
        :return: Betti curve that results from the addition
        '''

        if isinstance(other, numbers.Number):
            return BettiCurve(self._data + x)

        # Not a number, so let's re-index the Betti curve and perform
        # addition for the new curves.

        new_index = self._data.index.union(other._data.index)

        # The `fillna` is required because we might have a filtration
        # value that *precedes* the first index of one of the frames.
        left = self._data.reindex(new_index, method='ffill').fillna(0)
        right = other._data.reindex(new_index, method='ffill').fillna(0)

        return BettiCurve(left + right)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __neg__(self):
        '''
        Negates the current values of the Betti curves, i.e. applies
        a unary minus operation to the curve.

        :return: Negated Betti curve
        '''

        return BettiCurve(-self._data)

    def __sub__(self, other):
        return self.__add__(-other)

    def __abs__(self):
        '''
        Calculates the absolute value of the Betti curve. Does not
        modify the current Betti curve.

        :return: Absolute value of the Betti curve
        '''

        return BettiCurve(abs(self._data))

    def __truediv__(self, x):
        '''
        Elementwise division of a Betti curve by some number.

        :param x: Number to divide the Betti curve by
        :return: Betti curve divided by x
        '''

        return BettiCurve(self._data / x)

    def norm(self, p=1.0):
        '''
        Calculates an $L_p$ norm of the Betti curve and returns the
        result.

        :param p: Exponent for the corresponding $L_p$ norm
        :return: $L_p$ norm of the current Betti curve
        '''

        result = 0.0
        for (x1, y1), (x2, y2) in zip(
                self._data.iteritems(),
                self._data.shift(axis='index').dropna().iteritems()):

            m = (y2 - y1) / (x2 - x1)
            c = y1 - m * x1

            def evaluator(x):
                if m == 0.0:
                    return math.pow(c, p) * x
                else:
                    return math.pow(m*x + c, p+1) / (m * (p + 1))

            integral = abs(evaluator(x2) - evaluator(x1))
            result += integral

        return math.pow(result, 1.0 / p)

    def distance(self, other, p=1.0):
        '''
        Calculates the distance between the current Betti curve and
        another one, subject to a certain $L_p$ norm.

        :param other: Other Betti curve
        :param p: Exponent for the corresponding $L_p$ norm

        :return: Distance between the two curves
        '''

        return abs(self - other).norm(p)

import numpy

from molml.atom import LocalEncodedBond
from molml.utils import SMOOTHING_FUNCTIONS, SPACING_FUNCTIONS, get_depth_threshold_mask_connections

class LocalAxisEncodedBond(LocalEncodedBond):
    '''
    A smoothed histogram of atomic distances that is axis aligned.

    This extends the regular LocalEncoded

    Parameters
    ----------
    input_type : string, default='list'
        Specifies the format the input values will be (must be one of 'list'
        or 'filename').

    n_jobs : int, default=1
        Specifies the number of processes to create when generating the
        features. Positive numbers specify a specifc amount, and numbers less
        than 1 will use the number of cores the computer has.

    segments : int, default=100
        The number of bins/segments to use when generating the histogram.
        Empirically, it has been found that values beyond 50-100 have little
        benefit.

    smoothing : string or callable, default='norm'
        A string or callable to use to smooth the histogram values. If a
        callable is given, it must take just a single argument that is a float.
        For a list of supported default functions look at SMOOTHING_FUNCTIONS.

    start : float, default=-6.0
        The starting point for the histogram sampling in angstroms.

    end : float, default=6.0
        The ending point for the histogram sampling in angstroms.

    slope : float, default=20.
        A parameter to tune the smoothing values. This is applied as a
        multiplication before calling the smoothing function.

    max_depth : int, default=0
        A parameter to set the maximum geodesic distance to include in the
        interactions. A value of 0 signifies that all interactions are
        included.

    spacing : string, default="linear"
        The histogram interval spacing type. Must be one of ("linear",
        "inverse", or "log"). Linear spacing is normal spacing. Inverse takes
        and evaluates the distances as 1/r and the start and end points are
        1/x. For log spacing, the distances are evaluated as numpy.log(r)
        and the start and end points are numpy.log(x).

    Attributes
    ----------
    _element : list
        A list of all the element pairs in the fit molecules.
    '''
    def __init__(self, input_type='list', n_jobs=1, segments=100,
                 smoothing="norm", start=-6.0, end=6.0, slope=20., max_depth=0,
                 spacing="linear"):
        super(LocalAxisEncodedBond, self).__init__(
                                               input_type=input_type,
                                               n_jobs=n_jobs,
                                               segments=segments,
                                               smoothing=smoothing,
                                               start=start,
                                               end=end,
                                               slope=slope,
                                               max_depth=max_depth,
                                               spacing=spacing)
    
    def _para_transform(self, X, y=None):
        '''
        A single instance of the transform procedure

        This is formulated in a way that the transformations can be done
        completely parallel with map.

        Parameters
        ----------
        X : object
            An object to use for the transform

        Returns
        -------
        value : array, shape=(n_atoms, len(self._elements) * self.segments)
            The features extracted from the molecule
        '''
        if self._elements is None:
            msg = "This %s instance is not fitted yet. Call 'fit' first."
            raise ValueError(msg % type(self).__name__)

        try:
            smoothing_func = SMOOTHING_FUNCTIONS[self.smoothing]
        except KeyError:
            msg = "The value '%s' is not a valid spacing type."
            raise KeyError(msg % self.smoothing)

        pair_idxs = {key: i for i, key in enumerate(self._elements)}

        data = self.convert_input(X)

        vector = numpy.zeros((len(data.elements), len(self._elements),
                              3, self.segments))

        try:
            theta_func = SPACING_FUNCTIONS[self.spacing]
        except KeyError:
            msg = "The value '%s' is not a valid spacing type."
            raise KeyError(msg % self.spacing)

        theta = numpy.linspace(theta_func(self.start), theta_func(self.end),
                               self.segments)
        mat = get_depth_threshold_mask_connections(data.connections,
                                                   max_depth=self.max_depth)

	coords = numpy.array(data.coords)
        distances = coords - coords[:, None, :]
        for i, ele1 in enumerate(data.elements):
            for j, ele2 in enumerate(data.elements):
                if i == j or not mat[i, j]:
                    continue

                for k in xrange(3):
                    diff = theta - theta_func(distances[i, j, k])
                    value = smoothing_func(self.slope * diff)
                    vector[i, pair_idxs[ele2], k] += value
        return vector.reshape(len(data.elements), -1)

import sys

import numpy as np

from spatialpandas import GeoDataFrame
from spatialpandas.geometry import (
    LineArray,
    MultiLineArray,
    MultiPointArray,
    MultiPolygonArray,
    PointArray,
    PolygonArray,
    RingArray,
)
from spatialpandas.io import to_tiledb

point_array = PointArray(
    [
        [1, 2],
        [3, 4],
        [5, 6],
    ],
    np.uint16,
)

multipoint_array = MultiPointArray(
    [
        [1, 2, 3, 4, 5, 6],
        [7, 8],
        [9, 10, 11, 12]
    ],
    np.int16,
)

line_array = LineArray(
    [
        [1, 2, 3, 4, 5, 6],
        [7, 8],
        [9, 10, 11, 12],
    ],
    np.uint32,
)

multiline_array = MultiLineArray(
    [
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10]],
        [[11, 12]],
        [[13, 14, 15, 16], [17, 18, 18, 20, 21, 22], [23, 24]],
    ],
    np.int32,
)

# Square from (0, 0) to (1, 1) in CCW order
outline0 = [0, 0, 1, 0, 1, 1, 0, 1, 0, 0]

# Square from (2, 2) to (5, 5) in CCW order
outline1 = [2, 2, 2, 2, 5, 5, 2, 5, 2, 2]

outline2 = [3, 1, 4, 1, 3, 2, 3, 1]

# Triangle hole in CW order
hole1 = [3, 3, 4, 3, 3, 4, 3, 3]

ring_array = RingArray([outline0, outline1, outline2], np.uint32)

polygon_array = PolygonArray([[outline0], [outline1, hole1], [outline2]], np.uint64)

multipolygon_array = MultiPolygonArray(
    [
        [[outline0], [outline1, hole1]],
        [[outline2]],
        # None,
        [[outline0, outline2]],
    ]
)

# Build dataframe
df = GeoDataFrame(
    dict(
        a=np.arange(len(point_array)),
        point_array=point_array,
        multipoint_array=multipoint_array,
        line_array=line_array,
        ring_array=ring_array,
        multiline_array=multiline_array,
        polygon_array=polygon_array,
        multipolygon_array=multipolygon_array,
    )
)

to_tiledb(df, sys.argv[1])

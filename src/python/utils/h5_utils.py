"""
==========================================================================
CoQuí: Correlated Quantum ínterface

Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==========================================================================
"""

import numpy as np


def write_complex_dataset(grp, name, a):
    """Write a complex array to an h5py group in the TRIQS/nda layout read by the C++ side: float64
    with a trailing (re, im) dimension and a ``__complex__`` attribute."""
    a = np.ascontiguousarray(a, dtype=np.complex128)
    ds = grp.create_dataset(name, data=a.view(np.float64).reshape(a.shape + (2,)))
    ds.attrs["__complex__"] = np.int8(1)

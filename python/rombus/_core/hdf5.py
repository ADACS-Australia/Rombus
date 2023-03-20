import h5py

from typing import Union

# The following are used for type hints
file = h5py._hl.files.File
filename = str
file_or_filename = Union[file, str]


def ensure_open(file_in: file_or_filename) -> file:
    if isinstance(file_in, filename):
        return h5py.File(file_in, "r")
    else:
        return file_in

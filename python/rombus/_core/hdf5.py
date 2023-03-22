import h5py  # type: ignore

from typing import TypeVar, TypeAlias

# The following are used for type hints
File: TypeAlias = h5py.File
Filename: TypeAlias = str
FileOrFilename = TypeVar("FileOrFilename", Filename, File)


def ensure_open(file_in: FileOrFilename) -> File:
    if type(file_in) == Filename:
        return h5py.File(file_in, "r"), True
    else:
        return file_in, False

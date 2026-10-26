"""Secrets come only from keys.txt (FR-14); never from the environment, never printed."""
from pathlib import Path


class MissingKey(KeyError):
    pass


class Keys(dict):
    def require(self, name):
        if name not in self:
            raise MissingKey(name)
        return self[name]

    def __repr__(self):
        return f"Keys({sorted(self)})"

    __str__ = __repr__


def load_keys(path="keys.txt"):
    keys = Keys()
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, value = line.partition("=")
        if not sep or not name.strip():
            raise ValueError(f"line {n}: expected NAME=value")
        if value.strip():  # empty means unset
            keys[name.strip()] = value.strip()
    return keys

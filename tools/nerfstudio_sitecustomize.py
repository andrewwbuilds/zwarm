"""nerfstudio 1.1.5 predates Python 3.11's rule against unhashable dataclass defaults.
Make only nerfstudio's own dataclasses hashable so its config classes can load."""
import dataclasses

_original = dataclasses.dataclass


def dataclass(cls=None, /, **kwargs):
    def wrap(c):
        options = dict(kwargs)
        if c.__module__.startswith("nerfstudio") and "__hash__" not in c.__dict__:
            options.setdefault("unsafe_hash", True)
        return _original(c, **options)
    return wrap if cls is None else wrap(cls)


dataclasses.dataclass = dataclass

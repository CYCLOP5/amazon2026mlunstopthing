'tiny plug-in registry: components register under a name that configs refer to'


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items = {}

    def register(self, name: str):
        def deco(obj):
            if name in self._items:
                raise ValueError(f"{self.kind} {name!r} registered twice")
            self._items[name] = obj
            return obj
        return deco

    def get(self, name: str):
        if name not in self._items:
            raise KeyError(f"unknown {self.kind} {name!r}; available: {sorted(self._items)}")
        return self._items[name]

    def names(self):
        return sorted(self._items)

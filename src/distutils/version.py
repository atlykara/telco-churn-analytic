# src/distutils/version.py

import re
from functools import total_ordering


@total_ordering
class LooseVersion:
    """
    Basit bir LooseVersion implementasyonu.

    PySpark sadece sürümleri karşılaştırmak için kullanıyor.
    Burada stringi parçalara ayırıp (sayı / harf) liste halinde karşılaştırıyoruz.
    """

    def __init__(self, vstring=None):
        self.vstring = str(vstring) if vstring is not None else ""
        self._components = self._parse(self.vstring)

    def _parse(self, s: str):
        # Örn: "3.5.1" -> ["3", ".", "5", ".", "1"] -> [3, ".", 5, ".", 1]
        parts = re.split(r"(\d+)", s)
        comps = []
        for p in parts:
            if not p:
                continue
            if p.isdigit():
                comps.append(int(p))
            else:
                comps.append(p)
        return comps

    def __repr__(self):
        return f"LooseVersion('{self.vstring}')"

    def __eq__(self, other):
        if not isinstance(other, LooseVersion):
            other = LooseVersion(other)
        return self._components == other._components

    def __lt__(self, other):
        if not isinstance(other, LooseVersion):
            other = LooseVersion(other)
        return self._components < other._components

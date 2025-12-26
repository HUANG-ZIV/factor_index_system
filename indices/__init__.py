# -*- coding: utf-8 -*-
"""
指數定義模組
"""

from .base_index import BaseIndexConfig, BaseIndex
from .quality_index import QualityIndexConfig, QualityIndex
#from .value_index import ValueIndexConfig, ValueIndex
from .momentum_index import MomentumIndexConfig, MomentumIndex
from .size_index import SizeIndexConfig, SizeIndex
from .dividend_index import DividendIndexConfig, DividendIndex
from .low_vol_index import LowVolIndexConfig, LowVolIndex

__all__ = [
    'BaseIndexConfig', 'BaseIndex',
    'QualityIndexConfig', 'QualityIndex',
    'ValueIndexConfig', 'ValueIndex',
    'MomentumIndexConfig', 'MomentumIndex',
    'SizeIndexConfig', 'SizeIndex',
    'DividendIndexConfig', 'DividendIndex',
    'LowVolIndexConfig', 'LowVolIndex',
]
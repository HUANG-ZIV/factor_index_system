# -*- coding: utf-8 -*-
"""
價值因子指數
定義價值指數的設定與因子計算邏輯
"""

import pandas as pd
import numpy as np
from .base_index import BaseIndexConfig, BaseIndex


class ValueIndexConfig(BaseIndexConfig):
    """價值指數設定"""
    
    # ========== 指數基本資訊 ==========
    INDEX_NAME = "價值因子指數"
    INDEX_CODE = "VALUE"
    BASE_DATE = "2024-01-03"
    BASE_VALUE = 100
    
    # ========== 調倉時程 ==========
    REBALANCE_FREQ = "Q"                    # 季調倉
    REBALANCE_MONTHS = [3, 6, 9, 12]
    REVIEW_DAY = "last_business_day"
    EFFECTIVE_DAYS = 5
    
    # ========== 股票池篩選 ==========
    MARKET_CAP_FILTER = {
        "method": "top_n",
        "value": 300
    }
    
    LIQUIDITY_FILTER = {
        "method": "top_percent",
        "value": 0.90,
        "metric": "avg_daily_value"
    }
    
    EXCLUDE_SECTORS = None
    EXCLUDE_STOCKS = None
    
    # ========== 選股設定 ==========
    SELECTION_METHOD = "top_n"              # 選前 N 檔
    TOP_PERCENT = None
    TOP_N = 50                              # 前 50 檔
    
    # ========== 權重設定 ==========
    WEIGHTING_METHOD = "factor"             # 因子加權
    WEIGHT_CAP = 0.10                       # 單檔上限 10%
    WEIGHT_FLOOR = None
    MIXED_WEIGHT_ALPHA = 0.5
    
    # ========== 因子設定 ==========
    # 三個因子等權重，都是越低越好（direction = -1）
    FACTORS = {
        "pe_ratio": {
            "file": "本益比.csv",
            "weight": 1/3,
            "direction": -1               # 低本益比較好
        },
        "pb_ratio": {
            "file": "股價淨值比.csv",
            "weight": 1/3,
            "direction": -1               # 低股價淨值比較好
        },
        "leverage": {
            "file": "槓桿比率.csv",
            "weight": 1/3,
            "direction": -1               # 低槓桿較好
        }
    }
    
    # ========== 標準化設定 ==========
    STANDARDIZE_METHOD = "percentile"
    WINSORIZE = True
    WINSORIZE_LIMITS = (0.01, 0.99)


class ValueIndex(BaseIndex):
    """價值因子指數"""
    
    config = ValueIndexConfig
    
    def calc_factor_score(self, date):
        """
        計算價值因子綜合分數
        
        Args:
            date: 計算日期
            
        Returns:
            pd.Series: 股票代碼為索引，價值分數為值
        """
        return self._calc_composite_score(date)
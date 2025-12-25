# -*- coding: utf-8 -*-
"""
動能因子指數
定義動能指數的設定與因子計算邏輯
"""

import pandas as pd
import numpy as np
from .base_index import BaseIndexConfig, BaseIndex


class MomentumIndexConfig(BaseIndexConfig):
    """動能指數設定"""
    
    # ========== 指數基本資訊 ==========
    INDEX_NAME = "動能因子指數"
    INDEX_CODE = "MOMENTUM"
    BASE_DATE = "2005-01-03"
    BASE_VALUE = 100
    
    # ========== 調倉時程 ==========
    REBALANCE_FREQ = "M"
    REBALANCE_MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
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
    SELECTION_METHOD = "top_percent"
    TOP_PERCENT = 0.20
    TOP_N = None
    
    # ========== 權重設定 ==========
    WEIGHTING_METHOD = "equal"
    WEIGHT_CAP = 0.05
    WEIGHT_FLOOR = None
    MIXED_WEIGHT_ALPHA = 0.5
    
    # ========== 因子設定 ==========
    FACTORS = {
        "mom_12_1": {
            "file": None,               # 需計算：過去12個月報酬，排除近1個月
            "weight": 0.60,
            "direction": 1,
            "lookback": 252,            # 回溯交易日數
            "skip": 21                  # 排除近期交易日數
        },
        "mom_6": {
            "file": None,               # 需計算：過去6個月報酬
            "weight": 0.40,
            "direction": 1,
            "lookback": 126,
            "skip": 0
        }
    }
    
    # ========== 標準化設定 ==========
    STANDARDIZE_METHOD = "percentile"
    WINSORIZE = True
    WINSORIZE_LIMITS = (0.01, 0.99)


class MomentumIndex(BaseIndex):
    """動能因子指數"""
    
    config = MomentumIndexConfig
    
    def calc_factor_score(self, date):
        """
        計算動能因子綜合分數
        
        Args:
            date: 計算日期
            
        Returns:
            pd.Series: 股票代碼為索引，動能分數為值
        """
        pass
    
    def _calc_momentum(self, date, lookback, skip=0):
        """
        計算動能（過去N日報酬率）
        
        Args:
            date: 計算日期
            lookback: 回溯天數
            skip: 排除近期天數
            
        Returns:
            pd.Series: 股票代碼為索引，動能為值
        """
        pass

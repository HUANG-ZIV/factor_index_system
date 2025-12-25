# -*- coding: utf-8 -*-
"""
品質因子指數
定義品質指數的設定與因子計算邏輯
"""

import pandas as pd
import numpy as np
from .base_index import BaseIndexConfig, BaseIndex


class QualityIndexConfig(BaseIndexConfig):
    """品質指數設定"""
    
    # ========== 指數基本資訊 ==========
    INDEX_NAME = "品質因子指數"
    INDEX_CODE = "QUALITY"
    BASE_DATE = "2005-01-03"
    BASE_VALUE = 100
    
    # ========== 調倉時程 ==========
    REBALANCE_FREQ = "Q"                    # 季調倉
    REBALANCE_MONTHS = [3, 6, 9, 12]        # 調倉月份
    REVIEW_DAY = "last_business_day"        # 調整日：當月最後一個交易日
    EFFECTIVE_DAYS = 5                      # 5 個交易日後生效
    
    # ========== 股票池篩選 ==========
    MARKET_CAP_FILTER = {
        "method": "top_n",                  # 市值前 N 大
        "value": 300
    }
    
    LIQUIDITY_FILTER = {
        "method": "top_percent",            # 流動性前 N%
        "value": 0.90,
        "metric": "avg_daily_value"
    }
    
    EXCLUDE_SECTORS = None                  # 不排除產業
    EXCLUDE_STOCKS = None                   # 不排除特定股票
    
    # ========== 選股設定 ==========
    SELECTION_METHOD = "top_percent"        # 選前 N%
    TOP_PERCENT = 0.20                      # 前 20%
    TOP_N = None
    
    # ========== 權重設定 ==========
    WEIGHTING_METHOD = "market_cap"         # 市值加權
    WEIGHT_CAP = 0.10                       # 單檔上限 10%
    WEIGHT_FLOOR = None                     # 無下限
    MIXED_WEIGHT_ALPHA = 0.5
    
    # ========== 因子設定 ==========
    FACTORS = {
        "roe": {
            "file": "稅後權益報酬率.csv",
            "weight": 0.30,
            "direction": 1                  # 越高越好
        },
        "gross_margin": {
            "file": "毛利率.csv",
            "weight": 0.25,
            "direction": 1
        },
        "operating_margin": {
            "file": "營業利益率.csv",
            "weight": 0.25,
            "direction": 1
        },
        "debt_ratio": {
            "file": "負債比率.csv",
            "weight": 0.20,
            "direction": -1                 # 越低越好
        }
    }
    
    # ========== 標準化設定 ==========
    STANDARDIZE_METHOD = "percentile"       # 百分位排序
    WINSORIZE = True                        # 縮尾處理
    WINSORIZE_LIMITS = (0.01, 0.99)         # 1% ~ 99%


class QualityIndex(BaseIndex):
    """品質因子指數"""
    
    config = QualityIndexConfig
    
    def calc_factor_score(self, date):
        """
        計算品質因子綜合分數
        
        Args:
            date: 計算日期
            
        Returns:
            pd.Series: 股票代碼為索引，品質分數為值
        """
        # 使用基類的通用方法計算綜合分數
        return self._calc_composite_score(date)
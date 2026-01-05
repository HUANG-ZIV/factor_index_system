# -*- coding: utf-8 -*-
"""
規模因子指數（大型股）
定義規模指數的設定與因子計算邏輯
"""

import pandas as pd
import numpy as np
from .base_index import BaseIndexConfig, BaseIndex


class SizeIndexConfig(BaseIndexConfig):
    """規模指數設定（大型股）"""
    
    # ========== 指數基本資訊 ==========
    INDEX_NAME = "規模因子指數"
    INDEX_CODE = "SIZE"
    BASE_DATE = "2024-01-03"
    BASE_VALUE = 100
    
    # ========== 調倉時程 ==========
    REBALANCE_FREQ = "Q"
    REBALANCE_MONTHS = [3, 6, 9, 12]
    REVIEW_DAY = "last_business_day"
    EFFECTIVE_DAYS = 5
    
    # ========== 股票池篩選 ==========
    MARKET_CAP_FILTER = None
    
    LIQUIDITY_FILTER = {
        "method": "top_percent",
        "value": 0.90,
        "metric": "avg_daily_value"
    }
    
    EXCLUDE_SECTORS = None
    EXCLUDE_STOCKS = None
    
    # ========== 選股設定 ==========
    SELECTION_METHOD = "top_n"
    TOP_PERCENT = None
    TOP_N = 3  # 或您要的數量
    
    # ========== 權重設定 ==========
    WEIGHTING_METHOD = "market_cap"
    WEIGHT_CAP = None
    WEIGHT_FLOOR = None
    MIXED_WEIGHT_ALPHA = 0.5


class SizeIndex(BaseIndex):
    """規模因子指數（大型股）"""
    
    config = SizeIndexConfig
    
    def calc_factor_score(self, date):
        """
        計算規模因子分數（直接用市值排序）
        
        Args:
            date: 計算日期
            
        Returns:
            pd.Series: 股票代碼為索引，市值為值
        """
        date = pd.to_datetime(date)
        
        # 取得市值
        market_cap = self.data_manager.get_market_cap(date)
        
        if market_cap is None or len(market_cap) == 0:
            return pd.Series(dtype=float)
        
        # 移除無效值
        market_cap = market_cap.dropna()
        market_cap = market_cap[market_cap > 0]
        
        # 直接返回市值（市值越大分數越高）
        return market_cap
    
    def calc_weights(self, stocks, date, factor_scores=None):
        """
        計算權重（市值加權）
        """
        if self.config.WEIGHTING_METHOD != "market_cap":
            return super().calc_weights(stocks, date, factor_scores)
        
        market_cap = self.data_manager.get_market_cap(date)
        market_cap = market_cap.reindex(stocks).dropna()
        
        if len(market_cap) == 0:
            return self._calc_equal_weights(stocks)
        
        weights = market_cap / market_cap.sum()
        weights = self._apply_weight_constraints(weights)
        
        return weights
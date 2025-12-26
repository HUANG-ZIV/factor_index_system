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
    REBALANCE_FREQ = "Q"                    # 季調倉
    REBALANCE_MONTHS = [3, 6, 9, 12]
    REVIEW_DAY = "last_business_day"
    EFFECTIVE_DAYS = 5
    
    # ========== 股票池篩選 ==========
    # 不預先篩選，直接從全市場選市值最大的股票
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
    TOP_N = 50                              # 市值前 50 大
    
    # ========== 權重設定 ==========
    WEIGHTING_METHOD = "market_cap"         # 市值加權
    WEIGHT_CAP = 0.30                       # 單一股票上限 30%
    WEIGHT_FLOOR = None
    MIXED_WEIGHT_ALPHA = 0.5
    
    # ========== 因子設定 ==========
    FACTORS = {}
    
    # ========== 標準化設定 ==========
    STANDARDIZE_METHOD = "percentile"
    WINSORIZE = True
    WINSORIZE_LIMITS = (0.01, 0.99)


class SizeIndex(BaseIndex):
    """規模因子指數（大型股）"""
    
    config = SizeIndexConfig
    
    def calc_factor_score(self, date):
        """
        計算規模因子分數（市值越大分數越高）
        
        Args:
            date: 計算日期
            
        Returns:
            pd.Series: 股票代碼為索引，規模分數為值
        """
        date = pd.to_datetime(date)
        
        # 取得市值
        market_cap = self.data_manager.get_market_cap(date)
        
        if market_cap is None or len(market_cap) == 0:
            return pd.Series(dtype=float)
        
        # 移除無效值
        market_cap = market_cap.dropna()
        market_cap = market_cap[market_cap > 0]
        
        if len(market_cap) == 0:
            return pd.Series(dtype=float)
        
        # 標準化（市值越大分數越高）
        standardized = self._standardize(market_cap)
        
        return standardized
    
    def calc_weights(self, stocks, date, factor_scores=None):
        """
        計算權重（市值加權）
        
        Args:
            stocks: 成分股列表
            date: 日期
            factor_scores: 因子分數（未使用）
            
        Returns:
            pd.Series: 股票代碼為索引，權重為值
        """
        if self.config.WEIGHTING_METHOD != "market_cap":
            return super().calc_weights(stocks, date, factor_scores)
        
        # 取得市值
        market_cap = self.data_manager.get_market_cap(date)
        market_cap = market_cap.reindex(stocks).dropna()
        
        if len(market_cap) == 0:
            return self._calc_equal_weights(stocks)
        
        # 計算市值權重
        weights = market_cap / market_cap.sum()
        
        # 套用權重上下限
        weights = self._apply_weight_constraints(weights)
        
        return weights
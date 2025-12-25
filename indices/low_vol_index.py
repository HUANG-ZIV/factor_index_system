# -*- coding: utf-8 -*-
"""
低波動因子指數
定義低波動指數的設定與因子計算邏輯
"""

import pandas as pd
import numpy as np
from .base_index import BaseIndexConfig, BaseIndex


class LowVolIndexConfig(BaseIndexConfig):
    """低波動指數設定"""
    
    # ========== 指數基本資訊 ==========
    INDEX_NAME = "低波動因子指數"
    INDEX_CODE = "LOWVOL"
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
    SELECTION_METHOD = "top_n"       # "top_n", "top_percent"
    TOP_PERCENT = None
    TOP_N = 50
    
    # ========== 權重設定 ==========
    WEIGHTING_METHOD = "factor"             # 依波動率倒數加權（低波動高權重）
    WEIGHT_CAP = 0.08
    WEIGHT_FLOOR = None
    MIXED_WEIGHT_ALPHA = 0.5
    
    # ========== 波動率設定 ==========
    VOLATILITY_LOOKBACK = 252               # 回溯交易日數（約一年）
    MIN_TRADING_DAYS = 120                  # 最少需要的交易日數
    
    # ========== 因子設定 ==========
    FACTORS = {}
    
    # ========== 標準化設定 ==========
    STANDARDIZE_METHOD = "percentile"
    WINSORIZE = True
    WINSORIZE_LIMITS = (0.01, 0.99)


class LowVolIndex(BaseIndex):
    """低波動因子指數"""
    
    config = LowVolIndexConfig
    
    def calc_factor_score(self, date):
        """
        計算低波動因子分數
        
        Args:
            date: 計算日期
            
        Returns:
            pd.Series: 股票代碼為索引，低波動分數為值
        """
        date = pd.to_datetime(date)
        
        # 計算波動率
        volatility = self._calc_volatility(date)
        
        if len(volatility) == 0:
            return pd.Series(dtype=float)
        
        # 標準化
        standardized = self._standardize(volatility)
        
        # 反轉（低波動得高分）
        score = 1 - standardized
        
        return score
    
    def _calc_volatility(self, date):
        """
        計算波動率（過去 N 日報酬標準差，年化）
        
        Args:
            date: 計算日期
            
        Returns:
            pd.Series: 股票代碼為索引，波動率為值
        """
        date = pd.to_datetime(date)
        lookback = self.config.VOLATILITY_LOOKBACK
        min_days = self.config.MIN_TRADING_DAYS
        
        # 取得交易日
        trading_dates = self.data_manager.trading_dates
        
        try:
            end_idx = trading_dates.get_loc(date)
        except KeyError:
            return pd.Series(dtype=float)
        
        start_idx = max(0, end_idx - lookback + 1)
        
        if end_idx - start_idx < min_days:
            return pd.Series(dtype=float)
        
        start_date = trading_dates[start_idx]
        end_date = trading_dates[end_idx]
        
        # 取得報酬率序列
        returns = self.data_manager.get_price_return_series(
            start_date, end_date, use_adjusted=True
        )
        
        if len(returns) == 0:
            return pd.Series(dtype=float)
        
        # 計算標準差
        volatility = returns.std()
        
        # 年化（假設 252 個交易日）
        volatility = volatility * np.sqrt(252)
        
        # 移除無效值
        volatility = volatility.dropna()
        volatility = volatility[volatility > 0]
        
        return volatility
    
    def calc_weights(self, stocks, date, factor_scores=None):
        """
        覆寫權重計算（低波動使用波動率倒數加權）
        
        Args:
            stocks: 成分股列表
            date: 日期
            factor_scores: 因子分數
            
        Returns:
            pd.Series: 股票代碼為索引，權重為值
        """
        if self.config.WEIGHTING_METHOD != "factor":
            return super().calc_weights(stocks, date, factor_scores)
        
        # 取得波動率
        volatility = self._calc_volatility(date)
        volatility = volatility.reindex(stocks).dropna()
        
        if len(volatility) == 0:
            return self._calc_equal_weights(stocks)
        
        # 波動率倒數作為權重（低波動高權重）
        inv_volatility = 1 / volatility
        
        # 正規化
        weights = inv_volatility / inv_volatility.sum()
        
        # 套用權重上下限
        weights = self._apply_weight_constraints(weights)
        
        return weights
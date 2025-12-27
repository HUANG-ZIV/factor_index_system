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
    BASE_DATE = "2024-01-03"
    BASE_VALUE = 100
    
    # ========== 調倉時程 ==========
    REBALANCE_FREQ = "M"                    # 月調倉
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
    SELECTION_METHOD = "top_n"              # 選前 N 檔
    TOP_PERCENT = None
    TOP_N = 50                              # 前 50 檔
    
    # ========== 權重設定 ==========
    WEIGHTING_METHOD = "factor"             # 因子加權
    WEIGHT_CAP = 0.10                       # 單檔上限 10%
    WEIGHT_FLOOR = None
    MIXED_WEIGHT_ALPHA = 0.5
    
    # ========== 動能設定 ==========
    MOMENTUM_LOOKBACK = 20                  # 過去 20 個交易日
    MOMENTUM_SKIP = 0                       # 不排除近期
    
    # ========== 因子設定（動能不使用檔案，自行計算）==========
    FACTORS = {}
    
    # ========== 標準化設定 ==========
    STANDARDIZE_METHOD = "percentile"
    WINSORIZE = True
    WINSORIZE_LIMITS = (0.01, 0.99)


class MomentumIndex(BaseIndex):
    """動能因子指數"""
    
    config = MomentumIndexConfig
    
    def calc_factor_score(self, date):
        """
        計算動能因子分數
        
        Args:
            date: 計算日期
            
        Returns:
            pd.Series: 股票代碼為索引，動能分數為值
        """
        date = pd.to_datetime(date)
        
        # 計算動能
        momentum = self._calc_momentum(
            date, 
            lookback=self.config.MOMENTUM_LOOKBACK,
            skip=self.config.MOMENTUM_SKIP
        )
        
        if len(momentum) == 0:
            return pd.Series(dtype=float)
        
        # 標準化（動能越高分數越高）
        standardized = self._standardize(momentum)
        
        return standardized
    
    def _calc_momentum(self, date, lookback, skip=0):
        """
        計算動能（過去 N 日報酬率，可排除近 M 日）
        
        Args:
            date: 計算日期
            lookback: 回溯天數
            skip: 排除近期天數
            
        Returns:
            pd.Series: 股票代碼為索引，動能為值
        """
        date = pd.to_datetime(date)
        
        # 取得日期索引
        trading_dates = self.data_manager.trading_dates
        
        try:
            end_idx = trading_dates.get_loc(date)
        except KeyError:
            return pd.Series(dtype=float)
        
        # 計算區間
        # 排除近 skip 天
        momentum_end_idx = end_idx - skip
        momentum_start_idx = momentum_end_idx - lookback
        
        if momentum_start_idx < 0 or momentum_end_idx < 0:
            return pd.Series(dtype=float)
        
        start_date = trading_dates[momentum_start_idx]
        end_date = trading_dates[momentum_end_idx]
        
        # 取得價格（使用還原價）
        price_df = self.data_manager.adjusted_price_df
        
        if price_df is None:
            return pd.Series(dtype=float)
        
        try:
            start_price = price_df.loc[start_date]
            end_price = price_df.loc[end_date]
        except KeyError:
            return pd.Series(dtype=float)
        
        # 計算報酬率
        momentum = (end_price - start_price) / start_price
        momentum = momentum.dropna()
        
        return momentum
# -*- coding: utf-8 -*-
"""
股利因子指數
定義股利指數的設定與因子計算邏輯
"""

import pandas as pd
import numpy as np
from .base_index import BaseIndexConfig, BaseIndex


class DividendIndexConfig(BaseIndexConfig):
    """股利指數設定"""
    
    # ========== 指數基本資訊 ==========
    INDEX_NAME = "股利因子指數"
    INDEX_CODE = "DIVIDEND"
    BASE_DATE = "2005-01-03"
    BASE_VALUE = 100
    
    # ========== 調倉時程 ==========
    REBALANCE_FREQ = "A"                    # 年調倉
    REBALANCE_MONTHS = [6]                  # 6 月調倉（股利公告後）
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
    WEIGHTING_METHOD = "factor"             # 依殖利率加權
    WEIGHT_CAP = 0.08
    WEIGHT_FLOOR = None
    MIXED_WEIGHT_ALPHA = 0.5
    
    # ========== 股利設定 ==========
    DIVIDEND_FILE = "現金股利合計.csv"
    DIVIDEND_LOOKBACK_DAYS = 365            # 回溯一年的股利
    
    # ========== 因子設定 ==========
    FACTORS = {}
    
    # ========== 標準化設定 ==========
    STANDARDIZE_METHOD = "percentile"
    WINSORIZE = True
    WINSORIZE_LIMITS = (0.01, 0.99)


class DividendIndex(BaseIndex):
    """股利因子指數"""
    
    config = DividendIndexConfig
    
    def calc_factor_score(self, date):
        """
        計算股利因子分數（殖利率）
        
        Args:
            date: 計算日期
            
        Returns:
            pd.Series: 股票代碼為索引，殖利率為值
        """
        date = pd.to_datetime(date)
        
        # 計算殖利率
        dividend_yield = self._calc_dividend_yield(date)
        
        if len(dividend_yield) == 0:
            return pd.Series(dtype=float)
        
        # 標準化
        standardized = self._standardize(dividend_yield)
        
        return standardized
    
    def _calc_dividend_yield(self, date):
        """
        計算殖利率（過去一年股利 / 當前股價）
        
        Args:
            date: 計算日期
            
        Returns:
            pd.Series: 股票代碼為索引，殖利率為值
        """
        date = pd.to_datetime(date)
        
        # 回溯期間
        lookback = self.config.DIVIDEND_LOOKBACK_DAYS
        start_date = date - pd.Timedelta(days=lookback)
        
        # 取得股利資料
        dividend_df = self.data_manager.dividend_df
        
        if dividend_df is None:
            return pd.Series(dtype=float)
        
        # 篩選期間內的股利
        mask = (dividend_df.index >= start_date) & (dividend_df.index <= date)
        period_dividends = dividend_df.loc[mask]
        
        # 加總每檔股票的股利
        total_dividends = period_dividends.sum()
        
        # 取得當前股價
        current_price = self.data_manager.get_close_price(date)
        
        # 計算殖利率
        dividend_yield = total_dividends / current_price
        dividend_yield = dividend_yield.replace([np.inf, -np.inf], np.nan)
        dividend_yield = dividend_yield.dropna()
        
        # 排除殖利率為 0 或負值的股票
        dividend_yield = dividend_yield[dividend_yield > 0]
        
        return dividend_yield
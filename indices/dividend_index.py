# -*- coding: utf-8 -*-
"""
高股息因子指數
定義高股息指數的設定與因子計算邏輯
"""

import pandas as pd
import numpy as np
from .base_index import BaseIndexConfig, BaseIndex


class DividendIndexConfig(BaseIndexConfig):
    """高股息指數設定"""
    
    # ========== 指數基本資訊 ==========
    INDEX_NAME = "高股息因子指數"
    INDEX_CODE = "DIVIDEND"
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
    
    # ========== 殖利率設定 ==========
    DIVIDEND_LOOKBACK_DAYS = 252            # 回溯一年（約 252 個交易日）
    
    # ========== 因子設定（殖利率自行計算，不使用檔案）==========
    FACTORS = {}
    
    # ========== 標準化設定 ==========
    STANDARDIZE_METHOD = "percentile"
    WINSORIZE = True
    WINSORIZE_LIMITS = (0.01, 0.99)


class DividendIndex(BaseIndex):
    """高股息因子指數"""
    
    config = DividendIndexConfig
    
    def calc_factor_score(self, date):
        """
        計算殖利率因子分數
        
        Args:
            date: 計算日期
            
        Returns:
            pd.Series: 股票代碼為索引，殖利率分數為值
        """
        date = pd.to_datetime(date)
        
        # 計算殖利率
        dividend_yield = self._calc_dividend_yield(
            date, 
            lookback_days=self.config.DIVIDEND_LOOKBACK_DAYS
        )
        
        if len(dividend_yield) == 0:
            return pd.Series(dtype=float)
        
        # 標準化（殖利率越高分數越高）
        standardized = self._standardize(dividend_yield)
        
        return standardized
    
    def _calc_dividend_yield(self, date, lookback_days=252):
        """
        計算殖利率
        
        殖利率 = Σ (除息金額 ÷ 除息前一日股價)
        
        Args:
            date: 計算日期
            lookback_days: 回溯天數（預設 252 個交易日，約一年）
            
        Returns:
            pd.Series: 股票代碼為索引，殖利率為值
        """
        date = pd.to_datetime(date)
        
        # 取得日期索引
        trading_dates = self.data_manager.trading_dates
        
        try:
            end_idx = trading_dates.get_loc(date)
        except KeyError:
            return pd.Series(dtype=float)
        
        # 計算回溯起始日
        start_idx = max(0, end_idx - lookback_days)
        start_date = trading_dates[start_idx]
        
        # 取得股利資料
        dividend_df = self.data_manager.dividend_df
        if dividend_df is None:
            return pd.Series(dtype=float)
        
        # 取得股價資料
        price_df = self.data_manager.close_price_df
        if price_df is None:
            return pd.Series(dtype=float)
        
        # 篩選回溯期間的股利資料
        mask = (dividend_df.index >= start_date) & (dividend_df.index <= date)
        dividend_period = dividend_df.loc[mask]
        
        if len(dividend_period) == 0:
            return pd.Series(dtype=float)
        
        # 計算每檔股票的殖利率
        all_stocks = dividend_df.columns
        dividend_yields = pd.Series(0.0, index=all_stocks)
        
        for ex_date in dividend_period.index:
            # 取得除息前一日
            try:
                ex_idx = trading_dates.get_loc(ex_date)
                if ex_idx == 0:
                    continue
                prev_date = trading_dates[ex_idx - 1]
            except KeyError:
                continue
            
            # 取得除息金額
            dividends = dividend_period.loc[ex_date]
            
            # 取得前一日股價
            try:
                prev_prices = price_df.loc[prev_date]
            except KeyError:
                continue
            
            # 計算單次殖利率
            valid_mask = (dividends > 0) & (prev_prices > 0)
            single_yield = dividends / prev_prices
            single_yield = single_yield.where(valid_mask, 0)
            
            # 累加
            dividend_yields = dividend_yields.add(single_yield, fill_value=0)
        
        # 只保留有殖利率的股票
        dividend_yields = dividend_yields[dividend_yields > 0]
        
        return dividend_yields
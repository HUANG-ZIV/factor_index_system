# -*- coding: utf-8 -*-
"""
自訂指數
直接指定成份股與權重
"""

import pandas as pd
import numpy as np
import os
from .base_index import BaseIndexConfig, BaseIndex


class CustomIndexConfig(BaseIndexConfig):
    """自訂指數設定"""
    
    # ========== 指數基本資訊 ==========
    INDEX_NAME = "自訂指數"
    INDEX_CODE = "CUSTOM"
    BASE_DATE = "2024-01-03"
    BASE_VALUE = 100
    
    # ========== 調倉時程 ==========
    REBALANCE_FREQ = "Q"
    REBALANCE_MONTHS = [3, 6, 9, 12]
    REVIEW_DAY = "last_business_day"
    EFFECTIVE_DAYS = 5
    
    # ========== 自訂成份股檔案 ==========
    CONSTITUENTS_FILE = "custom_constituents.csv"
    
    # ========== 停用篩選（由檔案指定）==========
    MARKET_CAP_FILTER = None
    LIQUIDITY_FILTER = None
    EXCLUDE_SECTORS = None
    EXCLUDE_STOCKS = None
    
    # ========== 停用選股設定（由檔案指定）==========
    SELECTION_METHOD = None
    TOP_PERCENT = None
    TOP_N = None
    
    # ========== 停用權重設定（由檔案指定）==========
    WEIGHTING_METHOD = "custom"
    WEIGHT_CAP = None
    WEIGHT_FLOOR = None


class CustomIndex(BaseIndex):
    """自訂指數 - 從檔案讀取成份股與權重"""
    
    config = CustomIndexConfig
    
    def __init__(self, data_manager):
        """
        初始化自訂指數
        
        Args:
            data_manager: DataManager 實例
        """
        super().__init__(data_manager)
        self.constituents_df = self._load_constituents()
    
    def _load_constituents(self):
        """
        載入成份股檔案
        
        檔案格式：
        date,stock_id,weight
        2024-01-03,2330,0.50
        2024-01-03,2317,0.30
        2024-01-03,2454,0.20
        
        Returns:
            pd.DataFrame: 成份股資料
        """
        filepath = os.path.join(
            self.data_manager.data_root, 
            self.config.CONSTITUENTS_FILE
        )
        
        if not os.path.exists(filepath):
            self.logger.error(f"成份股檔案不存在: {filepath}")
            return None
        
        try:
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            df['stock_id'] = df['stock_id'].astype(str)
            df['weight'] = df['weight'].astype(float)
            
            # 驗證每個日期的權重總和
            for date in df['date'].unique():
                date_weights = df[df['date'] == date]['weight'].sum()
                if abs(date_weights - 1.0) > 0.001:
                    self.logger.warning(
                        f"{date.strftime('%Y-%m-%d')}: 權重總和 = {date_weights:.4f}，將自動標準化"
                    )
            
            self.logger.info(
                f"載入成份股: {len(df['date'].unique())} 個調倉日，"
                f"共 {len(df)} 筆資料"
            )
            return df
            
        except Exception as e:
            self.logger.error(f"載入成份股檔案失敗: {e}")
            return None
    
    def _get_constituents_for_date(self, date):
        """
        取得指定日期的成份股與權重
        
        會自動找該日期或之前最近一期的資料
        
        Args:
            date: 查詢日期
            
        Returns:
            pd.DataFrame: 該期的成份股資料，或 None
        """
        if self.constituents_df is None:
            return None
        
        date = pd.to_datetime(date)
        
        # 找該日期或之前最近的資料
        available_dates = self.constituents_df['date'].unique()
        valid_dates = available_dates[available_dates <= date]
        
        if len(valid_dates) == 0:
            self.logger.warning(f"{date.strftime('%Y-%m-%d')}: 無可用的成份股資料")
            return None
        
        latest_date = valid_dates.max()
        
        # 篩選該日期的成份股
        mask = self.constituents_df['date'] == latest_date
        data = self.constituents_df[mask].copy()
        
        return data
    
    def filter_stock_pool(self, date):
        """
        覆寫：直接返回指定的成份股作為股票池
        
        Args:
            date: 日期
            
        Returns:
            list: 成份股代碼列表
        """
        data = self._get_constituents_for_date(date)
        
        if data is None:
            return []
        
        return list(data['stock_id'])
    
    def calc_factor_score(self, date):
        """
        覆寫：返回權重作為分數
        
        用權重當分數可確保排序後順序正確
        
        Args:
            date: 計算日期
            
        Returns:
            pd.Series: 股票代碼為索引，權重為值
        """
        data = self._get_constituents_for_date(date)
        
        if data is None:
            return pd.Series(dtype=float)
        
        scores = pd.Series(
            data['weight'].values,
            index=data['stock_id'].values
        )
        
        return scores
    
    def select_stocks(self, date):
        """
        覆寫：直接返回指定的成份股
        
        Args:
            date: 選股日期
            
        Returns:
            tuple: (成份股列表, 權重 Series, 空的落選列表)
        """
        data = self._get_constituents_for_date(date)
        
        if data is None:
            self.logger.warning(f"{date}: 無成份股資料")
            return [], pd.Series(dtype=float), []
        
        stocks = list(data['stock_id'])
        scores = pd.Series(
            data['weight'].values, 
            index=data['stock_id'].values
        )
        
        self.logger.info(f"{date}: 載入 {len(stocks)} 檔成份股")
        
        return stocks, scores, []
    
    def calc_weights(self, stocks, date, factor_scores=None):
        """
        覆寫：直接返回指定的權重
        
        Args:
            stocks: 成份股列表
            date: 日期
            factor_scores: 因子分數（未使用）
            
        Returns:
            pd.Series: 股票代碼為索引，權重為值
        """
        data = self._get_constituents_for_date(date)
        
        if data is None:
            self.logger.warning(f"{date}: 無權重資料，使用等權重")
            return self._calc_equal_weights(stocks)
        
        # 建立權重 Series
        weights = pd.Series(
            data['weight'].values,
            index=data['stock_id'].values
        )
        
        # 只保留指定的股票
        weights = weights.reindex(stocks)
        
        # 確保權重總和為 1
        total = weights.sum()
        if total > 0 and abs(total - 1.0) > 0.0001:
            weights = weights / total
        
        return weights
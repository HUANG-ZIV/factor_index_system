# -*- coding: utf-8 -*-
"""
指數基類
定義所有因子指數共用的設定結構與計算邏輯
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import logging
from config import *


class BaseIndexConfig:
    """指數設定基類 - 定義所有可設定參數的預設值"""
    
    # ========== 指數基本資訊 ==========
    INDEX_NAME = "Base Index"
    INDEX_CODE = "BASE"
    BASE_DATE = "2005-01-03"
    BASE_VALUE = 100
    
    # ========== 調倉時程 ==========
    REBALANCE_FREQ = "Q"                    # M=月, Q=季, SA=半年, A=年
    REBALANCE_MONTHS = [3, 6, 9, 12]        # 調倉月份
    REVIEW_DAY = "last_business_day"        # 調整日定義
    EFFECTIVE_DAYS = 5                      # 0=最快生效(T+1), N=T+N生效
    
    # ========== 股票池篩選 ==========
    MARKET_CAP_FILTER = {
        "method": "top_n",                  # top_n / top_percent / min_value / None
        "value": 300
    }
    
    LIQUIDITY_FILTER = {
        "method": "top_percent",            # top_n / top_percent / min_value / None
        "value": 0.90,
        "metric": "avg_daily_value"         # avg_volume / avg_daily_value
    }
    
    EXCLUDE_SECTORS = None                  # None = 不排除
    EXCLUDE_STOCKS = None                   # None = 不排除
    
    # ========== 選股設定 ==========
    SELECTION_METHOD = "top_percent"        # top_percent / top_n
    TOP_PERCENT = 0.20                      # 前 20%
    TOP_N = None                            # 或固定 N 檔
    
    # ========== 權重設定 ==========
    WEIGHTING_METHOD = "market_cap"         # equal / market_cap / factor / mixed
    WEIGHT_CAP = 0.10                       # 單檔上限，None = 不限
    WEIGHT_FLOOR = None                     # 單檔下限，None = 不限
    MIXED_WEIGHT_ALPHA = 0.5                # mixed 時，市值權重佔比
    
    # ========== 因子設定 ==========
    FACTORS = {}                            # 子類需覆寫
    
    # ========== 標準化設定 ==========
    STANDARDIZE_METHOD = "percentile"       # percentile / zscore
    WINSORIZE = True                        # 是否縮尾處理
    WINSORIZE_LIMITS = (0.01, 0.99)         # 縮尾範圍


class BaseIndex(ABC):
    """指數基類 - 定義所有因子指數共用的計算邏輯"""
    
    config = BaseIndexConfig
    
    def __init__(self, data_manager):
        """
        初始化指數
        
        Args:
            data_manager: DataManager 實例
        """
        self.data_manager = data_manager
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """設定日誌"""
        logger = logging.getLogger(self.config.INDEX_CODE)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(LOG_FORMAT)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(LOG_LEVEL)
        return logger
    
    # ========== 子類必須實作的方法 ==========
    
    @abstractmethod
    def calc_factor_score(self, date):
        """
        計算因子分數（子類必須實作）
        
        Args:
            date: 計算日期
            
        Returns:
            pd.Series: 股票代碼為索引，因子分數為值
        """
        pass
    
    # ========== 標準化方法 ==========
    
    def _standardize(self, series, method=None):
        """
        標準化因子值
        
        Args:
            series: 原始因子值
            method: 標準化方法（percentile / zscore），None 則使用設定值
            
        Returns:
            pd.Series: 標準化後的值
        """
        if method is None:
            method = self.config.STANDARDIZE_METHOD
        
        # 移除空值
        valid_series = series.dropna()
        
        if len(valid_series) == 0:
            return series
        
        # 縮尾處理
        if self.config.WINSORIZE:
            valid_series = self._winsorize(valid_series)
        
        if method == "percentile":
            # 百分位排序：排名 / 總數
            ranks = valid_series.rank(method='average')
            result = ranks / len(valid_series)
        
        elif method == "zscore":
            # Z-score：(值 - 平均) / 標準差
            mean = valid_series.mean()
            std = valid_series.std()
            if std == 0 or pd.isna(std):
                result = pd.Series(0.5, index=valid_series.index)
            else:
                result = (valid_series - mean) / std
        
        else:
            raise ValueError(f"未知的標準化方法: {method}")
        
        return result
    
    def _winsorize(self, series, limits=None):
        """
        縮尾處理
        
        Args:
            series: 原始值
            limits: 縮尾範圍，None 則使用設定值
            
        Returns:
            pd.Series: 縮尾後的值
        """
        if limits is None:
            limits = self.config.WINSORIZE_LIMITS
        
        lower, upper = limits
        lower_bound = series.quantile(lower)
        upper_bound = series.quantile(upper)
        
        return series.clip(lower=lower_bound, upper=upper_bound)
    
    # ========== 股票池篩選方法 ==========
    
    def filter_stock_pool(self, date):
        """
        篩選股票池
        
        Args:
            date: 篩選日期
            
        Returns:
            list: 通過篩選的股票代碼列表
        """
        # 取得所有股票
        all_stocks = self.data_manager.get_all_stocks()
        valid_stocks = set(all_stocks)
        
        self.logger.debug(f"初始股票數: {len(valid_stocks)}")
        
        # 市值篩選
        valid_stocks = self._filter_by_market_cap(valid_stocks, date)
        
        # 流動性篩選
        valid_stocks = self._filter_by_liquidity(valid_stocks, date)
        
        # 排除產業
        if self.config.EXCLUDE_SECTORS:
            sectors = self.data_manager.get_sectors()
            if len(sectors) > 0:
                excluded = set(sectors[sectors.isin(self.config.EXCLUDE_SECTORS)].index)
                valid_stocks = valid_stocks - excluded
                self.logger.debug(f"排除產業後: {len(valid_stocks)}")
        
        # 排除特定股票
        if self.config.EXCLUDE_STOCKS:
            exclude_set = set(str(s) for s in self.config.EXCLUDE_STOCKS)
            valid_stocks = valid_stocks - exclude_set
            self.logger.debug(f"排除特定股票後: {len(valid_stocks)}")
        
        return list(valid_stocks)
    
    def _filter_by_market_cap(self, stocks, date):
        """
        依市值篩選
        
        Args:
            stocks: 股票集合
            date: 日期
            
        Returns:
            set: 通過篩選的股票集合
        """
        config = self.config.MARKET_CAP_FILTER
        
        if not config or not config.get("method"):
            return stocks
        
        market_cap = self.data_manager.get_market_cap(date)
        market_cap = market_cap.dropna().sort_values(ascending=False)
        
        method = config["method"]
        value = config["value"]
        
        if method == "top_n":
            passed = set(market_cap.head(int(value)).index)
        elif method == "top_percent":
            n = max(1, int(len(market_cap) * value))
            passed = set(market_cap.head(n).index)
        elif method == "min_value":
            passed = set(market_cap[market_cap >= value].index)
        else:
            return stocks
        
        result = stocks.intersection(passed)
        self.logger.debug(f"市值篩選後: {len(result)}")
        return result
    
    def _filter_by_liquidity(self, stocks, date):
        """
        依流動性篩選
        
        Args:
            stocks: 股票集合
            date: 日期
            
        Returns:
            set: 通過篩選的股票集合
        """
        config = self.config.LIQUIDITY_FILTER
        
        if not config or not config.get("method"):
            return stocks
        
        metric = config.get("metric", "avg_daily_value")
        liquidity = self.data_manager.get_liquidity(date, metric=metric)
        liquidity = liquidity.dropna().sort_values(ascending=False)
        
        method = config["method"]
        value = config["value"]
        
        if method == "top_n":
            passed = set(liquidity.head(int(value)).index)
        elif method == "top_percent":
            n = max(1, int(len(liquidity) * value))
            passed = set(liquidity.head(n).index)
        elif method == "min_value":
            passed = set(liquidity[liquidity >= value].index)
        else:
            return stocks
        
        result = stocks.intersection(passed)
        self.logger.debug(f"流動性篩選後: {len(result)}")
        return result
    
    # ========== 選股方法 ==========
    
    def select_stocks(self, date):
        """
        根據因子分數選股
        
        Args:
            date: 選股日期
            
        Returns:
            tuple: (入選股票列表, 因子分數 Series, 落選資訊 list)
        """
        # 篩選股票池
        stock_pool = self.filter_stock_pool(date)
        
        if len(stock_pool) == 0:
            self.logger.warning(f"{date}: 股票池為空")
            return [], pd.Series(dtype=float), []
        
        # 計算因子分數
        factor_scores = self.calc_factor_score(date)
        
        # 只保留股票池內的股票
        factor_scores = factor_scores[factor_scores.index.isin(stock_pool)]
        factor_scores = factor_scores.dropna()
        
        if len(factor_scores) == 0:
            self.logger.warning(f"{date}: 無有效因子分數")
            return [], pd.Series(dtype=float), []
        
        # 排序（分數高的在前）
        factor_scores = factor_scores.sort_values(ascending=False)
        
        # 選股
        if self.config.SELECTION_METHOD == "top_percent":
            n = max(1, int(len(factor_scores) * self.config.TOP_PERCENT))
        elif self.config.SELECTION_METHOD == "top_n":
            n = min(self.config.TOP_N or len(factor_scores), len(factor_scores))
        else:
            n = len(factor_scores)
        
        selected_stocks = list(factor_scores.head(n).index)
        selected_scores = factor_scores.loc[selected_stocks]
        
        # 落選資訊
        rejected_stocks = list(factor_scores.iloc[n:].index)
        rejected_info = []
        for stock in rejected_stocks:
            rejected_info.append({
                "stock_id": stock,
                "score": factor_scores[stock],
                "rank": list(factor_scores.index).index(stock) + 1,
                "reason": "排名未達標準"
            })
        
        self.logger.info(f"{date}: 選出 {len(selected_stocks)} 檔股票")
        
        return selected_stocks, selected_scores, rejected_info
    
    # ========== 權重計算方法 ==========
    
    def calc_weights(self, stocks, date, factor_scores=None):
        """
        計算權重
        
        Args:
            stocks: 成分股列表
            date: 日期
            factor_scores: 因子分數，factor 加權時使用
            
        Returns:
            pd.Series: 股票代碼為索引，權重為值
        """
        if len(stocks) == 0:
            return pd.Series(dtype=float)
        
        method = self.config.WEIGHTING_METHOD
        
        if method == "equal":
            weights = self._calc_equal_weights(stocks)
        elif method == "market_cap":
            weights = self._calc_market_cap_weights(stocks, date)
        elif method == "factor":
            weights = self._calc_factor_weights(stocks, factor_scores)
        elif method == "mixed":
            weights = self._calc_mixed_weights(stocks, date, factor_scores)
        else:
            self.logger.warning(f"未知的權重方法: {method}，使用等權重")
            weights = self._calc_equal_weights(stocks)
        
        # 套用權重上下限
        weights = self._apply_weight_constraints(weights)
        
        return weights
    
    def _calc_equal_weights(self, stocks):
        """計算等權重"""
        n = len(stocks)
        weight = 1.0 / n
        return pd.Series(weight, index=stocks)
    
    def _calc_market_cap_weights(self, stocks, date):
        """計算市值加權"""
        market_cap = self.data_manager.get_market_cap(date, stocks)
        market_cap = market_cap.dropna()
        
        if market_cap.sum() == 0:
            return self._calc_equal_weights(stocks)
        
        weights = market_cap / market_cap.sum()
        return weights
    
    def _calc_factor_weights(self, stocks, factor_scores):
        """計算因子加權"""
        if factor_scores is None or len(factor_scores) == 0:
            return self._calc_equal_weights(stocks)
        
        scores = factor_scores.reindex(stocks).dropna()
        
        # 處理負值（Z-score 可能為負）
        if scores.min() <= 0:
            scores = scores - scores.min() + 0.01
        
        if scores.sum() == 0:
            return self._calc_equal_weights(stocks)
        
        weights = scores / scores.sum()
        return weights
    
    def _calc_mixed_weights(self, stocks, date, factor_scores):
        """計算混合加權"""
        alpha = self.config.MIXED_WEIGHT_ALPHA
        
        cap_weights = self._calc_market_cap_weights(stocks, date)
        factor_weights = self._calc_factor_weights(stocks, factor_scores)
        
        # 對齊索引
        common_stocks = cap_weights.index.intersection(factor_weights.index)
        cap_weights = cap_weights.reindex(common_stocks).fillna(0)
        factor_weights = factor_weights.reindex(common_stocks).fillna(0)
        
        weights = alpha * cap_weights + (1 - alpha) * factor_weights
        
        # 重新正規化
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        return weights
    
    def _apply_weight_constraints(self, weights):
        """套用權重上下限"""
        if len(weights) == 0:
            return weights
        
        cap = self.config.WEIGHT_CAP
        floor = self.config.WEIGHT_FLOOR
        
        # 無限制則直接回傳
        if cap is None and floor is None:
            return weights
        
        # 迭代調整
        for _ in range(MAX_WEIGHT_ITERATIONS):
            adjusted = False
            
            # 套用上限
            if cap is not None:
                excess = weights[weights > cap] - cap
                if len(excess) > 0:
                    weights[weights > cap] = cap
                    # 將超額分配給未達上限的股票
                    under_cap = weights[weights < cap]
                    if len(under_cap) > 0:
                        redistribution = excess.sum() * (under_cap / under_cap.sum())
                        weights[under_cap.index] += redistribution
                    adjusted = True
            
            # 套用下限
            if floor is not None:
                deficit = floor - weights[weights < floor]
                if len(deficit) > 0:
                    weights[weights < floor] = floor
                    # 從超過下限的股票扣除
                    above_floor = weights[weights > floor]
                    if len(above_floor) > 0:
                        reduction = deficit.sum() * (above_floor / above_floor.sum())
                        weights[above_floor.index] -= reduction
                    adjusted = True
            
            if not adjusted:
                break
        
        # 最終正規化
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        return weights
    
    # ========== 通用因子計算方法（子類可使用）==========
    
    def _calc_composite_score(self, date, factors_config=None):
        """
        計算綜合因子分數
        
        Args:
            date: 計算日期
            factors_config: 因子設定，None 則使用 self.config.FACTORS
            
        Returns:
            pd.Series: 股票代碼為索引，綜合分數為值
        """
        if factors_config is None:
            factors_config = self.config.FACTORS
        
        if not factors_config:
            return pd.Series(dtype=float)
        
        all_scores = {}
        
        for factor_name, factor_info in factors_config.items():
            factor_file = factor_info.get("file")
            weight = factor_info.get("weight", 1.0)
            direction = factor_info.get("direction", 1)
            
            if factor_file is None:
                continue
            
            # 取得因子資料
            raw_data = self.data_manager.get_factor_data(factor_file, date)
            
            if len(raw_data) == 0:
                self.logger.warning(f"因子 {factor_name} 無資料")
                continue
            
            # 標準化
            standardized = self._standardize(raw_data)
            
            # 方向調整
            if direction < 0:
                standardized = 1 - standardized  # percentile 反轉
                # 若為 zscore，則乘以 -1
                if self.config.STANDARDIZE_METHOD == "zscore":
                    standardized = -standardized
            
            # 加權
            all_scores[factor_name] = standardized * weight
        
        if not all_scores:
            return pd.Series(dtype=float)
        
        # 合併所有因子
        scores_df = pd.DataFrame(all_scores)
        
        # 計算綜合分數（加總）
        composite = scores_df.sum(axis=1)
        
        return composite
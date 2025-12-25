# -*- coding: utf-8 -*-
"""
指數計算引擎
執行指數計算的核心邏輯
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from config import *


class IndexEngine:
    """指數計算引擎"""
    
    def __init__(self, index, data_manager):
        """
        初始化引擎
        
        Args:
            index: 指數實例（如 QualityIndex）
            data_manager: DataManager 實例
        """
        self.index = index
        self.data_manager = data_manager
        self.config = index.config
        self.logger = self._setup_logger()
        
        # 指數序列
        self.price_index_series = {}        # {date: value}
        self.total_return_index_series = {} # {date: value}
        
        # 每日資料
        self.daily_weights = {}             # {date: pd.Series}
        self.daily_prices = {}              # {date: pd.Series}
        self.daily_returns = {}             # {date: pd.Series}
        self.daily_contributions = {}       # {date: pd.Series}
        
        # 調倉資料
        self.rebalance_schedule = []        # [(調整日, 生效日), ...]
        self.rebalance_history = []         # 調倉紀錄
        
        # 過渡期資料
        self.transition_data = {}           # {date: {'old': weights, 'new': weights}}
        
        # 當前投資組合
        self.current_stocks = []
        self.current_weights = pd.Series(dtype=float)
        
        # 待生效投資組合（過渡期間）
        self.pending_stocks = None
        self.pending_weights = None
        self.pending_effective_date = None
    
    def _setup_logger(self):
        """設定日誌"""
        logger = logging.getLogger(f"Engine_{self.config.INDEX_CODE}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(LOG_FORMAT)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(LOG_LEVEL)
        return logger
    
    # ========== 主要執行方法 ==========
    
    def run(self, start_date=None, end_date=None):
        """
        執行完整指數計算
        
        Args:
            start_date: 起始日期，None 則使用 BASE_DATE
            end_date: 結束日期，None 則使用資料最後日期
            
        Returns:
            dict: 計算結果
        """
        self.logger.info(f"=== 開始計算 {self.config.INDEX_NAME} ===")
        
        # 設定日期範圍
        if start_date is None:
            start_date = self.config.BASE_DATE
        start_date = pd.to_datetime(start_date)
        
        if end_date is None:
            end_date = self.data_manager.trading_dates[-1]
        end_date = pd.to_datetime(end_date)
        
        # 取得交易日
        trading_dates = self.data_manager.get_trading_dates(start_date, end_date)
        
        if len(trading_dates) == 0:
            self.logger.error("無交易日資料")
            return None
        
        self.logger.info(f"計算期間: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        self.logger.info(f"交易日數: {len(trading_dates)}")
        
        # 產生調倉日期
        self._generate_rebalance_schedule(start_date, end_date)
        self.logger.info(f"調倉次數: {len(self.rebalance_schedule)}")
        
        # 初始化
        self._initialize(trading_dates[0])
        
        # 逐日計算
        for i, date in enumerate(trading_dates):
            self._process_date(date)
            
            # 進度顯示
            if (i + 1) % 250 == 0:
                self.logger.info(f"已處理 {i + 1} / {len(trading_dates)} 天")
        
        self.logger.info(f"=== {self.config.INDEX_NAME} 計算完成 ===")
        
        return self.get_results()
    
    def _initialize(self, start_date):
        """
        初始化指數
        
        Args:
            start_date: 起始日期
        """
        self.logger.info(f"初始化指數，基準日: {start_date.strftime('%Y-%m-%d')}")
        
        # 初始選股
        stocks, scores, rejected = self.index.select_stocks(start_date)
        
        if len(stocks) == 0:
            self.logger.error("初始化失敗：無法選出成分股")
            return
        
        # 計算初始權重
        weights = self.index.calc_weights(stocks, start_date, scores)
        
        # 設定當前投組
        self.current_stocks = stocks
        self.current_weights = weights
        
        # 設定基準值
        self.price_index_series[start_date] = self.config.BASE_VALUE
        self.total_return_index_series[start_date] = self.config.BASE_VALUE
        
        # 記錄每日資料
        self.daily_weights[start_date] = weights.copy()
        self.daily_prices[start_date] = self.data_manager.get_close_price(start_date, stocks)
        self.daily_returns[start_date] = pd.Series(0, index=stocks)
        self.daily_contributions[start_date] = pd.Series(0, index=stocks)
        
        # 記錄初始調倉
        self._record_rebalance(start_date, start_date, stocks, weights, scores, rejected)
        
        self.logger.info(f"初始成分股: {len(stocks)} 檔")
    
    def _process_date(self, date):
        """
        處理單一交易日
        
        Args:
            date: 交易日期
        """
        # 檢查是否為調整日
        if self._is_review_date(date):
            self._process_review(date)
        
        # 檢查是否需要切換投組（生效日前一天收盤後）
        if self._should_switch_portfolio(date):
            self._switch_portfolio(date)
        
        # 計算當日指數（跳過基準日）
        if date not in self.price_index_series:
            self._process_trading_day(date)
    
    # ========== 調倉日期相關 ==========
    
    def _generate_rebalance_schedule(self, start_date, end_date):
        """
        產生調倉日期列表
        
        Args:
            start_date: 起始日期
            end_date: 結束日期
        """
        self.rebalance_schedule = []
        
        # 取得調倉月份
        months = self.config.REBALANCE_MONTHS
        
        # 遍歷每年每月
        current_year = start_date.year
        end_year = end_date.year
        
        for year in range(current_year, end_year + 1):
            for month in months:
                review_date = self._get_review_date(year, month)
                
                if review_date is None:
                    continue
                
                # 檢查是否在範圍內
                if review_date < start_date or review_date > end_date:
                    continue
                
                effective_date = self._get_effective_date(review_date)
                
                if effective_date is not None and effective_date <= end_date:
                    self.rebalance_schedule.append((review_date, effective_date))
    
    def _get_review_date(self, year, month):
        """
        取得調整日（依 REVIEW_DAY 設定）
        
        Args:
            year: 年
            month: 月
            
        Returns:
            datetime: 調整日
        """
        # 該月最後一天
        if month == 12:
            next_month_first = pd.Timestamp(year + 1, 1, 1)
        else:
            next_month_first = pd.Timestamp(year, month + 1, 1)
        
        last_day_of_month = next_month_first - pd.Timedelta(days=1)
        
        # 找該月最後一個交易日
        month_dates = self.data_manager.trading_dates[
            (self.data_manager.trading_dates.year == year) & 
            (self.data_manager.trading_dates.month == month)
        ]
        
        if len(month_dates) == 0:
            return None
        
        if self.config.REVIEW_DAY == "last_business_day":
            return month_dates[-1]
        else:
            # 可擴充其他規則
            return month_dates[-1]
    
    def _get_effective_date(self, review_date):
        """
        取得生效日
        
        Args:
            review_date: 調整日
            
        Returns:
            datetime: 生效日
        """
        n = self.config.EFFECTIVE_DAYS
        
        if n == 0:
            # 最快生效 = 下一個交易日
            return self.data_manager.get_next_trading_date(review_date)
        else:
            # T + N
            return self.data_manager.get_nth_trading_date_after(review_date, n)
    
    def _is_review_date(self, date):
        """判斷是否為調整日"""
        for review_date, _ in self.rebalance_schedule:
            if date == review_date:
                return True
        return False
    
    def _get_effective_date_for_review(self, review_date):
        """取得調整日對應的生效日"""
        for r_date, e_date in self.rebalance_schedule:
            if r_date == review_date:
                return e_date
        return None
    
    def _should_switch_portfolio(self, date):
        """
        判斷是否應該切換投組（生效日前一天收盤後）
        
        Args:
            date: 當前日期
            
        Returns:
            bool
        """
        if self.pending_effective_date is None:
            return False
        
        # 生效日的前一個交易日
        prev_date = self.data_manager.get_previous_trading_date(self.pending_effective_date)
        
        return date == prev_date
    
    def _is_in_transition(self, date):
        """判斷是否在過渡期"""
        return self.pending_effective_date is not None
    
    # ========== 調倉處理 ==========
    
    def _process_review(self, date):
        """
        處理調整日
        
        Args:
            date: 調整日
        """
        self.logger.info(f"調整日: {date.strftime('%Y-%m-%d')}")
        
        # 選股
        stocks, scores, rejected = self.index.select_stocks(date)
        
        if len(stocks) == 0:
            self.logger.warning(f"{date}: 無法選出成分股，維持原投組")
            return
        
        # 計算權重
        weights = self.index.calc_weights(stocks, date, scores)
        
        # 取得生效日
        effective_date = self._get_effective_date_for_review(date)
        
        if effective_date is None:
            self.logger.warning(f"{date}: 無對應生效日")
            return
        
        # 設定待生效投組
        self.pending_stocks = stocks
        self.pending_weights = weights.copy()
        self.pending_effective_date = effective_date
        
        # 記錄調倉
        self._record_rebalance(date, effective_date, stocks, weights, scores, rejected)
        
        self.logger.info(f"新投組將於 {effective_date.strftime('%Y-%m-%d')} 生效，成分股: {len(stocks)} 檔")
    
    def _record_rebalance(self, review_date, effective_date, stocks, weights, factor_scores, rejected):
        """記錄調倉資訊"""
        # 計算異動
        old_stocks = set(self.current_stocks)
        new_stocks = set(stocks)
        
        added = new_stocks - old_stocks
        removed = old_stocks - new_stocks
        maintained = new_stocks & old_stocks
        
        # 計算換手率
        turnover = self._calc_turnover(self.current_weights, weights)
        
        record = {
            "review_date": review_date,
            "effective_date": effective_date,
            "stocks": stocks,
            "weights": weights.copy(),
            "factor_scores": factor_scores.copy() if isinstance(factor_scores, pd.Series) else factor_scores,
            "rejected": rejected,
            "added": list(added),
            "removed": list(removed),
            "maintained": list(maintained),
            "turnover": turnover,
            "old_weights": self.current_weights.copy() if len(self.current_weights) > 0 else None
        }
        
        self.rebalance_history.append(record)
    
    def _calc_turnover(self, old_weights, new_weights):
        """計算換手率"""
        if old_weights is None or len(old_weights) == 0:
            return 1.0
        
        # 合併所有股票
        all_stocks = set(old_weights.index) | set(new_weights.index)
        
        old_w = old_weights.reindex(all_stocks).fillna(0)
        new_w = new_weights.reindex(all_stocks).fillna(0)
        
        # 換手率 = 權重變化絕對值總和 / 2
        turnover = abs(new_w - old_w).sum() / 2
        
        return turnover
    
    # ========== 投資組合切換 ==========
    
    def _switch_portfolio(self, date):
        """
        切換投資組合（生效日前一天收盤後）
        
        Args:
            date: 當前日期（生效日前一天）
        """
        if self.pending_stocks is None:
            return
        
        self.logger.info(f"切換投組: {date.strftime('%Y-%m-%d')} 收盤後")
        
        # 計算新投組在過渡期間的權重漂移
        drifted_weights = self._calc_drifted_pending_weights(date)
        
        # 切換
        self.current_stocks = self.pending_stocks
        self.current_weights = drifted_weights
        
        # 清除待生效
        self.pending_stocks = None
        self.pending_weights = None
        self.pending_effective_date = None
    
    def _calc_drifted_pending_weights(self, date):
        """
        計算待生效投組的漂移權重
        
        Args:
            date: 當前日期
            
        Returns:
            pd.Series: 漂移後權重
        """
        if self.pending_weights is None:
            return pd.Series(dtype=float)
        
        weights = self.pending_weights.copy()
        stocks = list(weights.index)
        
        # 取得當日報酬
        returns = self.data_manager.get_price_return(date, stocks, use_adjusted=False)
        returns = returns.reindex(stocks).fillna(0)
        
        # 計算漂移
        new_values = weights * (1 + returns)
        total_value = new_values.sum()
        
        if total_value > 0:
            drifted = new_values / total_value
        else:
            drifted = weights
        
        return drifted
    
    # ========== 每日計算 ==========
    
    def _process_trading_day(self, date):
        """
        處理一般交易日
        
        Args:
            date: 交易日期
        """
        prev_date = self.data_manager.get_previous_trading_date(date)
        
        if prev_date is None or prev_date not in self.price_index_series:
            return
        
        stocks = self.current_stocks
        if len(stocks) == 0:
            return
        
        # 取得前一日權重
        prev_weights = self.daily_weights.get(prev_date, self.current_weights)
        
        # 計算報酬
        price_return, dividend_return, contributions = self._calc_portfolio_return(date, prev_weights)
        
        # 更新指數
        prev_price_index = self.price_index_series[prev_date]
        prev_tr_index = self.total_return_index_series[prev_date]
        
        self.price_index_series[date] = prev_price_index * (1 + price_return)
        self.total_return_index_series[date] = prev_tr_index * (1 + price_return + dividend_return)
        
        # 計算權重漂移
        drifted_weights = self._calc_drifted_weights(date, prev_weights)
        
        # 記錄每日資料
        self.daily_weights[date] = drifted_weights
        self.daily_prices[date] = self.data_manager.get_close_price(date, stocks)
        self.daily_returns[date] = self.data_manager.get_price_return(date, stocks, use_adjusted=False)
        self.daily_contributions[date] = contributions
        
        # 更新當前權重
        self.current_weights = drifted_weights
        
        # 更新過渡期資料
        if self._is_in_transition(date):
            self._update_transition_data(date, prev_weights, drifted_weights)
    
    def _calc_portfolio_return(self, date, weights):
        """
        計算投資組合報酬
        
        Args:
            date: 當前日期
            weights: 前一日權重
            
        Returns:
            tuple: (價格報酬, 股息報酬, 各股票貢獻)
        """
        stocks = list(weights.index)
        
        # 取得日報酬（使用收盤價計算，不含股息）
        returns = self.data_manager.get_price_return(date, stocks, use_adjusted=False)
        returns = returns.reindex(stocks).fillna(0)
        
        # 各股票貢獻
        contributions = weights * returns
        
        # 投組報酬
        price_return = contributions.sum()
        
        # 股息報酬
        dividend_return = self._calc_dividend_return(date, weights)
        
        return price_return, dividend_return, contributions
    
    def _calc_dividend_return(self, date, weights):
        """
        計算股息報酬
        
        Args:
            date: 當前日期
            weights: 前一日權重
            
        Returns:
            float: 股息報酬率
        """
        stocks = list(weights.index)
        
        # 取得除息日股利
        dividends = self.data_manager.get_dividend(date, stocks)
        dividends = dividends.reindex(stocks).fillna(0)
        
        if dividends.sum() == 0:
            return 0.0
        
        # 取得前一日收盤價
        prev_date = self.data_manager.get_previous_trading_date(date)
        prev_prices = self.data_manager.get_close_price(prev_date, stocks)
        prev_prices = prev_prices.reindex(stocks)
        
        # 殖利率
        yields = dividends / prev_prices
        yields = yields.fillna(0)
        
        # 加權殖利率
        dividend_return = (weights * yields).sum()
        
        return dividend_return
    
    def _calc_drifted_weights(self, date, prev_weights):
        """
        計算權重漂移
        
        Args:
            date: 當前日期
            prev_weights: 前一日權重
            
        Returns:
            pd.Series: 漂移後權重
        """
        stocks = list(prev_weights.index)
        
        # 取得日報酬
        returns = self.data_manager.get_price_return(date, stocks, use_adjusted=False)
        returns = returns.reindex(stocks).fillna(0)
        
        # 計算漂移
        new_values = prev_weights * (1 + returns)
        total_value = new_values.sum()
        
        if total_value > 0:
            drifted = new_values / total_value
        else:
            drifted = prev_weights
        
        return drifted
    
    def _update_transition_data(self, date, old_weights, new_weights):
        """記錄過渡期權重資料"""
        self.transition_data[date] = {
            "old_portfolio": old_weights.copy(),
            "new_portfolio": self.pending_weights.copy() if self.pending_weights is not None else None
        }
    
    # ========== 結果彙整 ==========
    
    def get_results(self):
        """
        取得計算結果
        
        Returns:
            dict: 包含所有計算結果的字典
        """
        # 轉換為 DataFrame
        price_index_df = pd.Series(self.price_index_series).sort_index()
        tr_index_df = pd.Series(self.total_return_index_series).sort_index()
        
        # 計算日報酬
        price_daily_return = price_index_df.pct_change()
        tr_daily_return = tr_index_df.pct_change()
        
        results = {
            "index_name": self.config.INDEX_NAME,
            "index_code": self.config.INDEX_CODE,
            "price_index": price_index_df,
            "total_return_index": tr_index_df,
            "price_daily_return": price_daily_return,
            "tr_daily_return": tr_daily_return,
            "daily_weights": self.daily_weights,
            "daily_prices": self.daily_prices,
            "daily_returns": self.daily_returns,
            "daily_contributions": self.daily_contributions,
            "rebalance_history": self.rebalance_history,
            "transition_data": self.transition_data,
            "rebalance_schedule": self.rebalance_schedule
        }
        
        # 統計資訊
        if len(price_index_df) > 1:
            total_days = len(price_index_df)
            total_return = (price_index_df.iloc[-1] / price_index_df.iloc[0] - 1) * 100
            
            self.logger.info(f"--- 計算結果摘要 ---")
            self.logger.info(f"計算天數: {total_days}")
            self.logger.info(f"價格指數: {price_index_df.iloc[0]:.2f} → {price_index_df.iloc[-1]:.2f}")
            self.logger.info(f"累積報酬: {total_return:.2f}%")
            self.logger.info(f"調倉次數: {len(self.rebalance_history)}")
        
        return results
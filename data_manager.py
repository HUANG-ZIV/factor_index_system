# -*- coding: utf-8 -*-
"""
資料管理模組
負責讀取、清洗、提供資料
"""

import pandas as pd
import numpy as np
import os
import logging
from config import *


class DataManager:
    """資料管理器"""
    
    def __init__(self, data_root=None):
        """
        初始化資料管理器
        
        Args:
            data_root: 資料根目錄，None 則使用 config 設定
        """
        self.data_root = data_root or DATA_ROOT
        self.logger = self._setup_logger()
        
        # 日頻資料容器
        self.close_price_df = None
        self.adjusted_price_df = None
        self.volume_df = None
        self.market_cap_df = None
        self.dividend_df = None
        
        # 產業分類
        self.sector_df = None
        
        # 因子資料快取（避免重複讀取）
        self.factor_data_cache = {}
        
        # 交易日曆
        self.trading_dates = None
        
        # 載入資料
        self._load_all_data()
    
    def _setup_logger(self):
        """設定日誌"""
        logger = logging.getLogger("DataManager")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(LOG_FORMAT)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(LOG_LEVEL)
        return logger
    
    # ========== 資料載入方法 ==========
    
    def _load_all_data(self):
        """載入所有資料"""
        self.logger.info("開始載入資料...")
        
        self._load_price_data()
        self._load_volume_data()
        self._load_market_cap_data()
        self._load_dividend_data()
        self._load_sector_data()
        self._build_trading_calendar()
        
        self.logger.info("資料載入完成")
    
    def _load_csv(self, filepath, name=""):
        """
        讀取 CSV 並統一日期格式
        
        Args:
            filepath: 檔案路徑
            name: 資料名稱（用於日誌）
            
        Returns:
            pd.DataFrame: 日期為索引，股票為欄位
        """
        if not os.path.exists(filepath):
            self.logger.warning(f"{name} 檔案不存在: {filepath}")
            return None
        
        try:
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df = df.sort_index()
            
            # 欄位名稱轉為字串（股票代碼）
            df.columns = df.columns.astype(str)
            
            self.logger.info(f"載入 {name}: {len(df)} 列, {len(df.columns)} 檔股票")
            return df
        except Exception as e:
            self.logger.error(f"載入 {name} 失敗: {e}")
            return None
    
    def _load_price_data(self):
        """載入股價資料"""
        # 收盤價
        filepath = os.path.join(self.data_root, CLOSE_PRICE_FILE)
        self.close_price_df = self._load_csv(filepath, "收盤價")
        
        # 還原價（選配）
        filepath = os.path.join(self.data_root, ADJUSTED_CLOSE_FILE)
        self.adjusted_price_df = self._load_csv(filepath, "還原價")
        
        # 如果沒有還原價，使用收盤價代替
        if self.adjusted_price_df is None and self.close_price_df is not None:
            self.logger.info("無還原價資料，使用收盤價代替")
            self.adjusted_price_df = self.close_price_df.copy()
    
    def _load_volume_data(self):
        """載入成交量資料"""
        filepath = os.path.join(self.data_root, VOLUME_FILE)
        self.volume_df = self._load_csv(filepath, "成交量")
    
    def _load_market_cap_data(self):
        """載入市值資料"""
        filepath = os.path.join(self.data_root, MARKET_CAP_FILE)
        self.market_cap_df = self._load_csv(filepath, "總市值")
    
    def _load_dividend_data(self):
        """載入股利資料"""
        filepath = os.path.join(self.data_root, "現金股利合計.csv")
        self.dividend_df = self._load_csv(filepath, "現金股利")
    
    def _load_sector_data(self):
        """載入產業分類資料"""
        filepath = os.path.join(self.data_root, "產業分類.csv")
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                # 假設格式：stock_id, sector
                df['stock_id'] = df['stock_id'].astype(str)
                self.sector_df = df.set_index('stock_id')['sector']
                self.logger.info(f"載入產業分類: {len(self.sector_df)} 檔股票")
            except Exception as e:
                self.logger.warning(f"載入產業分類失敗: {e}")
                self.sector_df = None
        else:
            self.logger.info("無產業分類資料")
            self.sector_df = None
    
    def _build_trading_calendar(self):
        """建立交易日曆（從收盤價資料取得）"""
        if self.close_price_df is not None:
            self.trading_dates = self.close_price_df.index
            self.logger.info(f"交易日曆: {self.trading_dates[0].strftime('%Y-%m-%d')} ~ {self.trading_dates[-1].strftime('%Y-%m-%d')}, 共 {len(self.trading_dates)} 天")
        else:
            self.trading_dates = pd.DatetimeIndex([])
            self.logger.warning("無法建立交易日曆")
    
    def _load_factor_file(self, factor_file):
        """
        載入因子資料檔案（有快取機制）
        
        Args:
            factor_file: 因子檔案名稱
            
        Returns:
            pd.DataFrame
        """
        if factor_file in self.factor_data_cache:
            return self.factor_data_cache[factor_file]
        
        # 嘗試在 data_root 下尋找
        filepath = os.path.join(self.data_root, factor_file)
        
        if not os.path.exists(filepath):
            self.logger.warning(f"因子檔案不存在: {filepath}")
            return None
        
        df = self._load_csv(filepath, factor_file)
        if df is not None:
            self.factor_data_cache[factor_file] = df
        
        return df
    
    # ========== 資料取得方法 ==========
    
    def get_close_price(self, date, stocks=None):
        """
        取得收盤價
        
        Args:
            date: 日期
            stocks: 股票列表，None 則回傳所有股票
            
        Returns:
            pd.Series: 股票代碼為索引，收盤價為值
        """
        return self._get_data_at_date(self.close_price_df, date, stocks)
    
    def get_adjusted_price(self, date, stocks=None):
        """
        取得還原價
        
        Args:
            date: 日期
            stocks: 股票列表，None 則回傳所有股票
            
        Returns:
            pd.Series: 股票代碼為索引，還原價為值
        """
        return self._get_data_at_date(self.adjusted_price_df, date, stocks)
    
    def get_market_cap(self, date, stocks=None):
        """
        取得市值
        
        Args:
            date: 日期
            stocks: 股票列表，None 則回傳所有股票
            
        Returns:
            pd.Series: 股票代碼為索引，市值為值
        """
        return self._get_data_at_date(self.market_cap_df, date, stocks)
    
    def get_volume(self, date, stocks=None):
        """
        取得成交量
        
        Args:
            date: 日期
            stocks: 股票列表，None 則回傳所有股票
            
        Returns:
            pd.Series: 股票代碼為索引，成交量為值
        """
        return self._get_data_at_date(self.volume_df, date, stocks)
    
    def get_liquidity(self, date, metric="avg_daily_value", lookback=20, stocks=None):
        """
        取得流動性指標
        
        Args:
            date: 日期
            metric: 指標類型
                - avg_volume: 平均成交量
                - avg_daily_value: 平均成交金額
            lookback: 回溯天數
            stocks: 股票列表，None 則回傳所有股票
            
        Returns:
            pd.Series: 股票代碼為索引，流動性指標為值
        """
        date = pd.to_datetime(date)
        
        # 取得回溯期間的資料
        end_idx = self.trading_dates.get_loc(date)
        start_idx = max(0, end_idx - lookback + 1)
        period_dates = self.trading_dates[start_idx:end_idx + 1]
        
        if metric == "avg_volume":
            if self.volume_df is None:
                return pd.Series(dtype=float)
            data = self.volume_df.loc[period_dates].mean()
        
        elif metric == "avg_daily_value":
            if self.volume_df is None or self.close_price_df is None:
                return pd.Series(dtype=float)
            volume = self.volume_df.loc[period_dates]
            price = self.close_price_df.loc[period_dates]
            daily_value = volume * price
            data = daily_value.mean()
        
        else:
            self.logger.warning(f"未知的流動性指標: {metric}")
            return pd.Series(dtype=float)
        
        if stocks is not None:
            stocks = [str(s) for s in stocks]
            data = data.reindex(stocks)
        
        return data
    
    def get_factor_data(self, factor_file, date):
        """
        取得因子資料（處理季頻資料對齊）
        
        對於季頻資料，會自動向前找最近一期的資料。
        
        Args:
            factor_file: 因子資料檔案名稱
            date: 資料日期
            
        Returns:
            pd.Series: 股票代碼為索引，因子值為值
        """
        df = self._load_factor_file(factor_file)
        if df is None:
            return pd.Series(dtype=float)
        
        date = pd.to_datetime(date)
        
        # 檢查是否為日頻或季頻資料（假設季頻資料列數較少）
        if len(df) < 100:
            # 季頻資料：找 date 當天或之前最近一筆資料
            available_dates = df.index[df.index <= date]
            
            if len(available_dates) == 0:
                return pd.Series(dtype=float)
            
            latest_date = available_dates[-1]
            return df.loc[latest_date]
        else:
            # 日頻資料：直接取該日
            return self._get_data_at_date(df, date)
    
    def get_dividend(self, date, stocks=None):
        """
        取得股利資料（除息日當天的股利）
        
        Args:
            date: 日期
            stocks: 股票列表，None 則回傳所有股票
            
        Returns:
            pd.Series: 股票代碼為索引，股利為值（無除息則為 0）
        """
        if self.dividend_df is None:
            return pd.Series(dtype=float)
        
        data = self._get_data_at_date(self.dividend_df, date, stocks)
        
        # 將 NaN 替換為 0（無除息）
        return data.fillna(0)
    
    def get_all_stocks(self):
        """
        取得所有股票代碼
        
        Returns:
            list: 股票代碼列表
        """
        if self.close_price_df is not None:
            return list(self.close_price_df.columns)
        return []
    
    def get_trading_dates(self, start_date=None, end_date=None):
        """
        取得交易日曆
        
        Args:
            start_date: 起始日期，None 則從頭開始
            end_date: 結束日期，None 則到最後
            
        Returns:
            pd.DatetimeIndex: 交易日期
        """
        dates = self.trading_dates
        
        if start_date is not None:
            start_date = pd.to_datetime(start_date)
            dates = dates[dates >= start_date]
        
        if end_date is not None:
            end_date = pd.to_datetime(end_date)
            dates = dates[dates <= end_date]
        
        return dates
    
    def get_sectors(self):
        """
        取得產業分類
        
        Returns:
            pd.Series: 股票代碼為索引，產業為值
        """
        if self.sector_df is not None:
            return self.sector_df.copy()
        return pd.Series(dtype=str)
    
    def get_price_return(self, date, stocks=None, use_adjusted=True):
        """
        取得日報酬率
        
        Args:
            date: 日期
            stocks: 股票列表，None 則回傳所有股票
            use_adjusted: 是否使用還原價計算
            
        Returns:
            pd.Series: 股票代碼為索引，日報酬率為值
        """
        date = pd.to_datetime(date)
        prev_date = self.get_previous_trading_date(date)
        
        if prev_date is None:
            return pd.Series(dtype=float)
        
        price_df = self.adjusted_price_df if use_adjusted else self.close_price_df
        if price_df is None:
            return pd.Series(dtype=float)
        
        try:
            price_today = price_df.loc[date]
            price_yesterday = price_df.loc[prev_date]
            returns = (price_today - price_yesterday) / price_yesterday
            
            if stocks is not None:
                stocks = [str(s) for s in stocks]
                returns = returns.reindex(stocks)
            
            return returns
        except KeyError:
            return pd.Series(dtype=float)
    
    def get_price_return_series(self, start_date, end_date, stocks=None, use_adjusted=True):
        """
        取得一段期間的報酬率序列
        
        Args:
            start_date: 起始日期
            end_date: 結束日期
            stocks: 股票列表
            use_adjusted: 是否使用還原價
            
        Returns:
            pd.DataFrame: 日期為索引，股票為欄位
        """
        price_df = self.adjusted_price_df if use_adjusted else self.close_price_df
        if price_df is None:
            return pd.DataFrame()
        
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # 需要多取一天來計算第一天的報酬
        prev_start = self.get_previous_trading_date(start_date)
        if prev_start is None:
            prev_start = start_date
        
        prices = price_df.loc[prev_start:end_date]
        returns = prices.pct_change(fill_method=None).loc[start_date:end_date]
        
        if stocks is not None:
            stocks = [str(s) for s in stocks]
            returns = returns[returns.columns.intersection(stocks)]
        
        return returns
    
    def get_previous_trading_date(self, date):
        """
        取得前一個交易日
        
        Args:
            date: 日期
            
        Returns:
            datetime: 前一個交易日，若無則回傳 None
        """
        date = pd.to_datetime(date)
        
        try:
            idx = self.trading_dates.get_loc(date)
            if idx > 0:
                return self.trading_dates[idx - 1]
        except KeyError:
            # 如果 date 不在交易日中，找最近的前一個交易日
            earlier_dates = self.trading_dates[self.trading_dates < date]
            if len(earlier_dates) > 0:
                return earlier_dates[-1]
        
        return None
    
    def get_next_trading_date(self, date):
        """
        取得下一個交易日
        
        Args:
            date: 日期
            
        Returns:
            datetime: 下一個交易日，若無則回傳 None
        """
        date = pd.to_datetime(date)
        
        try:
            idx = self.trading_dates.get_loc(date)
            if idx < len(self.trading_dates) - 1:
                return self.trading_dates[idx + 1]
        except KeyError:
            # 如果 date 不在交易日中，找最近的下一個交易日
            later_dates = self.trading_dates[self.trading_dates > date]
            if len(later_dates) > 0:
                return later_dates[0]
        
        return None
    
    def get_nth_trading_date_after(self, date, n):
        """
        取得 N 個交易日後的日期
        
        Args:
            date: 起始日期
            n: 交易日數（0 表示當天，1 表示下一個交易日）
            
        Returns:
            datetime: 目標日期，若超出範圍則回傳 None
        """
        date = pd.to_datetime(date)
        
        try:
            idx = self.trading_dates.get_loc(date)
            target_idx = idx + n
            if 0 <= target_idx < len(self.trading_dates):
                return self.trading_dates[target_idx]
        except KeyError:
            pass
        
        return None
    
    # ========== 輔助方法 ==========
    
    def _get_data_at_date(self, df, date, stocks=None):
        """
        取得指定日期的資料
        
        Args:
            df: DataFrame
            date: 日期
            stocks: 股票列表
            
        Returns:
            pd.Series
        """
        if df is None:
            return pd.Series(dtype=float)
        
        date = pd.to_datetime(date)
        
        try:
            data = df.loc[date]
        except KeyError:
            # 日期不存在，回傳空 Series
            return pd.Series(dtype=float)
        
        if stocks is not None:
            stocks = [str(s) for s in stocks]
            data = data.reindex(stocks)
        
        return data
    
    def validate_data(self):
        """
        驗證資料完整性
        
        Returns:
            dict: 驗證結果
        """
        results = {
            "close_price": self.close_price_df is not None,
            "adjusted_price": self.adjusted_price_df is not None,
            "volume": self.volume_df is not None,
            "market_cap": self.market_cap_df is not None,
            "dividend": self.dividend_df is not None,
            "sector": self.sector_df is not None,
            "trading_dates": len(self.trading_dates) > 0
        }
        
        self.logger.info("=== 資料驗證結果 ===")
        for key, value in results.items():
            status = "✓" if value else "✗"
            self.logger.info(f"  {status} {key}")
        
        return results


# 測試用
if __name__ == "__main__":
    dm = DataManager()
    dm.validate_data()
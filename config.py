# -*- coding: utf-8 -*-
"""
全域設定檔
定義資料路徑、交易日曆等所有指數共用的參數
"""

import os
import logging

# ========== 資料路徑設定 ==========
DATA_ROOT = "./data"
PRICE_DATA_PATH = os.path.join(DATA_ROOT, "price")
FINANCIAL_DATA_PATH = os.path.join(DATA_ROOT, "financial")
DIVIDEND_DATA_PATH = os.path.join(DATA_ROOT, "dividend")

# 特定資料檔案
CLOSE_PRICE_FILE = "收盤價.csv"
ADJUSTED_CLOSE_FILE = "收盤價_還原.csv"
VOLUME_FILE = "成交量.csv"
MARKET_CAP_FILE = "總市值.csv"

# ========== 輸出路徑設定 ==========
OUTPUT_ROOT = "./output"

# ========== 日期格式設定 ==========
DATE_FORMAT = "%Y-%m-%d"

# ========== 指數基礎設定 ==========
DEFAULT_BASE_VALUE = 100
DEFAULT_BASE_DATE = "2005-01-03"

# ========== 計算設定 ==========
MIN_VALID_STOCKS = 10
MAX_WEIGHT_ITERATIONS = 100
WEIGHT_TOLERANCE = 1e-6

# ========== 日誌設定 ==========
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

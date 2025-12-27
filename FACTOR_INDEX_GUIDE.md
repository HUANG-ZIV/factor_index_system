# 因子指數設定指南

## 一、調整現有指數參數

直接修改 `indices/` 資料夾內的 `*_index.py` 檔案。

### 1.1 基本設定

```python
INDEX_NAME = "指數名稱"
INDEX_CODE = "CODE"
BASE_DATE = "2024-01-03"      # 基準日
BASE_VALUE = 100              # 基準值
```

### 1.2 調倉設定

```python
REBALANCE_FREQ = "Q"          # M=月, Q=季, SA=半年, A=年
REBALANCE_MONTHS = [3,6,9,12] # 調倉月份
EFFECTIVE_DAYS = 5            # 審核日後幾天生效
```

### 1.3 股票池篩選

```python
MARKET_CAP_FILTER = {
    "method": "top_n",        # top_n / top_percent / min_value / None
    "value": 300              # 前 300 大市值
}

LIQUIDITY_FILTER = {
    "method": "top_percent",
    "value": 0.90,            # 流動性前 90%
    "metric": "avg_daily_value"
}
```

### 1.4 選股設定

```python
SELECTION_METHOD = "top_n"    # top_n / top_percent
TOP_N = 50                    # 選幾檔
TOP_PERCENT = 0.20            # 或選前 20%
```

### 1.5 權重設定

```python
WEIGHTING_METHOD = "factor"   # equal / market_cap / factor / mixed
WEIGHT_CAP = 0.10             # 單檔上限 10%
WEIGHT_FLOOR = None           # 單檔下限
```

### 1.6 因子設定（多因子）

```python
FACTORS = {
    "factor_name": {
        "file": "檔案名.csv",  # 因子資料檔
        "weight": 0.5,         # 權重
        "direction": 1         # 1=越高越好, -1=越低越好
    }
}
```

---

## 二、新增指數步驟

### Step 1：建立設定檔

在 `indices/` 建立 `xxx_index.py`：

```python
# -*- coding: utf-8 -*-
from .base_index import BaseIndexConfig, BaseIndex

class XxxIndexConfig(BaseIndexConfig):
    INDEX_NAME = "新指數名稱"
    INDEX_CODE = "XXX"
    BASE_DATE = "2024-01-03"
    
    # 調倉設定
    REBALANCE_FREQ = "Q"
    REBALANCE_MONTHS = [3, 6, 9, 12]
    EFFECTIVE_DAYS = 5
    
    # 股票池篩選
    MARKET_CAP_FILTER = {
        "method": "top_n",
        "value": 300
    }
    
    LIQUIDITY_FILTER = {
        "method": "top_percent",
        "value": 0.90,
        "metric": "avg_daily_value"
    }
    
    # 選股設定
    SELECTION_METHOD = "top_n"
    TOP_N = 50
    
    # 權重設定
    WEIGHTING_METHOD = "factor"
    WEIGHT_CAP = 0.10
    
    # 因子設定
    FACTORS = {}
    
    # 標準化設定
    STANDARDIZE_METHOD = "percentile"
    WINSORIZE = True
    WINSORIZE_LIMITS = (0.01, 0.99)


class XxxIndex(BaseIndex):
    config = XxxIndexConfig
    
    def calc_factor_score(self, date):
        # 實作因子計算邏輯
        pass
```

### Step 2：註冊指數

**`indices/__init__.py`** 加入：

```python
from .xxx_index import XxxIndex, XxxIndexConfig
```

**`main.py`** 的 `INDEX_MAP` 加入：

```python
"XXX": XxxIndex,
```

### Step 3：執行測試

```bash
python main.py --index XXX
```

---

## 三、因子計算範例

### 3.1 動能因子（自行計算）

```python
class MomentumIndexConfig(BaseIndexConfig):
    # ...
    MOMENTUM_LOOKBACK = 20    # 回溯天數
    MOMENTUM_SKIP = 0         # 排除近期天數

class MomentumIndex(BaseIndex):
    config = MomentumIndexConfig
    
    def calc_factor_score(self, date):
        momentum = self._calc_momentum(
            date, 
            lookback=self.config.MOMENTUM_LOOKBACK,
            skip=self.config.MOMENTUM_SKIP
        )
        return self._standardize(momentum)
    
    def _calc_momentum(self, date, lookback, skip=0):
        date = pd.to_datetime(date)
        trading_dates = self.data_manager.trading_dates
        
        end_idx = trading_dates.get_loc(date)
        momentum_end_idx = end_idx - skip
        momentum_start_idx = momentum_end_idx - lookback
        
        start_date = trading_dates[momentum_start_idx]
        end_date = trading_dates[momentum_end_idx]
        
        price_df = self.data_manager.adjusted_price_df
        start_price = price_df.loc[start_date]
        end_price = price_df.loc[end_date]
        
        momentum = (end_price - start_price) / start_price
        return momentum.dropna()
```

### 3.2 殖利率因子（自行計算）

```python
class DividendIndexConfig(BaseIndexConfig):
    # ...
    DIVIDEND_LOOKBACK_DAYS = 252  # 回溯一年

class DividendIndex(BaseIndex):
    config = DividendIndexConfig
    
    def calc_factor_score(self, date):
        dividend_yield = self._calc_dividend_yield(
            date, 
            lookback_days=self.config.DIVIDEND_LOOKBACK_DAYS
        )
        return self._standardize(dividend_yield)
    
    def _calc_dividend_yield(self, date, lookback_days=252):
        # 計算邏輯：
        # 殖利率 = Σ (除息金額 ÷ 除息前一日股價)
        # 詳見 dividend_index.py
        pass
```

### 3.3 單因子（讀檔案）

```python
class QualityIndex(BaseIndex):
    config = QualityIndexConfig
    
    def calc_factor_score(self, date):
        # 使用 _calc_composite_score 自動處理 FACTORS 設定
        return self._calc_composite_score(date)
```

對應設定：

```python
FACTORS = {
    "roe": {
        "file": "稅後權益報酬率.csv",
        "weight": 1.0,
        "direction": 1    # 越高越好
    }
}
```

### 3.4 多因子（混合）

**方法一：使用 FACTORS 設定**

```python
FACTORS = {
    "roe": {
        "file": "稅後權益報酬率.csv",
        "weight": 0.4,
        "direction": 1
    },
    "pe": {
        "file": "本益比.csv",
        "weight": 0.3,
        "direction": -1   # 越低越好
    },
    "pb": {
        "file": "股價淨值比.csv",
        "weight": 0.3,
        "direction": -1
    }
}

class MultiFactorIndex(BaseIndex):
    def calc_factor_score(self, date):
        return self._calc_composite_score(date)
```

**方法二：自訂計算邏輯**

```python
def calc_factor_score(self, date):
    # 讀取因子
    roe = self.data_manager.get_factor_data("稅後權益報酬率.csv", date)
    pe = self.data_manager.get_factor_data("本益比.csv", date)
    momentum = self._calc_momentum(date, 20)
    
    # 標準化
    roe_std = self._standardize(roe)           # 越高越好
    pe_std = 1 - self._standardize(pe)         # 越低越好（反向）
    mom_std = self._standardize(momentum)
    
    # 加權合併
    score = 0.4 * roe_std + 0.3 * pe_std + 0.3 * mom_std
    return score
```

---

## 四、參數速查表

### 4.1 調倉頻率

| 參數值 | 說明 |
|-------|------|
| `M` | 月調倉 |
| `Q` | 季調倉 |
| `SA` | 半年調倉 |
| `A` | 年調倉 |

### 4.2 篩選方式

| 參數值 | 說明 |
|-------|------|
| `top_n` | 前 N 檔 |
| `top_percent` | 前 X% |
| `min_value` | 最低門檻值 |
| `None` | 不篩選 |

### 4.3 權重方式

| 參數值 | 說明 |
|-------|------|
| `equal` | 等權重 |
| `market_cap` | 市值加權 |
| `factor` | 因子加權 |
| `mixed` | 混合加權（市值 + 因子）|

### 4.4 因子方向

| 參數值 | 說明 |
|-------|------|
| `1` | 越高越好（如 ROE、殖利率）|
| `-1` | 越低越好（如本益比、負債比）|

---

## 五、現有指數總覽

| 指數 | 代碼 | 調倉 | 成分股 | 權重 |
|-----|------|------|-------|------|
| 低波動 | LOWVOL | 季 | 50 | 因子加權 |
| 規模 | SIZE | 季 | 50 | 市值加權 |
| 品質 | QUALITY | 季 | 50 | 因子加權 |
| 動能 | MOMENTUM | 月 | 50 | 因子加權 |
| 價值 | VALUE | 季 | 50 | 因子加權 |
| 高股息 | DIVIDEND | 季 | 50 | 因子加權 |

---

## 六、執行指令

```bash
# 計算單一指數
python main.py --index QUALITY

# 計算所有指數
python main.py --index ALL

# 指定日期範圍
python main.py --index QUALITY --start 2024-01-03 --end 2025-12-31
```

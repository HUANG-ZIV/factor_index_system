# -*- coding: utf-8 -*-
"""
輸出管理模組
負責輸出結果檔案
"""

import pandas as pd
import numpy as np
import os
import logging
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
from config import *


class OutputManager:
    """輸出管理器"""
    
    def __init__(self, index, engine, results):
        """
        初始化輸出管理器
        
        Args:
            index: 指數實例
            engine: IndexEngine 實例
            results: engine.get_results() 的結果
        """
        self.index = index
        self.engine = engine
        self.results = results
        self.config = index.config
        self.logger = self._setup_logger()
        
        # 輸出路徑
        self.output_dir = os.path.join(OUTPUT_ROOT, self.config.INDEX_CODE)
        self.rebalance_dir = os.path.join(self.output_dir, "rebalance")
    
    def _setup_logger(self):
        """設定日誌"""
        logger = logging.getLogger(f"Output_{self.config.INDEX_CODE}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(LOG_FORMAT)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(LOG_LEVEL)
        return logger
    
    def _ensure_output_dirs(self):
        """確保輸出目錄存在"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.rebalance_dir, exist_ok=True)
    
    # ========== 主要輸出方法 ==========
    
    def export_all(self):
        """輸出所有檔案"""
        self._ensure_output_dirs()
        
        self.logger.info(f"開始輸出 {self.config.INDEX_NAME} 結果...")
        
        # 主要檔案
        main_file = self.export_main_file()
        self.logger.info(f"主檔案: {main_file}")
        
        # 調倉詳細檔案
        rebalance_files = self.export_rebalance_files()
        self.logger.info(f"調倉檔案: {len(rebalance_files)} 個")
        
        self.logger.info("輸出完成")
        
        return {
            "main_file": main_file,
            "rebalance_files": rebalance_files
        }
    
    def export_main_file(self):
        """
        輸出主要 Excel 檔案
        
        輸出檔案：{INDEX_CODE}_index.xlsx
        """
        filename = f"{self.config.INDEX_CODE}_index.xlsx"
        filepath = os.path.join(self.output_dir, filename)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 工作表1：每日指數值
            self._write_daily_index(writer)
            
            # 工作表2：每日權重
            self._write_daily_weights(writer)
            
            # 工作表3：每日收盤價
            self._write_daily_prices(writer)
            
            # 工作表4：每日報酬率
            self._write_daily_returns(writer)
            
            # 工作表5：每日貢獻度
            self._write_daily_contributions(writer)
            
            # 工作表6：調倉摘要
            self._write_rebalance_summary(writer)
            
            # 工作表7：過渡期權重
            self._write_transition_weights(writer)
        
        return filepath
    
    def export_rebalance_files(self):
        """
        輸出調倉詳細檔案
        
        輸出檔案：rebalance/rebalance_YYYYMMDD.xlsx
        """
        files = []
        
        for record in self.results["rebalance_history"]:
            review_date = record["review_date"]
            filename = f"rebalance_{review_date.strftime('%Y%m%d')}.xlsx"
            filepath = os.path.join(self.rebalance_dir, filename)
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # 工作表1：調倉資訊
                self._write_rebalance_info(writer, record)
                
                # 工作表2：入選成分股
                self._write_selected_stocks(writer, record)
                
                # 工作表3：未入選股票
                self._write_rejected_stocks(writer, record)
                
                # 工作表4：權重變化明細
                self._write_weight_changes(writer, record)
            
            files.append(filepath)
        
        return files
    
    # ========== 主檔工作表 ==========
    
    def _write_daily_index(self, writer):
        """工作表1：每日指數值"""
        price_index = self.results["price_index"]
        tr_index = self.results["total_return_index"]
        price_return = self.results["price_daily_return"]
        tr_return = self.results["tr_daily_return"]
        
        df = pd.DataFrame({
            "date": price_index.index,
            "price_index": price_index.values,
            "total_return_index": tr_index.values,
            "price_daily_return": price_return.values,
            "tr_daily_return": tr_return.values
        })
        
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        
        df.to_excel(writer, sheet_name="每日指數值", index=False)
    
    def _write_daily_weights(self, writer):
        """工作表2：每日權重（寬表）"""
        df = self._dict_to_wide_df(self.results["daily_weights"])
        
        if len(df) > 0:
            df.index = df.index.strftime("%Y-%m-%d")
            df = df.reset_index().rename(columns={"index": "date"})
            df.to_excel(writer, sheet_name="每日權重", index=False)
        else:
            pd.DataFrame({"message": ["無資料"]}).to_excel(writer, sheet_name="每日權重", index=False)
    
    def _write_daily_prices(self, writer):
        """工作表3：每日收盤價（寬表）"""
        df = self._dict_to_wide_df(self.results["daily_prices"])
        
        if len(df) > 0:
            df.index = df.index.strftime("%Y-%m-%d")
            df = df.reset_index().rename(columns={"index": "date"})
            df.to_excel(writer, sheet_name="每日收盤價", index=False)
        else:
            pd.DataFrame({"message": ["無資料"]}).to_excel(writer, sheet_name="每日收盤價", index=False)
    
    def _write_daily_returns(self, writer):
        """工作表4：每日報酬率（寬表）"""
        df = self._dict_to_wide_df(self.results["daily_returns"])
        
        if len(df) > 0:
            df.index = df.index.strftime("%Y-%m-%d")
            df = df.reset_index().rename(columns={"index": "date"})
            df.to_excel(writer, sheet_name="每日報酬率", index=False)
        else:
            pd.DataFrame({"message": ["無資料"]}).to_excel(writer, sheet_name="每日報酬率", index=False)
    
    def _write_daily_contributions(self, writer):
        """工作表5：每日貢獻度（寬表）"""
        df = self._dict_to_wide_df(self.results["daily_contributions"])
        
        if len(df) > 0:
            # 加入 total 欄位
            df["total"] = df.sum(axis=1)
            df.index = df.index.strftime("%Y-%m-%d")
            df = df.reset_index().rename(columns={"index": "date"})
            df.to_excel(writer, sheet_name="每日貢獻度", index=False)
        else:
            pd.DataFrame({"message": ["無資料"]}).to_excel(writer, sheet_name="每日貢獻度", index=False)
    
    def _write_rebalance_summary(self, writer):
        """工作表6：調倉摘要"""
        records = []
        
        for record in self.results["rebalance_history"]:
            records.append({
                "review_date": record["review_date"].strftime("%Y-%m-%d"),
                "effective_date": record["effective_date"].strftime("%Y-%m-%d"),
                "total_stocks": len(record["stocks"]),
                "stocks_added": len(record["added"]),
                "stocks_removed": len(record["removed"]),
                "stocks_maintained": len(record["maintained"]),
                "turnover": record["turnover"]
            })
        
        df = pd.DataFrame(records)
        df.to_excel(writer, sheet_name="調倉摘要", index=False)
    
    def _write_transition_weights(self, writer):
        """工作表7：過渡期權重"""
        transition_data = self.results["transition_data"]
        
        if not transition_data:
            pd.DataFrame({"message": ["無過渡期資料"]}).to_excel(writer, sheet_name="過渡期權重", index=False)
            return
        
        rows = []
        for date, data in transition_data.items():
            old_portfolio = data.get("old_portfolio")
            new_portfolio = data.get("new_portfolio")
            
            if old_portfolio is not None:
                for stock, weight in old_portfolio.items():
                    rows.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "portfolio_type": "old",
                        "stock_id": stock,
                        "weight": weight
                    })
            
            if new_portfolio is not None:
                for stock, weight in new_portfolio.items():
                    rows.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "portfolio_type": "new",
                        "stock_id": stock,
                        "weight": weight
                    })
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_excel(writer, sheet_name="過渡期權重", index=False)
        else:
            pd.DataFrame({"message": ["無過渡期資料"]}).to_excel(writer, sheet_name="過渡期權重", index=False)
    
    # ========== 調倉詳細檔工作表 ==========
    
    def _write_rebalance_info(self, writer, record):
        """調倉詳細檔工作表1：調倉資訊"""
        info = {
            "項目": [
                "指數名稱",
                "指數代碼",
                "調整日",
                "生效日",
                "選股方法",
                "選股比例/數量",
                "權重方法",
                "權重上限",
                "成分股數量",
                "新增股票數",
                "刪除股票數",
                "維持股票數",
                "換手率"
            ],
            "值": [
                self.config.INDEX_NAME,
                self.config.INDEX_CODE,
                record["review_date"].strftime("%Y-%m-%d"),
                record["effective_date"].strftime("%Y-%m-%d"),
                self.config.SELECTION_METHOD,
                self.config.TOP_PERCENT if self.config.SELECTION_METHOD == "top_percent" else self.config.TOP_N,
                self.config.WEIGHTING_METHOD,
                self.config.WEIGHT_CAP,
                len(record["stocks"]),
                len(record["added"]),
                len(record["removed"]),
                len(record["maintained"]),
                f"{record['turnover']:.2%}"
            ]
        }
        
        df = pd.DataFrame(info)
        df.to_excel(writer, sheet_name="調倉資訊", index=False)
    
    def _write_selected_stocks(self, writer, record):
        """調倉詳細檔工作表2：入選成分股"""
        stocks = record["stocks"]
        weights = record["weights"]
        factor_scores = record["factor_scores"]
        added = set(record["added"])
        
        rows = []
        for i, stock in enumerate(stocks):
            row = {
                "rank": i + 1,
                "stock_id": stock,
                "weight": weights.get(stock, 0) if isinstance(weights, dict) else weights.loc[stock] if stock in weights.index else 0,
                "factor_score": factor_scores.get(stock, None) if isinstance(factor_scores, dict) else factor_scores.loc[stock] if isinstance(factor_scores, pd.Series) and stock in factor_scores.index else None,
                "action": "新增" if stock in added else "維持"
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name="入選成分股", index=False)
    
    def _write_rejected_stocks(self, writer, record):
        """調倉詳細檔工作表3：未入選股票"""
        rejected = record["rejected"]
        
        if not rejected:
            pd.DataFrame({"message": ["無落選股票資料"]}).to_excel(writer, sheet_name="未入選股票", index=False)
            return
        
        rows = []
        for item in rejected:
            rows.append({
                "rank": item.get("rank", ""),
                "stock_id": item.get("stock_id", ""),
                "factor_score": item.get("score", ""),
                "reason": item.get("reason", "")
            })
        
        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name="未入選股票", index=False)
    
    def _write_weight_changes(self, writer, record):
        """調倉詳細檔工作表4：權重變化明細"""
        new_weights = record["weights"]
        old_weights = record["old_weights"]
        
        if old_weights is None:
            old_weights = pd.Series(dtype=float)
        
        # 合併所有股票
        all_stocks = set(new_weights.index) | set(old_weights.index)
        
        rows = []
        for stock in sorted(all_stocks):
            old_w = old_weights.get(stock, 0) if isinstance(old_weights, dict) else old_weights.loc[stock] if stock in old_weights.index else 0
            new_w = new_weights.get(stock, 0) if isinstance(new_weights, dict) else new_weights.loc[stock] if stock in new_weights.index else 0
            
            change = new_w - old_w
            
            if old_w == 0 and new_w > 0:
                action = "新增"
            elif old_w > 0 and new_w == 0:
                action = "刪除"
            elif change > 0:
                action = "增加"
            elif change < 0:
                action = "減少"
            else:
                action = "維持"
            
            rows.append({
                "stock_id": stock,
                "old_weight": old_w,
                "new_weight": new_w,
                "weight_change": change,
                "action": action
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values("new_weight", ascending=False)
        df.to_excel(writer, sheet_name="權重變化明細", index=False)
    
    # ========== 輔助方法 ==========
    
    def _dict_to_wide_df(self, data_dict):
        """
        將 {date: Series} 字典轉換為寬表 DataFrame
        
        Args:
            data_dict: {date: pd.Series} 格式的字典
            
        Returns:
            pd.DataFrame: 寬表格式（日期為索引，股票為欄位）
        """
        if not data_dict:
            return pd.DataFrame()
        
        # 合併所有 Series
        df = pd.DataFrame(data_dict).T
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        return df
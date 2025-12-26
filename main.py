# -*- coding: utf-8 -*-
"""
主程式
串接各模組，執行指數建構
"""

import argparse
import logging
import sys
import os

# 確保可以匯入同目錄模組
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from data_manager import DataManager
from index_engine import IndexEngine
from output_manager import OutputManager

# 匯入所有指數
from indices import (
    QualityIndex,
    #ValueIndex,
    MomentumIndex,
    SizeIndex,
    DividendIndex,
    LowVolIndex
)


# 指數對照表
INDEX_MAP = {
    "QUALITY": QualityIndex,
    #"VALUE": ValueIndex,
    "MOMENTUM": MomentumIndex,
    "SIZE": SizeIndex,
    "DIVIDEND": DividendIndex,
    "LOWVOL": LowVolIndex
}


def setup_logging():
    """設定日誌"""
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT
    )


def run_index(index_code, start_date=None, end_date=None, data_root=None):
    """
    執行單一指數計算
    
    Args:
        index_code: 指數代碼（QUALITY / VALUE / MOMENTUM / SIZE / DIVIDEND / LOWVOL）
        start_date: 起始日期
        end_date: 結束日期
        data_root: 資料根目錄
        
    Returns:
        dict: 計算結果
    """
    logger = logging.getLogger("Main")
    
    # 檢查指數代碼
    if index_code not in INDEX_MAP:
        logger.error(f"未知的指數代碼: {index_code}")
        logger.info(f"可用的指數: {list(INDEX_MAP.keys())}")
        return None
    
    # 初始化資料管理器
    logger.info("載入資料...")
    data_manager = DataManager(data_root)
    
    # 驗證資料
    validation = data_manager.validate_data()
    if not validation["close_price"]:
        logger.error("缺少必要資料：收盤價")
        return None
    
    # 初始化指數
    IndexClass = INDEX_MAP[index_code]
    index = IndexClass(data_manager)
    
    logger.info(f"計算指數: {index.config.INDEX_NAME}")
    
    # 初始化引擎
    engine = IndexEngine(index, data_manager)
    
    # 執行計算
    results = engine.run(start_date, end_date)
    
    if results is None:
        logger.error("指數計算失敗")
        return None
    
    # 輸出結果
    logger.info("輸出結果...")
    output_manager = OutputManager(index, engine, results)
    output_files = output_manager.export_all()
    
    logger.info(f"主檔案: {output_files['main_file']}")
    logger.info(f"調倉檔案數: {len(output_files['rebalance_files'])}")
    
    return results


def run_all_indices(start_date=None, end_date=None, data_root=None):
    """
    執行所有指數計算
    
    Args:
        start_date: 起始日期
        end_date: 結束日期
        data_root: 資料根目錄
        
    Returns:
        dict: 各指數計算結果
    """
    logger = logging.getLogger("Main")
    
    all_results = {}
    
    for index_code in INDEX_MAP.keys():
        logger.info(f"\n{'='*60}")
        logger.info(f"計算 {index_code} 指數")
        logger.info(f"{'='*60}")
        
        try:
            results = run_index(index_code, start_date, end_date, data_root)
            all_results[index_code] = results
        except Exception as e:
            logger.error(f"{index_code} 計算失敗: {e}")
            all_results[index_code] = None
    
    return all_results


def main():
    """主程式進入點"""
    
    # 解析命令列參數
    parser = argparse.ArgumentParser(description="因子指數建構系統")
    parser.add_argument(
        "--index", 
        type=str, 
        default="LOWVOL",
        choices=list(INDEX_MAP.keys()) + ["ALL"],
        help="要計算的指數代碼，或 ALL 計算所有指數"
    )
    parser.add_argument(
        "--start", 
        type=str, 
        default=None,
        help="起始日期 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", 
        type=str, 
        default=None,
        help="結束日期 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        default=None,
        help="資料根目錄"
    )
    
    args = parser.parse_args()
    
    # 設定日誌
    setup_logging()
    logger = logging.getLogger("Main")
    
    logger.info("=" * 60)
    logger.info("因子指數建構系統")
    logger.info("=" * 60)
    
    # 執行
    if args.index == "ALL":
        logger.info("計算所有指數...")
        results = run_all_indices(args.start, args.end, args.data)
    else:
        logger.info(f"計算 {args.index} 指數...")
        results = run_index(args.index, args.start, args.end, args.data)
    
    logger.info("=" * 60)
    logger.info("計算完成")
    logger.info("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
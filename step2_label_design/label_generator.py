"""
Step2 标签生成器模块
根据给定的周期(label_period)为每个交易对生成未来收益率标签。

标签定义：
    label = (close.shift(-label_period) - close) / close

说明：
    - 使用未来的收盘价与当前收盘价计算简单收益率
    - 底部最后 label_period 行将产生 NaN 标签，保存时由 utils.save_labeled_data* 清理
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class LabelGenerator:
    """标签生成器

    根据 label_period 计算未来收益率标签，并提供异步批量生成与验证方法。
    """

    def __init__(self, label_period: int = 1):
        if label_period <= 0:
            raise ValueError("label_period 必须为正整数")
        self.label_period = int(label_period)

    def generate_single(self, df: pd.DataFrame) -> pd.DataFrame:
        """为单个交易对数据生成标签列

        要求 df 包含列：['timestamp', 'open', 'high', 'low', 'close', 'volume']
        返回：在原DataFrame基础上新增 'label' 列
        """
        required_columns = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"数据缺少必要列: {missing}")

        # 按时间排序以确保计算顺序正确
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)

        # 未来收益率标签（简单收益率）
        future_close = df['close'].shift(-self.label_period)
        current_close = df['close']
        label = (future_close - current_close) / current_close

        out = df.copy()
        out['label'] = label
        return out

    async def generate_labels_async(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """异步批量生成标签

        Args:
            raw_data: {symbol: dataframe}

        Returns:
            {symbol: labeled_dataframe}
        """
        if not raw_data:
            logger.warning("raw_data 为空，无法生成标签")
            return {}

        loop = asyncio.get_event_loop()

        async def process_symbol(symbol: str, df: pd.DataFrame) -> Tuple[str, Optional[pd.DataFrame]]:
            try:
                labeled = await loop.run_in_executor(None, self.generate_single, df)
                return symbol, labeled
            except Exception as e:
                logger.error(f"{symbol}: 标签生成失败 - {e}")
                return symbol, None

        tasks = [process_symbol(symbol, df) for symbol, df in raw_data.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        labeled_data: Dict[str, pd.DataFrame] = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"标签生成异步异常: {result}")
                continue
            symbol, labeled_df = result
            if labeled_df is not None:
                labeled_data[symbol] = labeled_df

        logger.info(f"标签生成完成：成功 {len(labeled_data)}/{len(raw_data)}")
        return labeled_data

    def validate_labels(self, labeled_data: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
        """验证标签质量

        规则：
            - 存在 'label' 列
            - 非全 NaN（允许尾部 NaN）
            - 行数不大于原始数据（通常相等，因仅新增列）
        """
        results: Dict[str, bool] = {}
        for symbol, df in labeled_data.items():
            try:
                if df is None or df.empty:
                    results[symbol] = False
                    continue
                if 'label' not in df.columns:
                    results[symbol] = False
                    continue
                # 非全 NaN
                if df['label'].notna().sum() == 0:
                    results[symbol] = False
                    continue
                results[symbol] = True
            except Exception:
                results[symbol] = False
        return results
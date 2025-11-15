"""
VWMA_BIAS 成交量加权移动平均偏离度因子
计算当前收盘价相对于VWMA的偏离比率
"""
import pandas as pd
import numpy as np

def VWMA_BIAS(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    计算VWMA偏离度因子
    
    Args:
        df: 包含OHLCV数据的DataFrame，必须包含'close'和'volume'列
        period: VWMA计算周期，默认20
        
    Returns:
        VWMA偏离度的Series，计算公式: (当前价格 / VWMA) - 1
    """
    try:
        if 'close' not in df.columns:
            raise ValueError("DataFrame必须包含'close'列")
        
        if 'volume' not in df.columns:
            raise ValueError("DataFrame必须包含'volume'列")
        
        if len(df) < period:
            # 数据不足，返回NaN
            return pd.Series([np.nan] * len(df), index=df.index)
        
        close = df['close'].copy()
        volume = df['volume'].copy()
        
        # 计算成交量加权移动平均 (VWMA)
        # VWMA = sum(price * volume) / sum(volume) over period
        price_volume = close * volume
        
        # 使用滚动窗口计算VWMA
        vwma = price_volume.rolling(window=period).sum() / volume.rolling(window=period).sum()
        
        # 计算偏离度: (当前价格 / VWMA) - 1
        vwma_bias = (close / vwma) - 1
        
        # 处理除零和无效值
        vwma_bias = vwma_bias.replace([np.inf, -np.inf], np.nan)
        
        return vwma_bias
        
    except Exception as e:
        # 发生错误时返回NaN序列
        return pd.Series([np.nan] * len(df), index=df.index)

def calculate(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    VWMA偏离度因子计算的别名函数，与VWMA_BIAS函数功能相同
    
    Args:
        df: 包含OHLCV数据的DataFrame
        period: VWMA计算周期
        
    Returns:
        VWMA偏离度的Series
    """
    return VWMA_BIAS(df, period)
"""
EMA_BIAS 指数移动平均偏离度因子
计算当前收盘价相对于EMA的偏离比率
"""
import pandas as pd
import numpy as np

def EMA_BIAS(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    计算EMA偏离度因子
    
    Args:
        df: 包含OHLCV数据的DataFrame，必须包含'close'列
        period: EMA计算周期，默认20
        
    Returns:
        EMA偏离度的Series，计算公式: (当前价格 / EMA) - 1
    """
    try:
        if 'close' not in df.columns:
            raise ValueError("DataFrame必须包含'close'列")
        
        if len(df) < period:
            # 数据不足，返回NaN
            return pd.Series([np.nan] * len(df), index=df.index)
        
        close = df['close'].copy()
        
        # 计算指数移动平均
        ema = close.ewm(span=period, adjust=False).mean()
        
        # 计算偏离度: (当前价格 / EMA) - 1
        ema_bias = (close / ema) - 1
        
        # 处理除零和无效值
        ema_bias = ema_bias.replace([np.inf, -np.inf], np.nan)
        
        return ema_bias
        
    except Exception as e:
        # 发生错误时返回NaN序列
        return pd.Series([np.nan] * len(df), index=df.index)

def calculate(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    EMA偏离度因子计算的别名函数，与EMA_BIAS函数功能相同
    
    Args:
        df: 包含OHLCV数据的DataFrame
        period: EMA计算周期
        
    Returns:
        EMA偏离度的Series
    """
    return EMA_BIAS(df, period)
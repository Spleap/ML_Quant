"""
Momentum 动量因子
计算当前价格相对于n周期前价格的变化率
"""
import pandas as pd
import numpy as np

def MOM(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """
    计算动量因子
    
    Args:
        df: 包含OHLCV数据的DataFrame，必须包含'close'列
        period: 动量计算周期，默认10
        
    Returns:
        动量因子的Series，计算公式: (当前价格 / n周期前价格) - 1
    """
    try:
        if 'close' not in df.columns:
            raise ValueError("DataFrame必须包含'close'列")
        
        if len(df) < period:
            # 数据不足，返回NaN
            return pd.Series([np.nan] * len(df), index=df.index)
        
        close = df['close'].copy()
        
        # 计算n周期前的价格
        close_n_periods_ago = close.shift(period)
        
        # 计算动量: (当前价格 / n周期前价格) - 1
        momentum = (close / close_n_periods_ago) - 1
        
        # 处理除零和无效值
        momentum = momentum.replace([np.inf, -np.inf], np.nan)
        
        return momentum
        
    except Exception as e:
        # 发生错误时返回NaN序列
        return pd.Series([np.nan] * len(df), index=df.index)

def calculate(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """
    动量因子计算的别名函数，与MOM函数功能相同
    
    Args:
        df: 包含OHLCV数据的DataFrame
        period: 动量计算周期
        
    Returns:
        动量因子的Series
    """
    return MOM(df, period)
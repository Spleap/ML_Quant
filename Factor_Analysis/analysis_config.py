"""
Factor_Analysis 配置文件（单因子分析，图像仅保存不显示）

请在此文件配置本次分析的参数，运行 Factor_Analysis/main.py 即可完成整个流程。

配置说明：
- factor_config: 因子配置，支持两种格式：
  1. 字符串格式（旧版兼容）：直接指定完整因子列名，如 "MOM_10"、"RSI_14"
  2. 元组格式（推荐）：(因子名, 参数列表)，如 ("DDRF", [0.25, 0.25, 0.30, 0.20])
- time_range: 分析的时间范围，字典格式 {"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}。
  - 如不限制时间段，可将两者留空字符串或 None。
- remove_outliers: 是否在统计前进行去极值（按分位裁剪，排除极端值对统计的影响）。
- ir_window: IR 计算的滑动窗口长度（单位：时间截面个数，如 60、90、120）。
- use_parallel: 是否在数据读取、统计、IC/IR 计算中启用并行加速。
"""

# 因子配置 - 支持两种格式：
# 格式1（推荐）：元组格式 (因子名, 参数列表)
# 格式2（兼容）：字符串格式，直接指定完整列名
factor_config = ("RSI", [14])

# 自动生成因子列名的函数
def _generate_factor_column_name(config):
    """根据配置生成因子列名"""
    if isinstance(config, str):
        # 字符串格式，直接返回
        return config
    elif isinstance(config, (tuple, list)) and len(config) == 2:
        # 元组格式，生成列名
        factor_name, params = config
        if not params:
            return factor_name
        # 将参数转换为字符串并用下划线连接
        param_str = "_".join(str(p) for p in params)
        return f"{factor_name}_{param_str}"
    else:
        raise ValueError(f"不支持的因子配置格式: {config}")

# 生成实际的因子列名
single_factor = _generate_factor_column_name(factor_config)

# 分析时间段（如不限制，可将值设为空字符串或 None）
time_range = {
    "start_date": "2021-01-01",  # 例如："2023-01-01"
    "end_date": "2025-12-31"     # 例如："2023-12-31"
}

# 是否在统计前进行去极值（True/False）
remove_outliers = True

# IR 计算的滑动窗口长度（例如 60、90、120）
ir_window = 60

# 是否在处理过程中使用并行（数据加载、统计、IC/IR 计算）。True 表示启用。
use_parallel = True

# IC/IR 绘图时的采样频率（控制绘图点数，减少图像复杂度）
# 例如：0.1 表示仅绘制 1/10 的数据点，1.0 表示绘制全部数据点
sampling_frequency = 1

# 为兼容旧版本的配置项（可选）：提供别名 factor_name
# 外部代码如仍引用 factor_name，将使用 single_factor 的值
factor_name = single_factor
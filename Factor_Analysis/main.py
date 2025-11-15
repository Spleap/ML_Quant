"""
Factor_Analysis 主入口
按照 analysis_config.py 的配置，完成单因子分析与可视化：
1) 因子分布直方图
2) 因子分箱下标签平均值柱状图
3) 因子值 vs 标签 热力图
4) 截面 IC 时间序列图
5) 截面 IR 时间序列图

运行方式：
在 Factor_Analysis/analysis_config.py 配置好 single_factor、time_range、remove_outliers、ir_window 后，直接运行本文件即可：
python Factor_Analysis/main.py
"""
import sys
import os
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 本模块相对于项目根目录的路径处理
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent

# 确保项目根目录在路径中，以便绝对导入包
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 引入公共工具（使用包绝对导入，保证脚本运行）
from Factor_Analysis.utils.utils_common import (
    configure_chinese_font, safe_import_config
)
from Factor_Analysis.utils.data_loader import load_factor_data
from Factor_Analysis.utils.factor_stats import compute_factor_hist, compute_label_mean_by_bin
from Factor_Analysis.utils.heatmap_plot import create_factor_label_heatmap
from Factor_Analysis.utils.enhanced_plots import (
    create_enhanced_factor_histogram, create_enhanced_label_mean_plot, create_enhanced_heatmap
)
from Factor_Analysis.utils.ic_ir_plot import compute_ic_series, compute_ir_series, plot_ic_series, plot_ir_series
from Factor_Analysis.utils.result_saver import save_analysis_results

# 导入用户分析配置
from Factor_Analysis.analysis_config import single_factor, time_range, remove_outliers, ir_window, use_parallel, sampling_frequency


logger = logging.getLogger(__name__)


def _setup_logging():
    cfg = safe_import_config()
    # 优先使用项目内置日志配置
    try:
        cfg.setup_logging()
    except Exception:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    _setup_logging()
    # 运行过程中不显示任何 warnings
    warnings.filterwarnings("ignore")
    configure_chinese_font()
    # 强制不显示图像：切换到无 GUI 后端 Agg
    try:
        plt.switch_backend('Agg')
        logger.info("已关闭图像显示（使用 Agg 后端），仅保存图像")
    except Exception:
        pass

    logger.info("=" * 60)
    logger.info("开始执行 Factor_Analysis 单因子分析")
    # 为兼容旧代码，使用本地变量 factor_name
    factor_name = single_factor
    logger.info(f"分析因子: {factor_name}")
    logger.info(f"IR窗口长度: {ir_window}")
    logger.info("=" * 60)

    cfg = safe_import_config()
    base_output = Path(CURRENT_DIR) / "Output"
    base_output.mkdir(parents=True, exist_ok=True)

    # 解析时间段配置
    start_ts = None
    end_ts = None
    try:
        if isinstance(time_range, dict):
            s = time_range.get("start_date")
            e = time_range.get("end_date")
            import pandas as pd
            start_ts = pd.to_datetime(s) if s else None
            end_ts = pd.to_datetime(e) if e else None
    except Exception:
        start_ts = None
        end_ts = None

    # 1) 数据读取
    logger.info("步骤1：加载特征数据（并行）...")
    if start_ts or end_ts:
        logger.info(f"时间段: {start_ts if start_ts else '-∞'} ~ {end_ts if end_ts else '+∞'}")
    df, load_stats = load_factor_data(factor_name=factor_name, use_parallel=use_parallel, time_range=(start_ts, end_ts))
    logger.info(f"检查文件数: {load_stats['files_checked']}，有效文件数: {load_stats['files_used']}，总行数: {load_stats['rows_total']}")
    if not load_stats['symbols']:
        logger.info("没有有效的交易对数据，分析流程结束")
    else:
        logger.info(f"交易对数量: {len(load_stats['symbols'])}")

    if df.empty:
        logger.error("未能加载到有效数据，终止分析")
        return

    # 列名统一
    label_col = 'label'
    time_col = 'timestamp' if 'timestamp' in df.columns else None

    # 2) 因子统计
    logger.info("步骤2：计算因子分布与标签均值统计（使用自适应分箱）...")
    max_workers = getattr(cfg, 'MAX_WORKERS', 8)
    hist_data = compute_factor_hist(df, factor_col=factor_name, use_parallel=use_parallel, 
                                   max_workers=max_workers, remove_outliers=remove_outliers, 
                                   binning_method='auto')
    mean_data = compute_label_mean_by_bin(df, factor_col=factor_name, label_col=label_col, 
                                         bin_edges=hist_data['bin_edges'])

    # 构建图1：增强的因子值分布直方图
    logger.info("生成图1：增强的因子值分布直方图...")
    fig1 = create_enhanced_factor_histogram(hist_data, factor_name, remove_outliers)

    # 构建图2：增强的因子分箱标签均值图
    logger.info("生成图2：增强的因子分箱标签均值图...")
    fig2 = create_enhanced_label_mean_plot(mean_data, factor_name, remove_outliers)

    # 3) 热力图：因子值 vs 标签
    logger.info("步骤3：生成因子值 vs 标签 热力图...")
    fig3 = create_factor_label_heatmap(df[factor_name].values, df[label_col].values, title=f"图3：{factor_name} 因子值 vs 标签 热力图")

    # 4) IC/IR 计算与绘制
    logger.info("步骤4：计算截面 IC/IR 并绘图...")
    if time_col is None:
        logger.info("缺少时间列，无法计算 IC/IR。将跳过图4、图5。")
        fig4 = plt.figure(); plt.close(fig4)
        fig5 = plt.figure(); plt.close(fig5)
        ic_series = pd.Series(dtype=float)
        ir_series = pd.Series(dtype=float)
    else:
        ic_series = compute_ic_series(df, factor_col=factor_name, label_col=label_col, time_col=time_col,
                                      use_parallel=use_parallel, max_workers=max_workers)
        ir_series = compute_ir_series(ic_series, window=ir_window)
        fig4 = plot_ic_series(ic_series, title="图4：截面 IC 随时间（Spearman）", sampling_frequency=sampling_frequency)
        fig5 = plot_ir_series(ir_series, title=f"图5：截面 IR 随时间（窗口={ir_window}）", sampling_frequency=sampling_frequency)

    # 5) 仅保存图像与日志，不显示
    logger.info("步骤5：仅保存图像与日志，不显示图像...")

    # 6) 保存结果
    logger.info("步骤6：保存图像与日志...")
    # 简要日志内容
    # 时间范围（若有时间列）
    time_span = "N/A"
    if time_col is not None and not df['timestamp'].empty:
        time_span = f"{df['timestamp'].min()} ~ {df['timestamp'].max()}"

    log_lines = [
        f"分析因子: {factor_name}",
        f"IR窗口长度: {ir_window}",
        f"使用并行: {use_parallel}",
        f"是否去极值: {remove_outliers}",
        f"配置的时间段: {start_ts if start_ts else '-∞'} ~ {end_ts if end_ts else '+∞'}",
        f"读取文件数: {load_stats['files_checked']}",
        f"有效文件数: {load_stats['files_used']}",
        f"交易对数量: {len(load_stats['symbols'])}",
        f"总样本数: {load_stats['rows_total']}",
        f"数据实际时间范围: {time_span}",
        f"直方图分箱数: {hist_data['n_bins']}",
        f"每箱样本数（示例前5）: {hist_data['counts'][:5].tolist()}",
        f"标签均值（示例前5）: {pd.Series(mean_data['label_mean']).dropna()[:5].round(6).tolist()}",
        f"IC统计: 非 NaN 点 {int(np.isfinite(ic_series.values).sum())} / {len(ic_series)}",
        f"IR统计: 非 NaN 点 {int(np.isfinite(ir_series.values).sum())} / {len(ir_series)}",
    ]

    figures = {
        'fig1_factor_hist': fig1,
        'fig2_label_mean': fig2,
        'fig3_heatmap': fig3,
        'fig4_ic': fig4,
        'fig5_ir': fig5,
    }

    output_dir = save_analysis_results(base_output=base_output, factor_name=factor_name, figures=figures, log_lines=log_lines)
    logger.info(f"本次分析结果已保存至：{output_dir}")
    logger.info("分析流程完成！")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("用户中断执行")
    except Exception as e:
        logger.error(f"执行失败：{e}")
        raise
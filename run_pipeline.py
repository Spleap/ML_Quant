"""
一键运行完整Pipeline：Step1 → Step2 → Step3 → 因子分析

使用说明：
- 只需修改项目根目录下的 config.py，即可控制运行行为：
  0) RUN_STEP1/RUN_STEP2/RUN_STEP3/RUN_FACTOR_ANALYSIS：各步骤的运行开关
  1) FACTORS_TO_COMPUTE：Step3要计算的因子列表（格式：[("因子名", [参数列表])...]）
  2) ANALYZE_ALL_FACTORS：因子分析阶段是否分析所有因子；
     - True：依次分析 FACTORS_TO_COMPUTE 中的所有因子
     - False：只分析 FACTORS_TO_COMPUTE 列表中的第一个因子

运行方式：
  在项目根目录执行：
    python run_pipeline.py

注意：
- 本脚本在Windows环境下自动设置异步事件循环策略，兼容各步骤的异步处理。
- 因子分析结果会保存到 Factor_Analysis/Output 目录（图像与日志），不弹窗显示。
"""
import os
import sys
import logging
import importlib
import asyncio
from pathlib import Path

# 项目根目录加入路径
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 读取主配置
from config import (
    setup_logging,
    ANALYZE_ALL_FACTORS,
    FACTORS_TO_COMPUTE,
    RUN_STEP1,
    RUN_STEP2,
    RUN_STEP3,
    RUN_FACTOR_ANALYSIS,
)

logger = logging.getLogger(__name__)


def _set_windows_event_loop_policy():
    """在Windows上设置兼容的事件循环策略"""
    try:
        if sys.platform.startswith('win'):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            logger.info("已设置Windows异步事件循环策略")
    except Exception:
        # 在非Windows或低版本环境下安全忽略
        pass


def run_step1():
    """运行 Step1：原始数据处理"""
    logger.info("=" * 60)
    logger.info("开始执行 Step1：原始数据处理 (CSV → Parquet)")
    logger.info("=" * 60)
    try:
        from step1_data_processing import main as step1_main
        _set_windows_event_loop_policy()
        asyncio.run(step1_main.main())
        logger.info("Step1 执行完成")
    except Exception as e:
        logger.error(f"Step1 执行失败：{e}")
        raise


def run_step2():
    """运行 Step2：标签设计"""
    logger.info("=" * 60)
    logger.info("开始执行 Step2：标签设计")
    logger.info("=" * 60)
    try:
        from step2_label_design import main as step2_main
        # Step2 自带 run_step2 封装
        step2_main.run_step2()
        logger.info("Step2 执行完成")
    except Exception as e:
        logger.error(f"Step2 执行失败：{e}")
        raise


def run_step3():
    """运行 Step3：特征工程（因子计算与保存）"""
    logger.info("=" * 60)
    logger.info("开始执行 Step3：特征工程（计算因子并保存）")
    logger.info("=" * 60)
    try:
        from step3_feature_engineering import main as step3_main
        # Step3 自带 run_step3 封装（含Windows事件循环策略）
        step3_main.run_step3()
        logger.info("Step3 执行完成")
    except Exception as e:
        logger.error(f"Step3 执行失败：{e}")
        raise


def _analysis_set_factor(factor_tuple):
    """将因子分析配置切换为指定因子

    Args:
        factor_tuple: (因子名, [参数列表])
    """
    try:
        from Factor_Analysis import analysis_config as an_cfg
        # 设置当前分析的因子
        an_cfg.factor_config = factor_tuple
        # 更新 single_factor 列名（如 "RSI_14"）
        if hasattr(an_cfg, '_generate_factor_column_name'):
            an_cfg.single_factor = an_cfg._generate_factor_column_name(an_cfg.factor_config)
        else:
            # 兜底：直接用因子名或因子名_参数
            name, params = factor_tuple
            an_cfg.single_factor = name if not params else f"{name}_{'_'.join(map(str, params))}"
        logger.info(f"切换因子分析配置为：{an_cfg.single_factor}")
        return True
    except Exception as e:
        logger.error(f"设置因子分析配置失败：{e}")
        return False


def run_factor_analysis():
    """运行因子分析：根据 ANALYZE_ALL_FACTORS 决定分析范围"""
    logger.info("=" * 60)
    logger.info("开始执行 因子分析：生成图像与日志（自动保存）")
    logger.info("=" * 60)

    # 准备待分析因子列表
    factors_for_analysis = []
    if ANALYZE_ALL_FACTORS:
        factors_for_analysis = FACTORS_TO_COMPUTE[:]
        logger.info(f"将依次分析所有因子，共 {len(factors_for_analysis)} 个")
    else:
        if not FACTORS_TO_COMPUTE:
            logger.error("FACTORS_TO_COMPUTE 为空，无法进行单因子分析")
            return
        factors_for_analysis = [FACTORS_TO_COMPUTE[0]]
        logger.info(f"仅分析第一个配置的因子：{factors_for_analysis[0]}")

    # 逐个因子运行分析主程序
    for idx, factor_tuple in enumerate(factors_for_analysis, start=1):
        try:
            logger.info("-" * 60)
            logger.info(f"开始分析第 {idx}/{len(factors_for_analysis)} 个因子：{factor_tuple}")
            # 更新分析配置
            if not _analysis_set_factor(factor_tuple):
                logger.warning("切换配置失败，跳过该因子")
                continue
            # 重新加载分析主程序，使其绑定新的配置
            import Factor_Analysis.main as fa_main
            importlib.reload(fa_main)
            # 执行分析
            fa_main.main()
            logger.info(f"因子 {factor_tuple} 分析完成")
        except Exception as e:
            logger.error(f"因子 {factor_tuple} 分析失败：{e}")
            # 不中断整体流程，继续下一个因子
            continue

    logger.info("因子分析阶段完成")


def main():
    # 统一日志配置
    setup_logging()
    logger.info("== ML_Quant 一键Pipeline 开始 ==")

    # 显示运行开关状态
    logger.info(
        f"步骤开关状态：Step1={RUN_STEP1}, Step2={RUN_STEP2}, Step3={RUN_STEP3}, FactorAnalysis={RUN_FACTOR_ANALYSIS}"
    )

    # 按开关依次执行各步骤
    if RUN_STEP1:
        run_step1()
    else:
        logger.info("跳过 Step1（原始数据处理）")

    if RUN_STEP2:
        run_step2()
    else:
        logger.info("跳过 Step2（标签设计）")

    if RUN_STEP3:
        run_step3()
    else:
        logger.info("跳过 Step3（特征工程/因子计算）")

    if RUN_FACTOR_ANALYSIS:
        run_factor_analysis()
    else:
        logger.info("跳过 因子分析")

    logger.info("== ML_Quant 一键Pipeline 完成 ==")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("用户中断执行")
    except Exception as e:
        logger.error(f"Pipeline 执行失败：{e}")
        raise
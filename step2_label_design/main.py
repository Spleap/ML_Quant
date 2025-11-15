"""
Step2 主执行文件
标签设计：在原始数据基础上增加标签列
每次运行前清空labeled_data目录
"""
import asyncio
import logging
import sys
import os
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    LABELED_DATA_PATH, LABEL_PERIOD, clear_directory, ensure_data_dirs,
    LOG_LEVEL, LOG_FORMAT
)
from step2_label_design.label_generator import LabelGenerator
from step2_label_design.utils import (
    load_raw_data_async, save_labeled_data_async, validate_data_consistency,
    log_processing_summary, cleanup_intermediate_files
)

# 配置日志
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

async def main():
    """主执行函数"""
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("开始执行 Step2: 标签设计")
    logger.info(f"标签周期: {LABEL_PERIOD}")
    logger.info("=" * 60)
    
    try:
        # 确保数据目录存在
        ensure_data_dirs()
        
        # 清空labeled_data目录
        logger.info("清空labeled_data目录...")
        clear_directory(LABELED_DATA_PATH)
        
        # 异步加载原始数据
        logger.info("开始加载原始数据...")
        raw_data = await load_raw_data_async()
        
        if not raw_data:
            logger.error("未能加载任何原始数据，请确保Step1已正确执行")
            return
        
        logger.info(f"成功加载 {len(raw_data)} 个交易对的原始数据")
        
        # 验证原始数据质量
        valid_symbols = []
        invalid_symbols = []
        
        for symbol, df in raw_data.items():
            if df.empty:
                logger.warning(f"{symbol}: 原始数据为空")
                invalid_symbols.append(symbol)
                continue
            
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"{symbol}: 缺少必要列 {missing_columns}")
                invalid_symbols.append(symbol)
                continue
            
            if len(df) < LABEL_PERIOD + 10:  # 需要足够的数据来计算标签
                logger.warning(f"{symbol}: 数据量不足 ({len(df)} 行)")
                invalid_symbols.append(symbol)
                continue
            
            valid_symbols.append(symbol)
        
        if invalid_symbols:
            logger.warning(f"跳过 {len(invalid_symbols)} 个无效交易对: {invalid_symbols[:5]}...")
            # 移除无效数据
            for symbol in invalid_symbols:
                raw_data.pop(symbol, None)
        
        if not raw_data:
            logger.error("没有有效的原始数据可处理")
            return
        
        logger.info(f"有效交易对数: {len(valid_symbols)}")
        
        # 创建标签生成器
        label_generator = LabelGenerator(label_period=LABEL_PERIOD)
        
        # 异步生成标签
        logger.info("开始生成标签...")
        labeled_data = await label_generator.generate_labels_async(raw_data)
        
        if not labeled_data:
            logger.error("标签生成失败")
            return
        
        # 验证标签质量
        logger.info("验证标签质量...")
        validation_results = label_generator.validate_labels(labeled_data)
        
        valid_labeled_data = {}
        failed_validation = []
        
        for symbol, is_valid in validation_results.items():
            if is_valid and symbol in labeled_data:
                valid_labeled_data[symbol] = labeled_data[symbol]
            else:
                failed_validation.append(symbol)
        
        if failed_validation:
            logger.warning(f"标签验证失败的交易对: {failed_validation[:5]}...")
        
        if not valid_labeled_data:
            logger.error("没有通过验证的标签数据")
            return
        
        logger.info(f"通过验证的交易对数: {len(valid_labeled_data)}")
        
        # 验证数据一致性
        logger.info("验证数据一致性...")
        consistency_check = validate_data_consistency(raw_data, valid_labeled_data)
        
        if not consistency_check:
            logger.warning("数据一致性检查未完全通过，但继续处理")
        
        # 异步保存标签数据
        logger.info("开始保存标签数据...")
        save_results = await save_labeled_data_async(valid_labeled_data)
        
        # 统计保存结果
        successful_saves = sum(save_results.values())
        failed_saves = len(save_results) - successful_saves
        
        if failed_saves > 0:
            failed_symbols = [symbol for symbol, success in save_results.items() if not success]
            logger.warning(f"保存失败的交易对: {failed_symbols[:5]}...")
        
        # 验证输出文件
        output_files = list(LABELED_DATA_PATH.glob("*.parquet"))
        logger.info(f"生成的标签数据文件数量: {len(output_files)}")
        
        if len(output_files) != successful_saves:
            logger.warning("输出文件数量与保存成功数量不匹配")
        
        # 清理中间文件
        cleanup_intermediate_files()
        
        # 记录处理汇总
        total_symbols = len(raw_data)
        successful_symbols = successful_saves
        failed_symbols = total_symbols - successful_symbols
        
        log_processing_summary(total_symbols, successful_symbols, failed_symbols)
        
        end_time = time.time()
        logger.info(f"Step2 总耗时: {end_time - start_time:.2f} 秒")
        
        # 最终验证
        if successful_symbols == 0:
            logger.error("Step2 执行失败：没有成功处理任何交易对")
            return
        
        logger.info("Step2 执行成功完成")
        
    except Exception as e:
        logger.error(f"Step2 执行失败: {str(e)}")
        raise
    
    finally:
        logger.info("Step2 执行结束")

def run_step2():
    """运行Step2的便捷函数"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("用户中断执行")
    except Exception as e:
        logger.error(f"Step2 运行异常: {str(e)}")
        raise

if __name__ == "__main__":
    run_step2()
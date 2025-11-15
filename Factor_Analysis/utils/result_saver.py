"""
结果保存模块
在 Factor_Analysis/Output/ 下创建独立子文件夹（包含时间戳和因子名），保存图1~图5和因子分析日志。
不保存中间测试文件或大图。
硬件加速优化版：支持内存管理、图像压缩、批量处理。
"""
import logging
import gc
import psutil
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 强制使用无GUI后端

from .utils_common import get_timestamp, ensure_dir, safe_import_config


logger = logging.getLogger(__name__)


def _check_memory_usage() -> float:
    """检查当前内存使用率"""
    try:
        return psutil.virtual_memory().percent / 100.0
    except Exception:
        return 0.0


def _trigger_gc_if_needed(memory_threshold: float = 0.7):
    """如果内存使用超过阈值，触发垃圾回收"""
    if _check_memory_usage() > memory_threshold:
        gc.collect()
        logger.debug(f"内存使用超过{memory_threshold*100:.1f}%，已触发垃圾回收")


def create_output_subdir(base_dir: Path, factor_name: str) -> Path:
    ts = get_timestamp()
    subdir = base_dir / f"{ts}_{factor_name}"
    ensure_dir(subdir)
    return subdir


def save_figures(figs: Dict[str, plt.Figure], output_dir: Path) -> Dict[str, str]:
    """保存图像到指定目录，返回 {name: file_path} 映射。硬件加速优化版。"""
    # 应用硬件加速配置
    cfg = safe_import_config()
    memory_limit_gb = getattr(cfg, 'MEMORY_LIMIT_GB', 8.0)
    gc_threshold = getattr(cfg, 'GC_THRESHOLD', 0.7)
    
    # 优化的图像保存参数
    save_params = {
        'dpi': 150,  # 适中的分辨率，平衡质量和文件大小
        'bbox_inches': 'tight',
        'pad_inches': 0.1,
        'format': 'png',
        'facecolor': 'white',
        'edgecolor': 'none',
        'transparent': False
    }
    
    saved: Dict[str, str] = {}
    total_figs = len(figs)
    
    for i, (name, fig) in enumerate(figs.items(), 1):
        try:
            file_path = output_dir / f"{name}.png"
            
            # 保存图像
            fig.savefig(file_path, **save_params)
            saved[name] = str(file_path)
            
            # 立即关闭图像并释放内存
            try:
                plt.close(fig)
                del fig  # 显式删除引用
            except Exception:
                pass
            
            # 定期检查内存并触发垃圾回收
            if i % 2 == 0 or i == total_figs:  # 每2个图像或最后一个图像后检查
                _trigger_gc_if_needed(gc_threshold)
            
            logger.debug(f"已保存图像 {name} ({i}/{total_figs})")
            
        except Exception as e:
            logger.error(f"保存图像失败 {name}: {e}")
            # 即使保存失败也要尝试关闭图像
            try:
                plt.close(fig)
                del fig
            except Exception:
                pass
    
    # 最终内存清理
    _trigger_gc_if_needed(0.5)  # 更积极的清理
    logger.info(f"已保存 {len(saved)}/{total_figs} 个图像，当前内存使用: {_check_memory_usage()*100:.1f}%")
    
    return saved


def save_log(output_dir: Path, content_lines: Sequence[str]) -> str:
    """保存分析日志为 txt 文件，返回文件路径。"""
    log_path = output_dir / "analysis_log.txt"
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            for line in content_lines:
                f.write(line.rstrip('\n') + '\n')
        return str(log_path)
    except Exception as e:
        logger.error(f"保存分析日志失败：{e}")
        return str(log_path)


def save_analysis_results(base_output: Path, factor_name: str, figures: Dict[str, plt.Figure],
                          log_lines: Sequence[str]) -> Path:
    """创建输出子目录并保存所有结果，返回子目录路径。硬件加速优化版。"""
    # 应用硬件加速配置
    cfg = safe_import_config()
    gc_threshold = getattr(cfg, 'GC_THRESHOLD', 0.7)
    
    logger.info(f"开始保存分析结果，图像数量: {len(figures)}")
    initial_memory = _check_memory_usage()
    
    try:
        # 创建输出目录
        subdir = create_output_subdir(base_output, factor_name)
        
        # 保存图像（已优化内存管理）
        saved_figures = save_figures(figures, subdir)
        
        # 清理图像引用
        figures.clear()
        _trigger_gc_if_needed(gc_threshold)
        
        # 保存日志
        log_path = save_log(subdir, log_lines)
        
        # 最终内存清理
        _trigger_gc_if_needed(0.5)
        final_memory = _check_memory_usage()
        
        logger.info(f"分析结果保存完成，目录: {subdir}")
        logger.info(f"内存使用变化: {initial_memory*100:.1f}% -> {final_memory*100:.1f}%")
        
        return subdir
        
    except Exception as e:
        logger.error(f"保存分析结果失败: {e}")
        # 错误时也要清理内存
        try:
            figures.clear()
            _trigger_gc_if_needed(0.5)
        except Exception:
            pass
        raise
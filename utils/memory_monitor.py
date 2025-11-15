"""
内存监控工具
用于监控和管理大规模数据处理时的内存使用
"""
import gc
import psutil
import logging
from typing import Optional
from functools import wraps
import time

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self, memory_limit_gb: float = 20, gc_threshold: float = 0.8):
        """
        初始化内存监控器
        
        Args:
            memory_limit_gb: 内存使用上限（GB）
            gc_threshold: 触发垃圾回收的内存使用率阈值
        """
        self.memory_limit_bytes = memory_limit_gb * 1024 ** 3
        self.gc_threshold = gc_threshold
        self.process = psutil.Process()
        
    def get_memory_usage(self) -> dict:
        """获取当前内存使用情况"""
        memory_info = self.process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            'process_memory_mb': memory_info.rss / 1024 ** 2,
            'process_memory_gb': memory_info.rss / 1024 ** 3,
            'system_memory_percent': system_memory.percent,
            'system_available_gb': system_memory.available / 1024 ** 3,
            'memory_limit_gb': self.memory_limit_bytes / 1024 ** 3
        }
    
    def check_memory_usage(self) -> bool:
        """检查内存使用是否超过阈值"""
        memory_info = self.get_memory_usage()
        
        # 检查进程内存是否超过限制
        if memory_info['process_memory_gb'] > memory_info['memory_limit_gb']:
            logger.warning(f"进程内存使用超过限制: {memory_info['process_memory_gb']:.2f}GB > {memory_info['memory_limit_gb']:.2f}GB")
            return False
            
        # 检查系统内存使用率
        if memory_info['system_memory_percent'] > self.gc_threshold * 100:
            logger.warning(f"系统内存使用率过高: {memory_info['system_memory_percent']:.1f}%")
            return False
            
        return True
    
    def force_gc(self):
        """强制垃圾回收"""
        logger.info("执行垃圾回收...")
        collected = gc.collect()
        logger.info(f"垃圾回收完成，回收对象数: {collected}")
        
        # 记录回收后的内存使用
        memory_info = self.get_memory_usage()
        logger.info(f"垃圾回收后内存使用: {memory_info['process_memory_gb']:.2f}GB")
    
    def auto_gc_if_needed(self):
        """根据内存使用情况自动执行垃圾回收"""
        if not self.check_memory_usage():
            self.force_gc()
    
    def log_memory_status(self):
        """记录当前内存状态"""
        memory_info = self.get_memory_usage()
        logger.info(
            f"内存状态 - 进程: {memory_info['process_memory_gb']:.2f}GB, "
            f"系统使用率: {memory_info['system_memory_percent']:.1f}%, "
            f"系统可用: {memory_info['system_available_gb']:.2f}GB"
        )

def memory_monitor_decorator(monitor: Optional[MemoryMonitor] = None):
    """
    内存监控装饰器
    在函数执行前后监控内存使用情况
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if monitor is None:
                return func(*args, **kwargs)
                
            # 执行前检查内存
            monitor.log_memory_status()
            monitor.auto_gc_if_needed()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # 执行后再次检查内存
                monitor.auto_gc_if_needed()
                monitor.log_memory_status()
                
        return wrapper
    return decorator

# 全局内存监控器实例
_global_monitor: Optional[MemoryMonitor] = None

def get_global_monitor() -> Optional[MemoryMonitor]:
    """获取全局内存监控器"""
    return _global_monitor

def init_global_monitor(memory_limit_gb: float = 20, gc_threshold: float = 0.8):
    """初始化全局内存监控器"""
    global _global_monitor
    _global_monitor = MemoryMonitor(memory_limit_gb, gc_threshold)
    logger.info(f"内存监控器已初始化 - 限制: {memory_limit_gb}GB, GC阈值: {gc_threshold}")
    return _global_monitor

def auto_gc_if_needed():
    """如果有全局监控器，则自动执行垃圾回收检查"""
    if _global_monitor:
        _global_monitor.auto_gc_if_needed()

def time_perf_decorator(name: Optional[str] = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                dt_ms = (time.perf_counter() - t0) * 1000.0
                label = name or func.__name__
                logger.info(f"{label} 耗时 {dt_ms:.3f} ms")
        return wrapper
    return decorator

if __name__ == "__main__":
    # 测试内存监控工具
    print("=== 内存监控工具测试 ===")
    
    # 初始化监控器
    monitor = MemoryMonitor(memory_limit_gb=20, gc_threshold=0.8)
    print("✓ 内存监控器初始化成功")
    
    # 获取内存使用情况
    memory_info = monitor.get_memory_usage()
    print(f"当前进程内存: {memory_info['process_memory_gb']:.2f}GB")
    print(f"系统内存使用率: {memory_info['system_memory_percent']:.1f}%")
    print(f"系统可用内存: {memory_info['system_available_gb']:.2f}GB")
    
    # 测试内存检查
    is_ok = monitor.check_memory_usage()
    print(f"内存使用状态: {'正常' if is_ok else '超限'}")
    
    # 测试全局监控器
    init_global_monitor(memory_limit_gb=20, gc_threshold=0.8)
    print("✓ 全局内存监控器初始化成功")
    
    print("✓ 内存监控工具测试完成")

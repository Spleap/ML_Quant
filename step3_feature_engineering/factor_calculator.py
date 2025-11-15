"""
Step3 因子计算器
动态调用factors目录下的因子计算函数
"""
import pandas as pd
import numpy as np
import logging
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    FACTORS_TO_COMPUTE, MAX_WORKERS, FACTOR_CALC_WORKERS, 
    ENABLE_MEMORY_MONITORING, MEMORY_LIMIT_GB, GC_THRESHOLD,
    ENABLE_PARALLEL_FACTOR_CALC, ENABLE_VECTORIZED_OPERATIONS,
    PRELOAD_DATA_TO_MEMORY
)

# 导入内存监控工具
try:
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils'))
    from memory_monitor import MemoryMonitor, memory_monitor_decorator, init_global_monitor, time_perf_decorator
    MEMORY_MONITOR_AVAILABLE = True
except ImportError:
    logger.warning("内存监控工具不可用，将跳过内存监控功能")
    MEMORY_MONITOR_AVAILABLE = False
    MemoryMonitor = None
    memory_monitor_decorator = lambda x: lambda f: f

logger = logging.getLogger(__name__)

class FactorCalculator:
    """因子计算器类"""
    
    def __init__(self, factors_dir: Path = None, max_workers: int = MAX_WORKERS, factors_to_compute: List[Tuple[str, List]] = None):
        self.factors_dir = factors_dir or Path(__file__).parent / "factors"
        self.max_workers = max_workers
        self.factor_calc_workers = FACTOR_CALC_WORKERS
        self.enable_parallel_calc = ENABLE_PARALLEL_FACTOR_CALC
        self.enable_vectorized = ENABLE_VECTORIZED_OPERATIONS
        self.preload_data = PRELOAD_DATA_TO_MEMORY
        
        # 初始化线程池
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        if self.enable_parallel_calc:
            self.factor_executor = ThreadPoolExecutor(max_workers=self.factor_calc_workers)
        else:
            self.factor_executor = None
            
        # 初始化内存监控
        self.memory_monitor = None
        if MEMORY_MONITOR_AVAILABLE and ENABLE_MEMORY_MONITORING:
            self.memory_monitor = init_global_monitor(MEMORY_LIMIT_GB, GC_THRESHOLD)
            logger.info("内存监控已启用")
        
        self.available_factors = {}
        self.factors_to_compute = factors_to_compute or FACTORS_TO_COMPUTE
        self._load_required_factors()
    
    def _load_required_factors(self):
        """只加载config.py中指定的因子"""
        try:
            if not self.factors_dir.exists():
                logger.error(f"因子目录不存在: {self.factors_dir}")
                return
            
            # 添加factors目录到Python路径
            if str(self.factors_dir) not in sys.path:
                sys.path.insert(0, str(self.factors_dir))
            
            # 获取需要加载的因子名称
            required_factor_names = {factor_name for factor_name, _ in self.factors_to_compute}
            
            if not required_factor_names:
                logger.warning("没有指定需要计算的因子")
                return
            
            # 只加载需要的因子
            for factor_name in required_factor_names:
                py_file = self.factors_dir / f"{factor_name}.py"
                
                if not py_file.exists():
                    logger.error(f"因子文件不存在: {py_file}")
                    continue
                
                try:
                    # 动态导入模块
                    spec = importlib.util.spec_from_file_location(factor_name, py_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # 查找因子计算函数
                    factor_function = self._find_factor_function(module, factor_name)
                    
                    if factor_function:
                        self.available_factors[factor_name] = factor_function
                        logger.info(f"加载因子: {factor_name}")
                    else:
                        logger.warning(f"在模块 {factor_name} 中未找到有效的因子函数")
                        
                except Exception as e:
                    logger.error(f"加载因子模块失败 {py_file}: {str(e)}")
            
            logger.info(f"成功加载 {len(self.available_factors)} 个因子: {list(self.available_factors.keys())}")
            
        except Exception as e:
            logger.error(f"加载因子失败: {str(e)}")
    
    def _find_factor_function(self, module: Any, module_name: str) -> Optional[Callable]:
        """
        在模块中查找因子计算函数
        
        Args:
            module: 导入的模块
            module_name: 模块名称
            
        Returns:
            因子计算函数或None
        """
        try:
            # 优先查找与模块名同名的函数
            if hasattr(module, module_name):
                func = getattr(module, module_name)
                if callable(func):
                    return func
            
            # 查找名为calculate的函数
            if hasattr(module, 'calculate'):
                func = getattr(module, 'calculate')
                if callable(func):
                    return func
            
            # 查找第一个可调用的非私有函数
            for name, obj in inspect.getmembers(module):
                if (callable(obj) and 
                    not name.startswith('_') and 
                    inspect.isfunction(obj) and
                    obj.__module__ == module.__name__):
                    return obj
            
            return None
            
        except Exception as e:
            logger.error(f"查找因子函数失败 {module_name}: {str(e)}")
            return None
    
    def get_available_factors(self) -> List[str]:
        """获取所有可用因子列表"""
        return list(self.available_factors.keys())
    
    def validate_factor_request(self, factor_name: str, params: List) -> bool:
        """
        验证因子请求是否有效
        
        Args:
            factor_name: 因子名称
            params: 参数列表
            
        Returns:
            是否有效
        """
        if factor_name not in self.available_factors:
            logger.warning(f"因子 {factor_name} 不可用")
            return False
        
        try:
            # 检查函数签名
            func = self.available_factors[factor_name]
            sig = inspect.signature(func)
            
            # 基本验证：函数应该接受DataFrame和参数
            param_names = list(sig.parameters.keys())
            
            if len(param_names) < 1:
                logger.warning(f"因子函数 {factor_name} 参数不足")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"验证因子请求失败 {factor_name}: {str(e)}")
            return False
    
    @time_perf_decorator()
    @memory_monitor_decorator()
    def calculate_single_factor(self, df: pd.DataFrame, factor_name: str, params: List, symbol: str) -> Optional[pd.Series]:
        """
        计算单个因子
        
        Args:
            df: 输入数据DataFrame
            factor_name: 因子名称
            params: 参数列表
            symbol: 交易对符号
            
        Returns:
            因子值序列或None
        """
        try:
            if not self.validate_factor_request(factor_name, params):
                return None
            
            func = self.available_factors[factor_name]
            
            # 调用因子计算函数
            if params:
                result = func(df, *params)
            else:
                result = func(df)
            
            # 确保返回Series
            if isinstance(result, pd.Series):
                factor_series = result
            elif isinstance(result, (list, np.ndarray)):
                factor_series = pd.Series(result, index=df.index)
            elif isinstance(result, (int, float)):
                factor_series = pd.Series([result] * len(df), index=df.index)
            else:
                logger.error(f"{symbol}: 因子 {factor_name} 返回类型无效: {type(result)}")
                return None
            
            # 基本验证
            if len(factor_series) != len(df):
                logger.warning(f"{symbol}: 因子 {factor_name} 长度不匹配")
                # 尝试对齐
                factor_series = factor_series.reindex(df.index)
            
            # 生成因子列名
            if params:
                factor_column_name = f"{factor_name}_{'_'.join(map(str, params))}"
            else:
                factor_column_name = factor_name
            
            factor_series.name = factor_column_name
            
            logger.info(f"{symbol}: 成功计算因子 {factor_column_name}")
            return factor_series
            
        except Exception as e:
            logger.error(f"{symbol}: 计算因子失败 {factor_name}: {str(e)}")
            return None
    


    @time_perf_decorator()
    @memory_monitor_decorator()
    def calculate_factors_for_symbol(self, df: pd.DataFrame, symbol: str, 
                                   factors_to_compute: List[Tuple[str, List]] = None) -> pd.DataFrame:
        """
        为单个交易对计算所有指定因子
        
        Args:
            df: 输入数据DataFrame
            symbol: 交易对符号
            factors_to_compute: 要计算的因子列表，格式: [(factor_name, params), ...]
            
        Returns:
            添加因子列的DataFrame，删除因子列有NaN的行
        """
        try:
            if factors_to_compute is None:
                factors_to_compute = FACTORS_TO_COMPUTE
            
            logger.info(f"{symbol}: 开始硬性计算 {len(factors_to_compute)} 个因子")
            
            result_df = df.copy()
            calculated_factors = []
            
            # 硬性计算所有因子，不管数据是否充足
            for factor_name, params in factors_to_compute:
                factor_series = self.calculate_single_factor(df, factor_name, params, symbol)
                
                if factor_series is not None:
                    result_df[factor_series.name] = factor_series
                    calculated_factors.append(factor_series.name)
                else:
                    logger.warning(f"{symbol}: 跳过因子 {factor_name}")
            
            logger.info(f"{symbol}: 成功计算 {len(calculated_factors)} 个因子: {calculated_factors}")
            
            # 新的NaN处理逻辑：删除因子列有NaN的所有行
            if calculated_factors:
                original_length = len(result_df)
                
                # 找到因子列中有任何NaN的行
                factor_columns = calculated_factors
                has_nan_mask = result_df[factor_columns].isnull().any(axis=1)
                
                # 删除有NaN的行
                result_df = result_df[~has_nan_mask].reset_index(drop=True)
                
                removed_rows = original_length - len(result_df)
                logger.info(f"{symbol}: 因子NaN处理 - 删除{removed_rows}行包含NaN的数据，保留{len(result_df)}行有效数据")
                
                # 验证处理后的数据确实无NaN值
                remaining_nans = result_df[factor_columns].isnull().sum().sum()
                if remaining_nans > 0:
                    logger.warning(f"{symbol}: 警告 - 处理后仍有{remaining_nans}个NaN值")
                else:
                    logger.info(f"{symbol}: ✓ 因子列严格无NaN值")
                
                # 如果删除后没有数据了，返回None
                if len(result_df) == 0:
                    logger.warning(f"{symbol}: 删除NaN行后无剩余数据，返回None")
                    return None
            
            return result_df
            
        except Exception as e:
            logger.error(f"{symbol}: 因子计算失败: {str(e)}")
            return None
    
    @time_perf_decorator()
    async def calculate_factors_async(self, data_dict: Dict[str, pd.DataFrame], 
                                    factors_to_compute: List[Tuple[str, List]] = None) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
        """
        异步批量计算因子
        
        Args:
            data_dict: {symbol: dataframe} 字典
            factors_to_compute: 要计算的因子列表
            
        Returns:
            ({symbol: dataframe_with_factors} 字典, {舍弃的symbol: 舍弃原因} 字典)
        """
        try:
            if factors_to_compute is None:
                factors_to_compute = FACTORS_TO_COMPUTE
            
            logger.info(f"开始异步计算 {len(data_dict)} 个交易对的因子")
            logger.info(f"要计算的因子: {factors_to_compute}")
            
            # 创建异步任务
            loop = asyncio.get_event_loop()
            tasks = []
            
            for symbol, df in data_dict.items():
                task = loop.run_in_executor(
                    self.executor,
                    self.calculate_factors_for_symbol,
                    df,
                    symbol,
                    factors_to_compute
                )
                tasks.append((symbol, task))
            
            # 等待所有任务完成
            result_dict = {}
            skipped_dict = {}
            
            for symbol, task in tasks:
                try:
                    result_df = await task
                    if result_df is not None:
                        result_dict[symbol] = result_df
                    else:
                        # 计算失败或删除NaN后无数据，记录到舍弃列表
                        skipped_dict[symbol] = "删除NaN行后无剩余数据或计算失败"
                except Exception as e:
                    logger.error(f"{symbol}: 异步因子计算失败 - {str(e)}")
                    skipped_dict[symbol] = f"计算异常: {str(e)}"
            
            logger.info(f"异步因子计算完成: 成功{len(result_dict)}个，舍弃{len(skipped_dict)}个")
            return result_dict, skipped_dict
            
        except Exception as e:
            logger.error(f"异步因子计算失败: {str(e)}")
            return data_dict, {}
    
    def validate_factor_results(self, data_dict: Dict[str, pd.DataFrame], 
                              factors_to_compute: List[Tuple[str, List]] = None) -> Dict[str, Dict]:
        """
        验证因子计算结果
        
        Args:
            data_dict: 计算结果字典
            factors_to_compute: 预期的因子列表
            
        Returns:
            验证结果字典
        """
        try:
            if factors_to_compute is None:
                factors_to_compute = FACTORS_TO_COMPUTE
            
            validation_results = {}
            
            # 生成预期的因子列名
            expected_factor_columns = []
            for factor_name, params in factors_to_compute:
                if params:
                    column_name = f"{factor_name}_{'_'.join(map(str, params))}"
                else:
                    column_name = factor_name
                expected_factor_columns.append(column_name)
            
            for symbol, df in data_dict.items():
                result = {
                    'total_factors': len(expected_factor_columns),
                    'calculated_factors': 0,
                    'missing_factors': [],
                    'invalid_factors': [],
                    'factor_stats': {}
                }
                
                for factor_col in expected_factor_columns:
                    if factor_col in df.columns:
                        factor_values = df[factor_col].dropna()
                        
                        if len(factor_values) > 0:
                            result['calculated_factors'] += 1
                            
                            # 计算因子统计信息
                            result['factor_stats'][factor_col] = {
                                'count': len(factor_values),
                                'mean': factor_values.mean(),
                                'std': factor_values.std(),
                                'min': factor_values.min(),
                                'max': factor_values.max(),
                                'na_count': df[factor_col].isna().sum()
                            }
                            
                            # 检查异常值
                            if factor_values.std() == 0:
                                result['invalid_factors'].append(f"{factor_col}(常数)")
                            elif (factor_values == np.inf).any() or (factor_values == -np.inf).any():
                                result['invalid_factors'].append(f"{factor_col}(无穷大)")
                        else:
                            result['missing_factors'].append(factor_col)
                    else:
                        result['missing_factors'].append(factor_col)
                
                validation_results[symbol] = result
            
            return validation_results
            
        except Exception as e:
            logger.error(f"验证因子结果失败: {str(e)}")
            return {}
    
    def log_factor_summary(self, validation_results: Dict[str, Dict]):
        """
        记录因子计算汇总信息
        
        Args:
            validation_results: 验证结果字典
        """
        try:
            if not validation_results:
                logger.warning("没有因子验证结果可显示")
                return
            
            logger.info("=" * 60)
            logger.info("因子计算汇总:")
            logger.info("=" * 60)
            
            total_symbols = len(validation_results)
            total_expected_factors = 0
            total_calculated_factors = 0
            
            for symbol, result in validation_results.items():
                total_expected_factors += result['total_factors']
                total_calculated_factors += result['calculated_factors']
            
            success_rate = (total_calculated_factors / total_expected_factors * 100) if total_expected_factors > 0 else 0
            
            logger.info(f"总交易对数: {total_symbols}")
            logger.info(f"预期因子总数: {total_expected_factors}")
            logger.info(f"成功计算因子数: {total_calculated_factors}")
            logger.info(f"因子计算成功率: {success_rate:.2f}%")
            
            # 显示前几个交易对的详细信息
            logger.info("\n前5个交易对因子计算详情:")
            for i, (symbol, result) in enumerate(list(validation_results.items())[:5]):
                logger.info(f"{symbol}: {result['calculated_factors']}/{result['total_factors']} 因子成功, "
                           f"缺失: {len(result['missing_factors'])}, 异常: {len(result['invalid_factors'])}")
            
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"记录因子汇总失败: {str(e)}")
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        if hasattr(self, 'factor_executor') and self.factor_executor:
            self.factor_executor.shutdown(wait=True)

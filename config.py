"""
ML_Quant 统一配置文件
管理数据路径、性能参数、标签周期和因子列表
"""
import os
from pathlib import Path

# =============================
# 生产环境：一键运行与核心参数（前置）
# =============================
# 每个Step是否运行（可按需开关）
RUN_STEP1 = True               # 原始数据处理（CSV→Parquet）
RUN_STEP2 = True               # 标签设计
RUN_STEP3 = True               # 特征工程（计算因子）
RUN_FACTOR_ANALYSIS = True     # 因子分析（保存图像与日志）

# 输入数据路径（原始CSV文件目录）
INPUT_CSV_PATH = r"xxxxxxxxxxxxxxxxxx"

# 标签设计（生产常改）
LABEL_PERIOD = 1  # 标签计算周期n

# 因子计算与分析（生产常改）
# 格式: [("因子名", [参数列表])]
FACTORS_TO_COMPUTE = [
    ("RSI", [14])
]
# 因子分析控制：True分析所有；False仅分析FACTORS_TO_COMPUTE的第一个因子
ANALYZE_ALL_FACTORS = True

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 数据路径配置
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_ROOT / "raw_data"
LABELED_DATA_PATH = DATA_ROOT / "labeled_data"
FEATURE_DATA_PATH = DATA_ROOT / "feature_data"

# （提示）上方已设置 INPUT_CSV_PATH、LABEL_PERIOD、FACTORS_TO_COMPUTE、ANALYZE_ALL_FACTORS 与步骤开关

# 性能配置（基于实际数据：603个币种，43.78MB，平均672条/文件）
MAX_WORKERS = 8  # 最大线程数（适应多小文件场景，避免I/O竞争）
CHUNK_SIZE = 5000  # 分块读取大小（适应小文件，平均672条记录）
BATCH_SIZE = 50  # 批量处理大小（适合小文件批处理）
ASYNC_SEMAPHORE = 30  # 异步并发限制（小文件可以增加并发）

# 内存管理配置（基于43.78MB总数据量优化）
MEMORY_LIMIT_GB = 32  # 内存使用上限（GB），数据量小，降低内存占用
ENABLE_MEMORY_MONITORING = True  # 启用内存监控
GC_THRESHOLD = 0.7  # 内存使用率达到70%时触发垃圾回收（更积极）

# 数据处理配置
CSV_ENCODING = "gbk"  # CSV文件编码
SKIP_ROWS = 1  # 跳过CSV文件前几行
FILE_SUFFIX = "-USDT"  # 只处理以此结尾的文件

# 数据处理优化配置（针对小文件优化）
ENABLE_PARALLEL_FACTOR_CALC = True  # 启用因子并行计算
FACTOR_CALC_WORKERS = 8  # 因子计算专用线程数（避免过度并发）
ENABLE_VECTORIZED_OPERATIONS = True  # 启用向量化操作
PRELOAD_DATA_TO_MEMORY = True  # 预加载数据到内存（小文件全部预加载）
USE_OPTIMIZED_BIN_STATS = True
USE_PARQUET_COLUMN_PRUNE = True
USE_RANK_BASED_IC = True
ENABLE_RESULT_CACHE = False

# 数据合并配置
ENABLE_1H_TO_1D_MERGE = True  # 是否启用1小时数据合并为1天数据
MERGE_MAX_WORKERS = MAX_WORKERS  # 数据合并的最大线程数
MERGE_BATCH_SIZE = BATCH_SIZE  # 数据合并的批量处理大小

# 标签设计配置（已前置到生产参数区）

# 因子计算配置与因子分析控制（已前置到生产参数区）

# Parquet文件配置（针对小文件优化）
PARQUET_ENGINE = "pyarrow"
PARQUET_COMPRESSION = "lz4"  # 更快的压缩算法，适合大量小文件
PARQUET_ROW_GROUP_SIZE = 10000  # 行组大小优化（适应平均672条记录的小文件）
PARQUET_PAGE_SIZE = 256 * 1024  # 页面大小优化（256KB，适合小文件）
PARQUET_USE_DICTIONARY = True  # 启用字典编码

# 日志配置
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 日志设置函数 - 自动配置日志系统
def setup_logging():
    """设置日志配置"""
    import logging
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
        ]
    )

# 确保数据目录存在
def ensure_data_dirs():
    """确保所有数据目录存在"""
    for path in [RAW_DATA_PATH, LABELED_DATA_PATH, FEATURE_DATA_PATH]:
        path.mkdir(parents=True, exist_ok=True)

# 别名函数，保持兼容性
def ensure_data_directories():
    """确保所有数据目录存在（别名函数）"""
    ensure_data_dirs()

# 清空目录函数
def clear_directory(directory_path):
    """清空指定目录下的所有文件"""
    import shutil
    if directory_path.exists():
        shutil.rmtree(directory_path)
    directory_path.mkdir(parents=True, exist_ok=True)

def clear_all_data():
    for path in [RAW_DATA_PATH, LABELED_DATA_PATH, FEATURE_DATA_PATH]:
        clear_directory(path)

def get_analysis_output_path():
    return PROJECT_ROOT / "Factor_Analysis" / "Output"

def clear_analysis_output():
    clear_directory(get_analysis_output_path())

def clear_caches():
    import shutil
    for root, dirs, files in os.walk(PROJECT_ROOT):
        for d in list(dirs):
            if d in {"__pycache__", ".ipynb_checkpoints"}:
                target = Path(root) / d
                try:
                    shutil.rmtree(target, ignore_errors=True)
                except Exception:
                    pass
        for f in list(files):
            if f.endswith((".pyc", ".pyo")):
                target = Path(root) / f
                try:
                    target.unlink(missing_ok=True)
                except Exception:
                    pass

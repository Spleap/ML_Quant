# ML_Quant

一个面向加密交易对的量化研究项目，提供三步数据管道（CSV→原始Parquet→标签→特征/因子）与单因子分析模块（自动生成可视化与日志）。支持异步/并行与内存监控，开箱即用。

## 仓库结构
- `run_pipeline.py`：一键执行完整流水线（Step1→Step2→Step3→因子分析）
- `config.py`：统一配置与清理函数（数据路径、因子列表、并行/内存等）
- `step1_data_processing/`：原始数据处理（CSV→Parquet）
- `step2_label_design/`：标签设计（在原始数据上生成标签列）
- `step3_feature_engineering/`：特征工程（计算因子并保存）
- `Factor_Analysis/`：单因子分析（读取特征数据，生成图与日志）
- `data/`：数据输出根目录（`raw_data/`、`labeled_data/`、`feature_data/`）

## 安装与环境
- Python：建议 `>= 3.10`
- 安装依赖：
  - `pip install -r requirements.txt`

## 快速开始
1. 在 `config.py` 设置：
   - `INPUT_CSV_PATH`：原始 CSV 文件目录
   - `FACTORS_TO_COMPUTE`：要计算的因子及参数（如：`[("RSI", [14])]`）
   - `ANALYZE_ALL_FACTORS`：因子分析是否遍历所有配置的因子
   - 步骤开关：`RUN_STEP1/RUN_STEP2/RUN_STEP3/RUN_FACTOR_ANALYSIS`
2. 运行一键脚本：
   - `python run_pipeline.py`
3. 产物位置：
   - 原始数据 Parquet：`data/raw_data/`
   - 标签数据 Parquet：`data/labeled_data/`
   - 特征数据 Parquet：`data/feature_data/`
   - 因子分析输出（图与日志）：`Factor_Analysis/Output/<timestamp>_<factor>/`

## 因子分析配置
- 修改 `Factor_Analysis/analysis_config.py` 中的：
  - `single_factor`：当前分析因子列名（如 `RSI_14`）
  - `time_range`：时间区间（可选）
  - `remove_outliers`：是否去极值
  - `ir_window`：IR 计算窗口长度
  - `use_parallel`、`sampling_frequency`：并行与采样频率
- 直接运行：`python Factor_Analysis/main.py`

## 清理指南
- 内置便捷函数（在 `config.py`）：
  - `clear_all_data()`：清空 `data/raw_data/`、`data/labeled_data/`、`data/feature_data/`
  - `clear_analysis_output()`：清空 `Factor_Analysis/Output/`
  - `clear_caches()`：递归删除 `__pycache__/`、`*.pyc`、`*.pyo`、`.ipynb_checkpoints`
- 也可在运行各步骤前自动清理（各 Step 主程序已调用各自目录清理）

## 关键实现参考
- 入口与开关：`run_pipeline.py:167-198`
- 数据目录清理：
  - Step1：`step1_data_processing/main.py:158-160`
  - Step2：`step2_label_design/main.py:44-46`
  - Step3：`step3_feature_engineering/main.py:46-48`
  - 清理函数：`config.py:111-117`
- 因子分析输出：`Factor_Analysis/main.py:81-84`，保存：`Factor_Analysis/utils/result_saver.py:116-147`

## 性能与内存
- 并行/异步：`MAX_WORKERS`、`ASYNC_SEMAPHORE`、`FactorCalculator` 异步计算
- 内存监控与回收：`ENABLE_MEMORY_MONITORING`、`MEMORY_LIMIT_GB`、`GC_THRESHOLD`
- Parquet 写出优化：`pyarrow` + 压缩与行组配置

## 常见问题
- 原始 CSV 必需列：`timestamp/open/high/low/close/volume`
- 因子前置 NaN：如 `RSI_14` 前 14 行为 NaN，保存前会清理这些行
- Windows 异步策略：入口已设置兼容策略，避免事件循环报错

## 版本控制建议
- `.gitignore` 已忽略大数据与输出产物及缓存目录
- 如需保留历史产物，请先备份再清理

## 许可证
- 可根据需要添加开源许可证（如 MIT）。

## GitHub 上传步骤（Windows）
1. 在 GitHub 新建仓库（记下远程地址）
2. 在项目根目录执行：
   - `git init`
   - `git config core.autocrlf true`
   - `git add .`
   - `git commit -m "Initialize ML_Quant project"`
   - `git branch -M main`
   - `git remote add origin https://github.com/<用户名>/ML_Quant.git`
   - `git push -u origin main`
## 目标
- 清空 `data` 子目录内所有文件与子文件夹（`raw_data/`、`labeled_data/`、`feature_data/`）
- 清理工作区缓存文件（`__pycache__/`、`*.pyc`、`*.pyo`、`.ipynb_checkpoints` 等）
- 清空因子分析输出目录 `Factor_Analysis/Output/`
- 通读代码并撰写 `README.md`
- 增加 `.gitignore` 并整理依赖，项目可直接上传至 GitHub
- 提供在 Windows 上上传的详细步骤

## 清理范围与依据
- 数据输出目录（内置清理逻辑）：
  - Step1 清理：`step1_data_processing/main.py:158-160` 调用 `config.clear_directory(RAW_DATA_PATH)`
  - Step2 清理：`step2_label_design/main.py:44-46` 调用 `config.clear_directory(LABELED_DATA_PATH)`
  - Step3 清理：`step3_feature_engineering/main.py:46-48` 调用 `config.clear_directory(FEATURE_DATA_PATH)`
  - 清理函数定义：`config.py:111-117`（删除后重建目录）
- 因子分析输出目录：运行时创建 `Factor_Analysis/Output/`（`Factor_Analysis/main.py:81-84`、结果保存 `Factor_Analysis/utils/result_saver.py:116-147`）。需新增一个便捷清理函数。
- 缓存与临时文件：当前仓库未内置统一清理；存在 `__pycache__/`（如 `step3_feature_engineering/__pycache__/...`）。需新增递归清理。

## 实施方案
- 在 `config.py` 中新增便捷清理方法，集中操作：
  - `clear_all_data()`：调用 `clear_directory(RAW_DATA_PATH/LABELED_DATA_PATH/FEATURE_DATA_PATH)`
  - `clear_analysis_output()`：定位 `PROJECT_ROOT/Factor_Analysis/Output` 并清空
  - `clear_caches()`：递归删除工作区内 `**/__pycache__/`、`*.pyc`、`*.pyo`、`.ipynb_checkpoints` 等
- 提供一键脚本入口（不改变现有运行入口）：
  - 在 `run_pipeline.py` 内保留现状；如需可新增可选 `--clean` 执行上述三项清理，再按开关运行流水线
  - 保持破坏性操作仅针对 `data/` 与 `Factor_Analysis/Output/`

## README.md 结构（将撰写）
- 项目简介：三步数据管道 + 单因子分析
- 仓库结构与关键模块：
  - 入口：`run_pipeline.py`（一键执行）
  - 配置：`config.py`（数据路径、开关、参数）
  - Step1：`step1_data_processing/*`
  - Step2：`step2_label_design/*`
  - Step3：`step3_feature_engineering/*`
  - 因子分析：`Factor_Analysis/*` 与 `analysis_config.py`
- 快速开始：安装依赖、配置 `INPUT_CSV_PATH`、运行 `python run_pipeline.py`
- 配置说明：运行开关、`FACTORS_TO_COMPUTE`、`ANALYZE_ALL_FACTORS`、`LABEL_PERIOD`
- 数据输入/输出：CSV 要求与 Parquet 产物位置（`data/raw_data/`、`data/labeled_data/`、`data/feature_data/`）
- 因子分析输出：`Factor_Analysis/Output/<timestamp>_<factor>/`
- 清理指南：调用 `config` 中新增的三个清理函数或 `--clean` 选项
- 性能与内存：并行/异步、内存监控、垃圾回收阈值
- 常见问题：数据列缺失、因子前置 NaN、Windows 事件循环策略
- 许可证与贡献（如需）

## Git忽略与依赖
- 新增 `.gitignore`：
  - `data/**`（大文件不入库）
  - `Factor_Analysis/Output/**`
  - `__pycache__/`, `*.pyc`, `*.pyo`, `.ipynb_checkpoints`, `.DS_Store`
- 新增依赖清单 `requirements.txt`（根据导入）：
  - `pandas`、`numpy`、`pyarrow`、`matplotlib`、`psutil`
  - Python 版本建议：`>=3.10`

## 验证步骤
- 运行清理函数后检查目标目录为空
- 执行 `python run_pipeline.py` 验证三步产物与因子分析输出
- 检查 `.gitignore` 是否生效（`git status` 无大数据/输出文件）

## 上传到 GitHub（Windows PowerShell）
- 在 GitHub 创建空仓库（公开/私有均可），记下远程地址 `https://github.com/<用户名>/ML_Quant.git`
- 在项目根目录执行：
  - `git init`
  - `git config core.autocrlf true`
  - `git add .`
  - `git commit -m "Initialize ML_Quant project"`
  - `git branch -M main`
  - `git remote add origin https://github.com/<用户名>/ML_Quant.git`
  - `git push -u origin main`

## 注意与风险
- 清理操作为破坏性：会删除并重建输出目录；不会触及源代码
- `.gitignore` 将避免将大量 Parquet 数据与绘图产物上传到仓库
- 如需保留历史产物，请先备份后再清理

确认后我将：
1) 实现 `config.py` 中的三个清理函数与（可选）`--clean` 入口
2) 编写 `README.md` 与 `.gitignore`、`requirements.txt`
3) 完成一次本地验证并提供上传演示
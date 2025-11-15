## 总览
- 代码入口：`run_pipeline.py:167-205` 控制 Step1→Step2→Step3→因子分析全流程，读取 `config.py` 的运行开关。
- 架构分层：Step1（原始数据→清洗→Parquet）、Step2（标签生成）、Step3（因子计算）、Factor_Analysis（数据加载→统计/绘图）。广泛使用 `asyncio` + `ThreadPoolExecutor` 做并发；暂无显式缓存与测试套件。
- 主要潜在瓶颈：
  - 分箱统计：`Factor_Analysis/utils/factor_stats.py:61-227, 230-396, 398-410`
  - 截面相关：`Factor_Analysis/utils/ic_ir_plot.py:55-110`
  - 因子拼接与NaN清理：`step3_feature_engineering/factor_calculator.py:249-317`
  - 多文件批读取与合并：`Factor_Analysis/utils/data_loader.py:158-260, 260-319`

## 阶段一：性能剖析与基线
1. 运行级剖析：
   - 在 `run_pipeline.py:167-205` 外层使用 `cProfile` 生成全流程基线；按开关分别跑 Step1/2/3/分析，记录阶段耗时与函数热点。
   - 插入统一计时装饰器（`time.perf_counter`）到热点函数，输出粒度到日志：
     - `factor_stats.compute_factor_hist` 与 `compute_label_mean_by_bin`
     - `data_loader.load_factor_data`
     - `factor_calculator.calculate_single_factor/calculate_factors_for_symbol/calculate_factors_async`
     - `ic_ir_plot.compute_ic_series`
   - 采样剖析：用 `py-spy` 或 `yappi` 做采样，确认 C 层向量化 vs Python 循环的占比。
2. 内存剖析：
   - 复用 `utils/memory_monitor.py`，新增计时装饰器；记录每次 GC 触发与峰值内存。
3. 基线数据：
   - 数据规模：每批 10/50/100 文件；每因子 1e5/5e5/1e6 行。
   - 指标：阶段耗时、CPU 利用、峰值/平均内存、GC 次数、线程数。

## 阶段二：微基准与A/B
- 微基准脚本（不改业务逻辑，仅插桩与对比）：
  - 分箱均值：在 `factor_stats.py` 上用随机向量 + 真实样本做 `digitize+np.bincount/np.histogram(weights)` vs 现实现的循环/并行路径对比。
  - IC 计算：`groupby→rank(pct=True)→pearson` 与逐截面 `spearmanr` 对比。
  - Parquet 读取：`pd.read_parquet(columns=...)` 只读必要列 vs 读全量后选列。
  - 因子函数：`RSI/EMA_BIAS/MOM` 在 1e5/5e5 行上验证是否已充分向量化。
- A/B 验证：通过 `config.py` 开关切换旧/新实现，统计差异与速度提升；功能一致性以数值误差阈值（例如 1e-12）判定。

## 阶段三：优化实施（保持接口不变）
1. 算法优化（首要）：
   - `compute_label_mean_by_bin`（`factor_stats.py:230-396`）：
     - 用 `np.digitize` 获得 `bin_indices` 后，替换循环与分块累计为向量化：
       - 计数：`counts = np.bincount(bin_indices, minlength=n_bins)`
       - 加权和：`label_sums = np.bincount(bin_indices, weights=label_values, minlength=n_bins)`
       - 均值：`label_mean = label_sums / counts`（安全除法）
     - 并行路径仅在数据极大时分批 `np.bincount` 后相加，避免 Python 层 `for i in range(n_bins)`。
   - `compute_factor_hist`（`factor_stats.py:61-227`）：
     - 保持 `np.histogram` 主路径；并行时分块调用 `np.histogram` 已较优，仅微调分块阈值与批次大小，减少 futures 过多开销。
   - `compute_ic_series`（`ic_ir_plot.py:55-110`）：
     - 选项A（可切）：预先 `df.groupby(time)['factor'].rank(pct=True)` 与 `df.groupby(time)['label'].rank(pct=True)`，随后每截面用 `np.corrcoef` 计算皮尔逊，速度通常优于频繁 `spearmanr`。
     - 选项B：维持线程池，但对极大 `times` 做批次 `executor.map`，减少 `as_completed` 管理开销。
2. I/O 优化：
   - `data_loader._read_single_file`（`data_loader.py:60-156`）：
     - 读取时直接 `pd.read_parquet(file_path, engine=..., columns=keep_cols)`，避免读全量后再选列；保留向量化清理与类型优化。
   - 统一为 `float32` 与 `category`：
     - `combined['symbol'] = combined['symbol'].astype('category')`；继续对数值列做 `float32` 降维（已实现，维持）。
3. 并发处理：
   - CPU 密集段优先彻底向量化；若仍存在 Python 循环（仅少量场景），改为 `ProcessPoolExecutor`（按批聚合）或直接移除循环。
   - IO 段维持线程池；对批次 `batch_size`、`max_workers` 做配置化调优（来自 `config.py`）。
4. 缓存策略：
   - 分析阶段数据缓存：`load_factor_data(factor, time_range)` 结果按键（因子+时间窗）缓存到 `Feature_Cache/`（Parquet）；复用时直接读取缓存，减少重复IO与合并。
   - Step3 中间因子缓存：对单个 symbol 的因子列结果按（symbol, factor, params）命名缓存，重跑时复用；保留显式失效策略（通过 `config` 开关）。
5. 延迟加载：
   - 因子分析仅加载所需因子列与时间范围；绘图前做下采样（已支持），继续配置化。

## 阶段四：监控与度量
- 在 `utils/memory_monitor.py` 增加 `time_perf_decorator`（统一计时），并在上述热点函数上启用。
- 日志中统一输出：耗时（ms）、处理行数、线程数、GC 次数、内存峰值；形成 CSV/JSON 结果用于报告汇总。

## 阶段五：一致性与A/B验证
- 配置开关（`config.py`）：`USE_OPTIMIZED_BIN_STATS`、`USE_PARQUET_COLUMN_PRUNE`、`USE_RANK_BASED_IC`、`ENABLE_RESULT_CACHE`。
- A/B 比较：
  - 统计差异：分箱均值误差、IC/IR 差异分布（均值/方差/最大绝对误差）。
  - 性能对比：耗时、内存、IO 次数、线程数量；以固定数据规模输出对比表。
- 保证功能完整：失败回退到旧实现；日志明确标注所用路径。

## 交付物
- 性能优化报告：
  - 基线 vs 优化 后对比表（每阶段/每函数），误差分析与结论。
  - 剖析火焰图（若使用采样分析）、关键日志摘录与指标汇总（CSV/JSON）。
- 代码更新：
  - 保持代码风格一致，关键优化点增加说明性注释。
  - 通过现有流程跑通（现无测试套件，将以流水线成功完成与结果一致性作为验收标准）。

## 拟改动文件（精确定位）
- `Factor_Analysis/utils/factor_stats.py`
  - `compute_label_mean_by_bin`：替换循环为 `np.bincount` 向量化累计（约 `339-349` 与并行块 `302-325`）。
  - `compute_factor_hist`：调优并行阈值与批次（约 `161-183`）。
- `Factor_Analysis/utils/data_loader.py`
  - `_read_single_file`：`pd.read_parquet(..., columns=keep_cols)` 只读必要列（约 `65-76`）。
  - 合并后 `symbol`→`category`（约 `301-310`）。
- `Factor_Analysis/utils/ic_ir_plot.py`
  - `compute_ic_series`：可选 rank→pearson 实现与批次 `executor.map`（约 `70-110`）。
- `step3_feature_engineering/factor_calculator.py`
  - 为 `calculate_factors_for_symbol/calculate_factors_async` 增加结果级缓存（保持接口）。
- `utils/memory_monitor.py`
  - 新增 `time_perf_decorator` 并应用于热点函数。
- `config.py`
  - 新增优化与缓存开关、并发与批次配置项；不破坏现有变量。

—— 请确认以上计划；确认后我将按此方案插桩、编写微基准与逐项优化，并提供对比报告。
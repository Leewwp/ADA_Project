# 项目进度与组内交接

## 当前已完成

### 数据收集与 ETL

- 已完成香港 AQI 与气象数据的真实 API 抓取脚本整理。
- 已将数据采集默认模式改为真实 API，不再静默回退到示例数据。
- 已新增 `collection_metadata.json`，可核验：
  - 是否真实采集
  - 是否 fallback 到 sample
  - 每个时间分块的请求状态
  - 是否使用天气代理补全
- 已新增 `preprocessing_report.json`，可核验：
  - 缺失值处理
  - 异常值审计
  - 物理边界修正
  - 清洗后数据校验
- 已新增 `data/data_dictionary.csv` 与 `report/data_collection_etl.md`，可直接支持报告写作。

### 当前真实数据状态

- 当前主数据已成功通过真实 API 采集。
- `data/collection_metadata.json` 显示：
  - `source_mode = live_api`
  - `fallback_to_sample = false`
  - `weather_synthetic_fill.used = false`
- 当前原始数据与清洗后数据均为：
  - 17544 条小时级记录
  - 时间范围 `2024-01-01 00:00:00` 到 `2025-12-31 23:00:00`
  - 缺失值为 0
  - 重复时间戳为 0

## 最新策略调整

- 预处理默认异常值模式改为 `flag`。
- 原因：空气污染尖峰可能是真实污染事件，不应被 IQR 规则直接裁剪掉。
- 如果建模同学需要做对比实验，可以额外测试：
  - `python src/02_preprocessing.py --outlier-mode clip`
  - `python src/02_preprocessing.py --outlier-mode remove`

## 对建模同学的建议

- 优先使用当前默认的 `flag` 版清洗数据做主实验。
- 可以额外做一组 `clip` 对照实验，比较 RMSE / MAE / R2 是否明显变化。
- 当前项目仍需重点完善：
  - LSTM 是否真正跑通并纳入比较
  - `07_forecast.py` 是否真正调用最优模型，而不是简单趋势外推
  - `09_final_evaluation.py` 在不同环境中的稳定运行问题
- 如果后续需要更严格的标签定义，可进一步把 `aqi` 从 PM2.5 proxy 升级为多污染物 AQI。

## 对报告/汇报同学的建议

- 报告的数据来源与转换部分，建议直接引用以下文件内容：
  - `report/data_collection_etl.md`
  - `data/data_dictionary.csv`
  - `data/collection_metadata.json`
  - `data/preprocessing_report.json`
- 建议在报告中明确写出：
  - 数据来自 Open-Meteo API
  - 采集位置是香港单点经纬度
  - 时区统一为 `Asia/Hong_Kong`
  - 清洗策略以保留真实污染波动为主
  - 当前 `aqi` 是 PM2.5-based AQI proxy

## 还需要继续完善的地方

### 数据侧

- 可以考虑补充多站点或更高代表性的香港空气质量数据源。
- 可以考虑补充节假日、交通、季风/台风等外部变量。
- 可以考虑把 AQI 标签从 PM2.5 proxy 升级为更完整的官方口径。

### 模型侧

- 需要真正打通 LSTM 或 Prophet 的可运行实验。
- 需要修正最终预测脚本与最优模型之间的脱节。
- 需要复核测试集时间口径与 README/报告中的表述是否完全一致。
- `03_feature_engineering.py` 当前还能运行，但有两个技术债：
  - `aqi_percentile_24h` 的 rolling lambda 里用了 `x[-1]`，后续最好改成 `x.iloc[-1]`
  - 大量逐列插入特征导致 DataFrame fragmentation warning，后续可改成分组构造后 `pd.concat`

### 展示侧

- 建议补一页“数据采集与质量审计”展示：
  - 数据来源
  - 采集范围
  - 缺失值情况
  - 是否发生 fallback
  - 是否发生代理补全
- 这样可以显著增强项目可信度和完整性。

## 建议的组内分工衔接

- 数据收集：已基本完成第一轮稳定化与文档化，可继续负责多源数据补强和标签定义升级。
- 模型调优：继续负责 LSTM / Prophet / forecast 逻辑修正与最终性能比较。
- 报告汇报：开始根据 ETL 文档和审计文件撰写“数据来源与转换”“分析步骤与尝试”部分。

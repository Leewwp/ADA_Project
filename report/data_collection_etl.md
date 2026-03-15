# 数据收集与 ETL 说明

## 1. 数据来源

- 空气污染物数据：Open-Meteo Air Quality API
- 气象数据：Open-Meteo Weather Archive API
- 采集地点：Hong Kong（`22.2855, 114.1577`）
- 时间范围：默认 `2024-01-01` 至 `2025-12-31`
- 时间口径：统一按 `Asia/Hong_Kong` 时区处理

## 2. 原始字段

原始数据文件为 `data/raw_aqi_data.csv`。

- 污染物：`pm2_5`、`pm10`、`nitrogen_dioxide`、`ozone`
- 气象变量：`temperature_2m`、`relative_humidity_2m`、`wind_speed_10m`、`wind_direction_10m`、`surface_pressure`、`precipitation`、`cloud_cover`

字段级说明见 `data/data_dictionary.csv`。

## 3. ETL 流程

1. 使用 `src/01_data_collection.py` 按时间分块调用 API，默认每 6 个月为一个抓取窗口。
2. 将空气质量数据与气象数据按小时级时间戳对齐，并以 AQI 数据时间轴为主进行左连接。
3. 对采集结果生成审计元数据 `collection_metadata.json`，记录：
   - 抓取时间
   - 请求时间范围
   - API 成功/失败情况
   - 每个分块的记录数
   - 缺失值与天气补全情况
4. 使用 `src/02_preprocessing.py` 进行清洗：
   - 时间序列插值补缺
   - 默认仅对异常值进行 IQR 标记，不直接改写污染峰值
   - 如需对比实验，可选用 `--outlier-mode clip` 或 `--outlier-mode remove`
   - 无穷值替换
   - 物理边界约束（如湿度/云量限制在 0-100，降水和污染物不得为负）
5. 输出清洗后数据 `processed_aqi_data.csv`，并生成 `preprocessing_report.json`。

## 4. 审计产物

- `data/collection_metadata.json`：记录数据来源、API 请求状态、缺失情况、是否使用天气代理补全
- `data/preprocessing_report.json`：记录清洗前后摘要、缺失值处理、异常值处理、边界校正结果
- `data/data_dictionary.csv`：字段说明

## 5. 运行说明

真实数据采集：

```bash
python src/01_data_collection.py
```

仅在离线调试时生成示例数据：

```bash
python src/01_data_collection.py --mode sample
```

预处理：

```bash
python src/02_preprocessing.py
```

如果需要做异常值处理对比实验：

```bash
python src/02_preprocessing.py --outlier-mode flag
python src/02_preprocessing.py --outlier-mode clip
python src/02_preprocessing.py --outlier-mode remove
```

## 6. 当前限制

- 当前 `aqi` 标签是基于 `pm2_5` 推导的 AQI proxy，而不是完整多污染物官方 AQI。
- 当前采集点是香港单一经纬度点，适合作为区域代表性样本，但不等同于多站点官方监测网络。
- 如果天气 API 局部缺失，脚本会仅对天气字段进行代理补全，并在 `collection_metadata.json` 中明确记录。
- 当前默认采取保守预处理策略，优先保留真实污染波动；若后续建模需要更强清洗，应将实验设置单独记录在报告中。

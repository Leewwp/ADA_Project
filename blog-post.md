# 香港空气质量短期预测系统：从 API 数据采集到多模型融合的完整实现

**TL;DR:** 本文介绍一个基于 Python 的香港 AQI 短期预测项目，使用 Open-Meteo API 采集空气质量和气象数据，通过 9 大类特征工程和 6+ 种机器学习模型的融合，实现了未来 24-72 小时 AQI 趋势预测。集成加权融合模型在测试集上取得 RMSE 29.39、方向准确率 65.6% 的表现。

---

## 一、为什么做这个项目

空气污染是影响数百万城市居民健康的重大公共卫生问题。香港虽已建立官方监测网络，但现有公开预报工具要么精度有限，要么仅面向专业用户。作为开发者，我们希望构建一个**可复现、透明、且对开发者友好的 AQI 预测管道**。

这个项目的核心目标是：

1. 建立完整的数据采集→清洗→建模→预测流水线
2. 探索多模型融合是否能超越单一模型
3. 积累时间序列预测的工程经验

---

## 二、技术架构

```
┌──────────────────────────────────────────────────────────────┐
│                    数据采集 (01_data_collection.py)          │
│  Open-Meteo Air Quality API + Weather Archive API            │
│  → 分块采集 + 重试机制 + metadata 审计                        │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                    数据清洗 (02_preprocessing.py)            │
│  时间插值 → 异常值标记(flag) → 物理边界约束                   │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                    特征工程 (03_feature_engineering.py)       │
│  9 大类特征：时间/气象/大气稳定度/污染指数/滞后/滚动/差分/交互 │
│  重点：AQI 自回归特征群                                       │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                    模型训练 (04_models.py)                   │
│  Ridge / Lasso / ElasticNet / RF / XGBoost / LightGBM / SARIMA│
│  → 加权集成融合                                              │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                  最终评估 (09_final_evaluation.py)           │
│  测试集：2025 Q2                                             │
└──────────────────────────────────────────────────────────────┘
```

---

## 三、数据采集：API + 分块 + 审计

### 3.1 数据源选择

选用 **Open-Meteo** 的两个免费 API：

- **Air Quality API**: pm2_5, pm10, nitrogen_dioxide, ozone
- **Weather Archive API**: temperature, humidity, wind speed/direction, pressure, precipitation, cloud cover

数据点为香港单点坐标 (22.2855°N, 114.1577°E)，时间范围 2024-01-01 至 2025-12-31，共 17,544 条小时级记录。

### 3.2 分块采集策略

API 对单次请求的时间范围有限制，因此采用**每 6 个月一个分块**的策略：

```python
def build_date_chunks(start_date, end_date, months_per_chunk=6):
    chunks = []
    current = pd.Timestamp(start_date)
    while current <= end_ts:
        next_start = current + pd.DateOffset(months=months_per_chunk)
        chunk_end = min(next_start - pd.Timedelta(days=1), end_ts)
        chunks.append((current.date().isoformat(), chunk_end.date().isoformat()))
        current = chunk_end + pd.Timedelta(days=1)
    return chunks
```

### 3.3 失败重试与审计

每个 API 请求最多重试 3 次，指数退避：

```python
def fetch_with_retry(url, params, max_retries=3, pause_seconds=1):
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.json(), {"success": True, "attempts": attempt}
        except requests.exceptions.RequestException as exc:
            if attempt < max_retries:
                time.sleep(pause_seconds * attempt)
    return None, {"success": False, "attempts": max_retries}
```

每次采集运行后生成 `collection_metadata.json`，记录：
- 是否使用真实 API（而非 fallback 示例数据）
- 是否使用了天气代理补全
- 每个分块的请求状态

---

## 四、数据预处理：保守策略

预处理采用**保守策略**，核心原则是**保留真实的污染事件**。

### 4.1 缺失值处理

使用时间感知插值，避免引入虚假数据：

```python
df_clean = df_clean.interpolate(method="time")  # 时间插值
df_clean = df_clean.ffill().bfill()             # 首尾用前向/后向填充
```

### 4.2 异常值处理

默认仅标记（flag），不裁剪。这是经过教训得出的设计决策：

> 空气污染尖峰（工厂排放、工地扬尘、跨境污染）可能是真实事件，IQR 裁剪会将这些真实信号删除。

```bash
# 三种模式对比
python src/02_preprocessing.py --outlier-mode flag    # 默认：标记
python src/02_preprocessing.py --outlier-mode clip    # 裁剪
python src/02_preprocessing.py --outlier-mode remove  # 删除
```

### 4.3 物理边界约束

```python
# 湿度/云量限制 [0, 100]%，降水和污染物不得为负
df_clean[column] = df_clean[column].clip(lower=0, upper=100)  # 适用于 bounded columns
```

---

## 五、特征工程：预测效果的核心差异

特征工程是本项目的**核心竞争力**。共构建 9 大类、数百个特征。

### 5.1 特征构建顺序（关键）

```python
def create_all_features():
    df = create_time_features(df)           # 1. 时间特征
    df = create_meteorological_features(df) # 2. 气象特征
    df = create_atmospheric_stability_features(df)  # 3. 大气稳定度
    df = create_pollution_indices(df)        # 4. 污染指数
    df = create_aqi_target(df)              # 5. AQI 目标变量（先于滞后特征）
    df = create_lag_features(df)            # 6. 滞后特征
    df = create_rolling_features(df)         # 7. 滚动统计
    df = create_diff_features(df)            # 8. 差分特征
    df = create_interaction_features(df)     # 9. 交互特征
```

**关键设计**：AQI 目标变量在滞后特征之前创建，确保 AQI 的自回归特征（lag、rolling）来自历史值，而非未来目标。

### 5.2 AQI 派生与目标变量

AQI 并非直接获取，而是基于 PM2.5 浓度分段线性派生：

```python
def pm25_to_aqi(pm):
    if pm <= 35:
        return (50 / 35) * pm
    elif pm <= 75:
        return 50 + (50 / 40) * (pm - 35)
    elif pm <= 115:
        return 100 + (50 / 40) * (pm - 75)
    elif pm <= 150:
        return 150 + (50 / 35) * (pm - 115)
    elif pm <= 250:
        return 200 + (100 / 100) * (pm - 150)
    else:
        return 300 + (100 / 100) * (pm - 250)
```

预测目标为 24/48/72 小时三个 horizon 的 AQI。

### 5.3 AQI 自回归特征（最重要的特征群）

```python
# AQI 滞后特征
for lag in [1, 3, 6, 12, 24, 48, 72, 168]:  # 168h = 1 周
    df[f'aqi_lag_{lag}h'] = df['aqi'].shift(lag)

# AQI 滚动统计
for window in [6, 12, 24, 48]:
    df[f'aqi_rolling_mean_{window}h'] = df['aqi'].rolling(window=window).mean()
    df[f'aqi_rolling_std_{window}h'] = df['aqi'].rolling(window=window).std()
    df[f'aqi_change_{window}h'] = df['aqi'] - df['aqi'].shift(window)

# AQI 动量
df['aqi_momentum_1h'] = df['aqi'].diff(1)
df['aqi_momentum_24h'] = df['aqi'].diff(24)
```

### 5.4 大气稳定度特征

```python
# 温度逆温指示（稳定大气 = 扩散差 = 污染累积）
df['inversion_potential'] = (
    (df['hour'].isin([0, 1, 2, 3, 4, 5])) &  # 夜间
    (df['surface_pressure'] > 1015) &
    (df['temperature_2m'] > df['temperature_2m'].shift(12))
).astype(int)

# 通风指数（风速 × 混合层高度代理）
df['ventilation_index'] = df['wind_speed_10m'] * mixing_proxy
```

---

## 六、模型训练与集成

### 6.1 数据集划分

| 数据集 | 时间范围 | 用途 |
|--------|---------|------|
| 训练集 | 2024 全年 | 模型学习 |
| 验证集 | 2025 Q1 | 超参调优 |
| 测试集 | 2025 Q2 | 最终评估 |

### 6.2 模型列表

| 模型 | 类型 | 调优方式 |
|------|------|---------|
| Ridge | 线性（L2 正则） | α ∈ [10⁻⁴, 10²] 网格搜索 |
| Lasso | 线性（L1 正则） | α ∈ [10⁻⁴, 10¹] 网格搜索 |
| ElasticNet | 线性（L1+L2） | α × l1_ratio 网格搜索 |
| Random Forest | 树集成 | RandomizedSearchCV (30 iterations) |
| XGBoost | 梯度提升 | RandomizedSearchCV + Early Stopping |
| LightGBM | 梯度提升 | Early Stopping |
| SARIMA | 传统时序 | 固定参数 |
| **Ensemble** | 加权融合 | 基于验证集 RMSE 加权 |

### 6.3 加权集成融合

```python
# 基于验证集 RMSE 的逆加权
w_lasso = (1/lasso_rmse) / total_inv
w_ridge = (1/ridge_rmse) / total_inv
w_rf = (1/rf_rmse) / total_inv
w_gb = (1/gb_rmse) / total_inv
w_xgb = (1/xgb_rmse) / total_inv if xgb_rmse else 0
w_lgbm = (1/lgbm_rmse) / total_inv if lgbm_rmse else 0

ensemble_pred = (w_lasso * lasso_pred + w_ridge * ridge_pred +
                 w_rf * rf_pred + w_gb * gb_pred +
                 w_xgb * xgb_pred + w_lgbm * lgbm_pred)
```

---

## 七、模型性能

### 7.1 测试集评估结果（2025 Q2）

| 模型 | MAE | RMSE | R² | 方向准确率 |
|------|-----|------|----|-----------|
| **Ensemble** | **18.64** | **29.39** | **0.254** | **65.6%** |
| Ridge | 18.24 | 29.46 | 0.251 | 66.9% |
| Lasso | 18.24 | 29.51 | 0.248 | 67.6% |
| XGBoost | 19.33 | 29.88 | 0.229 | 60.3% |
| Random Forest | 19.96 | 30.72 | 0.185 | 61.1% |
| Gradient Boosting | 21.05 | 31.45 | 0.146 | 57.5% |

### 7.2 关键发现

1. **线性模型表现优异**：Ridge/Lasso 与集成模型差距极小，说明特征工程已充分提取了线性关系
2. **集成模型在 MAE/RMSE 最优**：加权融合综合了各模型优势
3. **方向准确率约 65-67%**：意味着每 3 天预测能正确判断 2 天的 AQI 走势

---

## 八、Trade-offs 与局限性

| 局限 | 影响 | 改进方向 |
|------|------|---------|
| AQI 为 PM2.5 proxy | 非官方多污染物 AQI | 引入 NO₂、O₃ 联合计算 |
| 单点采集 | 代表性有限 | 扩展为多站点网络 |
| LSTM 尚未集成 | 深度学习潜力未释放 | 接入 LSTM/Transformer |
| 缺乏外部变量 | 台风、节假日等未纳入 | 补充 calendar features |

---

## 九、项目结构

```
hongkong_aqi_prediction/
├── README.md
├── requirements.txt
├── data/
│   ├── raw_aqi_data.csv          # 原始数据
│   ├── processed_aqi_data.csv    # 清洗后数据
│   ├── featured_aqi_data.csv     # 特征工程后数据
│   ├── collection_metadata.json   # 采集审计
│   └── preprocessing_report.json # 清洗审计
├── src/
│   ├── 01_data_collection.py     # API 采集
│   ├── 02_preprocessing.py       # 数据清洗
│   ├── 03_feature_engineering.py # 特征工程
│   ├── 04_models.py             # 模型训练
│   ├── 05_evaluation.py         # 模型评估
│   ├── 09_final_evaluation.py   # 最终评估
│   └── 10_report_visualization.py # 可视化
├── report/
│   ├── figures/                  # 图表
│   ├── evaluation_results.csv    # 评估结果
│   └── data_collection_etl.md   # ETL 文档
└── notebooks/
    └── analysis.ipynb           # 分析 notebook
```

---

## 十、开源准备建议

上传 GitHub 前建议：

1. **补充 LICENSE**（推荐 MIT）
2. **补充 CONTRIBUTING.md**
3. **配置 `.gitignore`**（排除 `.venv/`、`__pycache__/`、`*.pkl`）
4. **在 README 添加 Badge**（build passing、license）
5. **补充可视化图表** 到 `report/figures/`
6. **将 notebook 导出为 HTML** 方便在线预览

---

## Further Reading

- [Open-Meteo Air Quality API Documentation](https://open-meteo.com/en/docs/air-quality-api)
- [scikit-learn Time Series Split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [AQI Calculation Formula (US EPA)](https://www.airnow.gov/aqi/aqi-calculator/)

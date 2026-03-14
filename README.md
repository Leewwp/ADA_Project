# Hong Kong AQI Short-Term Prediction

## Project Overview

- **Topic**: Hong Kong Air Quality Index (AQI) Short-Term Prediction
- **Objective**: Predict AQI for the next 1-3 days to provide recommendations for residents
- **Data Type**: Multi-factor time series (pollutant concentration + meteorological factors)
- **Course**: COMP5152 - Data Mining and Machine Learning

## 当前项目状态

### 最新测试结果 (2025 Q2)

| Model | RMSE | R² | Direction Accuracy |
|-------|------|-----|-------------------|
| **Ensemble (Weighted)** | **19.13** | **0.3257** | **62.2%** |
| Ridge | 19.25 | 0.3174 | 62.9% |
| Lasso | 19.46 | 0.3023 | 63.6% |
| XGBoost | 19.86 | 0.2730 | 58.0% |
| Random Forest | 19.92 | 0.2684 | 58.7% |
| Gradient Boosting | 20.61 | 0.2173 | 55.7% |

### 关键发现

1. **最佳模型**: Ensemble (Lasso + Ridge + RF + XGBoost + GradientBoosting)
2. **特征数量**: 40个（原始17个 + 23个AQI自回归特征）
3. **最重要特征**: pm2_5, aqi_lag_24h, aqi_lag_48h, aqi_rolling_mean_24h

---

## 项目结构

```
hongkong_aqi_prediction/
├── README.md                       # 本文件
├── requirements.txt                # 依赖
├── data/
│   ├── raw_aqi_data.csv           # 原始API数据
│   ├── processed_aqi_data.csv      # 清洗后数据
│   └── featured_aqi_data.csv       # 特征工程后数据 (290列)
├── src/
│   ├── 01_data_collection.py       # API数据收集
│   ├── 02_preprocessing.py        # 缺失值 & 异常值处理
│   ├── 03_feature_engineering.py   # 特征工程
│   ├── 04_models.py               # 模型实现
│   ├── 04b_lstm_model.py          # LSTM模型 (新增但未使用)
│   ├── 05_evaluation.py           # 模型评估
│   ├── 06_visualization.py        # 可视化
│   ├── 07_forecast.py             # 最终预测
│   ├── 08_model_optimization.py   # 高级优化
│   └── 09_final_evaluation.py    # 最终评估
├── models/                         # 保存的模型
├── report/
│   ├── evaluation_results.csv     # 模型对比结果
│   ├── aqi_forecast.csv           # 72小时预测
│   └── figures/                   # 可视化图表
└── notebooks/
    └── analysis.ipynb              # Jupyter分析笔记
```

---

## 运行步骤

```bash
# 1. 数据收集
python src/01_data_collection.py

# 2. 数据预处理
python src/02_preprocessing.py

# 3. 特征工程
python src/03_feature_engineering.py

# 4. 模型训练
python src/04_models.py

# 5. 模型优化 (可选)
python src/08_model_optimization.py

# 6. 最终评估
python src/09_final_evaluation.py

# 7. 生成预测
python src/07_forecast.py
```

---

## 待办事项 (TODO)

### 高优先级 (P0) - 项目必须完成

- [ ] **LSTM模型实现并调优** - 项目要求必须有LSTM，需成功运行
- [ ] **生成最终预测结果** - 使用最佳模型生成72小时AQI预测
- [ ] **完成分析报告** - 撰写完整的结果分析和可视化

### 中优先级 (P1) - 提升项目质量

- [ ] **LightGBM安装和测试** - requirements.txt已添加，需验证
- [ ] **Prophet模型集成** - requirements.txt已有prophet，可尝试
- [ ] **调优超参数** - 进一步优化模型参数提升R²
- [ ] **增加更多可视化** - 时间序列图、特征重要性图、预测对比图

### 低优先级 (P2) - 完善项目

- [ ] **Jupyter Notebook整理** - 完整分析流程
- [ ] **结果分析文档** - 详细解释模型表现
- [ ] **README完善** - 添加更多技术细节

---

## 优化记录

### 已尝试的优化及效果

| 优化 | 效果 | 状态 |
|------|------|------|
| 添加AQI自回归特征 | R² 0.318 → 0.326 | ✅ 有效 |
| 固定特征选择(40个) | 比动态选择更好 | ✅ 有效 |
| 跳过污染物outlier处理 | 降低性能 | ❌ 回退 |
| 减少rolling特征数量 | 降低性能 | ❌ 回退 |
| LSTM模型 | 未完成调优 | 🔄 待完成 |
| LightGBM | 依赖未安装 | 🔄 待安装 |

---

## 依赖

```
pandas>=1.5.0
numpy>=1.21.0
requests>=2.28.0
scikit-learn>=1.2.0
statsmodels>=0.13.0
matplotlib>=3.6.0
seaborn>=0.12.0
prophet>=1.1.0
xgboost>=1.7.0
torch>=2.0.0
lightgbm>=3.0.0
jupyter>=1.0.0
```

**注意**: 需要运行 `pip install -r requirements.txt` 安装所有依赖

---

## 团队分工建议

| 成员 | 任务 |
|------|------|
| 成员A | LSTM调优、可视化完善、Notebook整理 |
| 成员B | LightGBM测试、Prophet集成、报告撰写 |

---

## 数据说明

- **训练集**: 2024年 (8,616样本)
- **验证集**: 2025 Q1 (2,160样本)
- **测试集**: 2025 Q2 (6,528样本)

### 特征工程 (290个特征)

| 类别 | 数量 | 示例 |
|------|------|------|
| 时间特征 | 20 | hour, day_of_week, month, cyclical encoding |
| 气象特征 | 18 | wind categories, temp categories, humidity |
| 大气稳定 | 3 | inversion_potential, ventilation_index |
| 污染指数 | 7 | pm25_pm10_ratio, pollution_index |
| 滞后特征 | 36 | lag_1h ~ lag_168h |
| 滚动统计 | 126 | mean, std, range (3h~168h窗口) |
| 差分特征 | 24 | diff_1h, pct_change |
| **AQI自回归** | 23 | aqi_lag, aqi_rolling, aqi_momentum |
| 交互特征 | 8 | temp*wind, humidity*wind |

---

## 参考

- Open-Meteo Air Quality API: https://open-meteo.com/air-quality
- Open-Meteo Weather API: https://open-meteo.com/weather
- AQI Calculation: China GB3095-2012 Standard

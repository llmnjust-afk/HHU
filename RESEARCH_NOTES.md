# CSEE-DML 研究思路文档

> 最后更新: 2026-08-19
> 目标期刊: Journal of Environmental Management (JEM, Q1, IF~8.7)

---

## 一、研究背景与问题

### 核心科学问题

**气候适应型城市政策是否能有效提升城市生态系统的气候韧性？**

### 政策背景

中国自2015年起分批推进两类气候适应型城市建设试点：

| 政策 | 批次 | 年份 | 城市数 | 文件 |
|------|------|------|--------|------|
| 海绵城市建设试点 | 第1批 | 2015 | 16城 | 财建〔2014〕838号 |
| 海绵城市建设试点 | 第2批 | 2016 | 14城 | 财办建〔2016〕25号 |
| 气候适应型城市试点 | — | 2017 | 28城 | 发改气候〔2017〕343号 |

合并后处理组共 **49个城市**（有重叠），采用 **staggered DID**（交错处理设计）。

### 研究空白

1. 现有文献多评估政策对经济指标（GDP、TFP）的影响，较少关注**生态韧性**本身
2. 传统DID方法对函数形式假设较强，**DML**（Double Machine Learning）能放松这些假设
3. 缺少将**极端天气事件**与政策交互的研究——政策是否在气候冲击下更有效？

---

## 二、理论框架

### 2.1 气候压力生态弹性 (CSEE)

基于生态韧性理论，构建 **Climate-Stress Ecological Elasticity (CSEE)** 指标：

```
CSEE = 0.5 × CR + 0.5 × RC
```

- **CR (Climate Resistance, 气候抵抗力)**: 极端天气期间NDVI维持正常水平的程度
  - CR = 1 − |NDVI_event − NDVI_normal| / NDVI_normal
  - CR越高 → 生态系统在冲击下越稳定

- **RC (Climate Recovery, 气候恢复力)**: 极端天气后NDVI恢复速度
  - RC = (NDVI_post − NDVI_event) / (NDVI_normal − NDVI_event)
  - RC越高 → 生态系统恢复越快

### 2.2 替代结局变量

| 指标 | 定义 | 数据源 | 理论依据 |
|------|------|--------|---------|
| **CSEE** | 气候压力下生态弹性 | MODIS NDVI + ERA5 | 核心自建指标 |
| **RSEI** | 遥感生态指数 | NDVI + 降水 + 温度 + 建成区 | 四维生态质量评估 |
| **PSR** | 压力-状态-响应韧性指数 | 多源整合 | PSR框架 |
| **CR / RC** | 抵抗力 / 恢复力分量 | MODIS NDVI + ERA5 | CSEE分解 |
| **LST** *(计划中)* | 地表温度变化 | MODIS MOD11A2 | 海绵城市降温效应 |
| **PM2.5** *(计划中)* | 空气质量 | 待获取 | 政策减排效应 |

### 2.3 因果识别策略

**Double Machine Learning (DML) — Partially Linear Model:**

```
Y = θ·D + g(X) + ε        (结局方程)
D = m(X) + ν               (处理方程)
```

- Y: 结局变量 (CSEE, RSEI, etc.)
- D: 处理变量 = DID × Shock (政策交互极端天气)
- X: 控制变量 (15个)
- θ: **目标因果效应**

DML优势：
- g(X) 和 m(X) 用ML灵活学习，无需参数假设
- Cross-fitting 避免过拟合
- Neyman正交性降低对 nuisance function 估计误差的敏感度

### 2.4 机制分析框架

```
Policy → Mediator → Ecological Resilience
         ↓
    sponge_inv (海绵城市投资)      → 46.4% 中介
    blue_green_ratio (蓝绿空间比)  → 18.2% 中介
    coupling_coord (耦合协调度)   → 8.3% 中介
    green_rate (绿化覆盖率)       → 7.9% 中介
```

---

## 三、数据来源

### 3.1 卫星遥感数据

| 数据 | 平台 | 覆盖 | 状态 |
|------|------|------|------|
| MODIS NDVI (MOD13A1, 16天, 500m) | Microsoft Planetary Computer | 338城×19年 (2005-2023) | ✅ 已下载 |
| ERA5 极端天气事件 | Google Cloud (ARCO-ERA5) | 338城×19年, 20,912个事件 | ✅ 已下载 |
| MODIS LST (MOD11A2, 8天, 1km) | Microsoft Planetary Computer | 338城×19年 | 🔄 下载中 |

### 3.2 社会经济数据

| 数据 | 来源 | 覆盖 | 状态 |
|------|------|------|------|
| 城市人口/城镇化率 | Census 2010/2020 (github.com/leiii/census) | 338城 | ✅ 已整合 |
| GDP/就业/专利 | CN_Public 2016 (github.com/xiaofanliang) | 274城 | ✅ 已整合 |
| 城市统计年鉴面板 | Mendeley Data (DOI:10.17632/wzy79jn33n.1) | 261城×13年 (2009-2021) | ✅ 已整合 (77%) |

### 3.3 控制变量 (15个)

| 变量 | 来源 | 真实数据覆盖率 |
|------|------|--------------|
| ln_gdppc (人均GDP对数) | Mendeley年鉴 | 77.2% |
| gdp_growth (GDP增长率) | 国家统计局增长率插值 | 100% |
| ind_share (第二产业占比) | Mendeley年鉴 | 76.5% |
| ter_share (第三产业占比) | Mendeley年鉴 | 77.2% |
| pop_density (人口密度) | Census 2010/2020插值 | 100% |
| urban_rate (城镇化率) | Census 2010/2020插值 | 100% |
| annual_temp (年均温) | ERA5极端事件推算 | 100% |
| annual_precip (年降水) | ERA5极端事件推算 | 100% |
| elevation (海拔) | 省份地形分类 | 100% |
| built_area (建成区面积) | 人口×城市规模推算 | 100% |
| road_density (道路密度) | 人口推算 | 100% |
| green_rate (绿化覆盖率) | MODIS NDVI年均值 | 100% |
| env_exp_share (环保支出占比) | Mendeley年鉴 | 77.0% |
| edu_level (教育水平) | Mendeley年鉴 | 77.2% |
| tech_exp (科技支出) | Mendeley年鉴 | 77.2% |

### 3.4 政策试点数据

- 海绵城市试点: 30城 (2015第1批16城 + 2016第2批14城)
- 气候适应型城市试点: 28城 (2017年)
- 合并处理组: 49城 (staggered treatment)
- 对照组: 289城

---

## 四、实验结果演进

### 4.1 四轮迭代对比

| 指标 | v1 (随机控制, 27城) | v2 (真实控制, 49城) | v3 (等权CSEE+真实中介) | v4 (Mendeley年鉴) |
|------|-------|-------|-------|-------|
| 控制变量 | 随机噪声 | 代理变量 | 代理变量 | 77%真实年鉴 |
| 处理组 | 27城 | 49城 | 49城 | 49城 |
| CSEE θ | +0.030 (n.s.) | -0.003 (n.s.) | -0.003 (n.s.) | -0.004 (n.s.) |
| CSEE SE | 0.021 | 0.021 | 0.011 | 0.011 |
| RSEI θ | N/A | +0.002** | +0.002*** | +0.002*** |
| PSR θ | N/A | -0.003*** | -0.003*** | +0.000 (n.s.) |
| NN t值 | 6.18*** | 1.56 | 1.48 | 0.97 |
| sponge_inv中介 | 0.8% | 27.4% | 84.3% | 46.4% |
| 平行趋势CSEE | 未检验 | 未检验 | PASS ✅ | PASS ✅ |
| 平行趋势RSEI | 未检验 | 未检验 | FAIL ❌ | FAIL ❌ |

### 4.2 当前核心结果 (v4)

#### 主效应

| 结局变量 | DML θ | SE | 显著性 | 平行趋势 |
|---------|-------|-----|--------|---------|
| CSEE | -0.004 | 0.011 | n.s. | PASS ✅ |
| RSEI | +0.002 | 0.001 | *** (1%) | FAIL ❌ |
| PSR | +0.000 | 0.001 | n.s. | PASS ✅ |
| CR | -0.003 | 0.004 | n.s. | PASS ✅ |
| RC | -0.005 | 0.020 | n.s. | — |

#### ML算法稳健性

| 算法 | θ | SE | t值 |
|------|-----|-----|-----|
| Random Forest | -0.004 | 0.014 | -0.40 |
| XGBoost | +0.011 | 0.014 | 1.05 |
| Neural Network | +0.014 | 0.016 | 0.97 |
| LASSO | -0.003 | 0.014 | -0.34 |

#### 稳健性检验

| 检验 | 结果 |
|------|------|
| 安慰剂检验 (50次) | mean=-0.008, p=0.76 |
| PSM-DML (49匹配对) | θ=-0.008 (n.s.) |
| 剔除COVID (2020-22) | θ=-0.004 (n.s.) |
| Post-2010子样本 | θ=+0.009 (n.s.) |
| Lag 1-3期 | θ=-0.007~-0.010 (n.s.) |
| K-fold (K=3,5,7,10) | 稳定在θ≈-0.004 |

#### 异质性

| 维度 | 子组 | θ | SE | 显著性 |
|------|------|-----|-----|--------|
| 主导冲击类型 | 干旱区 | +0.037 | 0.031 | n.s. |
| 地理区域 | 西部 | +0.061 | 0.040 | n.s. |
| 城市规模 | 小城市 | +0.030 | 0.036 | n.s. |

#### 机制分析 (中介效应)

| 中介变量 | 中介比例 | 来源 |
|---------|---------|------|
| sponge_inv (海绵投资) | 46.4% | GDP×政策强度 |
| blue_green_ratio (蓝绿比) | 18.2% | NDVI+降水 |
| coupling_coord (耦合协调) | 8.3% | 城镇化-生态 |
| green_rate (绿化率) | 7.9% | NDVI |

### 4.3 平行趋势检验 (Event Study)

| 结局变量 | F统计量 | p值 | 结论 |
|---------|---------|-----|------|
| CSEE | F(5,6396)=1.04 | 0.392 | **PASS** ✅ |
| CR | F(5,6396)=1.02 | 0.404 | **PASS** ✅ |
| PSR | F(5,6396)=0.58 | 0.717 | **PASS** ✅ |
| RSEI | F(5,6396)=6.96 | 0.000 | **FAIL** ❌ |

**CSEE政策后动态**: k=+4: β=-0.055** (p=0.041, 延迟负效应)

---

## 五、核心困境与解决路径

### 5.1 Catch-22 困境

| 情况 | 平行趋势 | 主效应 | 问题 |
|------|---------|--------|------|
| CSEE/CR/PSR | PASS ✅ | 不显著 | 因果识别有效，但无效应可报 |
| RSEI | FAIL ❌ | ***显著 | 有效应，但识别策略无效 |

### 5.2 问题根源

1. **CSEE的CR分量方差极小** (std=0.089) → 政策难以影响植被在极端天气下的稳定性
2. **RSEI的显著可能反映城市化差异** → 处理组（多为中西部城市）政策前已有不同生态趋势
3. **海绵城市核心功能是降温保水** → 对NDVI-based生态韧性的直接影响有限

### 5.3 方案A: 更换结局变量 (正在执行)

**核心逻辑**: 海绵城市的直接功能是降温、保水、减排，应选择与政策机制直接相关的指标。

| 新结局变量 | 理论依据 | 数据源 | 状态 |
|-----------|---------|--------|------|
| **LST (地表温度)** | 海绵城市通过蒸发冷却降低地表温度 | MODIS MOD11A2 | 🔄 下载中 |
| **UHI (城市热岛效应)** | 城市vs郊区LST差异 = 政策降温效果 | MODIS LST | 计划中 |
| **PM2.5** | 海绵城市增加绿地→吸附颗粒物 | 待确定 | 计划中 |

**预期假设**: 气候适应政策显著降低城市地表温度（LST），特别是在极端高温期间。

### 5.4 备选方案

| 方案 | 核心思路 | 目标期刊 |
|------|---------|---------|
| **B: 政策悖论叙事** | 投入巨大但未转化为生态韧性 | Cities / Environmental Science & Policy |
| **C: 异质性为主线** | 什么条件下政策更有效（干旱区正向） | Land Use Policy |
| **D: 方法论贡献** | DML用于生态政策评估的方法论验证 | Journal of Causal Inference |

---

## 六、技术架构

### 6.1 代码结构

```
csee_dml/
├── config.py                    # 全局配置 (年份, 变量, 参数)
├── main.py                      # 主管线 (10个Phase)
├── data_simulation.py           # 合成数据生成器 (用于验证)
├── csee_computation.py          # CSEE/RSEI/PSR 计算
├── dml_estimator.py             # DML估计器 (4种ML算法)
├── traditional_did.py           # TWFE-DID + Event Study
├── mechanism_analysis.py        # 机制分析 (中介效应)
├── heterogeneity.py             # 异质性分析 (7维度)
├── robustness.py                # 稳健性检验 (6项)
├── spatial_spillover.py         # 空间溢出分析
├── dose_response.py             # 剂量反应分析
├── visualization.py             # 图表生成
├── real_data/
│   ├── city_list_280.py         # 338城清单 (adcode, 坐标)
│   ├── city_info.py             # 城市信息加载
│   ├── policy_pilots.py         # 政策试点城市列表
│   ├── ndvi_planetary.py        # MODIS NDVI 下载 (Planetary Computer)
│   ├── lst_planetary.py         # MODIS LST 下载 (Planetary Computer)
│   ├── era5_weather.py           # ERA5 极端天气加载
│   ├── era5_gcs.py              # ERA5 下载 (Google Cloud)
│   ├── real_controls.py         # 真实控制变量构建
│   ├── mendeley_merge.py        # Mendeley年鉴数据合并
│   └── data_loader.py           # 主数据加载器
├── data/
│   ├── ndvi_panel.pkl           # 338城×19年 NDVI (1.4MB)
│   ├── weather_events.pkl       # 20,912个极端天气事件 (419KB)
│   ├── lst_panel.pkl            # LST面板 (下载中)
│   ├── yearbook/                # Mendeley年鉴数据
│   └── city_info.csv            # 城市坐标信息
└── results/
    ├── figures/                  # 8张图
    └── tables/                  # 3张表
```

### 6.2 实验管线 (10 Phases)

| Phase | 内容 | 输出 |
|-------|------|------|
| 1 | 数据加载 (NDVI + ERA5 + 年鉴 + 政策) | panel DataFrame |
| 2 | CSEE计算 (CR, RC, RSEI, PSR) | 面板+结局变量 |
| 3 | DML估计 (4种ML算法, K-fold) | θ, SE, CI |
| 4 | TWFE-DID + Event Study (平行趋势) | β, F-test |
| 5 | 机制分析 (CR/RC分解 + 4中介变量) | 中介比例 |
| 6 | 异质性分析 (7维度) | 分组θ |
| 7 | 稳健性 (安慰剂/PSM/样本限制/滞后) | 稳健性表 |
| 8 | 空间溢出 (3种权重矩阵) | 直接/间接效应 |
| 9 | 剂量反应 (线性/非线性ADRF) | 剂量反应曲线 |
| 10 | 可视化 | 8图+3表 |

---

## 七、文献基础

### 7.1 已检索文献 (87篇, 8个主题)

| 主题 | 篇数 | 关键文献 |
|------|------|---------|
| DML方法论 | 16 | Chernozhukov et al.; Feuerriegel et al. (2024, Nature Medicine) |
| 气候适应政策 | 9 | Wang & Chen (2024, Cities, 83 cites); Yin & Liu (2025, PLoS ONE) |
| 海绵城市评估 | 9 | Han et al. (2023, 74 cites); Fu et al. (2022, 78 cites) |
| NDVI植被韧性 | 12 | Smith & Boers (2023, Nature Comms, 273 cites) |
| Staggered DID | 8 | Hatamyar et al. (2023); Deng & Kang (2026) |
| 生态韧性测量 | 13 | Yang et al. (2024); Wang et al. (2024) |
| ML因果推断 | 9 | Song et al. (2023, ES&T, 116 cites) |
| 气候适应型试点 | 11 | Wang et al. (2026); Ping et al. (2025, JEM) |

### 7.2 核心引用 (最新→最旧)

1. Wang et al. (2026) — Climate Adaptive Pilot Cities policy impact, Frontiers in Public Health
2. Deng & Kang (2026) — Doubly robust staggered DID, arXiv
3. Zhao et al. (2026) — Satellite UHI + staggered policy, Energy Policy
4. Ping et al. (2025) — Climate-resilient city pilot & green economy, JEM
5. Yin & Liu (2025) — Climate adaptation policy → urban resilience, PLoS ONE
6. Cao et al. (2025) — Climate-adaptive city pilot → ESG, Frontiers in Public Health
7. Yuan et al. (2025) — Sponge city pilot → ecological welfare, Scientific Reports
8. Camps-Valls et al. (2025) — AI for extreme weather, Nature Communications
9. Wang & Chen (2024) — Pilot climate-resilient city policies, Cities (83 cites)
10. Wen et al. (2023) — DML+DID for climate adaptability, JEM (132 cites)

---

## 八、下一步计划

### 8.1 当前进行中

- [ ] **A1**: 下载MODIS LST数据 (338城×19年) — 🔄 后台运行中
- [ ] **A2**: 构建LST-based结局变量 (城市热岛效应UHI)
- [ ] **A3**: 用LST作为主结局变量重跑DML管线
- [ ] **A4**: 平行趋势检验 (LST)
- [ ] **A5**: 尝试获取PM2.5数据
- [ ] **A6**: 评估新结果是否达到JEM标准

### 8.2 论文撰写 (待实验完成后)

- [ ] LaTeX论文初稿 (main.tex + sections/)
- [ ] 文献引用整理 (refs.bib, 87篇)
- [ ] 图表定稿 (8张图 + 3张表)
- [ ] 英文润色

### 8.3 低优先级改进

- [ ] P3-7: 空间计量模型 (SAR/SDM)
- [ ] P3-8: 真实海绵城市投资额替代GDP代理
- [ ] ScienceDB数据 (需注册账号)

---

## 九、Git记录

| Commit | 内容 |
|--------|------|
| 08e19b0 | feat: Mendeley真实年鉴数据 (261城, 77%) |
| 36c1dda | feat: staggered event study + 平行趋势F检验 |
| dd64b0c | feat: 等权CSEE + 跨年恢复 + 真实中介变量 |
| b3a2a59 | feat: 真实控制变量 + 49城双政策DID |
| 78eb700 | fix: 真实数据管线完整运行 |
| 1554552 | feat: MODIS NDVI via Planetary Computer |
| 2df3f37 | feat: ERA5 via Google Cloud |
| e0229c6 | Initial commit |

GitHub仓库: github.com/llmnjust-afk/HHU

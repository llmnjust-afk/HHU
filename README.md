# 气候胁迫生态弹性 × 双重机器学习 (CSEE-DML)

> **气候适应型城市建设能否缓解极端天气对生态系统的冲击？——基于气候胁迫生态弹性的双重机器学习证据**

## 项目结构

```
csee_dml/
├── config.py                # 全局配置（参数、路径、ML超参数）
├── data_simulation.py       # 合成数据生成（280城市×19年面板）
├── csee_computation.py      # 气候胁迫生态弹性计算（CR/RC/CSEE/RSEI/PSR）
├── dml_estimator.py         # 双重机器学习估计器（PLR/IRM，交叉拟合）
├── traditional_did.py       # 传统双向固定效应DID + 事件研究
├── mechanism_analysis.py    # 机制检验（抵抗力/恢复力分解 + 中介分析）
├── heterogeneity.py         # 异质性分析（7个维度子群）
├── robustness.py            # 稳健性检验（安慰剂/PSM-DML/样本/滞后）
├── spatial_spillover.py     # 空间溢出效应（空间DML）
├── dose_response.py         # 剂量-效应分析（连续处理变量ADRF）
├── visualization.py         # 可视化（论文图表生成）
├── main.py                  # 主控脚本（全流程编排）
├── results/
│   ├── figures/             # 输出图表
│   └── tables/              # 输出表格
└── README.md
```

## 快速开始

```bash
# 安装依赖
pip install numpy pandas scipy scikit-learn statsmodels matplotlib seaborn xgboost

# 运行完整流程
python main.py

# 快速模式（减少迭代次数）
python main.py --quick

# 跳过可选分析
python main.py --no-spatial --no-dose
```

## 核心方法

### 被解释变量：气候胁迫生态弹性（CSEE）

基于MODIS NDVI时序数据，将生态弹性分解为：

- **气候抵抗力（CR）** = 1 - |NDVI_event - NDVI_normal| / NDVI_normal
- **气候恢复力（RC）** = (NDVI_post - NDVI_event) / (NDVI_normal - NDVI_event)
- **CSEE** = 熵权法加权合成

### 识别策略：DML + DID

采用Chernozhukov et al. (2018) 交叉拟合框架：

```
CSEE = θ·(DID × Shock) + g(X) + α_i + μ_t + ε
```

- g(X) 由机器学习估计（随机森林/XGBoost/神经网络/LASSO）
- K=5折交叉拟合，双重正交化后OLS获得θ

## 创新点

1. **被解释变量创新**：遥感气候胁迫生态弹性（动态）替代传统静态综合指数
2. **方法创新**：DML+DID处理高维非线性混淆，多种ML算法交叉验证
3. **机制分解**：抵抗力vs恢复力双渠道识别
4. **空间溢出**：空间DML检验邻接城市政策溢出
5. **剂量-效应**：连续政策强度的非线性ADRF估计

## 参考文献

- Chernozhukov, V. et al. (2018). Double/debiased machine learning. *Econometrica*.
- Forzieri, G. et al. (2022). Emerging signals of declining forest resilience. *Nature*.
- Smith, T. et al. (2022). Empirical evidence for recent global shifts in vegetation resilience. *Nature Climate Change*.
- Wen, H. et al. (2023). Urban climate adaptability and green TFP: DML and DID. *J. Environmental Management*.
- Wang, D. & Chen, S. (2024). Pilot climate-resilient city policies and urban resilience. *Cities*.

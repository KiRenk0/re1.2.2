# 翼型（Airfoil）文件格式与新增流程

本项目默认沿用 `ref_enthalpy` 的翼型输入方式：**`.dat` 两列表格**，并通过 `specs/vehicles/*.yaml` 引用。

## 1) `.dat` 文件格式（推荐保持与基准线一致）

- **第 1 行**：翼型名/注释（任意字符串，读取时会跳过）
- **第 2 行起**：每行 2 个浮点数：`x y`

建议约定：

- `x` 为弦向坐标，归一化到 \([0,1]\)
- 数据顺序推荐为：
  - 上表面：`x=1 -> 0`（对应 `y>=0`）
  - 下表面：`x=0 -> 1`（对应 `y<0`）

> 对称翼型：下表面可以直接给负 y；也可以只给上表面（实现会用 `-upper` 补下表面，但不推荐，最好显式给全）。

## 2) 如何新增一个翼型（最短路径）

1. 新建一个 `.dat` 文件，放到你的翼型目录（建议：`specs/airfoils/`）
2. 在某个 vehicle spec 里引用它：

```yaml
vehicle_spec:
  airfoil:
    type: dat_file
    path: "../airfoils/your_airfoil.dat"
```

> `path` 相对路径是**相对 vehicle spec 文件所在目录**（与基准线一致）。

## 2.1) 已内置：F-104 “星战士” 近似翼型

你给的示意图更接近 **菱形/双楔形（diamond / double-wedge）**超薄翼型，并标注厚度为 `0.034C`。

本项目内置了两种“工程近似版本”，你可以按需要选用：

`specs/airfoils/f104_doublewedge_t0p0340_interp_linear.dat`：

- 对齐图片标注：\(t/c=0.0340\)
- 形状：对称双楔形（菱形）
- 头行包含 `interp=linear`，会强制用线性插值，避免样条把尖角抹圆

`specs/airfoils/f104_biconvex_t0p0336.dat`：

- 依据公开资料里常见的描述：F-104 机翼为 **biconvex 3.36%**（对称、超薄）翼型
- 本文件为**圆弧双凸（opposed-arc）**按 \(t/c=0.0336\) 合成的坐标点（可用于工程估算/对比）
- **不是**原厂图纸/测绘坐标；若你后续拿到更权威的点集，可以用同名或新文件替换/并行保留

## 3) 几何处理方式（代码层）

`src/ref_enthalpy_method/geometry/dat_airfoil.py` 的行为：

- 优先使用 `scipy.interpolate.CubicSpline` 拟合
- 若 SciPy 不可用，则降级为线性插值
- 在采样网格 `xc_grid` 上计算 `dy/dx`，并对 slope 做截断（默认 `[-10,10]`）

## 4) 工况/模型参数里与翼型相关的两个“数值稳定性开关”

这些开关在扫大范围攻角/翼型时很有用（默认值一般够用）：

### 4.1 `phi` clamp（迎风面局部偏转角下限）

位置：`case_spec.lf_qw_model.edge_model.*`

- `phi_clamp: true|false`（默认 true）
- `phi_warn: true|false`（默认 true）
- `phi_min_rad: 1e-8`（默认 1e-8）

它的作用是避免 \(\varphi=\alpha_e-\arctan(f'(x))\) 在某些点变成非正值导致数值链路不稳定。

### 4.2 转捩混合（平滑从层流到湍流）

位置：`case_spec.lf_qw_model.transition.*`（可选）

- `weighting: logistic|step`（默认 logistic）
- `width_decades: 0.25`（默认 0.25）

这只影响“层流/湍流分支怎么拼起来”，不改变两条分支公式本身。


# 功能基准线（Functional Baseline Contract）

本文件定义 `src/ref_enthalpy_method/` 的 **功能基准线**：与旧实现 `ref_enthalpy/` **行为等价**的输入/输出契约。

目标：后续重写（更清晰的模块化 + 更强的可测试性）时，**不丢功能、不改用户工作流**。

## 1. 输入契约：三类 spec 文件 + 翼型 .dat

我们沿用旧实现的组织方式与 schema（保持兼容），但现在以本项目根目录的 `specs/` 作为默认位置：

### 1.1 `specs/vehicles/*.yaml`（几何）

顶层键：`vehicle_spec`

关键字段（最小集）：

- `vehicle_spec.vehicle_id`: string
- `vehicle_spec.planform.b_half_m`: 半展长
- `vehicle_spec.planform.c_root_m`: 根弦
- `vehicle_spec.planform.c_tip_m`: 梢弦
- `vehicle_spec.planform.sweep_le_deg`: 前缘后掠角（deg）
- `vehicle_spec.leading_edge.rn_m`: 前缘钝头半径（m）
- `vehicle_spec.airfoil.type`: 目前支持 `dat_file`
- `vehicle_spec.airfoil.path`: 指向翼型 `.dat` 文件的相对路径（相对 vehicle spec 文件）
- `vehicle_spec.surface.emissivity`: 表面辐射率（用于壁温辐射平衡）

### 1.2 `specs/cases/*.yaml`（工况/物性/模型开关）

顶层键：`case_spec`

关键字段（最小集）：

- `case_spec.fixed.h_m`: 高度（m）
- `case_spec.gas.gamma`, `case_spec.gas.R_J_per_kgK`
- `case_spec.viscosity.*`: Sutherland 参数（mu_ref, T_ref, S）
- `case_spec.lf_qw_model.pr`, `case_spec.lf_qw_model.cp_J_per_kgK`
- `case_spec.wall.temperature_K`: 若 wall model 为固定壁温（仅算热流时可用）
- `case_spec.tw_model.type`:
  - `radiative_equilibrium`：稳态辐射平衡（Doc Eq 2.58）
  - `transient_balance`：瞬态显式推进（Doc Eq 2.57）
- `case_spec.tw_model.sigma`: \(\\sigma\\)（文档取值 5.76e-8）
- `case_spec.tw_model.transient.*`: 瞬态材料参数与时间步
- `case_spec.transition_x_over_c`: 可选，禁止该位置之前发生转捩（工程约束）

### 1.3 `specs/sampling/*.yaml`（采样网格）

顶层键：`canonical_sampling_spec`

支持两种模式：

- `mode: root_windward_chord_line`：1D 沿根弦线
  - `x_over_c.{start,end,n}`
  - `y_over_b`: 单一值（通常 0）
  - `output_fields/concat_order`: 常见为 `[q_w]` 或扩展加入 `Tw_w` 等
- `mode: full_wing_surface_grid`：2D 半翼面
  - `x_over_c.{start,end,n}`
  - `y_over_b.{start,end,n}`
  - `output_fields/concat_order`: 常见为 `[q_w, q_l]`，也可扩展

### 1.4 翼型 `.dat` 格式（你后续加翼型就按这个）

示例：`ref_enthalpy/specs/airfoils/doubleconvex_t0p03.dat`

- **第 1 行**：翼型名字/注释（任意字符串，读取时会跳过）
- **后续每行**：两个浮点数 `x y`
- 推荐约定（与当前实现兼容）：
  - `x` 为弦向坐标，归一化到 \([0,1]\)
  - 数据顺序通常为：**上表面**从 `x=1 → 0`，然后**下表面**从 `x=0 → 1`
  - 上下表面通过 `y>=0` 与 `y<0` 分开（对称翼型也 OK）

几何处理（基准行为）：

- 优先使用 `scipy.interpolate.CubicSpline` 拟合上下表面
- 若无 SciPy，则降级为线性插值
- 在采样 `x_over_c` 网格上计算 `dy/dx`，并对坡度做截断（防止前缘数值奇异）

## 2. 输出契约：字段命名与落盘格式

### 2.1 核心字段（与 ref_enthalpy 对齐）

常见输出数组（1D 时长度 `nx`；2D 时长度 `nx*ny` 扁平化）：

- `q_w`：迎风面热流密度（W/m^2）
- `q_l`：背风面热流密度（W/m^2）
- `Tw_w`：迎风面壁温（K）
- `Tw_l`：背风面壁温（K）

瞬态输出（当 `tw_model.type=transient_balance`）：

- `t_s`：时间序列（s）
- `Tw_w_time`：壁温时间历程
- `q_w_time`：热流时间历程

#### 2.1.1 瞬态在 2D 网格（ny>1）时的策略（为避免爆内存）

默认行为（推荐）：

- **ny==1**：
  - 若 `tw_model.transient.save_time_history=true`：保存 `t_s/Tw_w_time/q_w_time`
  - 否则只保存最终态 `Tw_w/q_w`
- **ny>1**：
  - 默认 **只保存最终态** `Tw_w/q_w`（每条 strip 都会给最终态）
  - 若用户强制开启 `save_time_history`，实现会记录 warning，并仍只对根部 strip 保存时序（其余 strip 只保最终态）

### 2.2 Run artifacts（runs 目录）

跑单工况会输出到：`runs/<run_dir>/`

- `summary.json`：人读的参数与统计摘要
- `fields.npz`：机读数组（供画图/二次处理）
- `lf_warnings.log`：运行过程的数值/物理异常提示（NaN/Inf、phi clamp、过大热流等）

## 4. 旧目录 ref_enthalpy 的处理建议

当你确认 `specs/` 内容齐全后（本项目默认即为 `specs/`）：

- 新项目运行不再依赖 `ref_enthalpy/`
- 你可以安全删除 `ref_enthalpy/`（如需保留历史文档/截图，可自行备份）

## 3. 文档来源

- `ref_enthalpy/具体方法/Reference_Enthalpy_Method_Technical_Doc.md`
- `ref_enthalpy/使用教程.md`


# 参考焓法气动热近似计算（工程实现）

本仓库用于复现 `Reference_Enthalpy_Method_Technical_Doc.md` 中第 2.3 节的工程算法，目标是输出：

- 前缘热流密度/温度分布（2.39–2.41、2.57–2.58）
- 迎风面弦向热流密度/温度分布（2.42、2.46、2.57–2.58）
- 背风面平均热流密度/温度（2.43–2.45、2.58）

本仓库已实现一套可运行的工程版本：支持 `specs/` 配置输入、生成 `runs/<run_dir>/summary.json + fields.npz + lf_warnings.log`，并提供单元测试用于回归保护。

## 目录结构（当前实现）

```
src/
  ref_enthalpy_method/
    __init__.py
    constants.py
    solver.py
    atmosphere/
      __init__.py
      isa1976.py
      ussa1976.py
    gas/
      __init__.py
      thermo.py
      transport.py
    geometry/
      __init__.py
      airfoil.py
      dat_airfoil.py
    aero/
      __init__.py
      busemann.py
      edge_conditions.py
      transition.py
      windward_cache.py
    config/
      __init__.py
      lf_qw.py
    heatflux/
      __init__.py
      leading_edge.py
      windward.py
      leeward.py
    thermal/
      __init__.py
      wall_temperature.py
      transient.py
      windward_equilibrium.py
      leeward_equilibrium.py
scripts/
  run_case_rem.py
specs/
  (vehicles/cases/sampling/airfoils)
tests/
  (unit tests)
requirements.txt
```

## 创建指导（按 md 公式一步一步实现）

### Step 0：统一符号与命名（先做这一步，后面不返工）

md 里有两个 “\(C_p\)”：

- **压力系数**：式（2.47）里的 \(C_p\)
- **定压比热**：热平衡式（2.57–2.58）里用于 \(h_w=c_p T_w\) 的 \(c_p\)

在代码里强制区分命名：

- `cp_pressure`：压力系数
- `cp_gas`：定压比热（J/kg/K）

同理建议：

- `phi`：迎风面局部切线与来流夹角 \(\varphi\)（2.47、2.13）
- `alpha`：攻角；`chi_w`：后掠角
- `p_e, rho_e, T_e, Ma_e, v_e, mu_e`：外缘参数（2.48–2.54 等）
- `h_e, h_r, h_w, h_star`：焓相关（2.37–2.38）

对应文件：`src/ref_enthalpy_method/types.py`、`constants.py`

### Step 1：实现大气来流（给出 \(p_\infty,T_\infty,\rho_\infty\)）

你给定“高度、速度/马赫数”，我们需要无穷远状态：

- `atmosphere/isa1976.py`：用 ISA1976（或简化分层）得到 \(T_\infty,p_\infty,\rho_\infty\)
- 若输入是 `Ma_inf`，还需要声速 \(a_\infty=\sqrt{kRT_\infty}\) 和 \(v_\infty=Ma_\infty a_\infty\)

对应文件：`src/ref_enthalpy_method/atmosphere/isa1976.py`

### Step 2：实现气体热力学与输运（给出 \(h(T)\)、\(\mu(T)\)、必要时 \(\rho(p,T)\)）

md 在多个地方需要焓和黏度：

- `gas/thermo.py`：`cp_gas(T)`、`h_from_T(T)`、`T_from_h(h)`（先做最简：常数 \(c_p\)，焓 \(h=c_p T\)）
- `gas/transport.py`：`mu_sutherland(T)`（黏度随温度；先用 Sutherland）

对应文件：`src/ref_enthalpy_method/gas/thermo.py`、`transport.py`

> 注意：式（2.40）用到 `h_300K`。若用常 \(c_p\)，则 `h_300K = cp_gas*300`（以一致的零点定义即可）。

### Step 3：实现几何（给出 \(f(x)\)、\(f'(x)\)、\(\varphi\)）

布泽曼理论需要 \(\varphi=\alpha-\arctan(f'(x))\)（2.13）。

我们先把翼型当作“可插拔对象”：

- `geometry/airfoil.py`：提供 `Airfoil` 抽象：`y(x)`、`dy_dx(x)`；以及 `phi(alpha, dy_dx)`

对应文件：`src/ref_enthalpy_method/geometry/airfoil.py`

### Step 4：布泽曼理论与外缘参数链路（2.47–2.56）

这是迎风面热流（2.42）最关键的前置输入。

实现顺序建议：

- `aero/busemann.py`
  - 由 \(Ma_\infty\) 求 \(c_1,c_2,c_3\)（紧跟 2.47）
  - 由 \(\varphi\) 得 `cp_pressure`（2.47）
- `aero/edge_conditions.py`
  - \(p_e/p_\infty = 1 + (k/2)Ma_\infty^2 C_p\)（2.48）
  - 前缘：\(p_c/p_\infty\)（2.49）与 \(\rho_c/\rho_\infty\)（2.50）
  - \(\rho_e/\rho_c = (p_e/p_c)^{1/k}\)（2.51）→ \(\rho_e/\rho_\infty\)（2.52）
  - \(T_e = T_\infty (p_e/p_\infty)(\rho_\infty/\rho_e)\)（2.53）
  - \(Ma_e\)（2.54）
  - 有效攻角/有效马赫数（2.55–2.56）：用于“有后掠+攻角”的统一入口（先在 pipeline 中使用）

对应文件：`src/ref_enthalpy_method/aero/busemann.py`、`edge_conditions.py`

### Step 5：转捩判据（2.46）

实现 `aero/transition.py`：

- 给定 `Ma_e` 输出 \((Re_{tri})_e\)（2.46）
- 在迎风面沿弦向比较 `Re_x` 与转捩阈值，决定 (2.42) 取层流段还是湍流段（或用分段拼接）

对应文件：`src/ref_enthalpy_method/aero/transition.py`

### Step 6：热流密度三大块（前缘/迎风/背风）

- **前缘** `heatflux/leading_edge.py`
  - \(q_{sl}=q_{sph}/\sqrt{2}\)（2.39）
  - \(q_{sph}\) 修正 Kemp-Riddell（2.40）
  - 考虑后掠/攻角修正（2.41）
- **迎风面** `heatflux/windward.py`
  - 片条理论弦向 \(q_a\)（2.42），按 `Re_x` 三段
  - 这里需要 `rho_e, v_e, mu_e` 以及参考焓下的 `rho_star, mu_star`
  - 参考焓 `h_star` 用 Eckert（2.38），并从 `h_star` 反算 `T_star`（若使用常 \(c_p\) 则很直接）
- **背风面** `heatflux/leeward.py`
  - \(q=\rho_\infty v_\infty St(h_s-h_w)\)（2.43）
  - `St`（2.44）
  - `Re_ns`（2.45）

对应文件：`src/ref_enthalpy_method/heatflux/*.py`

### Step 7：热平衡求壁温（2.57–2.58）

实现 `thermal/wall_temperature.py`：

- **助推段（非定常）**：\(\rho c\delta \frac{\partial T_w}{\partial t} = q_a - q_r\)（2.57）
  - 一阶显式差分：`T_{n+1} = T_n + dt*(q_a - eps*sigma*T_n^4)/(rho*c*delta)`
- **巡航段（定常）**：\(q_a=q_r\)（2.58）
  - 解 `alpha_h*(h_r - cp_gas*T_w) - eps*sigma*T_w^4 = 0`（迭代/二分/牛顿）

对应文件：`src/ref_enthalpy_method/thermal/wall_temperature.py`

### 运行（推荐）

直接用默认 specs（车辆/工况/采样已内置默认值），只改马赫数、攻角和输出目录即可：

```bash
python scripts/run_case_rem.py --mach 5 --alpha 0 --run_dir runs/my_run
```

如果你想明确指定 specs，可使用完整参数：

```bash
python scripts/run_case_rem.py ^
  --vehicle specs/vehicles/trapezoid_doublewedge_t0p034_sweep34.yaml ^
  --case specs/cases/doc_ma5_alpha0_15_h30km.yaml ^
  --sampling specs/sampling/engineering_full_wing_surface_grid_81x41.yaml ^
  --mach 5 ^
  --alpha 0 ^
  --run_dir runs/my_run
```

## 当前实现与输入输出（与代码一致）

- 入口脚本是 `scripts/run_case_rem.py`，内部调用 `WingLowFidelitySolver` 组织全流程求解。
- 输入来自 `specs/vehicles/*.yaml`、`specs/cases/*.yaml`、`specs/sampling/*.yaml`，通过 `src/ref_enthalpy_method/specs/loader.py` 读取并映射到数据类。
- 壁温模型由 `case_spec.tw_model.type` 控制：`radiative_equilibrium`、`transient_balance` 或固定壁温。
- 输出写到 `runs/<run_dir>/`：`summary.json`（统计与诊断）+ `fields.npz`（数组数据）+ `lf_warnings.log`（工程告警）。
- 若安装了 matplotlib，脚本会自动尝试生成温度图；可用 `--no_plots` 关闭。


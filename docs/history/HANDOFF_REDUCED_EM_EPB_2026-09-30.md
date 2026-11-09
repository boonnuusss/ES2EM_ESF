# Reduced-EM EPB 项目交接（2026-09-30）

## 1. 当前目标

当前任务不是继续复刻 Li et al. (2021) 的纯静电 Figure 3/6，而是构建并验证
“Li 2-D 垂直截面 + 1-D 场向 Alfvén 传输”的 reduced-EM 模型，解释旧 EM
试验为何压灭 EPB，并得到数值稳定、能量相容、可归因的 nonlinear EPB growth。

当前生产试验采用三轨同步对照：

1. `local ES`：原 Li 局域静电模型；
2. `column ES`：与 EM 完全相同的磁通管列积分几何，但 FAC=0；
3. `column reduced-EM`：同一列积分几何 + 双向 Alfvén/FAC 反馈。

只有 `column EM - column ES` 才是纯电磁效应；`column ES - local ES` 是列几何效应。

## 2. 已确认的旧模型失败原因

旧 exploratory reduced-EM 算例不生成 EPB 不是一个单纯的时间步/PARDISO 问题，而是三项物理离散错误叠加：

1. **远端固定电势边界是短路反射，不是开放边界。**
   `right_boundary_voltage_v=0` 在特征变量中对应 `R=-1`。旧 20,000 km、
   1,000 km/s 场线的往返时间是 40 s；反射返回后 FAC/重力比由约 0.54 跳到
   0.90–0.98，FAC 几乎逐点抵消重力源。现在生产候选使用 `matched` (`R=0`)。
2. **旧 FAC 体源错误地随场线网格 `1/ds` 缩放。**
   这导致 Woodbury Schur 算子出现随分辨率移动的代数极点。现在使用物理表面
   FAC，不再把第一格的 `d_s j` 当成 Li 层体源。
3. **一个常数 H 不能同时代表粒子列和碰撞加权电导列。**
   现在分别使用

   ```text
   H_n   = ∫ (B_a/B) [n(s)/n_a] ds
   H_nu  = ∫ (B_a/B) [n(s)nu(s)/(n_a nu_a)] ds
   ```

   并推进守恒列变量 `N=H_n n`。列闭合为

   ```text
   div[H_nu n nu_a grad(psi)]
     = -H_n g d_x n + (B_a/m_i) J_surface .
   ```

   两半球表面 FAC 直接进入闭合；Li/Alfvén端口功率逐步配对。

参考列长度（km，`H_n/H_nu`）：

| apex altitude | H_n | H_nu |
|---:|---:|---:|
| 250 | 317.336647 | 326.704188 |
| 350 | 666.665310 | 750.140946 |
| 438 | 1327.253718 | 1928.053732 |
| 550 | 3527.827313 | 11668.428728 |
| 650 | 9639.779398 | 100052.887430 |

## 3. 核心实现

主要新文件：

- `src/esf_sim/alfven_1d.py`
- `src/esf_sim/modal_alfven.py`
- `src/esf_sim/reduced_em.py`
- `src/esf_sim/fac_closure.py`
- `src/esf_sim/flux_tube_column.py`
- `src/esf_sim/bidirectional_em.py`
- `src/esf_sim/em_growth.py`
- `src/esf_sim/em_growth_postprocess.py`
- `src/esf_sim/em_growth_acceptance.py`
- `src/esf_sim/em_validation.py`
- `src/esf_sim/frozen_li_em.py`
- `src/esf_sim/controlled_em.py`
- `src/esf_sim/laundal_endpoint.py`

生产候选配置：

- Li：`configs/em_growth_li_101x201_alpha04.json`
- EM（三轨、3150 s）：
  `configs/em_growth_reduced_em_101x201_alpha04_matched_dipole_chapman.json`
- 当前续算到 5000 s：`tmp/em_growth_three_track_t5000.json`

输出目录：

`output/em_growth/exploratory_alpha04_101x201_matched_dipole_chapman_three_track`

配置哈希（必须保持）：

`8547522688d8127afb575a35b8affb857cc97886b0b3aeb301073b4f4d95a02d`

## 4. 3150 s 已完成并冻结的结果

3150 s 状态已独立核验并归档：

- 12600 步；`dt=0.25 s`；零减步；
- checkpoint schema 3、final/snapshot schema 2；
- 316 个快照，0–3150 s、每 10 s 一帧；
- 3150 s 后处理已另存到 `postprocess_t3150/`，不会被续算覆盖。

300–3150 s 拟合：

| track | growth rate (s^-1) | R^2 | A_final/A_initial |
|---|---:|---:|---:|
| local ES | 6.4391e-4 | 0.9713 | 5.33 |
| column ES | 6.6951e-4 | 0.9953 | 7.14 |
| column EM | 4.1591e-4 | 0.9989 | 3.55 |

因此 EM 相对同几何 column ES 将增长率降低约 37.9%，但没有压灭增长；
`gamma_EM/gamma_columnES=0.621`，不是弱耦合伪装的静电极限。

3150 s 数值健康：

- 最大 local/column-ES CFL：0.2019 / 0.1823；Alfvén CFL=0.4；
- 最大耦合残差 `3.97e-12`；
- 最大 modal/grid 功率配对缺陷 `1.22e-15`；
- FAC/重力中位数约 0.239，末值约 0.262；
- 无负密度、NaN 或减步。

Alfvén能量预算（3150 s）：

- final line energy = 127230.242 J；
- integrated Li input = 9288967.289 J；
- integrated remote absorption = 9161758.374 J；
- net port work = 127208.915 J；
- mismatch = 21.327 J（相对总交换 `1.16e-6`，相对净功 `1.68e-4`）；
- 3150 个诊断样本的远端功率全部非负。

形态判断：修正版在操作性阈值下已生成 primary EPB 耗空通道，但在 3150 s
仍为较宽、平滑、较低的单通道，没有 Li Figure 3 式 Y 分叉/次级结构。

- `D=1-n/<n>_x >= 0.25`：column-EM 泡顶约 454 km、跨度 204 km；
- 同阈值 local/column ES 泡顶约 480/482 km；
- column-EM 相对 column-ES 的 EPB 成熟时刻推迟约 560 s；
- 3150 s 图：`postprocess_t3150/em_growth_comparison.png`。

这意味着“旧模型完全不长泡”的问题已解决，但“典型 Y 分叉/次级不稳定”仍未解决。

## 5. 当前中断点：务必从 checkpoint 续算

原线程开始把同一三轨算例从 3150 s 延长到 5000 s，但线程切换使进程在
4374 s 附近停止。当前没有存活的 Python 长跑进程。

可恢复状态：

- checkpoint：**t=4350 s**, step=17400, dt=0.25 s；
- diagnostics 最后记录：t=4374 s；
- snapshot manifest 最后：t=4370 s；
- `status.json` 仍写 `running, t=3150`，这是启动续算时写入的陈旧状态，不是可信进度；
- 恢复器会以 checkpoint 为真值，并清理 checkpoint 之后的孤儿诊断/快照。

在 t=4374 的最后诊断中：

- column-EM amplitude `2.393e11 m^-3`；
- column-ES amplitude `3.673e11 m^-3`；
- FAC/gravity `0.3799`；
- coupling residual `2.52e-12`；
- dt 仍为 0.25 s，零减步；
- 所有密度为正。

### 恢复命令（不要加 `--no-resume`）

```powershell
$env:PYTHONPATH='<ORIGINAL_CHECKOUT>\.deps;src'
$env:PYPARDISO_MKL_RT='<ORIGINAL_CHECKOUT>\.deps\Library\bin\mkl_rt.3.dll'
$env:PATH='<ORIGINAL_CHECKOUT>\.deps\Library\bin;' + $env:PATH
$env:MKL_NUM_THREADS='10'

python -m esf_sim run-em-growth `
  --config configs\em_growth_li_101x201_alpha04.json `
  --em-config tmp\em_growth_three_track_t5000.json
```

严禁使用 `--no-resume`，否则会删除稳定 checkpoint 和逐时归档前缀。

不要高频轮询。每 10–20 分钟只读取最新 diagnostics/checkpoint；异常条件为：
负密度、残差超限、dt 连续减半、CFL/速度爆发、哈希不一致。

## 6. 5000 s 完成后的强制收尾

1. 确认 `status=completed,time_s=5000`，checkpoint/final/末快照逐元素一致。
2. 运行后处理：

   ```powershell
   python -m esf_sim.em_growth_postprocess `
     output\em_growth\exploratory_alpha04_101x201_matched_dipole_chapman_three_track
   ```

3. 运行严格验收：

   ```powershell
   python -m esf_sim.em_growth_acceptance `
     output\em_growth\exploratory_alpha04_101x201_matched_dipole_chapman_three_track
   ```

   验收器已加固：严格核对配置哈希、schema、终时、checkpoint/final/diagnostics；
   检查周期端点、逐记录 FAC 平衡、matched 远端逐时被动性、Alfvén能量预算、
   最大周期连续泡宽度。开放 y 边界下域内 column-content drift 仅作诊断，不能误当
   数值守恒误差。
4. 用 `view_image` 检查最终 PNG，不能仅凭振幅或阈值宣称 Y 分叉。
5. 运行：

   ```powershell
   python -m esf_sim validate-em
   python -m pytest -q `
     tests/test_alfven_1d.py tests/test_modal_alfven.py `
     tests/test_reduced_em_es_limit.py tests/test_laundal_endpoint.py `
     tests/test_frozen_li_em.py tests/test_fac_closure.py `
     tests/test_flux_tube_column.py tests/test_bidirectional_em.py `
     tests/test_em_validation.py tests/test_em_growth.py `
     tests/test_em_growth_postprocess.py tests/test_em_growth_acceptance.py `
     tests/test_controlled_em.py tests/test_pardiso_phase_cache.py `
     tests/test_em_exploratory_config.py
   ```

   预期为 132 项左右全部通过；最后一次严格 acceptance 聚焦测试为 12 passed。

## 7. 若 5000 s 仍无 Y 分叉，下一步不要盲调参数

按以下顺序做小规模 A/B，且始终保留 `column ES` 对照：

1. **分辨率/模态收敛**：101x201 + 44 个低波数 FAC 模态可能抹平次级结构。
   先做 201x401 与更高 `max_x/y_wavenumber` 的短窗/中窗对照；不要仅增加密度网格而
   不增加 FAC 模态。
2. **真实 E-region 终端电导**：`matched` 是被动、无反射基线，但不等同于真实
   夜间 E 层。用 `conductance` 边界扫描有文献依据的 Pedersen conductance，检查
   `R=(Sigma_A-Sigma_P)/(Sigma_A+Sigma_P)`；禁止回到 fixed-voltage `R=-1`，也禁止
   翻转 FAC 符号。
3. **缺失的 postsunset 驱动**：当前 Li 初始条件无背景 prereversal upward ExB drift、
   neutral wind/shear。文献表明背景上升漂移是 ESF 的关键驱动；若前两项收敛后仍只有
   平滑耗空，应新增默认 0、哈希保护的 `background_vertical_drift_m_s`，同时对 local/
   column/EM 三轨施加相同速度，再做 0/10/20 m/s A/B。不要只给 EM 轨加漂移。
4. **几何参数**：当前 field-line length=20,000 km、constant VA=1000 km/s 是探索值；
   低纬 F-to-E 实际场线长度短得多。matched 终端的低频导纳与长度无关，但传播瞬态、
   line energy 和反射时刻有关，需单独做长度/VA 收敛。
5. **未包含物理**：common-mode parallel mass transport、完整双半球 metric、化学、
   背景漂移和动态 E-region conductance 仍缺失，不能把当前探索性结果写成定量预测。

相关一级文献：

- Bhattacharyya & Burke (2000), transmission-line coupling of F/E regions.
- Bhattacharyya (2004), E-region conductivity and bubble dynamics:
  https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2003GL018960
- Dao et al. (2013), finite Alfvén mapping and E-region absorption:
  https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/jgra.50216
- Stephan et al. (2002), E-region conductance suppressing ESF:
  https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2001JA000162
- Ghadjari et al. (2022), standing Alfvén waves in EPBs:
  https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021GL097526

## 8. 工作树注意事项

- 工作树是 dirty 的；现有 modified/untracked 文件都是本项目成果或用户历史结果，
  不要 reset、checkout 或删除。
- `tmp_three/` 是早期 9x9 smoke 目录，可忽略；不要为清理而冒险递归删除。
- Git 尚未提交当前 reduced-EM 大改；用户之前只要求初始化仓库，没有要求本轮 commit。
- 原 Li 静电代码与正式 Figure 3/6 结果必须继续作为 regression oracle。

## 9. 可直接粘贴给新 Codex 的提示词

```text
继续 EPB_simulation 项目的 reduced-EM nonlinear EPB 任务。请先完整阅读
docs/HANDOFF_REDUCED_EM_EPB_2026-09-30.md，并检查当前工作树和输出，不要重做已经完成的
推导、实现或3150 s算例。

第一优先级：从现有 schema-v3 checkpoint（t=4350 s）安全恢复三轨算例到5000 s。
使用 configs/em_growth_li_101x201_alpha04.json 和
tmp/em_growth_three_track_t5000.json；绝对不要使用 --no-resume。当前 status.json 的
running/t=3150 是陈旧启动状态，可信恢复点是 checkpoint t=4350；diagnostics/snapshots
到4374/4370的尾部属于checkpoint之后的孤儿，runner应自动reconcile。配置哈希必须保持
8547522688d8127afb575a35b8affb857cc97886b0b3aeb301073b4f4d95a02d。

不要高频轮询，每10–20分钟检查一次最新diagnostics/checkpoint即可。5000 s完成后，必须：
1)核对checkpoint/final/末快照和哈希；2)运行em_growth_postprocess；
3)运行严格em_growth_acceptance；4)用view_image做最终视觉QA；5)运行validate-em和交接文档
列出的132项reduced-EM回归测试。

科学判断必须谨慎：3150 s已经恢复primary EPB growth，但仍只有宽而平滑的耗空通道，
尚未出现Li式Y分叉或次级结构。5000 s若仍无分叉，不要宣称完全解决，也不要翻转FAC符号、
回到fixed-voltage短路边界或用弱feedback作弊。按交接文档顺序做分辨率/模态、真实E层终端
电导、三轨共同背景上升ExB漂移、场线长度/VA的受控A/B。任何新物理参数必须默认0或保持旧
行为、进入配置哈希、具备测试，并保留local ES / column ES / column EM三轨归因。

最终给出：失败根因、修复方程与代码、数值/能量门禁、5000 s形态与增长率、仍未解决的
物理限制，以及可点击的图/JSON/源数据路径。不要修改或删除用户已有输出，不要git reset。
```

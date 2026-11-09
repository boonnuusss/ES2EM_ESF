# Reduced-EM nonlinear EPB：5000 s 续算与严格验收

## 结论与适用范围

从 schema-v3 checkpoint t=4350 s、step=17400 安全续算至 **5000 s、22279 步**。未重做 3150 s 算例，未使用 `--no-resume`。生产配置哈希始终为：

`8547522688d8127afb575a35b8affb857cc97886b0b3aeb301073b4f4d95a02d`

**旧模型压灭 primary EPB growth 的问题已得到纠正，但 column reduced-EM 到 5000 s 仍未出现清晰的 Li 式 Y 分叉。不能称为完全解决。** 最终 PNG 经 `view_image` 直接检查：local ES 有细长通道和明显两侧展开结构；column ES 有清晰分支及次级结构；column EM 仍是宽而平滑的主要耗空通道。阈值连通域数量不等同于 Y 分叉计数。

生产物理参数未改动：matched R=0，feedback_gain=1，双半球表面 FAC，粒子/碰撞电导两个独立列矩；保留 local ES、column ES、column EM 三轨。

## 失败根因与已有方程修复

以下是交接中已建立、此次沿用的结论，不是此次重新推导或重跑旧案例的结果：

1. 固定远端电势等价于 R=-1 短路反射，不是开放边界。旧 FAC 在反射返回后几乎抵消重力驱动；生产改用 matched R=0 被动终端。
2. 将第一场线格上的电流梯度作为体源，错误引入 1/ds 缩放和随网格漂移的 Schur 代数极点；现用物理表面 FAC。
3. 单一有效长度不能同时代表粒子列和碰撞加权电导列。使用

   H_n = ∫(B_a/B)(n(s)/n_a) ds

   H_nu = ∫(B_a/B)(n(s)nu(s)/(n_a nu_a)) ds

   N = H_n n

   div[H_nu n nu_a grad(psi)] = -g d_x(H_n n) + (B_a/m_i) J_surface.

   当前 H_n 仅依赖高度，因此右侧重力项等于 -H_n g d_x n。推进守恒列变量 N，两半球表面电流直接闭合，并配对 modal/grid 端口功率。

对应代码：[列几何](../src/esf_sim/flux_tube_column.py)、[FAC 闭合](../src/esf_sim/fac_closure.py)、[双向反馈与列输运](../src/esf_sim/bidirectional_em.py)、[Alfvén 模态与终端](../src/esf_sim/modal_alfven.py)、[三轨 runner](../src/esf_sim/em_growth.py)。

## 本次发现并修复的归档问题

4430.25 s 的候选步 density CFL=0.250015，超过 0.25，runner 拒步并将 dt 从 0.25 减至 0.125 s；之后没有继续减半。最终总步数为 22279，不能被诊断间隔 4 整除。旧 runner 因而把 diagnostics 停在 4999.625 s，尽管 checkpoint/final/snapshot 已正确到达 5000 s。严格终态核对捕获了这个真实缺陷。

修复内容：无论诊断间隔，今后的最后接受步必须写诊断；对已完成但缺终点诊断的 checkpoint，仅重算该状态的 RHS 并追加 endpoint 样本，不推进时间。追加记录标记 `reconstructed_terminal_state_rhs_not_stage_maximum`。此记录不是从历史 RK-stage 恢复的峰值 CFL。原诊断前缀字节完全保留，checkpoint 所有数组完全相同，final NPZ 文件字节完全相同。新增两个测试验证非整除终点、无时间推进恢复及重复恢复幂等性。

## 数值与能量门禁

- checkpoint schema 3；final/snapshot schema 2；501 个快照覆盖 0–5000 s、间隔 10 s；5570 条诊断严格递增并到达终点。
- checkpoint/final 的三轨密度和 EM 特征变量逐元素一致；最后快照三轨密度与 checkpoint 逐元素一致；所有快照哈希、时间、正密度与有限性检查通过。
- 最大 local ES / column ES CFL：0.2498138223 / 0.2033135328；最大 combined EM CFL：0.4。最终 dt=0.125 s，累计一次减步。
- 最大耦合残差 4.2039e-12；最大 modal/grid 功率配对相对缺陷 1.2241e-15。
- FAC/重力有效期中位数 0.25116，最大值（终点）0.45993；逐记录净表面电流归一化失衡最大 8.638e-17。
- 全部远端功率非负；反射系数始终为 0。
- 终态线能量 993170.975 J；积分 Li 端口输入 46568659.827 J；远端吸收 45575691.919 J；净边界功 992967.937 J。
- 能量增量比净边界功大 203.038 J；相对总交换误差 4.455e-6，相对净功约 2.045e-4。通过现有 5e-3 数值生能门禁，但不能称为机器精度离散能量守恒。
- 能量工作积分来自采样端口功率的梯形积分，末点为明确标记的终态 RHS 样本；未归档逐 SSPRK stage 的精确功积分。开放垂直边界下的总粒子列漂移仍只是诊断，尚无边界通量逐步闭合预算。
- 严格 acceptance：36 个检查，零失败，4 个警告：local ES 周期端点差异 1.859e-9 接近首级警告阈值、local CFL 接近上限、EM combined CFL 接近上限、开放边界粒子预算未闭合。
- `validate-em` 九阶段通过；交接指定的 reduced-EM 回归加本次测试：**143 passed**。

## 5000 s 形态与增长率

阈值 D=1-n/<n>_x >= 0.25：column EM 泡顶 **488 km**、垂直跨度 **238 km**、最大周期连续宽度 **147.5 km**；column ES 泡顶 634 km。local ES 阈值通道达到 650 km 域顶，因此不能把其泡顶当作无边界影响的物理高度。

原后处理的 300–5000 s OLS（每个 checkpoint 历史步样本等权）：

| track | gamma (s^-1) | R² | A(5000)/A(0) |
|---|---:|---:|---:|
| local ES | 3.6321175e-4 | 0.79581 | 3.64182 |
| column ES | 4.9159683e-4 | 0.94818 | 10.1537 |
| column EM | 4.2019054e-4 | 0.99952 | 7.39695 |

原拟合 EM 比同几何 column ES 低约 14.5%。由于一次减步令晚期样本加密，另以等间隔 1 s 插值后拟合：local ES 3.9912731e-4、column ES 5.1643918e-4、column EM 4.2139190e-4 s^-1；EM 相对 column ES 低约 **18.4%**。两个估计均保留，采样方法不同不能混用。

在 3150–5000 s 的等时间拟合中，local ES / column ES / column EM 分别为 -1.33135e-4 / 1.67338e-4 / 4.03618e-4 s^-1。末 100 s 分别约 -5.181e-4 / 1.5944e-4 / 2.9488e-4 s^-1。ES 分支形态和横向 RMS 已进入不同非线性阶段，不能用晚期 RMS 斜率反转宣称 EM 提高了线性不稳定增长率。纯 EM 归因始终使用 column EM − column ES。

## 仍未解决的问题与下一步顺序

5000 s EM 通道仍光滑；低波数 FAC 截断、网格、终端负载和缺失驱动的作用尚未分离。本次首先开展独立的 201x401 + FAC (8,8) / 152 模态三轨对照，原基线为 101x201 + FAC (4,4) / 44 模态。先做 60 s 稳定性门禁，再延长至 300 s；这不足以证明 5000 s 的非线性分叉形态收敛。后续状态与量化比较见同目录的分辨率对照报告（生成后链接补充）。

在分辨率证据之后，依次做真实 E 层电导扫描、默认零且三轨共同施加的背景上升漂移 0/10/20 m/s、场线长度/VA 敏感性。本次没有增加漂移或其他物理参数，没有修改 FAC 符号、回到 fixed-voltage 或减弱反馈来制造增长。

Bhattacharyya (2004) 的示例包含约 1 S 电导、1600 km 场线长度和 300 km/s VA，是后续参数范围的文献锚点，不是本算例的环境实测值。[原文](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2003GL018960)。Dao et al. (2013) 展示 E 区部分吸收/反射与快 Alfvén 映射，说明需区分快速电动力学与慢密度演化；它并不定量验证当前简化线路。[原文](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/jgra.50216)。

缺失物理仍包括 common-mode 平行质量输运、完整双半球 metric、化学、中性风/剪切、背景 PRE 漂移与动态 E-region 电导。当前是可归因的探索性 reduced-EM 模型，不是对真实 EPB 的定量预测。

## 可复查交付物

全部生产结果根目录：`../output/em_growth/exploratory_alpha04_101x201_matched_dipole_chapman_three_track/`。

- [最终 PNG](../output/em_growth/exploratory_alpha04_101x201_matched_dipole_chapman_three_track/postprocess_t5000/em_growth_comparison.png)
- [矢量 PDF](../output/em_growth/exploratory_alpha04_101x201_matched_dipole_chapman_three_track/postprocess_t5000/em_growth_comparison.pdf)
- [后处理 metrics JSON](../output/em_growth/exploratory_alpha04_101x201_matched_dipole_chapman_three_track/postprocess_t5000/em_growth_metrics.json)
- [源数据 NPZ](../output/em_growth/exploratory_alpha04_101x201_matched_dipole_chapman_three_track/postprocess_t5000/em_growth_source_data.npz)
- [严格 acceptance JSON](../output/em_growth/exploratory_alpha04_101x201_matched_dipole_chapman_three_track/em_growth_acceptance_t5000.json)
- [终态与快照审计](../output/em_growth/exploratory_alpha04_101x201_matched_dipole_chapman_three_track/terminal_audit_t5000.json)
- [诊断修复不变性证明](../output/em_growth/exploratory_alpha04_101x201_matched_dipole_chapman_three_track/terminal_diagnostic_recovery_t5000.json)
- [等时间增长率](../output/em_growth/exploratory_alpha04_101x201_matched_dipole_chapman_three_track/growth_uniform_time_t5000.json)
- [validate-em](../output/em_growth/exploratory_alpha04_101x201_matched_dipole_chapman_three_track/validate_em_t5000.json)
- [143 项回归日志](../output/em_growth/exploratory_alpha04_101x201_matched_dipole_chapman_three_track/reduced_em_regression_t5000.log)
- [checkpoint](../output/em_growth/exploratory_alpha04_101x201_matched_dipole_chapman_three_track/checkpoint.npz)、[final states](../output/em_growth/exploratory_alpha04_101x201_matched_dipole_chapman_three_track/final_states.npz)、[原始 diagnostics](../output/em_growth/exploratory_alpha04_101x201_matched_dipole_chapman_three_track/diagnostics.jsonl)。

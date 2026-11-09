# EPB_simulation 项目进度与跨 LLM 交接

更新时间：2026-10-05（Asia/Shanghai）。本文件聚焦 reduced-EM nonlinear EPB；E 盘目录中另有 FARR 工作，本轮未审计，不应与 reduced-EM 工作树混用。

## 1. 接手者首先要知道的状态

**原 101×201 三轨算例已完成 5000 s 并验收；201×401 同步加密算例尚未完成，目前没有存活的 Python 计算进程，可信恢复点为 1050 s。** 本轮只读核查和文档交接，没有启动新计算。

| 对象 | 2026-10-05 核查结果 |
|---|---|
| 实际 reduced-EM 工作树 | `<ORIGINAL_CHECKOUT>` |
| 当前聊天默认目录 | `<ORIGINAL_CHECKOUT>`，不是上述代码版本 |
| 加密 checkpoint | schema 3；t=1050 s；step=4200；dt=0.25 s；零减步；浮点数组均有限、三轨密度为正 |
| 最新 diagnostics | t=1074 s、step=4296；属于 checkpoint 后尾部 |
| 最新快照 | t=1070 s，manifest 共 108 帧；1050 s 后帧属于未提交尾部 |
| status.json | running/t=300 s，是启动时陈旧状态，不是真实进度 |
| 计算进程 | 通过 Get-CimInstance 确认没有 python.exe；旧 PID 24236 已失效 |
| 加密 final/report | 尚为上次 300 s 完成结果，不能当作 1050/1074/3150 s 终态 |
| 中断原因 | 未确定；stderr 为空。额度限制曾中断代理操作，但不能据此断言 Python 进程停止的原因 |
| 定时跟进 | automation `epb` 已暂停，避免另一个 LLM 接手后重复运行或继续消耗额度 |
| Git | dirty；大量 reduced-EM 源码与测试仍是 untracked，未提交。禁止 reset/checkout 清理 |

恢复时以 checkpoint 为真值，让 runner 自动 reconcile 超出 1050 s 的诊断/快照尾部。不要手工删除，不要伪造哈希，不要使用 `--no-resume`。进程状态只是本次核查结果，新代理启动前仍需重新检查。

完整机器可读核查记录：[EPB_HANDOFF_STATE_2026-10-05.json](<ORIGINAL_CHECKOUT>/docs/EPB_HANDOFF_STATE_2026-10-05.json)。

## 2. 研究目标与三轨归因

目标是验证“Li 2-D 垂直截面 + 1-D 场向 Alfvén 传输”的 reduced-EM 非线性模型，解释旧 EM 模型为何压灭 EPB，并检验恢复增长后是否产生可信的分叉/次级结构。

三轨始终为 local ES、相同磁通管列几何且 FAC=0 的 column ES、相同列几何且双向 FAC 反馈的 column EM。纯电磁效应必须用 column EM − column ES；column ES − local ES 是几何效应。

**当前结论：primary EPB growth 已恢复，但 coarse column EM 到 5000 s 仍无清晰 Li 式 Y 分叉。加密成熟期研究未完成，不能宣称完全解决或已经收敛。**

## 3. 已建立的失败根因和修复

这部分已完成，不要重新推导/实现，也不必重跑旧失败案例：

1. 远端固定电势是 R=-1 的短路反射，不是开放边界。旧 20000 km / 1000 km/s 场线往返 40 s，反射后 FAC 几乎抵消重力驱动。生产使用 matched R=0。
2. 旧 FAC 体源错误包含 1/ds，导致随网格漂移的 Schur 代数极点；现在使用两半球物理表面 FAC，不能改回第一格电流梯度体源。
3. 单一 H 不能同时代表粒子列和碰撞加权电导列。现在：

```text
H_n  = ∫ (B_a/B) [n(s)/n_a] ds
H_nu = ∫ (B_a/B) [n(s)nu(s)/(n_a nu_a)] ds
N = H_n n
div[H_nu n nu_a grad(psi)] = -g d_x(H_n n) + (B_a/m_i) J_surface
```

H_n 仅随高度变化，重力项也可写成 -H_n g d_x n。推进列变量 N，配对 modal/grid 端口功率。禁止翻转 FAC 符号、恢复 fixed-voltage 短路或降低 feedback_gain 来“制造成功”。

## 4. 已完成里程碑与结果

### 原网格 3150 s：已冻结，不重算

101×201，FAC (4,4)，44 模态；12600 步，dt=0.25 s，零减步。300–3150 s 增长率 local/column ES/EM 分别约 6.4391e-4 / 6.6951e-4 / 4.1591e-4 s^-1。EM 比 column ES 低约 37.9%，单位反馈下仍有增长。图件独立保存在 `postprocess_t3150/`。

### 原网格 5000 s：已完成全部强制收尾

从 4350 s checkpoint 续算至 5000 s，22279 步。4430.25 s 候选步 density CFL=0.250015 超过 0.25，拒步后 dt 减到 0.125 s，之后无连续减步。

| 指标 | 结果 |
|---|---:|
| checkpoint/final/snapshot schema | 3 / 2 / 2 |
| 快照 | 501 帧，0–5000 s，每 10 s |
| diagnostics | 5570 条，到终点 |
| 最大 local / column ES CFL | 0.2498138 / 0.2033135 |
| 最大 EM combined CFL | 0.4 |
| 最大耦合残差 | 4.2039e-12 |
| 最大功率配对缺陷 | 1.2241e-15 |
| FAC/重力中位数 / 终点 | 0.25116 / 0.45993 |
| 终态线能量 | 993170.975 J |
| 积分 Li 输入 / 远端吸收 | 46568659.827 / 45575691.919 J |
| 能量比净功多出的量 | 203.038 J，相对总交换 4.455e-6 |
| 严格 acceptance | 36 检查、0 失败、4 警告 |
| validate-em / reduced-EM 回归 | 九阶段通过 / 143 passed |

checkpoint/final 三轨密度与 EM 特征变量逐元素一致，末快照密度逐元素一致，全部快照元数据、有限性与正密度已核验。已用 view_image 检查最终图：column ES 有明显分支/次级结构，column EM 仍宽而平滑。D=1-n/<n>_x≥0.25 时 EM 泡顶 488 km、跨度 238 km、最大连续宽度 147.5 km。local ES 达到域顶 650 km，存在边界限制。

| 300–5000 s 拟合 | local ES | column ES | column EM |
|---|---:|---:|---:|
| 每个时间步等权 gamma (s^-1) | 3.63212e-4 | 4.91597e-4 | 4.20191e-4 |
| R² | 0.79581 | 0.94818 | 0.99952 |
| 等间隔 1 s 插值拟合 gamma | 3.99127e-4 | 5.16439e-4 | 4.21392e-4 |

减步令晚期历史样本更密，因此两种拟合分开报告。EM 相对 column ES 抑制分别约 14.5% / 18.4%。晚期 ES RMS 饱和/下降不能解释为 EM 提高线性增长率。

### 本次修复：终点诊断遗漏

旧 runner 只在 step 能被间隔 4 整除时输出诊断；22279 不整除，原文件止于 4999.625 s。现已强制写终点，并支持已完成 checkpoint 的终态 RHS 诊断重建（不推进时间）。补写记录标记 `reconstructed_terminal_state_rhs_not_stage_maximum`，不可冒充历史 RK-stage 峰值。修复证明：checkpoint 全数组不变、final NPZ 字节不变、既有诊断前缀不变。新增非整除终点和无推进幂等恢复测试，已包含在 143 项中。

### 加密 201×401 + FAC (8,8)：60/300 s 已验收，3150 s 未完成

152 模态；物理条件保持一致，feedback=1，matched，三轨。60/300 s 严格门禁通过，300 s 终态数组一致且完成图像 QA；零减步、最大残差 4.511e-12、配对缺陷 6.170e-16。300 s 完整状态保存在 `archive_t300/`。

相同 300 s、细网格取共同节点并去除重复周期端点：EM 总密度相对 L2 差 0.0138633%，横向扰动 L2 差 0.157415%，EM−column 效应差 3.1054%。100–300 s 的 EM 增长率粗/细分别 3.208565e-4 / 3.195274e-4 s^-1，差约 0.414%。这只是早期稳定性与敏感性证据，不是晚期形态收敛。

## 5. 配置、路径、哈希与安全恢复

所有下列相对路径均相对于 **C 盘实际工作树**。

| 用途 | Li 配置 | EM 配置 |
|---|---|---|
| 原网格已完成 5000 s | configs/em_growth_li_101x201_alpha04.json | tmp/em_growth_three_track_t5000.json |
| 加密目标 3150 s | configs/em_growth_li_201x401_alpha04_resolution.json | tmp/em_growth_resolution_201x401_k8_t3150.json |

原网格 hash：`8547522688d8127afb575a35b8affb857cc97886b0b3aeb301073b4f4d95a02d`

加密 hash：`f7aecc67b3ddaae0c9c6325d0ef6fee2b6084060db301c8c4df5d6b9b52ad055`

```powershell
# 接手时先重新检查，输出为空才可启动；不要信旧 PID。
Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
  Select-Object ProcessId, CommandLine

Set-Location '<ORIGINAL_CHECKOUT>'
$env:PYTHONPATH='<ORIGINAL_CHECKOUT>\.deps;src'
$env:PYPARDISO_MKL_RT='<ORIGINAL_CHECKOUT>\.deps\Library\bin\mkl_rt.3.dll'
$env:PATH='<ORIGINAL_CHECKOUT>\.deps\Library\bin;' + $env:PATH
$env:MKL_NUM_THREADS='10'

# 二选一，绝对不要同时运行：
# A. 既有 driver：包含续算、终态核对、后处理和严格验收
python '<ORIGINAL_CHECKOUT>/tmp/run_em_resolution_mature_20261002.py'

# B. 仅 runner，默认 resume；完成后自行运行收尾命令
# python -m esf_sim run-em-growth --config configs/em_growth_li_201x401_alpha04_resolution.json --em-config tmp/em_growth_resolution_201x401_k8_t3150.json
```

driver 会从当前 checkpoint 恢复，允许 300≤t≤3150；不是从 300 s 强制重跑。不要重新运行 `prepare_em_mature_20261002.py`，其准备/归档步骤已经完成。不要重新运行旧终点诊断修复脚本。输出路径不能切到空目录，否则可能启动新算例。

每 30 分钟检查一次 diagnostics/checkpoint；避免用 LLM 高频轮询消耗额度。单次运行 CPU 时间很长，不要因 stderr 为空或 status 陈旧而重复启动。要看 checkpoint 时间是否推进。任务切换、休眠或进程停止后均先核对，再从稳定 checkpoint 恢复。

## 6. 完成加密 3150 s 后的验收与后续顺序

1. 核对 status completed、实际终时、配置哈希、schema、checkpoint/final/末快照逐元素一致、诊断终时和去重。
2. 执行后处理及严格 acceptance；driver 已自动安排，先检查是否成功，不要盲目重做。
3. 用图像查看工具做最终视觉 QA，与粗网格同一 3150 s 的快照/冻结图件比较。不能用不同终时对比替代分辨率研究。
4. 若需比较成熟的 5000 s 形态，先归档细网格 3150 s 成果，再仅延长 t_end_s 到 5000 s，验证同一 hash，不用 --no-resume。
5. 分辨率/模态证据之后，再做有文献依据的真实 E-region 终端电导扫描。不能把 matched 当成真实夜间电导；R=(Sigma_A-Sigma_P)/(Sigma_A+Sigma_P)。
6. 若仍需要背景驱动，再新增默认 0 的 background_vertical_drift_m_s，三轨共同施加，保持旧行为/哈希兼容、非零值进入哈希、有测试，再做 0/10/20 m/s 对照。
7. 最后分别研究场线长度和 VA；VA 同时改变传播速度和导纳，不能全部归因为时延。

**电导扫描、背景漂移实现/扫描、长度/VA 扫描尚未开展。** 两个分辨率或早期小差异都不能证明非线性分叉收敛。不要跳过阶段去盲调参数。

```powershell
$r='output/em_growth/resolution_201x401_k8_matched_dipole_chapman_three_track_20260930'
python -m esf_sim.em_growth_postprocess $r --output "$r/postprocess_t3150"
python -m esf_sim.em_growth_acceptance $r --output "$r/acceptance_t3150.json"
python -m esf_sim validate-em
python -m pytest -q tests/test_alfven_1d.py tests/test_modal_alfven.py tests/test_reduced_em_es_limit.py tests/test_laundal_endpoint.py tests/test_frozen_li_em.py tests/test_fac_closure.py tests/test_flux_tube_column.py tests/test_bidirectional_em.py tests/test_em_validation.py tests/test_em_growth.py tests/test_em_growth_postprocess.py tests/test_em_growth_acceptance.py tests/test_controlled_em.py tests/test_pardiso_phase_cache.py tests/test_em_exploratory_config.py
```

## 7. 仍需谨慎的科学限制

- 能量门禁基于采样端口功率的梯形积分，非逐 RK-stage 精确功积分；终态补写诊断的采样来源已注明。
- 开放垂直边界的列含量漂移只是诊断，尚未归档完整边界粒子通量账，不能当数值守恒误差。
- 缺少 common-mode 平行质量输运、完整双半球 metric、化学、中性风/剪切、PRE 漂移、动态 E-region 电导。
- 常数 VA=1000 km/s、场线长度 20000 km 是探索设置，不是环境实测；当前不能给真实 EPB 定量预测。
- 电导文献已查阅：Bhattacharyya 2004，DOI 10.1029/2003GL018960；Dao 2013，DOI 10.1002/jgra.50216。前者的约 1 S、1600 km、300 km/s 只是示例锚点，不能直接称为本场景实测。

## 8. 关键代码与证据索引

- [flux_tube_column.py](<ORIGINAL_CHECKOUT>/src/esf_sim/flux_tube_column.py)
- [fac_closure.py](<ORIGINAL_CHECKOUT>/src/esf_sim/fac_closure.py)
- [bidirectional_em.py](<ORIGINAL_CHECKOUT>/src/esf_sim/bidirectional_em.py)
- [modal_alfven.py](<ORIGINAL_CHECKOUT>/src/esf_sim/modal_alfven.py)
- [em_growth.py](<ORIGINAL_CHECKOUT>/src/esf_sim/em_growth.py)
- [em_growth_acceptance.py](<ORIGINAL_CHECKOUT>/src/esf_sim/em_growth_acceptance.py)
- [em_growth_postprocess.py](<ORIGINAL_CHECKOUT>/src/esf_sim/em_growth_postprocess.py)
- [em_validation.py](<ORIGINAL_CHECKOUT>/src/esf_sim/em_validation.py)

已验证成果：

- [5000 s 完整科学报告](<ORIGINAL_CHECKOUT>/docs/REDUCED_EM_EPB_T5000_2026-09-30.md)
- [5000 s 图件](<ORIGINAL_CHECKOUT>/output/em_growth/exploratory_alpha04_101x201_matched_dipole_chapman_three_track/postprocess_t5000/em_growth_comparison.png)
- [5000 s 后处理指标](<ORIGINAL_CHECKOUT>/output/em_growth/exploratory_alpha04_101x201_matched_dipole_chapman_three_track/postprocess_t5000/em_growth_metrics.json)
- [5000 s 源数据 NPZ](<ORIGINAL_CHECKOUT>/output/em_growth/exploratory_alpha04_101x201_matched_dipole_chapman_three_track/postprocess_t5000/em_growth_source_data.npz)
- [5000 s 严格验收](<ORIGINAL_CHECKOUT>/output/em_growth/exploratory_alpha04_101x201_matched_dipole_chapman_three_track/em_growth_acceptance_t5000.json)
- [5000 s 终态审计](<ORIGINAL_CHECKOUT>/output/em_growth/exploratory_alpha04_101x201_matched_dipole_chapman_three_track/terminal_audit_t5000.json)
- [终点诊断修复不变性](<ORIGINAL_CHECKOUT>/output/em_growth/exploratory_alpha04_101x201_matched_dipole_chapman_three_track/terminal_diagnostic_recovery_t5000.json)
- [等间隔时间增长率](<ORIGINAL_CHECKOUT>/output/em_growth/exploratory_alpha04_101x201_matched_dipole_chapman_three_track/growth_uniform_time_t5000.json)
- [回归日志](<ORIGINAL_CHECKOUT>/output/em_growth/exploratory_alpha04_101x201_matched_dipole_chapman_three_track/reduced_em_regression_t5000.log)
- [加密 300 s 验收](<ORIGINAL_CHECKOUT>/output/em_growth/resolution_201x401_k8_matched_dipole_chapman_three_track_20260930/acceptance_t300.json)
- [粗细网格 300 s 对照 JSON](<ORIGINAL_CHECKOUT>/output/em_growth/resolution_201x401_k8_matched_dipole_chapman_three_track_20260930/resolution_comparison_t300.json)
- [加密 300 s 图件](<ORIGINAL_CHECKOUT>/output/em_growth/resolution_201x401_k8_matched_dipole_chapman_three_track_20260930/postprocess_t300/em_growth_comparison.png)
- [加密 checkpoint（接手时重新核查）](<ORIGINAL_CHECKOUT>/output/em_growth/resolution_201x401_k8_matched_dipole_chapman_three_track_20260930/checkpoint.npz)
- [原始历史交接](<ORIGINAL_CHECKOUT>/docs/HANDOFF_REDUCED_EM_EPB_2026-09-30.md)

## 9. 文件交接与额度限制

本次新文档位于 E 盘默认项目 docs，计算代码/状态在 C 盘 worktree，依赖库和少量辅助 driver 在 E 盘。只把 E 盘 Git 仓库交给下一模型会漏掉大量实际代码；只把 C 盘 Git 已提交内容交出去同样会漏掉 untracked 成果。新 LLM 若在同一电脑，应读取上述实际路径；若在另一环境，需连同 C 盘当前未提交源码、configs、tests、tmp 配置、加密 checkpoint/diagnostics/timestep events/快照前缀及 E 盘辅助 driver 一并转移，再安装依赖。不要复制密钥或不相关个人配置。

Codex 自动检查 `epb` 已暂停。本轮未删除输出、未 git reset、未重新运行算例或回归。143 passed 是上次代码修复后真实运行结果，本次仅引用已归档日志。避免要求下一 LLM 自动重复所有已完成工作；首先安全恢复加密 checkpoint，推进当前未完成阶段。

## 10. 新代理接手续算记录（2026-10-05 20:18，Asia/Shanghai）

前三份指定交接文件已完整阅读。只读核对 C 盘 dirty 工作树、配置和 runner 的哈希/reconcile/resume 路径；未改源码、物理参数或配置，未重跑任何已完成算例/回归。重新按命令行检查 python.exe/pythonw.exe 后确认无活跃 Python，启动前再次复核为空。

- 20:16:30 启动唯一既有 driver `<ORIGINAL_CHECKOUT>/tmp/run_em_resolution_mature_20261002.py`，后台 PID **28312**。20:18 启动确认按命令行核实为该 driver，只有一个 Python 进程。PID 仅作本次记录，后续检查须重新核对命令行。
- 实际代码工作树仍是 `<ORIGINAL_CHECKOUT>`，依赖和环境按用户所给设置，MKL_NUM_THREADS=10。
- 当前源码计算的配置 hash 与 schema-3 checkpoint 完全一致：`f7aecc67b3ddaae0c9c6325d0ef6fee2b6084060db301c8c4df5d6b9b52ad055`。恢复点 1050 s、4200 步、dt=0.25 s、零减步；浮点数组有限，三轨密度为正。
- stdout 明确 `Resuming refined run at 1050.0 s`，启动确认时 stderr 为空。runner 自动 reconcile 后 snapshot manifest 从108帧/1070 s回到106帧/1050 s，诊断已重新推进到1055 s/4220步。没有手动清理尾部，没有运行prepare，没有使用--no-resume。
- 20:18 最新 checkpoint 仍为1050 s，尚未到下一个25 s保存点；不能将1055 s诊断当可信恢复点。最新诊断耦合残差4.5263e-12，功率配对缺陷2.2137e-16，local/column ES CFL分别0.0034568/0.0034413，EM combined CFL=0.4。
- 原 automation `epb` 继续保持暂停；当前接手聊天新建 heartbeat **epb-2**，每30分钟仅检查一次，稳定且无可行动变化时不通知。禁止两个聊天重复启动进程。

[启动审计 JSON](<ORIGINAL_CHECKOUT>/docs/EPB_RESUME_AUDIT_2026-10-05.json)、[启动记录](<ORIGINAL_CHECKOUT>/tmp/em_resolution_resume_launch_20261005.json)、[stdout](<ORIGINAL_CHECKOUT>/tmp/em_resolution_resume_20261005_201630.stdout.log)、[stderr](<ORIGINAL_CHECKOUT>/tmp/em_resolution_resume_20261005_201630.stderr.log)。机器可读交接 JSON 保留原始核查字段，在 `latest_resume` 中增加接手状态。

**仍未完成：**细网格3150 s终态验收、图像QA、粗细同3150 s增长率/形态/FAC/能量对照；必要时细网格5000 s延长；E-region电导扫描、背景漂移实现与扫描、长度/VA扫描。driver将自动做3150 s终态核对、后处理与严格验收，完成后先检查已有结果，避免重复运行。旧中断原因仍未知。旧失败根因与修复方程继续采用第3节结论；不将启动成功称为3150 s已通过，也不将两个网格或300 s早期小差异称严格收敛。

### 20:48 定时检查（Asia/Shanghai）

唯一Python进程PID28312，命令行确认是既有细网格driver，未重复启动。可信checkpoint已到1125 s/4500步，schema3、hash不变、dt=0.25 s、零减步，全部浮点数组有限、三轨密度为正（最小值分别1.528984835e7、1.465643836e7、1.528985019e7 m^-3）。诊断到1147 s，快照到1140 s；status仍为启动1050 s，不用它判断进度。续算以来最大耦合残差4.6150e-12、功率配对缺陷8.7823e-16，local/column ES CFL最大0.0040873/0.0040996，EM combined CFL=0.4；诊断时间严格递增且数值有限，无减步事件，stderr为空。运行稳定，3150 s及后续阶段仍未完成，本次不启动其他算例。

[20:48审计JSON](<ORIGINAL_CHECKOUT>/docs/EPB_MONITOR_AUDIT_2026-10-05_2048.json)。

### 21:18 定时检查（Asia/Shanghai）

单一PID28312命令行仍为既有driver。可信checkpoint推进到1225 s/4900步，hash/schema不变、dt=0.25 s、零减步，所有浮点数组有限、三轨正密度；诊断到1238 s、快照到1230 s。续算以来最大耦合残差4.6843e-12、功率配对缺陷8.7823e-16；local/column ES最大CFL=0.0048959/0.0049541，EM combined CFL=0.4；诊断严格递增、数值有限，无减步事件，stderr为空。稳定运行，无需操作；3150 s及后续阶段仍待完成。

[21:18审计JSON](<ORIGINAL_CHECKOUT>/docs/EPB_MONITOR_AUDIT_2026-10-05_2118.json)。

### 21:48 定时检查（Asia/Shanghai）

单一PID28312命令行确认为既有driver。可信checkpoint推进到1325 s/5300步，schema3、hash不变、dt=0.25 s、零减步，浮点数组全部有限、三轨正密度；诊断到1328 s、快照到1320 s。续算以来最大耦合残差4.8515e-12、功率配对缺陷8.7823e-16；local/column ES最大CFL=0.0059730/0.0061067，EM combined CFL=0.4。诊断时间严格递增、数值有限，无减步事件，stderr为空。运行稳定，无需操作，未重复启动；3150 s及后续阶段仍待完成。

[21:48审计JSON](<ORIGINAL_CHECKOUT>/docs/EPB_MONITOR_AUDIT_2026-10-05_2148.json)。

### 22:18 定时检查（Asia/Shanghai）

命令行核查仅有既有driver PID28312。可信checkpoint到1400 s/5600步，schema3、hash不变、dt=0.25 s、零减步，全部浮点数组有限、三轨正密度；诊断到1418 s，快照到1410 s。续算以来最大耦合残差4.9391e-12、功率配对缺陷8.7823e-16，local/column ES最大CFL=0.0075028/0.0077785，EM combined CFL=0.4。诊断时间严格递增、数值有限，无减步事件，stderr为空。运行稳定，无需操作，未重复启动；3150 s及后续阶段仍未完成。

[22:18审计JSON](<ORIGINAL_CHECKOUT>/docs/EPB_MONITOR_AUDIT_2026-10-05_2218.json)。

### 22:48 定时检查（Asia/Shanghai）

按命令行仅有既有driver PID28312。可信checkpoint到1500 s/6000步，schema3、hash不变、dt=0.25 s、零减步，浮点数组有限、三轨正密度；诊断到1508 s，快照到1500 s。续算以来最大耦合残差4.9880e-12、功率配对缺陷8.7823e-16，local/column ES最大CFL=0.0098584/0.0104379，EM combined CFL=0.4。诊断时间严格递增、数值有限，无减步事件，stderr为空。运行稳定，无需操作，未重复启动；3150 s及后续阶段仍未完成。

[22:48审计JSON](<ORIGINAL_CHECKOUT>/docs/EPB_MONITOR_AUDIT_2026-10-05_2248.json)。

### 23:18 定时检查（Asia/Shanghai）

命令行确认单一既有driver PID28312。可信checkpoint到1575 s/6300步，schema3、hash不变、dt=0.25 s、零减步，所有浮点数组有限、三轨正密度；诊断到1598 s，快照到1590 s。续算以来最大耦合残差5.0441e-12、功率配对缺陷8.7823e-16，local/column ES最大CFL=0.0139577/0.0153131，EM combined CFL=0.4。诊断时间严格递增、数值有限，无减步事件，stderr为空。运行稳定，无需操作、未重复启动；3150 s及后续阶段仍待完成。

[23:18审计JSON](<ORIGINAL_CHECKOUT>/docs/EPB_MONITOR_AUDIT_2026-10-05_2318.json)。

### 23:48 定时检查（Asia/Shanghai）

按命令行确认唯一既有driver PID28312。可信checkpoint推进至1675 s/6700步，schema3、hash不变、dt=0.25 s、零减步，所有浮点数组有限、三轨正密度；诊断到1687 s、快照到1680 s。续算以来最大耦合残差5.0642e-12、功率配对缺陷8.7823e-16；local/column ES最大CFL=0.0224297/0.0263446，EM combined CFL=0.4。诊断严格递增且数值有限，无减步事件，stderr为空。运行稳定，无需操作，未重复启动；3150 s及后续阶段仍未完成。

[23:48审计JSON](<ORIGINAL_CHECKOUT>/docs/EPB_MONITOR_AUDIT_2026-10-05_2348.json)。

### 2026-10-06 00:18 定时检查（Asia/Shanghai）

命令行确认唯一既有driver PID28312。可信checkpoint到1775 s/7100步，schema3、hash不变、dt=0.25 s、零减步，所有浮点数组有限、三轨正密度；诊断到1777 s、快照到1770 s。续算以来最大耦合残差5.0642e-12、功率配对缺陷8.7823e-16，local/column ES最大CFL=0.0436844/0.0554159，EM combined CFL=0.4。诊断严格递增、数值有限，无减步事件，stderr为空。运行稳定，无需操作，未重复启动；3150 s及后续阶段仍未完成。

[00:18审计JSON](<ORIGINAL_CHECKOUT>/docs/EPB_MONITOR_AUDIT_2026-10-06_0018.json)。

### 2026-10-06 00:48 定时检查（Asia/Shanghai）

按命令行确认唯一既有driver PID28312。可信checkpoint到1850 s/7400步，schema3、hash不变、dt=0.25 s、零减步，所有浮点数组有限、三轨正密度；诊断到1867 s、快照到1860 s。续算以来最大耦合残差5.0642e-12、功率配对缺陷8.7823e-16，local/column ES最大CFL=0.0703596/0.0801382，EM combined CFL=0.4。诊断严格递增、数值有限，无减步事件，stderr为空。运行稳定，无需操作，未重复启动；3150 s及后续阶段仍未完成。

[00:48审计JSON](<ORIGINAL_CHECKOUT>/docs/EPB_MONITOR_AUDIT_2026-10-06_0048.json)。

### 2026-10-06 01:18 定时检查（Asia/Shanghai）

命令行确认唯一既有driver PID28312。可信checkpoint到1950 s/7800步，schema3、hash不变、dt=0.25 s、零减步，所有浮点数组有限、三轨正密度；诊断到1956 s、快照到1950 s。续算以来最大耦合残差5.0642e-12、功率配对缺陷8.7823e-16，local/column ES最大CFL=0.0844717/0.0951952，EM combined CFL=0.4。诊断严格递增、数值有限，无减步事件，stderr为空。运行稳定，无需操作，未重复启动；3150 s及后续阶段仍未完成。

[01:18审计JSON](<ORIGINAL_CHECKOUT>/docs/EPB_MONITOR_AUDIT_2026-10-06_0118.json)。

### 2026-10-06 01:48 定时检查（Asia/Shanghai）

按命令行确认唯一既有driver PID28312。可信checkpoint到2025 s/8100步，schema3、hash不变、dt=0.25 s、零减步，所有浮点数组有限、三轨正密度；诊断到2045 s、快照到2040 s。续算以来最大耦合残差5.0642e-12、功率配对缺陷8.7823e-16，local/column ES最大CFL=0.0990587/0.1107585，EM combined CFL=0.4。诊断严格递增、数值有限，无减步事件，stderr为空。运行稳定，无需操作，未重复启动；3150 s及后续阶段仍未完成。

[01:48审计JSON](<ORIGINAL_CHECKOUT>/docs/EPB_MONITOR_AUDIT_2026-10-06_0148.json)。

### 2026-10-06 02:18 定时检查（Asia/Shanghai）

按命令行确认唯一既有driver PID28312。可信checkpoint到2125 s/8500步，schema3、hash不变、dt=0.25 s、零减步，所有浮点数组有限、三轨正密度；诊断到2134 s、快照到2130 s。续算以来最大耦合残差5.0642e-12、功率配对缺陷8.7823e-16，local/column ES最大CFL=0.1163704/0.1289058，EM combined CFL=0.4。诊断严格递增、数值有限，无减步事件，stderr为空。运行稳定，无需操作，未重复启动；3150 s及后续阶段仍未完成。

[02:18审计JSON](<ORIGINAL_CHECKOUT>/docs/EPB_MONITOR_AUDIT_2026-10-06_0218.json)。

### 2026-10-06 02:48 定时检查（Asia/Shanghai）

按命令行确认唯一既有driver PID28312。可信checkpoint到2200 s/8800步，schema3、hash不变、dt=0.25 s、零减步，所有浮点数组有限、三轨正密度；诊断到2223 s、快照到2220 s。续算以来最大耦合残差5.0642e-12、功率配对缺陷8.7823e-16，local/column ES最大CFL=0.1368115/0.1502671，EM combined CFL=0.4。诊断严格递增、数值有限，无减步事件，stderr为空。运行稳定，无需操作，未重复启动；3150 s及后续阶段仍未完成。

[02:48审计JSON](<ORIGINAL_CHECKOUT>/docs/EPB_MONITOR_AUDIT_2026-10-06_0248.json)。

### 2026-10-06 03:18 定时检查（Asia/Shanghai）

按命令行确认唯一既有driver PID28312。可信checkpoint到2300 s/9200步，schema3、hash不变、dt=0.25 s、零减步，所有浮点数组有限、三轨正密度；诊断到2313 s、快照到2310 s。续算以来最大耦合残差5.0642e-12、功率配对缺陷8.7823e-16，local/column ES最大CFL=0.1622783/0.1785238，EM combined CFL=0.4。诊断严格递增、数值有限，无减步事件，stderr为空。运行稳定，无需操作，未重复启动；3150 s及后续阶段仍未完成。

[03:18审计JSON](<ORIGINAL_CHECKOUT>/docs/EPB_MONITOR_AUDIT_2026-10-06_0318.json)。

### 2026-10-06 03:48 定时检查（Asia/Shanghai）

按命令行确认唯一既有driver PID28312。可信checkpoint到2400 s/9600步，schema3、hash不变、dt=0.25 s、零减步，所有浮点数组有限、三轨正密度；诊断到2404 s、快照到2400 s。续算以来最大耦合残差5.0642e-12、功率配对缺陷9.8956e-16，local/column ES最大CFL=0.1982591/0.2213765，均低于0.25门限，EM combined CFL=0.4。诊断严格递增、数值有限，无减步事件，stderr为空。ES CFL随演化上升，继续由runner门禁处理，不人为改参数或抢跑。运行稳定，无需操作，未重复启动；3150 s及后续阶段仍未完成。

[03:48审计JSON](<ORIGINAL_CHECKOUT>/docs/EPB_MONITOR_AUDIT_2026-10-06_0348.json)。

### 2026-10-06 04:18 定时检查（Asia/Shanghai）

按命令行确认唯一既有driver PID28312。可信checkpoint到2475 s/9900步，schema3、hash不变、宏观dt=0.25 s、零宏观减步，浮点数组有限、三轨正密度；诊断到2492 s、快照到2490 s。续算以来最大耦合残差5.0642e-12、功率配对缺陷9.8956e-16；local/column ES接受步诊断最大CFL=0.2483744/0.2494876，低于0.25，EM combined CFL=0.4。ES已自动启用两子步（子步dt=0.125 s），最新诊断local/column子步CFL=0.1284132/0.1349909；事件尾部计数local23、column150。这是既有runner正常subcycling，不是宏观减步或连续拒步，未人为调参数。诊断严格递增、数值有限，stderr为空。稳定运行，3150 s及后续阶段仍未完成。

[04:18审计JSON](<ORIGINAL_CHECKOUT>/docs/EPB_MONITOR_AUDIT_2026-10-06_0418.json)。

### 2026-10-06 04:48 定时检查（Asia/Shanghai）

按命令行确认唯一既有driver PID28312。可信checkpoint到2550 s/10200步，schema3、hash不变、宏观dt=0.25 s、零宏观减步，浮点数组有限、三轨正密度；诊断到2574 s、快照到2570 s。续算以来最大耦合残差5.0642e-12、功率配对缺陷9.8956e-16，local/column ES最大接受步诊断CFL=0.2483744/0.2494876，EM combined CFL=0.4。ES继续正常两子步（0.125 s），最新local/column子步CFL=0.1560375/0.1756061；事件尾部计数local351、column478。诊断严格递增、数值有限，stderr为空；没有宏观减步或连续拒步。运行稳定，无需操作，未重复启动；3150 s及后续阶段仍未完成。

[04:48审计JSON](<ORIGINAL_CHECKOUT>/docs/EPB_MONITOR_AUDIT_2026-10-06_0448.json)。

### 2026-10-06 05:18 定时检查（Asia/Shanghai）

按命令行确认唯一既有driver PID28312。可信checkpoint到2650 s/10600步，schema3、hash不变、宏观dt=0.25 s、零宏观减步，浮点数组有限、三轨正密度；诊断到2657 s、快照到2650 s。续算以来最大耦合残差5.0642e-12、功率配对缺陷9.8956e-16，local/column ES最大接受步诊断CFL=0.2483744/0.2494876，EM combined CFL=0.4。ES继续正常两子步（0.125 s），最新local/column子步CFL=0.1728259/0.1753018；事件尾部计数local685、column812。诊断严格递增、数值有限，stderr为空；没有宏观减步或连续拒步。运行稳定，无需操作，未重复启动；3150 s及后续阶段仍未完成。

[05:18审计JSON](<ORIGINAL_CHECKOUT>/docs/EPB_MONITOR_AUDIT_2026-10-06_0518.json)。

### 2026-10-06 05:48 定时检查（Asia/Shanghai）

按命令行确认唯一既有driver PID28312。可信checkpoint到2725 s/10900步，schema3、hash不变、宏观dt=0.25 s、零宏观减步，浮点数组有限、三轨正密度；诊断到2738 s、快照到2730 s。续算以来最大耦合残差5.0642e-12、功率配对缺陷9.8956e-16，local/column ES最大接受步诊断CFL=0.2483744/0.2494876，EM combined CFL=0.4。最新ES两子步dt=0.125 s，local/column子步CFL=0.1963297/0.2020478；事件尾部计数local1009（额外子步1009）、column1136（额外子步1202，历史有多于两子步）。诊断严格递增、数值有限，stderr为空；没有宏观减步或连续拒步。运行稳定，无需操作，未重复启动；3150 s及后续阶段仍未完成。

[05:48审计JSON](<ORIGINAL_CHECKOUT>/docs/EPB_MONITOR_AUDIT_2026-10-06_0548.json)。

### 2026-10-06 06:18 定时检查（Asia/Shanghai）

按命令行确认唯一既有driver PID28312。可信checkpoint到2800 s/11200步，schema3、hash不变、宏观dt=0.25 s、零宏观减步，浮点数组有限、三轨正密度；诊断到2820 s、快照到2820 s。续算以来最大耦合残差5.0642e-12、功率配对缺陷9.8956e-16，local/column ES最大接受步诊断CFL=0.2483744/0.2494876，EM combined CFL=0.4。最新ES两子步dt=0.125 s，local/column子步CFL=0.2345068/0.2402687，仍低于0.25；事件尾部计数local1337（额外1337）、column1464（额外1530）。诊断严格递增、数值有限，stderr为空；没有宏观减步或连续拒步。运行稳定，无需操作，未重复启动；3150 s及后续阶段仍未完成。

[06:18审计JSON](<ORIGINAL_CHECKOUT>/docs/EPB_MONITOR_AUDIT_2026-10-06_0618.json)。

### 2026-10-06 06:48 定时检查（Asia/Shanghai）

按命令行确认唯一既有driver PID28312。可信checkpoint到2875 s/11500步，schema3、hash不变、宏观dt=0.25 s、零宏观减步，浮点数组有限、三轨正密度；诊断到2894 s、快照到2890 s。续算以来最大耦合残差5.0642e-12、功率配对缺陷9.8956e-16，local/column ES最大接受步诊断CFL=0.2498382/0.2497356，EM combined CFL=0.4。最新ES自动四子步dt=0.0625 s，local/column子步CFL=0.1373130/0.2386006，均低于0.25；事件尾部计数local1632（额外2010）、column1759（额外2225）。诊断严格递增、数值有限，stderr为空；没有宏观减步或连续拒步。运行稳定，无需操作，未重复启动；3150 s及后续阶段仍未完成。

[06:48审计JSON](<ORIGINAL_CHECKOUT>/docs/EPB_MONITOR_AUDIT_2026-10-06_0648.json)。

### 2026-10-06 07:18 定时检查（Asia/Shanghai）

按命令行确认唯一既有driver PID28312。可信checkpoint到2950 s/11800步，schema3、hash不变、宏观dt=0.25 s、零宏观减步，浮点数组有限、三轨正密度；诊断到2965 s、快照到2960 s。续算以来最大耦合残差5.0642e-12、功率配对缺陷9.8956e-16，local/column ES最大接受步诊断CFL=0.2498382/0.2497356，EM combined CFL=0.4。最新ES四子步dt=0.0625 s，local/column子步CFL=0.1505911/0.1468607；事件尾部计数local1915（额外2859）、column2042（额外3074）。诊断严格递增、数值有限，stderr为空；没有宏观减步或连续拒步。运行稳定，无需操作，未重复启动；3150 s及后续阶段仍未完成。

[07:18审计JSON](<ORIGINAL_CHECKOUT>/docs/EPB_MONITOR_AUDIT_2026-10-06_0718.json)。

### 2026-10-06 07:48 定时检查（Asia/Shanghai）

按命令行确认唯一既有driver PID28312。可信checkpoint到3025 s/12100步，schema3、hash不变、宏观dt=0.25 s、零宏观减步，浮点数组有限、三轨正密度；诊断到3035 s、快照到3030 s。续算以来最大耦合残差5.0642e-12、功率配对缺陷9.8956e-16，local/column ES最大接受步诊断CFL=0.2498382/0.2497356，EM combined CFL=0.4。最新ES四子步dt=0.0625 s，local/column子步CFL=0.1623239/0.1558162；事件尾部计数local2197（额外3705）、column2324（额外3920）。诊断严格递增、数值有限，stderr为空；没有宏观减步或连续拒步。运行稳定，无需操作，未重复启动；3150 s及后续阶段仍未完成。

[07:48审计JSON](<ORIGINAL_CHECKOUT>/docs/EPB_MONITOR_AUDIT_2026-10-06_0748.json)。

### 2026-10-06 08:18 定时检查（Asia/Shanghai）

按命令行确认唯一既有driver PID28312。可信checkpoint到3100 s/12400步，schema3、hash不变、宏观dt=0.25 s、零宏观减步，浮点数组有限、三轨正密度；诊断到3106 s、快照到3100 s。续算以来最大耦合残差5.0642e-12、功率配对缺陷9.8956e-16，local/column ES最大接受步诊断CFL=0.2498382/0.2497356，EM combined CFL=0.4。最新ES四子步dt=0.0625 s，local/column子步CFL=0.1672329/0.1619703；事件尾部计数local2478（额外4548）、column2605（额外4763）。诊断严格递增、数值有限，stderr为空；没有宏观减步或连续拒步。运行稳定，无需操作，未重复启动；3150 s及后续阶段仍未完成，不能提前宣称终态验收通过。

[08:18审计JSON](<ORIGINAL_CHECKOUT>/docs/EPB_MONITOR_AUDIT_2026-10-06_0818.json)。

## 11. 2026-10-06 08:48：细网格3150 s完成，但严格验收失败

唯一driver已结束；checkpoint/final/末快照三轨密度及EM特征变量一致性通过，hash不变、12600步、dt=0.25 s、零宏观减步。严格acceptance为2失败/4警告：local/column ES周期重复端点差3.6087e-5/1.3077e-4超1e-6；EM5.7146e-15。数值有限、正密度、残差、CFL、FAC/功率配对、采样能量门禁通过。不可称严格验收通过，不继续5000 s或跳到物理扫描。

源码存在重复周期点与全列np.roll推进语义不一致；既有PDM7小型算子复现确认首尾相等初值不保证RHS首尾相等。细快照首次超门限column ES2600 s、local ES2810 s。源码尚未修复，旧档未改、不放宽阈值、不手改hash；下一步针对周期离散语义审计/修复及测试，改变轨迹须显式版本/hash隔离。

粗细同3150 s图件均已view_image QA：细ES更多分支，EM仍宽平滑、无清晰Y。300–3150 s同宏观步等权gamma粗/细EM4.15913e-4/4.09892e-4 s^-1（差-1.4478%）；EM共同节点横向扰动L2差2.8841%。ES横向结构差大，不能宣称严格收敛。

[完整失败分析、数值与能量、同时间对比及限制报告](<ORIGINAL_CHECKOUT>/docs/EPB_FINE_T3150_REVIEW_2026-10-06.md)、[对照JSON](<ORIGINAL_CHECKOUT>/docs/EPB_RESOLUTION_COMPARISON_T3150_2026-10-06.json)。后续定时跟进首先解决本数值失败，不重复运行已完成3150 s轨迹、不静默修改旧f7轨迹。

## 12. 2026-10-06 09:18跟进：周期离散契约修正，39项测试通过

FAC与所有输运/势算子均使用全部存储列为独立周期格，历史细/粗实际计算周期251.25/252.5 km，而种子波长250 km。增加默认关闭、显式进入hash的unique_periodic_cells；默认行为和两生产hash保持。验收/图像对新布局保留所有唯一列；没有改旧数组或放宽旧门限。相关测试39项通过，只有0.02 s小型三轨验证。已准备独立200×401/FAC(8,8)的60 s配置，未启动；须明确同意新hash独立验证后才能开展，不能用f7 checkpoint静默启用。无python进程；不继续失败轨迹到5000 s，不跳到新物理扫描。

[修正JSON](<ORIGINAL_CHECKOUT>/docs/EPB_PERIODIC_GRID_CORRECTION_2026-10-06.json)、[完整报告](<ORIGINAL_CHECKOUT>/docs/EPB_FINE_T3150_REVIEW_2026-10-06.md)、[测试日志](<ORIGINAL_CHECKOUT>/tmp/periodic_grid_focused_tests_20261006.log)。

## 13. 2026-10-06 14:10：用户授权并启动独立唯一周期格60 s验证

用户明确回复“启动”。按命令行再次核查，无活跃EPB Python计算；以新driver、resume=True启动独立200×401/FAC(8,8)152模态三轨，初始PID23740。新hash为28c1470354b3f7bb60a6129ba29566fb2d6bbe937aba3426fd5b703a0b357fa1，唯一200列覆盖250 km周期。实际源码与新输出仍在C盘工作树；原f7的3150 s输出/checkpoint/hash保留，没有重启旧driver、prepare或重复粗网格算例。新driver自带60 s终态数组/hash审计、后处理和严格验收。启动不等于完成或验收通过；完成后仍需view_image QA。仅60 s独立验证获授权，未启动300/3150/5000 s延伸和新物理扫描。

[启动记录与日志路径](<ORIGINAL_CHECKOUT>/tmp/em_unique_periodic_t60_launch_20261006.json)、[新driver](<ORIGINAL_CHECKOUT>/tmp/run_em_unique_periodic_t60_20261006.py)。不早于14:40再做进度检查，此后每30分钟至多一次，不高频轮询。


## 14. 2026-10-06 14:48：独立唯一周期格60 s验收通过、视觉QA完成

按命令行查无Python进程，driver正常完成且stderr为空。可信schema3 checkpoint为60 s/240步，hash28c1470354b3f7bb60a6129ba29566fb2d6bbe937aba3426fd5b703a0b357fa1；终态checkpoint/final/末快照密度逐元素一致，线变量checkpoint/final一致，所有浮点数组有限、三轨正密度。dt=0.25 s，零宏观减步；60条诊断严格递增到60 s。唯一200列计算周期250 km，不删除末列、不要求两个不同周期格首尾相等；历史f7验收失败保持原判。

严格60 s门禁通过，0失败、2警告（EM combined CFL=0.4接近设定上限；开放垂直边界未记录通量以闭合粒子账）。ES CFL最大local0.0160429、column0.00266135；最大耦合残差6.72059e-13、功率配对缺陷5.93530e-16、表面电流平衡比7.73666e-17。FAC/重力有效期中位数0.242602。R=0和feedback=1保持。

终态线能量36509.49783 J；采样净边界功36514.38699 J，功减能4.88916 J，相对差9.37869e-5、生能比例0；远端吸收52130.49922 J。此为代表stage功率梯形积分，非精确RK-stage功；开放边界粒子列漂移不等于封闭守恒误差。

0–60 s每0.25 s等权OLS拟合ln RMS，241样本：local ES gamma=-4.24420e-4、column ES=4.03876e-4、column EM=3.12259e-4 s^-1，含20 s feedback ramp，因此只是早期窗。EM相对column ES增长率降低约22.68%。图件已view_image：三轨均早期平滑扰动，无D>=0.25泡或Y分叉；渲染无明显接缝/缺陷。60 s成功不能证明旧晚期端点风险完全消除或成熟期收敛。

下一建议：先保留/归档60 s成果，再用同新hash从现有checkpoint延伸300 s；当前未启动延伸。旧3150/5000 s均未重跑。后续物理扫描未实施。

[本次审计JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_T60_AUDIT_2026-10-06.json)、[严格验收](<ORIGINAL_CHECKOUT>/output/em_growth/unique_periodic_200x401_k8_three_track_20261006/acceptance_t60.json)、[图件](<ORIGINAL_CHECKOUT>/output/em_growth/unique_periodic_200x401_k8_three_track_20261006/postprocess_t60/em_growth_comparison.png)、[源数据](<ORIGINAL_CHECKOUT>/output/em_growth/unique_periodic_200x401_k8_three_track_20261006/postprocess_t60/em_growth_source_data.npz)。

## 15. 2026-10-06 15:17：用户授权同hash从60 s续算300 s

启动前命令行核查无EPB Python进程。已复制保护20个60 s成果文件至原输出下archive_t60（包括checkpoint、final、report、诊断、全部快照和后处理）；SHA256清单已保存，checkpoint副本逐字节相同，不删除或移动原成果。用户明确授权从现有checkpoint同hash续算300 s，唯一新driver初始PID16656，resume=True。同hash28c1470354b3f7bb60a6129ba29566fb2d6bbe937aba3426fd5b703a0b357fa1已核对；只修改EM runner的t_end_s，Li配置完整保留（其t_end_s字段参与hash，EM runner以EM配置终点为准）。先前尝试同步修改Li终点被hash断言拦截，未启动计算或改checkpoint；已改为保持Li配置的正确同hash路径。

新driver包含300 s终态数组/hash/有限正密度审计、后处理和严格验收。下一进度检查不早于15:47，此后每30分钟至多一次。当前只是启动，尚不能宣称300 s验收通过。旧f7和粗3150/5000 s成果未改，未启动新3150/5000 s或后三项物理扫描。

[启动与日志记录](<ORIGINAL_CHECKOUT>/tmp/em_unique_periodic_t300_launch_20261006.json)、[续算driver](<ORIGINAL_CHECKOUT>/tmp/run_em_unique_periodic_t300_20261006.py)、[60 s归档SHA256清单](<ORIGINAL_CHECKOUT>/output/em_growth/unique_periodic_200x401_k8_three_track_20261006/archive_t60/archive_manifest.json)。

### 2026-10-06 15:48同hash300 s续算检查

按命令行确认唯一driver PID16656。可信checkpoint150 s/600步，schema3、新hash保持，全部浮点数组有限且三轨正密度；诊断到158.0 s、严格递增，dt0.25 s、零减步，无ES subcycling。stderr为空；耦合残差/功率配对/CFL门禁无异常。继续现有进程，不重复启动。300 s验收与视觉QA仍待完成。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-06_1548.json)。

### 2026-10-06 16:18同hash300 s续算检查

按命令行确认唯一driver PID16656。可信checkpoint225 s/900步，schema3、新hash保持，所有浮点数组有限且三轨正密度；诊断到247 s且严格递增。dt0.25 s、零减步，最大耦合残差6.74112e-13、功率配对缺陷6.22969e-16、CFL local0.0160429/column0.00266135/EM0.4；stderr为空。稳定运行，不重复启动；300 s终态验收、后处理与视觉QA未完成，下次不早于16:48检查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-06_1618.json)。

## 16. 2026-10-06 16:48：修正三轨300 s验收通过，视觉QA与旧存档对照完成

现有checkpoint同hash续算完成300 s/1200步，schema3、新hash保持；driver退出且stderr空，按命令行无Python进程。终态一致性、有限数组、正密度、零减步、残差/功率配对及采样能量门禁通过。严格验收0失败2警告（EM CFL上限、开放粒子通量预算未闭合）。新与旧细同300 s图件均view_image QA，EM无D>=0.25泡、无Y。200–300 s等权401样本拟合：修正EM gamma3.24473e-4 s^-1，对旧细变化−0.0453%；横向结构L2差0.013762%。不同计算周期/FAC截断及早期证据不能证明成熟期或严格收敛。建议归档300 s后同hash续算3150 s；当前尚未启动，后续物理扫描未实施。

[完整报告与图件/源数据](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_T300_REVIEW_2026-10-06.md)、[审计JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_T300_AUDIT_2026-10-06.json)。旧f7验收失败保持原判，所有旧成果保留。

## 17. 2026-10-06 20:38：用户授权同hash从300 s续算3150 s

已归档300 s成果51文件至archive_t300，含checkpoint/final/report/诊断/全部快照/后处理与验收；SHA256清单保存，checkpoint复制逐字节一致，不覆盖archive_t60。启动前两次按命令行确认无EPB Python计算。唯一新driver初始PID21148，resume=True，从300 s/1200步接续至3150 s，同hash28c1470354b3f7bb60a6129ba29566fb2d6bbe937aba3426fd5b703a0b357fa1。仅EM runner终止时间变化，Li配置完整保留。driver含终态hash/三轨数组/有限正密度审计、后处理和严格验收；完成后另需view_image及同3150 s粗/旧细比较。现在仅启动，不声称验收完成。旧f7、粗3150/5000 s成果未改，未启动修正5000 s或后续物理扫描。

不早于21:08再查进度，此后每30分钟至多一次；以checkpoint而非300 s旧final/report/status为真值。现有唯一driver持续运行时禁止重复启动。

[启动记录与日志](<ORIGINAL_CHECKOUT>/tmp/em_unique_periodic_t3150_launch_20261006.json)、[新续算driver](<ORIGINAL_CHECKOUT>/tmp/run_em_unique_periodic_t3150_20261006.py)、[300 s归档清单](<ORIGINAL_CHECKOUT>/output/em_growth/unique_periodic_200x401_k8_three_track_20261006/archive_t300/archive_manifest.json)。

### 2026-10-06 21:19同hash3150 s续算检查

命令行确认唯一新driver PID21148。可信checkpoint400 s/1600步，schema3、新hash保持、浮点数组有限、三轨正密度；诊断到422 s且严格递增。dt0.25 s、零减步，两ES仍单子步；最大耦合残差6.74112e-13、功率配对缺陷6.22969e-16，CFL local0.0160429/column0.00266135/EM0.4；stderr为空。运行稳定，无需操作，不重复启动。3150 s终态及严格验收、视觉QA尚未完成。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-06_2119.json)。

### 2026-10-06 21:50同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint500.0 s/2000步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到514.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-06_2150.json)。

### 2026-10-06 22:49同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint675.0 s/2700步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到692.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-06_2249.json)。

### 2026-10-06 23:19同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint775.0 s/3100步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到782.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-06_2319.json)。

### 2026-10-06 23:49同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint850.0 s/3400步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到872.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-06_2349.json)。

### 2026-10-07 00:19同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint950.0 s/3800步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到961.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_0019.json)。

### 2026-10-07 00:49同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint1050.0 s/4200步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到1051.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_0049.json)。

### 2026-10-07 01:19同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint1125.0 s/4500步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到1141.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_0119.json)。

### 2026-10-07 01:49同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint1225.0 s/4900步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到1231.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_0149.json)。

### 2026-10-07 02:19同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint1300.0 s/5200步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到1321.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_0219.json)。

### 2026-10-07 02:49同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint1400.0 s/5600步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到1412.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_0249.json)。

### 2026-10-07 03:19同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint1500.0 s/6000步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到1503.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_0319.json)。

### 2026-10-07 03:49同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint1575.0 s/6300步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到1594.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_0349.json)。

### 2026-10-07 04:19同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint1675.0 s/6700步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到1685.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_0419.json)。

### 2026-10-07 04:49同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint1775.0 s/7100步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到1776.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_0449.json)。

### 2026-10-07 05:19同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint1850.0 s/7400步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到1867.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_0519.json)。

### 2026-10-07 05:49同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint1950.0 s/7800步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到1959.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_0549.json)。

### 2026-10-07 06:19同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint2050.0 s/8200步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到2050.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_0619.json)。

### 2026-10-07 06:49同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint2125.0 s/8500步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到2141.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_0649.json)。

### 2026-10-07 07:19同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint2225.0 s/8900步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到2232.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_0719.json)。

### 2026-10-07 07:49同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint2300.0 s/9200步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到2324.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_0749.json)。

### 2026-10-07 08:19同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint2400.0 s/9600步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到2415.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_0819.json)。

### 2026-10-07 08:49同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint2500.0 s/10000步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到2502.0 s，宏观减步累计0；最新ES子步[2, 2]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_0849.json)。

### 2026-10-07 09:19同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint2575.0 s/10300步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到2585.0 s，宏观减步累计0；最新ES子步[2, 2]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_0919.json)。

### 2026-10-07 09:49同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint2650.0 s/10600步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到2667.0 s，宏观减步累计0；最新ES子步[2, 2]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_0949.json)。

### 2026-10-07 10:19同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint2725.0 s/10900步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到2749.0 s，宏观减步累计0；最新ES子步[2, 2]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_1019.json)。

### 2026-10-07 10:49同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint2825.0 s/11300步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到2829.0 s，宏观减步累计0；最新ES子步[2, 4]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_1049.json)。

### 2026-10-07 11:19同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint2900.0 s/11600步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到2901.0 s，宏观减步累计0；最新ES子步[4, 4]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_1119.json)。

### 2026-10-07 11:49同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint2950.0 s/11800步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到2971.0 s，宏观减步累计0；最新ES子步[4, 8]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_1149.json)。

### 2026-10-07 12:19同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint3025.0 s/12100步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到3028.0 s，宏观减步累计0；最新ES子步[4, 16]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_1219.json)。

### 2026-10-07 12:49同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint3075.0 s/12300步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到3086.0 s，宏观减步累计0；最新ES子步[4, 8]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_1249.json)。

### 2026-10-07 13:19同hash3150 s续算检查

命令行核查EPB进程数0；可信checkpoint3150.0 s/12600步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到3150.0 s，宏观减步累计0；最新ES子步[4, 8]。检查结果terminal_requires_review，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_1319.json)。


## 2026-10-07 13:19 唯一周期3150 s完成复核

三轨同hash到3150 s/12600步，终态独立审计、严格阶段验收及三张同时间图像QA完成：0失败4警告。EM仍宽平滑无清晰Y。共同300–3150 s窗口EM增长率4.0990676e-4 s^-1；旧细差+0.00365%，旧粗差−1.44416%。旧f7两ES端点失败原判不变。5000 s尚未授权或启动，物理后三项扫描未实施。详情见EPB_UNIQUE_PERIODIC_T3150_REVIEW_2026-10-07.md及同名AUDIT JSON。


## 2026-10-07 13:42 同hash续算5000 s启动

用户明确授权先归档3150 s再续算5000 s。archive_t3150复制344文件，全文件SHA256核对一致、checkpoint逐字节一致；未删除原成果。唯一driver run_em_unique_periodic_t5000_20261007.py 从3150 s/12600步以resume=True启动，初始PID6392（后续必须按命令行核查），hash28c1470354b3f7bb60a6129ba29566fb2d6bbe937aba3426fd5b703a0b357fa1不变。Li配置保留，只改EM runner t_end_s=5000。下一检查不早于14:12；每30分钟最多一次，稳定保持安静。完成后核查terminal_audit_t5000.json/acceptance_t5000.json、view_image QA和同时间对照。3150 s审计保留；后三项物理扫描未实施。启动日志与记录tmp/em_unique_periodic_t5000_launch_20261007.json。

### 2026-10-07 14:19同hash5000 s续算检查

命令行核查EPB进程数1；可信checkpoint3225.0 s/12900步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到3228.0 s，宏观减步累计0；最新ES子步[4, 4]。检查结果running_stable，异常[]。不重复启动；5000 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_1419.json)。

### 2026-10-07 14:49同hash5000 s续算检查

命令行核查EPB进程数1；可信checkpoint3275.0 s/13100步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到3296.0 s，宏观减步累计0；最新ES子步[4, 4]。检查结果running_stable，异常[]。不重复启动；5000 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_1449.json)。

### 2026-10-07 15:19同hash5000 s续算检查

命令行核查EPB进程数1；可信checkpoint3350.0 s/13400步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到3366.0 s，宏观减步累计0；最新ES子步[4, 4]。检查结果running_stable，异常[]。不重复启动；5000 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_1519.json)。

### 2026-10-07 15:49同hash5000 s续算检查

命令行核查EPB进程数1；可信checkpoint3425.0 s/13700步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到3436.0 s，宏观减步累计0；最新ES子步[4, 4]。检查结果running_stable，异常[]。不重复启动；5000 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_1549.json)。

### 2026-10-07 16:19同hash5000 s续算检查

命令行核查EPB进程数1；可信checkpoint3500.0 s/14000步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到3505.0 s，宏观减步累计0；最新ES子步[4, 4]。检查结果running_stable，异常[]。不重复启动；5000 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_1619.json)。

### 2026-10-07 16:49同hash5000 s续算检查

命令行核查EPB进程数1；可信checkpoint3575.0 s/14300步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到3575.0 s，宏观减步累计0；最新ES子步[4, 4]。检查结果running_stable，异常[]。不重复启动；5000 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_1649.json)。

### 2026-10-07 17:19同hash5000 s续算检查

命令行核查EPB进程数1；可信checkpoint3625.0 s/14500步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到3645.0 s，宏观减步累计0；最新ES子步[4, 4]。检查结果running_stable，异常[]。不重复启动；5000 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_1719.json)。

### 2026-10-07 17:49同hash5000 s续算检查

命令行核查EPB进程数1；可信checkpoint3700.0 s/14800步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到3714.0 s，宏观减步累计0；最新ES子步[4, 4]。检查结果running_stable，异常[]。不重复启动；5000 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_1749.json)。

### 2026-10-07 18:19同hash5000 s续算检查

命令行核查EPB进程数1；可信checkpoint3775.0 s/15100步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到3783.0 s，宏观减步累计0；最新ES子步[4, 4]。检查结果running_stable，异常[]。不重复启动；5000 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_1819.json)。

### 2026-10-07 18:49同hash5000 s续算检查

命令行核查EPB进程数1；可信checkpoint3850.0 s/15400步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到3853.0 s，宏观减步累计0；最新ES子步[4, 4]。检查结果running_stable，异常[]。不重复启动；5000 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_1849.json)。

### 2026-10-07 19:19同hash5000 s续算检查

命令行核查EPB进程数1；可信checkpoint3900.0 s/15600步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到3922.0 s，宏观减步累计0；最新ES子步[4, 4]。检查结果running_stable，异常[]。不重复启动；5000 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_1919.json)。

### 2026-10-07 19:49同hash5000 s续算检查

命令行核查EPB进程数1；可信checkpoint3975.0 s/15900步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到3992.0 s，宏观减步累计0；最新ES子步[4, 4]。检查结果running_stable，异常[]。不重复启动；5000 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_1949.json)。

### 2026-10-07 20:20同hash5000 s续算检查

命令行核查EPB进程数1；可信checkpoint4050.0 s/16200步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到4059.0 s，宏观减步累计0；最新ES子步[4, 8]。检查结果running_stable，异常[]。不重复启动；5000 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_2020.json)。

### 2026-10-07 21:20同hash5000 s续算检查

命令行核查EPB进程数1；可信checkpoint4175.0 s/16700步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到4193.0 s，宏观减步累计0；最新ES子步[4, 8]。检查结果running_stable，异常[]。不重复启动；5000 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_2120.json)。

### 2026-10-07 22:19同hash5000 s续算检查

命令行核查EPB进程数1；可信checkpoint4300.0 s/17200步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到4318.0 s，宏观减步累计0；最新ES子步[4, 8]。检查结果running_stable，异常[]。不重复启动；5000 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_2219.json)。

### 2026-10-07 22:50同hash5000 s续算检查

命令行核查EPB进程数1；可信checkpoint4375.0 s/17500步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到4381.0 s，宏观减步累计0；最新ES子步[4, 8]。检查结果running_stable，异常[]。不重复启动；5000 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_2250.json)。

### 2026-10-07 23:49同hash5000 s续算检查

命令行核查EPB进程数1；可信checkpoint4500.0 s/18000步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到4506.0 s，宏观减步累计0；最新ES子步[4, 8]。检查结果running_stable，异常[]。不重复启动；5000 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-07_2349.json)。

### 2026-10-08 00:49同hash5000 s续算检查

命令行核查EPB进程数1；可信checkpoint4625.0 s/18500步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到4632.0 s，宏观减步累计0；最新ES子步[4, 8]。检查结果running_stable，异常[]。不重复启动；5000 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-08_0049.json)。

### 2026-10-08 01:49同hash5000 s续算检查

命令行核查EPB进程数1；可信checkpoint4750.0 s/19000步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到4758.0 s，宏观减步累计0；最新ES子步[4, 8]。检查结果running_stable，异常[]。不重复启动；5000 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-08_0149.json)。

### 2026-10-08 02:49同hash5000 s续算检查

命令行核查EPB进程数1；可信checkpoint4875.0 s/19500步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到4887.0 s，宏观减步累计0；最新ES子步[2, 4]。检查结果running_stable，异常[]。不重复启动；5000 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-08_0249.json)。

### 2026-10-08 03:50同hash5000 s续算检查

命令行核查EPB进程数0；可信checkpoint5000.0 s/20000步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到5000.0 s，宏观减步累计0；最新ES子步[2, 4]。检查结果terminal_requires_review，异常[]。不重复启动；5000 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_UNIQUE_PERIODIC_MONITOR_2026-10-08_0350.json)。


## 2026-10-08 03:50 唯一周期5000 s完成复核

同hash5000 s/20000步完成；终态独立数组审计、成熟期严格验收0失败4警告、同5000 s新细/旧粗图像QA通过。EM顶494 km仍宽平滑无Y；column ES到650 km边界有域截断限制。等时300–5000 s gammaEM4.1151655e-4 vs旧粗4.2139957e-4，EM结构差8.8808%，非严格收敛。采样能量生成4.48182e-6；开放粒子账缺通量。下一同步密度/FAC证据，后三项物理扫描未实施。本轮无额外生产启动。见EPB_UNIQUE_PERIODIC_T5000_REVIEW_2026-10-08.md及AUDIT JSON。


## 2026-10-08 04:19 同周期加密控制方案

源码发现FAC垂直模态长度ny*dy与面积归一化的附加混杂：旧粗402 km、新细401 km，面积差0.24938%。已准备固定ny401/dy1、横向周期250 km的100/200横向格点×FAC4/8四组合同；A/B/C配置解析及新hash通过，D复用已完成5000 s。没有启动新生产或修改源码/输出。该设计仅作横向网格与截断归因，不称二维收敛；垂直加密须另审几何约定。见EPB_MATCHED_PERIOD_REFINEMENT_PLAN_2026-10-08.md和JSON。


## 2026-10-08 11:51 EM无Y原因核查与控制实验启动

新增只读谱/冻结终态电势核查：同密度FAC垂直流速RMS为零FAC对照51.48%，横向为2.2168倍；5000s EM密度kx>8功率占比1.4193e-6，而column ES为5.667%。FAC改变流动方向，大尺度主导已确认，因果尚未证明。matched负载0.795775S、R0、传播20s非真实E电导测量。用户授权优化后启动C200x401/FAC4,4独立60s三轨控制，hash90347ce5dc069c5d9a799a589f2997593d1b99478caff6f6c13101d799fb2e5a，resume=True，新输出，初PID10844仅启动记录。下一检查不早于12:21:30，阶段验收/QA/归档后同hash300/3150，对照复用D。没有改旧源码/输出或弱反馈；未声称Y已解决。自动跟进实际PAUSED，已询问是否恢复，尚未更改状态。见EPB_EM_BIFURCATION_DIAGNOSIS_2026-10-08.md及JSON。


## 2026-10-08T13:52:21.3455610+08:00 C组FAC截断控制60s验收完成，续算300s

C60s/240步终态数组一致、有限正密度、零减步，严格0失败2警告（EM CFL近上限、开放粒子通量未记录），view_image QA通过，仍早期平滑。归档22文件全部SHA256一致、checkpoint逐字节一致。只修改EM终点，Li配置保留，从60s同hash90347ce5dc069c5d9a799a589f2997593d1b99478caff6f6c13101d799fb2e5a、resume=True续算300s。新driver tmp/run_em_bifurcation_control_C_t300_20261008.py；初PID16592仅启动记录。自动跟进实际仍PAUSED，待用户是否恢复选择。成熟期原因未确定，未声称Y已改善。


用户2026-10-08明确要求恢复30分钟自动跟进；已将epb-2恢复ACTIVE，更新提示为C60→300/后续3150阶段审计和同hash恢复。旧epb仍暂停；稳定状态安静，仅实质发现/完成/失败/异常/需用户处理通知。


2026-10-08 14:24 C300续算例行检查：checkpoint200.0s/800步，hash/有限正密度通过，零减步；实际driver命令行确认，stderr空，残差/功率配对/CFL正常。稳定无行动变化，不重复启动。见EPB_CONTROL_C_MONITOR_2026-10-08_1424.json。


2026-10-08 14:54 C300续算例行检查：checkpoint275.0s/1100步，hash/有限正密度通过，零减步；实际driver命令行确认，stderr空，残差/功率配对/CFL通过。诊断到276.0s，尚未完成，不重复启动。见EPB_CONTROL_C_MONITOR_2026-10-08_1454.json。


# C组300 s验收与同周期FAC截断对照（2026-10-08）

固定200×401网格、250 km唯一周期及401 km垂直模态尺度，C采用FAC(4,4)44模态，D采用(8,8)152模态；三轨、反馈1、matched边界及背景配置保持一致。

C完成300 s/1200步，hash保持90347ce5dc069c5d9a799a589f2997593d1b99478caff6f6c13101d799fb2e5a。独立审计确认checkpoint/final/快照密度及线变量一致、有限正密度、零宏步减半；严格验收0失败2警告（EM CFL近上限、开放粒子通量账缺失）。两组300 s图像已用view_image核查，早期平滑，无清晰Y。

以D为分母：EM密度L2差0.0115954%，横向结构L2差0.148901%，纯EM差场L2差2.978104%；两条ES密度差约1e-15，几何差场差约6e-14。共同200–300 s窗口、0.25 s等时logRMS插值OLS：gammaEM C=3.259830925e-4 s^-1，D=3.244725760e-4 s^-1，相差+0.465530%。这是早期固定网格的FAC截断敏感性，不能证明成熟期分叉原因或差场收敛。

archive_t300复制53文件，全部SHA256一致，checkpoint逐字节一致；archive_t60及所有原输出保留。16:00:03 Asia/Shanghai已从300 s checkpoint同hash、resume=True启动唯一3150 s driver，仅改EM runner终点，Li配置完整保留。下一进度检查不早于16:30:03。完成后独立审计、严格验收、图像QA，并与D归档3150 s对照；不得按有无Y修改阈值。

采样功率积分非精确RK-stage功；开放粒子漂移不是封闭守恒残差。A/B尚未启动，后续真实E-region电导、默认0共同ExB及长度/Alfven速度扫描尚未实施。模型缺平行质量、完整双半球metric、化学、风剪切/PRE及动态E电导，非定量预测。

### 2026-10-08 16:55同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint675.0 s/2700步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到687.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_C_MONITOR_2026-10-08_1655.json)。

### 2026-10-08 17:54同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint1100.0 s/4400步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到1106.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_C_MONITOR_2026-10-08_1754.json)。

### 2026-10-08 18:54同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint1525.0 s/6100步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到1529.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_C_MONITOR_2026-10-08_1854.json)。

### 2026-10-08 19:54同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint1950.0 s/7800步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到1951.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_C_MONITOR_2026-10-08_1954.json)。

### 2026-10-08 20:56同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint2375.0 s/9500步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到2392.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_C_MONITOR_2026-10-08_2056.json)。

### 2026-10-08 21:54同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint2725.0 s/10900步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到2734.0 s，宏观减步累计0；最新ES子步[2, 2]。检查结果anomaly，异常['CFL gate']。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_C_MONITOR_2026-10-08_2154.json)。

21:54检查修正：监控文本转换误将原ES CFL门槛0.2500001写为0.2315001；现已恢复原门槛，仅复核同批保存数据。实际CFL local0.2494993/column0.2498158/EM0.4均通过；撤销CFL误报，结果running_stable。未改实验参数/验收阈值，未重新轮询。

### 2026-10-08 22:54同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint3000.0 s/12000步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到3012.0 s，宏观减步累计0；最新ES子步[4, 4]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_C_MONITOR_2026-10-08_2254.json)。

### 2026-10-08 23:54同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint3150.0 s/12600步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到3150.0 s，宏观减步累计0；最新ES子步[4, 4]。检查结果terminal_requires_review，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_C_MONITOR_2026-10-08_2354.json)。


# C组3150 s成熟期FAC截断对照（2026-10-08）

C组3150 s/12600步完成，hash保持，checkpoint/final/快照密度和线变量一致、有限正密度、零宏步减半。严格验收0失败4警告（三轨CFL近上限及开放粒子通量账缺失）。C/D同3150 s图像已view_image QA：ES分叉明显，EM均宽平滑无清晰Li式Y，图像无渲染缺陷。

共同300–3150 s等时0.25 s logRMS插值OLS gammaEM C=4.161006231e-4、D=4.099067571e-4 s^-1，C高1.51104%。以D为分母EM横向结构L2差2.93896%，密度差0.78933%，纯EM差场19.21605%，几何差场21.40456%。

关键限制：column ES横向结构差14.93733%，local ES差0.11453%。Li配置字典完全一致，EM只差FAC截断和运行标签/输出/终点；源代码evaluate_column_es_rhs明确为零FAC同几何对照。ES对照0–2000 s差约1e-14，2500 s column差1.03e-7，3000 s增至9.86277%，3150 s增至14.93733%。非线性数值敏感性是待核查解释，尚未证明唯一原因；纯EM差场不能全部解释成FAC截断效应。

下一步先检查ES数值重复性、子循环事件及执行环境/求解器路径，再推进A/B同周期网格对照；不重跑既有旧粗/D轨迹，不为制造Y修改反馈、FAC符号或验收门槛，不自动延长C5000。两种截断不能证明渐近收敛。采样功率非精确RK-stage功，开放粒子漂移非封闭守恒残差，模型非定量预测。


# ES重复性只读核查（2026-10-09）

只读审查已归档C/D3150 s诊断，无生产轨迹重跑或推进。共同3150个诊断样本的宏步dt与local ES子步数完全一致；column ES在220个共同样本子步数不一致，首次记录为2811 s（step11244）：C为2，D为4。

两组column ES子循环宏步事件均2763，额外子步C5557、D9071；采样最大子步C8、D16。首次采样子循环均2460 s，local ES均2493 s。子循环计数差异是已确认的数值路径差异，但不能证明它是最初触发因素：2500 s已有微小密度结构差，首次采样子步不一致晚于该差异。

源代码evaluate_column_es_rhs通过同几何零FAC重力电势求解，无FAC模态投影。Li参数一致；因此成熟ES差异不能当作物理FAC效应。非线性误差放大和后续自适应分支差异是证据支持的待核查解释，起始扰动来源仍未证实。配置hash未归档每次运行的dirty源码快照、MKL/PARDISO线程/runtime，不能假设这些完全相同。

下一步审查运行日志环境溯源与差异起始快照；制定独立、有限成本ES重复性诊断，不重跑禁用的旧粗/D生产轨迹，不改历史hash/数组/验收阈值，不自动延长C5000。三轨差场结论暂保留混杂限制。


# 冻结密度ES求解重复性诊断（2026-10-09）

采用同一C组2500 s归档column ES密度，分别在FAC4/8模型、MKL线程1/10四个独立进程中各求解3次零FAC RHS；未积分、未改checkpoint或生产输出。

{
  "C_threads1_vs_D_threads1": {
    "array_equal": true,
    "rhs_relative_l2": 0.0
  },
  "C_threads10_vs_D_threads10": {
    "array_equal": false,
    "rhs_relative_l2": 4.149274290569737e-15
  },
  "C_threads10_vs_C_threads1": {
    "array_equal": false,
    "rhs_relative_l2": 2.7696242176207197e-14
  },
  "D_threads10_vs_D_threads1": {
    "array_equal": false,
    "rhs_relative_l2": 2.75993149922352e-14
  }
}

单线程C/D RHS逐字节一致，重复调用一致。10线程重复调用和跨模型/进程差异约浮点舍入尺度，电势残差约5e-13。这确认多线程求解可提供微小数值扰动，尚不能证明历史14.94%结构差来自它；冻结密度与新模型未复现整个轨迹或求解器缓存历史。当前零FAC RHS未显示FAC截断造成可分辨物理差异。

后续新诊断和A/B运行须记录线程/runtime、源码SHA256及初值SHA256；已有结果不重写，不靠修改阈值消除差异。先定义可重复的数值对照，再解释成熟期差场；尚无Y改进结果。

### 2026-10-09 02:26 A组60 s控制实验启动

A100×401/dx2.5、250 km唯一周期、FAC4,4/44模态，独立初值三轨60 s，hash15f3e86753a082c729152f9ec6b78d688f49fd414ee5716b04b0fe9efd6cdfcd。MKL/OMP/OpenBLAS单线程，源码ZIP/SHA256及runtime/package/config溯源已保存。未改物理参数、旧hash、C/D源码或输出；同历史C/D比较须保留线程数值混杂。下一检查不早于02:56:17；60 s完成先终态审计/严格验收/图像QA/归档，再阶段续算；不重复启动。

### 2026-10-09 03:25 A组60 s验收/归档后续300 s

终态独立审计、严格0失败2警告及view_image QA通过，60 s/240步同hash、有限正密度、零宏步减半。archive_t60复制22文件SHA256一致，checkpoint逐字节一致。单线程自身checkpoint resume=True续300 s已启动，Li配置完整保留，只改EM终点，下一检查不早于03:55:56。


# A组300 s同周期网格对照（2026-10-09）

A300 s/1200步完成，hash保持，独立终态数组审计/有限正密度/零宏步减半/严格0失败2警告/view_image QA通过。archive_t300复制53文件，全部SHA256一致、checkpoint逐字节一致。

共同物理节点A与C每隔一列对齐，同ny401、周期250 km、FAC4,4。以C为分母EM横向结构差0.0094357%、密度差0.0007358%，200–300 s等时logRMS OLS增长率A3.257300198e-4 vsC3.259830925e-4 s^-1（−0.0776337%）。local ES结构差0.0174557%、column ES0.0130123%。早期误差小不证明成熟形态或渐近收敛；A单线程/C历史多线程混杂保留。

04:25:55从A自身300 s checkpoint同hash resume=True续3150 s已启动，仅改EM终点，Li保持、单线程设置保持；下一检查不早于04:55:55。完成后独立审计、严格验收、view_image QA及归档，和C同3150 s比较。B尚未启动，C/D不重跑，无5000自动延长。

### 2026-10-09 05:24同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint800.0 s/3200步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到817.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_A_MONITOR_2026-10-09_0524.json)。

### 2026-10-09 06:24同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint1325.0 s/5300步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到1343.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_A_MONITOR_2026-10-09_0624.json)。

### 2026-10-09 07:24同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint1850.0 s/7400步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到1869.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_A_MONITOR_2026-10-09_0724.json)。

### 2026-10-09 08:24同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint2375.0 s/9500步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到2394.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_A_MONITOR_2026-10-09_0824.json)。

### 2026-10-09 10:02同hash3150 s续算检查

命令行核查EPB进程数0；可信checkpoint3150.0 s/12600步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到3150.0 s，宏观减步累计0；最新ES子步[2, 2]。检查结果terminal_requires_review，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_A_MONITOR_2026-10-09_1002.json)。

### 2026-10-09 10:05 A3150完成与B60启动

A组100×401、250 km唯一周期、FAC(4,4)44模态三轨完成3150 s/12600步。checkpoint/final/3150 s快照密度逐数组一致，线变量与final一致；hash保持15f3e86753a082c729152f9ec6b78d688f49fd414ee5716b04b0fe9efd6cdfcd，数组有限、密度正值、零宏步减半。严格验收0失败4警告（三轨CFL接近上限及开放粒子通量账缺失）。未降低阈值。3150归档346文件全部SHA256一致，checkpoint逐字节一致，原输出全部保留。

已用view_image检查A/C同3150图件。A两ES轨出现宽分支，C有更细指状结构；两EM轨仍宽平滑、无清晰Y。同物理节点比较：EM横向结构L2差0.1290172%，密度L2差0.0332560%；共同300–3150 s、0.25 s等时logRMS插值OLS增长率A4.157578486889497e-4、C4.1610062311785606e-4 s^-1，相对差−0.0823778%。local ES结构差62.27717%，column ES33.17537%，纯EM差场43.39809%，几何差场69.55454%。增长率相近不能代替成熟期形态/差场收敛。A单线程与历史C多线程环境仍混杂，不能将所有差异归于网格或证明Y唯一原因。

未续算A5000。按已授权串行阶段流程，B组100×401、同周期250 km、FAC(8,8)152模态从独立初值启动60 s门禁；保持三轨、feedback1、matched、背景及时间步，单线程MKL/OMP/OpenBLAS、MKL_DYNAMIC FALSE。89个源码文件与A/B冻结SHA256清单一致，保存库/配置/线程运行溯源。B必须通过60/300/3150各阶段独立审计、严格验收、view_image QA和SHA256归档；恢复仅B自身同hash checkpoint resume=True，不移植C/D checkpoint。

采样功率能量不是精确RK-stage功，开放粒子漂移不是封闭守恒残差；模型尚缺平行质量/完整双半球metric/化学/风剪切/PRE/动态E电导，不作定量预测。

B启动2026-10-09T10:05:13.2037809+08:00；初始PID仅记录，下一检查不得早于2026-10-09T10:35:13.203780+08:00。

### 2026-10-09 10:39 B60完成与300续算

B100×401、250 km唯一周期、FAC(8,8)152模态三轨完成60 s/240步；独立数组检查确认checkpoint/final/快照密度一致、线变量与final一致，有限正密度、零宏步减半、hash保持。严格验收0失败2警告（EM CFL近上限与开放粒子通量账缺失）；view_image QA确认三轨早期平滑，图件正常，不作成熟期Y判断。

同网格、同单线程A/B早期比较：local ES和column ES密度逐字节一致；EM横向结构L2差0.0224005%。这支持当前零FAC对照的早期重复性，不能证明成熟期重复性、唯一Y原因或收敛。

archive_t60复制22文件并逐文件SHA256核验，checkpoint逐字节一致，保留原输出。已从B自身60 s checkpoint以同hash resume=True启动300 s唯一driver，仅修改EM runner终点，Li配置、单线程环境和三轨保持。300 s完成须独立终态审计/严格验收/view_image QA/归档门禁，通过后才续3150；不自动延长5000、不移植其他轨迹checkpoint。A/C/D不重跑。A/B与历史C/D线程混杂继续保留。

启动2026-10-09T10:39:03.2879051+08:00，下次检查不早于2026-10-09T11:09:03.287905+08:00。

### 2026-10-09 11:37同hash300 s续算检查

命令行核查EPB进程数1；可信checkpoint225.0 s/900步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到241.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；300 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_B_MONITOR_2026-10-09_1137.json)。

### 2026-10-09 12:37同hash300 s续算检查

命令行核查EPB进程数0；可信checkpoint300.0 s/1200步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到300.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果terminal_requires_review，异常[]。不重复启动；300 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_B_MONITOR_2026-10-09_1237.json)。

### 2026-10-09 12:39 B300完成与3150续算

B100×401、250 km唯一周期、FAC(8,8)152模态三轨完成300 s/1200步。独立检查checkpoint/final/快照密度一致、线变量与final一致，有限正密度、零宏步减半、hash保持。严格验收0失败2警告，view_image QA确认早期平滑三轨图件正常，无成熟期Y结论。archive_t300复制53文件全部SHA256一致、checkpoint逐字节一致，保留所有原输出。

A/B同网格同单线程：两ES密度逐字节一致；EM横向结构L2差0.1486987%，纯EM差场3.02901%。共同200–300 s、0.25 s等时logRMS插值OLS，gammaB3.2422139469299825e-4、A3.257300198434873e-4 s^-1，相对差−0.463152%。B/D同FAC同周期、共同节点EM结构差0.00935435%，gamma相对D差−0.0774122%；B单线程与D历史多线程仍混杂。早期一致不能证明成熟期/Y因果或严格收敛。

按授权从B自身300 s checkpoint同hash resume=True续3150 s，仅改EM runner终点；Li配置、三轨、反馈1、matched及单线程环境保持。3150终态仍须独立审计、严格验收、view_image QA、与A/D同3150比较及SHA256归档。不自动续5000，A/C/D不重跑，不移植checkpoint、不prepare/no-resume/手清尾部。

启动2026-10-09T12:38:57.9923809+08:00，下次检查不早于2026-10-09T13:08:57.992380+08:00。

### 2026-10-09 13:37同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint475.0 s/1900步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到482.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_B_MONITOR_2026-10-09_1337.json)。

### 2026-10-09 14:37同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint650.0 s/2600步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到669.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_B_MONITOR_2026-10-09_1437.json)。

### 2026-10-09 15:37同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint850.0 s/3400步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到855.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_B_MONITOR_2026-10-09_1537.json)。

### 2026-10-09 16:38同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint1025.0 s/4100步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到1043.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_B_MONITOR_2026-10-09_1638.json)。

### 2026-10-09 17:39同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint1225.0 s/4900步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到1230.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_B_MONITOR_2026-10-09_1739.json)。

### 2026-10-09 18:12同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint1325.0 s/5300步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到1332.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_B_MONITOR_2026-10-09_1812.json)。

### 2026-10-09 19:07同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint1500.0 s/6000步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到1501.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_B_MONITOR_2026-10-09_1907.json)。

### 2026-10-09 20:07同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint1675.0 s/6700步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到1686.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_B_MONITOR_2026-10-09_2007.json)。

### 2026-10-09 21:07同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint1775.0 s/7100步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到1798.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_B_MONITOR_2026-10-09_2107.json)。

### 2026-10-09 22:07同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint1825.0 s/7300步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到1835.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_B_MONITOR_2026-10-09_2207.json)。

### 2026-10-09 23:07同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint1850.0 s/7400步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到1872.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_B_MONITOR_2026-10-09_2307.json)。

### 2026-10-10 00:07同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint2050.0 s/8200步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到2053.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_B_MONITOR_2026-10-10_0007.json)。

### 2026-10-10 01:07同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint2225.0 s/8900步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到2236.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_B_MONITOR_2026-10-10_0107.json)。

### 2026-10-10 02:07同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint2400.0 s/9600步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到2420.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_B_MONITOR_2026-10-10_0207.json)。

### 2026-10-10 03:07同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint2600.0 s/10400步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到2603.0 s，宏观减步累计0；最新ES子步[1, 1]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_B_MONITOR_2026-10-10_0307.json)。

### 2026-10-10 04:07同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint2775.0 s/11100步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到2783.0 s，宏观减步累计0；最新ES子步[2, 2]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_B_MONITOR_2026-10-10_0407.json)。

### 2026-10-10 05:07同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint2950.0 s/11800步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到2953.0 s，宏观减步累计0；最新ES子步[2, 2]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_B_MONITOR_2026-10-10_0507.json)。

### 2026-10-10 06:07同hash3150 s续算检查

命令行核查EPB进程数1；可信checkpoint3100.0 s/12400步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到3124.0 s，宏观减步累计0；最新ES子步[2, 2]。检查结果running_stable，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_B_MONITOR_2026-10-10_0607.json)。

### 2026-10-10 07:07同hash3150 s续算检查

命令行核查EPB进程数0；可信checkpoint3150.0 s/12600步，schema3；hash核对、浮点有限、密度正值见JSON。诊断到3150.0 s，宏观减步累计0；最新ES子步[2, 2]。检查结果terminal_requires_review，异常[]。不重复启动；3150 s终态验收/视觉QA尚待核查。

[检查JSON](<ORIGINAL_CHECKOUT>/docs/EPB_CONTROL_B_MONITOR_2026-10-10_0707.json)。

### 2026-10-10 07:09 B3150完成与四组因素证据

B100×401、250 km唯一周期、FAC(8,8)152模态三轨完成3150 s/12600步。独立checkpoint/final/快照密度逐字节一致，线变量与final一致；hash保持，浮点有限、密度正值、零宏步减半。严格0失败4警告，B/A/D同3150图件view_image QA通过。archive_t3150复制346文件全部SHA256一致，checkpoint逐字节一致，原输出全部保留。实际Python进程为空，不重启，不延长B5000。

关键控制：同网格同单线程A/B两ES终态密度逐字节一致、共同300–3150 s增长率完全相同。B相对A的EM横向结构L2差2.8714586%，纯EM差场2.1219283%，几何差场零。共同0.25 s等时logRMS插值OLS，gammaB4.095824637125564e-4、A4.157578486889497e-4 s^-1，B相对A−1.4853321%。两EM仍宽而平滑，无清晰Y；在这组条件下FAC截断44增至152模态未产生Y。

B/D同FAC共同物理节点：EM结构差0.1136655%，gammaB相对D−0.0791139%。localES/columnES结构差62.2761457%/33.0449166%，纯EM差场42.9001564%；单线程A/B与历史C/D环境混杂仍保留，不能宣称严格网格或差场收敛。

四组均已有同3150秒成果。B−A受控FAC比较支持约1.49%增长率变化、约2.87%结构变化；历史D−C的EM增长率相对C−1.4885501%、结构差2.8856550%（D−C以C为范数分母，旧C−D以D为分母，方向不同不能直接替换旧数值）。两个网格上的EM截断响应相近，但历史C/D零FAC ES基线不一致，不能把历史纯EM差场全部归于FAC。共同100节点的D−C−B+A密度交互范数相对C为EM7.1094215e-5、columnES0.06576764；仅描述量，不是无混杂因果估计。两级因素证据不构成渐近收敛证明。

下一研究按原授权顺序，先文献与源码支持审查真实Eregion终端电导映射，再设计默认0的三轨共同ExB及长度/Alfven速度因素。后三项仍未实施，新的物理参数须向后兼容默认、显式入hash及物理/回归测试。保护已完成A/B/C/D和旧失败轨迹，不重复生产计算、不更改验收阈值。采样能量非精确RK-stage功；开放粒子漂移非封闭守恒；模型非定量预测，Y缺失原因尚未唯一确定。

### 2026-10-10T08:11:50.738633+08:00 E区终端映射审查

现有标量被动负载接口及hash保护已只读核查，89源码SHA一致，Bhash保持。matched 0.795774715459 S是无反射Alfven导纳，非实测E区电导；完整足点度量/两半球映射尚未建立。6个解析负载核查通过，无积分或新生产启动。下一实施默认0的三轨共同背景ExB及物理/回归测试。详见 EPB_EREGION_TERMINAL_MAPPING_REVIEW_2026-10-10.md 和 EPB_EREGION_TERMINAL_MAPPING_AUDIT_2026-10-10.json。

### 2026-10-10T09:13:55.187199+08:00 三轨共同背景ExB实现

已新增EMGrowthConfig及PhysicalFeedbackConfig参数background_exb_y_m_s，默认0，单位m/s，正值沿+y向上。localES、columnES、columnEM使用同一叠加后的竖直速度计算密度输运、速度诊断和CFL。柱轨道仍输运H*n并除H，局地轨道仍输运n。零漂移保留历史局地ES RHS和SSPRK快捷路径；非零漂移进入共同RHS积分，避免旧零FAC快捷路径绕过背景漂移。新增字段放在dataclass已有字段之后，保持已有位置参数顺序。

这是外加共同密度漂移敏感性接口，尚未把背景电场引起的电导电流及极化响应加入电流闭合，不能称完整PRE或自洽背景E区模型。背景漂移不进入FAC电势/线电压，未改变FAC符号、feedback_gain或远端负载。开放上下边界沿用原输运边界，因此常密度/常柱密度解析检验取模板不受边界影响的内部；不声称全域粒子守恒。

14项有针对性测试通过：默认零局地RHS/SSPRK逐字节回归，有限值验证，零/-零hash一致且非零hash不同，10/20/−10m/s方向与量纲及局地/柱解析输运，常柱密度内部零导数，周期x平移等变，CFL计入漂移并拒绝超限；0/10/20m/s的独立9×9微型三轨测试从0.02恢复到0.04s，密度有限正值且零减步，改变漂移拒绝checkpoint恢复。旧143整套和已完成生产轨迹未重跑。

A/B/C保存配置默认零hash逐一验证保持。89文件源码清单中仅em_growth.py与bidirectional_em.py发生预期变更；旧A/B清单及ZIP保留，新源码清单/ZIP独立保存。两个编辑前文件备份位于tmp/background_exb_backup_20261010；物理测试保存到实际worktree tests/test_background_exb.py。机器细节见同名AUDIT JSON。

本轮未启动生产。下一步审阅源码差异并准备新独立短阶段的0/10/20m/s计划：固定同网格、同FAC、负载、长度、VA及完整Li参数，单线程溯源；0基线优先引用现有归档，非零新hash与新目录，先短阶段独立数组审计/CFL/严格验收/图件QA，不直接启动长轨迹。新源码不能继续使用旧89文件清单作为当前源码一致声明，历史A/B仍以保存快照为准。Y缺失原因未确定。

### 2026-10-10T10:10:45.782397+08:00 ExB60s计划门禁

目的：在同B100×401、250km唯一周期、FAC(8,8)152模态、同matched负载、L=20000km、VA=1000km/s、同完整Li配置与单线程环境下，检验三轨共同外加密度漂移10/20m/s的早期稳定性。0m/s引用B的archive_t60，不重复生产基线。非零参数只表示外加密度输运，无背景电流极化闭合，不视为完整PRE；60s不提供成熟Y分叉证据。

初始全网格RHS均有限，columnES与EM密度RHS逐字节一致。10m/s初始localES/columnES/EM-density CFL为0.01860904/0.00515958/0.00515958；20m/s为0.02110904/0.00765958/0.00765958；线CFL均0.4。原门槛ES0.2500001/EM0.4000001不变。完整Li文件与B逐字节一致，SHA256为04e7db8fe3620d2c4895c7c4890024b6dad129fc4cf45d0f6270268082d799c9。

预检查发现新漂移hash整数10与浮点10.0会不同，已仅对该新参数的非零值做float规范化；默认0仍省略保持旧hash。14个测试再次通过，A/B/C历史默认零核查有效。早先实现AUDIT中的非零整数hash属于v1，应以本计划AUDIT浮点规范v2为准；旧文件/ZIP全部保留。新当前89源码hash清单与ZIP为tmp/background_exb_source_v2_20261010_sha256.json及.zip。

10m/s hash 2fc21197d5a2a30fc4e3f1bbf06b60951106956ec41d7b82d62e8bb9b933f527；20m/s hash fdafd006233807f6aa30f885b467753011e5fb260f7cfd71731bbcc5b3bc9112。各有独立完整配置、新目录与driver。使用resume=True从新目录初态启动，不移植任何checkpoint；若目录/旧checkpoint出现则初次启动门禁拒绝，不手工清理。

运行串行：先10m/s到60s，完成checkpoint/final/快照/线数组独立审计、有限正密度、零宏步减半、严格验收、图件view_image QA、SHA256归档后，再启动20m/s到60s。初始实际Python进程必须为空，按命令行核查；启动PID仅溯源。任何失败或异常保留现场并审查，不自动重启，不自动延长300/3150/5000。

### 2026-10-10T10:12:02.449746+08:00 ExB10m/s 60s启动

门禁通过后启动唯一driver，实际Python命令行匹配且stderr空，新hash 2fc21197d5a2a30fc4e3f1bbf06b60951106956ec41d7b82d62e8bb9b933f527，新目录，单线程resume=True。20m/s仍未启动，必须先10m/s终态审计/严格验收/QA/SHA归档；不自动延长。下一检查不早于 2026-10-10T10:42:02.449746+08:00。

### 2026-10-10T11:10:47.249601+08:00 ExB10m/s完成及20m/s串行启动

10m/s独立三轨60s/240步完成，hash2fc21197d5a2a30fc4e3f1bbf06b60951106956ec41d7b82d62e8bb9b933f527保持。独立checkpoint/final/snapshot密度逐字节一致，线变量与final一致，全部浮点有限、密度正值、零宏步减半。严格0失败2警告（线CFL临界与开放边界内容漂移诊断），view_image QA通过，图件/三轨曲线清楚且为平滑早期结构。archive_t60复制23文件SHA256一致，checkpoint逐字节一致；原输出保留。完成后实际Python为空，不重启10组。

同B零漂移archive_t60、同网格/单线程/完整Li/FAC8负载长度VA对照：localES/columnES/EM密度相对L2差0.6049031%/0.7664620%/0.7665101%，横向结构差0.6280238%/0.7609840%/0.7656892%。共同20–60s等时0.25s插值logRMS斜率EM为2.1888325103737276e-4 vs零漂移3.022244741147432e-4 s^-1，差−8.3341223e-5；columnES差−8.2648767e-5，localES差−1.8024020e-6。早期斜率受背景输运影响，不作为成熟增长率或Y因果证据。

纯EM差场相对零漂移差0.9507419%，几何差场94.2830139%；差场分母小且背景改变ES控制，不能把几何差场相对变化或EM斜率降低全归于FAC。外加密度漂移尚不包含背景电场电导电流/极化闭合；开放上下边界不构成封闭粒子守恒。

10组全部门禁通过后，串行启动唯一20m/s60s driver，hashfdafd006233807f6aa30f885b467753011e5fb260f7cfd71731bbcc5b3bc9112，独立新目录resume=True初态，实际命令行确认、stderr空。初PID23468只作溯源。仅目标60s，不自动延长300/3150/5000。后续完成同级审计/QA/归档及0/10/20汇总，再规划长度/VA与等效终端负载；不重复A/B/C/D生产轨迹。

下一检查不早于2026-10-10T11:40:47.249601+08:00。

### 2026-10-10T12:09:25.113262+08:00 0/10/20m/s60秒阶段汇总

10与20m/s各完成60s/240步，独立checkpoint/final/snapshot密度及线变量审计、有限正密度、零宏步减半、严格0失败2警告和view_image QA通过。各archive_t60为23文件SHA256一致，checkpoint逐字节一致，原输出保留；实际Python进程为空。0基线引用已有B archive_t60，没有重复生产或延长任一轨迹。两非零组源码同v2快照、同单线程、同网格FAC负载长度VA、Li完整逐字节一致。

| 外加+y漂移m/s | EM横向结构差（对0） | localES logRMS斜率s^-1 | columnES斜率s^-1 | EM斜率s^-1 |
|---:|---:|---:|---:|---:|
| 0 | 0 | −3.924311262e-4 | 4.058784303e-4 | 3.022244741e-4 |
| 10 | 0.7656892% | −3.942335282e-4 | 3.232296628e-4 | 2.188832510e-4 |
| 20 | 1.5277986% | −3.959861129e-4 | 2.405156210e-4 | 1.354714711e-4 |

共同20–60s、0.25s等时插值logRMS拟合。EM相对0的早期斜率差在10/20m/s为−8.3341223e-5/−1.6675300e-4；columnES为−8.2648767e-5/−1.6536281e-4。EM−columnES拟合斜率差分别为−1.0365395617e-4、−1.0434641175e-4、−1.0504414993e-4。两柱轨道的大部分早期斜率变化接近，支持共同柱输运敏感性描述，不是从差分斜率分离出的纯FAC因果量。

20m/s对0的localES/columnES/EM密度L2差1.2097615%/1.5293301%/1.5294262%；纯EM差场1.8706143%，几何差场188.14331%。10组纯EM0.9507419%、几何94.28301%。几何差场相对变化使用小差场分母，ES控制同时改变，不能读作整体密度大幅改变或全归于FAC。三组密度图仍为平滑早期形态；60s不能判断成熟Y。该接口仅外加密度漂移，没有背景电场电导电流/极化闭合，不称完整PRE。上下开放边界及柱权重保持原实现。

下一跟进先只读准备长度/VA与等效远端被动负载的因素门禁。既有参数已入hash，不新增重复参数。改变VA时matched同时改变SigmaA=1/(mu0 VA)；若要保持反射，控制SigmaP/SigmaA，若要研究绝对电导，固定S并把反射变化记为因素。改变L/VA须同时记录单程时间L/VA、线空间步长与CFL；不能因缩短L或增大VA而无记录地改变宏步或依赖减半。长度对照优先固定线空间步长并相应改变cells，VA对照优先共同满足最严格CFL的宏步，或明确数值混杂。参数取值须说明为敏感性假设，不作真实赤道E区标定。

本轮没有新生产启动，不自动恢复或延长10/20组300/3150/5000；A/B/C/D3150及旧失败原判/输出全部保护。下一参数门禁和有针对性传播/被动性核查通过后，才准备独立短阶段；Y缺失原因仍未唯一确定。


## 2026-10-10 13:10 长度/VA/负载只读门禁

7组配置账本（含B归档基线）hash互异、CFL通过；4项独立线传播/CFL/被动反射/匹配脉冲测试通过。89源码同v2快照，Bhash保持，无源码改动及EPB生产。VA2000在旧dt下CFL0.8不合当前0.4门槛；长线/VA500的80s往返未被60s覆盖。下一先核查候选柱权重/完整Li/初始RHS，再准备独立短阶段；不自动延长。详见EPB_LINE_FACTOR_GATE_REVIEW_2026-10-10.md及AUDIT。


## 2026-10-10 14:10 线因素初态预检

7配置dipole柱权重/初始三轨密度RHS与B逐字节一致，有限正密度、CFL/初始残差/配对通过；仅t0零反馈增益不证明成熟FAC。已准备独立L10000/16cells固定ds625km/dt.25/VA1000/matched零漂移60s配置与driver，未启动；完整Li保持、源同v2。详见EPB_LINE_FACTOR_PREFLIGHT_REVIEW_2026-10-10.md及AUDIT。下一实际命令行确认空及源/hash/单线程门禁后仅60s，完成独立审计/ES重复性/QA/SHA，禁止自动延长。


## 2026-10-10 15:10 独立短线60s启动

实际Python空门禁后启动唯一run_line_L10000_t60_20261010.py；启动确认命令行唯一、stderr空。L10000/16cells、ds625km、dt.25、VA1000、matched、FAC8、零漂移，完整Li与v2源SHA/hash门禁由driver核查。初PID24768仅溯源；后续按实际命令行/checkpoint。仅60s，完成独立数组/严格验收、B归档两ES重复性、view_imageQA/SHA后再决策，不自动延长。下一检查不早于15:40。


2026-10-10 16:08 短线常规检查：实际Python命令行唯一；checkpoint25s/100步，同hash、有限正密度、零宏步减半；36条诊断最大耦合残差4.9088e-13、配对6.8914e-16、EM CFL.4、stderr空，无异常。未到终态，不采用final/status替代进度。下次不早于16:40。


## 2026-10-10 17:12 L10000短线完成

60s240步同hash，独立终态有限正密度零减步、严格0失败2警告QA通过，23文件SHA归档cp字节一致。B两ES全程/终态一致，EM终态密度逐字节一致及20–60s斜率相同；仅matched/零初始线/60s范围，无成熟Y结论。原验收hash异常为零漂移规则遗漏，修复acceptance模块并6针对测试通过，仅重验收无重积分，原stderr保留。运行v2，当前验收源v3快照保存。下一只读特征机制/反射负载门禁，不直接生产。


## 2026-10-10 18:12 matched机制与反射计划

终态两matched线w_minus严格零，近端w_plus差2.7756e-17V非byteexact；无回传特征支持密度长度一致机制，但不推广全线/真实双半球。160cells孤立脉冲R±2/3主峰±.613625，反号尾瓣.000146144显式记录；边界被动性通过非全局能量证明。ratio.2/5新60s配置/driver已准备，初态/hash/CFL/v3SHA通过，未启动。下一32cells脉冲离散/被动功率门禁后串行ratio.2，审计QA/SHA后ratio5，不自动延长。见EPB_REFLECTION_T60_PLAN_REVIEW/AUDIT。


## 2026-10-10 19:10 反射比0.2独立60s启动

32cells/dt.25/CFL.4脉冲与28步被动边界通量核查通过，主峰±.318509反号尾瓣.0018511(0.581%)保留，脉冲欠分辨不称收敛/精确能量。实际Python空后唯一run_load_ratio0p2_t60_20261010.py启动，stderr空，初PID22896仅溯源。v3源/同hash/完整Li/单线程新目录resumeTrue仅60s，ratio5未启动。完成独立审计/两ES重复性/QA/SHA后才下一组，不自动延长。


2026-10-10 20:08 ratio0p2常规检查：实际Python命令行唯一，checkpoint25s/100步同hash、有限正密度零宏步減半。36诊断残差最大4.9088e-13/配对6.8914e-16，ES/EM原CFL门槛通过，stderr空、issues[]。未到60终态，ratio5仍未启动；下一不早于20:40。


2026-10-10 21:15 ratio0p2完成60s/240步，独立数组与实际图件QA通过，两ES与B全程一致；严格验收保留1失败2警告（conductance实验不满足matched_remote_boundary），未归档/未启动ratio5。原策略/阈值/源码/数组/hash保持，不重积分。详见EPB_LOAD_RATIO0P2_T60_EXCEPTION_REVIEW及AUDIT。下一只读设计独立反射负载验收策略与篡改/被动性测试；下一时间以handoff为准。


2026-10-11 conductance独立验收只读设计完成：原策略已有被动功率检查，唯一冲突为matched硬条件。显式profile/独立报告设计及12项内存故障拒绝原型通过；未接入完整验收，原1失败2警告及5输入SHA/89源码v3保持。ratio5未启动，ratio0p2未归档。下一实施显式profile默认回归及完整身份/阈值/故障门禁；详见EPB_CONDUCTANCE_ACCEPTANCE_DESIGN_REVIEW及AUDIT。


2026-10-13 conductance显式独立验收接口完成，20针对性检查通过，默认matched完整结果一致。ratio0p2新profile另存0失败2警告，原matched1失败2警告及五输入SHA保持。仅acceptance源码变化，新89源码v4快照；未重积分/归档，ratio5未启动。下一共同保留新旧报告SHA归档，再更新ratio5显式profile/v4启动门禁。详见IMPLEMENTATION_REVIEW/AUDIT。


2026-10-13 22:34 ratio0p2共25文件SHA归档通过、新旧判定共同保留；ratio5实际Python空门禁后唯一60s启动确认，stderr空，完整Li/v4源/显式profile门禁。无cp移植、不延长，下一检查时间以handoff为准。


2026-10-13 23:33 ratio5常规检查：actual命令行唯一，checkpoint25s/100步同hash、有限正密度零宏步减半；36诊断残差max4.9088e-13、配对6.8914e-16，原CFL通过EM.4，stderr空issues[]。未到60s，未使用final/status代进度；下一以handoff时间为准。

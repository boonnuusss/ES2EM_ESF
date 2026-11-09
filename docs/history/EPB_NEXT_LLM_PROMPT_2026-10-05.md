# 复制以下内容给接手 LLM

你接手 EPB_simulation 的 reduced-EM nonlinear EPB 研究。前一个代理频繁遇到额度限制；请优先利用现有成果，不要从头推导、实现或重跑已完成算例。

首先完整阅读：
1. <ORIGINAL_CHECKOUT>/docs/EPB_PROJECT_PROGRESS_2026-10-05.md
2. <ORIGINAL_CHECKOUT>/docs/EPB_HANDOFF_STATE_2026-10-05.json
3. <ORIGINAL_CHECKOUT>/docs/REDUCED_EM_EPB_T5000_2026-09-30.md

实际 reduced-EM 代码工作树是 <ORIGINAL_CHECKOUT>。<ORIGINAL_CHECKOUT> 是另一个 checkout，主要提供 .deps 和辅助脚本，不要混淆版本。C 盘工作树 dirty，很多成果未提交；不要 git reset/checkout 清理，不删除用户输出，不要创建重复计算进程。

截至 2026-10-05 已核查：
- 粗网格 101×201、FAC (4,4) 44 模态的三轨算例已到 5000 s，所有终态检查、后处理、严格验收、视觉 QA、validate-em 和 143 项回归已通过。禁止重跑已完成的 3150/5000 s。生产 hash 为 8547522688d8127afb575a35b8affb857cc97886b0b3aeb301073b4f4d95a02d。
- 5000 s column EM 仍为宽平滑耗空通道，泡顶约 488 km，无清晰 Li 式 Y 分叉；column ES 有明显分支。不能宣称完全解决。
- 加密 201×401 + FAC (8,8) 152 模态三轨已完成 60/300 s 验收，当前向 3150 s 的续算已中断。可信 checkpoint 为 schema 3、t=1050 s、step=4200、dt=0.25 s、零减步，所有浮点数组有限且密度为正。
- diagnostics 到1074 s、快照到1070 s，checkpoint 后尾部属于孤儿；status 的 running/t=300 s 是陈旧启动状态；final/report 也仍是300 s旧结果。当前无 python.exe，旧PID24236失效，但启动前必须再次按命令行检查。中断原因未知，不能仅归因于额度。
- 加密 hash 必须保持 f7aecc67b3ddaae0c9c6325d0ef6fee2b6084060db301c8c4df5d6b9b52ad055。
- Codex 30分钟定时跟进 epb 已暂停，避免你接手后抢跑。

第一优先级：检查当前源码和状态，在确认无活跃计算后，安全恢复现有加密 checkpoint 到3150 s。Li配置 configs/em_growth_li_201x401_alpha04_resolution.json；EM配置 tmp/em_growth_resolution_201x401_k8_t3150.json；输出 output/em_growth/resolution_201x401_k8_matched_dipole_chapman_three_track_20260930。绝对不要使用 --no-resume，不手动改hash或清理尾部；让runner自动reconcile。不要重新执行已完成的prepare脚本。

PowerShell 环境与命令：
```powershell
Set-Location '<ORIGINAL_CHECKOUT>'
$env:PYTHONPATH='<ORIGINAL_CHECKOUT>\.deps;src'
$env:PYPARDISO_MKL_RT='<ORIGINAL_CHECKOUT>\.deps\Library\bin\mkl_rt.3.dll'
$env:PATH='<ORIGINAL_CHECKOUT>\.deps\Library\bin;' + $env:PATH
$env:MKL_NUM_THREADS='10'
python '<ORIGINAL_CHECKOUT>/tmp/run_em_resolution_mature_20261002.py'
```
这个既有driver含续算、终态核对、后处理、严格验收。也可调用 python -m esf_sim run-em-growth 使用上述两份配置，但两种入口只能选择一种。每30分钟检查一次，严禁高频轮询。按checkpoint而非status判断进度；检查哈希、密度、残差、CFL、连续减步等异常。

完成3150 s后，用图像查看工具核对形态，比较粗/细网格相同物理时间，分别分析增长率、形态、FAC和能量。需要时归档细网格3150 s后续算5000 s，不重复粗网格算例。300 s 的细/粗 EM 横向结构差仅0.157%、增长率差0.414%，但这不证明晚期分叉收敛，也不能从两个分辨率宣称严格收敛。

始终保留 local ES / column ES / column EM。纯电磁效应用 column EM−column ES，几何效应用 column ES−local ES。旧失败根因已修复：固定电势R=-1短路反射；错误FAC体源1/ds；单一H混淆粒子列与电导列。现用matched R=0、物理表面FAC、独立H_n/H_nu并推进N=H_n*n。禁止翻转FAC符号、恢复fixed-voltage短路或减弱feedback作弊。

之后严格按顺序推进：同步密度/FAC模态加密研究→有文献依据的真实E-region终端电导扫描→默认0、三轨共同施加的背景上升ExB漂移0/10/20 m/s→场线长度/Alfvén速度敏感性。后三项尚未实施。任何新物理参数须向后兼容、进入配置哈希、有测试。

注意数值解释：终点诊断遗漏已修复，粗网格末点是明确标记的终态RHS重建样本，不是历史RK-stage峰值；能量预算为采样功率积分，非精确离散stage功；开放边界粒子列漂移不能当封闭守恒误差。步长改变后增长率样本权重不同，需说明拟合窗和时间采样方法。模型仍缺平行质量输运、完整双半球metric、化学、中性风/剪切、PRE及动态E层电导，不得称定量预测。

工作成果请持续写入进度文档，最后给出失败根因、修复方程/代码、数值与能量门禁、同时间形态和增长率、未解决物理限制，以及可点击图件/JSON/源数据路径。不要仅回复计划，也不要隐瞒未完成阶段。若没有本机文件访问能力，先向用户索取进度文档和必要代码/状态包，不能声称已运行或验收。

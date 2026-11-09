# 长度/VA/负载初始模型预检（2026-10-10）

7个配置（含B参考）构建及初始RHS检查通过。各组dipole_chapman_flux_tube重力/碰撞柱权重与B逐字节一致，初始localES/columnES/EM密度RHS分别与B逐字节一致。密度有限正，线RHS有限；ES宏步CFL最大0.0161090435，线CFL为0.4或VA500的0.2，原门槛不变。初始残差/功率配对缺陷为零。

这些检查发生在t=0、反馈ramp增益为零、零初始线状态，仅能确认初态配置、柱权重和初始输运一致，不能证明活跃FAC或成熟阶段一致。当前89源码与v2快照SHA一致，完整Li SHA保持04e7db8fe3620d2c4895c7c4890024b6dad129fc4cf45d0f6270268082d799c9。

已准备单个独立短线60秒方案：L10000km、16线单元、ds625km、VA1000km/s、dt0.25s、matched、FAC8,8、零背景漂移；单程10秒、往返20秒。Li原文件逐字节复制。hash为03edcda4cc46479b43373506c1d223992cf413469a9dfff5d9f2576057648823；物理hash与上一配置账本一致，终点改60秒不改变物理身份。

配置位于tmp/line_L10000_li_20261010.json及line_L10000_em_t60_20261010.json；driver为tmp/run_line_L10000_t60_20261010.py，已语法检查但未执行。新输出目录<ORIGINAL_CHECKOUT>/output/em_growth/line_L10000_x100_y401_k8_three_track_20261010尚未创建。对照只引用B archive_t60，不恢复旧轨迹。

下一启动门槛：实际Python命令行确认无生产运行；保持MKL/OMP/OpenBLAS单线程及MKL_DYNAMIC FALSE，核对v2源快照与配置hash，独立新目录resume=True且无checkpoint移植。只运行60秒，终态独立cp/final/snapshot密度与线数组核查、有限正密度、零宏步减半、严格0失败、实际view_image QA后SHA归档。必须核对两个ES全程/终态与B归档的重复性；偏离时保留原判并检查环境/求解器，不能把控制差异全归线长度。长度也改变总线容量，结果不称严格收敛或成熟Y结论。本轮未启动生产，不自动300/3150/5000。

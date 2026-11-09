# 32单元反射门禁与负载比0.2启动（2026-10-10）

32单元、ds625km、dt.25s、CFL.4孤立线脉冲7秒核查通过：matched返回波严格零，R±2/3得到相反主峰/带符号总和；28步采样的远端特征二次通量均满足被动非负吸收。反射主峰±.318508986V，反号尾瓣±.001851097V（主峰约0.58118%），全部有限。该脉冲宽度仅1.28单元，欠分辨；形状/幅度不作收敛或定量物理证据，数值尾瓣保留、未裁剪、未改生产验收。二次通量不是完整耦合系统精确RK能量审计。

门禁通过且实际Python为空后，串行启动唯一负载比0.2仅60秒：R=+2/3、S=.15915494309189537S，保持B100×401/唯一250km周期/FAC8,8/L20000/32cells/VA1000/dt.25/零漂移/完整Li。使用v3源快照，driver核查源SHA/hash/单线程/完整Li。新目录resume=True无checkpoint移植；启动确认实际命令行唯一、stderr空。初PID22896仅溯源，后续实际命令行为准。

driver tmp/run_load_ratio0p2_t60_20261010.py；launch/stdout/stderr前缀tmp/load_ratio0p2_t60_20261010；hash e33ebcff1629897ef7de6a09d33f00aaa9c2cf3253da1d356127c688e2f6bb63。输出<ORIGINAL_CHECKOUT>/output/em_growth/load_ratio0p2_x100_y401_k8_three_track_20261010。只运行60秒；ratio5准备但未启动。

下一不早于19:40检查，实际命令行导出数组到tmp/load_ratio0p2_t60_processes_20261010.json，然后运行monitor_load_ratio0p2_t60_20261010.py，checkpoint判断进度。终态独立数组/零宏步减半/有限正密度/严格0失败、与B两ES全程及终态重复性、返回波/远端功率诊断核查、实际view_image QA及SHA归档后，才可串行ratio5仅60秒。R改变的是等效被动终端，不是FAC符号或feedback增益；不是已标定E区。禁止自动延长、重复启动及修改旧轨迹。

# 背景ExB独立60秒阶段计划（2026-10-10）

目的：在同B100×401、250km唯一周期、FAC(8,8)152模态、同matched负载、L=20000km、VA=1000km/s、同完整Li配置与单线程环境下，检验三轨共同外加密度漂移10/20m/s的早期稳定性。0m/s引用B的archive_t60，不重复生产基线。非零参数只表示外加密度输运，无背景电流极化闭合，不视为完整PRE；60s不提供成熟Y分叉证据。

初始全网格RHS均有限，columnES与EM密度RHS逐字节一致。10m/s初始localES/columnES/EM-density CFL为0.01860904/0.00515958/0.00515958；20m/s为0.02110904/0.00765958/0.00765958；线CFL均0.4。原门槛ES0.2500001/EM0.4000001不变。完整Li文件与B逐字节一致，SHA256为04e7db8fe3620d2c4895c7c4890024b6dad129fc4cf45d0f6270268082d799c9。

预检查发现新漂移hash整数10与浮点10.0会不同，已仅对该新参数的非零值做float规范化；默认0仍省略保持旧hash。14个测试再次通过，A/B/C历史默认零核查有效。早先实现AUDIT中的非零整数hash属于v1，应以本计划AUDIT浮点规范v2为准；旧文件/ZIP全部保留。新当前89源码hash清单与ZIP为tmp/background_exb_source_v2_20261010_sha256.json及.zip。

10m/s hash 2fc21197d5a2a30fc4e3f1bbf06b60951106956ec41d7b82d62e8bb9b933f527；20m/s hash fdafd006233807f6aa30f885b467753011e5fb260f7cfd71731bbcc5b3bc9112。各有独立完整配置、新目录与driver。使用resume=True从新目录初态启动，不移植任何checkpoint；若目录/旧checkpoint出现则初次启动门禁拒绝，不手工清理。

运行串行：先10m/s到60s，完成checkpoint/final/快照/线数组独立审计、有限正密度、零宏步减半、严格验收、图件view_image QA、SHA256归档后，再启动20m/s到60s。初始实际Python进程必须为空，按命令行核查；启动PID仅溯源。任何失败或异常保留现场并审查，不自动重启，不自动延长300/3150/5000。

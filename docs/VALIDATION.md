# 本次迁移验证（2026-11-09）

- 在独立发布目录运行9×9三轨到0.04 s：终态有限、密度正，checkpoint/final/末快照密度和checkpoint/final线状态一致。见 `provenance/SMOKE_TERMINAL_AUDIT.json`。
- `python -m esf_sim validate-em --output output/migration_validation`：九阶段全部通过。见 `provenance/EM_VALIDATION.json`。
- reduced-EM、验收、后处理、周期布局、三轨配置、极限和列几何九个测试文件：79 passed (14.76 s)。
- 新迁移终态审计、背景漂移、验收哈希三个测试文件：24 passed (4.02 s)。合计103项回归。
- 9个案例独立子进程加载配置及指定源码，均与保留的历史受保护哈希匹配；load5只与其启动记录哈希匹配，没有历史完成证据。见 `provenance/CASE_CONFIG_CHECKS.json`。

没有重新执行60/300/3150/5000 s生产模拟，没有复验完整历史数据，没有在另一台电脑或全新虚拟环境验证pip安装。当前验证借助旧机既有依赖目录，但它不包含在发布内容中。新电脑必须先安装，再执行冒烟/九阶段/回归和选定案例。

可用 `python -m pip install -e ".[test]" -c constraints-validated.txt` 尝试匹配已记录版本；这只是版本约束而非包含所有传递依赖与分发文件哈希的跨平台锁文件。平台不支持时不要静默替换求解器后声称字节级复现。

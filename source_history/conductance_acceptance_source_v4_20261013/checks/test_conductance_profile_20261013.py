from pathlib import Path
import importlib.util,json,copy,shutil,tempfile,hashlib
from esf_sim import em_growth_acceptance as m
b=Path.cwd(); root=Path('C:/Users/12928/.codex/worktrees/865f/EPB_simulation/output/em_growth');run=root/'load_ratio0p2_x100_y401_k8_three_track_20261010';profile='equivalent_passive_conductance_t60_v1'
sp=importlib.util.spec_from_file_location('old_accept',b/'tmp/acceptance_before_conductance_20261013.py');old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
checks=[]
for name,r in [('conductance_default',run),('matched_B',root/'matched_period_x100_y401_k8_three_track_20261008/archive_t60')]:
    assert m.evaluate_run(r)==old.evaluate_run(r);checks.append(name)
try:m.evaluate_run(run,profile='invalid')
except ValueError:checks.append('unknown_profile_rejected')
else:raise AssertionError()
a=m.evaluate_run(run,profile=profile);assert a['pass'] and len(a['warnings'])==2;checks.append('separate_conductance_pass')
for out in [None,run/'acceptance_t60.json']:
    try:m.write_acceptance_report(run,out,profile=profile)
    except ValueError:checks.append('protect_original_output_'+str(out is None))
    else:raise AssertionError()
report=json.loads((run/'em_growth_report.json').read_text());rows=[json.loads(x) for x in (run/'diagnostics.jsonl').read_text().splitlines()]
for field,value in [('right_boundary_conductance_s',-1),('right_boundary_conductance_s',float('nan')),('alfven_speed_km_s',0),('right_boundary_voltage_v',1),('right_boundary_model','fixed_voltage')]:
    q=copy.deepcopy(report);q['em_config'][field]=value;g=m._GateBook();m._conductance_gates(g,q['em_config'],q['scope'],rows,'60',60,60);assert any(x['status']=='fail' for x in g.checks);checks.append('reject_'+field+'_'+str(value))
for loc,key,value in [('scope','remote_boundary_reflection_coefficient',0),('scope','remote_boundary_conductance_s',1),('row','remote_boundary_reflection_coefficient',float('nan')),('row','remote_boundary_power_modal_w',float('nan'))]:
    q=copy.deepcopy(report);d=copy.deepcopy(rows);(q['scope'] if loc=='scope' else d[-1])[key]=value;g=m._GateBook();m._conductance_gates(g,q['em_config'],q['scope'],d,'60',60,60);assert any(x['status']=='fail' for x in g.checks);checks.append('reject_'+loc+'_'+key)
with tempfile.TemporaryDirectory(dir=b/'tmp',prefix='conductance_test_') as td:
    t=Path(td)
    for f in ['em_growth_report.json','diagnostics.jsonl','final_states.npz','checkpoint.npz','status.json']:shutil.copy2(run/f,t/f)
    for key,value,expected in [('remote_boundary_power_modal_w',-1,'passive_remote_absorption'),('coupled_cfl',0.5,'coupled_cfl'),('coupling_relative_residual',1e-4,'residual')]:
        d=copy.deepcopy(rows);d[-1][key]=value;(t/'diagnostics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in d));res=m.evaluate_run(t,profile=profile);assert not res['pass'],key;checks.append('shared_gate_'+key)
    (t/'diagnostics.jsonl').write_bytes((run/'diagnostics.jsonl').read_bytes())
    q=copy.deepcopy(report);q['em_config']['right_boundary_conductance_s']=1;(t/'em_growth_report.json').write_text(json.dumps(q))
    try:m.evaluate_run(t,profile=profile)
    except ValueError:checks.append('physical_hash_tamper_rejected')
    else:raise AssertionError()
    (t/'em_growth_report.json').write_bytes((run/'em_growth_report.json').read_bytes());q=json.loads((t/'status.json').read_text());q['configuration_hash']='bad';(t/'status.json').write_text(json.dumps(q))
    try:m.evaluate_run(t,profile=profile)
    except ValueError:checks.append('cross_file_identity_rejected')
    else:raise AssertionError()
result=dict(checks=checks,count=len(checks),passed=True,original_verdict_preserved=not m.evaluate_run(run)['pass'],independent_verdict=a['pass'],production_rerun=False)
(b/'tmp/conductance_profile_tests_20261013.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

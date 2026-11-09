"""Explicit, data-free entry point for the catalogued reduced-EM cases."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]


def terminal_audit(directory, expected_time, expected_hash):
    import numpy as np
    with np.load(directory/'checkpoint.npz', allow_pickle=False) as cp, np.load(directory/'final_states.npz', allow_pickle=False) as final:
        if float(cp['time_s']) != expected_time or float(final['time_s']) != expected_time:
            raise ValueError('Terminal time mismatch')
        if str(cp['configuration_hash']) != expected_hash:
            raise ValueError('Checkpoint configuration hash mismatch')
        for archive in (cp, final):
            for key in archive.files:
                if np.issubdtype(archive[key].dtype, np.floating) and not np.isfinite(archive[key]).all():
                    raise ValueError('Non-finite state: '+key)
        for key in ('baseline_density_m3','column_control_density_m3','coupled_density_m3'):
            if not (cp[key] > 0).all():
                raise ValueError('Non-positive density: '+key)
            np.testing.assert_array_equal(cp[key], final[key])
        for key in ('coupled_w_plus_v','coupled_w_minus_v'):
            np.testing.assert_array_equal(cp[key], final[key])
        # Verify the actual terminal snapshot, never assume a frame index.
        snapshots = sorted((directory/'snapshots').glob('snapshot_*.npz'))
        terminal = None
        for path in reversed(snapshots):
            with np.load(path, allow_pickle=False) as snap:
                if float(snap['time_s']) == expected_time:
                    for key in ('baseline_density_m3','column_control_density_m3','coupled_density_m3'):
                        np.testing.assert_array_equal(cp[key], snap[key])
                    if str(snap['configuration_hash']) != expected_hash:
                        raise ValueError('Snapshot hash mismatch')
                    terminal = path.name
                    break
        if terminal is None:
            raise ValueError('Terminal snapshot missing')
    return {'passed':True,'time_s':expected_time,'configuration_hash':expected_hash,'terminal_snapshot':terminal,'finite_positive':True,'terminal_arrays_exact':True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('list','check','run'))
    parser.add_argument('case', nargs='?')
    parser.add_argument('--stage', type=int, default=60)
    parser.add_argument('--plan', action='store_true', help='Print config/hash without running')
    parser.add_argument('--resume', action='store_true', help='Resume this stage in its own output directory')
    args = parser.parse_args(argv)
    catalog = json.loads((ROOT/'cases.json').read_text(encoding='utf-8'))['cases']
    if args.action == 'list':
        for key, case in catalog.items():
            print(f"{key:10} grid={case['grid_points']} FAC={case['fac_cutoff']} stages={case['stages_s']}")
        return 0
    if args.case not in catalog and args.case != 'smoke':
        parser.error('Select a case from list, or smoke')
    smoke = args.case == 'smoke'
    case = catalog['B'] if smoke else catalog[args.case]
    if not smoke and args.stage not in case['stages_s']:
        parser.error('Stage not catalogued for this case')
    os.chdir(ROOT)
    # Historical snapshots are deliberately preferred to the editable package.
    source = ROOT/('src' if smoke else case['source_directory'])
    sys.path.insert(0, str(source))
    for name in ('MKL_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['MKL_DYNAMIC'] = 'FALSE'
    os.environ['MPLBACKEND'] = 'Agg'
    from esf_sim.config import SimulationConfig
    from esf_sim.em_growth import EMGrowthConfig, _configuration_hash, run_em_growth_experiment
    li = SimulationConfig.from_json(ROOT/case['li'])
    em = EMGrowthConfig.from_json(ROOT/case['em'])
    if smoke:
        li = replace(li,nx=9,ny=9,x_max_km=8,y_max_km=258,dx_km=1,dy_km=1,
                     perturbation_wavelength_km=9,dt_s=0.01,t_end_s=0.04)
        em = replace(em,max_x_wavenumber=1,max_y_wavenumber=0,field_line_length_km=8000,
                     field_line_cells=8,alfven_speed_km_s=800,line_spatial_order=1,use_pardiso=False,
                     dt_s=0.01,minimum_dt_s=0.001,t_end_s=0.04,growth_fit_start_s=0,
                     snapshot_interval_s=0.02,checkpoint_interval_s=0.02,diagnostics_interval_steps=1,
                     periodic_x_layout='unique_periodic_cells')
    target = em.t_end_s if smoke else float(args.stage)
    # Keep Li JSON unchanged: the entire Li dict is protected by the historical hash.
    em = replace(em,t_end_s=target,output_dir=f'output/reproduction/{args.case}_t{target:g}')
    li.validate()
    fingerprint = _configuration_hash(li,em)
    if not smoke and case.get('expected_configuration_hash') and fingerprint != case['expected_configuration_hash']:
        raise ValueError('Case configuration no longer matches its protected historical hash')
    if not smoke:
        from esf_sim.em_growth_acceptance import write_acceptance_report
        from esf_sim.em_growth_postprocess import postprocess_em_growth
        acceptance_kwargs = {} if case['acceptance_profile'] == 'matched' else {'profile':case['acceptance_profile']}
        inspect.signature(write_acceptance_report).bind(Path(em.output_dir),Path(em.output_dir)/'reproduction_acceptance.json',**acceptance_kwargs)
        inspect.signature(postprocess_em_growth).bind(Path(em.output_dir),output_directory=Path(em.output_dir)/'postprocess')
    packages = {}
    for name in ('numpy','scipy','matplotlib','pypardiso','mkl','pytest'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    provenance = {'case':args.case,'target_time_s':target,'configuration_hash':fingerprint,
                  'source_directory':source.relative_to(ROOT).as_posix(),
                  'python':sys.version,'platform':platform.platform(),'packages':packages,
                  'thread_environment':{k:os.environ.get(k) for k in ('MKL_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_DYNAMIC')},
                  'li_config':li.to_dict(),'em_config':em.to_dict(),
                  'source_sha256':{p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.rglob('*.py'))}}
    print(json.dumps({k:v for k,v in provenance.items() if k not in ('source_sha256','li_config','em_config')},indent=2))
    if args.action == 'check' or args.plan:
        return 0
    output = ROOT/em.output_dir
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f'{output} contains files; inspect then use --resume')
    output.mkdir(parents=True,exist_ok=True)
    lock = output/'.reproduction.lock'
    # An interrupted lock must be inspected by the user before manual removal.
    with lock.open('x',encoding='utf-8') as handle:
        handle.write(str(os.getpid()))
    try:
        (output/'reproduction_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
        run_em_growth_experiment(li,em,resume=args.resume)
        audit = terminal_audit(output,target,fingerprint)
        (output/'terminal_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
        if smoke:
            print('Smoke terminal audit passed; this is not a scientific-stage acceptance.')
            return 0
        from esf_sim.em_growth_postprocess import postprocess_em_growth
        from esf_sim.em_growth_acceptance import write_acceptance_report
        postprocess_em_growth(output,output_directory=output/'postprocess')
        if case['acceptance_profile'] == 'matched':
            # Historical matched snapshots predate the optional profile argument.
            report = write_acceptance_report(output,output/'reproduction_acceptance.json')
        else:
            report = write_acceptance_report(output,output/'reproduction_acceptance.json',profile=case['acceptance_profile'])
        print(json.dumps({'pass':report['pass'],'failures':report['failures'],'warnings':report['warnings']},indent=2))
        return 0 if report['pass'] else 1
    finally:
        lock.unlink()


if __name__ == '__main__':
    raise SystemExit(main())

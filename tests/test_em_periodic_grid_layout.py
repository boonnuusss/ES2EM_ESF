from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from esf_sim.config import SimulationConfig
from esf_sim.em_growth import EMGrowthConfig, _configuration_hash, _load_checkpoint, _build_models, run_em_growth_experiment
from esf_sim.em_growth_acceptance import _grid_metadata, _unique_x
from esf_sim.em_growth_postprocess import _coordinates
from esf_sim.numerics import advection_rhs

ROOT=Path('C:/Users/12928/.codex/worktrees/865f/EPB_simulation')

def tiny():
 li=SimulationConfig(nx=8,ny=9,x_max_km=7,dx_km=1,y_max_km=258,dy_km=1,perturbation_wavelength_km=8,potential_boundary_y='equal_extrapolation',potential_operator='mimetic_harmonic_log_rhs',dt_s=.01)
 em=EMGrowthConfig(periodic_x_layout='unique_periodic_cells',closure='mimetic',use_pardiso=False,field_line_cells=8,max_x_wavenumber=1,max_y_wavenumber=1,dt_s=.01,minimum_dt_s=.0025,growth_fit_start_s=0,t_end_s=.02,checkpoint_interval_s=.01,snapshot_interval_s=.01,initial_line_state='zero',right_boundary_model='matched',comparison_tracks='local_column_em',fac_source_model='symmetric_flux_tube',parallel_averaging_profile='dipole_chapman_flux_tube',dipole_quadrature_order=16)
 return li,em

@pytest.mark.parametrize('li_file,em_file,expected',[
 ('configs/em_growth_li_201x401_alpha04_resolution.json','tmp/em_growth_resolution_201x401_k8_t3150.json','f7aecc67b3ddaae0c9c6325d0ef6fee2b6084060db301c8c4df5d6b9b52ad055'),
 ('configs/em_growth_li_101x201_alpha04.json','tmp/em_growth_three_track_t5000.json','8547522688d8127afb575a35b8affb857cc97886b0b3aeb301073b4f4d95a02d')])
def test_historical_production_hashes_unchanged(li_file,em_file,expected):
 assert _configuration_hash(SimulationConfig.from_json(ROOT/li_file),EMGrowthConfig.from_json(ROOT/em_file))==expected

def test_new_layout_hash_rejects_legacy_checkpoint():
 li=SimulationConfig.from_json(ROOT/'configs/em_growth_li_201x401_alpha04_resolution.json')
 em=EMGrowthConfig.from_json(ROOT/'tmp/em_growth_resolution_201x401_k8_t3150.json')
 new=replace(em,periodic_x_layout='unique_periodic_cells')
 assert _configuration_hash(li,new)!=_configuration_hash(li,em)
 cp=ROOT/em.output_dir/'checkpoint.npz'
 with pytest.raises(ValueError,match='hash mismatch'):
  _load_checkpoint(cp,_configuration_hash(li,new),comparison_tracks='local_column_em')

def test_reject_duplicate_node_input_and_unknown_layout():
 li,em=tiny()
 with pytest.raises(ValueError,match='integer seed wavelengths'):
  _build_models(replace(li,nx=9,x_max_km=8),em)
 with pytest.raises(ValueError,match='periodic_x_layout'):replace(em,periodic_x_layout='unknown')

def test_unique_ring_translation_and_no_false_endpoint_removal():
 li,em=tiny(); report={'li_config':li.to_dict(),'em_config':em.to_dict()}
 q=np.tile(np.array([2,3,5,4,2,6,8,9.],float),(9,1)); u=np.ones_like(q); v=np.zeros_like(q)
 rhs=lambda a:advection_rhs(a,u,v,dx=1,dy=1,scheme='pdm7',pdm_alpha=.4,periodic_y=False)
 np.testing.assert_allclose(rhs(np.roll(q,1,axis=1)),np.roll(rhs(q),1,axis=1),rtol=0,atol=1e-14)
 grid=_grid_metadata(report,q.shape)
 assert not grid['periodic_endpoint_duplicate_removed'] and grid['nx_unique']==8 and grid['computational_period_x_km']==8
 np.testing.assert_array_equal(_unique_x(q,grid),q)
 x,y=_coordinates(report,q.shape); np.testing.assert_array_equal(x,np.arange(8)-4)

def test_three_track_tiny_run_and_resume_unique_grid(tmp_path):
 li,em=tiny(); em=replace(em,output_dir=str(tmp_path/'unique_ring'))
 report=run_em_growth_experiment(li,em,resume=True)
 assert report['completed_time_s']==.02 and report['em_config']['periodic_x_layout']=='unique_periodic_cells'
 _,baseline,coupled,basis,line=_build_models(li,em)
 assert basis.nx==baseline.config.nx==coupled.config.nx==8
 with np.load(Path(em.output_dir)/'checkpoint.npz',allow_pickle=False) as cp:
  arrays={k:cp[k].copy() for k in ('baseline_density_m3','column_control_density_m3','coupled_density_m3')}
  assert all(a.shape==(9,8) and np.isfinite(a).all() and (a>0).all() for a in arrays.values())
 assert run_em_growth_experiment(li,em,resume=True)['configuration_hash']==report['configuration_hash']
 with np.load(Path(em.output_dir)/'checkpoint.npz',allow_pickle=False) as cp:
  for k,a in arrays.items():np.testing.assert_array_equal(cp[k],a)


"""Physical checks for imposed common density drift, without production runs."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
from esf_sim.em_growth import EMGrowthConfig, _build_models, _configuration_hash
from esf_sim.bidirectional_em import DimensionalEMState, PhysicalFeedbackConfig
from esf_sim.modal_alfven import ModalAlfvenState
from test_em_growth import _li_config, _three_track_config

def models(tmp_path, drift):
    em=replace(_three_track_config(tmp_path/'unused',t_end_s=.04),background_exb_y_m_s=drift)
    return _build_models(replace(_li_config(),ny=33,y_max_km=282.0),em)

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')])
def test_drift_finite(bad):
    with pytest.raises(ValueError): EMGrowthConfig(background_exb_y_m_s=bad)
    with pytest.raises(ValueError): PhysicalFeedbackConfig(background_exb_y_m_s=bad)

def test_zero_hash_and_nonzero_protection(tmp_path):
    li=_li_config(); em=_three_track_config(tmp_path/'unused',t_end_s=.04)
    assert _configuration_hash(li,em)==_configuration_hash(li,replace(em,background_exb_y_m_s=-0.0))
    hashes={_configuration_hash(li,replace(em,background_exb_y_m_s=v)) for v in [0,10,20,-10]}
    assert len(hashes)==4
    assert _configuration_hash(li,replace(em,background_exb_y_m_s=10)) == _configuration_hash(li,replace(em,background_exb_y_m_s=10.0))

def test_zero_local_rhs_byte_exact(tmp_path):
    li,base,coupled,basis,line=models(tmp_path,0)
    n=np.ones((li.ny,li.nx))*1e11
    n[:,2]*=1.01
    expected,diag=base.legacy_li_model.evaluate_rhs(n)
    got,d=base.evaluate_rhs(0,DimensionalEMState(n,ModalAlfvenState(np.zeros((line.cells,line.mode_count)),np.zeros((line.cells,line.mode_count)))))
    assert expected.tobytes()==got.density_m3.tobytes()
    assert d.combined_cfl_rate_s_inv==diag.cfl_rate

@pytest.mark.parametrize('drift',[10.,20.,-10.])
def test_common_upward_transport_and_cfl(tmp_path,monkeypatch,drift):
    li,base,coupled,basis,line=models(tmp_path,drift)
    # Isolate imposed drift from polarization/FAC without altering its transport paths.
    monkeypatch.setattr(base.legacy_li_model,'velocity_from_potential',lambda p:(np.zeros_like(p),np.zeros_like(p)))
    y=np.arange(li.ny)[:,None]*li.dy_m
    n=np.broadcast_to(1e11+1e6*y,(li.ny,li.nx)).copy()
    zero=ModalAlfvenState(np.zeros((line.cells,line.mode_count)),np.zeros((line.cells,line.mode_count)))
    rhs,diag=base.evaluate_rhs(0,DimensionalEMState(n,zero))
    np.testing.assert_allclose(rhs.density_m3[4:-4],-drift*1e6,rtol=1e-10)
    assert diag.electrostatic.maximum_speed_y==abs(drift)
    assert diag.electrostatic.cfl_rate==abs(drift)/li.dy_m
    # Column transport is H*n; test a linear column and constant column separately.
    h=coupled.gravity_column_weight_field_m
    column_n=n/h
    col,col_diag=coupled.evaluate_column_es_rhs(column_n)
    em_rhs,em_diag=coupled.evaluate_rhs(0,DimensionalEMState(column_n,zero))
    np.testing.assert_allclose((h*col)[4:-4],-drift*1e6,rtol=1e-10)
    np.testing.assert_array_equal(col,em_rhs.density_m3)
    assert col_diag.cfl_rate==em_diag.electrostatic.cfl_rate==abs(drift)/li.dy_m
    constant,_=coupled.evaluate_column_es_rhs(np.ones_like(n)*1e11/h)
    np.testing.assert_allclose((h*constant)[4:-4],0,atol=1e-2)

def test_periodic_x_translation(tmp_path):
    li,base,coupled,basis,line=models(tmp_path,20)
    n=np.ones((li.ny,li.nx))*1e11
    n[:,2:4]*=1.01
    a,_=coupled.evaluate_column_es_rhs(n)
    b,_=coupled.evaluate_column_es_rhs(np.roll(n,2,axis=1))
    np.testing.assert_allclose(np.roll(a,2,axis=1),b,rtol=1e-7,atol=1e-2)

def test_nonzero_local_step_advects_and_rejects_cfl(tmp_path,monkeypatch):
    from esf_sim._model_impl import CFLViolation
    li,base,coupled,basis,line=models(tmp_path,20)
    monkeypatch.setattr(base.legacy_li_model,'velocity_from_potential',lambda p:(np.zeros_like(p),np.zeros_like(p)))
    y=np.arange(li.ny)[:,None]*li.dy_m
    n=np.broadcast_to(1e11+1e6*y,(li.ny,li.nx)).copy()
    state=DimensionalEMState(n,ModalAlfvenState(np.zeros((line.cells,line.mode_count)),np.zeros((line.cells,line.mode_count))))
    got,diag=base.ssprk3_step(0,state,.01)
    np.testing.assert_allclose(got.density_m3[12:-12],n[12:-12]-20*1e6*.01,rtol=1e-12)
    with pytest.raises(CFLViolation): base.ssprk3_step(0,state,20)


@pytest.mark.parametrize('drift',[0.,10.,20.])
def test_tiny_three_track_resume_and_physics_hash(tmp_path,drift):
    from esf_sim.em_growth import run_em_growth_experiment
    output=tmp_path/'tiny_new_test'
    li=_li_config()
    cfg=replace(_three_track_config(output,t_end_s=.02),background_exb_y_m_s=drift)
    run_em_growth_experiment(li,cfg,resume=False)
    report=run_em_growth_experiment(li,replace(cfg,t_end_s=.04),resume=True)
    assert report['completed_time_s']==.04
    assert report['scope']['background_exb_y_m_s']==drift
    with np.load(output/'checkpoint.npz',allow_pickle=False) as cp:
        assert int(cp['dt_reductions'])==0
        for key in ['baseline_density_m3','column_control_density_m3','coupled_density_m3']:
            assert np.isfinite(cp[key]).all() and (cp[key]>0).all()
    with pytest.raises(ValueError,match='configuration hash mismatch'):
        run_em_growth_experiment(li,replace(cfg,t_end_s=.04,background_exb_y_m_s=drift+1),resume=True)

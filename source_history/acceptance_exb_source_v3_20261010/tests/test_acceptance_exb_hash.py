from dataclasses import replace
from copy import deepcopy
import pytest
from esf_sim.config import SimulationConfig
from esf_sim.em_growth import EMGrowthConfig, _configuration_hash
from esf_sim.em_growth_acceptance import _configuration_hash_from_report

@pytest.mark.parametrize("drift", [0.0, -0.0, 10, 10.0, 20.0])
def test_report_hash_round_trip_and_physical_tamper_detection(drift):
    li=SimulationConfig()
    em=EMGrowthConfig(background_exb_y_m_s=drift,periodic_x_layout="unique_periodic_cells",right_boundary_model="matched")
    report={"li_config":li.to_dict(),"em_config":em.to_dict()}
    expected=_configuration_hash(li,em)
    assert _configuration_hash_from_report(report)==expected
    edited=deepcopy(report);edited["em_config"]["background_exb_y_m_s"]=30.0
    assert _configuration_hash_from_report(edited)!=expected

def test_historical_report_without_drift_field_keeps_identity():
    li=SimulationConfig();em=EMGrowthConfig(periodic_x_layout="unique_periodic_cells",right_boundary_model="matched")
    report={"li_config":li.to_dict(),"em_config":em.to_dict()}
    report["em_config"].pop("background_exb_y_m_s")
    assert _configuration_hash_from_report(report)==_configuration_hash(li,em)

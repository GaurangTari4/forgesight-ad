import numpy as np
from importlib.util import module_from_spec,spec_from_file_location
from pathlib import Path
import pytest
PATH=Path(__file__).resolve().parents[1]/'tools'/'analyze_calibration_stability.py';SPEC=spec_from_file_location('calibration_stability',PATH);m=module_from_spec(SPEC);SPEC.loader.exec_module(m)
def test_bootstrap_is_reproducible_and_quantile_is_upper_tail():
 x=np.array([1.,2.,3.,4.,10.]);a=m.bootstrap_quantile(x,42,200);b=m.bootstrap_quantile(x,42,200);assert a==b and m.q98(x)==10.
def test_bootstrap_and_robust_estimators_validate_input():
 with pytest.raises(ValueError):m.bootstrap_quantile([1.])
 r=m.robust_estimators([1.,2.,3.,4.]);assert set(r)=={'empirical_q98','tukey_upper_fence','median_plus_3_mad_scaled','normal_mean_plus_2_326_std'}

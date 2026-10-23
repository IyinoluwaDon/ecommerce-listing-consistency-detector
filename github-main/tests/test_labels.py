import importlib.util
from pathlib import Path
import pandas as pd
import pytest
spec=importlib.util.spec_from_file_location("labels",Path(__file__).parents[1]/"src/build_labels.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
@pytest.mark.parametrize("fraction",[0,1])
def test_swaps(tmp_path,fraction):
    pd.DataFrame({"designation":["book","toy"],"description":["a","b"],"productid":[1,2],"imageid":[10,20]}).to_csv(tmp_path/"X_train_update.csv")
    pd.DataFrame({"prdtypecode":[10,40]}).to_csv(tmp_path/"Y_train_CVw08PX.csv")
    m.build(str(tmp_path),str(tmp_path/"out"),fraction,42)
    result=pd.read_csv(tmp_path/"out/train_with_mismatch_labels.csv")
    assert result.label.tolist()==[fraction,fraction]
    assert result.productid_for_image.tolist()==([1,2] if fraction==0 else [2,1])
@pytest.mark.parametrize("fraction",[-1,1.1,float("nan")])
def test_invalid(tmp_path,fraction):
    with pytest.raises(ValueError):m.build(str(tmp_path),str(tmp_path),fraction,42)

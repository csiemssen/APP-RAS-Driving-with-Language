from pathlib import Path


src_dir = Path(__file__).parent.resolve()
data_dir = src_dir / ".." / "data"
drivelm_dir = data_dir / "drivelm"
drivelm_json = drivelm_dir / "v1_1_train_nus.json"

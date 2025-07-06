from pathlib import Path

src_dir = Path(__file__).parent.resolve()
data_dir = src_dir / ".." / "data"
drivelm_dir = data_dir / "drivelm"
nuscenes_dir = data_dir / "nuscenes"
grid_dir = nuscenes_dir / "samples" / "GRID"
drivelm_train_json = drivelm_dir / "v1_1_train_nus.json"
drivelm_val_json = drivelm_dir / "v1_1_val_nus_q_only.json"
drivelm_test_json = drivelm_dir / "v1_1_test_nus.json"
log_dir = data_dir / "logs"
model_dir = src_dir / ".." / "models"
model_output_dir = model_dir / "hf_dumps"
model_log_dir = model_dir / "logs"
fonts_dir = data_dir / "fonts"

IMAGE_SIZE = (900, 1600)  # (height, width)
GRID = (2, 3)  # (rows, cols)
GRID_IMG_SIZE = (
    IMAGE_SIZE[0] * GRID[0],
    IMAGE_SIZE[1] * GRID[1],
)  # (height, width)

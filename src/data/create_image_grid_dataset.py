import os

import tqdm
from PIL import Image, ImageDraw, ImageFont

from src.constants import drivelm_dir, fonts_dir, grid_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_grid_image_with_labels(
    image_paths: dict,
    text_position: str = "top-left",
) -> Image.Image:
    img_width, img_height = 1600, 900

    grid_cols, grid_rows = 3, 2
    canvas_width = img_width * grid_cols
    canvas_height = img_height * grid_rows

    grid_img = Image.new("RGB", (canvas_width, canvas_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(grid_img)

    # Adjust font size scaling using square root of resize factor
    font_size = 28
    font = ImageFont.load_default()
    try:
        font = fonts_dir / "Roboto-Regular.ttf"
        font = ImageFont.truetype(str(font), font_size)
    except IOError:
        font = ImageFont.load_default()
        print("Warning: Fallback to default font")

    positions = {
        "CAM_FRONT_LEFT": (0, 0),
        "CAM_FRONT": (1, 0),
        "CAM_FRONT_RIGHT": (2, 0),
        "CAM_BACK_LEFT": (0, 1),
        "CAM_BACK": (1, 1),
        "CAM_BACK_RIGHT": (2, 1),
    }

    for cam, (col, row) in positions.items():
        img_path = image_paths.get(cam)
        if img_path is None:
            continue

        img = Image.open(img_path).convert("RGB")

        x_offset = col * img_width
        y_offset = row * img_height
        grid_img.paste(img, (x_offset, y_offset))

        label_text = f"<{cam}>"
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        if text_position == "top-left":
            label_x = x_offset + 10
            label_y = y_offset + 10

        elif text_position == "bottom-left":
            label_x = x_offset + 10
            label_y = y_offset + img_height - text_height - 10

        elif text_position == "top-right":
            label_x = x_offset + img_width - text_width - 10
            label_y = y_offset + 10

        elif text_position == "bottom-right":
            label_x = x_offset + img_width - text_width - 10
            label_y = y_offset + img_height - text_height - 10

        else:
            label_x = x_offset + 10
            label_y = y_offset + 10

        draw.text((label_x, label_y), label_text, fill="red", font=font)

    return grid_img


def create_image_grid_dataset(data, override=False):
    grid_dir.mkdir(parents=True, exist_ok=True)

    for scene_id, scene_data in tqdm.tqdm(
        data.items(), desc="Creating grid images for scenes"
    ):
        for key_frame_id, key_frame_data in scene_data["key_frames"].items():
            image_paths = key_frame_data["image_paths"]
            image_name = f"{scene_id}_{key_frame_id}__GRID.jpg"
            grid_path = grid_dir / image_name
            image_paths["GRID"] = "../nuscenes/samples/GRID/" + image_name

            if not grid_path.exists() or override:
                image_paths = {
                    key: os.path.join(drivelm_dir, path)
                    for key, path in image_paths.items()
                }
                grid_img = create_grid_image_with_labels(image_paths)
                grid_img.save(grid_path)
                logger.debug(f"Saved grid image: {grid_path}")

    return data

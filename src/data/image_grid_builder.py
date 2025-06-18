from PIL import Image, ImageDraw, ImageFont

from src.constants import fonts_dir


def create_grid_image_with_labels(
    image_paths: dict,
    font_size: int = 24,
    resize_factor: float = 1.0,
    text_position: str = "top-left",
) -> Image.Image:
    base_width, base_height = 1600, 900
    img_width = int(base_width * resize_factor)
    img_height = int(base_height * resize_factor)

    grid_cols, grid_rows = 3, 2
    canvas_width = img_width * grid_cols
    canvas_height = img_height * grid_rows

    grid_img = Image.new("RGB", (canvas_width, canvas_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(grid_img)
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
        try:
            img = Image.open(img_path).convert("RGB")
            if resize_factor != 1.0:
                img = img.resize((img_width, img_height), Image.BICUBIC)

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
        except Exception as e:
            print(f"Error processing {cam}: {e}")

    return grid_img

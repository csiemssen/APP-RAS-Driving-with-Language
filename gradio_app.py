import copy
import os

import gradio as gr

from src.constants import bev_dir, drivelm_dir, grid_dir
from src.data.basic_dataset import DriveLMImageDataset
from src.data.load_dataset import load_dataset
from src.models.gemma_inference import GemmaInferenceEngine
from src.models.intern_vl_inference import InternVLInferenceEngine
from src.models.qwen_vl_inference import QwenVLInferenceEngine

PROVIDERS = {
    "Google": {
        "Gemma-3-4B-IT": {
            "engine_class": GemmaInferenceEngine,
            "kwargs": {"model_path": "google/gemma-3-4b-it"},
        },
    },
    "Qwen": {
        "Qwen2.5-VL-7B-Instruct": {
            "engine_class": QwenVLInferenceEngine,
            "kwargs": {"processor_path": "Qwen/Qwen2.5-VL-7B-Instruct"},
        },
        "Qwen2.5-VL-3B-Instruct": {
            "engine_class": QwenVLInferenceEngine,
            "kwargs": {"processor_path": "Qwen/Qwen2.5-VL-3B-Instruct"},
        },
    },
    "InternVL": {
        "OpenGVLab/InternVL3-2B": {
            "engine_class": InternVLInferenceEngine,
            "kwargs": {"model_path": "OpenGVLab/InternVL3-2B"},
        }
    },
    # Add more providers/models here as needed
}

raw_dataset = None
inference_engine = None
dataset = None
image_paths_list = []
selected_question_item = None
kois_active = True


def get_engine(provider, model):
    global inference_engine
    engine_info = PROVIDERS[provider][model]
    engine_class = engine_info["engine_class"]
    inference_engine = engine_class(**engine_info.get("kwargs", {}))
    inference_engine.load_model()
    return inference_engine.model_path


def get_dataset(split, add_kois, add_bev, use_grid, use_system_prompt):
    global dataset, selected_question_item, raw_dataset

    if inference_engine is None:
        raise gr.Error("Please load model first", duration=2)

    dataset = DriveLMImageDataset(
        message_format=inference_engine.message_formatter,
        split=split,
        front_cam=True,
        add_kois=add_kois,
        add_bev=add_bev,
        use_grid=use_grid,
        use_system_prompt=use_system_prompt,
    )
    raw_dataset = load_dataset(split)
    return split


def get_models(provider):
    models = list(PROVIDERS[provider].keys())
    return gr.update(choices=models, value=models[0] if models else None)


def parse_keyframe_id(id: str):
    return id.split("_")[1]


def parse_scene_id(id: str):
    return id.split("_")[0]


def parse_question_id(id: str):
    return id.split("_")[2]


def filter_question_items(
    items, scene_id=None, keyframe_id=None, question_type=None, question_id=None
):
    return [
        item
        for item in items
        if (scene_id is None or parse_scene_id(item.qa_id) == scene_id)
        and (keyframe_id is None or parse_keyframe_id(item.qa_id) == keyframe_id)
        and (question_type is None or item.qa_type == question_type)
        and (question_id is None or parse_question_id(item.qa_id) == question_id)
    ]


def get_scene_id():
    global selected_question_item
    if selected_question_item is not None:
        return parse_scene_id(selected_question_item.qa_id)
    return None


def get_scenes(items):
    return sorted({parse_scene_id(item.qa_id) for item in items})


def render_scenes(items):
    scene_ids = get_scenes(items)
    return gr.update(
        choices=scene_ids,
        value=get_scene_id(),
    )


def get_keyframe_id():
    global selected_question_item
    if selected_question_item is not None:
        return parse_keyframe_id(selected_question_item.qa_id)
    return None


def get_keyframes(items):
    return sorted({parse_keyframe_id(item.qa_id) for item in items})


def render_keyframes(items):
    keyframe_ids = get_keyframes(items)
    return gr.update(
        choices=keyframe_ids,
        value=get_keyframe_id(),
    )


def get_question_types(items):
    return sorted({item.qa_type for item in items})


def render_question_types(items):
    question_types = get_question_types(items)
    return gr.update(
        choices=question_types,
        value=get_question_type(),
    )


def get_question_type():
    global selected_question_item
    if selected_question_item is not None:
        return selected_question_item.qa_type
    return None


def get_question_id():
    global selected_question_item
    if selected_question_item is not None:
        return parse_question_id(selected_question_item.qa_id)
    return None


def get_question_ids(items):
    return sorted({parse_question_id(item.qa_id) for item in items})


def render_question_ids(items):
    question_ids = get_question_ids(items)
    return gr.update(
        choices=question_ids,
        value=get_question_id(),
    )


def get_images(items):
    global image_paths_list
    scene_id = get_scene_id()
    keyframe_id = get_keyframe_id()

    if raw_dataset is None:
        raise_dataset_error()
        return []

    scene_data = raw_dataset[scene_id]
    keyframe_data = scene_data["key_frames"][keyframe_id]

    all_native_image_paths = set(keyframe_data["image_paths"].values())

    other_image_files = [
        os.path.join(bev_dir, f"{scene_id}_{keyframe_id}__BEV_FRONT_CAM.jpg"),
        os.path.join(bev_dir, f"{scene_id}_{keyframe_id}__BEV.jpg"),
        os.path.join(grid_dir, f"{scene_id}_{keyframe_id}__GRID.jpg"),
    ]

    for file in other_image_files:
        if os.path.isfile(file):
            all_native_image_paths.add(file)

    image_paths_list = [
        os.path.join(drivelm_dir, path) for path in all_native_image_paths
    ]
    return image_paths_list


def get_image():
    global selected_question_item
    return (
        selected_question_item.image_path
        if selected_question_item is not None
        else None
    )


def update_image(evt: gr.SelectData):
    global image_paths_list, selected_question_item
    if 0 <= evt.index < len(image_paths_list):
        selected_question_item.image_path = image_paths_list[evt.index]
        return [selected_question_item.image_path, get_formatted_question()]
    return [None, get_formatted_question()]


def get_question():
    global selected_question_item
    return (
        selected_question_item.question if selected_question_item is not None else None
    )


def update_question(text):
    global selected_question_item
    if selected_question_item is None:
        raise_dataset_error()
        return None
    selected_question_item.question = text
    return [
        selected_question_item.question,
        get_formatted_question(),
        get_ground_truth(invalid=True),
    ]


def get_system_prompt():
    global selected_question_item
    return (
        selected_question_item.system_prompt
        if selected_question_item is not None
        else None
    )


def update_system_prompt(text):
    global selected_question_item
    if selected_question_item is None:
        raise_dataset_error()
        return None
    selected_question_item.system_prompt = text
    return [selected_question_item.system_prompt, get_formatted_question()]


def get_ground_truth(invalid=False):
    global selected_question_item

    if invalid:
        return gr.update(value=None, visible=False)

    if selected_question_item is not None:
        if (
            selected_question_item.ground_truth_answer is None
            or selected_question_item.ground_truth_answer == ""
        ):
            return gr.update(value=None, visible=False)

        return gr.update(value=selected_question_item.ground_truth_answer, visible=True)
    return gr.update(value=None, visible=False)


def get_kois():
    global selected_question_item
    if selected_question_item is not None:
        return [
            selected_question_item.key_object_info,
            get_formatted_question(),
        ]
    return [None, get_formatted_question()]


def update_kois_active(active):
    global kois_active
    kois_active = active
    return [active, get_formatted_question()]


def update_question_item():
    global dataset, selected_question_item
    if dataset is None or len(dataset) == 0:
        raise_dataset_error()
        return None

    selected_question_item = copy.deepcopy(dataset[0])
    scene_items = filter_question_items(dataset, scene_id=get_scene_id())
    return [
        render_scenes(dataset),
        *render_question_item_change_on_scene_id(scene_items),
    ]


def update_question_item_on_scene_id(scene_id):
    global dataset, selected_question_item
    if dataset is None or len(dataset) == 0:
        raise_dataset_error()
        return None
    for item in dataset:
        if parse_scene_id(item.qa_id) == scene_id:
            selected_question_item = copy.deepcopy(item)
            break

    scene_items = filter_question_items(items=dataset, scene_id=scene_id)
    return [*render_question_item_change_on_scene_id(scene_items)]


def update_question_item_on_keyframe_id(scene_id, keyframe_id):
    global dataset, selected_question_item
    keyframe_items = filter_question_items(
        items=dataset, scene_id=scene_id, keyframe_id=keyframe_id
    )
    for item in keyframe_items:
        selected_question_item = copy.deepcopy(item)
        break

    return [*render_question_item_change_on_keyframe(keyframe_items)]


def update_question_item_on_question_type(scene_id, keyframe_id, question_type):
    global dataset, selected_question_item

    question_type_items = filter_question_items(
        items=dataset,
        scene_id=scene_id,
        keyframe_id=keyframe_id,
        question_type=question_type,
    )

    for item in question_type_items:
        selected_question_item = copy.deepcopy(item)
        break

    return [
        *render_question_item_change_on_question_type(question_type_items),
    ]


def update_question_item_on_question_id(
    scene_id, keyframe_id, question_type, question_id
):
    global dataset, selected_question_item

    question_id_items = filter_question_items(
        items=dataset,
        scene_id=scene_id,
        keyframe_id=keyframe_id,
        question_type=question_type,
        question_id=question_id,
    )

    for item in question_id_items:
        if parse_question_id(item.qa_id) == question_id:
            selected_question_item = copy.deepcopy(item)
            break

    return [
        get_question(),
        get_system_prompt(),
        get_formatted_question(),
        get_ground_truth(),
    ]


def render_question_item_change_on_scene_id(items):
    keyframe_items = filter_question_items(items, keyframe_id=get_keyframe_id())
    return [
        render_keyframes(items),
        *render_question_item_change_on_keyframe(keyframe_items),
    ]


def render_question_item_change_on_keyframe(items):
    question_type_items = filter_question_items(
        items, question_type=get_question_type()
    )
    return [
        render_question_types(items),
        *render_question_item_change_on_question_type(question_type_items),
    ]


def render_question_item_change_on_question_type(items):
    return [
        render_question_ids(items),
        *render_question_item_change_on_question_id(),
        get_kois()[0],
        get_images(items),
        get_image(),
    ]


def render_question_item_change_on_question_id():
    return [
        get_question(),
        get_system_prompt(),
        get_formatted_question(),
        get_ground_truth(),
    ]


def get_formatted_question():
    global inference_engine, selected_question_item, kois_active

    if selected_question_item is None:
        raise_dataset_error()

    formatted_question = selected_question_item
    if not kois_active:
        formatted_question = copy.deepcopy(selected_question_item)
        formatted_question.key_object_info = None

    formatted_question.formatted_message = formatted_question.format_message(
        inference_engine.message_formatter
    )

    return formatted_question.formatted_message


def predict_question(formatted_question):
    global inference_engine

    responses = inference_engine.predict_batch([[formatted_question]])
    return responses[0]


def raise_dataset_error():
    raise gr.Error("Please load a dataset first.", duration=2)


with gr.Blocks() as demo:
    # UI COMPONENTS
    with gr.Row():
        with gr.Column(scale=0):
            loaded_model_textbox = gr.Textbox(
                label="Model",
                interactive=False,
                value="None",
            )
        with gr.Column(scale=0):
            loaded_dataset_textbox = gr.Textbox(
                label="Dataset",
                interactive=False,
                value="None",
            )
    with gr.Tab("Settings"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    default_provider = "Qwen"
                    default_model = "Qwen2.5-VL-7B-Instruct"
                    provider_dropdown = gr.Dropdown(
                        choices=list(PROVIDERS.keys()),
                        label="Provider",
                        value=default_provider,
                    )
                    model_dropdown = gr.Dropdown(
                        choices=list(PROVIDERS[default_provider].keys()),
                        label="Model",
                        value=default_model,
                    )

                with gr.Row():
                    load_model_button = gr.Button("Load Model", scale=0)

            with gr.Column():
                split = gr.Radio(
                    choices=["val", "test", "train"],
                    label="Split",
                    value="val",
                )
                add_kois = gr.Checkbox(label="Yolo KOIs", value=False)
                add_bev = gr.Checkbox(label="BEVs", value=False)
                use_grid = gr.Checkbox(label="Grid", value=True)
                use_system_prompt = gr.Checkbox(label="System Prompts", value=True)

                get_dataset_button = gr.Button("Load Dataset", scale=0)

    with gr.Tab("Chat"):
        with gr.Accordion("Question Picker", open=False):
            with gr.Row():
                scene_id_dropdown = gr.Dropdown(
                    label="Scene",
                    choices=[],
                    value=None,
                )
                keyframe_id_dropdown = gr.Dropdown(
                    label="Keyframe",
                    choices=[],
                    value=None,
                )
                question_type_radio = gr.Radio(
                    label="Question Type",
                    choices=[],
                    value=None,
                )
                question_ids_dropdown = gr.Dropdown(
                    label="Question",
                    choices=[],
                    value=None,
                )
        with gr.Row():
            with gr.Column():
                with gr.Accordion("Image Gallery", open=False):
                    image_gallery = gr.Gallery(
                        value=[],
                        object_fit="contain",
                        columns=3,
                        height="auto",
                    )
            with gr.Column():
                image = gr.Image(label="Selected Image", interactive=False)

        with gr.Accordion("Key Object Infos", open=False):
            kois_checkbox = gr.Checkbox(
                label="Pass Key Object Infos",
                value=kois_active,
            )

            kois_json = gr.JSON(label="Key Object Infos", value=None)

        system_prompt_textbox = gr.Textbox(
            label="System Prompt",
            value=None,
            interactive=True,
        )

        question_textbox = gr.Textbox(
            label="Question",
            value=None,
            interactive=True,
        )
        with gr.Accordion(label="Formatted Message", open=False):
            format_message_json = gr.JSON(value=None)

        response_textbox = gr.Textbox(label="Answer", value=None, interactive=False)

        ground_truth_textbox = gr.Textbox(
            label="Ground Truth Answer",
            value=None,
            interactive=False,
            visible=False,
        )

        send_button = gr.Button("Send")

    # EVENT HANDLERS
    get_dataset_button.click(
        fn=get_dataset,
        inputs=[
            split,
            add_kois,
            add_bev,
            use_grid,
            use_system_prompt,
        ],
        outputs=loaded_dataset_textbox,
    ).then(
        fn=update_question_item,
        inputs=None,
        outputs=[
            scene_id_dropdown,
            keyframe_id_dropdown,
            question_type_radio,
            question_ids_dropdown,
            question_textbox,
            system_prompt_textbox,
            format_message_json,
            ground_truth_textbox,
            kois_json,
            image_gallery,
            image,
        ],
    )

    load_model_button.click(
        fn=get_engine,
        inputs=[provider_dropdown, model_dropdown],
        outputs=loaded_model_textbox,
    )

    provider_dropdown.change(
        fn=get_models, inputs=provider_dropdown, outputs=model_dropdown
    )

    scene_id_dropdown.input(
        fn=update_question_item_on_scene_id,
        inputs=scene_id_dropdown,
        outputs=[
            keyframe_id_dropdown,
            question_type_radio,
            question_ids_dropdown,
            question_textbox,
            system_prompt_textbox,
            format_message_json,
            ground_truth_textbox,
            kois_json,
            image_gallery,
            image,
        ],
    )
    keyframe_id_dropdown.input(
        fn=update_question_item_on_keyframe_id,
        inputs=[scene_id_dropdown, keyframe_id_dropdown],
        outputs=[
            question_type_radio,
            question_ids_dropdown,
            question_textbox,
            system_prompt_textbox,
            format_message_json,
            ground_truth_textbox,
            kois_json,
            image_gallery,
            image,
        ],
    )

    question_type_radio.input(
        fn=update_question_item_on_question_type,
        inputs=[scene_id_dropdown, keyframe_id_dropdown, question_type_radio],
        outputs=[
            question_ids_dropdown,
            question_textbox,
            system_prompt_textbox,
            format_message_json,
            ground_truth_textbox,
            kois_json,
            image_gallery,
            image,
        ],
    )

    question_ids_dropdown.input(
        fn=update_question_item_on_question_id,
        inputs=[
            scene_id_dropdown,
            keyframe_id_dropdown,
            question_type_radio,
            question_ids_dropdown,
        ],
        outputs=[
            question_textbox,
            system_prompt_textbox,
            format_message_json,
            ground_truth_textbox,
        ],
    )

    image_gallery.select(
        fn=update_image,
        inputs=None,
        outputs=[image, format_message_json],
    )

    question_textbox.submit(
        fn=update_question,
        inputs=question_textbox,
        outputs=[question_textbox, format_message_json, ground_truth_textbox],
    )

    system_prompt_textbox.submit(
        fn=update_system_prompt,
        inputs=system_prompt_textbox,
        outputs=[system_prompt_textbox, format_message_json],
    )

    kois_checkbox.change(
        fn=update_kois_active,
        inputs=kois_checkbox,
        outputs=[kois_checkbox, format_message_json],
    )

    send_button.click(
        fn=predict_question,
        inputs=format_message_json,
        outputs=response_textbox,
    )


if __name__ == "__main__":
    demo.launch()

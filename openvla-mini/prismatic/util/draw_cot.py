import ast
import re
import textwrap
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from .cot_utils import CotTag, split_reasoning, get_cot_tags_list

def draw_cot(reasoning, action, obs, include_keys=None):
    parsed_reasoning = reasoning.replace('@', ' ')
    parsed_reasoning = split_reasoning(parsed_reasoning)
    metadata = get_metadata(parsed_reasoning)

    original_image = obs
    #original_image = obs["full_image"] 
    if isinstance(original_image, list):
        original_image = original_image[0]
    h, w = original_image.shape[:2]
    img_size = (w, h)

    annotated_img = draw_annotations(original_image, metadata, action, img_size)
    if include_keys:
        tags = get_cot_tags_list()
        tags = list(set(tags) & set(include_keys))
        annotated_img = create_side_text_panel(annotated_img, parsed_reasoning, tags, w, h)

    return annotated_img

def get_metadata(reasoning):
    metadata = {"gripper": [], "bboxes": dict()}

    gripper_key = f"{CotTag.GRIPPER_POSITION.value}"
    objects_key = f"{CotTag.VISIBLE_OBJECTS.value}"

    if gripper_key in reasoning:
        gripper_str = reasoning[gripper_key].strip()
        if gripper_str:
            try:
                gripper_pos = ast.literal_eval(gripper_str)
                pairs = [
                    (gripper_pos[2 * i], gripper_pos[2 * i + 1])
                    for i in range(len(gripper_pos) // 2)
                ]
                if pairs:
                    metadata["gripper"] = pairs
            except (ValueError, SyntaxError, TypeError):
                pass

    if objects_key in reasoning:
        objects_str = reasoning[objects_key].strip()
        if objects_str:
            try:
                parsed = ast.literal_eval(objects_str)
                if isinstance(parsed, dict) and parsed:
                    metadata["bboxes"] = parsed
            except (ValueError, SyntaxError):
                pass

    return metadata

def _sanitize_gripper_points(pos_list):
    """Convert parsed gripper pairs to integer pixel coordinates; drop invalid entries."""
    out = []
    for pos in pos_list:
        try:
            x = int(round(float(pos[0])))
            y = int(round(float(pos[1])))
        except (TypeError, ValueError, IndexError):
            continue
        out.append((x, y))
    return out


def draw_annotations(image_array, metadata, action_vector, img_size=(640, 480)):
    img = np.ascontiguousarray(image_array)
    gripper_pos = _sanitize_gripper_points(metadata["gripper"])
    bbs = metadata["bboxes"] if isinstance(metadata.get("bboxes"), dict) else {}

    if bbs:
        draw_bboxes(img, bbs, img_size)
    if gripper_pos:
        draw_gripper(img, gripper_pos, img_size)
        draw_action(img, gripper_pos, action_vector)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def draw_gripper(img, pos_list, img_size=(640, 480)):
    for i, pos in enumerate(reversed(pos_list)):
        #pos = resize_pos(pos, img_size)
        scale = 255 - int(255 * i / len(pos_list))
        cv2.circle(img, pos, 6, (0, 0, 0), -1)
        cv2.circle(img, pos, 5, (scale, scale, 255), -1)

def draw_bboxes(img, bboxes, img_size=(640, 480)):
    for name, bbox in bboxes.items():
        try:
            x1, y1 = int(bbox[0][0]), int(bbox[0][1])
            x2, y2 = int(bbox[1][0]), int(bbox[1][1])
        except (TypeError, ValueError, IndexError):
            continue
        show_name = str(name)
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            name_to_random_color(name),
            1,
        )
        cv2.putText(
            img,
            show_name,
            (x1, y1 + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

def draw_action(img, gripper_pos, action_vector):
    start_x, start_y = gripper_pos[0]
    scaled_start = (start_x, start_y)

    scale_factor = 500
    pixel_dx = int(action_vector[0] * scale_factor)
    pixel_dy = int(- action_vector[1] * scale_factor)
    scaled_end = (start_x + pixel_dx, start_y + pixel_dy)

    cv2.arrowedLine(
        img,
        scaled_start,
        scaled_end,
        (255, 0, 0),  # Red color in BGR
        2,            # Thickness of the line
        cv2.LINE_AA
    )

def create_side_text_panel(annotated_img, parsed_reasoning, display_keys, image_width, image_height):
    text_list = [tag + parsed_reasoning[tag] for tag in display_keys if tag in parsed_reasoning]


    caption = ""
    for t in text_list:
        wrapper = textwrap.TextWrapper(width=110, replace_whitespace=False)
        word_list = wrapper.wrap(text=t)
        caption_new = ''
        for ii in word_list[:-1]:
            caption_new = caption_new + ii + '\n      '
        caption_new += word_list[-1]

        caption += caption_new.lstrip() + "\n\n"

    # Create white base for the text panel
    base = Image.fromarray(np.ones((image_width, image_height, 3), dtype=np.uint8) * 255)
    draw = ImageDraw.Draw(base)
    
    # Load font
    font_path = os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, size=10)
    color = (0, 0, 0) # RGB
    
    # Draw text onto the base
    draw.text((15, 15), caption, color, font=font)

    # Concatenate the original image array and the text panel array
    text_arr = np.array(base)
    reasoning_img = np.concatenate([annotated_img, text_arr], axis=1)
    
    return reasoning_img

def name_to_random_color(name):
    return [(hash(name) // (256**i)) % 256 for i in range(3)]


def resize_pos(pos, img_size):
    return tuple((int(x) * int(size)) // 256 for x, size in zip(pos, img_size))
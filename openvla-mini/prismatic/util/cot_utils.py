import enum


class CotTag(enum.Enum):
    #TASK = "TASK:"
    PLAN = "PLAN:"
    VISIBLE_OBJECTS = "VISIBLE OBJECTS:"
    SUBTASK_REASONING = "SUBTASK REASONING:"
    SUBTASK = "SUBTASK:"
    MOVE_REASONING = "MOVE REASONING:"
    MOVE = "MOVE:"
    GRIPPER_POSITION = "GRIPPER POSITION:"
    ACTION = "ACTION:"

def split_reasoning(text):
    tags = get_cot_tags_list()
    new_parts = {None: text}

    for tag in tags:
        parts = new_parts
        new_parts = dict()

        for k, v in parts.items():
            if tag in v:
                # Add 1 to ensure it only splits the first instance of the tag
                s = v.split(tag, 1) 
                
                # Only keep the 'k' key if the text before the tag isn't empty
                if s[0].strip():
                    new_parts[k] = s[0]
                
                new_parts[tag] = s[1]
            else:
                new_parts[k] = v

    # Clean up the None key at the end if it's completely empty
    if None in new_parts and not new_parts[None].strip():
        del new_parts[None]

    return new_parts

def abbreviate_tag(tag: str):
    return tag[0] + tag[-2]


def get_cot_tags_list():
    return [
        #CotTag.TASK.value,
        CotTag.PLAN.value,
        CotTag.VISIBLE_OBJECTS.value,
        CotTag.SUBTASK_REASONING.value,
        CotTag.SUBTASK.value,
        CotTag.MOVE_REASONING.value,
        CotTag.MOVE.value,
        CotTag.GRIPPER_POSITION.value,
        CotTag.ACTION.value,
    ]


def get_cot_database_keys():
    return {
        #CotTag.TASK.value: "task",
        CotTag.PLAN.value: "plan",
        CotTag.VISIBLE_OBJECTS.value: "bboxes",
        CotTag.SUBTASK_REASONING.value: "subtask_reasoning",
        CotTag.SUBTASK.value: "subtask",
        CotTag.MOVE_REASONING.value: "movement_reasoning",
        CotTag.MOVE.value: "movement",
        CotTag.GRIPPER_POSITION.value: "gripper",
        CotTag.ACTION.value: "action",
    }

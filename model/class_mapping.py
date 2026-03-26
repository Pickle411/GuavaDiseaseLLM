from common.config import CLASS_NAMES

# index -> 英文類別
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# 英文類別 -> index
CLASS_TO_IDX = {name: idx for idx, name in IDX_TO_CLASS.items()}

# 英文類別 -> 中文類別
CLASS_TO_ZH = {
    "Anthracnose": "炭疽病",
    "fruit_fly": "果蠅危害",
    "healthy_guava": "健康芭樂",
}

# index -> 中文類別
IDX_TO_ZH = {idx: CLASS_TO_ZH[name] for idx, name in IDX_TO_CLASS.items()}


def get_class_name(class_idx: int) -> str:
    return IDX_TO_CLASS[class_idx]


def get_class_name_zh(class_name: str) -> str:
    return CLASS_TO_ZH[class_name]


def get_class_zh_by_idx(class_idx: int) -> str:
    return IDX_TO_ZH[class_idx]
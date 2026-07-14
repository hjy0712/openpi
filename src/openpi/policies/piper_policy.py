"""Piper 机械臂策略定义 - 用于 π₀.₅ 训练和推理。

Piper 机械臂有 6 个关节，无夹爪数据。模型直接输出关节角度。
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# Piper 机械臂关节数
PIPER_NUM_JOINTS = 7  # 6 个关节 + 1 个夹爪


def make_piper_example() -> dict:
    """创建一个随机示例输入用于测试。"""
    return {
        "observation/state": np.random.rand(PIPER_NUM_JOINTS),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the block",
    }


def _parse_image(image) -> np.ndarray:
    """将图像转为 uint8 (H,W,C) 格式。LeRobot 自动存为 float32 (C,H,W)。"""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """
    将 Piper 数据转换为模型期望的输入格式。

    Piper 有:
    - 1 个顶部摄像头 (base_0_rgb)
    - 1 个腕部摄像头 (left_wrist_0_rgb)
    - 6 个关节角度 (state)
    - 6 个关节角度动作 (actions)
    - 无右腕摄像头 → 用零填充
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Piper 没有右腕摄像头，填充零
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # π₀.₅ (PI0 模式) 对不存在的图像 mask 为 False
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # 动作（仅训练时有）
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # 语言指令
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    """
    将模型输出转为 Piper 的动作格式。

    模型输出被 PadStatesAndActions 填充到了 action_dim（默认为 32），
    这里裁剪回 Piper 的 6 个关节角度。
    """

    def __call__(self, data: dict) -> dict:
        # 只取前 6 个关节角度（去掉填充部分）
        return {"actions": np.asarray(data["actions"][..., :PIPER_NUM_JOINTS])}

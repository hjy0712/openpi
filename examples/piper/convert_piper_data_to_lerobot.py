"""
将 Piper 机械臂采集数据转换为 LeRobot 格式。

数据层级：
  piper_v2/
    {session_id}/               ← 一个 session = LeRobot 的一个 episode
      {step_id:06d}/            ← 一个文件夹 = 1 帧（五文件对应同一时刻）
        {ts}_rgb_top.npy       - 顶部摄像头 (480, 640, 3) uint8
        {ts}_rgb_wrist.npy     - 腕部摄像头 (480, 640, 3) uint8
        {ts}_joint.json        - 跟随臂关节角度 (6 DOF)
        {ts}_joint_leader.json - 主臂关节角度 (6 DOF)
        {ts}_ee.json           - 末端位姿

用法:
  uv run examples/piper/convert_piper_data_to_lerobot.py --data_dir /data/nvme2/houjunyi/piper_v2
"""

import json
import shutil
from pathlib import Path

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro

REPO_NAME = "houjunjun/piper"
DEFAULT_PROMPT = "pick up the blue bag and place it in the orange zone"


def load_frame(step_dir: Path) -> dict | None:
    """加载一个 step 文件夹（1帧 = 5个文件）。"""
    files = list(step_dir.iterdir())
    if not files:
        return None

    top_img = wrist_img = joint_file = None
    for f in files:
        name = f.name
        if name.endswith("_rgb_top.npy"):
            top_img = f
        elif name.endswith("_rgb_wrist.npy"):
            wrist_img = f
        elif name.endswith("_joint.json"):
            joint_file = f

    if not all([top_img, wrist_img, joint_file]):
        return None
    return {"top_image": top_img, "wrist_image": wrist_img, "joint": joint_file}


def main(data_dir: str, output_dir: str = "/data/nvme2/houjunyi/vla_piper_lerobot", *, push_to_hub: bool = False):
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found")

    root = Path(output_dir)
    output_path = root / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        root=root,
        robot_type="piper",
        fps=30,
        features={
            "image": {"dtype": "image", "shape": (480, 640, 3), "names": ["h", "w", "c"]},
            "wrist_image": {"dtype": "image", "shape": (480, 640, 3), "names": ["h", "w", "c"]},
            "state": {"dtype": "float32", "shape": (7,), "names": ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    total_frames = 0
    sessions = sorted(d for d in data_path.iterdir() if d.is_dir())
    print(f"Found {len(sessions)} sessions")

    for session_dir in sessions:
        # 按编号排序，每个文件夹 = 1 帧
        step_dirs = sorted(
            (d for d in session_dir.iterdir() if d.is_dir()),
            key=lambda x: int(x.name),
        )
        frames = [f for d in step_dirs if (f := load_frame(d))]
        if len(frames) < 2:
            continue

        # 前后帧配对: state=frame[i], action=frame[i+1]
        for i in range(len(frames) - 1):
            cur, nxt = frames[i], frames[i + 1]

            # BGR → RGB（OpenCV 采集的图像通道顺序）
            top_img = np.load(str(cur["top_image"]))[..., ::-1].copy()
            wrist_img = np.load(str(cur["wrist_image"]))[..., ::-1].copy()

            with open(cur["joint"]) as f:
                cur_joints = np.array(json.load(f)["position"], dtype=np.float32)
            with open(nxt["joint"]) as f:
                nxt_joints = np.array(json.load(f)["position"], dtype=np.float32)

            dataset.add_frame({
                "image": top_img,
                "wrist_image": wrist_img,
                "state": cur_joints,
                "actions": nxt_joints,
                "task": DEFAULT_PROMPT,
            })
            total_frames += 1

        dataset.save_episode()

    print(f"Total frames: {total_frames}, episodes: {len(sessions)} -> saved to {output_path}")

    if push_to_hub:
        dataset.push_to_hub(tags=["piper", "robot"], private=False, push_videos=True, license="apache-2.0")


if __name__ == "__main__":
    tyro.cli(main)

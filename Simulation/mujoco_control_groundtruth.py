import mujoco
import mujoco.viewer
import numpy as np
import time
import os

def control_hand():
    model_path = r"C:\Users\USER\Downloads\sahan\mujoco_menagerie-main\mujoco_menagerie-main\shadow_hand\scene_right.xml"
    label_path = r"C:\Users\USER\Downloads\sahan\y_test.npy"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"MuJoCo model not found: {model_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"y_test.npy not found: {label_path}")

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    open_hand = np.zeros(20)
    close_hand = np.array([
        0, 0, 0, 1.571, 0, 1.571, 0.17, 0, 1.571, 1.571,
        0, 1.571, 1.571, 0, 1.571, 0, 0, 1.571, 1.2, 0
    ])
    labels = np.load(label_path).flatten()
    print(f"Loaded {len(labels)} labels from y_test.npy")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for label in labels:
            target = open_hand if label == 1 else close_hand
            data.ctrl[:] = target
            start_time = time.time()
            while time.time() - start_time < 1.0:
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.001)

if __name__ == "__main__":
    control_hand()
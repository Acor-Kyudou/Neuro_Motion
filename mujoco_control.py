import mujoco
import mujoco.viewer
import numpy as np
import time
import os

def control_hand():
    model_path = r"C:\Users\USER\Downloads\sahan\mujoco_menagerie-main\mujoco_menagerie-main\shadow_hand\scene_right.xml"
    pred_path = r"C:\Users\USER\Downloads\sahan\predictions_hand.npy"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"MuJoCo model not found: {model_path}")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"predictions_hand.npy not found: {pred_path}")
    if 'foot' in pred_path.lower():
        raise ValueError("Foot data not supported for hand simulation")

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    open_hand = np.zeros(20)
    close_hand = np.array([
        0, 0, 0, 1.571, 0, 1.571, 0.17, 0, 1.571, 1.571,
        0, 1.571, 1.571, 0, 1.571, 0, 0, 1.571, 1.2, 0
    ])
    predictions = np.load(pred_path).flatten()
    print(f"Loaded {len(predictions)} hand predictions")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for pred in predictions:
            target = open_hand if pred == 1 else close_hand
            data.ctrl[:] = target
            start_time = time.time()
            while time.time() - start_time < 1.0:
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.001)

if __name__ == "__main__":
    control_hand()
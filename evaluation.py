import os
import numpy as np
import torch
from model.model import RestorationModel


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "weights/best_model.pth"
TEST_DIR = "test_images"
OUTPUT_DIR = "restored_outputs"


os.makedirs(OUTPUT_DIR, exist_ok=True)


model = RestorationModel().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()


files = [
    f for f in os.listdir(TEST_DIR)
    if f.endswith(".npy")
]


with torch.no_grad():

    for filename in files:

        input_path = os.path.join(
            TEST_DIR,
            filename
        )

        image = np.load(
            input_path
        ).astype(np.float32)

        image = torch.from_numpy(
            image
        ).unsqueeze(0).unsqueeze(0)

        image = image.to(DEVICE)

        restored = model(image)

        restored = restored.squeeze().cpu().numpy()

        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        np.save(
            output_path,
            restored
        )

        print(
            f"Restored: {filename}"
        )

print("Inference completed.")

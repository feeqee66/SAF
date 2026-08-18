import numpy as np

from metrics import calculate_metrics


# Simple test images in [0, 1]
gt = np.zeros((64, 64), dtype=np.float32)

pred = np.zeros((64, 64), dtype=np.float32)

# Introduce a small error
pred[20:30, 20:30] = 0.1


results = calculate_metrics(pred, gt)


print("==============================")
print("METRICS TEST")
print("==============================")

for name, value in results.items():
    print(f"{name}: {value}")

print("==============================")
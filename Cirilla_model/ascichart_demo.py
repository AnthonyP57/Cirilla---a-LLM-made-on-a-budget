import time
import os
import asciichartpy as asciichart

# Simulate training loss decreasing and validation loss decreasing more slowly
training_loss = []
validation_loss = []

config = {
    "height": 15,
    "colors": [asciichart.blue, asciichart.red]  # train=blue, val=red
}

EPOCHS = 20

for epoch in range(1, EPOCHS + 1):
    # Simulate new loss values
    new_train = 5 / (1 + 0.2 * epoch) + 0.1 * (0.5 - epoch % 2)  # a noisy decay
    new_val = 5 / (1 + 0.18 * epoch) + 0.15 * (0.5 - (epoch + 1) % 2)

    training_loss.append(new_train)
    validation_loss.append(new_val)

    # Clear terminal
    os.system("cls" if os.name == "nt" else "clear")

    print("Training Loss (blue) vs Validation Loss (red)\n")
    print(asciichart.plot([training_loss, validation_loss], config))

    print(f"Epoch: {epoch}/{EPOCHS}")
    print(f"Training Loss: {new_train:.4f}")
    print(f"Validation Loss: {new_val:.4f}")
    print("Model improving..." if new_val < new_train else "Possible overfitting?")

    time.sleep(0.5)  # pause to simulate training

import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw

from NN import load_model, predict, forward_pass


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

W1, b1, W2, b2 = load_model()


# --------------------------------------------------
# Window settings
# --------------------------------------------------

WINDOW_SIZE = 400
CANVAS_SIZE = 280
BRUSH_SIZE = 16  # slightly thinner brush -> closer to MNIST stroke proportions


# --------------------------------------------------
# Create window
# --------------------------------------------------

root = tk.Tk()

root.title(
    "Handwritten Digit Classifier"
)

root.geometry(
    "500x600"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

title = tk.Label(
    root,
    text="Draw a digit",
    font=("Arial", 24)
)

title.pack(
    pady=15
)


# --------------------------------------------------
# Canvas
# --------------------------------------------------

canvas = tk.Canvas(
    root,
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    bg="black"
)

canvas.pack()


# --------------------------------------------------
# PIL image
# --------------------------------------------------

image = Image.new(
    "L",
    (CANVAS_SIZE, CANVAS_SIZE),
    0
)

draw = ImageDraw.Draw(
    image
)


# --------------------------------------------------
# Drawing (now with continuous strokes, not just dabs)
# --------------------------------------------------

last_x, last_y = None, None


def draw_digit(event):
    global last_x, last_y

    x, y = event.x, event.y
    r = BRUSH_SIZE // 2

    if last_x is not None:
        # Connect the previous point to this one so fast mouse
        # movement doesn't leave gaps in the stroke.
        canvas.create_line(
            last_x, last_y, x, y,
            fill="white",
            width=BRUSH_SIZE,
            capstyle=tk.ROUND,
            smooth=True
        )
        draw.line(
            (last_x, last_y, x, y),
            fill=255,
            width=BRUSH_SIZE,
            joint="curve"
        )

    # Still stamp a round dab so single clicks (dots) and stroke
    # ends/starts look round rather than square.
    canvas.create_oval(
        x - r, y - r, x + r, y + r,
        fill="white", outline="white"
    )
    draw.ellipse(
        (x - r, y - r, x + r, y + r),
        fill=255
    )

    last_x, last_y = x, y


def reset_stroke(event):
    global last_x, last_y
    last_x, last_y = None, None


canvas.bind("<B1-Motion>", draw_digit)
canvas.bind("<ButtonRelease-1>", reset_stroke)


# --------------------------------------------------
# Predict
# --------------------------------------------------

def preprocess(image):
    """
    Crop to the drawn digit's bounding box, then paste it centered
    into a 28x28 frame -- mirroring how MNIST digits are normalized
    (centered by bounding box inside the frame) instead of just
    squashing the whole blank canvas down to 28x28.
    """
    bbox = image.getbbox()

    if bbox is None:
        # Nothing drawn yet
        return np.zeros((1, 784), dtype=np.float32)

    cropped = image.crop(bbox)

    # Fit the digit into a 20x20 box, preserving aspect ratio,
    # then pad to 28x28 with a 4px border on each side (this is
    # the same convention the real MNIST dataset uses).
    cropped.thumbnail((20, 20), Image.LANCZOS)

    canvas_28 = Image.new("L", (28, 28), 0)
    upper_left = (
        (28 - cropped.width) // 2,
        (28 - cropped.height) // 2
    )
    canvas_28.paste(cropped, upper_left)

    img_array = np.array(canvas_28, dtype=np.float32)
    img_array /= 255.0
    img_array = img_array.reshape(1, 784)

    return img_array


def classify_digit():

    img_array = preprocess(image)

    if not img_array.any():
        result_label.config(text="Draw a digit first!")
        return

    # Get prediction
    prediction = predict(
        img_array,
        W1,
        b1,
        W2,
        b2
    )[0]

    # Get probabilities
    _, _, _, A2 = forward_pass(
        img_array,
        W1,
        b1,
        W2,
        b2
    )

    confidence = A2[
        prediction,
        0
    ] * 100

    result_label.config(
        text=(
            f"Prediction: {prediction}\n"
            f"Confidence: {confidence:.2f}%"
        )
    )


# --------------------------------------------------
# Clear canvas
# --------------------------------------------------

def clear_canvas():

    canvas.delete(
        "all"
    )

    draw.rectangle(
        (
            0,
            0,
            CANVAS_SIZE,
            CANVAS_SIZE
        ),
        fill=0
    )

    result_label.config(
        text="Prediction: -"
    )


# --------------------------------------------------
# Buttons
# --------------------------------------------------

button_frame = tk.Frame(
    root
)

button_frame.pack(
    pady=20
)


predict_button = tk.Button(
    button_frame,
    text="Predict",
    font=("Arial", 16),
    command=classify_digit
)

predict_button.pack(
    side=tk.LEFT,
    padx=10
)


clear_button = tk.Button(
    button_frame,
    text="Clear",
    font=("Arial", 16),
    command=clear_canvas
)

clear_button.pack(
    side=tk.LEFT,
    padx=10
)


# --------------------------------------------------
# Result
# --------------------------------------------------

result_label = tk.Label(
    root,
    text="Prediction: -",
    font=("Arial", 20)
)

result_label.pack(
    pady=10
)


# --------------------------------------------------
# Start application
# --------------------------------------------------

root.mainloop()
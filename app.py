import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the pre-trained model
model = keras.models.load_model('object_tracking.h5')

# Initialize OpenCV video capture
cap = None

def open_file():
    global cap
    file_path = filedialog.askopenfilename()
    if file_path:
        cap = cv2.VideoCapture(file_path)
        start_tracking()


# Initialize object tracker (you can choose different tracker types)
tracker = cv2.TrackerKCF_create()

# Initialize variables to store the initial bounding box
bbox = None
tracking = False

def start_tracking():
    global tracking, bbox

    if cap is not None:
        tracking = True
        while tracking:
            ret, frame = cap.read()
            if not ret:
                break

            if bbox is not None:
                # Update the tracker with the new frame
                tracking, bbox = tracker.update(frame)

                if tracking:
                    # Draw the bounding box on the frame
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the frame in the GUI
            display_frame(frame)

def stop_tracking():
    global tracking, bbox
    tracking = False
    bbox = None

def display_frame(frame):
    # Convert BGR frame to RGB for displaying with tkinter
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
    video_label.config(image=photo)
    video_label.image = photo
    video_label.update()


# Create a tkinter window
window = tk.Tk()
window.title("Object Tracking GUI")
window.geometry("800x600")

# Add a File Dialog button
open_button = tk.Button(window, text="Open Video File", command=open_file)
open_button.pack()

# Add a Label to display the video feed
video_label = tk.Label(window)
video_label.pack()

# Add Start and Stop buttons
start_button = tk.Button(window, text="Start Tracking", command=start_tracking)
start_button.pack()

stop_button = tk.Button(window, text="Stop Tracking", command=stop_tracking)
stop_button.pack()

# Start the tkinter main loop
window.mainloop()

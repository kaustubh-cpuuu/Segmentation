import tkinter as tk
from tkinter import ttk
import subprocess

def set_brightness(value):
    brightness = float(value) / 100  # Convert the value to a float between 0.0 and 1.0
    command = f"xrandr --output eDP-1 --brightness {brightness}"
    subprocess.run(command, shell=True)

# Create the main window with a larger size
root = tk.Tk()
root.title("Brightness Slider")
root.geometry("400x200")  # Set the window size to 400x200 pixels

# Create a label with a larger font size
label = ttk.Label(root, text="Adjust Brightness", font=("Helvetica", 14))
label.pack(pady=20)

# Create the slider with a larger scale
slider = ttk.Scale(root, from_=10, to=100, orient="horizontal", command=set_brightness, length=300)
slider.set(70)  # Set the default brightness level (70%)
slider.pack(padx=20, pady=20)

# Start the GUI event loop
root.mainloop()


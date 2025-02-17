import os
from PIL import Image

# Settings
frame_folder = "galaxy_frames"  # Folder containing frames
output_gif = "galaxy.gif"    # Output GIF file
fps = 30                     # Frames per second

frames = sorted([os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith(".png")])

if not frames:
    print("No frames found in 'me_frames' folder!")
    exit()

images = [Image.open(frame) for frame in frames]
images[0].save(output_gif, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)
print(f"GIF saved as {output_gif}")

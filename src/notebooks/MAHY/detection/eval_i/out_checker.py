import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import numpy as np
from pathlib import Path

IMG_DIR = '/media/plerolland/LaBoite/MAHY_eval/i-reboot'
COLORMAP = 'inferno'

image_files = (glob.glob(os.path.join(IMG_DIR, '*.png')))
np.random.seed(0)
np.random.shuffle(image_files)
current_index = [0]

pos_file, neg_file, skipped_file, shots_file = "pos.txt", "neg.txt", "skipped.txt", "shots.txt"

def write_pos(text):
    with open(pos_file, "a") as f:
        f.write(text + "\n")
    print(f"add {text}")

def write_neg(text):
    with open(neg_file, "a") as f:
        f.write(text + "\n")
    print(f"del {text}")

def write_skipped(text):
    with open(skipped_file, "a") as f:
        f.write(text + "\n")
    print(f"skip {text}")

def write_shots(text):
    with open(shots_file, "a") as f:
        f.write(text + "\n")
    print(f"shot {text}")

def on_key(event):
    if event.key == 'enter':
        filename = os.path.basename(image_files[current_index[0]])
        write_pos(filename.split("/")[-1][:-4])
        plt.close()
    elif event.key == 'backspace':
        filename = os.path.basename(image_files[current_index[0]])
        write_neg(filename.split("/")[-1][:-4])
        plt.close()
    elif event.key == 'escape':
        filename = os.path.basename(image_files[current_index[0]])
        write_skipped(filename.split("/")[-1][:-4])
        plt.close()
    elif event.key == 'delete':
        filename = os.path.basename(image_files[current_index[0]])
        write_shots(filename.split("/")[-1][:-4])
        plt.close()

def on_click(event):
    if event.inaxes and event.button == 3:
        date = os.path.basename(image_files[current_index[0]]).split("/")[-1][:-4].split("_")
        station, date = date[0], "_".join(date[1:])
        date = datetime.datetime.strptime(date, "%Y%m%d_%H%M%S_%f")
        print(f"Clic à x = {event.xdata:.2f}")

        date += datetime.timedelta(seconds=event.xdata*0.25 - 100)
        write_pos(f'{station}_{date.strftime("%Y%m%d_%H%M%S_%f")}')

def show_image(index):
    img = mpimg.imread(image_files[index])
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 6)
    cax = ax.imshow(img, cmap=COLORMAP, aspect="auto")
    ax.set_title(str(index) + "_" + os.path.basename(image_files[index]))
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

if __name__=="__main__":
    done = []
    if Path(pos_file).exists():
        with open(pos_file, "r") as f:
            done.extend(f.read().splitlines())
    if Path(neg_file).exists():
        with open(neg_file, "r") as f:
            done.extend(f.read().splitlines())
    if Path(skipped_file).exists():
        with open(skipped_file, "r") as f:
            done.extend(f.read().splitlines())
    if Path(shots_file).exists():
        with open(shots_file, "r") as f:
            done.extend(f.read().splitlines())
    while current_index[0] < len(image_files) and os.path.basename(image_files[current_index[0]]).split("/")[-1][:-4] in done:
        current_index[0] += 1


    while current_index[0] < len(image_files):
        show_image(current_index[0])
        current_index[0] += 1

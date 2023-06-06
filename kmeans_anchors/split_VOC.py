import os
import random
import math

prefix_path = "F:/EmojiYoLo/kmeans_anchors/Annotations/"
folder_path = 'F:/EmojiYoLo/kmeans_anchors/Annotations'

def shuffle_files_in_folder(folder_path):
    file_list = os.listdir(folder_path)
    random.shuffle(file_list)

    # train_size= math.floor(len(file_list) * (3/5))
    # val_size= len(file_list) - train_size
    train_size= len(file_list)
    val_size= len(file_list)
    
    shuffled_paths = []
    
    for i, filename in enumerate(file_list, start=1):
        dst = os.path.join(prefix_path, filename)
        shuffled_paths.append(dst)
    
    with open("train.txt", "w") as f:
        f.write("\n".join(shuffled_paths[:train_size]))

shuffle_files_in_folder(folder_path)


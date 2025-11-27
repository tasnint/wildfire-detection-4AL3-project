import os

def count_images(folder):
    fire = len(os.listdir(os.path.join(folder, "fire")))
    nofire = len(os.listdir(os.path.join(folder, "nofire")))
    total = fire + nofire
    return fire, nofire, total

base = "../data"

train = count_images(os.path.join(base, "train"))
val = count_images(os.path.join(base, "val"))
test = count_images(os.path.join(base, "test"))

print("TRAIN:", train)
print("VAL:", val)
print("TEST:", test)
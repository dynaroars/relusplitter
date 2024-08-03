import numpy as np
from PIL import Image
from pathlib import Path
from vnnlib.compat import read_vnnlib_simple



def label2onehot(label, num_classes):
    onehot = np.zeros(num_classes)
    onehot[label] = 1
    return onehot

def img2vnnlib_cifar10(img, label, eps, fp, info=""):
    # works for classificattion only
    # expect the label to be in one-hot encoding

    img = img.flatten().tolist()
    img = [(i-eps, i+eps) for i in img]


    s = f"; {info}\n"
    for i, _ in enumerate(img):
        s += f"(declare-const X_{i} Real)\n"

    for i, l in enumerate(label):
        s += f"(declare-const Y_{i} Real)\n"

    for i, (l, u) in enumerate(img):
        s += f"(assert (<= X_{i} {u}))\n"
        s += f"(assert (>= X_{i} {l}))\n"


    label_idx = np.argmax(label)
    s += f"(assert (or \n"
    for i, l in enumerate(label):
        if i == label_idx:
            pass
        else:
            s += f"(and (>= Y_{i} Y_{label_idx}))\n"
    s += "))\n"

    with open(fp, 'w') as f:
        f.write(s)

    return fp
    

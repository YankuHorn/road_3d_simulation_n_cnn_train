import numpy as np
import matplotlib.pyplot as plt


def type2color(type):
    color = [0, 0, 0]
    if type == 'solid':
        color = [0, 255, 255]
    elif type == 'dashed':
        color = [255, 255, 0]
    elif type == 'vehicle':
        color = [255, 0, 255]
    else:
        print("no color for this type yet:", type)
    return color

import copy

def swap(x1, x2):
    return x2, x1

def draw_line2(img, r1, c1, r2, c2, clr, width=5):

    if abs(r1 - r2) > abs(c1 - c2):

        # a = (c1-c2)/(float(r1-r2))
        # b = c1 - a * r1
        if r1 > r2:
            r1, r2 = swap(r1, r2)
        for r in range(r1, r2):
            c = int((c1-c2)/(float(r1-r2)) * r + c1 - ((c1-c2)/(float(r1-r2))) * r1)
            if (r > 0) and (c > 0) and (r < img.shape[0]) and (c < img.shape[1]):
                img[r, c - width // 2: c + width // 2] = clr
    else:
        # a = (r1-r2)/(float(c1-c2))
        # b = c1 - a * r1
        if c1 > c2:
            c1, c2 = swap(c1, c2)
        for c in range(c1, c2):
            r = int((r1 - r2) / (float(c1 - c2)) * c + r1 - ((r1 - r2) / (float(c1 - c2))) * c1)
            if (r1 > 0) and (c1 - 2 > 0) and (r < img.shape[0]) and (c + 3 < img.shape[1]):
                img[r - width // 2: r + width // 2, c] = clr
            # img[r, c] = clr
    return img


def draw_rect(img, r1, c1, r2, c2, clr, fill_clr=None, width=15):

    img = draw_line2(img, r1, c1, r1, c2, clr, width)
    img = draw_line2(img, r1, c1, r2, c1, clr, width)

    img = draw_line2(img, r1, c2, r2, c2, clr, width)
    img = draw_line2(img, r2, c1, r2, c2, clr, width)
    if fill_clr is not None:
        if r1 > r2:
            r1, r2 = swap(r1, r2)
        if c1 > c2:
            c1, c2 = swap(c1, c2)
        img[r1+width//2:r2-width//2, c1+width//2:c2-width//2] = fill_clr
    return img


if __name__ == "__main__":
    img = np.zeros((320, 480, 3), dtype=np.int16)
    img2 = draw_line2(img, r1=101, c1=20, r2=20, c2=100, clr=[254, 0, 254])
    img3 = draw_rect(img2, r1=201, c1=220, r2=280, c2=300, clr=[0, 0, 254], fill_clr=[0, 254, 0])
    plt.imshow(img2)
    plt.show()
    print("for the b")
# ========================================
# Found in
# Python Imaging Library ihttps://stackoverflow.com/questions/12638790/drawing-a-rectangle-inside-a-2d-numpy-array

# ========================================
# import Image
# import ImageDraw
#
#
# def get_rect(x, y, width, height, angle):
#     rect = np.array([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])
#     theta = (np.pi / 180.0) * angle
#     R = np.array([[np.cos(theta), -np.sin(theta)],
#                   [np.sin(theta), np.cos(theta)]])
#     offset = np.array([x, y])
#     transformed_rect = np.dot(rect, R) + offset
#     return transformed_rect
#
#
# def get_data():
#     """Make an array for the demonstration."""
#     X, Y = np.meshgrid(np.linspace(0, np.pi, 512), np.linspace(0, 2, 512))
#     z = (np.sin(X) + np.cos(Y)) ** 2 + 0.25
#     data = (255 * (z / z.max())).astype(int)
#     return data
#
#
# if __name__ == "__main__":
#     data = get_data()
#
#     # Convert the numpy array to an Image object.
#     img = Image.fromarray(data)
#
#     # Draw a rotated rectangle on the image.
#     draw = ImageDraw.Draw(img)
#     rect = get_rect(x=120, y=80, width=100, height=40, angle=30.0)
#     draw.polygon([tuple(p) for p in rect], fill=0)
#     # Convert the Image data to a numpy array.
#     new_data = np.asarray(img)
#
#     # Display the result using matplotlib.  (`img.show()` could also be used.)
#     plt.imshow(new_data, cmap=plt.cm.gray)
#     plt.show()
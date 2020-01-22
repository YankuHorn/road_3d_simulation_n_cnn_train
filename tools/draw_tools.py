import numpy as np
import matplotlib.pyplot as plt

def idx2type(idx):
    if idx == 3:
        type = 'solid'
    elif idx == 4:
        type = 'dashed'
    elif idx == 8:
        type = 'vehicle'
    else:
        print("no type for this idx yet:", idx)
    return type


def type2idx(type):

    idx = None
    if type == 'solid':
        idx = 3
    elif type == 'dashed':
        idx = 4
    elif type == 'vehicle':
        idx = 8
    else:
        print("no index for this type yet:", type)
    return idx

def draw_horizon_cross(img, x_center, y_center, cam_roll=0, clr=[0, 0, 255]):
    left_of_x_center_r = round(y_center - 30 * np.math.sin(cam_roll))
    left_of_x_center_c = round(x_center - 30)
    right_of_x_center_c = round(x_center + 30)
    right_of_x_center_r = round(y_center + 30 * np.math.sin(cam_roll))
    draw_line2(img, left_of_x_center_r, left_of_x_center_c,
               right_of_x_center_r, right_of_x_center_c, clr=clr, width=2)
    top_y_center_r = round(y_center + 10)
    top_y_center_c = round(x_center)
    bot_y_center_r = round(y_center - 10)
    bot_y_center_c = round(x_center)
    draw_line2(img, top_y_center_r, top_y_center_c,
               bot_y_center_r, bot_y_center_c, clr=clr, width=5)
    return img


def seg_img2clr_img(seg_img):

    clr_image = np.zeros(shape=(seg_img.shape[0], seg_img.shape[1], 3), dtype=np.uint8)

    solid_indices = np.where(seg_img[:, :] == type_name2type_idx('solid'))
    dashed_indices = np.where(seg_img[:, :] == type_name2type_idx('dashed'))
    vcl_indices = np.where(seg_img[:, :] == type_name2type_idx('vehicle'))
    clr_image[solid_indices[0], solid_indices[1]] = type_name2color('solid')
    clr_image[dashed_indices[0], dashed_indices[1]] = type_name2color('dashed')
    clr_image[vcl_indices[0], vcl_indices[1]] = type_name2color('vehicle')
    return clr_image


def keep_indices_in_seg_img(seg_img, indices_list):

    img2keep = np.zeros(shape=seg_img.shape, dtype=np.uint8)
    for obj_idx in indices_list:
        img_indices = np.where(seg_img[:, :] == obj_idx)
        img2keep[img_indices[0], img_indices[1]] = obj_idx

    return img2keep


def type_name2type_idx(type):
    """
    # pay attention - the first channel ('R', indexed 0) should be unique per type!!!
    """

    type_idx = None
    if type == 'solid':
        type_idx = 3
    elif type == 'dashed':
        type_idx = 4
    elif type == 'vehicle':
        type_idx = 8
    else:
        print("no color for this type yet:", type)
    return type_idx


def type_name2color(type):
    """
    # pay attention - the first channel ('R', indexed 0) should be unique per type!!!
    """

    color = [0, 0, 0]
    if type == 'solid':
        color = [1, 255, 255]
    elif type == 'dashed':
        color = [253, 255, 0]
    elif type == 'vehicle':
        color = [252, 0, 255]
    else:
        print("no color for this type yet:", type)
    return color


def type_idx2color(type):
    """
    # pay attention - the first channel ('R', indexed 0) should be unique per type!!!
    """

    color = [0, 0, 0]
    if type == 3:
        color = [1, 255, 255]
    elif type == 4:
        color = [253, 255, 0]
    elif type == 8:
        color = [252, 0, 255]
    else:
        print("no color for this type yet:", type)
    return color


def color2type_name(color):
    """
    # pay attention - the first channel ('R', indexed 0) should be unique per type!!!
    """

    type = None
    if np.all(color == [1, 255, 255]):
        type = 'solid'
    elif np.all(color == [253, 255, 0]):
        type = 'dashed'
    elif np.all(color == [252, 0, 255]):
        type = 'vehicle'
    else:
        print("no type for this color yet:", color)
    return type


def color2type_idx(color):
    """
    # pay attention - the first channel ('R', indexed 0) should be unique per type!!!
    """

    type = None
    if np.all(color == [1, 255, 255]):
        type = 3
    elif np.all(color == [253, 255, 0]):
        type = 4
    elif np.all(color == [252, 0, 255]):
        type = 8
    else:
        print("no type for this color yet:", color)
    return type

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
import numpy as np
from skimage.draw import line_aa
import torch


def add_line(img, start_pos, end_pos):
    """
    Add a white line to an RGB image
    """
    # Get the array indices and values corresponding to the line
    rr, cc, val = line_aa(*start_pos, *end_pos)
    # Copy val to two extra channels for use in RGB images
    val3d = np.squeeze(np.dstack([val]*3))
    # Make the line white
    img[rr, cc, :] = val3d * 255
    return img

def arrow_images(size=100):
    """
    Creates images in the form numpy arrays of shape (size,size,3) that display a black square with
    a white arrow pointing left, left+up, up, up+right, right, right+down, down and down+left).
    """
    # Fill background to all black
    background = np.zeros([size, size, 3], dtype=np.uint8)
    # Parameter for the size of the arrow head
    offset = size // 10
    # Give the position in the grid of the three arrowhead endpoints based on the action direction.
    # Note that we need to indent by 1 pixel otherwise we get index errors
    size_diag = int((size//2) * ((size//2) / ((size//2)**2 + (size//2)**2)**0.5))

    arrow_points = {
        "down": {
            "tip": (size-1, size//2), "left": (size-1-offset, size//2-offset),
            "right": (size-1-offset, size//2+offset)
        },
        "downleft": {
            "tip": (size//2 + size_diag, size//2 - size_diag), "left": (size//2 + size_diag-int(offset*1.41), size//2 - size_diag),
            "right": (size//2 + size_diag, size//2 - size_diag+int(offset*1.41))
        },
        "left": {
            "tip": (size//2, 1), "left": (size//2+offset, 1+offset),
            "right": (size//2-offset, 1+offset)
        },
        "upleft": {
            "tip": (size//2 - size_diag, size//2 - size_diag), "left": (size//2 - size_diag, size//2 - size_diag+int(offset*1.41)),
            "right": (size//2 - size_diag+int(offset*1.41), size//2 - size_diag)
        },
        "up": {
            "tip": (1, size//2), "left": (1+offset, size//2-offset),
            "right": (1+offset, size//2+offset)
        },
        "upright": {
            "tip": (size//2 - size_diag, size//2 + size_diag), "left": (size//2 - size_diag, size//2 + size_diag-int(offset*1.41)),
            "right": (size//2 - size_diag+int(offset*1.41), size//2 + size_diag)
        },
        "right": {
            "tip": (size//2, size-1), "left": (size//2-offset, size-1-offset),
            "right": (size//2+offset, size-1-offset)
        },
        "downright": {
            "tip": (size//2 + size_diag, size//2 + size_diag), "left": (size//2 + size_diag, size//2 + size_diag-int(offset*1.41)),
            "right": (size//2 + size_diag-int(offset*1.41), size//2 + size_diag)
        },
    }
    imgs = {}
    for direction in arrow_points:
        img = background.copy()
        img = add_line(img, (size//2, size//2), arrow_points[direction]["tip"])
        img = add_line(img, arrow_points[direction]["tip"], arrow_points[direction]["left"])
        img = add_line(img, arrow_points[direction]["tip"], arrow_points[direction]["right"])
        imgs[direction] = img

    # Default image needed for procgen actions that don't map to directions
    imgs[None] = background.copy()
    return imgs

def overlay_actions(obs, actions, size=16):
    """
    Grab the image corresponding to the action at each timestep, at overlay it on the obs array
    """
    arrows = arrow_images(size)
    coinrun_actions = {0: 'downleft', 1: 'left', 2: 'upleft',
                       3: 'down', 4: None, 5: 'up',
                       6: 'downright', 7: 'right', 8: 'upright',
                       9: None, 10: None, 11: None,
                       12: None, 13: None, 14: None}

    for timestep in range(actions.shape[0]):
        action_str = coinrun_actions[actions[timestep]]
        action_img = arrows[action_str]
        # Overlay in the top right corner
        action_img_size = action_img.shape[0]
        half_ob_size = obs.shape[1] //2
        obs[timestep][:action_img_size, half_ob_size-action_img_size//2:half_ob_size+action_img_size//2] = action_img
    return obs

def overlay_box_var(im_seq, indicator_variable, left_right='left'):
    box_dim = 4
    t = im_seq.shape[0]
    b = im_seq.shape[1]
    c = im_seq.shape[-1]
    indicator_variable = torch.clamp(indicator_variable, 1, 0)
    box = torch.ones(t, box_dim, box_dim, 3, device=im_seq.device) * indicator_variable.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    box = box.squeeze()
    if left_right == 'left':
        im_seq[:, 0:box_dim, 0:box_dim, :] = 0.
        im_seq[:, 0:box_dim, 0:box_dim, :] = box
    elif left_right == 'right':
        im_seq[:, 0:box_dim, -box_dim-1:-1, :] = 0.
        im_seq[:, 0:box_dim, -box_dim-1:-1, :] = box
    return im_seq

if __name__ == "__main__":
    arrow_images()

from repvitsam.utils.transforms import ResizeLongestSide
import numpy as np
import torch
import cv2

def compute_logits_from_mask_torch(mask, eps=1e-6, expected_shape = (256, 256)):

    logits = torch.full_like(mask,eps, dtype=torch.float32)
    logits[mask == 1] = 1 - eps
    #logits[mask == 0] = eps
    #logits = torch.where(mask == 1, 1 - eps, eps, out=logits)
    logits = torch.logit(logits)

    # resize to the expected mask shape of SAM (256x256)
    
    if logits.shape == expected_shape:  # shape matches, do nothing
        pass

    elif logits.shape[-2] == logits.shape[-1]:  # shape is square
        trafo = ResizeLongestSide(expected_shape[0])
        logits = trafo.apply_image_torch(logits.unsqueeze(1))

    else:  # shape is not square
        # resize the longest side to expected shape
        trafo = ResizeLongestSide(expected_shape[0])
        logits = trafo.apply_image_torch(logits.unsqueeze(1))

        # pad the other side
        b,c, h, w = logits.shape
        padh = expected_shape[0] - h
        padw = expected_shape[1] - w
        # IMPORTANT: need to pad with zero, otherwise SAM doesn't understand the padding
        pad_width = (0,0,0, padh, 0, padw)
        logits = torch.nn.functional.pad(logits, pad_width, mode="constant", value=0)

    return logits

def compute_sam_box(sam_predictor, mask, original_size, margin=0.5):
    boxes = torch.stack([compute_bounding_box_torch(mask) for mask in mask.unsqueeze(0)]) # BxCxHxW
    boxes = sam_predictor.transform.apply_boxes_torch(boxes, original_size) # Bx4
    heights = boxes[:,2] - boxes[:,0]
    widths = boxes[:,3] - boxes[:,1]
    boxes[:,0] -= margin * heights
    boxes[:,2] += margin * heights
    boxes[:,1] -= margin * widths
    boxes[:,3] += margin * widths
    return boxes[0]


def queries_from_points(points, t=0, device=None):
    if torch.is_tensor(points):
        additional_points = torch.full((points.shape[0],1), t, device=points.device, dtype=torch.float)
        queries = torch.hstack((additional_points, points)).unsqueeze(0)
    else:
        additional_points = np.hstack((np.full((len(points),1),t),points))
        queries = torch.tensor(additional_points.reshape(1,-1,3), device=device, dtype=torch.float)
    return queries



def points_to_mask(points, shape, linetype=cv2.LINE_8):
    mask = np.zeros(shape, dtype=np.uint8)
    if points.shape[0] == 0: # no points
        return mask

    mask = cv2.fillPoly(mask, [(points).astype(np.int32)], 1, lineType=linetype)
    return mask

def mask_from_tracks(tracks, visibility, original_size):

    pred_tracks_np = tracks.cpu().numpy()
    pred_visibility_np = visibility.cpu().numpy()

    masks = np.array([points_to_mask(pred_tracks_np[0,i,pred_visibility_np[0,i]], original_size) for i in range(pred_tracks_np.shape[1])])
    masks = torch.tensor(masks, device=tracks.device)
    return masks


def compute_largest_contour(binary_mask):
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    largest_contour = max(contours, key=lambda x: len(x))
    return np.array(largest_contour)[:,0,:]

def interp_1d(x, size=100):
    # basic linear interpolation
    if size is None:
        return x
    a, b = 0, 1
    dx = (b-a)/len(x)
    d_x = np.diff(x, axis=0)
    i = np.linspace(a,b,size)
    return x[(i/dx -1).astype(int)] + d_x[(i/dx-2).astype(int)] * (i % dx)[...,None]

def compute_bounding_box(mask):
    """
    Compute the bounding box of a binary mask.

    Args:
        mask (numpy.ndarray): The binary mask.

    Returns:
        tuple: Bounding box coordinates (top, left, bottom, right).
    """
    nonzero_indices = np.argwhere(mask > 0)
    if len(nonzero_indices) == 0:
        return -1, -1, -1, -1
    top = np.min(nonzero_indices[:, 1])
    left = np.min(nonzero_indices[:, 0])
    bottom = np.max(nonzero_indices[:, 1])
    right = np.max(nonzero_indices[:, 0])
    return top, left, bottom, right

def compute_bounding_box_torch(mask):
    """
    Compute the bounding box of a binary mask.

    Args:
        mask (numpy.ndarray): The binary mask.

    Returns:
        tuple: Bounding box coordinates (top, left, bottom, right).
    """
    nonzero_indices = torch.argwhere(mask > 0)
    if len(nonzero_indices) == 0:
        return torch.tensor((-1, -1, -1, -1), device=mask.device)
    top = torch.min(nonzero_indices[:, 1])
    left = torch.min(nonzero_indices[:, 0])
    bottom = torch.max(nonzero_indices[:, 1])
    right = torch.max(nonzero_indices[:, 0])
    return torch.tensor((top, left, bottom, right), device=mask.device)

def compute_dice_score_torch(mask1, mask2):
    """
    Calculate the Dice coefficient between two masks
    shape (b, x, y)
    """
    assert mask1.shape == mask2.shape, "Mask shapes do not match"
    non_batch_axis = (-2,-1)
    
    # Calculate intersection and union
    intersection = torch.logical_and(mask1, mask2).sum(axis=non_batch_axis).float()
    union = (torch.count_nonzero(mask1, axis=non_batch_axis).float() + torch.count_nonzero(mask2, axis=non_batch_axis)).float()
    
    # Add a small epsilon to the union to avoid division by zero
    epsilon = 1e-9
    union += epsilon

    # Calculate the Dice coefficient
    dice_score = torch.where(union < 1, 0.0, (2.0 * intersection) / union)
    return dice_score
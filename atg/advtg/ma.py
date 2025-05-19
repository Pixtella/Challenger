import torch
import math
from typing import Optional
from pyquaternion import Quaternion
import numpy as np
from scipy.interpolate import interp1d



def state_se2_tensor_to_transform_matrix(
    input_data: torch.Tensor, precision: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Transforms a state of the form [x, y, heading] into a 3x3 transform matrix.
    :param input_data: the input data as a 3-d tensor.
    :return: The output 3x3 transformation matrix.
    """

    if precision is None:
        precision = input_data.dtype

    x: float = float(input_data[0].item())
    y: float = float(input_data[1].item())
    h: float = float(input_data[2].item())

    cosine: float = math.cos(h)
    sine: float = math.sin(h)

    return torch.tensor([[cosine, -sine, x], [sine, cosine, y], [0.0, 0.0, 1.0]], dtype=precision, device=input_data.device)

def state_se2_tensor_to_transform_matrix_batch(
    input_data: torch.Tensor, precision: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Transforms a tensor of states of the form Nx3 (x, y, heading) into a Nx3x3 transform tensor.
    :param input_data: the input data as a Nx3 tensor.
    :param precision: The precision with which to create the output tensor. If None, then it will be inferred from the input tensor.
    :return: The output Nx3x3 batch transformation tensor.
    """

    if precision is None:
        precision = input_data.dtype

    # Transform the incoming coordinates so transformation can be done with a simple matrix multiply.
    #
    # [x1, y1, phi1]  => [x1, y1, cos1, sin1, 1]
    # [x2, y2, phi2]     [x2, y2, cos2, sin2, 1]
    # ...          ...
    # [xn, yn, phiN]     [xn, yn, cosN, sinN, 1]
    processed_input = torch.column_stack(
        (
            input_data[:, 0],
            input_data[:, 1],
            torch.cos(input_data[:, 2]),
            torch.sin(input_data[:, 2]),
            torch.ones_like(input_data[:, 0], dtype=precision),
        )
    )

    # See below for reshaping example
    reshaping_tensor = torch.tensor(
        [
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, -1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=precision,
        device=input_data.device,
    )
    # Builds the transform matrix
    # First computes the components of each transform as rows of a Nx9 tensor, and then reshapes to a Nx3x3 tensor
    # Below is outlined how the Nx9 representation looks like (s1 and c1 are cos1 and sin1)
    # [x1, y1, c1, s1, 1]  => [c1, -s1, x1, s1, c1, y1, 0, 0, 1]  =>  [[c1, -s1, x1], [s1, c1, y1], [0, 0, 1]]
    # [x2, y2, c2, s2, 1]     [c2, -s2, x2, s2, c2, y2, 0, 0, 1]  =>  [[c2, -s2, x2], [s2, c2, y2], [0, 0, 1]]
    # ...          ...
    # [xn, yn, cN, sN, 1]     [cN, -sN, xN, sN, cN, yN, 0, 0, 1]
    return (processed_input @ reshaping_tensor).reshape(-1, 3, 3)

def transform_matrix_to_state_se2_tensor_batch(input_data: torch.Tensor) -> torch.Tensor:
    """
    Converts a Nx3x3 batch transformation matrix into a Nx3 tensor of [x, y, heading] rows.
    :param input_data: The 3x3 transformation matrix.
    :return: The converted tensor.
    """

    # Picks the entries, the third column will be overwritten with the headings [x, y, _]
    first_columns = input_data[:, :, 0].reshape(-1, 3)
    angles = torch.atan2(first_columns[:, 1], first_columns[:, 0])

    result = input_data[:, :, 2]
    result[:, 2] = angles

    return result



def global_state_se2_tensor_to_local(
    global_states: torch.Tensor, local_state: torch.Tensor, precision: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Transforms the StateSE2 in tensor from to the frame of reference in local_frame.

    :param global_states: A tensor of Nx3, where the columns are [x, y, heading].
    :param local_state: A tensor of [x, y, h] of the frame to which to transform.
    :param precision: The precision with which to allocate the intermediate tensors. If None, then it will be inferred from the input precisions.
    :return: The transformed coordinates.
    """

    if precision is None:
        if global_states.dtype != local_state.dtype:
            raise ValueError("Mixed datatypes provided to coordinates_to_local_frame without precision specifier.")
        precision = global_states.dtype

    local_xform = state_se2_tensor_to_transform_matrix(local_state, precision=precision)
    local_xform_inv = torch.linalg.inv(local_xform)

    transforms = state_se2_tensor_to_transform_matrix_batch(global_states, precision=precision)

    transforms = torch.matmul(local_xform_inv, transforms)

    output = transform_matrix_to_state_se2_tensor_batch(transforms)

    return output


def convert_to_global(ref, points):
    points = points.copy()

    rot = np.array([
        [np.cos(ref[2]), -np.sin(ref[2])],
        [np.sin(ref[2]),  np.cos(ref[2])]
    ])

    original_shape = points.shape

    points_dim = points.shape[-1]
    assert points_dim in (2,3), f'Received invalid feature dimension in convert_to_local, got {points_dim} expected 2 or 3'
    points = points.reshape(-1,points_dim)
    points[:,:2] = points[:,:2] @ rot.T
    points += ref[:points_dim]

    points = points.reshape(original_shape)

    return points

def convert_to_local(ref, points):
    points = points.copy()

    rot = np.array([
        [np.cos(ref[2]), -np.sin(ref[2])],
        [np.sin(ref[2]),  np.cos(ref[2])]
    ])

    original_shape = points.shape

    points_dim = points.shape[-1]
    assert points_dim in (2,3), f'Received invalid feature dimension in convert_to_local, got {points_dim} expected 2 or 3'
    points = points.reshape(-1,points_dim)
    points -= ref[:points_dim]
    points[:,:2] = points[:,:2] @ rot

    points = points.reshape(original_shape)

    return points

def qua_list_to_mat(qua: list):
    qua = Quaternion(qua)
    return torch.tensor(qua.rotation_matrix)


def interpolate_trajectory(trajectories, npoints):
    old_xs = np.arange(trajectories.shape[1])
    new_xs = np.linspace(0, trajectories.shape[1]-1, npoints)
    trajectories = interp1d(old_xs, trajectories, axis=1)(new_xs)
    return trajectories

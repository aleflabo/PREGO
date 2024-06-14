import random
import torch
import numpy as np


def dict2tensor(d):
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            d[key] = torch.from_numpy(np.array(value))
    return d


def Time2FrameNumber(t, ori_fps, fps=10):
    """function to convert segment annotations given in seconds to frame numbers
    input:
        ori_fps: is the original fps of the video
        fps: is the fps that we are using to extract frames from the video
        num_frames: is the number of frames in the video (under fps)
        t: is the time (in seconds) that we want to convert to frame number
    output:
        numf: the frame number corresponding to the time t of a video encoded at fps
    """
    ori2fps_ratio = int(ori_fps / fps)
    ori_numf = t * ori_fps
    numf = int(ori_numf / ori2fps_ratio)
    return numf


def subsample(sample, rate=2):
    K = sample["step_features"].shape[0]
    N = sample["frame_features"].shape[0]
    annotations = torch.zeros([K, N], dtype=int)
    for si in range(K):
        start_i = sample["step_starts"][si]
        end_i = sample["step_ends"][si]
        annotations[si, start_i : (end_i + 1)] = sample["step_ids"][si]

    keep = torch.zeros(N, dtype=bool)
    keep[random.randint(0, rate - 1) :: rate] = True
    new_N = keep.to(int).sum()

    # ensuring that we didn't filter out entire steps
    reserved_points = []
    for si in range(K):
        if annotations[si][keep].sum() == 0:
            print("happened")
            points_of_interest = torch.range(0, N - 1)[annotations[si] > 0].cpu()
            step_point = random.randint(points_of_interest.min(), points_of_interest.max())

            # find a point to switch the sample point with
            left_end, right_end = step_point, step_point
            while not keep[left_end] and not keep[right_end]:
                left_end = max(left_end - 1, 0)
                right_end = min(right_end + 1, N - 1)
                while left_end in reserved_points:
                    left_end = max(left_end - 1, 0)
                while right_end in reserved_points:
                    right_end = min(right_end + 1, N - 1)

            swappoint = left_end if keep[left_end] else right_end
            reserved_points.append(swappoint)
            keep[swappoint] = False
            keep[step_point] = True

    # writing filtered out features and gt into sample dict
    step_starts = torch.zeros_like(sample["step_starts"])
    step_ends = torch.zeros_like(sample["step_starts"])
    subsampled_annotations = annotations[:, keep]
    for si in range(K):
        step_scope = torch.range(0, new_N - 1)[subsampled_annotations[si] > 0]
        step_starts[si] = step_scope.min()
        step_ends[si] = step_scope.max()
    sample["step_starts"] = step_starts
    sample["step_ends"] = step_ends
    sample["step_starts_sec"] = step_starts * 3.2 * rate
    sample["step_ends_sec"] = step_ends * 3.2 * rate
    sample["frame_features"] = sample["frame_features"][keep]
    sample["num_frames"] = sample["frame_features"].shape[0]
    return sample


def sample_to_device(sample, device):
    for k, v in sample.items():
        try:
            sample[k] = v.to(device)
        except:
            pass
    return sample

import yaml
import torch
import os
from glob import glob
import numpy as np


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def load_yaml(filename):
    with open(filename, 'r') as stream:
        try:
            dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return dict


def save_checkpoint(state, checkpoint_dir, epoch, n_ckpt=10):
    torch.save(state, os.path.join(checkpoint_dir, "epoch{:0>4d}.pth.tar".format(epoch)))
    if epoch - n_ckpt >= 0:
        oldest_ckpt = os.path.join(checkpoint_dir, "epoch{:0>4d}.pth.tar".format(epoch - n_ckpt)) 
        if os.path.isfile(oldest_ckpt):
            os.remove(oldest_ckpt)

def get_last_checkpoint(checkpoint_dir):
    all_ckpt = glob(os.path.join(checkpoint_dir, 'epoch*.pth.tar'))
    if all_ckpt:
        all_ckpt = sorted(all_ckpt)
        return all_ckpt[-1]
    else:
        return ''

def Time2FrameNumber(t, ori_fps, fps=10):
    """ function to convert segment annotations given in seconds to frame numbers
        input:
            ori_fps: is the original fps of the video
            fps: is the fps that we are using to extract frames from the video
            num_frames: is the number of frames in the video (under fps)
            t: is the time (in seconds) that we want to convert to frame number 
        output: 
            numf: the frame number corresponding to the time t of a video encoded at fps
    """
    ori2fps_ratio = int(ori_fps/fps)
    ori_numf = t*ori_fps
    numf = int(ori_numf / ori2fps_ratio)
    return numf

def RemoveDuplicates(a):
    """ function to remove duplicate steps """
    filtered = []
    keep_ids = []
    nums = a.shape[0]
    for i in range(nums):
        if a[i] in filtered:
            continue
        else:
            filtered.append(a[i])
            keep_ids.append(i)
    filtered = torch.stack(filtered)
    keep_ids = torch.tensor(keep_ids)
    return filtered, keep_ids

def MergeConsec(a, a_st, a_ed):
    """ merge consecutibe steps"""
    
    # find consecutive steps
    a_old = 10000
    merge_ids = []
    mids = []
    merge_st = []
    mst = []
    for i in range(a.shape[0]):
        if a[i] == a_old:
            mst.append(a[i-1])
            mst.append(a[i])
            mids.append(i-1)
            mids.append(i)
        else:
            merge_ids.append(mids)
            merge_st.append(mst)
            mids = []
            mst = []
        a_old = a[i]
        if i == a.shape[0]-1:
            merge_ids.append(mids)
            merge_st.append(mst)
    # remove empty entries from list
    merge_ids = list(filter(None, merge_ids))
    merge_st = list(filter(None, merge_st))
    # merge consec start and end times
    for i in range(len(merge_ids)):
        a_st[merge_ids[i][-1]] = a_st[merge_ids[i][0]]
        a_ed[merge_ids[i][0]] = a_ed[merge_ids[i][-1]]
    
    return a_st, a_ed

def VidList2Batch(samples, VID_LEN=224):
    """ create a batch of videos of the same size from input sequences """
    # create data needed for training
    vids = []
    batch_size = len(samples)
    for b in range(batch_size):
        numf = samples[b]['frame_features'].shape[0]
        unpadded_vid = samples[b]['frame_features'].T
        # if video is shorter than desired length ==> PAD
        if numf < VID_LEN:
            pad = torch.nn.ConstantPad1d((0, VID_LEN-numf), 0)
            vids.append(pad(unpadded_vid))
        # if video is longer than desired length ==> STRIDED SAMPLING
        elif numf > VID_LEN:
            stride = int(numf//VID_LEN)
            pad = unpadded_vid[:,::stride]
            vids.append(pad[:,:VID_LEN])
        else:
            pad = unpadded_vid
            vids.append(pad)
    vids = torch.stack(vids, dim=0)

    return vids

def Steps2Batch(steps, num_steps):
    """ create a list of lists of the steps """
    st = 0
    batched_steps = []
    for i in range(len(num_steps)):
        ed = st+ num_steps[i]
        batched_steps.append(steps[st:ed,:])
        st = ed
    return batched_steps

def cosine_sim(x, z):
    cos_sim_fn = torch.nn.CosineSimilarity(dim=1)
    return cos_sim_fn(x[..., None], z.T[None, ...])

def neg_l2_dist(x, z, device):
    """Computes pairwise distances between all rows of x and z."""
    #return -1*torch.cdist(x,z,2)
    
    norm1 = torch.sum(torch.square(x), dim=-1)
    norm1 = torch.reshape(norm1, [-1, 1])
    norm2 = torch.sum(torch.square(z), dim=-1)
    norm2 = torch.reshape(norm2, [1, -1])

    # Max to ensure matmul doesn't produce anything negative due to floating
    # point approximations.
    dist = -1*torch.maximum(
    norm1 + norm2 - torch.tensor([2.0]).to(device) * torch.matmul(x, z.T), torch.tensor([0.0]).to(device))
    
    return dist
    

def linear_sim(x, z):
    return x @ z.T

def whitening(frame_features, step_features, stats_file):
    """ do data whitening """
    # load dataset stats
    data_stats = np.load(stats_file, allow_pickle=True)
    vis_mean = data_stats.item().get('vis_mean')
    vis_std = data_stats.item().get('vis_std')
    lang_mean = data_stats.item().get('lang_mean')
    lang_std = data_stats.item().get('lang_std')
    
    frame_features = (frame_features - vis_mean) / vis_std
    step_features = (step_features - lang_mean) / lang_std
    
    return frame_features, step_features

def unique_softmax(sim, labels, gamma=1, dim=0):
    assert sim.shape[0] == labels.shape[0]
    labels = labels.detach().cpu().numpy()
    unique_labels, unique_index, unique_inverse_index = np.unique(
        labels, return_index=True, return_inverse=True)
    unique_sim = sim[unique_index]
    unique_softmax_sim = torch.nn.functional.softmax(unique_sim / gamma, dim=0)
    softmax_sim = unique_softmax_sim[unique_inverse_index]
    return softmax_sim

def framewise_accuracy(frame_assignment, sample, use_unlabeled=False):
    """ calculate framewise accuracy as done in COIN """
    # convert start and end times into clip-level labels
    num_steps = sample['num_steps'].numpy()
    num_frames = sample['num_frames'].numpy()
    # non-step frames/clips are assigned label = -1
    gt_assignment = -np.ones((num_frames,), dtype=np.int32)
    # convert start and end times to clip/frame -wise labels
    for s in range(num_steps):
        st_ed = np.arange(sample['step_starts'][s],sample['step_ends'][s]+1)
        gt_assignment[st_ed] = s #sample['step_ids'][s]
    
    # to discount unlabeled frames in gt
    if not use_unlabeled:
        unlabled = np.count_nonzero(gt_assignment == -1)
        num_frames = num_frames - unlabled
        fa = np.logical_and(frame_assignment == gt_assignment, gt_assignment!=-1).sum()
    else:
        fa = np.count_nonzero((frame_assignment == gt_assignment))
    # framewise accuracy
    fa = fa / num_frames if num_frames!=0 else 0
    
    return fa
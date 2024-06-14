import os.path as osp
import pickle

import numpy as np
import torch
import torch.utils.data as data
from ipdb import set_trace


class TRNTHUMOSDataLayer(data.Dataset):
    def __init__(self, args, phase="train"):
        # args.data_root = '/home/aleflabo/ego_procedural/OadTR/data/assembly/old_split' # ! Loki path
        # args.data_root = "/home/aleflabo/ego_procedural/OadTR/data/assembly/old_split_train+val/"  # ! Loki path
        # args.data_root = '/home/scofanol/data/EgoProcel/old_split' #!'/home/aleflabo/ego_procedural/OadTR/data/assembly/train' # DGX Path
        args.data_root = "/home/aleflabo/ego_procedural/OadTR/data/assembly/OadTR_assembly/train+val_allMistakes_onlyThis/"  # ! Loki path to allMistakes with only_this

        self.pickle_root = args.data_root
        self.data_root = args.data_root  # data/THUMOS
        self.pickle_root = args.data_root
        self.data_root = args.data_root  # data/THUMOS
        self.sessions = getattr(args, phase + "_session_set")  # video name
        self.enc_steps = args.enc_layers
        self.numclass = args.numclass
        self.dec_steps = args.query_num
        self.training = phase == "train"
        if self.training:
            self.sessions = self.sessions[:3]
        if args.debug:
            self.sessions = self.sessions[:3]

        self.feature_pretrain = (
            args.feature
        )  # 'Anet2016_feature'   # IncepV3_feature  Anet2016_feature
        self.inputs = []
        self.dataset = "assembly"
        self.add_labels = args.add_labels
        self.add_labels_mean = args.add_labels_mean
        self.add_labels_seq = args.add_labels_seq

        if self.dataset == "assembly":
            self.subnet = "train" if self.training else "test"
        else:
            self.subnet = "val" if self.training else "test"
        self.resize = args.resize_feature
        if self.resize:
            target_all = pickle.load(
                open(
                    osp.join(
                        self.pickle_root,
                        self.dataset + "_" + self.subnet + "_anno_resize.pickle",
                    ),
                    "rb",
                )
            )
        else:
            target_all = pickle.load(
                open(
                    osp.join(
                        self.pickle_root,
                        self.dataset + "_" + self.subnet + "_anno.pickle",
                    ),
                    "rb",
                )
            )
        for session in self.sessions:  # 改
            # target = np.load(osp.join(self.data_root, 'target', session+'.npy'))  # thumos_val_anno.pickle
            target = target_all[session]["anno"]
            seed = np.random.randint(self.enc_steps) if self.training else 0

            # TODO: remove
            seed = 9

            for start, end in zip(
                range(
                    seed, target.shape[0], 1
                ),  # self.enc_steps #! Changed to 64 for test
                range(seed + self.enc_steps, target.shape[0] - self.dec_steps, 1),
            ):  #! Changed to 64 for test
                enc_target = target[start:end]
                # dec_target = self.get_dec_target(target[start:end + self.dec_steps])
                dec_target = target[end : end + self.dec_steps]
                distance_target, class_h_target = self.get_distance_target(
                    target[start:end]
                )

                if self.subnet == "train" and self.add_labels:
                    # !Take the additional class before the start of the sequence only if it is different, otherwise take the one before
                    if start == 0:
                        # all the classes are 0
                        additional_class = np.zeros(self.numclass)
                    else:
                        if self.add_labels_mean == True:
                            # ! Uncomment for mean
                            additional_class = [
                                target[start - 1 - i] for i in range(start, 0, -1)
                            ]
                            additional_class = np.mean(additional_class, axis=0)

                        elif self.add_labels_seq == True:
                            window_size = 10
                            # if start - window_size < 0 then pad the additiona class until the start of the sequence
                            if start - window_size < 0:
                                additional_class = [
                                    target[start - 1 - i] for i in range(start, 0, -1)
                                ]
                                # pad the rest to cover the window size
                                additional_class = np.pad(
                                    additional_class,
                                    ((0, window_size - start), (0, 0)),
                                    "constant",
                                    constant_values=0,
                                )
                                # revert the order so that the non zer elements are at the end
                                additional_class = additional_class[::-1]

                            else:
                                additional_class = [
                                    target[start - 1 - i]
                                    for i in range(start, start - window_size, -1)
                                ]
                                # convert it to numpy array
                                additional_class = np.array(additional_class)

                        # ! Uncomment for last token only
                        # additional_class = target[start - 1]

                    self.inputs.append(
                        [
                            session,
                            start,
                            end,
                            enc_target,
                            distance_target,
                            class_h_target,
                            dec_target,
                            additional_class,
                        ]
                    )
                else:
                    self.inputs.append(
                        [
                            session,
                            start,
                            end,
                            enc_target,
                            distance_target,
                            class_h_target,
                            dec_target,
                        ]
                    )

        if "V3" in self.feature_pretrain:
            if osp.exists(
                osp.join(
                    self.pickle_root,
                    "thumos_all_feature_{}_V3.pickle".format(self.subnet),
                )
            ):
                self.feature_All = pickle.load(
                    open(
                        osp.join(
                            self.pickle_root,
                            "thumos_all_feature_{}_V3.pickle".format(self.subnet),
                        ),
                        "rb",
                    )
                )
                print("load thumos_all_feature_{}_V3.pickle !".format(self.subnet))
            else:
                self.feature_All = {}
                for session in self.sessions:
                    self.feature_All[session] = {}
                    self.feature_All[session]["rgb"] = np.load(
                        osp.join(
                            self.data_root, self.feature_pretrain, session + "_rgb.npy"
                        )
                    )
                    self.feature_All[session]["flow"] = np.load(
                        osp.join(
                            self.data_root, self.feature_pretrain, session + "_flow.npy"
                        )
                    )
                with open(
                    osp.join(
                        self.pickle_root,
                        "thumos_all_feature_{}_V3.pickle".format(self.subnet),
                    ),
                    "wb",
                ) as f:
                    pickle.dump(self.feature_All, f)
                print("dump thumos_all_feature_{}_V3.pickle !".format(self.subnet))
        elif "Anet2016_feature_v2" in self.feature_pretrain:
            if self.resize:
                if osp.exists(
                    osp.join(
                        self.pickle_root,
                        "{}_all_feature_{}_tsn_v2_resize.pickle".format(
                            self.dataset, self.subnet
                        ),
                    )
                ):
                    self.feature_All = pickle.load(
                        open(
                            osp.join(
                                self.pickle_root,
                                "{}_all_feature_{}_tsn_v2_resize.pickle".format(
                                    self.dataset, self.subnet
                                ),
                            ),
                            "rb",
                        )
                    )
                    print(
                        "load {}_all_feature_{}_tsn_v2_resize.pickle !".format(
                            self.dataset, self.subnet
                        )
                    )
            else:
                if osp.exists(
                    osp.join(
                        self.pickle_root,
                        "{}_all_feature_{}_tsn_v2.pickle".format(
                            self.dataset, self.subnet
                        ),
                    )
                ):
                    self.feature_All = pickle.load(
                        open(
                            osp.join(
                                self.pickle_root,
                                "{}_all_feature_{}_tsn_v2.pickle".format(
                                    self.dataset, self.subnet
                                ),
                            ),
                            "rb",
                        )
                    )
                    print(
                        "load {}_all_feature_{}_tsn_v2.pickle !".format(
                            self.dataset, self.subnet
                        )
                    )
                else:
                    self.feature_All = {}
                    for session in self.sessions:
                        self.feature_All[session] = {}
                        self.feature_All[session]["rgb"] = np.load(
                            osp.join(
                                self.data_root,
                                self.feature_pretrain,
                                session + "_rgb.npy",
                            )
                        )
                        self.feature_All[session]["flow"] = np.load(
                            osp.join(
                                self.data_root,
                                self.feature_pretrain,
                                session + "_flow.npy",
                            )
                        )
                    with open(
                        osp.join(
                            self.pickle_root,
                            "{}_all_feature_{}_tsn_v2.pickle".format(
                                self.dataset, self.subnet
                            ),
                        ),
                        "wb",
                    ) as f:
                        pickle.dump(self.feature_All, f)
                    print(
                        "dump {}_all_feature_{}_tsn_v2.pickle !".format(
                            self.dataset, self.subnet
                        )
                    )
        else:
            if osp.exists(
                osp.join(
                    self.pickle_root, "thumos_all_feature_{}.pickle".format(self.subnet)
                )
            ):
                self.feature_All = pickle.load(
                    open(
                        osp.join(
                            self.pickle_root,
                            "thumos_all_feature_{}.pickle".format(self.subnet),
                        ),
                        "rb",
                    )
                )
                print("load thumos_all_feature_{}.pickle !".format(self.subnet))
            else:
                self.feature_All = {}
                for session in self.sessions:
                    self.feature_All[session] = {}
                    self.feature_All[session]["rgb"] = np.load(
                        osp.join(
                            self.data_root, self.feature_pretrain, session + "_rgb.npy"
                        )
                    )
                    self.feature_All[session]["flow"] = np.load(
                        osp.join(
                            self.data_root, self.feature_pretrain, session + "_flow.npy"
                        )
                    )
                with open(
                    osp.join(
                        self.pickle_root,
                        "thumos_all_feature_{}.pickle".format(self.subnet),
                    ),
                    "wb",
                ) as f:
                    pickle.dump(self.feature_All, f)
                print("dump thumos_all_feature_{}.pickle !".format(self.subnet))

    def get_dec_target(self, target_vector):
        target_matrix = np.zeros(
            (self.enc_steps, self.dec_steps, target_vector.shape[-1])
        )
        for i in range(self.enc_steps):
            for j in range(self.dec_steps):
                # 0 -> [1, 2, 3]
                # target_matrix[i,j] = target_vector[i+j+1,:]
                # 0 -> [0, 1, 2]
                target_matrix[i, j] = target_vector[i + j, :]
        return target_matrix

    def get_distance_target(self, target_vector):
        target_matrix = np.zeros(self.enc_steps - 1)
        target_argmax = target_vector[self.enc_steps - 1].argmax()
        for i in range(self.enc_steps - 1):
            if target_vector[i].argmax() == target_argmax:
                target_matrix[i] = 1.0
        return target_matrix, target_vector[self.enc_steps - 1]

    def __getitem__(self, index):
        """self.inputs.append([
            session, start, end, enc_target, distance_target, class_h_target
        ])"""

        if self.subnet == "train" and self.add_labels:
            (
                session,
                start,
                end,
                enc_target,
                distance_target,
                class_h_target,
                dec_target,
                additional_class,
            ) = self.inputs[index]
            try:
                camera_inputs = self.feature_All[session]["rgb"][start:end]
            except:
                (
                    session,
                    start,
                    end,
                    enc_target,
                    distance_target,
                    class_h_target,
                    dec_target,
                ) = self.inputs[index - 1]
                camera_inputs = self.feature_All[session]["rgb"][start:end]
            camera_inputs = torch.tensor(camera_inputs)
            motion_inputs = self.feature_All[session]["flow"][start:end]

            motion_inputs = torch.tensor(motion_inputs)
            enc_target = torch.tensor(enc_target)
            distance_target = torch.tensor(distance_target)
            class_h_target = torch.tensor(class_h_target)
            dec_target = torch.tensor(dec_target)
            additional_class = torch.tensor(additional_class)

            # take the
            if camera_inputs.shape[0] != 512:
                camera_inputs = torch.cat(
                    [
                        camera_inputs,
                        torch.zeros(
                            512 - camera_inputs.shape[0], camera_inputs.shape[1]
                        ),
                    ],
                    axis=0,
                )
                motion_inputs = torch.cat(
                    [
                        motion_inputs,
                        torch.zeros(
                            512 - motion_inputs.shape[0], motion_inputs.shape[1]
                        ),
                    ],
                    axis=0,
                )

            return (
                session,
                start,
                end,
                camera_inputs,
                motion_inputs,
                enc_target,
                distance_target,
                class_h_target,
                dec_target,
                additional_class,
            )

        elif self.subnet == "test" or self.subnet == "train":
            (
                session,
                start,
                end,
                enc_target,
                distance_target,
                class_h_target,
                dec_target,
            ) = self.inputs[index]
            try:
                camera_inputs = self.feature_All[session]["rgb"][start:end]
            except:
                (
                    session,
                    start,
                    end,
                    enc_target,
                    distance_target,
                    class_h_target,
                    dec_target,
                ) = self.inputs[index - 1]
                camera_inputs = self.feature_All[session]["rgb"][start:end]
            camera_inputs = torch.tensor(camera_inputs)
            motion_inputs = self.feature_All[session]["flow"][start:end]
            motion_inputs = torch.tensor(motion_inputs)
            enc_target = torch.tensor(enc_target)
            distance_target = torch.tensor(distance_target)
            class_h_target = torch.tensor(class_h_target)
            dec_target = torch.tensor(dec_target)

            # take the
            if camera_inputs.shape[0] != 512:
                camera_inputs = torch.cat(
                    [
                        camera_inputs,
                        torch.zeros(
                            512 - camera_inputs.shape[0], camera_inputs.shape[1]
                        ),
                    ],
                    axis=0,
                )
                motion_inputs = torch.cat(
                    [
                        motion_inputs,
                        torch.zeros(
                            512 - motion_inputs.shape[0], motion_inputs.shape[1]
                        ),
                    ],
                    axis=0,
                )

            return (
                session,
                start,
                end,
                camera_inputs,
                motion_inputs,
                enc_target,
                distance_target,
                class_h_target,
                dec_target,
            )

    def __len__(self):
        return len(self.inputs)

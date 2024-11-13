# PREGO: online mistake detection in PRocedural EGOcentric videos (CVPR 2024)

### [PREGO paper [CVPR 2024]](https://openaccess.thecvf.com/content/CVPR2024/html/Flaborea_PREGO_Online_Mistake_Detection_in_PRocedural_EGOcentric_Videos_CVPR_2024_paper.html) [TI-PREGO paper [arxiv]](https://arxiv.org/abs/2411.02570)

The official PyTorch implementation of the IEEE/CVF Computer Vision and Pattern Recognition (CVPR) '24 paper **PREGO: online mistake detection in PRocedural EGOcentric videos** and of the follow-up paper **TI-PREGO: Chain of Thought and In-Context Learning for Online Mistake Detection in PRocedural EGOcentric Videos**.


PREGO is the first online one-class classification model for mistake detection in procedural egocentric videos. It uses an online action recognition component to model current actions and a symbolic reasoning module to predict next actions, detecting mistakes by comparing the recognized current action with the expected future one. We evaluate this on two adapted datasets, Assembly101-O and Epic-tent-O, for online benchmarking of procedural mistake detection.

![teaser_image](assets/teaser.png)

## News
 **[2024-11-12]** Uploaded the TSN features for Assembly101-O and Epic-tent-O[[GDrive]](https://drive.google.com/drive/u/1/folders/1gcOIEXhwysCE2o8-5C4vQnTShJ7p3CKH).

 **[2024-11-04]** Published the follow-up paper [[TI-PREGO]](https://arxiv.org/abs/2411.02570).
 
 **[2024-06-20]** PREGO presented at #CVPR2024.
 
 **[2024-06-16]** Uploaded the anticipation branch.

<!-- ## Data -->
<!-- *WIP* -->

## Usage

### Anticipation
```bash
cd step_anticipation
./scripts/anticipation.sh
```

## Reference
If you find our code or paper to be helpful, please consider citing:
```
@InProceedings{Flaborea_2024_CVPR,
    author    = {Flaborea, Alessandro and di Melendugno, Guido Maria D'Amely and Plini, Leonardo and Scofano, Luca and De Matteis, Edoardo and Furnari, Antonino and Farinella, Giovanni Maria and Galasso, Fabio},
    title     = {PREGO: Online Mistake Detection in PRocedural EGOcentric Videos},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {18483-18492}
}
```
```
@misc{plini2024tipregochainthoughtincontext,
      title={TI-PREGO: Chain of Thought and In-Context Learning for Online Mistake Detection in PRocedural EGOcentric Videos}, 
      author={Leonardo Plini and Luca Scofano and Edoardo De Matteis and Guido Maria D'Amely di Melendugno and Alessandro Flaborea and Andrea Sanchietti and Giovanni Maria Farinella and Fabio Galasso and Antonino Furnari},
      year={2024},
      eprint={2411.02570},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.02570}, 
}
```

import os
import sys
import random
from selfeeg import augmentation as aug
import numpy as np
import matplotlib.pyplot as plt
import torch

#seed = 11
#random.seed(seed)
#np.random.seed(seed)
#torch.manual_seed(seed)
#plt.style.use('seaborn-v0_8-white')
#plt.rcParams['figure.figsize'] = (15.0, 6.0)


#Creating sample EEG to create Augmenter
Fs = 128
SampleBatchEEG = torch.zeros(16,32,1024) + torch.sin(torch.linspace(0, 8*np.pi,1024))

#Phase 1 Augmentations
augBandNoise = aug.DynamicSingleAug(aug.add_band_noise, 
                                    discrete_arg={
                                        'samplerate': Fs, 
                                        'noise_range': [.05, .1],''
                                        'bandwidth': ["delta", "theta", "gamma", "beta"]}
                                        )

augArtifactNoise = aug.DynamicSingleAug(aug.add_eeg_artifact, 
                                        discrete_arg={
                                            'Fs': Fs, 
                                            'artifact': ["eye", "muscle"], 
                                            'batch_equal': False}
                                            )
augGaussianNoise = aug.DynamicSingleAug(aug.add_gaussian_noise, 
                                        range_arg={'std': [.05, .3]},
                                        range_type=[False]
                                        )

Phase1 = aug.RandomAug(augArtifactNoise, augBandNoise, augGaussianNoise)

augTimeMask = aug.DynamicSingleAug(aug.masking, 
                                   discrete_arg={'batch_equal': False}, 
                                   range_arg={'mask_number': [1, 3]}, 
                                   range_type = {'mask_number': 'int'}
                                       )

augWarp = aug.DynamicSingleAug(aug.warp_signal, discrete_arg={'batch_equal': False})

augCropResize = aug.DynamicSingleAug(aug.crop_and_resize, 
                                     discrete_arg={"batch_equal": False}, 
                                     range_arg={'N_cut': [1, 4], 'segments': [10,15]}, 
                                     range_type={'N_cut': True, 'segments':True})

Phase2 = aug.RandomAug(augCropResize,augTimeMask, augWarp, p=[.6, .2, .2])

augFlip = aug.RandomAug(aug.flip_horizontal, aug.flip_vertical)

augSame = aug.StaticSingleAug(aug.identity)

Phase3 = aug.RandomAug(augSame, augFlip, p=[.75, .25])

Augmenter = aug.SequentialAug(Phase1, Phase2, Phase3)

#Implementing Augmentations onto Sample EEG
AugmentedBatchEEG = Augmenter(SampleBatchEEG)


plt.plot(SampleBatchEEG[0,0],linewidth=2.5)
plt.plot(AugmentedBatchEEG[0,0])
plt.tick_params(axis='both', which='major', labelsize=12)
plt.title('Augmentation Visualized', fontsize=15)
plt.legend(['original sample', 'augmented sample'])
plt.show()
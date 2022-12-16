## SDP-GAN: Saliency Detail Preservation Generative Adversarial Networks for High Perceptual Quality Style Transfer

This is the Pytorch implementation of our TIP 2020 paper [SDP-GAN](http://liushuaicheng.org/TIP/SDPGAN/SDPGAN-TIP.pdf).

![image](./figs/pipeline.png)

## Dependencies

* Python=3.5
* Pytorch>=1.1.0
* Other requirements please refer to requirements.txt.


## Pre-trained models

The pre-trained models can be downloaded [here](https://drive.google.com/drive/folders/1agSGUuK0LuwLuxzqXADGdRa2rvD_CyWu?usp=sharing). Place the models in ./pretrained_models in order to test it.


## Testing

```
python3 test.py
```

## Results

![image](./figs/results.png)

## Citation

```
@article{li2020sdp-gan,
  title={SDP-GAN: Saliency Detail Preservation Generative Adversarial Networks for High Perceptual Quality Style Transfer},
  author={Li, Ru and Wu, Chi-Hao and Liu, Shuaicheng and Wang, Jue and Wang, Guangfu and Liu, Guanghui and Zeng, Bing},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={374--385},
  year={2020},
  publisher={IEEE}
}
```

# Acknowledgments

In this project we use (parts of) the implementations of the following works:

* [CartoonGAN-Test-Pytorch-Torch](https://github.com/Yijunmaverick/CartoonGAN-Test-Pytorch-Torch)
* [Pytorch-CartoonGAN](https://github.com/znxlwm/pytorch-CartoonGAN) 

We thank the respective authors for open sourcing of their implementations.

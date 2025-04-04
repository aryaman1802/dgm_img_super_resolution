# **dgm_img_super_resolution**

A repository containing deep generative models for image super resolution.

![Basic Diffusion Model Architecture](basic_diffusion_model.png "Basic Diffusion Model Architecture")

## **Repository Structure**

```bash
├── ddpm
│   ├── __init__.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── full_model.py
│   ├── utils.py
├── vae
|   ├── __init__.py
|   ├── encoder.py
|   ├── decoder.py
|   ├── full_model.py
|   ├── utils.py
├── notebooks
│   ├── basic_diffusion.ipynb
│   ├── vae.ipynb
│   ├── simple_gan.ipynb
│   ├── dcgan.ipynb
│   ├── wgan.ipynb
├── main.py
```


## **Introduction**

There are many deep generative models for image super resolution. The goal of this repo is not to build a model that outperforms the state of the art, but to build a model from scratch for educational purposes. We will focus on building a diffusion model for image super resolution. 

- The directories `ddpm` and `vae` contain the code for the diffusion model and the variational autoencoder, respectively.  They contain the code meant to be used for image super resolution. 
- The `notebooks` directory contains detailed explanations (along with the math) of some popular deep generative models like the Denoising Diffusion Probabilistic model (DDPM), Variational Autoencoder (VAE), and Generative Adversarial Networks (GANs). I highly recommend reading these notebooks to anybody who wants to get an intuitive understanding of these deep generative models. Note: these notebooks only contain the simplest version of the models. They are not meant to be used for image super resolution, but rather for educational purposes.
- The `main.py` file contains the code to train and evaluate the model.

In the future, I might add more models to this repo, such as GANs and VAEs, for image super resolution.

## **Dataset**

something

## **Evaluation**

something
# Thai-Food-Classification
 
Thai-Food-Classification is a web application that allows users to upload their Thai food images and it will predict what kind of food it is.

## Description

This application is built on the `streamlit` library. I used pretrained PyTorch [ViT(Vision Transformer)](https://github.com/lucidrains/vit-pytorch) model and trained it on my Thai food dataset. The dataset I used only contains 16 classes listed in [class.txt](class.txt) file, but it is in Thai.

## Usage

I provide my [checkpoint](https://drive.google.com/drive/folders/113kF1b1pMHqLYTcLYiL1uxTc2VVzi0N1?usp=sharing) that I use for this application. Feel free to use it. To run this application, follow these steps.

I use python `3.10.4`.

**Install the [requirements](requirements.txt)**

```bash
pip install -r requirements.txt
```

**Run the app using `streamlit run` command**
```bash
streamlit run app.py
```

### Running on docker

Build docker image
```bash
docker build -t thai-food .
```
```bash
docker run -p 8501:8501 thai-food:latest
```
import Net
import cv2
import os
import torch
from torch import optim
import numpy as np
from PIL import Image
from io import BytesIO
import torch.functional as F
import matplotlib.pyplot as plt
from torchvision import models, transforms
from sys import stdout


def display_video(video_path):
    cam = cv2.VideoCapture(video_path)
    while cam.isOpened():
        isFrame, frame = cam.read()
        if isFrame == 0:
            break
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = frame[100:500, :]
        cv2.imshow("1", frame)
        if cv2.waitKey(1) == 27:
            break
    cam.release()
    cv2.destroyAllWindows()


def load_model():
    vgg = models.vgg19(pretrained=True).features
    for para in vgg.parameters():
        para.requires_grad_(False)
    return vgg


def load_image(img_path, max_size=400, shape=None):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (400, 400))
    image = transforms.ToTensor()(image)
    image=transforms.Normalize((0.485, 0.456, 0.406),
                               (0.229, 0.224, 0.225))(image)
    image = image.unsqueeze(0)
    print(image.shape)
    return image


def im_convert(tensor):
    """ Display a tensor as an image. """

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image


def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',
                  '28': 'conv5_1'}

    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features


def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
vgg = load_model().to(device)

style = load_image("art2.jpg").to(device)
content = load_image("art3.jpg").to(device)

content_features = get_features(content, vgg)
style_features = get_features(style, vgg)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
target = content.clone().requires_grad_(True).to(device)
style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.75,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,
                 'conv5_1': 0.2}

content_weight = 1
style_weight = 1e3
show_every = 100

# iteration hyperparameters
optimizer = optim.Adam([target], lr=0.003)
steps = 2000  # decide how many iterations to update your image (5000)

for ii in range(1, steps + 1):

    target_features = get_features(target, vgg)
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        _, d, h, w = target_feature.shape
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
        style_loss += layer_style_loss / (d * h * w)
    stdout.write(".")
    stdout.flush()
    total_loss = content_weight * content_loss + style_weight * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # display intermediate images and print the loss
    if ii % show_every == 0:
        print('Total loss: ', total_loss.item())
        plt.imshow(im_convert(target))
        plt.show()

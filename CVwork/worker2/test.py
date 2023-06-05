from PIL import Image
import numpy as np
import PIL
import torch
import cv2
import torchvision.utils as vutils
from torchvision import transforms
import torchvision.models as models
import torchvision.transforms as T
import matplotlib.pyplot as plt
label_colors = np.array([(0, 0, 0), # background
                        (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                        # aeroplane, bicycle, bird, boat, bottle
                        (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                        # bus, car, cat, chair, cow
                        (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                        # dining table, dog, horse, motorbike, person
                        (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)
                        # potted plant, sheep, sofa, train, tv/monitor
                        ])

# 对输出进行编码
def decode_segmaps(image, label_colors=label_colors, nc=21):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for cls in range(0, nc):
        idx = image == cls # 得到对应类别的特定颜色
        r[idx] = label_colors[cls, 0]
        g[idx] = label_colors[cls, 1]
        b[idx] = label_colors[cls, 2]
    rgbimage = np.stack([r, g, b], axis=2) # 合并颜色
    cv2.imwrite('111.jpg',rgbimage)
    return rgbimage
img_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485,0.456,0.406],
                                                         std=[0.229,0.224,0.225])])

def get_mask(name,model,model2,img_transform=img_transform,label_colors=label_colors):
    image = PIL.Image.open(name)
    img_tensor = img_transform(image).unsqueeze(0)
    output = model(img_tensor)['out']
    output2 = model2(img_tensor)['out']
    output_arg = torch.argmax(output.squeeze(), dim=0).numpy()
    output_arg2 = torch.argmax(output2.squeeze(),dim=0).numpy()
    #print(output_arg)
    output_rgb = decode_segmaps(output_arg, label_colors)
    output_rgb2 = decode_segmaps(output_arg2, label_colors)
    plt.figure(figsize=(8, 10),dpi=300)
    #plt.subplot(1, 3, 1)
    #plt.imshow(image)
    #plt.axis('off')
    plt.subplot(1, 2, 1)
    plt.imshow(output_rgb)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(output_rgb2)
    plt.axis('off')
    plt.subplots_adjust(wspace=0.05)
    plt.savefig('4-6.png',bbox_inches='tight',pad_inches=0)
    plt.show()


model2 = models.segmentation.deeplabv3_resnet50(pretrained=True)
model= models.segmentation.fcn_resnet50(pretrained=True)
model2.eval()
model.eval()
get_mask('2007_000762.jpg',model,model2)


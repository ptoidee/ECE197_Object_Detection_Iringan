import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import torch
from initializer import ImageDataset,get_transform
import label_utils

def draw_bounding_box(pane, rect_coordinates):
    # Show bounding boxes

    # Create figure and axes
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(pane)

    
    # Create a Rectangle patch
    (x, y, w, h) = rect_coordinates
    rect = patches.Rectangle((x,y),w-x,h-y,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)

    plt.show() 

test_dict, test_classes = label_utils.build_label_dictionary("drinks/labels_test.csv")

dataset_test = ImageDataset(test_dict, get_transform(train=False))
# pick one image from the test set
img, _ = dataset_test[50]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# put the model in evaluation mode
model=torch.load('model.pth')
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])

print(prediction[0])
draw_bounding_box(img[0],prediction[0]['boxes'][0])
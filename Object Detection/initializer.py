import torch
import torch.utils.data
from PIL import Image
import torchvision
from torchvision.transforms import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dictionary, transforms=None):
        self.dictionary = dictionary
        self.transforms = transforms

    def __len__(self):
        return len(self.dictionary)

    def __getitem__(self, idx):
        # retrieve the image filename
        key = list(self.dictionary.keys())[idx]
        # retrieve all bounding boxes
        boxes = self.dictionary[key]
        # open the file as a PIL image
        img = Image.open(key).convert("RGB")
        # apply the necessary transforms
        box = []
        labeled = []
        for i in boxes:
            xmin = i[0]
            xmax = i[1]
            ymin = i[2]
            ymax = i[3]
            box.append([xmin, ymin, xmax, ymax])
            labeled.append(int(i[4]))

        box = torch.as_tensor(box, dtype=torch.float32)
        targets = {}
        targets["boxes"] = box
        targets["labels"] = torch.tensor(labeled, dtype=torch.int64)
        targets["image_id"] = torch.tensor([idx], dtype=torch.int64)
        targets["area"] = torch.tensor((box[:, 3] - box[:, 1]) * (box[:, 2] - box[:, 0]))
        targets["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.uint8)
        if self.transforms:
            img = self.transforms(img)
          
        # return a list of images and corresponding target
        return img, targets

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def bounding_box_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=False)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

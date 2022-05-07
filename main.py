import numpy as np
import label_utils
import torch
import torch.utils.data
from torchvision import transforms
from PIL import Image
import torchvision
from engine import train_one_epoch, evaluate
import utils
from torchvision.transforms import transforms as T
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.optim.lr_scheduler import StepLR

if __name__ == '__main__':
    
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
            # transforms like crop, resize, normalize, etc
            box = []
            labeled = []
            for i in boxes:
                xmin = i[0]
                xmax = i[1]
                ymin = i[2]
                ymax = i[3]
                box.append([xmin, ymin, xmax, ymax])
                labeled.append(i[4])

            box = torch.as_tensor(box, dtype=torch.float32)
            targets = {}
            targets["boxes"] = box
            targets["labels"] = torch.tensor(labeled, dtype=torch.int64)
            targets["image_id"] = torch.tensor([idx], dtype=torch.int64)
            targets["area"] = torch.tensor((box[:, 3] - box[:, 1]) * (box[:, 2] - box[:, 0]))
            targets["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.uint8)

            if self.transforms:
                img = self.transforms(img)
            
            # return a list of images and corresponding labels
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
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)

        # get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model

    test_dict, test_classes = label_utils.build_label_dictionary("drinks/labels_test.csv")
    train_dict, train_classes = label_utils.build_label_dictionary("drinks/labels_train.csv")

    dataset = ImageDataset(train_dict, get_transform(train=True))
    dataset_test = ImageDataset(test_dict, get_transform(train=False))

    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2, collate_fn=utils.collate_fn)
    test_dataloader = DataLoader(dataset_test, batch_size=8, shuffle=True, num_workers=2, collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 4

    # get the model using our helper function
    model = bounding_box_model(num_classes)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 1

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, test_dataloader, device=device)
        
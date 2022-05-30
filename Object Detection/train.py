from initializer import ImageDataset,get_transform,bounding_box_model
import torch
import label_utils
from engine import train_one_epoch
import utils
from torch.utils.data import DataLoader
import tensorflow as tf
from tensorflow import keras

def main():

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
    
    torch.save(model, 'model.pth')

if __name__ == "__main__":
    main()
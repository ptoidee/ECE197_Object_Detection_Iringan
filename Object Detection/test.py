from initializer import ImageDataset,get_transform
import torch
import label_utils
from engine import evaluate
import utils
from torch.utils.data import DataLoader

def main():   
    #load test dictionary
    test_dict, test_classes = label_utils.build_label_dictionary("drinks/labels_test.csv")
    #load test dataset
    dataset_test = ImageDataset(test_dict, get_transform(train=False))
    #load test dataloader
    test_dataloader = DataLoader(dataset_test, batch_size=8, shuffle=True, num_workers=2, collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load pretrained model
    model=torch.load('model.pth')

    # evaluate on the test dataset
    evaluate(model, test_dataloader, device=device)

if __name__ == "__main__":
    main()
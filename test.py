from initializer import ImageDataset,get_transform
import torch
import label_utils
from engine import evaluate
import utils
from torch.utils.data import DataLoader

def main():    
    test_dict, test_classes = label_utils.build_label_dictionary("drinks/labels_test.csv")

    dataset_test = ImageDataset(test_dict, get_transform(train=False))

    test_dataloader = DataLoader(dataset_test, batch_size=8, shuffle=True, num_workers=2, collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load pretrained model
    model=torch.load('model.pth')

        # evaluate on the test dataset
    evaluate(model, test_dataloader, device=device)

if __name__ == "__main__":
    main()
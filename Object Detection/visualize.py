# sample one mini-batch
images, boxes = next(iter(train_loader))
# map of label to class name
class_labels = {i: label_utils.index2class(i) for i in train_classes}

run.display(height=1000)
table = wandb.Table(columns=['Image'])

# we use wandb to visualize the objects and bounding boxes
for image, box in zip(images, boxes):
    dict = []
    for i in range(box.shape[0]):
        if box[i, -1] == 0:
            continue
        dict_item = {}
        dict_item["position"] = {
            "minX": box[i, 0].item(),
            "maxX": box[i, 1].item(),
            "minY": box[i, 2].item(),
            "maxY": box[i, 3].item(),
        }
        dict_item["domain"] = "pixel"
        dict_item["class_id"] = (int)(box[i, 4].item())
        dict_item["box_caption"] = label_utils.index2class(
            dict_item["class_id"])
        dict.append(dict_item)

    img = wandb.Image(image, boxes={
        "ground_truth": {
            "box_data": dict,
            "class_labels": class_labels
        }
    })
    table.add_data(img)

wandb.log({"train_loader": table})
wandb.finish()
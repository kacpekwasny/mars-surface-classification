import segmentation_models_pytorch as smp
import segmentation_models_pytorch.encoders as smpe


ENCODER = "resnet34"
WEIGHTS = "imagenet"

# model = smp.Unet(
#     encoder_name=ENCODER,
#     encoder_weights=WEIGHTS
# )

# process_input = smpe.get_preprocessing_fn(ENCODER, pretrained=WEIGHTS)

# model.train(True)


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import matplotlib.pyplot as plt
import cv2



DATA_DIR = '..\\UNet\\data'



x_train_dir = os.path.join(DATA_DIR, 'imgs')
y_train_dir = os.path.join(DATA_DIR, 'masks')

x_valid_dir = os.path.join(DATA_DIR, 'valid_imgs')
y_valid_dir = os.path.join(DATA_DIR, 'valid_masks')

x_test_dir = os.path.join(DATA_DIR, 'test_imgs')
y_test_dir = os.path.join(DATA_DIR, 'test_masks')




# helper function for data visualization
def visualize(**images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        print(name, image.shape)
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()





from torch.utils.data import DataLoader

from data_carvana import Dataset as CaravanDataset



if False:
    # Lets look at data we have
    dataset = CaravanDataset(x_train_dir, y_train_dir, classes=['car'])

    image, mask = dataset[4] # get some sample
    visualize(
        image=image, 
        cars_mask=mask[:, :, :, 0],
    )


# Skipping albumentation (data augmentation)


import torch
import numpy as np
import segmentation_models_pytorch as smp



ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['car']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'

# create segmentation model with pretrained encoder
model = smp.FPN(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


import albumentations as albu

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')
    
def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)



train_dataset = CaravanDataset(
    x_train_dir, 
    y_train_dir, 
    # augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = CaravanDataset(
    x_valid_dir, 
    y_valid_dir, 
    # augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)




# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])





# create epoch runners 
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)






# train model for 40 epochs
if __name__ == "__main__":
    max_score = 0

    for i in range(0, 40):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')
            
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')






    # load best saved checkpoint
    best_model = torch.load('./best_model.pth')






    # create test dataset
    test_dataset = CaravanDataset(
        x_test_dir, 
        y_test_dir, 
        augmentation=smp.get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    test_dataloader = DataLoader(test_dataset)






    # evaluate model on test set
    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )

    logs = test_epoch.run(test_dataloader)




    # test dataset without transformations for image visualization
    test_dataset_vis = Dataset(
        x_test_dir, y_test_dir, 
        classes=CLASSES,
    )




    for i in range(5):
        n = np.random.choice(len(test_dataset))
        
        image_vis = test_dataset_vis[n][0].astype('uint8')
        image, gt_mask = test_dataset[n]
        
        gt_mask = gt_mask.squeeze()
        
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
            
        visualize(
            image=image_vis, 
            ground_truth_mask=gt_mask, 
            predicted_mask=pr_mask
        )






















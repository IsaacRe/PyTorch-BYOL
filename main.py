import os

import numpy as np
import torch
import yaml
from experiment_utils.dataset import get_dataloader_incr, get_dataloader
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets
from data.multi_view_data_injector import MultiViewDataInjector
from data.transforms import get_simclr_data_transforms
from models.mlp_head import MLPHead
from models.resnet_base_network import ResNet18
from trainer import BYOLTrainer
from argparse import ArgumentParser

print(torch.__version__)
torch.manual_seed(0)


def main():
    parser = ArgumentParser()
    parser.add_argument('--incr', action='store_true', help='train representation incrementally')
    args = parser.parse_args()

    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
    n_class_epochs = config['incr']['n_class_epochs']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    data_transform = get_simclr_data_transforms(**config['data_transforms'])

    #train_dataset = datasets.STL10('../../data', split='train+unlabeled', download=False,
    #                               transform=MultiViewDataInjector([data_transform, data_transform]))

    # online network
    online_network = ResNet18(**config['network']).to(device)
    pretrained_folder = config['network']['fine_tune_from']

    # load pre-trained model if defined
    if pretrained_folder:
        try:
            checkpoints_folder = os.path.join('./runs', pretrained_folder, 'checkpoints')

            # load pre-trained parameters
            load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
                                     map_location=torch.device(torch.device(device)))

            online_network.load_state_dict(load_params['online_network_state_dict'])

        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    # predictor network
    predictor = MLPHead(in_channels=online_network.projection.net[-1].out_features,
                        **config['network']['projection_head']).to(device)

    # target encoder
    target_network = ResNet18(**config['network']).to(device)

    optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()),
                                **config['optimizer']['params'])

    trainer = BYOLTrainer(online_network=online_network,
                          target_network=target_network,
                          optimizer=optimizer,
                          predictor=predictor,
                          device=device,
                          **config['trainer'])

    num_workers = config['trainer']['num_workers']
    batch_size_train = 100
    batch_size_test = 200

    if args.incr:
        incr_train_loaders, incr_val_loaders = get_dataloader_incr(data_dir='../../data', base='CIFAR10', num_classes=10,
                                                                   img_size=224, classes_per_exposure=2, train=True,
                                                                   num_workers=num_workers, batch_size_train=batch_size_train,
                                                                   batch_size_test=batch_size_test,
                                                                   transform=MultiViewDataInjector([data_transform, data_transform]))

        # get train and val indices sampled
        train_indices = np.concatenate([ldr.sampler.indices for ldr in incr_train_loaders])

        train_class_dataloader = DataLoader(incr_train_loaders[0].dataset, sampler=SubsetRandomSampler(train_indices),
                                            batch_size=batch_size_train, num_workers=num_workers)
        #trainer.train(train_dataset)
        trainer.train_incr(incr_train_loaders, incr_val_loaders,
                           n_class_epochs=n_class_epochs,
                           train_class_dataloader=train_class_dataloader)
    else:
        train_loader, val_loader = get_dataloader(data_dir='../../data', base='CIFAR10', num_classes=10,
                                                  img_size=224, train=True, num_workers=num_workers,
                                                  batch_size_train=batch_size_train, batch_size_test=batch_size_test,
                                                  transform=MultiViewDataInjector([data_transform, data_transform]))
        trainer.train(train_loader, val_loader, n_class_epochs=n_class_epochs)


if __name__ == '__main__':
    main()

#DISCLAIMER this code is inspired from this repository :https://github.com/ankur219/ECG-Arrhythmia-classification


from __future__ import division, print_function
if __name__ == '__main__':
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import numpy as np
    import torch.optim as optim
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import numpy as np
    from torch import optim
    import torch.nn as nn
    from torchvision import models
    import gc
    #import normal_preprocessing as NO
    #import cropping as crop
    #import left_bundle_preprocessing as LB
    #import right_bundle_preprocessing as RB
    #import atrial_premature_preprocessing as A
    #import ventricular_escape_beat_preprocessing as E
    #import paced_beat_preprocessing as PB
    #import ventricular_premature_contraction_preprocessing as V
    #import create_dataset

    #########################################################
    #Image Generation
    #########################################################
    #NO.normal_image_generation()
    #LB.left_bundle_image_generation()
    #RB.right_bundle_image_generation()
    #A.atrial_premature_image_generation()
    #E.ventricular_escape_image_generation()
    #PB.paced_beat_image_generation()
    #V.ventricular_premature_image_generation()

    #########################################################
    #Data augmentation
    #########################################################
    #crop.directory_selection_cropping('images_normal')
    #crop.directory_selection_cropping('images_left_bundle')
    #crop.directory_selection_cropping('images_right_bundle')
    #crop.directory_selection_cropping('images_atrial_premature')
    #crop.directory_selection_cropping('images_ventricular_escape')
    #crop.directory_selection_cropping('images_paced_beat')
    #crop.directory_selection_cropping('images_ventricular_premature')
    #########################################################
    #Datase creation
    #########################################################
    #we clear the GPU memory
    gc.collect()
    torch.cuda.empty_cache()


    transform = transforms.Compose(
     [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset=torchvision.datasets.ImageFolder(root="/home/ens/AP69690/SYS843/database",transform=transform)
    #D:\geordi\ecole\Automn2022\SYS843\pycharmprojects\database
    #/home/ens/AP69690/SYS843/database

    N = len(dataset)
    print(N)
    # generate & shuffle indices
    indices = np.arange(N)
    indices = np.random.permutation(indices)

    # select train/test, for demo I am using 80,20 trains/test
    train_indices = indices[:int(0.8 * N)]
    test_indices = indices[int(0.8 * N):int(N)]


    train_set = torch.utils.data.Subset(dataset, train_indices)
    test_set = torch.utils.data.Subset(dataset, test_indices)
    print(dataset.classe())

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False,num_workers=2)

    class HeartNet(nn.Module):
        def __init__(self, num_classes=7):
            super(HeartNet, self).__init__()

            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.ELU(inplace=True),
                nn.BatchNorm2d(64, eps=0.001),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ELU(inplace=True),
                nn.BatchNorm2d(64, eps=0.001),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ELU(inplace=True),
                nn.BatchNorm2d(128, eps=0.001),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ELU(inplace=True),
                nn.BatchNorm2d(128, eps=0.001),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ELU(inplace=True),
                nn.BatchNorm2d(256, eps=0.001),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ELU(inplace=True),
                nn.BatchNorm2d(256, eps=0.001),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

            self.classifier = nn.Sequential(
                nn.Linear(16 * 16 * 256, 2048),
                nn.ELU(inplace=True),
                nn.BatchNorm1d(2048, eps=0.001),
                nn.Dropout(0.5),
                nn.Linear(2048, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), 16 * 16 * 256)
            x = self.classifier(x)
            return x

    net = HeartNet()

    # On utilise la descente de gradient stochastique comme optimiseur. D'autres méthodes sont existante mais celle-ci reste très utilisée.
    optimizer = optim.Adam(net.parameters(),lr=0.0001)

    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    torch.cuda.empty_cache()
    """
    gpu = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # Model et optimizer déjà définis dans les questions précédentes
    criterion = nn.CrossEntropyLoss() # Fonction de coût qui permettra le calcul de l'erreur
    net.to(gpu)
    for epoch in range(9): # loop sur le dataset 5 fois
      running_loss = 0.0
      net.train()
      for i, data in enumerate(trainloader, 0): # En mettant data de cette façon, data est un tuple tel que data = (image, label)

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(gpu), data[1].to(gpu)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs) # Forward propagation (On passe notre input au modèle à travers les différentes couches)
        loss = criterion(outputs, labels) # Calcul de l'erreur
        loss.backward() # On rétropropage l'erreur dans le réseau, donc on calcul le gradient de l'erreur pour chaque paramètre
        optimizer.step() # On actualise les poids en fonction des gradients

        # print statistics
        running_loss += loss.item() # .item() retourne la valeur dans le tenseur et non le tenseur lui même
        if i % 1000 == 999: # print every 1000 mini-batches (On a 50 000 données et mon batch size est de 10 donc on aura 5 000 itération sur le dataloader d'entraînement)
          print(f"[epoch {epoch + 1}, batch {i+1}/{int(len(dataset.targets)/10)}], loss : {running_loss / 1_000}")
          running_loss = 0.0
      correct = 0
      net.eval()
      for i, data in enumerate(testloader, 0):
        inputs, labels = data[0].to(gpu), data[1].to(gpu)
        outputs = net(inputs)
        pred = outputs.argmax() # mon_tenseur.argmax() donne l'index de l'élément le plus élevé de l'output, et donc on récupère la classe prédite par notre algo
                                # mon_tenseur.argmax(-1) donnera le même résultat
        if pred == labels: # On est pas obligé de sortir la donnée via pred[0] et labels[0] car il n'y a qu'une valeur dans le tenseur, mais on peut, les deux reviennent au même
          correct += 1

      print(f"Epoch : {epoch + 1} - Taux de classification global = {correct / len(testloader)}")
    print('Finished Training')
    """
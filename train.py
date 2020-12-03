# Source: https://towardsdatascience.com/a-practical-example-in-transfer-learning-with-pytorch-846bb835f2db

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms


# Load dataset
dataset = datasets.ImageFolder(
    'image_data/all_images',
    transforms.Compose([
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 20, 20])

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True#,
    # num_workers=4
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=True#,
    # num_workers=4
)


# Declare model
model = models.alexnet(pretrained=True)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)


NUM_EPOCHS = 10
BEST_MODEL_PATH = 'best_model.pth'
best_accuracy = 0.0
lr = 0.0001 #learning rate
print("lr =", lr)

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# Training
for epoch in range(NUM_EPOCHS):
    
    for images, labels in iter(train_loader):
        # images = images.to(device)
        # labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
    
    test_error_count = 0.0
    for images, labels in iter(test_loader):
        # images = images.to(device)
        # labels = labels.to(device)
        outputs = model(images)
        test_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))
    
    test_accuracy = 1.0 - float(test_error_count) / float(len(test_dataset))
    print('%d: %f' % (epoch, test_accuracy))
    print('loss: %f' % (loss))
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        best_accuracy = test_accuracy

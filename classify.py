import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
from torchvision import transforms
import os
import sys
from PIL import Image

#Note *Testing and training data split already for CIFAR10 dataset
# ----------------- prepare training data -----------------------
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomCrop(32, padding=4),  # Add padding and crop back to 32x32
    transforms.ToTensor(),  # Convert image to PyTorch tensor
])

train_data = torchvision.datasets.CIFAR10(
    root='./data.cifar10',                          # location of the dataset
    train=True,                                     # this is training data
    transform=transform_train,    # Converts a PIL.Image or numpy.ndarray to torch.FloatTensor of shape (C x H x W)
    download=True                                   # if you haven't had the dataset, this will automatically download it for you
)

train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)

# ----------------- prepare testing data -----------------------
test_data = torchvision.datasets.CIFAR10(root='./data.cifar10/', train=False, transform=torchvision.transforms.ToTensor())

test_loader = Data.DataLoader(dataset=test_data, batch_size=64, shuffle=False) 

# ----------------- build the model ------------------------
class My_XXXnet(nn.Module):
    def __init__(self, n_input, n_hidden, n_hidden2, n_output):
        super(My_XXXnet, self).__init__()
        self.hidden1 = torch.nn.Linear(n_input, n_hidden)
        self.dropout1 = torch.nn.Dropout(p=0.3)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden2)
        self.dropout2 = torch.nn.Dropout(p=0.3)
        self.output = torch.nn.Linear(n_hidden2, n_output)

    #makes data pass through layers in the forward direction
    def forward(self, x):
        x = x.view(-1, 32*32*3) #flatten the input tensor

        x = self.hidden1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.hidden2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.output(x)
        return x

model = My_XXXnet(3072, 1024, 1024, 10)
loss_func = nn.CrossEntropyLoss() #
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
acc_list = []

def train():
    #training loop
    print(f"{'Loop':<10}{'Train Loss':<15}{'Train Acc %':<15}{'Test Loss':<15}{'Test Acc %':<15}")  # Print table headers
    print("-" * 70)
    for epoch in range(10):
        correct = 0
        total = 0
        train_loss = 0
        for step, (input, target) in enumerate(train_loader):
            model.train()   # set the model in training mode. WHY? dropout layers work correctly only in training mode, disabled in eval mode

            prediction = model(input) #Get predictions from forward propagation 
            loss = loss_func(prediction, target) # calculate the loss 

            optimizer.zero_grad() #clear gradients
            loss.backward() #compute and update gradients
            optimizer.step() #update weights based on gradients and learning rate

            train_loss += loss.item() / len(train_loader) #add loss to total loss

            #Need to find index of max value
            pred = prediction.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        #calculate statistics
        train_acc = 100 * correct / total
        test_loss, test_acc = test()
        print(f"{epoch+1:<10}{train_loss:<15.4f}{train_acc:<15.4f}{test_loss:<15.4f}{test_acc:<15.4f}")

        #save model if most efficient yet
        if len(acc_list) == 0 or test_acc > max(acc_list):
            save_model(epoch, test_acc)
        acc_list.append(test_acc)


# ------ maybe some helper functions -----------
def test():
    model.eval()  # switch the model to evaluation mode

    #save for finding accuracy
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad(): #don't save gradients for testing data
        for input, target in test_loader:
            prediction = model(input) #get output for each image for each class
            loss = loss_func(prediction, target)
            total_loss += loss.item()

            #find index of max value
            max_values, max_indices = prediction.max(1) 
            predicted = max_indices 


            total += predicted.size(0)
            correct += predicted.eq(target).cpu().sum().item()
    test_loss = total_loss / len(test_loader)
    test_accuracy = 100 * correct/total
    return test_loss, test_accuracy


def test_one_img(img_path):
    CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer","dog", "frog", "horse", "ship", "truck"]

    load_model(model)
    model.eval()  # switch the model to evaluation mode

    #create transformations to be applied to image
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    
    #open image using PIL library
    try:
        img = Image.open(img_path).convert("RGB")
    except:
        print("Error loading image")
        return

    #apply transformations, add new dimension at idx 0, for batch
    img_tensor = transform(img).unsqueeze(0) 

    with torch.no_grad():
        outputs = model(img_tensor)
        max_values, max_index = outputs.max(1)
        class_prediction = CIFAR10_CLASSES[max_index.item()]
    print(f'prediction result: {class_prediction}')



def save_model(epoch, accuracy): 
    if not os.path.exists("model"):
        os.makedirs("model")

    model_path = f"./model/cifar10_best.pt"

    #clear previous model
    if os.path.exists(model_path):
        os.remove(model_path)

    #model state dict saves the model's weights, named after epoch and accuracy
    torch.save(model.state_dict(), f"./model/cifar10_best.pt")



def load_model(model): #use this in second part of hw
    model.load_state_dict(torch.load("./model/cifar10_best.pt"))
    model.eval()


def main():
  #if no arguments dont run
  if len(sys.argv) < 2:
    print("Invalid number of arguments")
    sys.exit()
  elif sys.argv[1] == "train":
    train()
  elif sys.argv[1] == "test":
    if len(sys.argv) < 3:
      print("Invalid number of arguments")
      sys.exit()
    else:
      test_one_img(sys.argv[2])
  else:
    print("Invalid commands")
    sys.exit()


if __name__ == "__main__":
  main()

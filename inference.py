from utils import torch, torchvision, plt, Image
from model import model
from data import testloader

classes = ["anne-hathaway", "blake-lively", "brad-pitt", "christian-bale", "emily-blunt", "james-bond",
"jennifer-garner", "leonardo-dicaprio", "margot-robbie", "robert-downey-jr"]

# loading model
<<<<<<< HEAD
model = model

# model.load_state_dict(torch.load(r"/home/abdullah/Desktop/celeb-classification/model.pth"))

model.load_state_dict(torch.load("model.pth"))
=======
model = Model()
<<<<<<< HEAD
model.load_state_dict(torch.load(r"/home/abdullah/Desktop/celeb-classification/model.pth"))
=======
model.load_state_dict(torch.load(r"C:\Users\D2\Desktop\celebrity-classification\model.pth"))
>>>>>>> cd1e927ba8f4fc72f7d91b4e0998ff2d68d63ad2
>>>>>>> 43f5170ff64629cbe5e0f23f5a3fa4a26383bde2
model.eval()


# test.py
correct_pred = {classname: 0 for classname in classes} # Initialize as dictionaries
total_pred = {classname: 0 for classname in classes}
with torch.no_grad():
  for images, labels in testloader:
    outputs = model(images)
    _, predictions = torch.max(outputs.data, 1)
    for prediction, label in zip(predictions, labels):
      if label == prediction:
        correct_pred[classes[label.item()]] += 1
      total_pred[classes[label.item()]] += 1

for classname, correct_count in correct_pred.items():
  accuracy = 100 * float(correct_count) / total_pred[classname]
  print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")
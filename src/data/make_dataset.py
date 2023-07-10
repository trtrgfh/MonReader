from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from PIL import Image
import os

torch.manual_seed(1234)
transform = transforms.Compose([
  transforms.Resize((224, 224)),  # Resize the image to a desired size
  transforms.ToTensor(),  # Convert the image to a tensor
  # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Load the images from the folders and create the dataset
def load_images_from_folder(folder, label):
  images = []
  for filename in os.listdir(folder):
    if filename.endswith(".jpg"):
      file_path = os.path.join(folder, filename)
      image = Image.open(file_path).convert("RGB")
      images.append(transform(image))
  labels = [label] * len(images)
  return images, labels

# Get training data
train_flip, train_flip_labels = load_images_from_folder("/content/images/training/flip", 1)
train_non_flip, train_non_flip_labels = load_images_from_folder("/content/images/training/notflip", 0)

train_images = train_flip + train_non_flip
train_labels = train_flip_labels + train_non_flip_labels

# Get testing data
test_flip, test_flip_labels = load_images_from_folder("/content/images/testing/flip", 1)
test_non_flip, test_non_flip_labels = load_images_from_folder("/content/images/testing/notflip", 0)

test_images = test_flip + test_non_flip
test_labels = test_flip_labels + test_non_flip_labels

# m = # of examples, train_data[m][0] is the features, train_data[m][1] is the labels
train_data = torch.utils.data.TensorDataset(torch.stack(train_images), torch.LongTensor(train_labels))
test_data = torch.utils.data.TensorDataset(torch.stack(test_images), torch.LongTensor(test_labels))
batch_size = 32

# Create a DataLoader to load the data in batches for training
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Create a testing DataLoader, no need to shuffle for testing
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

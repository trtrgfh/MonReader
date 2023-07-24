import torchvision
weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2
mobile_model = torchvision.models.mobilenet_v2(weights=weights)
torch.manual_seed(1234)

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()

test_loss, test_acc, f1, final_y_pred = test_model(mobile_model, loss_fn)
print(f"Test_loss: {test_loss}, Test_acc: {test_acc}, f1: {f1}")

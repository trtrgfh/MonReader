def test_model(model, loss_fn):
    # Testing
    test_loss, test_acc = 0, 0
    final_y_pred = []
    model.eval()
    with torch.inference_mode():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = loss_fn(outputs, labels)
            test_loss += loss
            test_acc += (torch.eq(labels, outputs.argmax(dim=1)).sum().item() / len(outputs)) * 100
            
            final_y_pred.extend(outputs.argmax(dim=1).cpu().numpy())
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        f1 = f1_score(test_labels, final_y_pred)

    return test_loss, test_acc, f1, final_y_pred

# Load model
load_model = pickle.load(open("cnn_model.pkl", "rb"))
loss_fn = nn.CrossEntropyLoss()
test_loss, test_acc, f1, final_y_pred = test_model(model, loss_fn)

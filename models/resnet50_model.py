class IdentityBlock(nn.Module):
  def __init__(self, in_channels, f, filters, stride=1):
    super(IdentityBlock, self).__init__()
    F1, F2, F3 = filters

    self.conv1 = nn.Conv2d(in_channels, F1, kernel_size=1, padding=0, stride=stride, bias=False)
    self.batch_norm1 = nn.BatchNorm2d(F1)
    self.conv2 = nn.Conv2d(F1, F2, kernel_size=f, padding=f//2, stride=stride, bias=False)
    self.batch_norm2 = nn.BatchNorm2d(F2)
    self.conv3 = nn.Conv2d(F2, F3, kernel_size=1, padding=0, stride=stride, bias=False)
    self.batch_norm3 = nn.BatchNorm2d(F3)

    self.stride = stride
    self.relu = nn.ReLU()

    # kaiming_uniform_ initializer is default
    # nn.init.kaiming_uniform_(self.conv1.weight)
    # nn.init.kaiming_uniform_(self.conv2.weight)
    # nn.init.kaiming_uniform_(self.conv3.weight)

  def forward(self, x):
    shortcut = x

    out = self.conv1(x)
    out = self.batch_norm1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.batch_norm2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.batch_norm3(out)

    out += shortcut
    out = self.relu(out)
    return out

class ConvolutionalBlock(nn.Module):
  def __init__(self, in_channels, f, filters, stride=2):
    super(ConvolutionalBlock, self).__init__()
    F1, F2, F3 = filters

    self.conv1 = nn.Conv2d(in_channels, F1, kernel_size=1, padding=0, stride=stride, bias=False)
    self.batch_norm1 = nn.BatchNorm2d(F1)
    self.conv2 = nn.Conv2d(F1, F2, kernel_size=f, padding=f//2, stride=1, bias=False)
    self.batch_norm2 = nn.BatchNorm2d(F2)
    self.conv3 = nn.Conv2d(F2, F3, kernel_size=1, padding=0, stride=1, bias=False)
    self.batch_norm3 = nn.BatchNorm2d(F3)

    self.shortcut = nn.Conv2d(in_channels, F3, kernel_size=1, padding=0, stride=stride, bias=False)
    self.shortcut_bn = nn.BatchNorm2d(F3)

    self.stride = stride
    self.relu = nn.ReLU()

  def forward(self, x):
    shortcut = x

    out = self.conv1(x)
    out = self.batch_norm1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.batch_norm2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.batch_norm3(out)

    shortcut = self.shortcut(shortcut)
    shortcut = self.shortcut_bn(shortcut)

    out += shortcut
    out = self.relu(out)
    return out

# Define the ResNet50 model
class ResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvolutionalBlock(64, f=3, filters=[64, 64, 256], stride=1)
        self.id_block1 = IdentityBlock(256, 3, [64, 64, 256])
        self.id_block2 = IdentityBlock(256, 3, [64, 64, 256])

        self.conv_block2 = ConvolutionalBlock(256, f=3, filters=[128, 128, 512], stride=2)
        self.id_block3 = IdentityBlock(512, 3, [128, 128, 512])
        self.id_block4 = IdentityBlock(512, 3, [128, 128, 512])
        self.id_block5 = IdentityBlock(512, 3, [128, 128, 512])

        self.conv_block3 = ConvolutionalBlock(512, f=3, filters=[256, 256, 1024], stride=2)
        self.id_block6 = IdentityBlock(1024, 3, [256, 256, 1024])
        self.id_block7 = IdentityBlock(1024, 3, [256, 256, 1024])
        self.id_block8 = IdentityBlock(1024, 3, [256, 256, 1024])
        self.id_block9 = IdentityBlock(1024, 3, [256, 256, 1024])
        self.id_block10 = IdentityBlock(1024, 3, [256, 256, 1024])

        self.conv_block4 = ConvolutionalBlock(1024, f=3, filters=[512, 512, 2048], stride=2)
        self.id_block11 = IdentityBlock(2048, 3, [512, 512, 2048])
        self.id_block12 = IdentityBlock(2048, 3, [512, 512, 2048])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.conv_block1(x)
        x = self.id_block1(x)
        x = self.id_block2(x)

        x = self.conv_block2(x)
        x = self.id_block3(x)
        x = self.id_block4(x)
        x = self.id_block5(x)

        x = self.conv_block3(x)
        x = self.id_block6(x)
        x = self.id_block7(x)
        x = self.id_block8(x)
        x = self.id_block9(x)
        x = self.id_block10(x)

        x = self.conv_block4(x)
        x = self.id_block11(x)
        x = self.id_block12(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.output(x)

        return x

resnet50_model = ResNet50(2)
torch.manual_seed(1234)

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5

train_model(resnet50_model, loss_fn, optimizer, num_epochs, "resnet50_model.pkl")

import matplotlib.pyplot as plt

mismatched_indices = np.where(np.array(final_y_pred) != np.array(test_labels))[0]
random_mismatched = np.random.choice(mismatched_indices, size=10, replace=False)

fig, axs = plt.subplots(2, 5, figsize=(12, 6))

# Assuming test_images is a list of tensor images
for i, index in enumerate(random_mismatched):
    image = test_images[index]

    # Convert tensor to NumPy array
    np_image = image.numpy()

    # Transpose dimensions to (height, width, channels)
    np_image = np.transpose(np_image, (1, 2, 0))

    # Plot the image in the current subplot
    ax = axs[i // 5, i % 5]
    ax.imshow(np_image)
    ax.axis('off')

    # Set the label on top of the image
    ax.set_title(f"{test_labels[index]}, {final_y_pred[index]}")

plt.tight_layout()
plt.show()

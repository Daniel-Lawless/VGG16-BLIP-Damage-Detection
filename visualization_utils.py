import matplotlib.pyplot as plt

# Class for visualization
class visualization_utils_images:

    # Function to perform the plotting of an image along with its true and predicted label
    def _plot_on_subplot(self, ax ,image, true_label, predicted_label):
        """A function to plot a single image on a given axis ax[i]."""
        ax.imshow(image)
        ax.set_title(f"True label: {true_label}\nPred label: {predicted_label}")
        ax.axis("off")

    # Function to show an image, and it's corresponding true label, along with its predicted label.
    def test_model_on_image(self, test_generator, model, num_to_plot):
        """A function to plot multiple images side by side in one row."""
        # Extract a batch of images and their corresponding labels.
        # We use the next() function to get the values it produces, since without it,
        # we would just be assigning test_images, test_labels to an ImageDataGenerator object.
        test_images, test_labels = next(test_generator)

        # Use the model to make predictions on the test_images. Outputs a 2d array (# of images, 1)
        predictions = model.predict(test_images)

        # If a probability in predictions is greater than 0.5 it gets a True value,
        # if it's less, it gets a False value, then we convert it to an int and flatten the
        # 2d array to a 1d array
        predicted_classes = (predictions > 0.5).astype(int).flatten()

        # Extract the class indices dictionary {"crack" : 0, "dent": 1}
        class_indices = test_generator.class_indices

        # Invert the dictionary to give {0 : "crack", 1: "dent"} This will make looking up
        # the corresponding class based on the prediction the model makes easier,since the predicted
        # class will be 0 or 1, not crack or dent.
        class_names = {value: key for key, value in class_indices.items()}

        # Define the number of axes. We want to plot them side by side in a row.
        figure, axes = plt.subplots(1, num_to_plot, figsize=(15, 5))
        for i in range(num_to_plot):
            # Specify the image to display and its corresponding true and pred label based on the index.
            image_to_plot = test_images[i]
            true_label = class_names[test_labels[i]]
            predicted_label = class_names[predicted_classes[i]]

            # Plot the current image on axes[i]
            self._plot_on_subplot(axes[i], image_to_plot, true_label, predicted_label)

        # Show the image.
        plt.tight_layout()
        plt.show()
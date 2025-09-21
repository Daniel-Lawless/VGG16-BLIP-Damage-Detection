import numpy as np
import tensorflow as tf
from PIL import Image

class BlipCaptionSummaryLayer(tf.keras.layers.Layer):

    # Initialize the model
    def __init__(self, processor, model, **kwargs):
        # This makes it so we don't have to specify all arguments for the parent constructor
        super().__init__(**kwargs)
        # initialize processor and model.
        self.processor = processor
        self.model = model

    def call(self, image_path, task):
        # tf.py_function() expects 3 arguments, the Python function we want to run, a list of input tensors to
        # pass to that function, and the output of that function. We have to do this because tensorflow builds a
        # computational graph of operations to perform operations effectively, however, it does not understand
        # Python libraries like Pillow or Numpy. By using this function we are saying, run this function,
        # expect this input, and expect this output to continue with.
        return tf.py_function(func=self.process_image, inp=[image_path, task], Tout=tf.string)


    def process_image(self, image_path, task):
        try:
            # image_path.numpy() convert the tensorflow tensor and converts into it's
            # equivalent numpy object. The result is a byte string, the decode("utf-8") converts
            # it from a byte string into human-readable text.
            image_path_str = image_path.numpy().decode("utf-8")

            # Open the image and convert it to an RGB format, so it ensures it has 3 channels.
            image = Image.open(image_path_str).convert("RGB")

            if task.numpy().decode("utf-8") == "caption":
                # Encourages the model to produce a simple, direct response
                prompt = "This is an image of"
            else:
                # Encourages a mode descriptive response
                prompt = "This is a detailed image showing"

            # The processor takes our image, resizes it to the dimensions the BLIP model expects, normalizes
            # the pixel values, and converts it to a tensor. It takes our text prompt, and passes it into
            # a tokenizer, it then returns the final prepared data as tensorflow tensors. inputs is a dictionary.
            inputs = self.processor(images=image, text=prompt, return_tensors="tf")

            # Pass our inputs into the model. the ** unpacks the inputs dictionary that our processor created
            # (e.g., pixel_values=..., input_ids=...) and passes them as keyword arguments to the generate method.
            outputs = self.model.generate(**inputs)

            # Converts the numerical representation of the caption/summary generated from the model into
            # human-readable text. The self.processor.decode() method does the opposite of the tokenizer,
            # it takes the token ID's and uses them to reconstruct the sentence. The generator can produce
            # multiple sequences, so we use outputs[0], to extract the first and only sequence in this case.
            # Models use special tokens for internal logic, the final argument removes them from our decoded
            # output.
            result = self.processor.decode(outputs[0], skip_special_tokens=True)

            # Return the decoded output
            return result

        # If there is an error in our try block, print an error.
        except Exception as e:
            print(f"Error: {e}")
            print("Error processing image")


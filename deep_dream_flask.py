"""
#
# Cloud for ML Final Project
# Divya Juneja
# deep_dream.py
#
"""
import os
from PIL import Image

import torch
from torch.optim import SGD
from torchvision import models
import flask

from misc_functions import preprocess_image, recreate_image, save_image

app = flask.Flask(__name__)

#port = int(os.getenv("PORT"))


class DeepDream():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter, im_path):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        # Generate a random image
        self.created_image = im_path.convert('RGB')
        # Hook the layers to get result of the convolution
        self.hook_layer()
        # Create the folder to export images if not exists
        if not os.path.exists('generated'):
            os.makedirs('generated')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]

        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

    def dream(self):
        # Process image and return variable
        self.processed_image = preprocess_image(self.created_image, True)
        plt.rcParams['figure.figsize'] = [24.0, 14.0]
        
        # Define optimizer for the image
        # Earlier layers need higher learning rates to visualize whereas layer layers need less
        optimizer = SGD([self.processed_image], lr=12,  weight_decay=1e-4)
        # Create a figure and plot the image
        fig, pl = plt.subplots(1, 1)
        for i in range(1, 10):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = self.processed_image
            for index, layer in enumerate(self.model):
                # Forward
                x = layer(x)
                # Only need to forward until we the selected layer is reached
                if index == self.selected_layer:
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image)
            # Save image every 20 iteration
            if i % 10 == 0:
                print(self.created_image.shape)
                im_path = 'generated/ddream_l' + str(self.selected_layer) + \
                    '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(self.created_image, im_path)
        plt.savefig('output.jpg')
        return send_file('output.jpg', mimetype='image/jpg')

@app.route('/main', methods=['POST'])
def main():
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            im = flask.request.files["image"].read()
            im = Image.open(io.BytesIO(im))
    cnn_layer = 34
    filter_pos = 94

    # Fully connected layer is not needed
    pretrained_model = models.vgg19(pretrained=True).features
    dd = DeepDream(pretrained_model, cnn_layer, filter_pos, im)
    # This operation can also be done without Pytorch hooks
    # See layer visualisation for the implementation without hooks
    dd.dream()
    return flask.jsonify({'class_name': class_name})
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)
    

    

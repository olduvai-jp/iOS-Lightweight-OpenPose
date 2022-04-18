import torch
from torch import nn
from torchvision import transforms
import coremltools as ct
from PIL import Image
import platform
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname('models')))
sys.path.append(os.path.join(os.path.dirname('modules')))
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state

def load_image(filename, size=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.LANCZOS)
    return img

def coreml_rename_output(model):
    # Rename the output of the model.
    spec = model.get_spec()
    feature_names = [
        'heat_map_1',
        'paf_1',
        'heat_map_2',
        'paf_2'
    ]
    for idx in range(len(spec.description.output)):
        old_name = spec.description.output[idx].name
        new_name = feature_names[idx]
        print(f'{old_name} -> {new_name}')
        ct.utils.rename_feature(spec, old_name, new_name)
    
    model = ct.models.MLModel(spec)
    return model

class ImgWrapNet(nn.Module):
    """Add a layer of the normalization to the neural network"""
    def __init__(self, checkpoint_path, scale=256.):
        super().__init__()
        self.scale = scale

        pose_model = PoseEstimationWithMobileNet()
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        load_state(pose_model, state_dict)
        self.pose_model = pose_model

    def forward(self, x):
        x = (x+(-128)) / self.scale
        x = self.pose_model(x)
        return x
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint-path', type=str, default='resources/checkpoint_iter_370000.pth',
                        help='path to the checkpoint')
    parser.add_argument('-o', '--onnx-output', type=str, default='pose_{}.onnx',
                        help='name of output model in ONNX format')
    parser.add_argument('-m', '--mlmodel-output', type=str, default='pose_{}.mlmodel',
                        help='name of output model in CoreML format')
    parser.add_argument('-i', '--image-path', type=str, default='data/preview.jpg', help='path to the image')
    parser.add_argument('-s', '--shape', type=int, default=368, help='width and height of the input image')
    args = parser.parse_args()

    assert args.onnx_output.endswith('.onnx'), "Export model file should end with .onnx"
    
    # prepare the input image
    content_file = args.image_path
    content_image = load_image(content_file, args.shape)
    img_tensor = transforms.functional.to_tensor(content_image)
    example_input = img_tensor.unsqueeze(0)

    torch_model = ImgWrapNet(args.checkpoint_path)
    # Set the model in evaluation mode.
    torch_model.eval()

    # Trace the model.
    traced_model = torch.jit.trace(torch_model, example_input)

    #torchscript_prediction(traced_model, example_input)

    # convert to CoreML
    mlmodel_path = args.mlmodel_output.format(args.shape)
    coreml_model = ct.convert(traced_model, inputs=[ct.ImageType(name="input_image", shape=example_input.shape)])
    coreml_model = coreml_rename_output(coreml_model)

    # Save to disk.
    coreml_model.save(mlmodel_path)
 
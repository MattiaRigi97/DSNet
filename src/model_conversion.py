
# python model_conversion.py anchor-free --model-dir ../models/pretrain_af_basic/ --splits ../splits/tvsum.yml ../splits/summe.yml

## PACKAGE
import logging
from pathlib import Path

# Helpers functions
from helpers import init_helper, data_helper
from modules.model_zoo import get_model

# Conversion libraries
import tensorflow as tf
import torch
from torch.autograd import Variable
import onnx
from onnx_tf.backend import prepare


logger = logging.getLogger()


def conversion(model):
    
    model.eval()
    with torch.no_grad():

        dummy_input = Variable(torch.randn(1, 1, 1024))

        if torch.cuda.is_available():
            dummy_input = dummy_input.to('cuda')
            model.to('cuda')

        torch.onnx.export(model, dummy_input, r"C:\Users\matti\github\DSNet\models\converted_model\model.onnx")

        onnx_model = onnx.load(r"C:\Users\matti\github\DSNet\models\converted_model\model.onnx")
        tf_rep = prepare(onnx_model) 

        # Input nodes to the model
        print('inputs:', tf_rep.inputs)
        print('outputs:', tf_rep.outputs)
        print('tensor_dict:')
        print(tf_rep.tensor_dict)

        tf_rep.export_graph(r"C:\Users\matti\github\DSNet\models\converted_model\model.pb")    
        
        # Convert the model
        converter = tf.lite.TFLiteConverter.from_saved_model(r"C:\Users\matti\github\DSNet\models\converted_model\model.pb") # path to the SavedModel directory
        tflite_model = converter.convert()

        with open(r"C:\Users\matti\github\DSNet\models\converted_model\model.tflite", 'wb') as f:
            f.write(tflite_model)


def main():

    args = init_helper.get_arguments()
    print(args)

    print(args.model)

    # Load the model
    model = get_model(args.model, **vars(args))
    model = model.eval().to(args.device)
    print(args.splits)
    
    for split_path in args.splits:
        split_path = Path(split_path)
        splits = data_helper.load_yaml(split_path)

        # For each split (train/test) in dataset.yml file (x5)
        for split_idx, split in enumerate(splits):
            
            # Load the model from the checkpoint folder
            ckpt_path = data_helper.get_ckpt_path(args.model_dir, split_path, split_idx)
            state_dict = torch.load(str(ckpt_path), map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)

    conversion(model)

    print("Conversion done")
   
    
if __name__ == '__main__':
    main()
import argparse
import torch

def convert_model(input_path, output_path):
    print(f"Loading original model from {input_path}...")
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("The 'ultralytics' library is required to extract weights. Install it via 'pip install ultralytics'.")
        
    model = YOLO(input_path)
    pytorch_model = model.model
    yaml_config = pytorch_model.yaml if hasattr(pytorch_model, 'yaml') else None
    metadata = {
        'nc': pytorch_model.nc,
        'names': pytorch_model.names,
        'stride': pytorch_model.stride.tolist() if hasattr(pytorch_model, 'stride') else [8, 16, 32]
    }
    state_dict = pytorch_model.state_dict()
    
    torch.save({
        'state_dict': state_dict,
        'metadata': metadata,
        'yaml': yaml_config
    }, output_path)
    print(f"Clean model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Ultralytics .pt model to clean .pth format without AGPL-3 dependencies.")
    parser.add_argument("input", help="Path to original .pt model")
    parser.add_argument("output", help="Path to save clean .pth model")
    args = parser.parse_args()
    
    convert_model(args.input, args.output)

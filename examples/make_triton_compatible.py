"""
This script makes the ONNX model compatible with Triton inference server.
"""

import sys
import numpy as np
import onnx
import onnxruntime as ort
import onnx_graphsurgeon as gs


def add_squeeze(graph, speed_input, speed_unsqueezed):
    """
    Add squeeze operation to the speed input to change shape from [batch_size, 1] to [batch_size]
    """
    # Create a squeeze node
    squeeze_node = gs.Node(
        op="Squeeze",
        name="speed_squeeze",
        inputs=[speed_unsqueezed],
        outputs=[gs.Variable(name="speed_squeezed", dtype=speed_unsqueezed.dtype)]
    )
    
    ## Find first node that has speed_unsqueezed as input
    insert_idx = 0
    for idx, node in enumerate(graph.nodes):
        for i, input_name in enumerate(node.inputs):
            if input_name.name == speed_unsqueezed.name:
                insert_idx = idx
                break
        if insert_idx != 0:
            break
    
    ## Add squeeze node to the graph
    insert_idx = min(0, insert_idx - 1)
    graph.nodes.insert(insert_idx, squeeze_node)
    
    # Update the speed input to point to the squeezed output
    for node in graph.nodes:
        for i, input_name in enumerate(node.inputs):
            if input_name.name == speed_input.name and not node.name == "speed_squeeze":
                node.inputs[i] = squeeze_node.outputs[0]
                
    return graph


def main():
    import argparse
    import os
    from kokoro.hf_utils import set_checkpoints_dir

    parser = argparse.ArgumentParser(description="Make ONNX model compatible with Triton")
    parser.add_argument("onnx_model_path", nargs="?", default=None, help="Path to input ONNX model")
    parser.add_argument(
        "--checkpoint-dir",
        "--checkpoint_dir",
        type=str,
        default=os.getenv("CHECKPOINT_DIR", "checkpoints"),
        help="Path to checkpoint directory (default: 'checkpoints' or $CHECKPOINT_DIR)",
    )
    args = parser.parse_args()
    set_checkpoints_dir(args.checkpoint_dir)

    if not args.onnx_model_path:
        candidate = os.path.join(args.checkpoint_dir, "onnx", "kokoro.onnx")
        if os.path.exists(candidate):
            args.onnx_model_path = candidate
        elif os.path.exists("onnx/kokoro.onnx"):
            args.onnx_model_path = "onnx/kokoro.onnx"
        else:
            parser.error("Usage: python make_triton_compatible.py <onnx_model_path>")

    onnx_model_path = args.onnx_model_path
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print("Model is valid")
    
    graph = gs.import_onnx(onnx_model)

    ## get input_id for speed
    speed_idx, speed = None, None
    for idx, input_ in enumerate(graph.inputs):
        if input_.name=="speed":
            speed_idx = idx
            speed = input_

    # Update the speed input to have shape [batch_size, 1]
    speed_unsqueezed = gs.Variable(name="speed", dtype=speed.dtype, shape=[speed.shape[0], 1])
    graph.inputs[speed_idx] = speed_unsqueezed
        
    ## Add squeeze to change speed shape from [batch_size, 1] to [batch_size]
    if speed is not None:
        print(f"Found speed input: {speed.name}")
        print(f"Found speed input shape: {speed.shape}")
        print(f"Found speed input dtype: {speed.dtype}")
        print(f"Found speed input: {speed}")
        print(f"Found speed input: {type(speed)}")
        graph = add_squeeze(graph, speed, speed_unsqueezed)
        
        # Export the modified graph back to ONNX
        modified_model = gs.export_onnx(graph)
        onnx.checker.check_model(modified_model)
        
        # Save the modified model
        output_path = onnx_model_path.replace('.onnx', '_triton.onnx')
        onnx.save(modified_model, output_path)
        print(f"Modified model saved to: {output_path}")
    else:
        print("Speed input not found in the model")


if __name__ == "__main__":
    main()

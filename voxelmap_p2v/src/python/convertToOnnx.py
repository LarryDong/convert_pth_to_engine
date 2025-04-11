import numpy as np
from tqdm import tqdm
import csv
from Point2VoxelNet import Point2VoxelNet # model

import torch
import pandas as pd
import onnxruntime as ort
import onnx





# ONNX 推理函数
def infer_with_onnx(onnx_path, points, p):
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1  # 设定单线程，避免亲和性错误
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    ort_session = ort.InferenceSession(onnx_path, sess_options)
    # 确保 points 和 p 是 numpy 数组
    points = points.cpu().numpy() if isinstance(points, torch.Tensor) else points
    p = p.cpu().numpy() if isinstance(p, torch.Tensor) else p

    # ort_inputs = {"points": points, "p": p}
    # return ort_session.run(None, ort_inputs)[0]

    ort_inputs = {"points": points, "p": p}
    p2v, weight, _ = ort_session.run(None, ort_inputs)
    return p2v, weight



if __name__ == '__main__':

    ## load data_new.csv
    data_filename = '/home/server/xec/voxelmap_p2v/data_new.csv'
    with open(data_filename, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    
    # Extract first line as p and remaining as points
    p = data[0]  # First line
    points = data[1:]  # Remaining 50 lines
    
    # Convert to float (assuming numeric data)
    p = [float(x) for x in p]
    points = [[float(x) for x in row] for row in points]
    
    # Convert to PyTorch tensors
    p = torch.tensor(p)
    points = torch.tensor(points)

    p = p.unsqueeze(0)
    points = points.unsqueeze(0)


    # for vis
    model_file = '/home/server/xec/voxelmap_p2v/best_voxel.pth'
    model = Point2VoxelNet(voxel_size=0.5).to('cpu')
    state_dict = torch.load(model_file, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict['model_state_dict'])



    #############################################################
    ## Convert to ONNX
    #############################################################

    device = torch.device("cpu")
    onnx_path = "/home/server/xec/voxelmap_p2v/best_voxel.onnx"

    dummy_points = torch.ones(1, 50, 3, device=device)
    dummy_p = torch.zeros(1, 3, device=device) 
    torch.onnx.export(
        model,
        (dummy_points, dummy_p),
        onnx_path,
        input_names=["points", "p"],
        output_names=["output_p2v", "output_weight", "feature"],
        opset_version=11
    )
    print(f"ONNX 模型已保存到 {onnx_path}")

    # 2.2 用 ONNX 推理
    p2v_onnx, weight_onnx = infer_with_onnx(onnx_path, points, p)
    print("------------------------- onnx output -------------------------")
    print("onnx输出:", p2v_onnx)
    print("onnx输出:", weight_onnx)
    print("--------------------------------------------------------------- ")


    #############################################################
    ## Compare with python's pth.
    #############################################################
    model.eval()
    with torch.no_grad():  # 关闭梯度计算
        p2v, weight, feat = model(points, p)  # 进行推理
    print("------------------------- onnx output -------------------------")
    print("pth输出:", p2v)
    print("pth输出:", weight)
    print("--------------------------------------------------------------- ")
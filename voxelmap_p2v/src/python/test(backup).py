import torch
import numpy as np
import csv
from Point2VoxelNet import Point2VoxelNet


if __name__ == "__main__":

    # load data_new.csv
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
    
    model.eval()
    with torch.no_grad():  # 关闭梯度计算
        p2v, weight, feat = model(points, p)  # 进行推理
    print(f"--> Python .pth predict p2v: {p2v}")

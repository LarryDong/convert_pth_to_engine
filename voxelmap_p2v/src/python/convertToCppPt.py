import torch
import sys
sys.path.append('/home/larry/codeGit/implict_voxel/src/feat_voxel_map')

from Model.Point2VoxelNet import Point2VoxelNet, MyLossFunction # model
import torch.nn as nn
import csv

if __name__ == "__main__":

    input_model = "/home/larry/codeGit/implict_voxel/src/feat_voxel_map/checkpoint/best_voxel.pth"
    output_model = "/home/larry/codeGit/implict_voxel/src/feat_voxel_map/checkpoint/best_voxel.pt"
    # for vis
    model = Point2VoxelNet(voxel_size=0.5).to('cpu')
    state_dict = torch.load(input_model, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()
    
    data_filename = '/home/larry/codeGit/implict_voxel/data/data_new.csv'
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
    
    traced_model = torch.jit.trace(model, (points, p))
    traced_model.save(output_model)     # must be saved by this method.

    print("Model saved")

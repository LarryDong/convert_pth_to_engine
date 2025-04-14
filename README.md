# convert_pth_to_engine
This code convert python's pth file to c++'s .engine


## Convert to C++ Engine
python's .pth -> .onnx -> c++'s .engine.


## Convert to C++'s libtorch
We can just use pytorch -> libtorch.

Attention:
- save `.pth` to `.pt` using torch.jit.trace
- load `.pt` to c++, and some conversion may necessary.


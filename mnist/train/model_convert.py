import torch 
import onnx 
# import tensorrt as trt  

device = torch.device('cuda:0') 

onnx_model = 'model.onnx' 
model = torch.load('cnn.pt').to(device)

class NaiveModel(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.pool = torch.nn.MaxPool2d(2, 2) 
 
    def forward(self, x): 
        return self.pool(x) 
 

 
# generate ONNX model 
# torch.onnx.export(NaiveModel(), 
torch.onnx.export(model, 
                  torch.randn(1, 1, 28, 28).to(device), 
                  onnx_model,
                  input_names=['input'], 
                  output_names=['output'],
                  opset_version=11)

# onnx_model = onnx.load(onnx_model)

# # create builder and network 
# logger = trt.Logger(trt.Logger.ERROR) 
# builder = trt.Builder(logger) 
# EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) 
# network = builder.create_network(EXPLICIT_BATCH) 
 
# # parse onnx 
# parser = trt.OnnxParser(network, logger) 

# if not parser.parse(onnx_model.SerializeToString()): 
#     error_msgs = '' 
#     for error in range(parser.num_errors): 
#         error_msgs += f'{parser.get_error(error)}\n' 
#     raise RuntimeError(f'Failed to parse onnx, {error_msgs}') 
 
# config = builder.create_builder_config() 
# config.max_workspace_size = 1<<20 
# profile = builder.create_optimization_profile() 
 
# profile.set_shape('input', [1,3 ,224 ,224], [1,3,224, 224], [1,3 ,224 ,224]) 
# config.add_optimization_profile(profile) 
# # create engine 
# with torch.cuda.device(device): 
#     engine = builder.build_engine(network, config) 
 
# with open('model.engine', mode='wb') as f: 
#     f.write(bytearray(engine.serialize())) 
#     print("generating file done!") 
 
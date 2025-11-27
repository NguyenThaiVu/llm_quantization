import os 
import torch 

input_size = 1024 * 8
output_size = 1024 * 8

X = torch.randn((input_size, output_size), device='cuda')
scale = 2.0

scale_vec = torch.Tensor([scale] * output_size)
scale_vec = scale_vec.to(device='cuda')

scale_matrix = torch.Tensor([scale] * input_size * output_size)
scale_matrix = scale_matrix.to(device='cuda')
scale_matrix = scale_matrix.view(input_size, output_size)

torch._dynamo.reset() # reset any previous dynamo state

# measure time for scalar scaling
# warm up
for i in range(5):
    Y = X * scale
    
torch.cuda.synchronize()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
for i in range(100):
    Y = X * scale
end_event.record()
torch.cuda.synchronize()
elapsed_time = start_event.elapsed_time(end_event)  # milliseconds
avg_time = elapsed_time / 100.0
print(f'Average time for scalar scaling: {avg_time} ms')

# measure time for vector scaling
# warm up
for i in range(5):
    Y = X * scale_vec[:, None]
    
torch.cuda.synchronize()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
for i in range(100):
    Y = X * scale_vec[:, None]
end_event.record()
torch.cuda.synchronize()

elapsed_time = start_event.elapsed_time(end_event)  # milliseconds
avg_time = elapsed_time / 100.0
print(f'Average time for vector scaling: {avg_time} ms')

# measure time for element-wise scaling
# warm up
for i in range(5):
    Y = X * scale_matrix

torch.cuda.synchronize()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
for i in range(100):
    Y = X * scale_matrix
end_event.record()
torch.cuda.synchronize()
elapsed_time = start_event.elapsed_time(end_event)  # milliseconds
avg_time = elapsed_time / 100.0
print(f'Average time for element-wise matrix scaling: {avg_time} ms')




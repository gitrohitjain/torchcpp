import torch
import torchvision

model = torchvision.models.resnet18()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
example = torch.rand(1, 3, 224, 224)

traced_script_module = torch.jit.trace(model, example)

# traced_script_module.save("traced_resnet_model.pt")

output = traced_script_module(torch.ones(1, 3, 224, 224))

print(output[0, :5])

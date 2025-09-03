from geoai.clay import Clay
import torch

clay_model = Clay(sensor_name="sentinel-2-l2a")


t1 = torch.rand((256, 256, 10))
embedding = clay_model.generate(t1)
print(embedding.shape)

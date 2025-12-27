from cirilla.Cirilla_model import HybridCirilla, HybridArgs

model = HybridCirilla(HybridArgs(n_layers=5, layer_pattern='5M'))
print(model.n_params / 1e6, "M")
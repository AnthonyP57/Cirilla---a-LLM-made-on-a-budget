from cirilla.Cirilla_model import HybridDecoder, HybridDecoderArgs

model = HybridDecoder(HybridDecoderArgs(
                    n_layers=5,
                    layer_pattern='AAAAA',
                    num_experts=4,
                    k=2))

print(model.n_params)
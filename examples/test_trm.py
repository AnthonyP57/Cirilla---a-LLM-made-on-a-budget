from cirilla.Cirilla_model import (
                            CirillaTRM,
                            TRMArgs,
                            trm_training_step,
                            trm_inference_step,
                            TrainingArgs,
                            CirillaTrainer,
                            CirillaTokenizer,
                            JSONLDataset,
                            # Encoder,
                            # EncoderArgs,
                            MixerArgs,
                            MLPMixer1D
                            )
from ema_pytorch import EMA
from types import MethodType
from functools import partial

# encoder = Encoder(EncoderArgs())
mixer = MLPMixer1D(MixerArgs())

model = CirillaTRM(mixer, TRMArgs())

targs = TrainingArgs(n_epoch=1000, save_checkpoint_min=9999, use_muon_optim=False)

trainer = CirillaTrainer(model, targs)

tokenizer = CirillaTokenizer(hub_url='AnthonyPa57/HF-torch-demo2')

dl = JSONLDataset(['./examples/data/example_bert.jsonl', './examples/data/example_bert.jsonl'],
                    shuffle_path=True, tokenizer=tokenizer, max_len=mixer.args.context_window)

max_recurrent_step = 16
halt_weight = 0.5
halt_thresh = 0.5

ema_model = EMA(
                model,
                beta=0.999,
                update_model_with_ema_every=1_000,
                forward_method_names=('predict',)
                )

train_method = partial(
    trm_training_step,
    max_recurrent_step=max_recurrent_step,
    halt_weight=halt_weight,
    halt_thresh=halt_thresh,
    ema_model=ema_model
)

inference_method = partial(
    trm_inference_step,
    max_recurrent_step=max_recurrent_step,
    halt_thresh=halt_thresh
)

trainer.training_step = MethodType(train_method, trainer)
trainer.inference_step = MethodType(inference_method, trainer)
trainer.criterion = None # the criterion is useless in that case

trainer.train(dl, dl)

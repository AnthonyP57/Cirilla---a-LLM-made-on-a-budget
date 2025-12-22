import os
import sys
import uuid
import glob
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from megablocks.layers.moe import clear_load_balancing_loss
from megablocks.layers.router import clear_router_zloss, batched_router_zloss
from cirilla.Cirilla_model import load_balancing_loss

with open(sys.argv[0]) as f:
    code = f.read()

def _register_hooks(model, optimizer_dict):
    
    def optimizer_hook(parameter):
        optimizer_dict[parameter].step()
        optimizer_dict[parameter].zero_grad(set_to_none=True)

    for p in model.parameters():
        if p in optimizer_dict:
            p.register_post_accumulate_grad_hook(optimizer_hook)
        else:
            print(f"Unknown param of shape: {p.shape}")

from cirilla.LLM_pieces import get_activation

moun_optim = get_activation("motif-technologies/optimizer")

def _fuse_optim(model, use_muon_optim, optim, lr, weight_decay):

    if not use_muon_optim:

        optimizer_by_name = {}
        for name, p in model.named_parameters():
            optimizer_by_name[name] = optim([p], lr=lr, weight_decay=weight_decay, fused=True, foreach=False)

        params_by_name = dict(model.named_parameters())
        optimizer_dict = {params_by_name[name]: opt for name, opt in optimizer_by_name.items()}

    else:
        get_default_muon_param_groups = moun_optim.muon.get_default_muon_param_groups
        muon_param_groups = get_default_muon_param_groups(model)

        optimizer_by_name = {}
        for name, p in model.named_parameters():

            if name in muon_param_groups[0]['names']: # matrices
                group = {
                    "params": [p],
                    "names": [name],
                    "use_muon": True
                }
                optimizer_by_name[name] = moun_optim.Muon([group], lr=lr, weight_decay=weight_decay)

            else: # biases, LayerNorm weights
                optimizer_by_name[name] = optim([p], lr=lr, weight_decay=weight_decay)

        params_by_name = dict(model.named_parameters())
        optimizer_dict = {params_by_name[name]: opt for name, opt in optimizer_by_name.items()}

    _register_hooks(model, optimizer_dict)

    return optimizer_dict


def get_optims(model, use_muon_optim, optim, lr, weight_decay):

    if use_muon_optim:
        get_default_muon_param_groups = moun_optim.muon.get_default_muon_param_groups
        muon_param_groups = get_default_muon_param_groups(model)

        moptim = moun_optim.Muon(muon_param_groups, lr=lr, weight_decay=weight_decay)

        rest_of_params = [p for n, p in model.named_parameters() if n not in muon_param_groups[0]['names']]

        roptim = optim(rest_of_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))

        return moptim, roptim
    
    else:
        return optim(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print(
            "---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README"
        )
        print(
            "---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try"
        )
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]  # number of tokens (claimed)
    return ntok  # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]  # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert (
            len(self.files) > 0
        ), f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = np.int64(0)
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(
            f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files"
        )

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):  # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

VAL_TOKENS = 1_000_000  # how many tokens of validation data. It's important to keep this fixed for consistent comparisons

def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


if __name__ == "__main__":
    import time
    import argparse

    print0(f"Running pytorch {torch.version.__version__}")

    parser = argparse.ArgumentParser()
    # file system input / output
    parser.add_argument(
        "--input_bin",
        type=str,
        default="data/fineweb10B/fineweb_train_*.bin",
        help="input .bin to train on",
    )
    parser.add_argument(
        "--input_val_bin",
        type=str,
        default="data/fineweb10B/fineweb_val_*.bin",
        help="input .bin to eval validation loss on",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="output directory to which to write logs and checkpoints",
    )
    # token layout for each step of the optimization
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="batch size, in units of #batch dimensions",
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        type=int,
        default=1,
        help="number of gradient accumulation steps",
    )
    parser.add_argument(
        "--sequence_length", type=int, default=64, help="sequence length"
    )
    # workload (number of steps)
    parser.add_argument(
        "--num_iterations", type=int, default=10, help="number of iterations to run"
    )
    # optimization
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="learning rate warmup iterations",
    )
    parser.add_argument(
        "--warmup_iters", type=int, default=0, help="learning rate warmup iterations"
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    # evaluation
    parser.add_argument(
        "--val_loss_every",
        type=int,
        default=0,
        help="every how mant steps to evaluate val loss?",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=16,
        help="how many batches of val to average?",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5000,
        help="every how many steps to save the checkpoint",
    )
    parser.add_argument(
        "--warmdown_iters",
        type=int,
        default=0,
        help="learning rate warmdown iterations",
    )

    args = parser.parse_args()

    # args error checking and convenience variables
    B, T = args.batch_size, args.sequence_length

    assert torch.cuda.is_available()

    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    assert (
        args.grad_accumulation_steps % ddp_world_size == 0
    ), "grad_accumulation_steps must be divisible by world size"
    args.grad_accumulation_steps //= (
        ddp_world_size  # each gpu does its fraction of the work
    )
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = 0  # each process gets the exact same seed
    print(f"using device: {device}")

    tokens_per_iter = B * T * ddp_world_size * args.grad_accumulation_steps
    print0(f"tokens per iteration: {tokens_per_iter:,}")

    # load tokens
    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = None
    tokens_per_iter_val = args.val_batch_size * T * ddp_world_size
    assert VAL_TOKENS % tokens_per_iter_val == 0
    val_steps = VAL_TOKENS // tokens_per_iter_val

    val_loader = DistributedDataLoader(
        args.input_val_bin, args.val_batch_size, T, ddp_rank, ddp_world_size
    )

    def get_lr(it):
        assert it <= args.num_iterations
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.learning_rate * (it + 1) / args.warmup_iters
        # 2) constant lr for a while
        elif it < args.num_iterations - args.warmdown_iters:
            return args.learning_rate
        # 3) linear warmdown
        else:
            decay_ratio = (args.num_iterations - it) / args.warmdown_iters
            return args.learning_rate * decay_ratio

    # init the model from scratch
    num_vocab = 50257

    from cirilla.Cirilla_model import Cirilla, Args

    model_args = Args(vocab_size=num_vocab,
                        n_layers=11,
                        context_window=args.sequence_length,
                        out_bias=True,
                        window_size=args.sequence_length,
                        d_ff=1024, dim=512,
                        num_experts=3, k=2,
                        n_heads=8, n_kv_heads=8,
                        moe_type='pytorch',
                        output_moe_weights=True,
                        use_sparse=False)
    
    model = Cirilla(model_args)

    for param in model.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=torch.nn.init.calculate_gain('leaky_relu'))

    print0(f' Number of parameters: {model.n_params/1e6:.2f} M')

    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    # optim_dict = _fuse_optim(model, True, torch.optim.AdamW, args.learning_rate, args.weight_decay)
    muon_opt, adam_opt = get_optims(model, True, torch.optim.AdamW, args.learning_rate, args.weight_decay)
    # scheduler_muon = torch.optim.lr_scheduler.ReduceLROnPlateau(muon_opt, factor=0.5, patience=2)
    # scheduler_adam = torch.optim.lr_scheduler.ReduceLROnPlateau(adam_opt, factor=0.5, patience=2)

    x, y = train_loader.next_batch()
    
    # # torch.cuda.memory._record_memory_history(
    # #     max_entries=100_000
    # # )

    for _ in range(2):
        torch.compiler.cudagraph_mark_step_begin()
        for _ in range(args.grad_accumulation_steps):
            # clear_load_balancing_loss()
            # clear_router_zloss()
            logits, moe_weight_list = model.pred(x)
            # logits = model.pred(x)

            lb_losses = [
                load_balancing_loss(w, num_experts=model.args.num_experts, top_k=model.args.k)
                for w in moe_weight_list
                    ]
            lb_loss = torch.stack(lb_losses).mean()

            loss = (F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1
            ) + 0.01 * lb_loss) / args.grad_accumulation_steps
            
            loss.backward()

        muon_opt.step()
        adam_opt.step()

        muon_opt.zero_grad(set_to_none=True)
        adam_opt.zero_grad(set_to_none=True)

    # torch.cuda.memory._dump_snapshot(f"vram_usage.pickle")

    # del x, y, loss, logits
    torch.cuda.empty_cache()

    run_id = str(uuid.uuid4())

    # create the logging directory if it does not exist
    logfile = None
    if master_process and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, "%s.log" % run_id)
        # create the log file "main.log" inside it, and wipe it clean
        with open(logfile, "w") as f:
            pass

    training_time_ms = 0.0
    # start the clock
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # begin training

    for step in range(args.num_iterations + 1):
        last_step = step == args.num_iterations

        # once in a while evaluate the validation dataset
        if args.val_loss_every > 0 and (step % args.val_loss_every == 0 or last_step):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            model.eval()
            val_loader.reset()  # reset the val loader so that it starts from the beginning
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(val_steps):  # always fiexed number of validation steps
                    x_val, y_val = val_loader.next_batch()
                    logits, _ = model.pred(x_val)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), y_val.view(-1), ignore_index=-1)
                    
                    val_loss += loss
                # dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                val_loss /= val_steps
            # log to console and to file
            print0(f"step:{step}/{args.num_iterations} | val loss {val_loss:.6f}")
            if master_process:
                if logfile is not None:
                    with open(logfile, "a") as f:
                        f.write("s:%d val:%f\n" % (step, val_loss))

            # restart the clock
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        # bit confusing: we want to make sure to eval on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()

        train_loss = torch.zeros(1, device=device)

        torch.compiler.cudagraph_mark_step_begin()

        for micro_step in range(args.grad_accumulation_steps):

            # clear_router_zloss()
            # clear_load_balancing_loss()

            x, y = train_loader.next_batch()

            logits, moe_weight_list = model.pred(x)
            # logits = model.pred(x)

            lb_losses = [
                load_balancing_loss(w, num_experts=model.args.num_experts, top_k=model.args.k)
                for w in moe_weight_list
                    ]
            lb_loss = torch.stack(lb_losses).mean()

            loss = (F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1
            # ) + 0.1 * batched_router_zloss(model.smoes[0].args).mean()) / args.grad_accumulation_steps
            ) + 0.01 * lb_loss) / args.grad_accumulation_steps

            train_loss += loss.detach()# / args.grad_accumulation_steps

            loss.backward()

        muon_opt.step()
        muon_opt.zero_grad(set_to_none=True)

        adam_opt.step()
        adam_opt.zero_grad(set_to_none=True)

        # --------------- TRAINING SECTION END -----------------

        lr = get_lr(step)
        for optimizer in [adam_opt, muon_opt]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        # scheduler_muon.step(val_loss)
        # scheduler_adam.step(val_loss)
        
        # for param, optim in optim_dict.items():
        #     for param_group in optim.param_groups:
        #         param_group["lr"] = lr

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # time and print
        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        # the 0th iteration is often an outlier (much slower) => skip logging it
        # tokens_per_second = ddp_world_size * B * T / (t1-t0)
        # dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        train_loss = train_loss.item()
        print0(
            f"step:{step}/{args.num_iterations} | loss {train_loss:.6f} | train_time:{approx_training_time_ms/1000:.2f}s | step_avg:{approx_training_time_ms/(step+1):.2f}ms"
        )
        # log to logile
        if master_process and logfile is not None:
            with open(logfile, "a") as f:
                f.write("s:%d trn:%f\n" % (step, train_loss))
        del train_loss

        if master_process and (step + 1) % args.save_every == 0:
            log = dict(model=model.state_dict(), code=code, args=args.__dict__)
            os.makedirs("logs/%s" % run_id, exist_ok=True)
            torch.save(log, "logs/%s/model_step%06d.pt" % (run_id, step))

    print0(
        f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB"
    )

    # -------------------------------------------------------------------------

    if master_process:
        log = dict(model=model.state_dict(), code=code, args=args.__dict__)
        os.makedirs("logs/%s" % run_id, exist_ok=True)
        torch.save(log, "logs/%s/final.pt" % run_id)

    # -------------------------------------------------------------------------
    # clean up nice
    # destroy_process_group()
    

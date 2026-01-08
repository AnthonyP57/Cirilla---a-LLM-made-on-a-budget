import torch
import torch.nn as nn

class GRPO(nn.Module):
    def __init__(self, epsilon, beta, eps=1e-8):
        super().__init__()

        self.epsilon = epsilon
        self.beta = beta
        self.eps

    def forward(
            self,
            model_log_probs:torch.Tensor, # (B,G,S)
            old_model_log_probs:torch.Tensor, # (B,G,S)
            reference_model_log_probs:torch.Tensor, # (B,G,S)
            rewards:torch.Tensor, # (B,G)
            mask:torch.Tensor # (B,G,S)
                ):
        
        A_matrix = (rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + self.eps) #(B,G)
        A_matrix = A_matrix.unsqueeze(-1) # (B,G,1)

        pi_div = (model_log_probs - old_model_log_probs).exp() # (B,G,S)

        clipped = torch.clip(pi_div, 1 - self.epsilon, 1 + self.epsilon) # (B,G,S)

        preference_loss = torch.min(pi_div * A_matrix, clipped * A_matrix) # (B,G,S)

        kl_loss = (reference_model_log_probs - model_log_probs).exp() - \
            (reference_model_log_probs - model_log_probs) - 1 # (B,G,S)
        
        loss = - (((preference_loss - self.beta * kl_loss) * mask).sum(dim=2) / mask.sum(dim=2)).mean() # scalar
        return loss

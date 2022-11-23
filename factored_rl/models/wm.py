from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Linear

from factored_rl import configs
from factored_rl.models.nnutils import Module, one_hot, attention
from factored_rl.models import MLP, losses
from factored_rl.models.ae import PairedAutoencoderModel

class WorldModel(PairedAutoencoderModel):
    def __init__(self, input_shape: Tuple, n_actions: int, cfg: configs.Config):
        super().__init__(input_shape, n_actions, cfg)
        self.arch = cfg.model.arch.predictor
        if self.arch == 'mlp':
            self.predictor = MLP.from_cfg(
                n_inputs=self.n_latent_dims + self.n_actions,
                n_outputs=self.n_latent_dims,
                cfg=cfg.model.wm.mlp,
            )
        elif self.arch == 'attn':
            self.predictor = AttnPredictor(self.n_latent_dims, self.n_actions, cfg.model.wm.attn)

    def predict(self, features: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        if self.arch == 'mlp':
            context = torch.concat((features, actions))
            effects, attn_weights = self.predictor(context), None
        elif self.arch == 'attn':
            effects, attn_weights = self.predictor(features, actions)
        return effects, attn_weights

    def parents_loss(self, attn_weights: Tensor):
        if self.cfg.loss.parents == 0:
            return 0.0
        if attn_weights is None:
            raise RuntimeError(
                f'Cannot compute parents_loss because predictor does not produce attn_weights.\n'
                f'  predictor = {self.arch}; cfg.loss.parents = {self.cfg.loss.parents}')
        parents_loss = losses.compute_sparsity(attn_weights, self.cfg.loss.sparsity)
        return parents_loss

    def training_step(self, batch, batch_idx):
        obs = batch['ob']
        actions = one_hot(batch['action'], self.n_actions)
        next_obs = batch['next_ob']
        z = self.encoder(obs)
        effects, attn_weights = self.predict(z, actions)
        next_z_hat = z + effects
        final_activation = torch.sigmoid if self.cfg.model.arch.decoder == 'cnn' else lambda x: x
        obs_hat = final_activation(self.decoder(z))
        next_obs_hat = final_activation(self.decoder(next_z_hat))
        losses = {
            'actions': self.action_semantics_loss(actions, effects),
            'effects': self.effects_loss(effects),
            'parents': self.parents_loss(attn_weights),
            'reconst': self.reconstruction_loss(obs, next_obs, obs_hat, next_obs_hat),
        }
        loss = sum([losses[key] * self.cfg.loss[key] for key in losses.keys()])
        losses = {('loss/' + key): value for key, value in losses.items()}
        losses['loss/train_loss'] = loss
        self.log_dict(losses)
        if batch_idx % self.cfg.trainer.log_every_n_steps == 0:
            # stack images along H dimension
            obs_stack = torch.cat((obs, obs_hat, (obs - obs_hat)), dim=2)
            next_obs_stack = torch.cat((obs, obs_hat, (obs - obs_hat)), dim=2)
            tensorboard = self.logger.experiment
            tensorboard.add_images('img/obs_reconst_diff', obs_stack, self.global_step)
            tensorboard.add_images('img/next_obs_reconst_diff', next_obs_stack, self.global_step)
        return loss

class AttnPredictor(Module):
    def __init__(self, n_latent_dims: int, n_actions: int, cfg: configs.AttnConfig):
        super().__init__()
        self.n_latent_dims = n_latent_dims # d
        self.n_actions = n_actions # A
        self.n_heads = n_latent_dims # h
        self.dropout_p = cfg.dropout

        vdim = (cfg.factor_embed_dim + cfg.action_embed_dim) #Ev
        kdim = cfg.key_embed_dim if cfg.key_embed_dim is not None else vdim #Ek
        self.action_query_projection = Linear(n_actions, kdim) #Ek
        self.factor_key_projection = Linear(n_latent_dims, kdim) #Ek
        self.action_val_projection = Linear(n_actions, cfg.action_embed_dim) #Ea
        self.factor_val_projection = Linear(n_latent_dims, cfg.factor_embed_dim) #Ef
        self.output_projection = Linear(vdim, 1) #Ev

    def forward(self, features: torch.Tensor, actions: torch.Tensor):
        z = features # (N,d) or (d,)
        a = actions # (N,A) or (A,)

        is_batched = (z.dim() == 2)
        if (a.dim() == 2) != is_batched:
            raise ValueError(f'Model cannot simultaneously process batch and non-batch inputs:\n'
                             f'  features.dim() = {z.dim()}; actions.dim() = {a.dim()}'
                             f'Check that actions are using a one-hot encoding?')
        if not is_batched:
            z = z.unsqueeze(0)
            a = a.unsqueeze(0)
        if a.shape[0] != z.shape[0]:
            raise ValueError(f"features and actions must have the same batch size")

        # repeat input for each head
        z = z.unsqueeze(-2).expand(-1, self.n_heads, -1) # (N,h,d)
        a = a.unsqueeze(-2).expand(-1, self.n_heads, -1) # (N,h,A)

        # split factors into one-hot tokens
        masked_factors = torch.diag_embed(z) # (N,h,d,d)

        # embed factors
        factor_embeds = self.factor_val_projection(masked_factors) # (N,h,d,Ef)
        keys = self.factor_key_projection(masked_factors) # (N,h,d,Ek)

        # embed action
        queries = self.action_query_projection(a).unsqueeze(-2) # (N,h,1,Ek)
        action_embeds = self.action_val_projection(a) # (N,h,Ea)

        # match shapes
        action_vals = action_embeds.unsqueeze(-2) # (N,h,1,Ea)
        action_vals = action_vals.expand(-1, -1, self.n_latent_dims, -1) # (N,h,d,Ea)
        factor_vals = factor_embeds.expand(-1, -1, self.n_latent_dims, -1) # (N,h,d,Ef)

        # concat values s.t. every input has both action and factor info
        values = torch.concat((factor_vals, action_vals), dim=-1) # (N,h,d,Ev)

        dropout_p = self.dropout_p if self.training else 0.0
        effect_embed, attn_weights = attention(queries, keys, values, dropout_p=dropout_p)
        # (N,h,1,Ev)   (N,h,1,d)
        #    ^            ^   ^
        #  head         head  token in sequence

        effect_embed = effect_embed.squeeze(-2) # (N,h,Ev)
        attn_weights = attn_weights.squeeze(-2) # (N,h,d)

        effect = self.output_projection(effect_embed) # (N,h,1)
        effect = effect.squeeze(-1) # (N,h)

        if not is_batched:
            effect = effect.squeeze(0) # (h,)
            attn_weights = attn_weights.squeeze(0) # (h,d)

        return effect, attn_weights

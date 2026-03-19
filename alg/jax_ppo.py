from dataclasses import dataclass
import functools
from typing import Any,Dict, Sequence, Callable

import distrax
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.nnx as nnx
from flax.nnx.nn.initializers import constant, orthogonal
from tqdm import tqdm




class ScannedRNN(nnx.Module):
    def __init__(self, input_features, hidden_size, rngs: nnx.Rngs):
        self.hidden_size = hidden_size
        self.cell = nnx.OptimizedLSTMCell(
            in_features=input_features,
            hidden_features=hidden_size,
            rngs=rngs
        )

    def __call__(self, carry,x):
 
        def step_fn(carry, x_step):
            new_carry, y = self.cell(carry,x_step)
            return new_carry, y

        scan = nnx.scan(
            step_fn,
            in_axes=(nnx.Carry,0),
            out_axes=(nnx.Carry,0)
        )

        final_carry, out = scan(carry, x)
        return final_carry, out
    
    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return jnp.zeros((batch_size, hidden_size)), jnp.zeros((batch_size, hidden_size))

class CNN(nnx.Module):

    def __init__(self,conv_layers: list, latent_dim: int, lstm_dim: int,rngs:nnx.Rngs, activation: Callable = nn.tanh, **kwargs):
        self.latent_dim = latent_dim
        self.lstm_dim = lstm_dim
        self.activation = activation
        self.depth = len(conv_layers)
        self.convs = nnx.List([])
        for i, (in_features,out_features, kernel_size) in enumerate(conv_layers):
            self.convs.append(nnx.Conv(
                in_features=in_features,
                out_features=out_features,
                kernel_size=(kernel_size, kernel_size),
                padding='SAME',
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
                rngs=rngs,
            ))
        self.dense_out = nnx.Linear(latent_dim,lstm_dim,rngs = rngs)

    def __call__(self, x):

        for i in range(self.depth):
            x = self.convs[i](x)
            x = self.activation(x)
        
        x = x.reshape((x.shape[0], -1))

        # Dense
        x = self.dense_out(x)
        x = self.activation(x)
        return x

class ActorCriticRNN(nnx.Module):
    def __init__(self,conv_layers: list, action_dim: int, latent_dim: int, lstm_dim: int,rngs:nnx.Rngs):
        self.cnn = CNN(
            conv_layers=conv_layers,
            latent_dim=latent_dim,
            lstm_dim=lstm_dim,
            rngs=rngs,
            activation=nn.tanh,
        )
        self.rnn = ScannedRNN(
            input_features=lstm_dim,
            hidden_size=lstm_dim,
            rngs=rngs,
        )
        self.actor = nnx.Linear(
            in_features=lstm_dim,
            out_features=action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.critic = nnx.Linear(
            in_features=lstm_dim,
            out_features=1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.action_dim = action_dim
        self.layer_norm = nnx.LayerNorm(lstm_dim,rngs=rngs)
        self.rngs = rngs

    def __call__(self, hidden, x):
        embedding = nnx.vmap(self.cnn)(x)

        embedding = self.layer_norm(embedding)

        rnn_in = (embedding)
        hidden, embedding = self.rnn(hidden, rnn_in)
        embedding = nnx.tanh(embedding)

        actor_mean = self.actor(embedding)

        pi = distrax.Categorical(logits=actor_mean)

        critic = self.critic(embedding)

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class ActorCriticRNN_combine(nnx.Module):
    def __init__(self,conv_layers: list,conv_layers2:list, action_dim: int, latent_dim: int,latent_dim2:int, lstm_dim: int,rngs:nnx.Rngs):
        self.cnn = CNN(
            conv_layers=conv_layers,
            latent_dim=latent_dim,
            lstm_dim=lstm_dim,
            rngs=rngs,
            activation=nn.tanh,
        )
        self.cnn2 = CNN(
            conv_layers=conv_layers2,
            latent_dim=latent_dim2,
            lstm_dim=lstm_dim,
            rngs=rngs,
            activation=nn.tanh,
        )
        self.rnn = ScannedRNN(
            input_features=lstm_dim*2,
            hidden_size=lstm_dim*2,
            rngs=rngs,
        )
        self.actor = nnx.Linear(
            in_features=lstm_dim*2,
            out_features=action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.critic = nnx.Linear(
            in_features=lstm_dim*2,
            out_features=1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.action_dim = action_dim
        self.layer_norm = nnx.LayerNorm(lstm_dim,rngs=rngs)
        self.rngs = rngs

    def __call__(self, hidden, x1,x2):
        embedding = nnx.vmap(self.cnn)(x1)

        embedding = self.layer_norm(embedding)

        embedding2 = nnx.vmap(self.cnn2)(x2)
        embedding2 = self.layer_norm(embedding2)

        rnn_in = jnp.concatenate([embedding, embedding2], axis=-1)
        hidden, embedding = self.rnn(hidden, rnn_in)
        embedding = nnx.tanh(embedding)

        actor_mean = self.actor(embedding)

        pi = distrax.Categorical(logits=actor_mean)

        critic = self.critic(embedding)

        return hidden, pi, jnp.squeeze(critic, axis=-1)



class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        embedding = x
        activation = nn.tanh

        embed_model = CNN(
            config=self.config,
            activation=activation,
        )
        embedding = jax.vmap(embed_model)(embedding)

        embedding = nn.LayerNorm()(embedding)

        rnn_in = (embedding)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        embedding = activation(embedding)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(embedding)

        pi = distrax.Categorical(logits=actor_mean)


        return hidden, pi

class CriticRNN(nn.Module): 
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        embedding = x
        activation = nn.tanh

        embed_model = CNN(
            config=self.config,
            activation=activation,
        )
        embedding = jax.vmap(embed_model)(embedding)

        embedding = nn.LayerNorm()(embedding)

        rnn_in = (embedding)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        embedding = activation(embedding)

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            embedding
        )

        return hidden, jnp.squeeze(critic, axis=-1)
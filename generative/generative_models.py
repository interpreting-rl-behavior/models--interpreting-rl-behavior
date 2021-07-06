import torch
import torch.nn as nn
from . import layers as lyr
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn import init
import torch.optim as optim
from torch.nn import Parameter as P
from .action_cond_lstm import ActionCondLSTMLayer, LSTMCell, ActionCondLSTMCell, ActionCondLayerNormLSTMCell, LSTMState

class LayeredConvNet(nn.Module):
    """Template class

    Note:
        Some notes

    Args:
        param1 (str): Description of `param1`.
        param2 (:obj:`int`, optional): Description of `param2`. Multiple
            lines are supported.
        param3 (:obj:`list` of :obj:`str`): Description of `param3`.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    def __init__(self, stride=2,
                 channels=[3, 64, 32, 32],
                 kernel_sizes=[6, 6, 6],
                 padding_hs=[1, 1, 1],
                 padding_ws=[1, 1, 1],
                 input_hw=64,
                 deconv=False,
                 layer_norm=False):
        super(LayeredConvNet, self).__init__()

        # Check all inps have right len
        if not len(channels)-1 == len(kernel_sizes) or \
           not len(kernel_sizes) == len(padding_hs) or \
           not len(padding_hs) == len(padding_ws):
            raise ValueError("One of your conv lists has the wrong length.")

        self.input_hw = input_hw
        layer_in_hw = self.input_hw
        self.nets = nn.ModuleList([])
        self.deconv = deconv
        ch_in = channels[0]
        self.lns = [] # for debugging only
        for i, (ch_out, k, p_h, p_w) in enumerate(zip(channels[1:],
                                                 kernel_sizes,
                                                 padding_hs,
                                                 padding_ws)):
            if deconv:
                net = nn.ConvTranspose2d(ch_in, ch_out,
                                         kernel_size=(k,k),
                                         stride=(stride,stride),
                                         padding=(p_h,p_w))
            else:
                net = nn.Conv2d(ch_in, ch_out,
                                kernel_size=(k,k),
                                stride=(stride,stride),
                                padding=(p_h,p_w))
            self.nets.append(net)
            if i < len(kernel_sizes) - 1: # Doesn't add actv or LN on last layer
                if layer_norm:
                    layer_in_hw = conv_output_size(layer_in_hw,
                                                   stride=stride,
                                                   padding=p_h,
                                                   kernel_size=k,
                                                   transposed=deconv)
                    ln = nn.LayerNorm([ch_out, layer_in_hw, layer_in_hw])
                    self.lns.append(ln)
                    self.nets.append(ln)
                # self.nets.append(nn.RReLU())
                self.nets.append(nn.LeakyReLU(negative_slope=((1/3)-0.125)))
            ch_in = ch_out

    def forward(self, x):
        # TODO convert outs to a dict and name each of the saved outputs
        #  according to the layer
        outs = []
        out = x
        for l in self.nets:
            out = l(out)
            outs.append(out)
        return out, outs

class NLayerPerceptron(nn.Module):
    """An N layer perceptron with a linear output.

    Note:
        Some notes

    Args:
        param1 (str): Description of `param1`.
        param2 (:obj:`int`, optional): Description of `param2`. Multiple
            lines are supported.
        param3 (:obj:`list` of :obj:`str`): Description of `param3`.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    def __init__(self, sizes=[256, 256], layer_norm=False):
        super(NLayerPerceptron, self).__init__()
        self.nets = nn.ModuleList([])
        for i in range(len(sizes)-1):
            net = nn.Linear(in_features=sizes[i],
                            out_features=sizes[i+1])
            self.nets.append(net)

            if i < len(sizes)-1:  # Doesn't add activation (or LN) to final layer
                if layer_norm:
                    self.nets.append(nn.LayerNorm(sizes[i+1]))
                #self.nets.append(nn.RReLU())
                self.nets.append(nn.LeakyReLU(negative_slope=((1/3)-0.125)))

    def forward(self, x):
        # TODO convert outs to a dict and name each of the saved outputs
        #  according to the layer
        outs = []
        out = x
        for l in self.nets:
            out = l(out)
            outs.append(out)
        return out, outs


class VAE(nn.Module):
    """The Variational Autoencoder that generates agent-environment sequences.

    Description

    Note:
        Some notes

    Args:
        agent (:obj:`PPO`): The agent that will be used in the simulated
            agent-environment system in the decoder.
        param3 (:obj:`list` of :obj:`str`): Description of `param3`.
        num_recon_obs (int): The number of observations that are input to the
            VAE encoder. In the decoder, these are reconstructed.
        num_pred_steps (`int`): The number of steps into the future that the
            decoder is tasked to predict, not just reconstruct.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """

    def __init__(self, agent, device, num_initialization_obs, num_obs_full):
        super(VAE, self).__init__()

        # Set settings
        hyperparams = Namespace(
            # Encoder
            num_initialization_obs=num_initialization_obs,
            num_obs_full=num_obs_full,
            num_sim_steps=num_obs_full-num_initialization_obs+1,  # Plus one is because the last obs of the init seq is the first obs of the simulated seq
            initializer_sample_dim=64,
            initializer_rnn_hidden_size=512,
            global_context_sample_dim=64,
            global_context_encoder_rnn_hidden_size=512,
            #Decoder
            agent_hidden_size=64,
            action_space_dim=agent.env.action_space.n,
            env_stepper_rnn_hidden_size=1024,
            layer_norm=True
        )
        self.device = device

        # Create networks
        self.encoder = Encoder(hyperparams)
        self.decoder = Decoder(hyperparams, agent)


    def forward(self, obs, agent_h0, actions, use_true_h0=False, use_true_actions=False):

        # Feed inputs into encoder and return the mean
        # and log(variance)

        mu_c, logvar_c, mu_g, logvar_g = self.encoder(obs, agent_h0)

        sigma_c = torch.exp(0.5 * logvar_c)  # log(var) -> standard deviation
        sigma_g = torch.exp(0.5 * logvar_g)

        # Reparametrisation trick
        sample_c = (torch.randn(sigma_c.size(), device=self.device) * sigma_c) + mu_c
        sample_g = (torch.randn(sigma_g.size(),
                                device=self.device) * sigma_g) + mu_g

        # Decode
        if use_true_h0:
            true_h0 = agent_h0
        else:
            true_h0 = None
            # therefore decoder doesn't use true agent h0 and uses the one
            # produced by the h0 decoder instead
        if use_true_actions:
            true_acts = actions
        else:
            true_acts = None
            # therefore decoder doesn't use true actions and uses the ones
            # produced by the simulated agent instead.

        # Feed sample(s) to decoder
        pred_obs, pred_rews, pred_dones, pred_agent_hs, pred_agent_logprobs, \
        pred_agent_values, pred_env_hs = self.decoder(z_c=sample_c,
                                         z_g=sample_g,
                                         true_h0=true_h0,
                                         true_actions=true_acts)


        # Collect outputs
        preds = {'obs': pred_obs,
                 'reward': pred_rews,
                 'done': pred_dones,
                 'hx': pred_agent_hs,
                 'act_log_probs': pred_agent_logprobs,
                 'values': pred_agent_values,
                 'env_hx': pred_env_hs,
                 'latent_vecs_c_and_g': (sample_c, sample_g)
                 }

        return mu_c, logvar_c, mu_g, logvar_g, preds


class Encoder(nn.Module):
    """Encoder network

    Consists of a GlobalContextEncoder and and InitializerEncoder

    Note:
        Some notes

    Args:
        param1 (str): Description of `param1`.
        param2 (:obj:`int`, optional): Description of `param2`. Multiple
            lines are supported.
        param3 (:obj:`list` of :obj:`str`): Description of `param3`.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    def __init__(self, hyperparams):
        super(Encoder, self).__init__()

        self.initializer_encoder = \
            InitializerEncoder(rnn_hidden_size=hyperparams.initializer_rnn_hidden_size,
                               agent_hidden_size=hyperparams.agent_hidden_size,
                               sample_dim=hyperparams.initializer_sample_dim,
                               stride=2,
                               channels=[3,64,32,32],
                               kernel_sizes=[6,4,3],
                               padding_hs=[1,1,1],
                               padding_ws=[1,1,1],
                               layer_norm=hyperparams.layer_norm)

        self.global_context_encoder = \
            GlobalContextEncoder(rnn_hidden_size=hyperparams.global_context_encoder_rnn_hidden_size,
                                 sample_dim=hyperparams.global_context_sample_dim,
                                 stride=2,
                                 channels=[3, 32, 16, 16],
                                 kernel_sizes=[6, 4, 3],
                                 padding_hs=[1, 1, 1],
                                 padding_ws=[1, 1, 1],
                                 layer_norm=hyperparams.layer_norm)

        self.init_seq_len = hyperparams.num_initialization_obs
        self.global_context_seq_len = hyperparams.num_sim_steps

    def forward(self, full_obs, agent_h0):
        #TODO be sure that you reset the hidden state of your env rnns
        # either here or elsewhere

        # Split frames into initialization frames and global context frames
        init_obs = full_obs[:, 0:self.init_seq_len]
        #init_obs = torch.flip(init_obs, dims=[1]) # reverse time
        glob_ctx_obs = full_obs[:, -self.global_context_seq_len:]

        # Run InitializerEncoder
        mu_c, logvar_c = self.initializer_encoder(init_obs, agent_h0)

        # Run GlobalContextEncoder
        mu_g, logvar_g = self.global_context_encoder(glob_ctx_obs)

        return mu_c, logvar_c, mu_g, logvar_g


class InitializerEncoder(nn.Module):
    """Template class

    Recurrent network that takes a seq of frames from t=-k to t=0 as input.
    The final output gets passed along with the final agent hidden state into
    an FC network. Outputs params of the distribution from which we sample z_c,
    which will get decoded into the the agent's initial hidden state (at t=0)
    and, when combined with z_g, the initial environment state.

    Note:
        Some notes

    Args:
        param1 (str): Description of `param1`.
        param2 (:obj:`int`, optional): Description of `param2`. Multiple
            lines are supported.
        param3 (:obj:`list` of :obj:`str`): Description of `param3`.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    def __init__(self, rnn_hidden_size, agent_hidden_size,
                 sample_dim, stride, channels,
                 kernel_sizes, padding_hs, padding_ws, layer_norm):
        super(InitializerEncoder, self).__init__()
        self.conv_input = LayeredConvNet(stride=stride,
                                         channels=channels,
                                         kernel_sizes=kernel_sizes,
                                         padding_hs=padding_hs,
                                         padding_ws=padding_ws,
                                         layer_norm=layer_norm)
        # conv will output tensor of shape 32 * 8 * 8

        self.rnn = nn.LSTM(input_size=32 * 8 * 8,
                           hidden_size=rnn_hidden_size,
                           num_layers=1,
                           batch_first=True)

        self.converter_base = nn.Sequential(
                                nn.Linear(rnn_hidden_size + agent_hidden_size,
                                          rnn_hidden_size),
                                nn.ReLU()
                                            )
        self.converter_split_mu = nn.Linear(rnn_hidden_size,
                                            sample_dim)
        self.converter_split_lv = nn.Linear(rnn_hidden_size,
                                            sample_dim)  # log var

    def forward(self, inp_obs_seq, agent_h0):
        """Takes the sequence of inputs to encode context to
        initialise the env stepper and agent initial hidden state"""
        # Flatten inp seqs along time dimension to pass all to conv nets
        # along batch dim
        x = inp_obs_seq
        batches = x.shape[0]
        ts = x.shape[1]
        h = x.shape[2]
        w = x.shape[3]
        ch = x.shape[4]

        x = x.reshape(batches*ts, h, w, ch)
        x, _ = self.conv_input(x)

        # Unflatten conv outputs again to reconstruct time dim

        x = x.view(batches, ts, x.shape[1], x.shape[2], x.shape[3])

        # Flatten conv outputs to size HxWxCH to get rnn input vecs
        x = x.view(batches, ts, -1)

        # Pass seq of vecs to initializer RNN
        x, _ = self.rnn(x)

        # Concat RNN output to agent h0 and then pass to Converter nets
        # to get mu_g and sigma_g
        x = x[:, -1]  # get last ts
        x = torch.cat([x, agent_h0], dim=1)

        x = self.converter_base(x)
        mu = self.converter_split_mu(x)
        logvar = self.converter_split_lv(x)

        return mu, logvar


class GlobalContextEncoder(nn.Module):
    """Encodes global latent variable by observing whole sequence

    Note:
        Some notes

    Args:
        param1 (str): Description of `param1`.
        param2 (:obj:`int`, optional): Description of `param2`. Multiple
            lines are supported.
        param3 (:obj:`list` of :obj:`str`): Description of `param3`.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    def __init__(self, rnn_hidden_size, sample_dim, stride, channels,
                 kernel_sizes, padding_hs, padding_ws, layer_norm):
        super(GlobalContextEncoder, self).__init__()
        self.conv_input = LayeredConvNet(stride=stride,
                                         channels=channels,
                                         kernel_sizes=kernel_sizes,
                                         padding_hs=padding_hs,
                                         padding_ws=padding_ws,
                                         layer_norm=layer_norm)
        # conv will output tensor of shape 16 * 8 * 8

        self.rnn = nn.LSTM(input_size=16 * 8 * 8, # TODO make output size an attribute of LayeredConvNet so that you don't have to hard code this.
                           hidden_size=rnn_hidden_size,
                           num_layers=1,
                           batch_first=True)

        self.converter_base = nn.Sequential(nn.Linear(rnn_hidden_size,
                                                      rnn_hidden_size),
                                            nn.ReLU())
        self.converter_split_mu = nn.Linear(rnn_hidden_size,
                                            sample_dim)
        self.converter_split_lv = nn.Linear(rnn_hidden_size,
                                            sample_dim) # log var

    def forward(self, x):

        # Flatten inp seqs along time dimension to pass all to conv nets
        # along batch dim
        batches = x.shape[0]
        ts = x.shape[1]
        h = x.shape[2]
        w = x.shape[3]
        ch = x.shape[4]

        # Get only the even-numbered timesteps (because it's wasteful that a
        # global context encoder should see every single frame)
        chosen_ts = [t for t in list(range(0, ts)) if t % 2 == 0] # gets even ts
        num_chosen_ts = len(chosen_ts)
        x = x[:,chosen_ts]


        x = x.reshape([batches*num_chosen_ts, h, w, ch])
        x, _ = self.conv_input(x)

        # Unflatten conv outputs again to reconstruct time dim

        x = x.reshape([batches, num_chosen_ts, x.shape[1], x.shape[2], x.shape[3]])

        # Flatten conv outputs to size HxWxCH to get rnn input vecs
        x = x.reshape([batches, num_chosen_ts, -1])

        # Pass seq of vecs to global context RNN
        x, _ = self.rnn(x)

        # Pass RNN output to Global Context Converter to get mu_g and sigma_g
        x = x[:, -1]  # get last ts

        x = self.converter_base(x)
        mu = self.converter_split_mu(x)
        logvar = self.converter_split_lv(x)

        return mu, logvar


class Decoder(nn.Module):
    """Template class

    Note:
        Some notes

    Args:
        param1 (str): Description of `param1`.
        param2 (:obj:`int`, optional): Description of `param2`. Multiple
            lines are supported.
        param3 (:obj:`list` of :obj:`str`): Description of `param3`.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    def __init__(self, hyperparams, agent):
        super(Decoder, self).__init__()

        z_c_size = hyperparams.initializer_sample_dim
        z_g_size = hyperparams.global_context_sample_dim
        self.action_space_dim = hyperparams.action_space_dim
        agent_h0_size = hyperparams.agent_hidden_size
        env_h_size = hyperparams.env_stepper_rnn_hidden_size
        layer_norm = hyperparams.layer_norm
        self.num_sim_steps = hyperparams.num_sim_steps

        env_conv_top_shape = [128, 8, 8]

        # Make agent initializer network
        self.agent_initializer = NLayerPerceptron([z_c_size,
                                                   256,
                                                   agent_h0_size],
                                                  layer_norm=layer_norm)

        # Make EnvStepperInitializerNetwork
        self.env_stepper_initializer = \
            EnvStepperStateInitializer(z_c_size, z_g_size, env_h_size, layer_norm=layer_norm)

        # Make EnvStepper
        self.env_stepper = EnvStepper(env_hidden_size=env_h_size,
                                      env_conv_top_shape=env_conv_top_shape,
                                      z_g_size=z_g_size,
                                      stride_out=2,  #2,
                                      channels_out=[128, 128, 256, 3],  #[64, 64, 256, 3],
                                      kernel_sizes_out=[3, 5, 6],
                                      padding_hs_out=[1, 1, 1],
                                      padding_ws_out=[1, 1, 1],
                                      layer_norm=layer_norm)

        # Make agent into an attribute of the decoder class
        self.agent = agent

    def forward(self, z_c, z_g, true_h0=None, true_actions=None):

        pred_obs = []
        pred_rews = []
        pred_dones = []
        pred_env_hs = []
        pred_agent_hs = []
        pred_acts = []
        pred_agent_logprobs = []
        pred_agent_values = []

        # Initialize env@t=0 and agent_h@t=0
        env_rnn_state = self.env_stepper_initializer(z_c, z_g)
        env_h, env_cell_state = env_rnn_state

        pred_agent_h0, _ = self.agent_initializer(z_c)
        pred_agent_hs.append(pred_agent_h0)

        # if we want to feed the correct h0 (the 0th hx) instead of the
        # guessed h0 (for purposes of being able to train the
        # agent and the generative model on different agents), then
        # we need to swap in the true h0 here, but we'll still store
        # and return the guessed h0 in preds so that it can be
        # trained to approximate true_h0.
        agent_h = true_h0 if true_h0 is not None else pred_agent_h0

        for i in range(self.num_sim_steps):
            # The first part of the for-loop happens only within t
            pred_env_hs.append(env_rnn_state)  # env@t
            if i > 0:
                pred_agent_hs.append(agent_h)  # agent_h@t

            ## Decode env_h@t to get ob/rew/done@t
            ob, rew, done = self.env_stepper.decode_hx(env_h)
            pred_obs.append(ob)
            pred_rews.append(rew)
            pred_dones.append(done)

            # Code in the for loop following here spans t AND t+1
            ## Step forward the agent to get act@t and logprobs@t, but
            ## agent_h@t+1
            act, logits, value, agent_h = self.agent.predict_STE(ob, agent_h,
                                                                 done)
            pred_acts.append(act)
            pred_agent_logprobs.append(logits)
            pred_agent_values.append(value)

            # Step forward the env using action@t and env_rnn_state@t and
            # ob@t and return env_rnn_state@t+1
            if true_actions is not None:
                # get true actions for that step and overwrite predicted
                # actions for input to env
                act = true_actions[:, i]
                act = torch.nn.functional.one_hot(act.long(), num_classes=self.action_space_dim)
                act = act.float()
            env_rnn_state = \
                self.env_stepper.encode_and_step(act, env_rnn_state, z_g)
            env_h, env_cell_state = env_rnn_state # env@t+1

            ## Get ready for new step
            self.agent.train_prev_recurrent_states = None


        return pred_obs, pred_rews, pred_dones, pred_agent_hs, \
               pred_agent_logprobs, pred_agent_values, pred_env_hs


class EnvStepperStateInitializer(nn.Module):
    """Template class

    Note:
        Some notes

    Args:
        param1 (str): Description of `param1`.
        param2 (:obj:`int`, optional): Description of `param2`. Multiple
            lines are supported.
        param3 (:obj:`list` of :obj:`str`): Description of `param3`.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    def __init__(self, z_c_size, z_g_size, env_h_size, layer_norm):
        super(EnvStepperStateInitializer, self).__init__()
        self.base = NLayerPerceptron([z_c_size + z_g_size,
                                      env_h_size,
                                      env_h_size * 2], layer_norm=layer_norm)
        self.cell_net = nn.Sequential(nn.ReLU(),
            nn.Linear(env_h_size * 2, env_h_size),
                                      nn.Sigmoid())
        self.hx_net = nn.Sequential(nn.ReLU(),
            nn.Linear(env_h_size * 2, env_h_size),
                                    nn.Tanh())

    def forward(self, z_c, z_g):
        z = torch.cat([z_c, z_g], dim=1)
        x, _ = self.base(z)
        cell_state = self.cell_net(x)
        hx_state = self.hx_net(x)
        return (hx_state, cell_state)

class EnvStepper(nn.Module):
    """Template class

    Note:
        Some notes

    Args:
        param1 (str): Description of `param1`.
        param2 (:obj:`int`, optional): Description of `param2`. Multiple
            lines are supported.
        param3 (:obj:`list` of :obj:`str`): Description of `param3`.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    def __init__(self, env_hidden_size, env_conv_top_shape, z_g_size,
                 stride_out, channels_out, kernel_sizes_out, padding_hs_out,
                 padding_ws_out, layer_norm):
        super(EnvStepper, self).__init__()

        self.env_h_size = env_hidden_size
        self.ob_conv_top_shape = env_conv_top_shape
        self.shp = self.ob_conv_top_shape  # legibility
        self.ob_conv_top_size = np.prod(self.ob_conv_top_shape)
        self.z_g_size = z_g_size

        # Networks in output step
        self.ob_decoder_fc = NLayerPerceptron([self.env_h_size,
                                               self.ob_conv_top_size])
        self.ob_decoder_conv = LayeredConvNet(stride=stride_out,
                                              channels=channels_out,
                                              kernel_sizes=kernel_sizes_out,
                                              padding_hs=padding_hs_out,
                                              padding_ws=padding_ws_out,
                                              deconv=True,
                                              layer_norm=layer_norm,
                                              input_hw=self.ob_conv_top_shape[1])
        self.done_decoder = NLayerPerceptron([env_hidden_size, 64, 1],
                                             layer_norm=layer_norm)
        self.reward_decoder = NLayerPerceptron([env_hidden_size, 64, 1],
                                               layer_norm=layer_norm)

        # Networks in forward step
        if layer_norm:
            cell = ActionCondLayerNormLSTMCell
        else:
            cell = ActionCondLSTMCell
        self.env_rnn = ActionCondLSTMLayer(cell,
                                 self.z_g_size,  # input size + self.env_h_size + self.z_g_size,
                                 self.env_h_size,  # hx size
                                 self.env_h_size*2,  # fusion size
                                 15,  # action space dim
                                 )

    def encode_and_step(self, act, rnn_state, z_g):

        rnn_state = LSTMState(rnn_state[0],rnn_state[1])

        # Unsqueeze to create unitary time dimension
        act = torch.unsqueeze(act, dim=1)
        z_g = torch.unsqueeze(z_g, dim=1)

        # Step RNN (Time increments here)
        in_vec = z_g
        out, out_state = self.env_rnn(in_vec, act, rnn_state)

        return out_state

    def decode_hx(self, hx):

        # Convert hidden state to image prediction
        x, _ = self.ob_decoder_fc(hx)
        x = x.view(x.shape[0], self.shp[0], self.shp[1], self.shp[2])
        ob, _ = self.ob_decoder_conv(x)
        ob = torch.sigmoid(ob)

        # Convert hidden state to reward prediction
        rew, _ = self.reward_decoder(hx)

        # Conver hidden state to done prediction
        done, _ = self.done_decoder(hx)
        done = torch.sigmoid(done)

        return ob, rew, done


# Probably better just to have two 'forward' like functions. One for a full
# step, the other for a half step that just produces its outputs given a cell
# state. This very function can be used in the full step.


class GenericClass(nn.Module):
    """Template class

    Note:
        Some notes

    Args:
        param1 (str): Description of `param1`.
        param2 (:obj:`int`, optional): Description of `param2`. Multiple
            lines are supported.
        param3 (:obj:`list` of :obj:`str`): Description of `param3`.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """
    def __init__(self, insize=256, outsize=256):
        super(TwoLayerPerceptron, self).__init__()
        self.net = \
            nn.Sequential(nn.Linear(insize, insize),
                          nn.ReLU(),
                          nn.Linear(insize, outsize))

    def forward(self, x):
        return self.net(x)



class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def conv_output_size(input_hw, stride, padding, kernel_size, transposed=False):
    """assumes a square input"""
    if transposed:
        out_hw = stride * (input_hw - 1) + kernel_size - 2 * padding
    else:
        out_hw = ( (input_hw - kernel_size + ( 2 * padding)) / stride) + 1
    return int(out_hw)
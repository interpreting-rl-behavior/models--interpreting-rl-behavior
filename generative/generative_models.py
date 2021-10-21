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
import os
from .layers import LayeredConvNet, LayeredResBlockUp, LayeredResBlockDown, NLayerPerceptron



class AgentEnvironmentSimulator(nn.Module): # TODO make agent optional.
    """
    Takes the VAE latent vector as input and simulates an agent-environment
    rollout
    """
    def __init__(self, agent, device, num_initialization_obs, num_obs_full):
        super(AgentEnvironmentSimulator, self).__init__()

        hyperparams = Namespace(
            num_initialization_obs=num_initialization_obs,
            num_obs_full=num_obs_full,
            num_sim_steps=num_obs_full-num_initialization_obs+1,  # Plus one is because the last obs of the init seq is the first obs of the simulated seq
            env_num_categs=32,
            env_categ_distribs=32,
            env_h_stoch_size=32*32,
            agent_hidden_size=64,
            action_space_dim=agent.env.action_space.n,
            env_stepper_rnn_hidden_size=512,
            initializer_rnn_hidden_size=256,
            layer_norm=True
        )

        self.action_space_dim = hyperparams.action_space_dim
        self.init_seq_len = hyperparams.num_initialization_obs
        self.num_sim_steps = hyperparams.num_sim_steps

        # Make EnvStepper
        self.env_stepper = EnvStepper(hyperparams)
        self.env_stepper_initializer = EnvStepperInitializer(hyperparams)

        # Make agent into an attribute of the decoder class
        self.agent = agent

        # Get directions for directions swapping exps
        hx_analysis_dir = os.path.join('analysis', 'hx_analysis_precomp')
        directions_path = os.path.join(hx_analysis_dir, 'pcomponents_4000.npy')
        hx_std_path = os.path.join(hx_analysis_dir, 'hx_std_4000.npy')

        if os.path.exists(directions_path):
            device = agent.device
            self.hx_std = torch.tensor(np.load(hx_std_path)).to(device).requires_grad_()
            directions = torch.tensor(np.load(
                directions_path)).to(device).requires_grad_()
            directions = directions.transpose(0, 1)
            directions = directions * self.hx_std
            directions = [directions[:,i] for i in
                               range(directions.shape[1])]
            self.directions = [direction/torch.norm(direction) for direction
                               in directions]  # Normalise vecs

    def forward(self, full_obs, true_h0, true_actions, use_true_actions=True, imagine=False,
                retain_grads=True, swap_directions=None):
        """
        imagine: Whether or not to use the generated images as input to
        the env model or whether to use true images (true images will be used
        during training).

        """

        pred_obs = []
        pred_rews = []
        pred_dones = []
        env_h_determs = []
        env_h_stoch_logits = []
        pred_env_h_stoch_logits = []
        env_h_stoch_samples = []
        pred_agent_hs = []
        pred_acts = []
        pred_agent_logprobs = []
        pred_agent_values = []

        # Initialize env@t=0 and agent_h@t=0
        init_obs = full_obs[:, 0:self.init_seq_len]
        env_h_determ_init = self.env_stepper_initializer(init_obs)
        env_h_determ = env_h_determ_init  # env_h_d_0

        agent_h = true_h0

        if retain_grads:
            # pred_agent_h0.retain_grad()
            env_h_determ.retain_grad()

        for i in range(self.num_sim_steps):
            # The first part of the for-loop happens only within t

            pred_env_stoch_logit, _ = \
                self.env_stepper.transition_predictor(env_h_determ)  # w_hat@t

            if imagine and i > 0:
                env_stoch_logit = pred_env_stoch_logit
            else:
                true_ob = full_obs[:, self.init_seq_len - 1 + i]
                if true_ob.ndim != 4:
                    print('boop')
                true_env_stoch_logit = \
                    self.env_stepper.representation_model(true_ob, env_h_determ)  # w@t
                env_stoch_logit = true_env_stoch_logit

            sample = self.env_stepper.sample_w_STE(env_stoch_logit)
            sample = sample.view(sample.shape[0], -1)
            env_h_stoch = self.env_stepper.sample_converter(sample)  # z@t

            # Save env stuff
            env_h_determs.append(env_h_determ)
            env_h_stoch_logits.append(env_stoch_logit)
            pred_env_h_stoch_logits.append(pred_env_stoch_logit)
            env_h_stoch_samples.append(env_h_stoch)
            env_h_cat = torch.cat([env_h_determ, env_h_stoch], dim=1)

            # Use the environment state to decode.
            pred_ob, pred_rew, pred_done = self.env_stepper.decode(env_h_cat)
            pred_obs.append(pred_ob)
            pred_rews.append(pred_rew)
            pred_dones.append(pred_done)

            # Code in the for loop following here spans t AND t+1
            ## Step forward the agent to get act@t and value@t, but
            ## agent_h@t+1
            pred_agent_hs.append(agent_h)
            old_agent_h = agent_h.clone().detach()
            act, act_logits, value, agent_h = \
                self.agent.predict_STE(pred_ob, agent_h, pred_done,
                                       retain_grads=retain_grads) # TODO confirm dones/pred_dones isn't doing anything weird here via masking.
            if swap_directions is not None:
                agent_h = self.swap_directions(swap_directions,
                                               old_agent_h,
                                               agent_h)
            pred_acts.append(act)
            pred_agent_logprobs.append(act_logits)
            pred_agent_values.append(value)

            # Step forward the env using action@t and env_rnn_state@t and
            # ob@t and return env_rnn_state@t+1
            if use_true_actions:
                # If true, the decoder doesn't use true actions and uses the ones
                # produced by the simulated agent instead.
                act = true_actions[:, i]
                act = torch.nn.functional.one_hot(act.long(), num_classes=self.action_space_dim)
                act = act.float()
                act.requires_grad = True
            env_h_determ = \
                self.env_stepper.transition_step(act, env_h_determ, env_h_stoch) # env@t+1

            if env_h_determ.ndim == 3:
                env_h_determ = torch.squeeze(env_h_determ)


            if retain_grads:
                pred_ob.retain_grad()
                agent_h.retain_grad()
                act.retain_grad()
                act_logits.retain_grad()
                env_h_determ.retain_grad()
                env_h_stoch.retain_grad()

            ## Get ready for new step
            self.agent.train_prev_recurrent_states = None

        preds_dict = {
            'obs': pred_obs,
            'reward': pred_rews,
            'done': pred_dones,
            'env_h_determs': env_h_determs,
            'env_h_stoch_logits': env_h_stoch_logits,
            'pred_env_h_stoch_logits': pred_env_h_stoch_logits,
            'env_h_stoch_samples': env_h_stoch_samples,
            'hx': pred_agent_hs,
            'acts': pred_acts,
            'act_log_probs': pred_agent_logprobs,
            'agent_values': pred_agent_values
        }

        return preds_dict

    def swap_directions(self, swap_directions, old_agent_h, agent_h):
        delta = agent_h - old_agent_h
        original_agent_h = agent_h - delta # for gradients
        bs = agent_h.shape[0]

        delta_subtract = torch.zeros_like(delta)
        delta_add = torch.zeros_like(delta)

        # Define the directions
        from_dirs = swap_directions[0]
        to_dirs = swap_directions[1]

        # project the delta onto the directions that we're swapping out and
        # keep the projection amounts. Also save the directions we want to
        # remove from the delta (i.e. delta_subtract)
        from_dirs_projection_amounts = []
        for from_dir in from_dirs:
            direction = self.directions[from_dir]
            projection_amount = delta @ direction # inner product
            direction = torch.stack([direction] * bs)
            delta_subtract += (direction.transpose(0,1) * projection_amount
                               ).transpose(0,1)
            from_dirs_projection_amounts.append(projection_amount)

        # Make a vector to add to the delta that has the same size of projection
        # but in a different direction
        null_dir = torch.zeros_like(self.directions[0])
        for to_dir, projection_amount in zip(to_dirs,
                                             from_dirs_projection_amounts):
            if to_dir is not None:
                direction = self.directions[from_dir]
            else:
                direction = null_dir
            direction = torch.stack([direction] * bs)
            delta_add += (direction.transpose(0,1) * projection_amount
                               ).transpose(0,1)

        delta = delta - delta_subtract + delta_add
        return delta


class EnvStepperInitializer(nn.Module):
    """

    Recurrent network that takes a seq of frames from t=-k to t=0 as input.
    The final output gets passed along with the final agent hidden state into
    an FC network. It outputs the initial state of the environment simulator.

    """
    def __init__(self, hyperparams):
        super(EnvStepperInitializer, self).__init__()

        self.init_seq_len = hyperparams.num_initialization_obs
        self.rnn_hidden_size = hyperparams.initializer_rnn_hidden_size
        self.env_dim = hyperparams.env_stepper_rnn_hidden_size

        self.image_embedder = LayeredResBlockDown(input_hw=64,
                                                  input_ch=3,
                                                  hidden_ch=64,
                                                  output_hw=8,
                                                  output_ch=32)

        self.rnn = nn.GRU(input_size=self.image_embedder.output_size,
                           hidden_size=self.rnn_hidden_size,
                           num_layers=1,
                           batch_first=True)

        self.mlp_out = nn.Sequential(
                                nn.Linear(self.rnn_hidden_size,
                                          self.rnn_hidden_size),
                                nn.ELU(),
                                nn.Linear(self.rnn_hidden_size,
                                          self.env_dim)
                                            )

    def forward(self, init_obs):
        """"""
        # Flatten inp seqs along time dimension to pass all to conv nets
        # along batch dim
        x = init_obs
        batches = x.shape[0]
        ts = x.shape[1]
        h = x.shape[2]
        w = x.shape[3]
        ch = x.shape[4]

        images = [x[:, i] for i in range(ts)]  # split along time dim
        embeddings = [self.image_embedder(im) for im in images]
        embeddings = [im for (im, _) in embeddings]
        x = torch.stack(embeddings, dim=1)  # stack along time dim

        # Flatten conv outputs to size (H*W*CH) to get rnn input vecs
        x = x.view(batches, ts, -1)

        # Pass seq of vecs to initializer RNN
        x, _ = self.rnn(x)

        # Concat RNN output to agent h0 and then pass to Converter nets
        # to get mu_g and sigma_g
        x = x[:, -1]  # get last ts
        init_env_state = self.mlp_out(x)
        return init_env_state

class EnvStepper(nn.Module):
    """

    """
    def __init__(self, hyperparams):
        super(EnvStepper, self).__init__()
        layer_norm = hyperparams.layer_norm
        self.env_h_determ_size = hyperparams.env_stepper_rnn_hidden_size
        self.env_h_stoch_size = hyperparams.env_h_stoch_size
        self.env_h_size = self.env_h_determ_size + self.env_h_stoch_size
        self.env_num_categs = hyperparams.env_num_categs
        self.env_categ_distribs = hyperparams.env_categ_distribs
        assert self.env_h_stoch_size == self.env_num_categs * self.env_categ_distribs

        action_dim = hyperparams.action_space_dim

        # Image Encoder
        self.ob_encoder_conv_top_shape = [32, 8, 8]
        self.ob_encoder_conv_top_size = np.prod(self.ob_encoder_conv_top_shape)
        self.ob_encoder_conv = LayeredResBlockDown(input_hw=64,
                                                 input_ch=3,
                                                 hidden_ch=32,
                                                 output_hw=self.ob_encoder_conv_top_shape[1],
                                                 output_ch=self.ob_encoder_conv_top_shape[0],)
        ob_encoder_fc_top_shape = self.env_h_stoch_size
        self.ob_encoder_fc = NLayerPerceptron([self.ob_encoder_conv_top_size,
                                               ob_encoder_fc_top_shape])
        self.sample_converter = nn.Linear(self.env_h_stoch_size,
                                          self.env_h_stoch_size)

        # Transition networks
        self.transition_predictor = NLayerPerceptron([self.env_h_determ_size,
                                                      self.env_h_stoch_size])
        self.env_rnn = nn.GRU(input_size=self.env_h_determ_size,
                              hidden_size=self.env_h_determ_size,
                              batch_first=True)
        self.input_feeder_net = nn.Linear(self.env_h_stoch_size+action_dim,
                                          self.env_h_determ_size)
        repr_model_inp_size = ob_encoder_fc_top_shape + self.env_h_determ_size
        self.representation_model_fc = NLayerPerceptron([repr_model_inp_size,
                                                     self.env_h_stoch_size])


        # Image Decoder
        self.ob_decoder_conv_top_shape = [32, 8, 8]
        self.ob_decoder_conv_top_size = np.prod(self.ob_decoder_conv_top_shape)
        self.ob_decoder_conv = LayeredResBlockUp(input_hw=self.ob_decoder_conv_top_shape[1],
                                                 input_ch=self.ob_decoder_conv_top_shape[0],
                                                 hidden_ch=32,
                                                 output_hw=64,
                                                 output_ch=3)
        self.ob_decoder_fc = NLayerPerceptron([self.env_h_size,
                                               self.ob_decoder_conv_top_size])

        # Other decoders
        self.done_decoder = NLayerPerceptron([self.env_h_size, 64, 1],
                                             layer_norm=layer_norm)
        self.reward_decoder = NLayerPerceptron([self.env_h_size, 64, 1],
                                               layer_norm=layer_norm)

    # def forward(self, env_h_determ, input_image, action, imagine=False):
    #     batch_size = input_image.shape[0]
    #
    #     predicted_logits = self.transition_predictor(env_h_determ)
    #     if imagine:
    #         sample = self.sample_w_STE(predicted_logits)
    #     else:
    #         repr_inp_vec = torch.cat([input_image, env_h_determ])
    #         true_logits = self.representation_model(repr_inp_vec)
    #         sample = self.sample_w_STE(true_logits)
    #
    #     env_h_stoch = sample.reshape(batch_size, self.env_h_stoch_size)
    #
    #     env_h_determ, _ = self.transition_step(action,
    #                                            env_h_determ,
    #                                            env_h_determ)
    #
    #     # decode
    #     env_h = torch.cat([env_h_determ, env_h_stoch])
    #     ob, rew, done = self.decode(env_h)
    #
    #     return

    def representation_model(self, image, env_h_determ):
        b = image.shape[0]
        embedding, _ = self.ob_encoder_conv(image)
        embedding = embedding.view(b, -1)
        embedding, _ = self.ob_encoder_fc(embedding)

        if env_h_determ.ndim == 3:
            env_h_determ = torch.squeeze(env_h_determ)

        if env_h_determ.ndim != embedding.ndim:
            print("Boop")
        repr_inp_vec = torch.cat([embedding, env_h_determ], dim=1)
        true_env_h_stoch_logits, _ = self.representation_model_fc(repr_inp_vec)
        return true_env_h_stoch_logits

    def transition_step(self, act, env_h_deterministic, env_h_stochastic):

        # Just concat these together and pass through an MLP
        in_vec = torch.cat([env_h_stochastic, act], dim=1)
        in_vec = self.input_feeder_net(in_vec)

        # Unsqueeze to create unitary time dimension
        in_vec = torch.unsqueeze(in_vec, dim=1)
        env_h_deterministic = torch.unsqueeze(env_h_deterministic, dim=0)
        # N.B dim 0 instead of 1. Not sure why the 'batch first' doesn't apply
        # also to the hidden state...

        # Step RNN (Time increments here)
        out, out_state = self.env_rnn(in_vec, env_h_deterministic)

        return out_state

    def decode(self, env_h):
        b = env_h.shape[0]
        ch = self.ob_decoder_conv_top_shape[0]
        h = self.ob_decoder_conv_top_shape[1]
        w = self.ob_decoder_conv_top_shape[2]

        # Convert hidden state to image prediction
        x, _ = self.ob_decoder_fc(env_h)
        x = x.view(x.shape[0], ch, h, w)
        x, _ = self.ob_decoder_conv(x)
        ob = torch.sigmoid(x)

        # Convert hidden state to reward prediction
        rew, _ = self.reward_decoder(env_h)

        # Conver hidden state to done prediction
        done, _ = self.done_decoder(env_h)
        done = torch.sigmoid(done)

        return ob, rew, done

    def sample_w_STE(self, logits_vec, modal=False):

        b = logits_vec.shape[0]
        device = logits_vec.device

        # Reshape logits to (batch, dimension, categories)
        logits = logits_vec.view(b, self.env_categ_distribs,
                                 self.env_num_categs)

        if modal:
            mode = logits.argmax(dim=2)
            sample = \
                torch.nn.functional.one_hot(
                    mode, num_classes=self.env_num_categs).to(device)
        else:
            # Sample using the logits
            sample = torch.distributions.OneHotCategorical(logits=logits).sample()

        # Straight-Through-Estimator step
        probs = torch.softmax(logits, dim=2)
        sample = sample + probs - probs.detach()

        return sample

# class GenericClass(nn.Module):
#     """Template class
#
#     Note:
#         Some notes
#
#     Args:
#         param1 (str): Description of `param1`.
#         param2 (:obj:`int`, optional): Description of `param2`. Multiple
#             lines are supported.
#         param3 (:obj:`list` of :obj:`str`): Description of `param3`.
#
#     Attributes:
#         attr1 (str): Description of `attr1`.
#         attr2 (:obj:`int`, optional): Description of `attr2`.
#
#     """
#     def __init__(self, insize=256, outsize=256):
#         super(TwoLayerPerceptron, self).__init__()
#         self.net = \
#             nn.Sequential(nn.Linear(insize, insize),
#                           nn.ReLU(),
#                           nn.Linear(insize, outsize))
#
#     def forward(self, x):
#         return self.net(x)

class Namespace:
    """
    Because they're nicer to work with than dictionaries
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)



### For deletion once RSSM is working:

# class GlobalContextEncoder(nn.Module):
#     """
#     Encodes global latent variable by observing whole sequence
#     """
#     def __init__(self, embedding_size, sample_dim):
#         super(GlobalContextEncoder, self).__init__()
#
#         self.image_conv_embedder = LayeredResBlockDown(input_hw=64,
#                                                   input_ch=3,
#                                                   hidden_ch=32,
#                                                   output_hw=8,
#                                                   output_ch=16)
#         self.image_fc_embedder = nn.Linear(in_features=self.image_conv_embedder.output_size,
#                                            out_features=embedding_size)
#
#         t_layer = torch.nn.TransformerEncoderLayer(d_model=embedding_size, nhead=1,
#                                              dim_feedforward=embedding_size,
#                                              dropout=0.0).to(torch.device('cuda:0'))
#         norm = nn.LayerNorm(normalized_shape=embedding_size, eps=1e-5)
#         self.seq_enc = torch.nn.TransformerEncoder(t_layer, num_layers=2, norm=norm).to(torch.device('cuda:0'))
#
#
#
#         self.converter_base = nn.Sequential(nn.Linear(embedding_size,
#                                                       embedding_size),
#                                             nn.ReLU())
#         self.converter_split_mu = nn.Linear(embedding_size,
#                                             sample_dim)
#         self.converter_split_lv = nn.Linear(embedding_size,
#                                             sample_dim) # log var
#
#     def forward(self, x):
#
#         # Flatten inp seqs along time dimension to pass all to conv nets
#         # along batch dim
#         batches = x.shape[0]
#         ts = x.shape[1]
#         h = x.shape[2]
#         w = x.shape[3]
#         ch = x.shape[4]
#
#         # Get only the timesteps at intervals (because it's wasteful that a
#         # global context encoder should see every single frame)
#         midpoint_intervals = np.arange(0, ts, step=4)
#         midpoint_intervals_rand = midpoint_intervals[1:-1]
#         random_interval_diffs = np.random.randint(-2, 2, len(midpoint_intervals_rand))
#         chosen_ts = [t+rand for t, rand in zip(midpoint_intervals_rand, random_interval_diffs)]
#         midpoint_intervals[1:-1] = chosen_ts
#         chosen_ts = midpoint_intervals
#         num_chosen_ts = len(chosen_ts)
#         x = x[:,chosen_ts]
#
#         # Embed images into vectors for the sequence encoder
#         embeddings = [self.image_conv_embedder(x[:,i]) for i in range(num_chosen_ts)]
#         embeddings = [x.reshape(batches, -1) for (x, _) in embeddings]
#         embeddings = [self.image_fc_embedder(x) for x in embeddings]
#         embeddings = torch.stack(embeddings, dim=1)  # Stack along time dim
#         embeddings = embeddings.permute([1,0,2])  # Swap batch and time dim for transformer.  -> [t, b, etc]
#
#         #Attn
#         x = self.seq_enc(embeddings)
#         x = x.permute([1,0,2]) # swap batch and t axis back again -> [b, t, etc]
#         x = torch.mean(x, dim=1)
#
#         # Convert transformer output into global vector sampling params
#         x = self.converter_base(x)
#         mu = self.converter_split_mu(x)
#         logvar = self.converter_split_lv(x)
#
#         return mu, logvar
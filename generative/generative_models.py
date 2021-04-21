import torch
import torch.nn as nn
from . import layers as lyr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P


"""Proposal for generative model:

    - An encoder network
        - EncoderInputNetwork
            - see diagram on tablet
            - image -> residual (64)
            - init hidden state added to the channel dim (64+128)
            - Resblock down all (image, init hidden, and resid output) into 1x1 conv (3+64+128 -> 128) (32x32x128)
            - layer norm
            - non-local layer (with residual connection)
            - layer norm
            - ResOnebyOne (dense from all(?) previous)
            - Resblockdown 
            - layer norm
        - EncoderRNN
            - convGRU (8x8x256)
            - layer norm after every step
        - EncoderEmbedder (takes final convGRU output only)
            - Resblock down (8x8x256) -> (4x4x256)
            - layer norm
            - split
                - fc to mu (should have around 256 neurons)
                - fc to sigma
    - A decoder network
        - Initializer networks
            - InitHiddenStateNetwork = TwoLayerPerceptron
                - fc
                - layer norm
                - fc
            - EnvStepper initializer
                - This takes a guess at initializing the stepper so 
                  it doesn't start with 0-tensors. Informative initialization 
                  like this should work better.  
                - outputs something the size of the 'EnvStepper hidden state shape' for env
                  unrolling         
            - UnrollerNet
                - AssimilatorResidualBlock (takes EnvStepper hidden state shape block and also noise vector and outputs EnvStepper hidden state shape block) 
                - layer norm
                - ResidualConv
                - layer norm
        - Side decoders (take a EnvStepper hidden state shape block and produce predictions for obs and rew
            -reward decoder
                - Residual shrink block
                - layer norm
                - FC -> 1
            - obs decoder
                - deconv growth residual block (but reduce channels)
                - layer norm
                - deconv growth residual block (but reduce channels)
                - layer norm
                - deconv growth residual block (but reduce channels)
                - layer norm
                - conv (but reduce channels) (3)
            
    -  AssimilatorResidualBlock
        - has-a:
            - AssimilatorBlock (1x1 conv to 2d conv)
            - residual connection between non vec inputs to AssimilatorBlock and its outputs
"""

"""
Classes we'll need:
- ResidualBlock for upsample and downsample and constant (from BigGAN)
- non-local layer (Attention)
- AssimilatorResidualBlock
- ResOnebyOne
- Encoder
    - EncoderInputNetwork
    - EncoderRNN
    - EncoderEmbedder
- Decoder
    - InitializerNets
        - IHANetwork
        - EnvStepperInitializer
    - EnvStepper
    - reward decoder
    - obs decoder
"""

"""
We can also explore whether or not it's worth learning the initialisation for
the encoder convGRU. For now, we'll just explore zero and noise 
initializations.
"""

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

    def __init__(self, agent, device, num_recon_obs, num_pred_steps):
        super(VAE, self).__init__()

        # Set settings
        self.num_recon_obs    = num_recon_obs
        self.num_pred_steps   = num_pred_steps
        self.num_unroll_steps = self.num_recon_obs + self.num_pred_steps
        self.device = device

        # Create networks
        self.encoder = EncoderNetwork(device).to(device)
        self.decoder = DecoderNetwork(device, agent,
                           num_unroll_steps=self.num_unroll_steps).to(device)

    def forward(self, obs, agent_hxs, actions, use_true_h0=False, use_true_actions=False):

        # Ensure the number of images in the input sequence is the same as the
        # number of observations that we're _reconstructing_.
        assert obs.shape[1] == self.num_recon_obs

        # Feed observation sequence into encoder, which returns the mean
        # and log(variance) for the latent sample (i.e. the encoded sequence)
        mu, logvar = self.encoder(obs, agent_hxs)

        sigma = torch.exp(0.5 * logvar)  # log(var) -> standard deviation

        # Reparametrisation trick
        sample = (torch.randn(sigma.size(), device=self.device) * sigma) + mu

        # Decode
        if use_true_h0:
            true_h0 = agent_hxs[:, 0, :]
        else:
            true_h0 = None  # therefore decoder doesn't use.
        if use_true_actions:
            true_acts = actions
        else:
            true_acts = None # therefore decoder doesn't use.
        pred_obs, pred_rews, pred_dones, pred_agent_hs, pred_agent_logprobs = \
            self.decoder(sample, true_h0=true_h0, true_actions=true_acts)

        preds = {'obs': pred_obs,
                 'reward': pred_rews,
                 'done': pred_dones,
                 'hx': pred_agent_hs,
                 'act_log_probs': pred_agent_logprobs}

        return mu, logvar, preds


class EncoderNetwork(nn.Module):
    """The Variational Autoencoder that generates agent-environment sequences.

    Description

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
    def __init__(self, device):
        super(EncoderNetwork, self).__init__()
        self.input_network = EncoderInputNetwork(device, agent_hidden_size=64).to(device)
        self.rnn = EncoderRNN(device)
        self.embedder = EncoderEmbedder(device)
        self.device = device

    def forward(self, obs, agent_hxs):

        # Reset encoder RNN's hidden state (note: not to be confused with the
        # agent's hidden state, agent_hxs)
        h = None

        obs_seq_len = obs.shape[1] # (B, *T*, Ch, H, W)

        # Pass each image in the input input sequence into the encoder input
        # network to get a sequence of latent representations (one per image)
        inps = []
        for i in range(obs_seq_len):
            inps.append(self.input_network(obs[:,i,:], agent_hxs[:,i,:]))
        inps = torch.stack(inps, dim=1) # stack along time dimension

        # Pass sequence of latent representations
        h = self.rnn(inps, h)

        # Convert sequence embedding into VAE latent params
        mu, sigma = self.embedder(h)

        return mu, sigma




class EncoderInputNetwork(nn.Module):
    """Input Network for the encoder.

    Takes (a batch of) single images at a single timestep.

    Consists of a feedforward convolutional network. It has many residual
    connections, which sometimes skip several layers. In that sense, it is
    similar to a `dense` convolutional network, which has many such
    connections.

    It also `assimilates` the initial agent hidden state (1D vector) into the
    convolutional representations (3D tensor).

    Its output is passed to a recurrent network, which thus accumulates
    information about each image in the sequence.

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
    def __init__(self, device, agent_hidden_size=256):
        super(EncoderInputNetwork, self).__init__()
        hid_ch = 64
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=hid_ch,
                               kernel_size=3, padding=1).to(device)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.norm1 = nn.LayerNorm([hid_ch,32,32])
        self.resdown1 = lyr.ResBlockDown(hid_ch,hid_ch,downsample=self.pool)
        self.assimilatehx = lyr.AssimilatorResidualBlock(hid_ch, agent_hidden_size)
        self.norm2 = nn.LayerNorm([hid_ch,32,32])
        self.resdown2 = lyr.ResBlockDown(hid_ch,hid_ch,downsample=self.pool)
        self.norm3 = nn.LayerNorm([hid_ch,16,16])
        self.attention = lyr.Attention(hid_ch)
        self.norm4 = nn.LayerNorm([hid_ch,16,16])
        self.res1x1   = lyr.ResOneByOne(hid_ch+hid_ch*3, hid_ch)
        self.resdown3 = lyr.ResBlockDown(hid_ch,hid_ch,downsample=self.pool) # was to hid_ch*2
        self.norm5 = nn.LayerNorm([hid_ch,8,8]) # was to hid_ch*2

    def forward(self, ob, hx):
        x  = ob
        z  = self.conv0(x)
        z  = self.resdown1(z)
        x1 = self.norm1(z)
        z  = self.assimilatehx(x1, hx)
        x2 = self.norm2(z)
        z  = self.resdown2(x2)
        x3 = self.norm3(z)
        z  = self.attention(x3)
        x4 = self.norm4(z)
        x123 = torch.cat([self.pool(x1), self.pool(x2), x3], dim=1)
        z  = self.res1x1(x4, x123)
        z  = self.resdown3(z)
        z  = self.norm5(z)
        return z

class EncoderRNN(nn.Module):
    """Recurrent network for input image sequences.

    The `EncoderRNN` takes as input the outputs of `EncoderInputNetwork`s for
    each input image. It thus takes as input a sequence of image
    representations (not raw images). It learns to encode the dynamics of the
    input image sequence in order that the decoder can reconstruct the input
    image sequence and also predict subsequent images that were not actually
    in the input sequence.

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
    def __init__(self, device):
        super(EncoderRNN, self).__init__()
        hid_ch = 64
        self.rnn = lyr.ConvGRU(input_size=[8, 8], # [H,W]
                               input_dim=hid_ch, # ch      # was hid_ch*2
                               hidden_dim=hid_ch,          # was hid_ch*2
                               kernel_size=(3,3),
                               num_layers=1,
                               device=device)#.to(device)
        self.norm = nn.LayerNorm([hid_ch, 8, 8])

    def forward(self, inp, h=None):
        h = self.rnn(inp, h)
        h = self.norm(h[0][:,-1])  # only use final hidden state
        return h


class EncoderEmbedder(nn.Module):
    """Converts the output of the RNN to the VAE's latent sample params.

    The encoder embedder is the final layer of the VAE encoder.

    The output of the EncoderRNN is passed to the encoder embedders, which
    simply does some non-recurrent processing in order to generate the
    mean and log(variance) of the sample in the latent space of the VAE that
    is used, by the decoder, to reconstruct the input image sequence and
    predict future images.

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
    def __init__(self, device):
        super(EncoderEmbedder, self).__init__()
        hid_ch = 64

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.resdown = lyr.ResBlockDown(in_channels=hid_ch,
                                        out_channels=hid_ch,
                                        downsample=self.pool).to(device)
        self.norm = nn.LayerNorm([hid_ch,4,4])
        self.fc_mu    = nn.Linear(4*4*hid_ch, 128).to(device)
        self.fc_sigma = nn.Linear(4*4*hid_ch, 128).to(device)

    def forward(self, inp):
        x = inp
        x = self.resdown(x)
        x = self.norm(x)
        mu = self.fc_mu(x.view(x.shape[0], -1))
        logvar = self.fc_sigma(x.view(x.shape[0], -1))
        return mu, logvar

class DecoderNetwork(nn.Module):
    """Reconstructs and predicts agent-environment sequences.

    Generates a whole agent-environment sequence from a latent sample from the
    VAE. It contains a copy of the agent. The VAE is trained such that the
    decoder produces:
        - a sequence of observations
        - a sequence of rewards
        - a sequence of hidden states of the agent
        - a sequence of actions from the agent (and their log probabilities)
    that match as closely as possible the sequences that were actually observed
    in real roll outs of the agent-environment system.

    It uses a recurrent architecture to generate sequences of arbitrary length,
    so sequences longer than the training sequences may be generated if wanted.
    The recurrent architecture consists of an environment and an agent part:
    the agent part is the agent itself, which takes as input an observation
    of the (simulated) environment and its own hidden state. The environment
    part consists of several networks:
        - An EnvStepper, which has 'environment hidden state' which unrolls
          through time
        - An ObservationDecoder, which takes the environment hidden state and
          converts it into an observation for that timestep.
        - A RewardDecoder, which does the same for the reward that agent
          received at that timestep.
        - A DoneDecoder, which predicts whether the episode is done at
          that timestep.

    Since both the agent and the EnvStepper are recurrent, they require inputs
    to get the ball rolling. There are several networks that convert the latent
    VAE sample into the inputs required. Those are:
        - inithidden_network, which generates the initial hidden state of the
          agent. We don't need to produce later hidden states, because the
          agent does that itself.
        - env_init_network, which generates the initial hidden state of the
          EnvStepper.

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
    def __init__(self, device, agent, num_unroll_steps):
        super(DecoderNetwork, self).__init__()
        self.action_dim = 15
        agent_hidden_size = 64
        hid_ch = 1024
        env_h_hw = 16

        # Initializers
        self.inithidden_network = TwoLayerPerceptron(
            insize=128,
            outsize=agent_hidden_size)
        self.env_init_network = EnvStepperInitializer(device=device, env_h_hw=env_h_hw, vae_latent_size=128)


        # Stepper (the one that gets unrolled)
        self.env_stepper      = EnvStepper(agent, env_h_ch=hid_ch, env_h_hw=env_h_hw, vae_latent_size=128)

        # Decoders (used at every timestep
        self.reward_decoder = RewardDecoder(device, env_h_hw=env_h_hw)
        self.done_decoder   = DoneDecoder(device, env_h_hw=env_h_hw)
        self.obs_decoder    = ObservationDecoder(device, env_h_hw=env_h_hw)
        self.agent = agent
        self.num_unroll_steps = num_unroll_steps

    def forward(self, sample, true_h0=None, true_actions=None):

        # Get initial inputs to the agent and EnvStepper (both recurrent)
        env_h = self.env_init_network(sample)  # t=0
        agent_h = self.inithidden_network(sample)  # t=0

        # Unroll the agent and EnvStepper and collect the generated data
        pred_obs = []
        pred_rews = []
        pred_dones = []
        pred_env_hs = []
        pred_agent_hs = []
        pred_acts = []
        pred_agent_logprobs = []

        for i in range(self.num_unroll_steps):
            # Within single timestep t
            ## Observations@t
            obs = self.obs_decoder(env_h)
            pred_obs.append(obs)

            ## Rewards@t
            rew = self.reward_decoder(env_h)
            pred_rews.append(rew)

            ## Dones@t
            done = self.done_decoder(env_h)
            pred_dones.append(done)

            # Moving forward in time: t <- t+1
            ## Store curr agent-hidden state

            pred_agent_hs.append(agent_h)

            ## Step the agent forward
            obs = obs.permute(0,2,3,1)
            if true_h0 is not None and i == 0:
                # if we want to feed the correct h0 instead of the
                # guessed h0 (for purposes of being able to train the
                # agent and the generative model on different agents), then
                # we need to swap in the true h0 here, but we'll still store
                # and return the guessed h0 in preds so that it can be
                # trained to approximate true_h0.
                agent_h = true_h0
            act, logits, value, agent_h = self.agent.predict_STE(obs, agent_h, done)
            ## Now it's act@t and logprobs@t, but agent_h@t+1
            # Note this 'act' is different from train and eval because it's a
            # one-hot vector. Also because we're passing gradients back
            # through the action using its logits as a straight through
            # estimator. (TODO the STE feature is now redundant now that
            #  we're using true actions during training).

            ## Store act and logprob
            pred_acts.append(act)
            pred_agent_logprobs.append(logits)

            ## Step environment forward: use env_h@t and act@t to get env_h@t+1

            if true_actions is not None:
                # get true actions for that step and overwrite predicted
                # actions for input to env
                act = true_actions[:, i]
                act = torch.nn.functional.one_hot(
                    act.long(), num_classes=15)
            pred_env_hs.append(env_h)
            env_h = self.env_stepper(sample, act, h=env_h)

            ## Get ready for new step
            self.agent.train_prev_recurrent_states = None

        return pred_obs, pred_rews, pred_dones, pred_agent_hs, pred_agent_logprobs


class TwoLayerPerceptron(nn.Module):
    """A two layer perceptron with layer norm and a linear output.

    It takes the VAE latent sample as input and outputs another vector.

    This class is used to make the inithidden_network. Its output is a vector
    the size of the agent's hidden state and initializes the agent.

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
                          nn.LayerNorm(insize),
                          nn.Linear(insize, outsize))

    def forward(self, x):
        return self.net(x)


class EnvStepperInitializer(nn.Module):
    """Initializes the EnvStepper

    The EnvStepper is recurrent and therefore needs an initial hidden state.
    The EnvStepperInitializer generates an initial hidden state by taking the
    VAE latent sample as input and outputting something the size of the
    EnvStepper hidden state.

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
    def __init__(self, device, vae_latent_size=256, env_h_ch=64, env_h_hw=8):
        super(EnvStepperInitializer, self).__init__()

        hid_ch = 64

        self.env_h_ch = env_h_ch
        self.env_h_hw = env_h_hw
        self.fc = nn.Linear(vae_latent_size,
                            int((env_h_ch*env_h_hw*env_h_hw)/(4**2)))
        self.resblockup1 = lyr.ResBlockUp(in_channels=hid_ch,
                                         out_channels=hid_ch,
                                         hw=4)
        self.resblockup2 = lyr.ResBlockUp(in_channels=hid_ch,
                                         out_channels=hid_ch,
                                         hw=8)
        self.norm1 = nn.LayerNorm([hid_ch, env_h_hw, env_h_hw])
        self.norm2 = nn.LayerNorm([hid_ch, env_h_hw, env_h_hw])
        self.norm3 = nn.LayerNorm([hid_ch, env_h_hw, env_h_hw])
        self.norm4 = nn.LayerNorm([hid_ch, env_h_hw, env_h_hw])

        self.resblock1 = lyr.ResidualBlock(hid_ch)
        self.resblock2 = lyr.ResidualBlock(hid_ch)
        self.attention = lyr.Attention(hid_ch)

    def forward(self, x):
        x = self.fc(x)
        x = self.resblockup1(x.view(x.shape[0],       self.env_h_ch,
                                   self.env_h_hw//4, self.env_h_hw//4)
                            )
        x = self.resblockup2(x)
        x = self.norm1(x)
        x = self.resblock1(x)
        x = self.norm2(x)
        x = self.attention(x)
        x = self.norm3(x)
        x = self.resblock2(x)
        x = self.norm4(x)
        return x

class EnvStepper(nn.Module):
    """A recurrent network that simulates the unrolling of the environment.

    The EnvStepper unrolls a latent representation of the environment through
    time. From its hidden state is decoded several things at each timestep:
      - The observation (by the ObservationDecoder), which is input to the
        agent.
      - The reward (by the RewardDecoder). It is trained to predict reward in
        the expectation that reward-salient aspects of the environment will be
        represented in the EnvStepper latent state.
      - The 'done' status (by the DoneDecoder). It is trained to predict
        whether the episode is done or not.

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
    def __init__(self, agent, env_h_ch=128, env_h_hw=8, vae_latent_size=256):
        super(EnvStepper, self).__init__()
        action_dim = 15
        hid_ch = 64

        self.assimilator = \
            lyr.AssimilatorResidualBlock(hid_ch,
                                         vec_size=(action_dim+vae_latent_size))
        # EnvStepper hidden state shape is 8x8x128
        self.attention = lyr.Attention(hid_ch)
        self.norm1 = nn.LayerNorm([hid_ch, env_h_hw, env_h_hw])
        self.norm2 = nn.LayerNorm([hid_ch, env_h_hw, env_h_hw])
        self.norm3 = nn.LayerNorm([hid_ch, env_h_hw, env_h_hw])
        self.resblock = lyr.ResidualBlock(hid_ch)

    def forward(self, vae_sample, act, h=None):
        vec = torch.cat([vae_sample, act], dim=1)
        new_h = self.assimilator(h, vec)
        new_h = self.norm1(new_h)
        new_h = self.attention(new_h)
        new_h = self.norm2(new_h)
        new_h = self.resblock(new_h)
        new_h = self.norm3(new_h)
        return new_h

class ObservationDecoder(nn.Module):
    """Decodes the observation from the EnvStepper latent state.

    At every timestep, the ObservationDecoder takes the EnvStepper latent state
    as input and outputs the observation (what the agent sees).

    It uses multiple upsampling residual blocks and a nonlocal layer
    (self-attention). It has a final tanh activation to ensure predicted pixel
    values have the same range as real pixels.

    """
    def __init__(self, device, env_h_ch=64, env_h_hw=8):
        super(ObservationDecoder, self).__init__()
        ch0 = env_h_ch
        hw0 = env_h_hw

        self.resblockup1 = lyr.ResBlockUp(ch0,    ch0//2, hw=hw0)
        self.resblockup2 = lyr.ResBlockUp(ch0//2, 3, hw=hw0*2)
        #self.resblockup3 = lyr.ResBlockUp(ch0//4, 3,      hw=hw0*4)
        self.attention = lyr.Attention(ch0 // 2)
        # self.net = nn.Sequential(self.resblockup1,
        #                          self.attention,
        #                          self.resblockup2,
        #                          self.resblockup3)

    def forward(self, x):
        x = self.resblockup1(x)
        x = self.attention(x)
        x = self.resblockup2(x)
        #x = self.resblockup3(x)
        x = ( torch.tanh(x) / 2.) #+ 0.5 #NEW
        return x


class RewardDecoder(nn.Module):
    """Decodes the current reward from the EnvStepper latent state.

    At every timestep, the RewardDecoder takes the EnvStepper latent state
    as input and outputs the (predicted) reward for that timestep. The agent
    doesn't see this reward directly. It is just used to train the VAE
    in the hope that reward-salient aspects of the environment will be
    represented in the EnvStepper latent state.

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
    def __init__(self, device, env_h_ch=64, env_h_hw=8):
        super(RewardDecoder, self).__init__()
        ch0 = env_h_ch
        self.resblockdown = lyr.ResBlockDown(ch0,ch0//4,
                               downsample=nn.MaxPool2d(kernel_size=2))
        half_env_h_hw = env_h_hw//2
        self.fc = nn.Linear(half_env_h_hw*half_env_h_hw*(ch0//4), 1)

    def forward(self, x):
        x = self.resblockdown(x)
        x = self.fc(x.view(x.shape[0], -1))
        return x

class DoneDecoder(nn.Module):
    """Decodes the 'Done' status from the EnvStepper latent state.

    At every timestep, the DoneDecoder takes the EnvStepper latent state
    as input and outputs the (predicted) 'Done' status for that timestep. This
    indicates when the similator thinks the episode is over (e.g. if the agent
    dies or completes the level).

    It is identical to the RewardDecoder apart from the final sigmoid
    activation, since we want to return a prediction for a boolean here.

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
    def __init__(self, device, env_h_ch=64, env_h_hw=8):
        super(DoneDecoder, self).__init__()
        ch0 = env_h_ch
        self.resblockdown = lyr.ResBlockDown(ch0,ch0//4,
                               downsample=nn.AvgPool2d(kernel_size=2))
        half_env_h_hw = env_h_hw//2
        self.fc = nn.Linear(half_env_h_hw*half_env_h_hw*(ch0//4), 1)

    def forward(self, x):
        x = self.resblockdown(x)
        x = self.fc(x.view(x.shape[0], -1))
        return torch.sigmoid(x)

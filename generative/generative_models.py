from .rssm.encoders import *
from .rssm.decoders import *
from .rssm.functions import dclamp, insert_dim, terminal_labels_to_mask
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import LayeredConvNet, LayeredResBlockUp, LayeredResBlockDown, NLayerPerceptron


# Note that dclamp is a custom clamp function to clip the values of the image
#  to be in [0,1]. From https://discuss.pytorch.org/t/exluding-torch-clamp-from-backpropagation-as-tf-stop-gradient-in-tensorflow/52404/6

class AgentEnvironmentSimulator(nn.Module):

    def __init__(self, agent, device, hyperparams):
        super(AgentEnvironmentSimulator, self).__init__()

        # Hyperparams
        self.num_init_steps  = hyperparams.num_init_steps
        self.num_sim_steps = hyperparams.num_sim_steps
        self.kl_balance    = hyperparams.kl_balance
        self.kl_weight     = hyperparams.kl_weight

        # Networks
        self.conv_in = MultiEncoder(cnn_depth=32, image_channels=3)
        hyperparams.__dict__.update({'embed_dim': self.conv_in.out_dim})
        self.env_stepper_initializer = EnvStepperInitializer(hyperparams, device)
        features_dim = hyperparams.deter_dim + hyperparams.stoch_dim * (hyperparams.stoch_discrete or 1)
        self.conv_out = MultiDecoder(features_dim, hyperparams)
        self.agent_env_stepper = AgentEnvStepper(hyperparams, agent)
        self.action_size = agent.env.action_space.n


    def forward(self, data, use_true_actions=True, imagine=False, modal_sampling=False,
                retain_grads=True, swap_directions=None):
        """
        imagine: Whether or not to use the generated images as input to
        the env model or whether to use true images (true images will be used
        during training).

        """

        # Get input data for generative model
        full_ims = data['ims']
        true_agent_h0 = data['hx'][self.num_init_steps-1] # -1 because 1st sim step is last init step
        agent_h_labels = data['hx'][self.num_init_steps-1:]
        true_actions_inds = data['action'][self.num_init_steps-2:] # -2 because we have to use a_{t-1} in combo with env_t to get o_t
        true_actions_1hot = torch.nn.functional.one_hot(true_actions_inds.long(),
                                                  self.action_size)
        true_actions_1hot = true_actions_1hot.float()

        reward_labels = data['reward'][self.num_init_steps-1:]
        terminal_labels = data['terminal'][self.num_init_steps-1:]
        # terminal_labels = terminal_labels_to_mask(terminal_labels)

        # Initialize env@t=0 and agent_h@t=0 # Later on, we're going to have to extract the initial steps here in order to make a VAE encoder from the initializer
        B = full_ims.shape[1]
        init_ims = full_ims[0:self.num_init_steps]
        im_labels = full_ims[-self.num_sim_steps:]

        if imagine:
            embeds = [None] * self.num_sim_steps
        else:
            embeds = self.conv_in(im_labels)
        env_h_prev, env_z_prev = self.env_stepper_initializer(init_ims)
        agent_h_prev = true_agent_h0
        action_prev = pred_action_1hot = true_actions_1hot[0]
        # TODO confirm the actions are one-hotted
        # TODO confirm whether you can get away without an action here
        #  (because will be better for gen model without)
        # TODO confirm the actions are aligned

        priors = []
        posts = []
        pred_actions = []
        pred_action_log_probs = []
        pred_values = []
        states_env_h = []
        samples = []
        agent_hs = []
        recon_losses = []
        metrics_list = []
        tensors_list = []

        for i in range(self.num_sim_steps):
            # Define the labels for the loss function because we calculate it
            #  in here.
            labels = {'image': im_labels[i],
                      'reward':reward_labels[i],
                      'terminal':terminal_labels[i],
                      'agent_h':agent_h_labels[i+1]} # +1 because ag_h_{t-1} is input to stepper and to agent, but it outputs ag_h_t(hat). We want the label to be ag_h_t.

            embed = embeds[i]

            (post,    # tensor(B, 2*S)
            pred_action_1hot,
            pred_action_log_prob,
            pred_value,
            image_rec, # _pred for predicted, _rec for reconstructed, i.e. basically the same thing.
            rew_rec,
            terminal_rec,
            (env_h, env_z),      # tensor(B, D+S+G)
            agent_h,
            loss_reconstr,
            metrics,
            tensors) = \
                self.agent_env_stepper.forward(embed=embed,
                                            action_prev=action_prev,
                                            agent_h_prev=agent_h_prev,
                                            env_state_prev=(env_h_prev, env_z_prev),
                                            imagine=imagine,
                                            modal_sampling=modal_sampling,
                                            labels=labels)
            posts.append(post)
            pred_actions.append(pred_action_1hot)
            pred_action_log_probs.append(pred_action_log_prob)
            pred_values.append(pred_value)
            states_env_h.append(env_h)
            samples.append(env_z)
            agent_hs.append(agent_h)
            recon_losses.append(loss_reconstr)
            metrics_list.append(metrics)
            tensors_list.append(tensors)

            if use_true_actions:
                action_prev = true_actions_1hot[i + 1]  # +1 because index0 is a_{t-1}
            else:
                action_prev = pred_action_1hot
            agent_h_prev = agent_h
            env_h_prev, env_z_prev = (env_h, env_z)

        posts = torch.stack(posts)                  # (T,B,2S)
        pred_actions = torch.stack(pred_actions)
        pred_action_log_probs = torch.stack(pred_action_log_probs)
        pred_values = torch.stack(pred_values)
        states_env_h = torch.stack(states_env_h)    # (T,B,D)
        samples = torch.stack(samples)              # (T,B,S)
        agent_hs = torch.stack(agent_hs)
        priors = self.agent_env_stepper.batch_prior(states_env_h)  # (T,B,2S)
        features = torch.cat([states_env_h, samples], dim=-1)   # (T,B,D+S)
        env_states = (states_env_h, samples)
        recon_losses = torch.stack(recon_losses).squeeze()

        # KL loss
        d = self.agent_env_stepper.zdistr
        dprior = d(priors)
        dpost = d(posts)
        loss_kl_exact = D.kl.kl_divergence(dpost, dprior)  # (T,B)

        # Analytic KL loss, standard for VAE
        if not self.kl_balance:
            loss_kl = loss_kl_exact
        else:
            loss_kl_postgrad = D.kl.kl_divergence(dpost, d(priors.detach()))
            loss_kl_priograd = D.kl.kl_divergence(d(posts.detach()), dprior)
            loss_kl = (1 - self.kl_balance) * loss_kl_postgrad + \
                      self.kl_balance       * loss_kl_priograd

        # Total loss
        assert loss_kl.shape == recon_losses.shape
        loss_model = self.kl_weight * loss_kl + recon_losses

        # Make preds_dict
        pred_images, pred_terminals, pred_rews = extract_preds_from_tensors(
            self.num_sim_steps, tensors_list)
        preds_dict = {'obs': pred_images,
                      'hx': agent_hs,
                      'reward': pred_rews,
                      'done': pred_terminals,
                      'act_log_probs': pred_action_log_probs,
                      'value': pred_values,
                      # 'sample_vecs': sample_vecs,
                      'env_h': states_env_h}

        return (
            loss_model,
            priors,                      # tensor(T,B,2S)
            posts,                       # tensor(T,B,2S)
            samples,                     # tensor(T,B,S)
            features,                    # tensor(T,B,D+S)
            env_states,
            (env_h.detach(), env_z.detach()),
            metrics_list,
            tensors_list,
            pred_actions,
            agent_hs
        )


class AgentEnvStepper(nn.Module):
    """

    """
    def __init__(self, hyperparams, agent):
        super(AgentEnvStepper, self).__init__()

        # Hyperparams
        self.image_range_min, self.image_range_max = (0, 1)
        self.stoch_dim = hyperparams.stoch_dim
        self.stoch_discrete = hyperparams.stoch_discrete
        self.deter_dim = hyperparams.deter_dim
        norm = nn.LayerNorm if hyperparams.layer_norm else NoNorm

        # Networks
        self.z_mlp = nn.Linear(hyperparams.stoch_dim * (hyperparams.stoch_discrete or 1), hyperparams.hidden_dim)
        self.a_mlp = nn.Linear(hyperparams.action_dim, hyperparams.hidden_dim, bias=False)  # No bias, because outputs are added
        self.in_norm = norm(hyperparams.hidden_dim, eps=1e-3)

        self.gru = GRUCellStack(hyperparams.hidden_dim, hyperparams.deter_dim, 1, 'gru_layernorm')

        self.prior_mlp_h = nn.Linear(hyperparams.deter_dim, hyperparams.hidden_dim)
        self.prior_norm = norm(hyperparams.hidden_dim, eps=1e-3)
        self.prior_mlp = nn.Linear(hyperparams.hidden_dim, hyperparams.stoch_dim * (hyperparams.stoch_discrete or 2))

        self.post_mlp_h = nn.Linear(hyperparams.deter_dim, hyperparams.hidden_dim)
        self.post_mlp_e = nn.Linear(hyperparams.embed_dim, hyperparams.hidden_dim, bias=False)
        self.post_norm = norm(hyperparams.hidden_dim, eps=1e-3)
        self.post_mlp = nn.Linear(hyperparams.hidden_dim, hyperparams.stoch_dim * (hyperparams.stoch_discrete or 2))

        features_dim = hyperparams.deter_dim + hyperparams.stoch_dim * (hyperparams.stoch_discrete or 1)
        self.decoder = MultiDecoder(features_dim, hyperparams)

        self.agent = agent
        self.device = agent.device

    def forward(self,
                embed: Tensor,                    # tensor(B,E)
                action_prev: Tensor,                   # tensor(B,A)
                env_state_prev: Tuple[Tensor, Tensor],
                agent_h_prev,
                imagine,
                modal_sampling,
                labels,
                ):

        in_h, in_z = env_state_prev
        B = action_prev.shape[0]

        x = self.z_mlp(in_z) + self.a_mlp(action_prev)  # (B,H)
        x = self.in_norm(x)
        za = F.elu(x)
        h = self.gru(za, in_h)              # (B, D)

        if imagine:
            x = self.prior_mlp_h(h)
            x = self.prior_norm(x)
            x = F.elu(x)
            prior = self.prior_mlp(x)       # (B,2S)
            prior_distr = self.zdistr(prior)
            if modal_sampling:
                # Uses Straight Through Gradients
                inds = prior_distr.mean.argmax(dim=2)
                mode_one_hot = torch.nn.functional.one_hot(inds, num_classes=self.stoch_discrete).to(self.agent.device)
                sample = prior_distr.mean + \
                      (mode_one_hot - prior_distr.mean).detach()
                sample = sample.reshape(B, -1)
            else:
                sample = prior_distr.rsample().reshape(B, -1)
            post_or_prior = prior
        else:
            x = self.post_mlp_h(h) + self.post_mlp_e(embed)
            x = self.post_norm(x)
            post_in = F.elu(x)
            post = self.post_mlp(post_in)   # (B, S*S)
            post_distr = self.zdistr(post)
            sample = post_distr.rsample().reshape(B, -1)
            post_or_prior = post

        feature = torch.cat([h, sample], dim=-1)
        BF_to_TBIF = lambda x: torch.unsqueeze(torch.unsqueeze(x, 1), 0)
        BF_to_TBF = lambda x: torch.unsqueeze(x, 0)
        feature = BF_to_TBIF(feature)
        labels = {k: BF_to_TBF(v) for k, v in labels.items()}
        # Now decode the env into ims
        loss_reconstr, metrics, tensors, image_rec, rew_rec, terminal_rec = \
            self.decoder.training_step(feature, labels)
        # Note that XXX_rec has grads but tensors['XXX_rec'] does not

        # Then use ims and agent_h to step the agent forward and produce an action
        image_rec = image_rec.squeeze()
        image_rec = dclamp(image_rec, self.image_range_min, self.image_range_max)
        no_masks = torch.zeros_like(labels['terminal'])
        pred_action, pred_action_logits, pred_value, agent_h = \
            self.agent.predict_STE(image_rec, agent_h_prev, no_masks,
                                   retain_grads=True) # Lee: I'm ignorant of whether the terminal-masks are doing anything potentially dangerous
        loss_reconstr_agent_h = self.agent_hx_loss(agent_h, labels['agent_h'], labels['terminal']) # TODO appropriate masking using terminal. We don't want agent hx to be trained after the episode is done
        loss_reconstr = loss_reconstr + loss_reconstr_agent_h # TODO try without this loss first because I'm not sure it's aligned or otherwise working properly
        return (
            post_or_prior,    # tensor(B, 2*S)
            pred_action,
            pred_action_logits,
            pred_value,
            image_rec, # _pred for predicted, _rec for reconstructed, i.e. basically the same thing.
            rew_rec,
            terminal_rec,
            (h, sample),      # tensor(B, D+S+G)
            agent_h,
            loss_reconstr,
            metrics,
            tensors
        )

    def agent_hx_loss(self, pred_agent_h, label_agent_h, terminals):
        """Based on jurgisp's 'vecobs_decoder'. To be honest I don't understand
         why the std is pre-specified like this. But it's unlikely to be hugely
         important; the agent_h_loss is auxiliary."""
        std = 0.3989422804
        var = std ** 2 # var cancels denominator, which makes loss = 0.5 (target-output)^2
        p = D.Normal(loc=pred_agent_h, scale=torch.ones_like(pred_agent_h) * std)
        p = D.independent.Independent(p, 1)  # Makes p.logprob() sum over last dim
        loss = -p.log_prob(label_agent_h) * var
        loss = loss * terminal_labels_to_mask(terminals)
        loss = loss.unsqueeze(-1)
        return loss

    def batch_prior(self,
                    h: Tensor,     # tensor(T, B, D)
                    ) -> Tensor:
        x = self.prior_mlp_h(h)
        x = self.prior_norm(x)
        x = F.elu(x)
        prior = self.prior_mlp(x)  # tensor(B,2S)
        return prior

    def zdistr(self, pp: Tensor) -> D.Distribution:
        # pp = post or prior
        if self.stoch_discrete:
            logits = pp.reshape(pp.shape[:-1] + (self.stoch_dim, self.stoch_discrete))
            distr = D.OneHotCategoricalStraightThrough(logits=logits.float())  # NOTE: .float() needed to force float32 on AMP
            distr = D.independent.Independent(distr, 1)  # This makes d.entropy() and d.kl() sum over stoch_dim
            return distr
        else:
            return diag_normal(pp)


class EnvStepperInitializer(nn.Module):
    """

    Recurrent network that takes a seq of frames from t=-k to t=0 as input.
    The final output gets passed along with the final agent hidden state into
    an FC network. It outputs the initial state of the environment simulator.

    """
    def __init__(self, hyperparams, device):
        super(EnvStepperInitializer, self).__init__()
        self.device = device
        self.num_init_steps = hyperparams.num_init_steps
        self.rnn_hidden_size = hyperparams.initializer_rnn_hidden_size
        self.env_dim = hyperparams.deter_dim
        self.env_h_stoch_size = hyperparams.env_h_stoch_size

        self.image_embedder = LayeredResBlockDown(input_hw=64,
                                                  input_ch=3,
                                                  hidden_ch=64,
                                                  output_hw=8,
                                                  output_ch=32)

        self.rnn = nn.GRU(input_size=self.image_embedder.output_size,
                           hidden_size=self.rnn_hidden_size,
                           num_layers=1,
                           batch_first=False)

        self.mlp_out = nn.Sequential(
                                nn.Linear(self.rnn_hidden_size,
                                          self.rnn_hidden_size),
                                nn.ELU(),
                                nn.Linear(self.rnn_hidden_size,
                                          self.env_dim)
                                            )

    def forward(self, init_ims):
        """"""
        # Flatten inp seqs along time dimension to pass all to conv nets
        # along batch dim
        x = init_ims
        ts = x.shape[0]
        batches = x.shape[1]
        h = x.shape[2]
        w = x.shape[3]
        ch = x.shape[4]

        images = [x[i] for i in range(ts)]  # split along time dim
        embeddings = [self.image_embedder(im) for im in images]
        embeddings = [im for (im, _) in embeddings]
        x = torch.stack(embeddings, dim=0)  # stack along time dim

        # Flatten conv outputs to size (H*W*CH) to get rnn input vecs
        x = x.view(ts, batches,  -1)

        # Pass seq of vecs to initializer RNN
        x, _ = self.rnn(x)

        # Concat RNN output to agent h0 and then pass to Converter nets
        # to get mu_g and sigma_g
        x = x[-1]  # get last ts
        init_env_determ_state = self.mlp_out(x)
        init_env_stoch_state = torch.zeros(batches, self.env_h_stoch_size, device=self.device)
        return init_env_determ_state, init_env_stoch_state # TODO confirm that it's okay that this is actually 0s

def extract_preds_from_tensors(num_sim_steps, tensors_list): # TODO duplicated in gen_model_experiment.py class
    pred_images = torch.cat(
        [tensors_list[t]['image_rec'] for t in range(num_sim_steps)],
        dim=0)
    pred_terminals = torch.cat(
        [tensors_list[t]['terminal_rec'] for t in range(num_sim_steps)],
        dim=0)
    pred_rews = torch.cat(
        [tensors_list[t]['reward_rec'] for t in range(num_sim_steps)],
        dim=0)
    return pred_images, pred_terminals, pred_rews
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

    def __init__(self,
                 agent,
                 device,
                 hyperparams):
        super(AgentEnvironmentSimulator, self).__init__()

        # Hyperparams
        self.num_init_steps = hyperparams.num_init_steps
        self.num_sim_steps  = hyperparams.num_sim_steps
        self.kl_balance     = hyperparams.kl_balance
        self.kl_weight      = hyperparams.kl_weight
        self.vae_kl_weight  = hyperparams.vae_kl_weight
        self.action_size = agent.env.action_space.n
        self.env_h_stoch_size = hyperparams.env_h_stoch_size
        self.latent_vec_size = hyperparams.latent_vec_size

        # Networks
        self.conv_in = MultiEncoder(cnn_depth=32, image_channels=3)
        hyperparams.__dict__.update({'embed_dim': self.conv_in.out_dim})
        self.encoder = VAEEncoder(hyperparams, device)
        self.latent_vec_converter_env = NLayerPerceptron(
            [hyperparams.latent_vec_size,
             int((hyperparams.latent_vec_size + hyperparams.deter_dim + hyperparams.env_h_stoch_size)/2),
             hyperparams.deter_dim + hyperparams.env_h_stoch_size],
        )
        self.latent_vec_converter_agent_h = NLayerPerceptron(
            [hyperparams.latent_vec_size,
             int((hyperparams.latent_vec_size + hyperparams.agent_hidden_size)/2),
             hyperparams.agent_hidden_size],
        ) # Note that this net does not influence the representations learned
        #  by the VAE latent vec because the VAE sample is detached before
        # passing to this network. Same for latent_vec_converter_action
        self.latent_vec_converter_action = NLayerPerceptron(
            [hyperparams.latent_vec_size,
             int((hyperparams.latent_vec_size + hyperparams.action_dim)/2),
             hyperparams.action_dim],
        )
        features_dim = hyperparams.deter_dim + hyperparams.stoch_dim * (hyperparams.stoch_discrete or 1)
        self.conv_out = MultiDecoder(features_dim, hyperparams)
        self.agent_env_stepper = AgentEnvStepper(hyperparams, agent)
        self.device = device


    def forward(self,
                data,
                use_true_actions=True,
                use_true_agent_h0=True,
                imagine=False,
                modal_sampling=False,
                retain_grads=True,
                swap_directions=None):

        calc_loss = not imagine

        init_ims = data['ims'][0:self.num_init_steps]
        #sample_vec, mu, logvar = self.encoder(init_ims)
        latent_vec = self.encoder(init_ims)

        B = init_ims.shape[1]

        # Get true agent h0 and true actions as aux vars for decoder
        if use_true_agent_h0:
            true_agent_h0 = data['hx'][self.num_init_steps - 1]  # -1 because 1st sim step is last init step
        else:
            true_agent_h0 = None  # generated in the VAE decoder

        if use_true_actions:
            true_actions_inds = data['action'][self.num_init_steps-2:] # -2 because we have to use a_{t-1} in combo with env_t to get o_t
            true_actions_1hot = torch.nn.functional.one_hot(true_actions_inds.long(), self.action_size)
            true_actions_1hot = true_actions_1hot.float()
        else:
            true_actions_1hot = torch.zeros(self.num_sim_steps, B, self.action_size, device=self.device)

        (   loss_model,
            loss_agent_h0,
            priors,  # tensor(T,B,2S)
            posts,  # tensor(T,B,2S)
            samples,  # tensor(T,B,S)
            features,  # tensor(T,B,D+S)
            env_states,
            (env_h, env_z),
            metrics_list,
            tensors_list,
            preds_dict,
        ) = self.vae_decode(latent_vec,
                             data,
                             true_actions_1hot=true_actions_1hot,
                             use_true_actions=use_true_actions,
                             true_agent_h0=true_agent_h0,
                             imagine=imagine,
                             calc_loss=calc_loss,
                             modal_sampling=modal_sampling,
                             retain_grads=True,)

        if calc_loss:
            # KL divergence loss for VAE bottleneck
            loss_latent_vec = 0. #0.5 * torch.sum(latent_vec.pow(2), dim=1)  # -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1)  # Sum over latent dim
        else:
            loss_latent_vec = 0.

        return (
            loss_model,
            loss_latent_vec,
            loss_agent_h0,
            priors,                      # tensor(T,B,2S)
            posts,                       # tensor(T,B,2S)
            samples,                     # tensor(T,B,S)
            features,                    # tensor(T,B,D+S)
            env_states,
            (env_h.detach(), env_z.detach()),
            metrics_list,
            tensors_list,
            preds_dict,
        )

    def vae_decode(self,
                latent_vec,
                data,
                true_actions_1hot=None,
                use_true_actions=True,
                true_agent_h0=None,
                use_true_agent_h0=True,
                imagine=False,
                calc_loss=True,
                modal_sampling=False,
                retain_grads=True,
                ):
        """
        imagine: Whether or not to use the generated images as input to
        the env model or whether to use true images (true images will be used
        during training).

        """

        # Get labels for loss function
        if calc_loss: # No need to calc loss
            agent_h_labels = data['hx'][self.num_init_steps-1:]
            reward_labels = data['reward'][self.num_init_steps-1:]
            terminal_labels = data['terminal'][self.num_init_steps-1:]
            before_terminal_labels = terminal_labels_to_mask(terminal_labels)
            im_labels = data['ims'][-self.num_sim_steps:]

        B = latent_vec.shape[0]

        # Use latent_vec to get init vectors: env_h_prev, env_z_prev,
        #  agent_h_prev, and action_prev

        ## First env_h_prev, env_z_prev (env_z_prev uses Straight Through
        ##  Gradients)
        env_prev, _ = self.latent_vec_converter_env(latent_vec)
        env_h_prev, env_z_prev = env_prev[:,:-self.env_h_stoch_size],\
                                 env_prev[:,-self.env_h_stoch_size:]
        init_z_dist = self.agent_env_stepper.zdistr(env_z_prev)
        ### Modal sampling
        # inds = init_z_dist.mean.argmax(dim=2)
        # mode_one_hot = torch.nn.functional.one_hot(inds,
        #                num_classes=self.agent_env_stepper.stoch_discrete).to(
        #                self.device)
        # env_z_prev = init_z_dist.mean + \
        #          (mode_one_hot - init_z_dist.mean).detach()
        # env_z_prev = env_z_prev.reshape(B, -1)
        ### Random sampling
        env_z_prev = init_z_dist.rsample().reshape(B, -1)


        ## Second, agent_h_prev
        pred_agent_h_prev, _ = self.latent_vec_converter_agent_h(latent_vec.detach()) # So that the sample vec is only learns to produce good env states, not contain any representations specific to an agent hx.
        pred_agent_h_prev = torch.tanh(pred_agent_h_prev)
        if use_true_agent_h0:
            agent_h_prev = true_agent_h0
        else:
            agent_h_prev = pred_agent_h_prev

        ## Third, action_prev (no straight through grads because the agent's init
        ##  vectors should be trained independently from the rest of the model)
        pred_action_prev_logits, _ = self.latent_vec_converter_action(latent_vec.detach())
        pred_action_prev_probs = torch.softmax(pred_action_prev_logits, dim=1)
        pred_action_prev_inds = pred_action_prev_probs.argmax(dim=1)
        pred_action_prev_1hot = torch.nn.functional.one_hot(pred_action_prev_inds,
                                                            num_classes=self.action_size).to(self.device).float()
        if use_true_actions:
            action_prev  = true_actions_1hot[0]
        else:
            action_prev = pred_action_prev_1hot

        if calc_loss:
            # MSE for h0
            loss_agent_h0 = torch.sum((pred_agent_h_prev - true_agent_h0)**2, dim=1)  # Sum over h dim

            # CE loss for action
            ce_loss = nn.CrossEntropyLoss(reduction='none')
            action_label = torch.argmax(true_actions_1hot[0], dim=1)
            loss_agent_act0 = ce_loss(pred_action_prev_logits,
                                      action_label)

            # Combine both auxiliary initialisation losses
            loss_agent_aux_init = loss_agent_h0 + loss_agent_act0
        else:
            loss_agent_aux_init = 0.

        # Finished getting initializing vectors.

        # Next, encode all the images to get the embeddings for the priors
        if imagine: # i.e. no need to calc loss therefore no need to have im_labels
            embeds = [None] * self.num_sim_steps
        else:
            embeds = self.conv_in(im_labels)

        priors = []
        posts = []
        pred_actions_1hot = []
        pred_action_log_probs = []
        pred_values = []
        pred_ims = []
        pred_rews = []
        pred_terminals = []
        states_env_h = []
        samples = []
        agent_hs = []
        recon_losses = []
        metrics_list = []
        tensors_list = []

        for i in range(self.num_sim_steps):
            # Define the labels for the loss function because we calculate it
            #  in here.
            if calc_loss:
                labels = {'image': im_labels[i],  #TODO make these keys consistent with the rest of the codebase, i.e. use 'ims', 'hx',
                          'reward':reward_labels[i],
                          'terminal':terminal_labels[i],
                          'before_terminal':before_terminal_labels[i],
                          'agent_h':agent_h_labels[i+1]}  #  +1 because ag_h_{t-1} is input to stepper and to agent, but it outputs ag_h_t(hat). We want the label to be ag_h_t.
            else:
                labels = None
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
                                            calc_loss=calc_loss,
                                            modal_sampling=modal_sampling,
                                            labels=labels)
            posts.append(post)
            pred_actions_1hot.append(pred_action_1hot)
            pred_action_log_probs.append(pred_action_log_prob)
            pred_values.append(pred_value)
            pred_ims.append(image_rec)
            pred_rews.append(rew_rec)
            pred_terminals.append(terminal_rec)
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
        pred_actions_1hot = torch.stack(pred_actions_1hot)
        pred_action_log_probs = torch.stack(pred_action_log_probs)
        pred_values = torch.stack(pred_values)
        pred_ims = torch.stack(pred_ims)
        pred_rews = torch.stack(pred_rews).squeeze()
        pred_terminals = torch.stack(pred_terminals).squeeze()
        states_env_h = torch.stack(states_env_h)    # (T,B,D)
        samples = torch.stack(samples)              # (T,B,S)
        agent_hs = torch.stack(agent_hs)
        priors = self.agent_env_stepper.batch_prior(states_env_h)  # (T,B,2S)
        features = torch.cat([states_env_h, samples], dim=-1)   # (T,B,D+S)
        env_states = (states_env_h, samples)

        if calc_loss:
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
            # are these all the same shape??
            loss_model = self.kl_weight * loss_kl + recon_losses
        else:
            loss_model = 0.

        # Make preds_dict
        preds_dict = {'action': pred_actions_1hot,
                      'act_log_prob': pred_action_log_probs,
                      'value': pred_values, # TODO go through whole codebase making consistent 'rec' vs 'pred'. Think I prefer rec because there's less confusion with 'prev'
                      'ims': pred_ims, # TODO go through whole codebase converting from 'obs' 'ob' 'ims' to 'im', for consistency with other labels
                      'hx': agent_hs,
                      'reward': pred_rews,
                      'terminal': pred_terminals,
                      'latent_vec': latent_vec,
                      'env_h': states_env_h}

        return (
            loss_model,
            loss_agent_aux_init,
            priors,                      # tensor(T,B,2S)
            posts,                       # tensor(T,B,2S)
            samples,                     # tensor(T,B,S)
            features,                    # tensor(T,B,D+S)
            env_states,
            (env_h.detach(), env_z.detach()),
            metrics_list,
            tensors_list,
            preds_dict,
        )


class AgentEnvStepper(nn.Module):
    """

    """
    def __init__(self,
                 hyperparams,
                 agent):
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
                calc_loss,
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
        if calc_loss:
            labels = {k: BF_to_TBF(v) for k, v in labels.items()}
            loss_reconstr, metrics, tensors, image_rec, rew_rec, terminal_rec = \
                self.decoder.training_step(feature, labels)
        else:
            labels = None
            loss_reconstr, metrics, tensors, image_rec, rew_rec, terminal_rec = \
                self.decoder.inference_step(feature)

        # Then use ims and agent_h to step the agent forward and produce an action
        image_rec = image_rec.squeeze()
        image_rec = dclamp(image_rec, self.image_range_min, self.image_range_max)
        no_masks = torch.zeros(1, image_rec.shape[0], device=self.device)  #(T,B)
        pred_action, pred_action_logits, pred_value, agent_h = \
            self.agent.predict_STE(image_rec, agent_h_prev, no_masks,
                                   retain_grads=True)
        if calc_loss:
            loss_reconstr_agent_h = self.agent_hx_loss(agent_h, labels['agent_h'], labels['before_terminal'])
            loss_reconstr = loss_reconstr + loss_reconstr_agent_h

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

    def agent_hx_loss(self,
                      pred_agent_h,
                      label_agent_h,
                      before_terminals):
        """Based on jurgisp's 'vecobs_decoder'. To be honest I don't understand
         why the std is pre-specified like this. But it's unlikely to be hugely
         important; the agent_h_loss is auxiliary."""
        std = 0.3989422804
        var = std ** 2 # var cancels denominator, which makes loss = 0.5 (target-output)^2
        p = D.Normal(loc=pred_agent_h, scale=torch.ones_like(pred_agent_h) * std)
        p = D.independent.Independent(p, 1)  # Makes p.logprob() sum over last dim

        loss = -p.log_prob(label_agent_h) * var
        loss = loss * before_terminals
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


class VAEEncoder(nn.Module):
    """

    Recurrent network that takes a seq of frames from t=-k to t=0 as input.
    The final output gets passed along with the final agent hidden state into
    an FC network. It outputs the initial state of the environment simulator.

    """
    def __init__(self,
                 hyperparams,
                 device):
        super(VAEEncoder, self).__init__()
        self.device = device
        self.num_init_steps = hyperparams.num_init_steps
        self.rnn_hidden_size = hyperparams.initializer_rnn_hidden_size
        self.env_dim = hyperparams.deter_dim
        self.env_h_stoch_size = hyperparams.env_h_stoch_size
        self.latent_vec_size = hyperparams.latent_vec_size

        # self.image_embedder = LayeredResBlockDown(input_hw=64,
        #                                           input_ch=3,
        #                                           hidden_ch=64,
        #                                           output_hw=8,
        #                                           output_ch=32)
        self.image_embedder = ConvEncoder(cnn_depth=32, in_channels=3)
        embedder_outsize = self.image_embedder.out_dim
        self.rnn = nn.GRU(input_size=embedder_outsize,
                          hidden_size=self.rnn_hidden_size,
                          num_layers=1,
                          batch_first=False)

        self.mlp_out = nn.Sequential(
                                nn.Linear(self.rnn_hidden_size,
                                          self.rnn_hidden_size),
                                nn.ELU(),
                                nn.Linear(self.rnn_hidden_size,
                                          self.rnn_hidden_size),
                                nn.ELU())

        self.mu_mlp = nn.Linear(self.rnn_hidden_size,
                                self.latent_vec_size)
        self.logvar_mlp = nn.Linear(self.rnn_hidden_size,
                                self.latent_vec_size)



    def forward(self,
                init_ims):
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
        # embeddings = [im for (im, _) in embeddings]
        x = torch.stack(embeddings, dim=0)  # stack along time dim

        # Flatten conv outputs to size (H*W*CH) to get rnn input vecs
        x = x.view(ts, batches,  -1)

        # Pass seq of vecs to initializer RNN
        x, _ = self.rnn(x)

        # Concat RNN output to agent h0 and then pass to Converter nets
        # to get mu_g and sigma_g
        x = x[-1]  # get last ts
        pre_vec = self.mlp_out(x)
        latent_vec = self.mu_mlp(pre_vec)
        norm = torch.norm(latent_vec, dim=1)
        norm_safe = torch.clip(norm, min=1e-8)
        latent_vec = latent_vec / norm_safe.unsqueeze(dim=1)
        # logvar =  self.logvar_mlp(pre_vec)
        # sigma = torch.exp(0.5 * logvar)
        # sample = mu #+ (sigma * torch.randn_like(mu, device=self.device))
        return latent_vec # sample, mu, logvar

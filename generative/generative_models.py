from .rssm.encoders import *
from .rssm.decoders import *
from .rssm.functions import dclamp, insert_dim, terminal_labels_to_mask, safe_normalize
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
        self.bottleneck_loss_weight  = hyperparams.bottleneck_loss_weight
        self.action_space_size = agent.env.action_space.n
        self.env_h_stoch_size = hyperparams.env_h_stoch_size
        self.bottleneck_vec_size = hyperparams.bottleneck_vec_size
        self.env_update_penalty_weight = hyperparams.env_update_penalty_weight

        # Networks
        self.conv_in = MultiEncoder(cnn_depth=32, image_channels=3)
        hyperparams.embed_dim = self.conv_in.out_dim
        self.encoder = AEEncoder(hyperparams, device)
        self.bottleneck_vec_converter_env = NLayerPerceptron(
            [hyperparams.bottleneck_vec_size,
             int((hyperparams.bottleneck_vec_size + hyperparams.deter_dim + hyperparams.env_h_stoch_size) / 2),
             int((hyperparams.bottleneck_vec_size + hyperparams.deter_dim + hyperparams.env_h_stoch_size) / 2),
             hyperparams.deter_dim + hyperparams.env_h_stoch_size],
        )
        self.bottleneck_vec_converter_agent_h = NLayerPerceptron(
            [hyperparams.bottleneck_vec_size,
             int((hyperparams.bottleneck_vec_size + hyperparams.agent_hidden_size)/2),
             int((hyperparams.bottleneck_vec_size + hyperparams.agent_hidden_size) / 2),
             hyperparams.agent_hidden_size],
        ) # Note that this net does not influence the representations learned
        #  by the AE latent vec because the AE sample is detached before
        # passing to this network. Same for bottleneck_vec_converter_action
        self.bottleneck_vec_converter_action = NLayerPerceptron(
            [hyperparams.bottleneck_vec_size,
             int((hyperparams.bottleneck_vec_size + hyperparams.action_space_size)/2),
             int((hyperparams.bottleneck_vec_size + hyperparams.action_space_size) / 2),
             hyperparams.action_space_size],
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
                calc_loss=False,
                modal_sampling=False,
                retain_grads=True,
                swap_directions=None):

        init_ims = data['ims'][0:self.num_init_steps]
        bottleneck_vec = self.encoder(init_ims)

        B = init_ims.shape[1]

        # Get true agent h0 and true actions as aux vars for decoder
        true_agent_h0 = data['hx'][self.num_init_steps - 1]  # -1 because 1st sim step is last init step
        true_agent_h0.requires_grad_()
        # N.B. true_agent_h0 may not actually be used for inference but is
        # needed for loss calculation.

        if use_true_actions:
            true_actions_inds = data['action'][self.num_init_steps-2:] # -2 because we have to use a_{t-1} in combo with env_t to get o_t
            true_actions_1hot = torch.nn.functional.one_hot(true_actions_inds.long(), self.action_space_size)
            true_actions_1hot = true_actions_1hot.float()
            true_actions_1hot.requires_grad_()
        else:
            true_actions_1hot = torch.zeros(self.num_sim_steps, B, self.action_space_size, device=self.device)

        (   loss_dict_no_grad,
            loss_model,
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
            unstacked_preds_dict,
        ) = self.ae_decode(bottleneck_vec,
                             data,
                             true_actions_1hot=true_actions_1hot,
                             use_true_actions=use_true_actions,
                             true_agent_h0=true_agent_h0,
                             use_true_agent_h0=use_true_agent_h0,
                             imagine=imagine,
                             calc_loss=calc_loss,
                             modal_sampling=modal_sampling,
                             retain_grads=True,
                             env_grads=True,)

        if calc_loss:
            # # Loss for autoencoder bottleneck
            # # pushes each vec away from others for best spread over hypersphere
            # similarities = 1-torch.mm(bottleneck_vec, bottleneck_vec.transpose(0,1))
            # eye = torch.eye(similarities.shape[0]).to(self.device)
            # not_eye = -(eye-1)
            # not_eye = not_eye.byte()
            # nonauto_sims = torch.where(not_eye, similarities, eye)
            # loss_bottleneck = - torch.log(1./torch.sum(torch.exp(nonauto_sims)))
            # loss_bottleneck *= self.bottleneck_loss_weight
            sq_dists = 2 * torch.mm(bottleneck_vec, bottleneck_vec.transpose(0,1)) - 2
            eye = torch.eye(sq_dists.shape[0]).to(self.device)
            not_eye = -(eye-1)
            not_eye = not_eye.byte()
            nonauto_sq_dists = torch.where(not_eye, sq_dists, eye)
            loss_bottleneck = torch.log(torch.sum(torch.exp(nonauto_sq_dists)))
            loss_bottleneck *= self.bottleneck_loss_weight
            loss_dict_no_grad['loss_bottleneck'] = loss_bottleneck.item()

        else:
            loss_bottleneck = 0.
            loss_dict_no_grad['loss_bottleneck'] = loss_bottleneck


        return (
            loss_dict_no_grad,
            loss_model,
            loss_bottleneck,
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
            unstacked_preds_dict,
        )

    def ae_decode(self,
                bottleneck_vec,
                data,
                true_actions_1hot=None,
                use_true_actions=True,
                true_agent_h0=None,
                use_true_agent_h0=True,
                imagine=False,
                calc_loss=True,
                modal_sampling=False,
                retain_grads=True,
                env_grads=True,
                ):
        """
        imagine: Whether or not to use the generated images as input to
        the env model or whether to use true images (true images will be used
        during training).

        """

        B = bottleneck_vec.shape[0]
        loss_dict_no_grad = {}

        # Get labels for loss function
        if calc_loss:  # No need to calc loss
            agent_h_labels = data['hx'][self.num_init_steps-1:]
            reward_labels = data['reward'][self.num_init_steps-1:]
            terminal_labels = data['terminal'][self.num_init_steps-1:]
            before_terminal_labels = terminal_labels_to_mask(terminal_labels)
            im_labels = data['ims'][-self.num_sim_steps:]

        # Use bottleneck_vec to get init vectors: env_h_prev, env_z_prev,
        #  agent_h_prev, and action_prev
        ## First env_h_prev, env_z_prev (env_z_prev uses Straight Through
        ##  Gradients because it is a random sample)

        env_prev, _ = self.bottleneck_vec_converter_env(bottleneck_vec)
        env_h_prev, env_z_prev = env_prev[:,:-self.env_h_stoch_size],\
                                 env_prev[:,-self.env_h_stoch_size:]
        init_z_dist = self.agent_env_stepper.zdistr(env_z_prev)

        if modal_sampling:
            ### Modal sampling
            inds = init_z_dist.mean.argmax(dim=2)
            mode_one_hot = torch.nn.functional.one_hot(inds,
                           num_classes=self.agent_env_stepper.stoch_discrete).to(
                           self.device)
            env_z_prev = init_z_dist.mean + \
                     (mode_one_hot - init_z_dist.mean).detach()
            env_z_prev = env_z_prev.reshape(B, -1)
        else:
            ### Random sampling
            env_z_prev = init_z_dist.rsample().reshape(B, -1)


        ## Second, agent_h_prev
        pred_agent_h_prev, _ = self.bottleneck_vec_converter_agent_h(bottleneck_vec.detach()) # So that the sample vec is only learns to produce good env states, not contain any representations specific to an agent hx.
        pred_agent_h_prev = torch.tanh(pred_agent_h_prev)
        if use_true_agent_h0:
            agent_h_prev = true_agent_h0
        else:
            agent_h_prev = pred_agent_h_prev

        ## Third, action_prev
        # Originally there was no straight through grads here because the agent's init
        ##  vectors should be trained independently from the rest of the model.
        ## But we're actually still training it separately from the model, but
        ## in fact we do need STE for causal consistency when updating the
        ## bottleneck vector. No need to retrain the whole model though.
        pred_action_prev_logits, _ = self.bottleneck_vec_converter_action(bottleneck_vec.detach())
        pred_action_prev_probs = torch.softmax(pred_action_prev_logits, dim=1)
        pred_action_prev_inds = pred_action_prev_probs.argmax(dim=1)
        pred_action_prev_1hot = torch.nn.functional.one_hot(pred_action_prev_inds,
                                                            num_classes=self.action_space_size).to(self.device).float()
        pred_action_prev_1hot = pred_action_prev_probs + \
                 (pred_action_prev_1hot - pred_action_prev_probs).detach()

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

            # prepare for calc of env_h update penalty
            env_update_losses = []

            # log
            loss_dict_no_grad['loss_agent_aux_init'] = torch.mean(
                loss_agent_aux_init).item()

        else:
            loss_agent_aux_init = 0.
            loss_dict_no_grad['loss_agent_aux_init'] = loss_agent_aux_init

        # Finished getting initializing vectors.

        # Next, encode all the images to get the embeddings for the priors
        if not calc_loss: #TODO remove this comment for its hubris # i.e. no need to calc loss therefore no need to have im_labels
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
        loss_reconstr_ims = []
        loss_reconstr_agent_hs = []
        metrics_list = []
        tensors_list = []

        if retain_grads:
            action_prev.retain_grad()
            agent_h_prev.retain_grad()
            env_h_prev.retain_grad()

        if not env_grads:
            action_prev = action_prev.detach()

        for i in range(self.num_sim_steps):
            # Define the labels for the loss function because we calculate it
            #  in here.
            if calc_loss:
                labels = {'ims': im_labels[i],
                          'reward':reward_labels[i],
                          'terminal':terminal_labels[i],
                          'before_terminal':before_terminal_labels[i],
                          'agent_h':agent_h_labels[i+1]}  #  +1 because ag_h_{t-1} is input to stepper and to agent, but it outputs ag_h_t(hat). We want the label to be ag_h_t.
            else:
                labels = None

            if not env_grads:
                action_prev = action_prev.detach()

            embed = embeds[i]

            (post,    # tensor(B, 2*S) # TODO currently only posts if imagine = False. But when imagine = True then this is the prior.
            prior,    # tensor(B, 2*S)
            pred_action_1hot,
            pred_action_log_prob,
            pred_value,
            pred_image, # _pred for predicted, _rec for reconstructed, i.e. basically the same thing.
            pred_rew,
            pred_terminal,
            (env_h, env_z),      # tensor(B, D+S+G)
            agent_h,
            loss_reconstr,
            loss_reconstr_im,
            loss_reconstr_agent_h,
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

            if retain_grads:
                pred_image.retain_grad()
                agent_h.retain_grad()
                pred_action_1hot.retain_grad()
                pred_action_log_prob.retain_grad()
                env_h.retain_grad()

            priors.append(prior)
            posts.append(post)
            pred_actions_1hot.append(pred_action_1hot)
            pred_action_log_probs.append(pred_action_log_prob)
            pred_values.append(pred_value)
            pred_ims.append(pred_image)
            pred_rews.append(pred_rew)
            pred_terminals.append(pred_terminal)
            states_env_h.append(env_h)
            samples.append(env_z)
            agent_hs.append(agent_h)
            recon_losses.append(loss_reconstr)
            loss_reconstr_ims.append(loss_reconstr_im)
            loss_reconstr_agent_hs.append(loss_reconstr_agent_h)


            metrics_list.append(metrics)
            tensors_list.append(tensors)

            if use_true_actions:
                action_prev = true_actions_1hot[i + 1]  # +1 because index0 is a_{t-1}
            else:
                action_prev = pred_action_1hot

            if calc_loss:
                diff = env_h - env_h_prev
                env_update_loss = torch.norm(diff, dim=1) # no norm on batch dim
                env_update_losses.append(env_update_loss)
            agent_h_prev = agent_h
            env_h_prev, env_z_prev = (env_h, env_z)

        # We keep an unstacked dict because for some reason when you retain
        # grads on the stacked tensors it doesn't actually retain the grads
        unstacked_preds_dict = {'action': pred_actions_1hot,
                                'act_log_prob': pred_action_log_probs,
                                'value': pred_values,
                                'ims': pred_ims,
                                'hx': agent_hs,
                                'reward': pred_rews,
                                'terminal': pred_terminals,
                                'bottleneck_vec': bottleneck_vec,
                                'env_h': states_env_h}

        posts = torch.stack(posts)                  # (T,B,2S)
        priors = torch.stack(priors)                  # (T,B,2S)
        pred_actions_1hot = torch.stack(pred_actions_1hot)
        pred_action_log_probs = torch.stack(pred_action_log_probs)
        pred_values = torch.stack(pred_values)
        pred_ims = torch.stack(pred_ims)
        pred_rews = torch.stack(pred_rews).squeeze()
        pred_terminals = torch.stack(pred_terminals).squeeze()
        states_env_h = torch.stack(states_env_h)    # (T,B,D)
        samples = torch.stack(samples)              # (T,B,S)
        agent_hs = torch.stack(agent_hs)
        # priors = self.agent_env_stepper.batch_prior(states_env_h)  # (T,B,2S) # Now doing this within stepper so that we can calculate the kl_rssm loss, whether imagine=True OR False.
        features = torch.cat([states_env_h, samples], dim=-1)   # (T,B,D+S)
        env_states = (states_env_h, samples)

        if calc_loss:
            recon_losses = torch.stack(recon_losses).squeeze()
            loss_reconstr_ims = torch.stack(loss_reconstr_ims).squeeze()
            loss_reconstr_agent_hs = torch.stack(loss_reconstr_agent_hs).squeeze()
            env_update_losses = torch.stack(env_update_losses).squeeze()

            # KL loss
            d = self.agent_env_stepper.zdistr
            dprior = d(priors)
            dpost = d(posts)
            loss_kl_exact = D.kl.kl_divergence(dpost, dprior)  # (T,B)

            # Analytic KL loss, standard for AE
            if not self.kl_balance:
                loss_kl = loss_kl_exact
            else:
                loss_kl_postgrad = D.kl.kl_divergence(dpost, d(priors.detach()))
                loss_kl_priograd = D.kl.kl_divergence(d(posts.detach()), dprior)
                loss_kl = (1 - self.kl_balance) * loss_kl_postgrad + \
                          self.kl_balance       * loss_kl_priograd


            # Total loss
            assert loss_kl.shape == recon_losses.shape
            loss_model = self.kl_weight * loss_kl + \
                         recon_losses + \
                         self.env_update_penalty_weight * env_update_losses

            loss_dict_no_grad['loss_reconstruction'] = recon_losses
            loss_dict_no_grad['loss_kl_rssm'] = loss_kl * self.kl_weight
            loss_dict_no_grad['loss_env_update'] = env_update_losses * self.env_update_penalty_weight
            loss_dict_no_grad['loss_reconstr_ims'] = loss_reconstr_ims
            loss_dict_no_grad['loss_reconstr_agent_hs'] = loss_reconstr_agent_hs
        else:
            loss_model = 0.
            loss_dict_no_grad['loss_reconstruction'] = 0.
            loss_dict_no_grad['loss_kl_rssm'] = 0.
            loss_dict_no_grad['loss_env_update'] = 0.
            loss_dict_no_grad['loss_reconstr_ims'] = 0.
            loss_dict_no_grad['loss_reconstr_agent_hs'] = 0.


        # Make preds_dict
        preds_dict = {'action': pred_actions_1hot,
                      'act_log_prob': pred_action_log_probs,
                      'value': pred_values,
                      'ims': pred_ims,
                      'hx': agent_hs,
                      'reward': pred_rews,
                      'terminal': pred_terminals,
                      'bottleneck_vec': bottleneck_vec,
                      'env_h': states_env_h}

        return (
            loss_dict_no_grad,
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
            unstacked_preds_dict,
        )

class GradientDestroyer(torch.autograd.Function):
    """
    The forward pass leaves the input unchanged, but the backward pass
    turns all gradients to 0.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # ctx.mark_non_differentiable(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        _, = ctx.saved_tensors
        grad_input = torch.zeros_like(grad_output)
        return grad_input


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
        self.a_mlp = nn.Linear(hyperparams.action_space_size, hyperparams.hidden_dim, bias=False)  # No bias, because outputs are added
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

    def get_prior(self, h):
        x = self.prior_mlp_h(h)
        x = self.prior_norm(x)
        x = F.elu(x)
        prior = self.prior_mlp(x)  # (B,2S)
        return prior

    def get_post(self, h, embed):
        x = self.post_mlp_h(h) + self.post_mlp_e(embed)
        x = self.post_norm(x)
        post_in = F.elu(x)
        post = self.post_mlp(post_in)  # (B, S*S)
        return post

    def sample_from_distr(self, distr, batch_size, modal_sampling=False):
        if modal_sampling:
            # Uses Straight Through Gradients
            inds = distr.mean.argmax(dim=2)
            mode_one_hot = torch.nn.functional.one_hot(inds,
                                                       num_classes=self.stoch_discrete).to(
                self.agent.device)
            sample = distr.mean + \
                     (mode_one_hot - distr.mean).detach()
            sample = sample.reshape(batch_size, -1)
        else:
            sample = distr.rsample().reshape(batch_size, -1)
        return sample

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

        # Summary of the below 2-level if tree: If we calc loss, then we always
        # need to calculate both prior and post, but what we sample from is
        # determined by whether or not we're imagining.
        # OTOH, if we're not calcing loss, then we only need the sample from the
        # distribution we care about, which is determined by whether or not
        # we're imagining.

        if calc_loss:
            prior = self.get_prior(h)
            prior_distr = self.zdistr(prior)
            post = self.get_post(h, embed)
            post_distr = self.zdistr(post)
            if imagine:
                sample = self.sample_from_distr(prior_distr, B,
                                                modal_sampling=modal_sampling)
            else:
                sample = self.sample_from_distr(post_distr, B,
                                                modal_sampling=False)
        else:
            if imagine:
                prior = self.get_prior(h)
                prior_distr = self.zdistr(prior)
                post = torch.tensor(float('nan'))
                sample = self.sample_from_distr(prior_distr, B,
                                                modal_sampling=modal_sampling)
            else:
                prior = torch.tensor(float('nan'))
                post = self.get_post(h, embed)
                post_distr = self.zdistr(post)
                sample = self.sample_from_distr(post_distr, B,
                                                modal_sampling=False)

        feature = torch.cat([h, sample], dim=-1)
        BF_to_TBIF = lambda x: torch.unsqueeze(torch.unsqueeze(x, 1), 0)
        BF_to_TBF = lambda x: torch.unsqueeze(x, 0)
        feature = BF_to_TBIF(feature)
        if calc_loss:
            labels = {k: BF_to_TBF(v) for k, v in labels.items()}
            loss_reconstr_im, metrics, tensors, pred_image, pred_rew, pred_terminal = \
                self.decoder.training_step(feature, labels)
        else:
            labels = None
            loss_reconstr_im, metrics, tensors, pred_image, pred_rew, pred_terminal = \
                self.decoder.inference_step(feature)

        # Then use ims and agent_h to step the agent forward and produce an action
        pred_image = pred_image.squeeze()
        pred_image = dclamp(pred_image, self.image_range_min, self.image_range_max)
        no_masks = torch.zeros(1, pred_image.shape[0], device=self.device)  #(T,B)
        pred_action, pred_action_logits, pred_value, agent_h = \
            self.agent.predict_STE(pred_image, agent_h_prev, no_masks,
                                   retain_grads=True)
        if calc_loss:
            loss_reconstr_agent_h = self.agent_hx_loss(agent_h, labels['agent_h'], labels['before_terminal'])
            loss_reconstr = loss_reconstr_im + loss_reconstr_agent_h
        else:
            loss_reconstr = 0.
            loss_reconstr_agent_h = 0.

        pred_action.retain_grad()
        pred_action_logits.retain_grad()
        pred_value.retain_grad()
        pred_image.retain_grad()
        agent_h.retain_grad()

        return (
            post,     # tensor(B, 2*S)
            prior,    # tensor(B, 2*S)
            pred_action,
            pred_action_logits,
            pred_value,
            pred_image, # _pred for predicted, _rec for reconstructed, i.e. basically the same thing.
            pred_rew,
            pred_terminal,
            (h, sample),      # tensor(B, D+S+G)
            agent_h,
            loss_reconstr,
            loss_reconstr_im,
            loss_reconstr_agent_h,
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


class AEEncoder(nn.Module):
    """

    Recurrent network that takes a seq of frames from t=-k to t=0 as input.
    The final output gets passed along with the final agent hidden state into
    an FC network. It outputs the initial state of the environment simulator.

    """
    def __init__(self,
                 hyperparams,
                 device):
        super(AEEncoder, self).__init__()
        self.device = device
        self.num_init_steps = hyperparams.num_init_steps
        self.rnn_hidden_size = hyperparams.initializer_rnn_hidden_size
        self.env_dim = hyperparams.deter_dim
        self.env_h_stoch_size = hyperparams.env_h_stoch_size
        self.bottleneck_vec_size = hyperparams.bottleneck_vec_size

        # self.image_embedder = LayeredResBlockDown(input_hw=64,
        #                                           input_ch=3,
        #                                           hidden_ch=64,
        #                                           output_hw=8,
        #                                           output_ch=32)

        self.image_embedder = LayeredResBlockDown(input_hw=64,
                                                  input_ch=3,
                                                  hidden_ch=64,
                                                  output_hw=4,
                                                  output_ch=128)

        self.rnn = nn.GRU(input_size=self.image_embedder.output_size,
                           hidden_size=self.rnn_hidden_size,
                           num_layers=1,
                           batch_first=False)


        # self.image_embedder = ConvEncoder(cnn_depth=32, in_channels=3)
        # embedder_outsize = self.image_embedder.out_dim
        # self.rnn = nn.GRU(input_size=embedder_outsize,
        #                   hidden_size=self.rnn_hidden_size,
        #                   num_layers=1,
        #                   batch_first=False)

        self.mlp_out = nn.Sequential(
                                nn.Linear(self.rnn_hidden_size,
                                          self.rnn_hidden_size),
                                nn.ELU(),
                                nn.Linear(self.rnn_hidden_size,
                                          self.rnn_hidden_size),
                                nn.ELU())

        self.bottleneck_mlp = nn.Linear(self.rnn_hidden_size,
                                self.bottleneck_vec_size)


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
        embeddings = [im for (im, _) in embeddings]
        x = torch.stack(embeddings, dim=0)  # stack along time dim

        # Flatten conv outputs to size (H*W*CH) to get rnn input vecs
        x = x.view(ts, batches,  -1)

        # Pass seq of vecs to initializer RNN
        x, _ = self.rnn(x)

        # Concat RNN output to agent h0 and then pass to Converter nets
        # to get bottneck vec
        x = x[-1]  # get last ts
        pre_vec = self.mlp_out(x)
        bottleneck_vec = self.bottleneck_mlp(pre_vec)
        bottleneck_vec = safe_normalize(bottleneck_vec)

        return bottleneck_vec

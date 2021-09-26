import torch
import torch.nn.functional as F


def compute_baseline_loss(advantages):
    return 0.5 * torch.mean(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.mean(policy * log_policy)


def compute_ppo_loss(logits, old_logits, actions, advantages, clip_par=0.15):
    logp = -F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    old_logp = -F.nll_loss(
        F.log_softmax(torch.flatten(old_logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    log_ratio = logp - old_logp.detach()
    ratio = log_ratio.exp()  # .clamp(1e-3, 1000)
    adv_targ = advantages
    adv_targ = (adv_targ - adv_targ.mean()) / (adv_targ.std() + 1e-8)
    adv_targ = torch.flatten(adv_targ, 0, 1).detach()
    surr1 = ratio * adv_targ
    surr2 = torch.clamp(ratio, 1.0 - clip_par,
                        1.0 + clip_par) * adv_targ
    return -torch.min(surr1, surr2).mean()


def compute_policy_gradient_loss(logits, actions, advantages):
    # print(torch.flatten(logits, 0, 1).shape,torch.flatten(actions, 0, 1).shape)
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.mean(cross_entropy * advantages.detach())

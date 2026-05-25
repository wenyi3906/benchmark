from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome import repr_tools
from ...util import nethook

from .emmet_hparams import EMMETHyperParams


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: EMMETHyperParams,
    layer: int,
    context_templates: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters
    if "neo" in model.config._name_or_path or "gpt2" in model.config._name_or_path:
        ln_f = nethook.get_module(model, hparams.ln_f_module)
        lm_head_module = nethook.get_module(model, hparams.lm_head_module)
        lm_w = nethook.get_parameter(lm_head_module, "weight").T
    else:
        lm_w, ln_f = (
            nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
            nethook.get_module(model, hparams.ln_f_module),
        )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids = tok.encode(request["target_new"], return_tensors="pt", add_special_tokens=False).to(f"cuda:{hparams.device}")[0]


    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1]) 
        for context_types in context_templates
        for context in context_types
    ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to(f"cuda:{hparams.device}")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device=f"cuda:{hparams.device}").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    if hasattr(model.config, 'n_embd'):
        delta = torch.zeros((model.config.n_embd,), requires_grad=True, device=f"cuda:{hparams.device}", dtype=model.dtype)
    elif hasattr(model.config, 'hidden_size'):
        delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device=f"cuda:{hparams.device}", dtype=model.dtype)
    else:
        raise NotImplementedError
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.layer_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                if type(cur_out) is tuple:
                    target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()
                else:
                    target_init = cur_out[0, lookup_idxs[0]].detach().clone()

            # 完美兼容 PyTorch 计算图的加法操作 (保证反向传播梯度 100% 成功流回 delta)
            base_tensor = cur_out[0] if type(cur_out) is tuple else cur_out
            
            # 建立掩码矩阵
            mask = torch.zeros(base_tensor.shape[:2], device=base_tensor.device, dtype=base_tensor.dtype)
            for i, idx in enumerate(lookup_idxs):
                mask[i, idx] = 1.0
            mask = mask.unsqueeze(-1)  # shape: (batch, seq, 1)
            
            # 通过自动广播机制相乘，安全建立计算图
            delta_expanded = mask * delta
            
            # 纯张量加法，完美反传
            new_tensor = base_tensor + delta_expanded

            if type(cur_out) is tuple:
                return (new_tensor,) + cur_out[1:]
            else:
                return new_tensor

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)
    nll_loss_factor = hparams.nll_loss_factor if hasattr(hparams, 'nll_loss_factor') else 1.0

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        out_repr = tr[hparams.layer_module_tmp.format(loss_layer)].output
        if type(out_repr) is tuple:
            full_repr = out_repr[0][: len(rewriting_prompts)]
        else:
            full_repr = out_repr[: len(rewriting_prompts)]

        logits = ln_f(full_repr) @ lm_w + lm_b
        log_probs = torch.log_softmax(logits, dim=-1)

        targets = rewriting_targets
        while targets.dim() < log_probs.dim() - 1:
            targets = targets.unsqueeze(0)
        if targets.size(0) != log_probs.size(0) and targets.size(0) == 1:
            targets = targets.expand(log_probs.size(0), *targets.shape[1:])

        loss = torch.gather(
            log_probs,
            -1,
            torch.where(targets != -100, targets, 0).unsqueeze(-1),
        ).squeeze(-1)
        mask = (targets != -100).float()
        max_probs = torch.max(log_probs, dim = 2)[0]
        max_prob = torch.exp((max_probs * mask).sum(1) / target_ids.size(0)).mean().item()
        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_factor * nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + weight_decay
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']}] "
            f"{np.round(max_prob, 4)}"
        )
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta

    return target


def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
    track: str = "both"
):
    """
    Retrieves detached representations for a list of words at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
        track=track,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        subtoken = fact_token_strategy[len("subject_") :]
        res = repr_tools.get_reprs_at_word_tokens(
            context_templates=context_templates,
            words=words,
            subtoken=subtoken,
            **word_repr_args,
        )
    elif fact_token_strategy == "last":
        res = repr_tools.get_reprs_at_idxs(
            contexts=[context.format(word) for context, word in zip(context_templates, words)],
            idxs=[[-1] for _ in words],
            **word_repr_args,
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    if track == "both":
        l_input, l_output = res
        if isinstance(l_input, tuple): l_input = l_input[0]
        if isinstance(l_output, tuple): l_output = l_output[0]
        return l_input.detach(), l_output.detach()
    else:
        l_res = res
        if isinstance(l_res, tuple): l_res = l_res[0]
        return l_res.detach()

def get_modules_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_templates: List[str],
    fact_token_strategy: str,
) -> Tuple[torch.Tensor, ...]:
    res = []
    for module_template in module_templates:
        # Default track='both' when fetching for multiple modules
        l_input, l_output = get_module_input_output_at_words(
            model,
            tok,
            layer,
            context_templates,
            words,
            module_template,
            fact_token_strategy,
            track="both"
        )
        res.append((l_input, l_output))
    return tuple(r[0] for r in res)

def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = []
    sentence = prompt.format(subject)
    if fact_token_strategy == "last":
        raise Exception("This is definitely bugged, fix it.")
        ret = -1
    elif "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        word = subject
        subwords = tok(word).input_ids
        subwords = subwords[1:] if subwords[0] == tok.bos_token_id else subwords
        subtoken = fact_token_strategy[len("subject_") :]
        
        if subtoken == "first":
            subtoken = subwords[0]
        elif subtoken == "last":
            subtoken = subwords[-1]
        elif subtoken == "first_after_last":
            subtoken = subwords[-1]
        else:
            raise ValueError(f"Unknown fact_token_strategy: {fact_token_strategy}")
            
        str_subtoken = tok.decode(subtoken)
        
        # Method 1: Strict Token ID match
        ret = [i for i, x in enumerate(tok(sentence)["input_ids"]) if x == subtoken]
        
        # Method 2: Token sequence sublist match
        if len(ret) == 0:
            prompt_words = tok(sentence).input_ids
            word_words = tok(word).input_ids
            for i in range(len(prompt_words) - len(word_words) + 1):
                if prompt_words[i:i+len(word_words)] == word_words:
                    if subtoken == subwords[-1]:
                        ret = [i + len(word_words) - 1]
                    elif subtoken == subwords[0]:
                        ret = [i]
                    break
                    
        # Method 3 (ULTIMATE FALLBACK): Pure String Decoding Accumulation
        if len(ret) == 0:
            try:
                prefix = prompt.split("{}")[0]
                target_char_len = len(prefix) + len(word)
                input_ids = tok(sentence)["input_ids"]
                
                if fact_token_strategy.endswith("last") or fact_token_strategy == "subject_first_after_last":
                    for i in range(len(input_ids)):
                        decoded_so_far = tok.decode(input_ids[:i+1])
                        if len(decoded_so_far) >= target_char_len:
                            ret = [i]
                            break
                elif fact_token_strategy.endswith("first"):
                    target_char_len = len(prefix)
                    for i in range(len(input_ids)):
                        decoded_so_far = tok.decode(input_ids[:i+1])
                        if len(decoded_so_far) > target_char_len:
                            ret = [i]
                            break
            except Exception as e:
                print(f"Fallback 3 failed: {e}")
                
        if len(ret) == 0:
            raise ValueError(f"Could not find subtoken {str_subtoken} in prompt {prompt} with subject {subject}")
            
        if fact_token_strategy == "subject_first_after_last":
            ret = [ret[0] + 1]
        ret = ret[0]
    else:
        raise ValueError(f"Unknown fact_token_strategy: {fact_token_strategy}")

    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret
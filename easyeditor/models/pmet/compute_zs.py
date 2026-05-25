from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome import repr_tools
from ...util import nethook

from .pmet_hparams import PMETHyperParams


def compute_zs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: PMETHyperParams,
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
    if "neo" in model.config._name_or_path or "llama" in model.config._name_or_path:
        delta_attn = torch.zeros((model.config.hidden_size,), requires_grad=True, device=f"cuda:{hparams.device}")
        delta_mlp = torch.zeros((model.config.hidden_size,), requires_grad=True, device=f"cuda:{hparams.device}")
    else:
        _dim = getattr(model.config, 'n_embd', getattr(model.config, 'hidden_size', None))
        delta_attn = torch.zeros((_dim,), requires_grad=True, device=f"cuda:{hparams.device}")
        delta_mlp = torch.zeros((_dim,), requires_grad=True, device=f"cuda:{hparams.device}")
    target_init_attn, target_init_mlp, kl_distr_init = None, None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init_attn, target_init_mlp

        if cur_layer == hparams.mlp_module_tmp.format(layer):
            if target_init_mlp is None:
                print("Recording initial value of v* in mlp")
                if type(cur_out) is tuple:
                    target_init_mlp = cur_out[0][0, lookup_idxs[0]].detach().clone()
                else:
                    target_init_mlp = cur_out[0, lookup_idxs[0]].detach().clone()

            if type(cur_out) is tuple:
                for i, idx in enumerate(lookup_idxs):
                    cur_out[0][i, idx, :] += delta_mlp
            else:
                for i, idx in enumerate(lookup_idxs):
                    cur_out[i, idx, :] += delta_mlp
        if cur_layer == hparams.attn_module_tmp.format(layer):
            if target_init_attn is None:
                print("Recording initial value of v* in attn")
                if type(cur_out) is tuple:
                    target_init_attn = cur_out[0][0, lookup_idxs[0]].detach().clone()
                else:
                    target_init_attn = cur_out[0, lookup_idxs[0]].detach().clone()

            if type(cur_out) is tuple:
                for i, idx in enumerate(lookup_idxs):
                    cur_out[0][i, idx, :] += delta_attn
            else:
                for i, idx in enumerate(lookup_idxs):
                    cur_out[i, idx, :] += delta_attn
        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta_mlp, delta_attn], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)
    nll_loss_factor = hparams.nll_loss_factor
    kl_factor = hparams.kl_factor
    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
                hparams.attn_module_tmp.format(layer),
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

        try:
            loss = torch.gather(
                log_probs,
                -1,
                torch.where(targets != -100, targets, 0).unsqueeze(-1),
            ).squeeze(-1)
        except Exception as e:
            print(f"Error! log_probs shape: {log_probs.shape}, targets shape: {targets.shape}, targets.unsqueeze(-1) shape: {torch.where(targets != -100, targets, 0).unsqueeze(-1).shape}")
            raise e
            
        mask = (targets != -100).float()
        max_probs = torch.max(log_probs, dim = 2)[0]
        max_prob = torch.exp((max_probs * mask).sum(1) / target_ids.size(0)).mean().item()
        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_factor * nll_loss_each.mean()
        kl_loss = kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta_mlp) / torch.norm(target_init_mlp) + torch.norm(delta_attn) / torch.norm(target_init_attn)
        )
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
        max_norm = hparams.clamp_norm_factor * target_init_mlp.norm()
        if delta_mlp.norm() > max_norm:
            with torch.no_grad():
                delta_mlp[...] = delta_mlp * max_norm / delta_mlp.norm()
        max_norm = hparams.clamp_norm_factor * target_init_attn.norm()
        if delta_attn.norm() > max_norm:
            with torch.no_grad():
                delta_attn[...] = delta_attn * max_norm / delta_attn.norm()

    target_mlp = target_init_mlp + delta_mlp
    target_attn = target_init_attn + delta_attn
    print(
        f"Init norm {target_init_mlp.norm()} | Delta norm {delta_mlp.norm()} | Target norm {target_mlp.norm()}"
    )
    print(
        f"Init norm {target_init_attn.norm()} | Delta norm {delta_attn.norm()} | Target norm {target_attn.norm()}"
    )

    return target_attn, target_mlp

# dummy alias if needed
compute_z = compute_zs

def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Retrieves detached representations for a list of words at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
        track="both",
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            context_templates=context_templates,
            words=words,
            subtoken=subtoken,
            **word_repr_args,
        )
    elif fact_token_strategy == "last":
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            contexts=[context.format(word) for context, word in zip(context_templates, words)],
            idxs=[[-1] for _ in words],
            **word_repr_args,
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    if isinstance(l_input, tuple): l_input = l_input[0]
    if isinstance(l_output, tuple): l_output = l_output[0]

    return l_input.detach(), l_output.detach()

def get_modules_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_templates: List[str],
    fact_token_strategy: str,
) -> Tuple[torch.Tensor, ...]:
    """
    Retrieves detached representations for a word at the input and
    output of multiple layer modules.
    """
    res = []
    for module_template in module_templates:
        l_input, l_output = get_module_input_output_at_words(
            model,
            tok,
            layer,
            context_templates,
            words,
            module_template,
            fact_token_strategy,
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
        # This solves ALL BPE tokenization inconsistencies (e.g. LLaMA prefix spaces)
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
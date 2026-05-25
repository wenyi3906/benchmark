import math
from typing import Dict, List, Tuple, Union
import copy

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome import repr_tools
from ...util import nethook

from .memit_hparams import MEMITHyperParams

def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: MEMITHyperParams,
    layer: int,
    context_templates: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the update.
    Unlike ROME, MEMIT computes the vector by optimizing a specific objective,
    which requires calculating the left vector (k) first.
    """

    # Get model parameters
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight"),
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    print("Computing right vector (v)")

    # Tokenize target into list of int
    target_ids = tok(request["target_new"], return_tensors="pt").to(f"cuda:{hparams.device}")[
        "input_ids"
    ][0]

    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]
    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context_type in context_templates
        for context in context_type
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

    if hasattr(model.config, 'n_embd'):
        delta = torch.zeros((model.config.n_embd,), requires_grad=True, device=f"cuda:{hparams.device}")
    elif hasattr(model.config, 'hidden_size'):
        delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device=f"cuda:{hparams.device}")
    else:
        raise NotImplementedError
    target_init, kl_distr_init = None, None

    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        def _unwrap_output(output):
            if isinstance(output, torch.Tensor):
                return output, None
            if isinstance(output, (list, tuple)):
                if len(output) == 0:
                    raise ValueError("Layer output container is empty.")
                # Check if the first element is a tuple (some models nest outputs)
                if isinstance(output[0], tuple) and len(output[0]) > 0 and isinstance(output[0][0], torch.Tensor):
                     return output[0][0], output
                return output[0], output
            if hasattr(output, 'items'):
                if 'hidden_states' in output and output['hidden_states'] is not None:
                    return output['hidden_states'], output
                for v in output.values():
                    if isinstance(v, torch.Tensor) and len(v.shape) == 3:
                        return v, output
                
                # It's possible that the output is a kwargs dictionary that got passed as kwargs 
                # to the hook (like from PyTorch forward hook kwargs capturing) and DOES NOT contain the actual output tensor.
                # In such rare cases, we return None and let the caller handle it.
                return None, output
                
            raise TypeError(f"Unsupported layer output type {type(output)} encountered in MEMIT.")

        def _rewrap_output(updated, original):
            if original is None:
                return updated
            if isinstance(original, list):
                new_out = list(original)
                if isinstance(new_out[0], tuple):
                    nested = list(new_out[0])
                    nested[0] = updated
                    new_out[0] = tuple(nested)
                else:
                    new_out[0] = updated
                return new_out
            elif isinstance(original, tuple):
                new_out = list(original)
                if isinstance(new_out[0], tuple):
                    nested = list(new_out[0])
                    nested[0] = updated
                    new_out[0] = tuple(nested)
                else:
                    new_out[0] = updated
                return tuple(new_out)
            elif hasattr(original, 'items'):
                # Mutate the dictionary-like object in-place to preserve its Exact Class and Behavior
                # This guarantees that if it is a HuggingFace ModelOutput, it retains its __getitem__ overrides
                # and won't throw KeyError when indexing into it.
                if 'hidden_states' in original:
                    original['hidden_states'] = updated
                else:
                    for k, v in original.items():
                        if isinstance(v, torch.Tensor) and len(v.shape) == 3:
                            original[k] = updated
                            break
                return original
            raise TypeError(f"Unsupported layer output container {type(original)} in MEMIT.")

        if cur_layer == hparams.layer_module_tmp.format(layer):
            layer_output, original_container = _unwrap_output(cur_out)

            # If layer_output is None, it means the hook captured something that wasn't the actual output
            # (e.g. kwargs dictionary containing only attention_masks). We should just ignore it and return the original.
            if layer_output is None:
                return cur_out

            if not isinstance(layer_output, torch.Tensor):
                raise RuntimeError(f"DEBUG INFO: layer_output is not a tensor! It is {type(layer_output)}. "
                                   f"cur_out type: {type(cur_out)}. "
                                   f"Are we hooking the correct module? Module output looks like: {cur_out}")

            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                target_init = layer_output[0, lookup_idxs[0]].detach().clone()

            # Add intervened delta
            # We must clone layer_output to avoid in-place modification issues for gradients
            new_layer_output = layer_output.clone()
            for i, idx in enumerate(lookup_idxs):
                new_layer_output[i, idx, :] += delta

            return _rewrap_output(new_layer_output, original_container)

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

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
            retain_input=[hparams.layer_module_tmp.format(loss_layer)],
            retain_output=[hparams.layer_module_tmp.format(loss_layer)] + [hparams.layer_module_tmp.format(layer)],
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
        full_repr = tr[hparams.layer_module_tmp.format(loss_layer)].output
        if hasattr(full_repr, 'items'):
             if 'hidden_states' in full_repr:
                 full_repr = full_repr['hidden_states']
             else:
                 for v in full_repr.values():
                     if isinstance(v, torch.Tensor) and len(v.shape) == 3:
                         full_repr = v
                         break
        elif isinstance(full_repr, tuple) or isinstance(full_repr, list):
             if isinstance(full_repr[0], tuple):
                 full_repr = full_repr[0][0]
             else:
                 full_repr = full_repr[0]
        
        log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w.to(full_repr.device).T + lm_b.to(full_repr.device), dim=2)
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(full_repr.device),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask.to(full_repr.device)).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        loss = nll_loss + kl_loss + weight_decay
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        loss.backward()
        opt.step()

        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    return target


def get_module_input_output_at_word(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_template: str,
    word: str,
    module_template: str,
    fact_token_strategy: str,
    track: str = "both",
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        subtok = fact_token_strategy[len("subject_") :]
        ret = repr_tools.get_reprs_at_word_tokens(
            track=track,
            subtoken=subtok,
            context_templates=[context_template],
            words=[word],
            **word_repr_args,
        )
    elif fact_token_strategy == "last":
        ret = repr_tools.get_reprs_at_idxs(
            track=track,
            contexts=[context_template.format(word)],
            idxs=[[-1]],
            **word_repr_args,
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    if isinstance(ret, tuple):
        return tuple(x.detach() for x in ret)
    return ret.detach()

def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
    track: str = "both",
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        subtok = fact_token_strategy[len("subject_") :]
        ret = repr_tools.get_reprs_at_word_tokens(
            track=track,
            subtoken=subtok,
            context_templates=context_templates,
            words=words,
            **word_repr_args,
        )
    elif fact_token_strategy == "last":
        ret = repr_tools.get_reprs_at_idxs(
            track=track,
            contexts=[
                context.format(word)
                for context, word in zip(context_templates, words)
            ],
            idxs=[[-1]] * len(context_templates),
            **word_repr_args,
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    if isinstance(ret, tuple):
        return tuple(x.detach() for x in ret)
    return ret.detach()

def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret
"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""
from ..models.melo.melo import LORA

import typing
from itertools import chain
from typing import List, Optional

import numpy as np
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from ..util import HyperParams
from .evaluate_utils import (
    test_seq2seq_batch_prediction_acc, 
    test_batch_prediction_acc, 
    test_prediction_acc,
    test_prediction_acc_LLM_judge,
    test_generation_quality, 
    test_concept_gen,
    test_safety_gen,
    test_instance_change,
    PPL,
    OOD_PPL,
    kl_loc_loss,
    es,
    es_per_icl,
    per_generation,
    F1
)

def compute_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device,
    eval_metric: str = 'token_em',
    test_generation = False,
    is_pre_edit = False,
    eval_locality = True
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    if isinstance(model,LORA):
        model=model.model
    # First, unpack rewrite evaluation record.
    target_new, ground_truth = (
        record[x] for x in ["target_new", "ground_truth"]
    )

    rewrite_prompts = record["prompt"]
    rephrase_prompts = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None

    def get_ans(prompts):
        if isinstance(prompts, str): prompts = [prompts]
        before_padding_side = tok.padding_side
        tok.padding_side = 'left'
        inputs = tok(prompts, return_tensors='pt', padding=True, truncation=True).to(f"cuda:{device}")
        tok.padding_side = before_padding_side
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=8,  # set 15
                pad_token_id=tok.eos_token_id,
                do_sample=False
            )
        if 't5' in model_name.lower():
            ans = [tok.decode(out, skip_special_tokens=True) for out in outputs]
        else:
            ans = [tok.decode(out[inputs['input_ids'].shape[1]:], skip_special_tokens=True) for out in outputs]
        return ans[0] if len(ans)==1 else ans

    ret = compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                              rewrite_prompts, target_new, device=device, eval_metric=eval_metric)
    
    if not is_pre_edit:
        ret['rewrite_ans'] = get_ans(rewrite_prompts)
    else:
        ret['rewrite_ans'] = ""

    ret['locality'] = {}
    ret['portability'] = {}
    if rephrase_prompts is not None:
        ret.update(
            compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                                rephrase_prompts, target_new, device=device, test_rephrase=True, eval_metric=eval_metric)
        )
        if not is_pre_edit:
            ret['rephrase_ans'] = get_ans(rephrase_prompts)
        else:
            ret['rephrase_ans'] = ""

    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            if not is_pre_edit or eval_locality:
                ret['locality'].update(
                    compute_locality_quality(model, model_name, hparams, tok, locality_key,
                                             record['locality'][locality_key]['prompt'],
                                             record['locality'][locality_key]['ground_truth'], device=device)
                )
                ret['locality'][f"{locality_key}_ans"] = get_ans(record['locality'][locality_key]['prompt'])
            else:
                ret['locality'][f"{locality_key}_ans"] = ""

    if 'portability' in record.keys() and any(record['portability']):
        for portability_key in record['portability'].keys():
            ret['portability'].update(
                compute_portability_quality(model, model_name, hparams, tok, portability_key,
                                            record['portability'][portability_key]['prompt'],
                                            record['portability'][portability_key]['ground_truth'], device=device)
            )
            if not is_pre_edit:
                ret['portability'][f"{portability_key}_ans"] = get_ans(record['portability'][portability_key]['prompt'])
            else:
                ret['portability'][f"{portability_key}_ans"] = ""

    if test_generation and not is_pre_edit:
        if hparams.alg_name == 'GRACE':
            ret['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=rewrite_prompts if isinstance(rewrite_prompts,list) else [rewrite_prompts,], max_out_len=100, vanilla_generation=True)
        else:
            ret['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=rewrite_prompts if isinstance(rewrite_prompts,list) else [rewrite_prompts,], max_out_len=100, vanilla_generation=False)
    return ret

def compute_rewrite_or_rephrase_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    prompt: str,
    target_new: str,
    device,
    test_rephrase: bool = False,
    eval_metric: str = 'token_em'
) -> typing.Dict:
    return {}

def compute_locality_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    locality_key: str,
    prompt: typing.Union[str, List[str]],
    locality_ground_truth: typing.Union[str, List[str]],
    device,
) -> typing.Dict:
    return {}

def compute_portability_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    portability_key: str,
    prompt: typing.Union[str, List[str]],
    ground_truth: typing.Union[str, List[str]],
    device,
) -> typing.Dict:
    return {}

def compute_icl_edit_quality(
        model,
        model_name,
        hparams: HyperParams,
        tok: AutoTokenizer,
        icl_examples,
        record: typing.Dict,
        device,
        pre_edit: bool = False,
        test_generation = False,
        eval_locality = True
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.
    """

    # First, unpack rewrite evaluation record.
    target_new, ground_truth = (
        record[x] for x in ["target_new", "ground_truth"]
    )
    prompt = record["prompt"]
    rephrase = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    new_fact = f'New Fact: {prompt} {target_new}\\nPrompt: {prompt}'

    def get_ans(icl_input, x_prefix, prompts):
        if isinstance(prompts, str): prompts = [prompts]
        ans = []
        for p in prompts:
            full_prompt = ''.join(icl_input) + f'{x_prefix}{p}'
            inputs = tok(full_prompt, return_tensors='pt').to(f"cuda:{device}")
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=15,
                    pad_token_id=tok.eos_token_id,
                    do_sample=False
                )
            if 't5' in model_name.lower():
                ans.append(tok.decode(outputs[0], skip_special_tokens=True))
            else:
                ans.append(tok.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True))
        return ans[0] if len(ans)==1 else ans

    icl_input_for_ans = [''] if pre_edit else icl_examples
    x_prefix_for_ans = "" if pre_edit else f"New Fact: {prompt} {target_new}\\nPrompt: "

    ret = {}
    
    if not pre_edit:
        ret['rewrite_ans'] = get_ans(icl_input_for_ans, x_prefix_for_ans, prompt)
    else:
        ret['rewrite_ans'] = ""
    
    ret['locality'] = {}
    ret['portability'] = {}
    if rephrase is not None:
        if not pre_edit:
            ret['rephrase_ans'] = get_ans(icl_input_for_ans, x_prefix_for_ans, rephrase)
        else:
            ret['rephrase_ans'] = ""

    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            if not pre_edit or eval_locality:
                ret['locality'][f"{locality_key}_ans"] = get_ans(icl_input_for_ans, x_prefix_for_ans, record['locality'][locality_key]['prompt'])
            else:
                ret['locality'][f"{locality_key}_ans"] = ""
            
    if 'portability' in record.keys() and any(record['portability']):
        for portability_key in record['portability'].keys():
            if not pre_edit:
                ret['portability'][f"{portability_key}_ans"] = get_ans(icl_input_for_ans, x_prefix_for_ans, record['portability'][portability_key]['prompt'])
            else:
                ret['portability'][f"{portability_key}_ans"] = ""

    if test_generation and not pre_edit:
        ret['fluency'] = test_generation_quality(model=model,tok=tok, prefixes=new_fact if isinstance(new_fact,list) else [new_fact,], max_out_len=100, vanilla_generation=False)
    return ret

def icl_lm_eval(
        model,
        model_name,
        hparams: HyperParams,
        tokenizer,
        icl_examples,
        target,
        x,
        neighborhood=False
)-> typing.Dict:
    device = torch.device(f'cuda:{hparams.device}')
    if 't5' in model_name.lower():
        target_len = len(tokenizer.encode(target))
        target_ids = tokenizer(f'{x} {target}', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples), return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids).logits
            ans = torch.argmax(logits, dim=-1)[:,-target_len:-1].squeeze()
            target_ids = target_ids[:,-target_len:-1]
            if neighborhood:
                return ans.squeeze().detach().cpu().numpy().tolist()
            return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()
    elif 'llama' in model_name.lower():
        target_ids = tokenizer(target, return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
        target_ids = target_ids[:,1:]
        if neighborhood:
            return ans.squeeze().detach().cpu().numpy().tolist()
        return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()
    else:
        target_ids = tokenizer(' ' + target + '\\n', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
        target_ids = target_ids[:,:-1]
        if neighborhood:
            return ans.squeeze().detach().cpu().numpy().tolist()
        return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()

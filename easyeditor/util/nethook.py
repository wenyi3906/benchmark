import contextlib
import copy
import inspect
import typing
from collections import OrderedDict
from typing import Any, Callable, Dict, Sequence, Union

import torch
import torch.nn as nn


class Trace(contextlib.ContextDecorator):
    """
    To retain the output of the named layer during the forward pass:

    with Trace(net, 'layer.name') as ret:
        _ = net(inp)
        representation = ret.output

    A layer module can be passed directly without a layer name, and
    its output will be retained.  By default, a clone of the tensor
    is returned, and the gradient is not retained.  To alter this,
    set clone=False, detach=False, retain_grad=True.
    """

    def __init__(
        self,
        module: nn.Module,
        layer: Union[str, Sequence[str]],
        retain_output: bool = True,
        retain_input: bool = False,
        clone: bool = True,
        detach: bool = True,
        retain_grad: bool = False,
        edit_output: Callable = None,
        stop: bool = False,
    ):
        """
        Method to replace a forward method with a closure that
        intercepts the call, and tracks the hook so that it can be reverted.
        """
        retainer = self
        self.layer = layer
        if isinstance(layer, str):
            module = get_module(module, layer)
        else:
            module = layer

        def retain_hook(m, inputs, kwargs, output):
            if retain_input:
                # Try to get input from positional args first
                if len(inputs) > 0:
                    retainer.input = recursive_copy(
                        inputs[0] if len(inputs) == 1 else inputs,
                        clone=clone,
                        detach=detach,
                        retain_grad=False,
                    )
                # If inputs is empty, try to get hidden_states from kwargs (modern transformers)
                elif kwargs is not None and 'hidden_states' in kwargs:
                    retainer.input = recursive_copy(
                        kwargs['hidden_states'],
                        clone=clone,
                        detach=detach,
                        retain_grad=False,
                    )
                else:
                    # Fallback: use output as input proxy for transformer blocks
                    # since they transform hidden_states -> hidden_states with same shape
                    retainer.input = None
            if edit_output:
                output = invoke_with_optional_args(
                    edit_output, output, self.layer, output=output, layer=self.layer
                )
            if retain_output:
                retainer.output = recursive_copy(
                    output, clone=clone, detach=detach, retain_grad=retain_grad
                )
                # When retain_grad is set, also insert a trivial
                # copy operation.  That allows in-place operations
                # to follow without error.
                if retain_grad:
                    output = recursive_copy(retainer.output, clone=True, detach=False)
            if stop:
                raise StopForward()
            return output

        # Try to use with_kwargs=True for PyTorch 2.0+ to capture kwargs like hidden_states
        try:
            # PyTorch 2.0+ supports with_kwargs parameter
            self.registered_hook = module.register_forward_hook(retain_hook, with_kwargs=True)
        except TypeError:
            # Fallback for older PyTorch versions - wrap the hook to ignore kwargs parameter
            def legacy_hook(m, inputs, output):
                return retain_hook(m, inputs, None, output)
            self.registered_hook = module.register_forward_hook(legacy_hook)
        self.stop = stop

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True

    def close(self):
        self.registered_hook.remove()


class TraceDict(OrderedDict, contextlib.ContextDecorator):
    """
    To retain the output of multiple named layers during a forward pass:

    with TraceDict(net, ['layer1.name1', 'layer2.name2']) as ret:
        _ = net(inp)
        representation = ret['layer1.name1'].output

    If edit_output is provided, it should be a function that takes
    two arguments: output, and the layer name; and then it returns the
    modified output.

    Other arguments are the same as Trace.  If stop is True, then the
    execution of the forward pass is stopped after the last layer
    is executed (i.e., a StopForward exception is raised and caught).
    """

    def __init__(
        self,
        module: nn.Module,
        layers: Sequence[str] = None,
        retain_output: bool = True,
        retain_input: bool = False,
        clone: bool = True,
        detach: bool = True,
        retain_grad: bool = False,
        edit_output: Callable = None,
        stop: bool = False,
    ):
        self.stop = stop

        def flag_make(p, name):
            if isinstance(p, dict):
                return p.get(name, False)
            if isinstance(p, list):
                return name in p
            return p

        def edit_make(e, name):
            if isinstance(e, dict):
                return e.get(name, None)
            return e

        if layers is None:
            layers = [n for n, m in module.named_modules()]

        for name in layers:
            if flag_make(stop, name) and name != layers[-1]:
                raise ValueError("Stop should only be True for the last layer.")
            self[name] = Trace(
                module=module,
                layer=name,
                retain_output=flag_make(retain_output, name),
                retain_input=flag_make(retain_input, name),
                clone=flag_make(clone, name),
                detach=flag_make(detach, name),
                retain_grad=flag_make(retain_grad, name),
                edit_output=edit_make(edit_output, name),
                stop=flag_make(stop, name),
            )

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True

    def close(self):
        for layer, trace in self.items():
            trace.close()


class StopForward(Exception):
    """
    If the only output needed from running a network is the retained
    submodule then Trace(submodule, stop=True) will stop execution
    immediately after the retained submodule by raising the StopForward()
    exception.  When Trace is used as context manager, it catches that
    exception and can be used as follows:

    with Trace(net, layername, stop=True) as tr:
        net(inp) # Only runs the network up to layername
    print(tr.output)
    """

    pass


def get_module(model, name):
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


def get_parameter(model, name):
    """
    Finds the named parameter within the given model.
    """
    for n, p in model.named_parameters():
        if n == name:
            return p
    raise LookupError(name)


def replace_module(model, name, new_module):
    """
    Replaces the named module within the given model.
    """
    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = get_module(model, parent_name)
        setattr(parent, child_name, new_module)
    else:
        setattr(model, name, new_module)


def recursive_copy(x, clone=None, detach=None, retain_grad=None):
    """
    Copies a reference to a tensor, or an object that contains tensors,
    optionally detaching and cloning the tensor(s).  If retain_grad is
    true, the original tensors are marked to have grads retained.
    """
    if not clone and not detach and not retain_grad:
        return x
    if isinstance(x, torch.Tensor):
        if retain_grad:
            if not x.requires_grad:
                x.requires_grad = True
            x.retain_grad()
        elif detach:
            x = x.detach()
        if clone:
            x = x.clone()
        return x
    # Only dicts, lists, and tuples (and subclasses) can be copied.
    if isinstance(x, dict):
        return type(x)({k: recursive_copy(v, clone, detach, retain_grad) for k, v in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)(
            [recursive_copy(v, clone, detach, retain_grad) for v in x]
        )
    else:
        # Return the object as is instead of crashing on custom objects like DynamicCache
        return x


def invoke_with_optional_args(fn, *args, **kwargs):
    """
    Invokes a function with only the arguments that it
    is written to accept, giving priority to arguments
    that match by-name, using the following rules.
    1. Arguments with matching names are passed by name.
    2. Remaining args are passed by order.
    3. Extra args are discarded.
    4. Extra kwargs are discarded.
    """
    argspec = inspect.getfullargspec(fn)
    pass_args = []
    used_kw = set()
    used_pos = 0
    for kw in argspec.args:
        if kw in kwargs:
            pass_args.append(kwargs[kw])
            used_kw.add(kw)
        elif used_pos < len(args):
            pass_args.append(args[used_pos])
            used_pos += 1
    pass_kw = {
        k: v
        for k, v in kwargs.items()
        if k in argspec.kwonlyargs and k not in used_kw
    }
    if argspec.varkw is not None:
        pass_kw.update(
            {k: v for k, v in kwargs.items() if k not in used_kw and k not in pass_kw}
        )
    if argspec.varargs is not None:
        pass_args += list(args[used_pos:])
    return fn(*pass_args, **pass_kw)


def set_requires_grad(requires_grad, *models):
    """
    Sets requires_grad true or false for all parameters within the
    models passed.
    """
    for model in models:
        if isinstance(model, torch.nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad
        else:
            assert False, "unknown type %r" % type(model)


class InstrumentedAttribute:
    """
    To intercept getting or setting of a specific object attribute
    during a forward pass:

    with InstrumentedAttribute(obj, 'attr_name') as ret:
        _ = net(inp)

    Within the block, the object obj has been temporarily altered so that
    getting the attribute attr_name calls the function get_fn(val)
    and setting the attribute attr_name calls the function set_fn(val).
    The original attribute is restored when the block exits.
    """

    def __init__(self, obj, attr_name, get_fn=None, set_fn=None):
        self.obj = obj
        self.attr_name = attr_name
        self.get_fn = get_fn
        self.set_fn = set_fn
        self.original_attr = getattr(obj, attr_name)
        
    def __enter__(self):
        class Wrapper:
            def __init__(self, original, get_fn, set_fn):
                self.original = original
                self.get_fn = get_fn
                self.set_fn = set_fn

            def __get__(self, instance, owner):
                val = self.original
                if self.get_fn:
                    val = self.get_fn(val)
                return val

            def __set__(self, instance, value):
                if self.set_fn:
                    value = self.set_fn(value)
                self.original = value

        setattr(self.obj.__class__, self.attr_name, Wrapper(self.original_attr, self.get_fn, self.set_fn))
        return self

    def __exit__(self, type, value, traceback):
        setattr(self.obj.__class__, self.attr_name, self.original_attr)
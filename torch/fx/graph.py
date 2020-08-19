# type: ignore
from .node import Node

import builtins
import torch

def _is_magic(x):
    return x.startswith('__') and x.endswith('__')

def snake_case(s):
    return ''.join(['_' + i.lower() if i.isupper() else i for i in s]).lstrip('_')

def _qualified_name(func):
    # things like getattr just appear in builtins
    if getattr(builtins, func.__name__, None) is func:
        return func.__name__
    name = func.__name__
    module = _find_module_of_method(func)
    module = module.replace('torch._ops', 'torch.ops')  # WAR for bug in how torch.ops assigns module
    return f'{module}.{name}'

# this is fixed on master, WAR for 1.5
def _find_module_of_method(orig_method):
    name = orig_method.__name__
    module = orig_method.__module__
    if module is not None:
        return module
    for guess in [torch, torch.nn.functional]:
        if getattr(guess, name, None) is orig_method:
            return guess.__name__
    raise RuntimeError(f'cannot find module for {orig_method}')

def _format_args(args, kwargs):
    args_s = ', '.join(repr(a) for a in args)
    kwargs_s = ', '.join(f'{k} = {repr(v)}' for k, v in kwargs.items())
    if args_s and kwargs_s:
        return f'{args_s}, {kwargs_s}'
    return args_s or kwargs_s

def _format_target(base, target):
    elems = target.split('.')
    r = base
    for e in elems:
        if not e.isidentifier():
            r = f'getattr({r}, "{e}")'
        else:
            r = f'{r}.{e}'
    return r

def map_arg(a, fn):
    """ apply fn to each Node appearing arg. arg may be a list, tuple, slice, or dict with string keys. """
    if isinstance(a, (tuple, list)):
        return type(a)(map_arg(elem, fn) for elem in a)
    elif isinstance(a, dict):
        return {k: map_arg(v, fn) for k, v in a.items()}
    elif isinstance(a, slice):
        return slice(map_arg(a.start, fn), map_arg(a.stop, fn), map_arg(a.step, fn))
    elif isinstance(a, Node):
        return fn(a)
    else:
        return a

class Graph:
    def __init__(self):
        self.nodes = []
        self._used_names = {}  # base name -> number

    def _mark_uses(self, a):
        def add_use(n: Node):
            n.uses += 1
            return n
        map_arg(a, add_use)

    def create_node(self, op, target=None, args=None, kwargs=None, name=None):
        assert op in ('call_function', 'call_method', 'get_param', 'call_module', 'placeholder')
        args = () if args is None else args
        kwargs = {} if kwargs is None else kwargs
        self._mark_uses(args)
        self._mark_uses(kwargs)
        n = Node(self, name if name is not None else self._name(target or op), op, target, args, kwargs)
        self.nodes.append(n)
        return n

    def node_copy(self, node, arg_transform=lambda x: x):
        """ copy a node from one graph into another. arg_transform needs to transform arguments from the graph of node
            to the graph of self"""
        return self.create_node(
            node.op, node.target, map_arg(node.args, arg_transform), map_arg(node.kwargs, arg_transform),
            self._name(node.name))

    def output(self, result):
        self.result = result
        self._mark_uses(result)


    def _name(self, op):
        if hasattr(op, '__name__'):
            op = op.__name__

        if _is_magic(op):
            op = op[2:-2]
        op = op.replace('.', '_')
        op = snake_case(op)

        if op not in self._used_names:
            self._used_names[op] = 0
            if not hasattr(torch, op) and not hasattr(torch.nn.functional, op) and not hasattr(torch.nn, op):
                return op
        i = self._used_names[op] = self._used_names[op] + 1
        return f'{op}_{i}'

    def get_param(self, target):
        return self.create_node('get_param', target)

    def placeholder(self, name):
        return self.create_node('placeholder', target=name, name=name.replace('*', ''))

    def python_code(self, root_module):
        free_vars = []
        body = []
        for node in self.nodes:
            if node.op == 'placeholder':
                free_vars.append(node.target)
                continue
            elif node.op == 'call_method':
                body.append(
                    f'{node.name} = {_format_target(repr(node.args[0]), node.target)}'
                    f'({_format_args(node.args[1:], node.kwargs)})\n')
                continue
            elif node.op == 'call_function':
                # pretty print operators
                if node.target.__module__ == '_operator' and node.target.__name__ in magic_methods:
                    body.append(f'{node.name} = {magic_methods[node.target.__name__].format(*(repr(a) for a in node.args))}\n')
                    continue
                qualified_name = _qualified_name(node.target)
                if qualified_name == 'getattr' and isinstance(node.args[1], str) and node.args[1].isidentifier():
                    # pretty print attribute access
                    body.append(f'{node.name} = {_format_target(repr(node.args[0]), node.args[1])}\n')
                    continue
                body.append(f'{node.name} = {qualified_name}({_format_args(node.args, node.kwargs)})\n')
                continue
            elif node.op == 'call_module':
                body.append(f'{node.name} = {_format_target(root_module,node.target)}({_format_args(node.args, node.kwargs)})\n')
                continue
            elif node.op == 'get_param':
                body.append(f'{node.name} = {_format_target(root_module, node.target)}\n')
                continue
            raise NotImplementedError(f'node: {node.op} {node.target}')

        src = ''.join(body)
        return src, str(self.result), free_vars

reflectable_magic_methods = {
    'add': '{} + {}',
    'sub': '{} - {}',
    'mul': '{} * {}',
    'floordiv': '{} // {}',
    'truediv': '{} / {}',
    'div': '{} / {}',
    'mod': '{} % {}',
    'pow': '{} ** {}',
    'lshift': '{} << {}',
    'rshift': '{} >> {}',
    'and': '{} & {}',
    'or': '{} | {}',
    'xor': '{} ^ {}',
    'getitem': '{}[{}]'
}

magic_methods = dict({
    'eq': '{} == {}',
    'ne': '{} != {}',
    'lt': '{} < {}',
    'gt': '{} > {}',
    'le': '{} <= {}',
    'ge': '{} >= {}',
    'pos': '+{}',
    'neg': '-{}',
    'invert': '~{}'}, **reflectable_magic_methods)

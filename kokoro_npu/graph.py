# SPDX-License-Identifier: Apache-2.0
"""Rewrite the Kokoro ONNX graph so eligible Conv nodes run on the NPU.

Each eligible ``Conv(X, W, B)`` becomes ``NpuOp(X)`` (domain ``kokoro.npu``)
followed by an ONNX ``Add`` for the bias. Weights leave the graph and are
returned in a side table keyed by ``op_id`` so the runtime can preload them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnx
from onnx import helper, numpy_helper

NPU_DOMAIN = "kokoro.npu"


@dataclass
class SnakeSpec:
    """x + beta * sin(alpha * x)^2 with per-channel alpha/beta ([C])."""

    op_id: int
    name: str
    alpha: np.ndarray
    beta: np.ndarray

    @property
    def channels(self) -> int:
        return int(self.alpha.shape[0])


@dataclass
class FusedSpec:
    """snake followed by a conv that is its only consumer (both on the NPU, one op)."""

    op_id: int
    snake: "SnakeSpec"
    conv: "ConvSpec"


@dataclass
class ConvSpec:
    op_id: int
    name: str
    weight: np.ndarray  # [cout, cin, taps] f32
    bias: np.ndarray | None
    taps: int
    dil: int
    pad_l: int
    pad_r: int

    @property
    def cout(self) -> int:
        return self.weight.shape[0]

    @property
    def cin(self) -> int:
        return self.weight.shape[1]

    @property
    def t_delta(self) -> int:
        return self.pad_l + self.pad_r - (self.taps - 1) * self.dil


def _attrs(node) -> dict:
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def conv_eligible(node, init: dict, cout_mult: int, name_filter=None) -> bool:
    if node.op_type != "Conv" or node.input[1] not in init:
        return False
    if name_filter is not None and not name_filter(node.name):
        return False
    w = init[node.input[1]]
    if len(w.dims) != 3:
        return False
    a = _attrs(node)
    strides = list(a.get("strides", [1]))
    dil = list(a.get("dilations", [1]))
    pads = list(a.get("pads", [0, 0]))
    if strides != [1] or a.get("group", 1) != 1 or len(dil) != 1 or dil[0] % 2 == 0:
        return False
    if a.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET"):
        return False
    if len(pads) != 2:
        return False
    if w.dims[0] % cout_mult:
        return False
    if len(node.input) > 2 and node.input[2] and node.input[2] not in init:
        return False
    return True


def _find_snake_chains(nodes, init: dict, channel_mult: int):
    """Return {first_mul_name: (nodes_to_drop, x_name, out_name, alpha, beta)}."""
    prod = {o: n for n in nodes for o in n.output}
    cons: dict[str, list] = {}
    for n in nodes:
        for i in n.input:
            cons.setdefault(i, []).append(n)

    def single(n):
        c = cons.get(n.output[0], [])
        return c[0] if len(c) == 1 else None

    def init_arr(name):
        return numpy_helper.to_array(init[name]) if name in init else None

    chains = {}
    for sin in nodes:
        if sin.op_type != "Sin":
            continue
        mul1 = prod.get(sin.input[0])
        if mul1 is None or mul1.op_type != "Mul" or single(mul1) is not sin:
            continue
        a_name = next((i for i in mul1.input if i in init), None)
        if a_name is None:
            continue
        x_name = next(i for i in mul1.input if i != a_name)
        alpha = init_arr(a_name)
        pw = single(sin)
        if pw is None or pw.op_type != "Pow":
            continue
        e = init_arr(pw.input[1])
        if e is None or e.size != 1 or float(e) != 2.0:
            continue
        mul2 = single(pw)
        if mul2 is None or mul2.op_type != "Mul":
            continue
        b_name = next((i for i in mul2.input if i in init), None)
        if b_name is None:
            continue
        beta = init_arr(b_name)
        add = single(mul2)
        if add is None or add.op_type != "Add" or x_name not in add.input:
            continue
        if alpha.shape != beta.shape or alpha.ndim != 3 or alpha.shape[0] != 1 or alpha.shape[2] != 1:
            continue
        c = alpha.shape[1]
        if c % channel_mult:
            continue
        chains[mul1.name] = ([mul1, sin, pw, mul2, add], x_name, add.output[0], alpha.reshape(-1), beta.reshape(-1))
    return chains


def rewrite(
    model: onnx.ModelProto, cout_mult: int = 128, name_filter=None, snake_channel_mult: int = 16, snake: bool = True,
    bias_in_kernel: bool = True,
) -> tuple[onnx.ModelProto, list[ConvSpec], list[SnakeSpec]]:
    """Rewrite eligible Convs and Snake chains into NpuOp nodes.

    With ``bias_in_kernel`` the conv bias is applied by the NPU kernel (ConvSpec.bias);
    otherwise an ONNX Add node is appended after the NpuOp.
    """
    g = model.graph
    init = {i.name: i for i in g.initializer}
    specs: list[ConvSpec] = []
    snakes: list[SnakeSpec] = []
    new_nodes = []
    consumed: dict[str, int] = {}
    for n in g.node:
        for i in n.input:
            consumed[i] = consumed.get(i, 0) + 1

    chains = _find_snake_chains(list(g.node), init, snake_channel_mult) if snake else {}
    drop = set()
    for drop_nodes, *_ in chains.values():
        drop.update(id(n) for n in drop_nodes)

    for node in g.node:
        if node.name in chains:
            drop_nodes, x_name, out_name, alpha, beta = chains[node.name]
            op_id = 1_000_000 + len(snakes)
            snakes.append(SnakeSpec(op_id=op_id, name=node.name, alpha=alpha.astype(np.float32), beta=beta.astype(np.float32)))
            new_nodes.append(
                helper.make_node(
                    "NpuOp", [x_name], [out_name], name=node.name + "_npu_snake", domain=NPU_DOMAIN,
                    op_id=op_id, cout=0, t_delta=0,
                )
            )
            for d in drop_nodes:
                for i in d.input:
                    consumed[i] -= 1
            consumed[x_name] += 1
            continue
        if id(node) in drop:
            continue
        if not conv_eligible(node, init, cout_mult, name_filter):
            new_nodes.append(node)
            continue
        a = _attrs(node)
        w = numpy_helper.to_array(init[node.input[1]]).astype(np.float32)
        b = None
        if len(node.input) > 2 and node.input[2]:
            b = numpy_helper.to_array(init[node.input[2]]).astype(np.float32)
        op_id = len(specs)
        spec = ConvSpec(
            op_id=op_id,
            name=node.name,
            weight=w,
            bias=b,
            taps=int(w.shape[2]),
            dil=int(list(a.get("dilations", [1]))[0]),
            pad_l=int(list(a.get("pads", [0, 0]))[0]),
            pad_r=int(list(a.get("pads", [0, 0]))[1]),
        )
        specs.append(spec)
        out = node.output[0]
        conv_out = out if (b is None or bias_in_kernel) else out + "_npu_nobias"
        new_nodes.append(
            helper.make_node(
                "NpuOp",
                [node.input[0]],
                [conv_out],
                name=node.name + "_npu",
                domain=NPU_DOMAIN,
                op_id=op_id,
                cout=int(w.shape[0]),
                t_delta=spec.t_delta,
            )
        )
        if b is not None and not bias_in_kernel:
            bname = node.input[2] + "_npu_b"
            if bname not in init:
                t = numpy_helper.from_array(b.reshape(1, -1, 1), bname)
                g.initializer.append(t)
                init[bname] = t
            new_nodes.append(helper.make_node("Add", [conv_out, bname], [out], name=node.name + "_npu_bias"))
        for i in node.input[1:]:
            consumed[i] -= 1

    # Drop initializers no longer referenced.
    keep = [i for i in g.initializer if consumed.get(i.name, 1) > 0 or i.name.endswith("_npu_b")]
    del g.initializer[:]
    g.initializer.extend(keep)
    del g.node[:]
    g.node.extend(new_nodes)

    if not any(o.domain == NPU_DOMAIN for o in model.opset_import):
        model.opset_import.append(helper.make_opsetid(NPU_DOMAIN, 1))
    return model, specs, snakes


def fuse_snake_conv(model: onnx.ModelProto, specs: list[ConvSpec], snakes: list[SnakeSpec], channel_mult: int = 64) -> list[FusedSpec]:
    """Merge NpuOp(snake) -> NpuOp(conv) pairs into one NpuOp node. Mutates the graph."""
    g = model.graph
    by_conv = {s.op_id: s for s in specs}
    by_snake = {s.op_id: s for s in snakes}
    cons: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    outputs = {o.name for o in g.output}

    def op_id(n):
        return next(helper.get_attribute_value(a) for a in n.attribute if a.name == "op_id")

    fused: list[FusedSpec] = []
    drop = set()
    for n in g.node:
        if n.op_type != "NpuOp" or n.domain != NPU_DOMAIN:
            continue
        sp = by_snake.get(op_id(n))
        if sp is None or n.output[0] in outputs:
            continue
        users = cons.get(n.output[0], [])
        if len(users) != 1 or users[0].op_type != "NpuOp":
            continue
        cv = by_conv.get(op_id(users[0]))
        if cv is None or cv.cin != sp.channels or cv.cin % channel_mult or cv.taps < 2:
            continue
        fid = 2_000_000 + len(fused)
        fused.append(FusedSpec(op_id=fid, snake=sp, conv=cv))
        conv_node = users[0]
        conv_node.input[0] = n.input[0]
        for a in conv_node.attribute:
            if a.name == "op_id":
                a.i = fid
        conv_node.name = n.name + "+" + conv_node.name
        drop.add(id(n))
    keep = [n for n in g.node if id(n) not in drop]
    del g.node[:]
    g.node.extend(keep)
    return fused


def rewrite_stft(model: onnx.ModelProto) -> int:
    """Replace ORT's slow STFT (onesided, constant window/step/length) with an
    equivalent strided Conv over a windowed DFT basis. Returns the number rewritten."""
    g = model.graph
    init = {i.name: i for i in g.initializer}
    new_nodes = []
    count = 0
    for node in g.node:
        a = _attrs(node)
        if node.op_type != "STFT" or a.get("onesided", 1) != 1 or len(node.input) < 4:
            new_nodes.append(node)
            continue
        sig, step_n, win_n, flen_n = node.input[:4]
        if not all(x in init for x in (step_n, win_n, flen_n)):
            new_nodes.append(node)
            continue
        step = int(numpy_helper.to_array(init[step_n]))
        win = numpy_helper.to_array(init[win_n]).astype(np.float64)
        flen = int(numpy_helper.to_array(init[flen_n]))
        if win.shape != (flen,):
            new_nodes.append(node)
            continue
        nb = flen // 2 + 1
        n = np.arange(flen)
        k = np.arange(nb)[:, None]
        basis = np.empty((nb, 2, flen), dtype=np.float64)
        basis[:, 0, :] = win * np.cos(2 * np.pi * k * n / flen)
        basis[:, 1, :] = -win * np.sin(2 * np.pi * k * n / flen)
        # The model derives a phase from atan(im/re) plus sign branches. For the DC and
        # Nyquist bins the true imaginary part is 0 and torch.atan2(0, re<0) = +pi, which
        # the exported branch structure only reproduces when im is a tiny positive number
        # for re < 0. ORT's STFT happens to give im ~ -1e-6*re there; emulate that
        # relationship exactly (magnitude unaffected at 1e-14).
        basis[np.abs(basis) < 1e-9] = 0.0
        for kk in range(nb):
            if np.all(basis[kk, 1, :] == 0.0):
                basis[kk, 1, :] = -1e-7 * basis[kk, 0, :]
        w_name = node.name + "_dft_w"
        g.initializer.append(numpy_helper.from_array(basis.reshape(nb * 2, 1, flen).astype(np.float32), w_name))
        s1 = node.name + "_shape3"
        s2 = node.name + "_shape4"
        g.initializer.append(numpy_helper.from_array(np.array([0, 1, -1], dtype=np.int64), s1))
        g.initializer.append(numpy_helper.from_array(np.array([0, nb, 2, -1], dtype=np.int64), s2))
        x3 = node.name + "_x3"
        y3 = node.name + "_y3"
        y4 = node.name + "_y4"
        new_nodes.append(helper.make_node("Reshape", [sig, s1], [x3], name=node.name + "_r1"))
        new_nodes.append(
            helper.make_node("Conv", [x3, w_name], [y3], name=node.name + "_conv", kernel_shape=[flen], strides=[step], pads=[0, 0])
        )
        new_nodes.append(helper.make_node("Reshape", [y3, s2], [y4], name=node.name + "_r2"))
        new_nodes.append(helper.make_node("Transpose", [y4], [node.output[0]], name=node.name + "_t", perm=[0, 3, 1, 2]))
        count += 1
    del g.node[:]
    g.node.extend(new_nodes)
    return count

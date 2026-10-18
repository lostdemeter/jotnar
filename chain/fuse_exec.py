"""Fused lattice execution: run advisor chains without materializing.

run_fused(text, chain_lines, payload, ...) delegates to chain/asm.run()
with a write-routing hook: ops whose bound line is in chain_lines write
their outputs to the hook env instead of feeds (chain-final ops --
consumed outside the chain, or sinks -- still materialize). Same
assemble, same op functions, same order, same checks = byte-identical
by construction (gated anyway, never trusted).

Consumes advise() output mechanically: chain_lines is the report's
`lines` for one chain (the gate asserts identity with the advisor's
current report -- hand-wiring a chain without the advisor link is a
scope violation). Generic over chains: zero per-chain code here.
Wall savings on lattice are feeds-dict traffic only (numpy still
allocates per-op temporaries in both arms) -- reported, never
asserted beyond the row. No emitter changes; CUDA emission of fused
chains is blocked (T-kind + missing BETA_V5/GAIN/SRGB patterns --
see docs/FUSION.md), not attempted here.
"""
from chain import asm as ASM


class _ChainStore:
    """Write-routing hook for asm.run(): chain-interior outs -> env.

    Interior = bound line in chain_lines AND every consumer also in
    chain_lines (derived from the bound consumer map -- no caller
    annotation, no per-chain code). dest() counts fused ops.
    """

    def __init__(self, chain_lines, prog):
        self.env = {}
        self.fused = 0
        self._lines = set(chain_lines)
        consumers = {}
        for outs, _, _, args, ln, _ in prog:
            for a in args:
                if ASM._is_literal(a):
                    continue
                consumers.setdefault(a, []).append(ln)
        self._consumers = consumers

    def dest(self, outs, ln):
        con = [c for o in outs for c in self._consumers.get(o, [])]
        if ln in self._lines and con and all(c in self._lines for c in con):
            self.fused += 1
            return True
        return False


def run_fused(text, chain_lines, payload, registry=None, sigs=None,
              basedir=".", trace=None, overrides=None):
    """Execute with the given bound lines fused (locals, not feeds).
    Returns (feeds, fused, env): feeds holds everything EXCEPT fused
    interior streams (assert their absence to prove non-materialization;
    their values live in env for byte-compare), fused counts ops run
    into the side env."""
    from chain.asm_ops import REGISTRY as _R, SIGS as _S
    config, inp, bound, _ = ASM.assemble(text, registry or _R,
                                         sigs or _S, basedir)
    hook = _ChainStore(chain_lines, bound)
    feeds = ASM.run((config, inp, bound), config, inp, payload,
                    trace=trace, overrides=overrides, store_hook=hook)
    return feeds, hook.fused, hook.env

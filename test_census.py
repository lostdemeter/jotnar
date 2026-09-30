"""Access-shape census gates (v1.3 gate 1): how data MOVES.

The census maps producer/consumer/fan-out per stream + access class per op,
statically (zero execution). Gates: planted sharing recovered exactly,
sharing-free control reports empty (tripwire against wallpaper censuses),
ACCESS covers all 42 mnemonics (drift-proof like asm-lang-coverage), and
the two production listings pin their real access maps (reuse made
visible: flagship A fan-out 3, xf_block XN fan-out 3).
Usage: python3 test_census.py
"""
import os
import sys

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phi-core")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chain import census as CS
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

FAIL = []

PLANTED = """IN a
IN b
IN m
S = ADD(a, b)
T = MUL(S, a)
U = MUL(S, S)
OUT = SELECT(m, T, U)
"""

CHAIN = """IN a
X = SILU(a)
OUT = GELU(X)
"""


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def main():
    r = CS.census_text(PLANTED, REGISTRY, SIGS)
    check("census-planted-shared", r["shared"] == {"S": 3, "a": 2},
          f"planted fan-out recovered {r['shared']}")
    check("census-planted-classes",
          r["classes"] == {"elementwise": 3, "select": 1},
          f"{r['classes']}")
    sp = r["streams"]["S"]
    check("census-planted-producer",
          sp["producer"] == ("ADD", sp["producer"][1]) and len(sp["consumers"]) == 3,
          "producer named + 3 consumers listed")
    c = CS.census_text(CHAIN, REGISTRY, SIGS)
    check("census-control-empty", c["shared"] == {},
          "sharing-free chain reports no sharing (tripwire)")
    check("census-control-sources",
          all(c["streams"][n]["source"] for n in ("a",)),
          "IN streams marked as sources")
    check("census-coverage", set(CS.ACCESS) >= set(REGISTRY),
          f"{len(CS.ACCESS)}/{len(REGISTRY)} mnemonics classified")
    f = CS.census_text(open(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "programs",
        "holo_flagship.asm")).read(), REGISTRY, SIGS,
        basedir=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "programs"))
    check("census-flagship", f["shared"] == {"A": 3, "LIN": 2, "Y": 2, "D": 2},
          f"amplitude read 3x, boundaries 2x ({f['shared']})")
    x = CS.census_text(open(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "programs",
        "xf_block.asm")).read(), REGISTRY, SIGS,
        basedir=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "programs"))
    check("census-xfblock", x["shared"] == {"XN": 3, "HN": 2, "x": 2, "pos": 2, "H": 2},
          f"normed streams fanned ({x['shared']})")
    check("census-xfblock-classes", x["classes"].get("reduce", 0) >= 10,
          f"attention block is reduction-heavy ({x['classes']})")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

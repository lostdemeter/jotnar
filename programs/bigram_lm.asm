# Bigram LM (v1.6 construction): next-token by frozen counts.
#
# The constructed creature's first organ: bank (V,V) bigram counts as IN
# data (frozen offline by scripts/freeze_lm.py), current token id as IN.
# ROW = GATHER(bank, tok) (the conditional distribution, unnormalized --
# argmax is monotone-invariant, no float needed); OUT = ARGMAX(ROW, 1)
# (greedy decode per the design: temperature/top-k stay host-boundary).
# Counts are exact integers (no trained weights anywhere in this program).
# Proof: test_lm.py demands train memorization exact + held-out stats.
IN tok
IN bank

ROW = GATHER(bank, tok)
OUT = ARGMAX(ROW, 1)

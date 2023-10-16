def generate(model, prefix, max_len=100):
    initial = model.encode([prefix])
    next_tokens = model.decode_inference([initial], max_len)
    # cutoff by EOS | make some constants.py ?

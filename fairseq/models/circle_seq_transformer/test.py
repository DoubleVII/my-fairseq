import torch
import random
padding_idx = 1
target_circle_rate = 0.8
middle = 299

tgt_tokens = torch.LongTensor([[2,3,4,5,6],[3,4,5,6,7],[2,3,4,5,1],[3,4,5,1,1]])

new_tgt_tokens = torch.zeros((tgt_tokens.size(0), tgt_tokens.size(1)+1)).to(tgt_tokens)
new_prev_output_tokens = torch.zeros((tgt_tokens.size(0), tgt_tokens.size(1)+1)).to(tgt_tokens)
padding_bias = (tgt_tokens == padding_idx).to(torch.int).sum(dim=1) + 1
for i in range(len(tgt_tokens)):
    tgt_tokens_len = tgt_tokens.size(1) - padding_bias[i]
    if random.random() > target_circle_rate:
        swap_position = tgt_tokens_len
    else:
        swap_position = random.randint(1, tgt_tokens.size(1) - padding_bias[i] -1)
    new_tgt_tokens[i,tgt_tokens_len+1:] = tgt_tokens[i,tgt_tokens_len:]
    new_tgt_tokens[i,:tgt_tokens_len+1] = torch.cat([tgt_tokens[i,swap_position:tgt_tokens_len], torch.LongTensor((middle,)).to(tgt_tokens), tgt_tokens[i,:swap_position]])
    new_prev_output_tokens[i,tgt_tokens_len+2:] = tgt_tokens[i,tgt_tokens_len+1:]
    # _ = torch.cat([tgt_tokens[i, tgt_tokens_len:tgt_tokens_len+1] , tgt_tokens[i,swap_position:tgt_tokens_len], torch.LongTensor((middle,)).to(tgt_tokens), tgt_tokens[i,:swap_position]])
    new_prev_output_tokens[i,:tgt_tokens_len+2] = torch.cat([tgt_tokens[i, tgt_tokens_len:tgt_tokens_len+1] , tgt_tokens[i,swap_position:tgt_tokens_len], torch.LongTensor((middle,)).to(tgt_tokens), tgt_tokens[i,:swap_position]])
tgt_tokens = new_tgt_tokens
print(tgt_tokens)
print(new_prev_output_tokens)
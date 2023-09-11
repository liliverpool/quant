from transformers import AutoTokenizer
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("/INPUT/lwb/k-emebedding/NORMAL/gpt2_tokenizer_bin13_H5")
vocab = tokenizer.get_vocab()

data = []
with open("/INPUT/lwb/k-emebedding/NORMAL/hybrid_data_train/5min/5min_lseq_total_FUT_filtered.txt",
          "r",
          encoding='gb2312') as infile:
    # 逐行读取并写入新的txt文件
    print(1)
    for line in infile:
        words = [w for w in line.strip().split(' ') if w not in vocab]
        print(len(data), words)
        data.append(words)
# print(data[-2:])

# with open("test.txt", "w", encoding='gb2312') as outfile:
#     for line in data:
#         outfile.write(line)

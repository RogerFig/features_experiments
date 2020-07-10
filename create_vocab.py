

vocab = []
with open('vocab/UserDictionary_pt.txt') as f:
     for linha in f.readlines():
         vocab.append(linha.strip().lower())

with open('vocab/DELAF_PB_v2/Delaf2015v04.dic', encoding="UTF-8-sig") as f:
    for linha in f.readlines():
        vocab.append(linha.strip().split(',')[0].lower())
        
vocab = set(vocab)
print(len(vocab))
with open('vocab/vocab.txt','w') as f:
    for item in vocab:
        f.write(item+'\n')

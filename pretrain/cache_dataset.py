from tokenizers import Tokenizer
from smart_open import open
from tqdm import tqdm


SEQ_LEN = 128 - 2
tokenizer = Tokenizer.from_file("../tokenizer.json")


documents = [[]]
for line in tqdm(open("../data/processed_dev/segmented.txt")):
    line = line.strip()

    if len(line) == 0:
        if len(documents[-1]) > 0:
            documents.append([])
        continue

    ids = tokenizer.encode(line, add_special_tokens=False).ids
    documents[-1].append(ids)


with open(f"../data/processed/cached_dev_{SEQ_LEN + 2}.txt", "w") as f:
    for document in tqdm(documents):
        segment = []
        for i, sentence in enumerate(document):
            segment += sentence

            if len(segment) > SEQ_LEN:
                segment = segment[:SEQ_LEN]
                subwords = [tokenizer.id_to_token(token_id) for token_id in segment]
                f.write(" ".join(subwords) + "\n")

                segment = [s for s in sentence]

        if len(segment) > 0:
            segment = segment[:SEQ_LEN]
            subwords = [tokenizer.id_to_token(token_id) for token_id in segment]
            f.write(" ".join(subwords) + "\n")

class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 4
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        self.word_to_id[self.pad_token] = 0
        self.id_to_word[0] = self.pad_token
        self.word_to_id[self.unk_token] = 1
        self.id_to_word[1] = self.unk_token
        self.word_to_id[self.bos_token] = 2
        self.id_to_word[2] = self.bos_token
        self.word_to_id[self.eos_token] = 3
        self.id_to_word[3] = self.eos_token
        vocab = set()
        for text in texts :
            for word in text.split(" "):
                normalized_word = word.lower().strip()
                if normalized_word not in vocab:
                    vocab.add(normalized_word)
                    self.vocab_size += 1
        list_vocab = sorted(list(vocab))
        for i, voc in enumerate(list_vocab,start=4):
            self.word_to_id[voc] = i
            self.id_to_word[i] = voc
                

    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        text_split = text.split()
        ids = []
        for element in text_split:
            ids.append(self.word_to_id.get(element.lower().strip(),1))
        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        elements = []
        for id in ids :
            elements.append(self.id_to_word.get(id,self.unk_token))
        return " ".join(elements)
        
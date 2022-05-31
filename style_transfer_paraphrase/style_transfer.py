from style_paraphrase.inference_utils import GPT2Generator


class PersonalityTransfer():
    def __init__(self, style_transfer_model_path, paraphrase_model_path,
                 generation_mode='nucleus_paraphrase', batch_size=32,
                 top_p=0.9, output_class=None, detokenize=True,
                 post_detokenize=True, lowercase=False, post_lowercase=False,
                 device='gpu'):
        self.detokenize = detokenize
        self.post_detokenize = post_detokenize
        self.post_lowercase = post_lowercase
        self.lowercase = lowercase
        self.batch_size = batch_size
        self.output_class = output_class
        self.paraphrase_model = GPT2Generator(
            paraphrase_model_path, upper_length="same_5", device=device
        )
        vec_data_dir = None  # FIXME os.path.dirname(os.path.dirname(args.input_file))
        self.style_transfer_model = GPT2Generator(style_transfer_model_path,
                                                  upper_length="same_10",
                                                  top_p=top_p,
                                                  data_dir=vec_data_dir,
                                                  device=device)
        self.change_mode(generation_mode, top_p)

    def change_mode(self, new_mode, top_p=0.9):
        self.mode = new_mode
        self.top_p = 0 if 'greedy' in new_mode or 'nucleus' not in new_mode else top_p
        self.style_transfer_model.modify_p(self.top_p)

    def paraprhase_wmodel(self, input_data):
        st_input_data = []
        for i in range(0, len(input_data), self.batch_size):
            st_input_data.extend(
                self.paraphrase_model.generate_batch(input_data[i:i + self.batch_size])[0]
            )
        return st_input_data

    def transfer_style(self, text):
        texts = text.strip().split('\n')
        if self.detokenize:
            texts = [PersonalityTransfer.detokenize(text) for text in texts]
        if self.lowercase:
            texts = [text.lower() for text in texts]

        if "paraphrase" in self.mode:
            texts = self.paraprhase_wmodel(texts)

        transferred_data = []
        if self.mode == "paraphrase":  # only paraphrase
            transferred_data = texts
        else:
            for i in range(0, len(texts), self.batch_size):
                if self.output_class is not None:
                    transferred_data.extend(
                        self.style_transfer_model.generate_batch(
                            contexts=texts[i:i + self.batch_size],
                            global_dense_features=[self.output_class for _ in texts[i:i + self.batch_size]]
                        )[0]
                    )
                else:
                    transferred_data.extend(
                        self.style_transfer_model.generate_batch(texts[i:i+self.batch_size])[0]
                    )
        if self.post_detokenize:
            transferred_data = [PersonalityTransfer.tokenize(x) for x in transferred_data]

        if self.post_lowercase:
            transferred_data = [x.lower() for x in transferred_data]

        transferred_data = [" ".join(x.split()) for x in transferred_data]
        return transferred_data

    @staticmethod
    def detokenize(x):
        x = x.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?").replace(" )", ")").replace("( ", "(")
        return x

    @staticmethod
    def tokenize(x):
        x = x.replace(".", " .").replace(",", " ,").replace("!", " !").replace("?", " ?").replace(")", " )").replace("(", "( ")
        return x

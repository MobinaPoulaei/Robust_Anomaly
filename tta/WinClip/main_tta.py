from WinCLIP import CLIPAD

if __name__ == "__main__":
    model, _, _ = CLIPAD.create_model_and_transforms(model_name=backbone, pretrained=pretrained_dataset, scales=scales,
                                                     precision=self.precision)
    tokenizer = CLIPAD.get_tokenizer(backbone)

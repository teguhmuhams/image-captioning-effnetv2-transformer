import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from inference import inference
import torchtext; torchtext.disable_torchtext_deprecation_warning()


def calculate_bleu(predicted, references):
    # Ensure that the punkt tokenizer is available
    nltk.download('punkt')

    # Tokenizing the sentences
    reference_tokens = [word_tokenize(ref) for ref in references]
    predicted_tokens = word_tokenize(predicted)

    # Use a smoothing function
    chencherry = SmoothingFunction()

    # Calculating BLEU scores for different n-gram lengths
    bleu_1 = sentence_bleu(reference_tokens, predicted_tokens, weights=(1, 0, 0, 0), smoothing_function=chencherry.method1)
    bleu_2 = sentence_bleu(reference_tokens, predicted_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method1)
    bleu_3 = sentence_bleu(reference_tokens, predicted_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=chencherry.method1)
    bleu_4 = sentence_bleu(reference_tokens, predicted_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)

    return {bleu_1, bleu_2, bleu_3, bleu_4}

def evaluate(data_root_path, test_df, model):
    print(model)
    print(test_df.iloc[0]['image_filename'])

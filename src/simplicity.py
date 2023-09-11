from nltk.tokenize import sent_tokenize
import readability


def count_newlines(text):
    return text.count("\n")


def count_sentences_nltk(text, language="german"):
    language = "german" if language == "de" else language
    return len(sent_tokenize(text, language=language))


def get_readability_grades(text, language="de"):
    return readability.getmeasures(text, lang=language)["readability grades"]

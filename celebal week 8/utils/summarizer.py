from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

summarizer = LexRankSummarizer()

def summarize_answer(question, docs):
    context = "\n".join(docs)
    parser = PlaintextParser.from_string(context, Tokenizer("english"))
    summary = summarizer(parser.document, sentences_count=3)

    return " ".join(str(sentence) for sentence in summary)

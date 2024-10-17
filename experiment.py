import pyterrier as pt
import os
import json
import re
import pandas as pd
import string

if not pt.started():
        pt.init()

META = {'docno' : 20, 'text': 10000}
FIELDS = ['text']

matchHTMLAndWhitespace = re.compile('\s+|<[^<>]+>')
matchHTMLTags = re.compile('<[^<>]+>')

symbolToTermMap = { '!' : ' exclamation ',
                    '+' : ' plus ',
                    '-' : ' minus ',
                    '*' : ' multiply ',
                    '/' : ' divides ',
                    '(' : ' lparenthesis ',
                    ')' : ' rparenthesis ',
                    '=' : ' equals ',
                    '.' : ' dot '
                   }

def basicPreprocessing(text):
    return text.translate(str.maketrans('', '', string.punctuation)).lower()

# Split the text based off of html tags, and join the substrings
# with a space between them.
def htmlPreprocessing(text):
    return ''.join([t.lower() for t in
                    matchHTMLTags.split(text) if
                    len(t) > 2])

def termSymbolProcessing(term):
    for symbol, symbolTerm in symbolToTermMap.items():
        term = term.replace(symbol, symbolTerm)
    return term

# Same as html preprocessing, but replace mathematical symbols with
# alphabetical tokens.
def htmlAndSymbolPreprocessing(text):
    return ''.join([' ' + termSymbolProcessing(t) for t in
                    matchHTMLAndWhitespace.split(text) if
                    len(t) > 2])

def main():
    def processDoc(doc, preprocessing):
        doc['docno'] = doc.pop('Id')
        doc['text'] = preprocessing(doc.pop('Text'))
        doc.pop('Score')
        
    processAnswers = lambda answers, p: [processDoc(d, p) for d in answers]
    
    basicAnswers = json.load(open('Puzzles/Answers.json', errors='ignore'))
    processAnswers(basicAnswers, basicPreprocessing)

    htmlAnswers = json.load(open('Puzzles/Answers.json', errors='ignore'))
    processAnswers(htmlAnswers, lambda t: basicPreprocessing( htmlPreprocessing(t) ))

    advAnswers = json.load(open('Puzzles/Answers.json', errors='ignore'))
    processAnswers(advAnswers, lambda t: basicPreprocessing( htmlAndSymbolPreprocessing(t) ))
        
    basicIndexer = pt.IterDictIndexer('./basicIndex',
                                      meta=META,
                                      overwrite=True,
                                      )
    htmlIndexer = pt.IterDictIndexer('./htmlIndex',
                                     meta=META,
                                     overwrite=True,
                                     )
    advIndexer = pt.IterDictIndexer('./advIndex',
                                    meta=META,
                                    overwrite=True,
                                    )

    basicRef = basicIndexer.index(basicAnswers)
    basicTFIDF = pt.terrier.Retriever(basicRef, wmodel="TF_IDF", num_results=100)
    basicBM25 = pt.terrier.Retriever(basicRef, wmodel="BM25", num_results=100)
    
    htmlRef = htmlIndexer.index(htmlAnswers)
    htmlTFIDF = pt.terrier.Retriever(htmlRef, wmodel="TF_IDF", num_results=100)
    htmlBM25 = pt.terrier.Retriever(htmlRef, wmodel="BM25", num_results=100)
    
    advRef = advIndexer.index(advAnswers)
    advTFIDF = pt.terrier.Retriever(advRef, wmodel="TF_IDF", num_results=100)
    advBM25 = pt.terrier.Retriever(advRef, wmodel="BM25", num_results=100)

    topics1 = json.load(open('Puzzles/topics_1.json', errors='ignore'))

    joinQuery = lambda q: ''.join([q['Title'], ' ', q['Body']])
    basicGetText = lambda q: basicPreprocessing( joinQuery(q) )
    htmlGetText = lambda q: basicPreprocessing( htmlPreprocessing( joinQuery(q) ))
    advGetText = lambda q: basicPreprocessing( htmlAndSymbolPreprocessing( joinQuery(q) ))
    getQueries = lambda getText: pd.DataFrame([[query['Id'], getText(query)]
                                               for query in topics1],
                                              columns=['qid', 'query'])
    
    basicQueries = getQueries(basicGetText)
    htmlQueries = getQueries(htmlGetText)
    advQueries = getQueries(advGetText)
    
    def runExperiment(queries, perq=False):
        return pt.Experiment(
                [basicTFIDF, basicBM25,
                 htmlTFIDF, htmlBM25,
                 advTFIDF, advBM25],
                queries,
                pt.io.read_qrels('Puzzles/qrel_1.tsv'),
                eval_metrics=['P.5', 'P.10', 'P.100',
                              'ndcg_cut.5', 'ndcg_cut.10', 'ndcg_cut.100',
                              'map', 'recip_rank'],
                round=4,
                names=['Basic TFIDF', 'Basic BM25',
                       'HTML TFIDF', 'HTML BM25',
                       'Advanced TFIDF', 'Advanced BM25'],
                verbose=True,
                perquery=perq)

    #Mean Results
    runExperiment(basicQueries).to_csv('basicResults.csv')
    runExperiment(htmlQueries).to_csv('htmlResults.csv')
    runExperiment(advQueries).to_csv('advResults.csv')

    #Per-Query Results
    runExperiment(basicQueries, True).to_csv('perqBasicResults.csv')
    runExperiment(htmlQueries, True).to_csv('perqHTMLResults.csv')
    runExperiment(advQueries, True).to_csv('perqAdvResults.csv')
        
if __name__ == '__main__':
    main()

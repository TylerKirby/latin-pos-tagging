import glob

def read_file(filename):
    infile = open(filename)
    contents = infile.read()
    infile.close()
    return contents

def reformat(text):
    
    split_sentences = text.split('\n')
    split_sentences_list = []
    
    for sentence in split_sentences:
        sentence_list = []
        sentence_list.append(sentence)
        split_sentences_list.append(sentence_list)
    
    pairs_list = []
    
    for sentence in split_sentences_list:
        broken_sentence = sentence[0].split()
        new_sentence_list = []
        for word in broken_sentence:
            if '/' in word and word != '/':
                split_word = word.split('/')
                word_pair = (split_word[0], split_word[1])
                new_sentence_list.append(word_pair)
            else:
                print(word)
                continue
        pairs_list.append(new_sentence_list)
    
    return pairs_list

paths = glob.glob('/Users/Elena/Documents/HDW-2019/pos-tagging/proiel-txt/*.txt')
for p in paths:
    shortened = p.split('proiel-txt/')[-1]
    file_name = shortened[:-4] + '_reformat.txt'
    print(file_name)
    f = open('/Users/Elena/Documents/HDW-2019/pos-tagging/proiel-reformat'+ '/' + file_name, 'w', encoding = 'utf-8')
    reformatted = reformat(read_file(p))
    f.write(str(reformatted))
    f.close
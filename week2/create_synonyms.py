import fasttext

model_file = '/workspace/datasets/fasttext/title_model_100.bin'
top_words_file = '/workspace/datasets/fasttext/top_words.txt'
synonyms_file = '/workspace/datasets/fasttext/synonyms.csv'
threshold = 0.75

model = fasttext.load_model(model_file)

with open(synonyms_file, 'w') as output_file:
    with open(top_words_file, 'r') as top_words:
        for word in top_words.readlines():
            word = word.rstrip('\n')
            synonyms = model.get_nearest_neighbors(word)
            output_list = [word]
            for synonym_pair in synonyms:
                score = synonym_pair[0]
                synonym = synonym_pair[1]
                if (synonym != word and score >= threshold):
                    output_list.append(synonym)
            if len(output_list) > 1:
                output_line = ','.join(output_list) + '\n'
                output_file.write(output_line)
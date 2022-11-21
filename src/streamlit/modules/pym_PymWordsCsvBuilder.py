class PymWordsCsvBuilder:

    SEPARATOR = ';'

    def __init__(self):
        return;    
            
    def build_words(self, letters_only=True):
        input_path = '/Users/sophieescaich/Documents/Github/Pymano/data/words.txt';
        output_path = '/Users/sophieescaich/Documents/Github/Pymano/data/words.csv';
        img_path = '/Users/sophieescaich/Documents/DATA/OCR/Données/words/'
        
        with open(output_path, 'w') as  output_f:
            with open(input_path) as input_f: #dossier où se trouve le fichier txt
                lines = input_f.readlines()
            headers = ["word_id","word_seg","graylevel","bounding_box_x","bounding_box_y","bounding_box_w","bounding_box_h", "gram_tag", "transcription_word","path\n"]
            header = self.SEPARATOR.join(headers)
            output_f.write(header)
            regex = re.compile("^[A-Za-z]*$")

            for line in lines:
                if (line[0] != '#'):
                    split_line = line.strip().split(' ')
                    text = ' '.join(split_line[8:])
                    if(not(letters_only) or regex.match(text)):                        
                        filenameSplit = split_line[0].split('-')
                        filepath = img_path + filenameSplit[0] + "/" + filenameSplit[0] + "-" + filenameSplit[1] + "/" + split_line[0] + ".png"
                        final_line = split_line[0] + self.SEPARATOR + split_line[1] + self.SEPARATOR + split_line[2] \
                        + self.SEPARATOR + split_line[3] + self.SEPARATOR + split_line[4] + self.SEPARATOR \
                        + split_line[5] + self.SEPARATOR + split_line[6] + self.SEPARATOR + split_line[7] \
                        + self.SEPARATOR + text + self.SEPARATOR + filepath +"\n"
                        output_f.write(final_line)

            input_f.close()
            output_f.close()
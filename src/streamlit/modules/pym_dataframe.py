import pandas as pd
import re
import tensorflow as tf
from os.path import exists
import csv

csv_path = './data/words.csv';
        
def get_clean_dataframe(force_csv_rebuild = False, force_img_check = False, lowercase_only = False, remove_space=True, separator = '\t'):
    
    if(force_csv_rebuild or not(exists(csv_path))):
        build_words_csv()
    
    # As we have double quote in our csv, tell pandas that they are not the beginning of a new string but a literal double quote
    df = pd.read_csv(csv_path, index_col=0, sep=separator, quoting = csv.QUOTE_NONE)
    
     # keep only rows with correct segmentation
    df = df[df["word_seg"] == "ok"]
    
    if(lowercase_only):
        df = df[df["transcription_word"].str.match("^[a-z]*$")]
    if(remove_space):
        df = df[df["transcription_word"].str.match("^\S+$")]
    
     # remove images in error
    if force_img_check:
        img_path_in_errors = get_imgs_in_error(df)
    else:
        # We don't want to check images in errors everytime, so here is the list
        img_path_in_errors = ['./images/words/a01/a01-117/a01-117-05-02.png','./images/words/r06/r06-022/r06-022-03-05.png']        
    df = df[~df["path"].isin(img_path_in_errors)]
    
     # take only wanted columns
    df = df[["transcription_word", "path"]]
    
    return df

def get_test_dataframe():
    df = pd.read_csv('./images/test/test.csv', index_col=0, sep='\t', quoting = csv.QUOTE_NONE)
    return df    

def build_words_csv(separator = '\t'):
    input_path = './data/words.txt';    
    img_path = './images/words/'

    with open(csv_path, 'w') as  output_f:
        with open(input_path) as input_f:
            lines = input_f.readlines()
        headers = ["word_id","word_seg","graylevel","bounding_box_x","bounding_box_y","bounding_box_w","bounding_box_h", "gram_tag", "transcription_word","path\n"]
        header = separator.join(headers)
        output_f.write(header)

        for line in lines:
            if (line[0] != '#'):
                split_line = line.strip().split(' ')
                text = ' '.join(split_line[8:])                
                filenameSplit = split_line[0].split('-')

                # word_id = a01-000u-00-00 => path = ./images/words/a01/a01-000u/a01-000u-00-00.png 
                filepath = img_path + filenameSplit[0] + "/" + filenameSplit[0] + "-" + filenameSplit[1] + "/" + split_line[0] + ".png"

                final_line = split_line[0] + separator + split_line[1] + separator + split_line[2] \
                + separator + split_line[3] + separator + split_line[4] + separator \
                + split_line[5] + separator + split_line[6] + separator + split_line[7] \
                + separator + text + separator + filepath +"\n"

                output_f.write(final_line)

        input_f.close()
        output_f.close()

def get_imgs_in_error(dataframe):    
    
    file_path_list = dataframe['path'].tolist()
    
    # Split the list into chuncks to increase loop performance
    chunks = [set(file_path_list[x:x+1000]) for x in range(0, len(file_path_list), 1000)]
    
    imgs_in_error = []
    for chunk in chunks:
        for filepath in chunk:
            error = False
            
            try:
                # Try to read the image. We use tf functions because it's the ones used in the preprocessing
                im = tf.io.read_file(filepath)
                im = tf.image.decode_png(im, channels=0)
            except:
                error = True

            # If the image was not read properly or if it is an empty one, add the path to the list
            if (im is None) or error:
                imgs_in_error.append(filepath)
    return imgs_in_error
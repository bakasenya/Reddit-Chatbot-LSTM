import sys
import pandas as pd
import re
import string

print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")

# Read input files
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

Lines_to = [line.strip() for line in read_file('to.txt')]
Lines_from = [line.strip() for line in read_file('from.txt')]

# Function to find URLs in text
def find_urls(text):
    return re.findall(r'https?://\S+|www\.\S+', text)

# Remove lines with URLs or too long text
def remove_url_and_long_text(data1, data2):
    filtered_from = []
    filtered_to = []
    
    for line_from, line_to in zip(data1, data2):
        if len(line_from) > 300 or len(line_to) > 300:
            continue
        
        if find_urls(line_from) or find_urls(line_to):
            continue
        
        filtered_from.append(line_from)
        filtered_to.append(line_to)
    
    return filtered_from, filtered_to

fro, to = remove_url_and_long_text(Lines_from, Lines_to)

# Create a DataFrame from the filtered data
df = pd.DataFrame({'from': fro, 'to': to})

# Text cleaning functions
def sort_clean(text):
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"we're", "we are", text)
    text = re.sub(r"i'd", "i would", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"[-()#/@;:<>{}`+=~|.!?,]", "", text)
    return text

def remove_url(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def remove_html(text):
    return re.sub(r'<.*?>', '', text)

def remove_emoji(text):
    emoji_pattern = re.compile("[" 
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_punct(text):
    return ''.join(ch for ch in text if ch not in string.punctuation)

def remove_newline_char(text):
    return text.replace('newlinechar', '')

def remove_numbers_out_of_range(text):
    words = [word for word in text.split() if word.isalpha()]
    text = ' '.join(words)
    return text if 3 <= len(text.split()) <= 25 else None

# Apply all cleaning functions
def clean_text(text):
    text = text.apply(lambda x: x.lower())
    text = text.apply(lambda x: sort_clean(x))
    text = text.apply(lambda x: remove_emoji(x))
    text = text.apply(lambda x: remove_html(x))
    text = text.apply(lambda x: remove_punct(x))
    text = text.apply(lambda x: remove_newline_char(x))
    text = text.apply(lambda x: remove_numbers_out_of_range(x))
    text = text.replace('\s+', ' ', regex=True)
    return text

# Clean the DataFrame columns
df['from'] = clean_text(df['from'])
df['to'] = clean_text(df['to'])

# Drop rows with NaN values
df.dropna(inplace=True)

# Save the cleaned data to a CSV file
df.to_csv('cleaned_data.csv', sep=';', index=False)

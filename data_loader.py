import os
import requests
import re

def download_data(data_dir='data'):

    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, 'tiny_shakespeare.txt')
    
    if not os.path.exists(file_path):
        try:
            url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

            print("Downloading Data")

            response = requests.get(url)
          
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(response.text)

            print("Data downloaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Download failed: {e}")

    
    return file_path
   
def load_and_preprocess():

    data = download_data()
    with open(data, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Data cleaning
    text = re.sub(r'[^\w\s\']', ' ', text)  # Remove special chars
   
    text = re.sub(r'\s+', ' ', text).strip()  # Remove whitespace

    tokens = text.lower().split() 
    unique_tokens = set(tokens)
    
    # print(f"Total tokens: {len(tokens)}")
    # print(f"Total unique tokens: {len(unique_tokens)}")
    # print(unique_tokens)

    return text



# load_and_preprocess()

# Result:-

# (venv) D:\Water\ChatGPT\group-9\Transformer_Pretraining>python data_loader.py
# Total tokens: 204089  
# Total unique tokens: 12632

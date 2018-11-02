import requests
import os.path
from os import path

file_url = "http://codex.cs.yale.edu/avi/db-book/db4/slide-dir/ch1-2.pdf"

def download_file(url, file_name):

    r = requests.get(url, stream=True)

    bool = path.exists('./' + file_name)
    if not bool:
        with open(file_name, "wb") as pdf:
            for chunk in r.iter_content(chunk_size=1024):

                # writing one chunk at a time to pdf file
                if chunk:
                    pdf.write(chunk)
    else:
        print(file_name + ' already exists')
    return file_name


#download_file(file_url, file_name='python.pdf')
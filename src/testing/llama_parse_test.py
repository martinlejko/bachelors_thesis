from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

parser = LlamaParse(
    api_key="llx-94uSiDnHtK5jIsliA0BYiXh4CclREU729ewfNxbqTWtW5dsf",
    result_type="markdown"
)

file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(input_files=['/Users/martinlejko/desktop/tabulka2.pdf'], file_extractor=file_extractor).load_data()
print(documents)
with open('/Users/martinlejko/Repos/github.com/martinlejko/bachelors_thesis/src/tabluka2.md', 'w') as f:
    f.write(documents[0].text)
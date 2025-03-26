# Contextual Retrieval Augmented Generation Example


## Build with

The project uses:
* [Python](https://www.python.org/)
* [LangChain](https://www.langchain.com/)
* [Okapi BM25](https://fr.wikipedia.org/wiki/Okapi_BM25)
* [FAISS](https://github.com/facebookresearch/faiss)
* [Openai API](https://platform.openai.com/)
* [Anthropic API](https://console.anthropic.com/)
* [GoogleGenerativeAI API](https://aistudio.google.com/)
* [Architecture example](https://www.anthropic.com/news/contextual-retrieval)


## Setup

### Prerequisites
- Python 3.8+
- pip (Python package manager)


### Installation

1. Clone this repository:
```bash
git clone https://github.com/Wizo17/generative_cag_application.git
cd generative_cag_application
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# Unix / MacOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```
<em>If you have some issues, use python 3.12.0 and requirements_all.txt</em>


5. Create .env file:
```bash
cp .env.example .env
```

6. **Update .env file**


## Running app

#### Build index or test in console
```bash
python main.py
```

#### Launch chatbot example
By launching streamlit, you can have competitive conflicts with `torch`. Not a problem for this version.
```bash
streamlit run chatbot.py
```


## Authors

* [@wizo17](https://github.com/Wizo17)

## License

This project is licensed under the ``MIT`` License - see [LICENSE](LICENSE.md) for more information.

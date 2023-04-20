# Master's Thesis
Repo for my Master's thesis project in spring 2022, titled: Entity-Related Multi-Document Summarization.

## Setup

The python version I used to run the code is 3.7.9. The reason for using that specific version was that I initially used using neuralcoref for coreference resolution, which required specific python versions. However, that package was later scrapped, and I have tested it with Python version 3.10.4 (newest version at the time of writing) on another PC and it works fine!

	1. Download python: https://www.python.org/downloads/release/python-379/
		a. Make sure python is added to Path
	2. Download the project - Zip or git
	3. Open a terminal where you installed the project
	4. Make sure you have python installed by running "python --version".
		a. If that doesn't give a Python 3.x.x version, try running "python3 --version". If that works, use "python3" instead of "python" and "pip3" instead of "pip" for the rest of the setup.
	5. Create a virtual environment by running "python -m venv .env"
		a. You might need to run "pip install virtualenv" first
	6. Activate the virtual environment
		a. Mac/linux: source .env/bin/activate
		b. Windows: .env\Scripts\activate
	7. Install requirements
		a. Upgrade pip: python -m pip install --upgrade pip
		b. Install requirements: pip install -r requirements.txt
			• NB: Some packages (like Spacy) might require
	8. Spacy specifics:
		a. python -m spacy download en_core_web_sm
	9. To run certain parts of the code you will have to download some transformer models, as well as the CNN/Dailymail dataset if you want to run the tests. Everything will be downloaded automatically when you run the code, but just expect it to take some time. 


## The main files to run
After you have completed the setup, the main files are found in ./src/extractive/main.py or ./src/abstractive/main.py, depending on which system you want to test. To generate a summary from the working dataset, assign a desired value to the _cluster_id_ parameter, and run the file.

All results from the test on CNN/DM can be seen by running ./src/extractive/eval_stats.py or ./src/abstractive/eval_stats.py.

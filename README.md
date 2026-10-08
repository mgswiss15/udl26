# Unsupervised Deep Learning - lab repository

Materials for the hands-on part of the course *Unsupervised Deep Learning* (Master in AI, THWS).  
**Lecturer:** Magda Gregorova (magda.gregorova@thws.de).

## What is this repository?

Throughout the course, there will be multiple **labs** where you implement models discussed in the lecture yourself and test them on a small dataset.  
This repo contains all you need for the labs:

- one folder per lab (`lab01/`, `lab02/`, ...) with the lab notebook and scripts
- a helper folder `udl/` with code we provide, so that you can concentrate on the models

New labs are added to this repository as the course goes on.

## Structure of repo

```
requirements.txt     Python packages needed for the labs
udl/                 provided helpers: datasets, densities, plotting, checks - you do not edit these
lab01/
    lab.ipynb        the lab notebook: instructions, checks, plots, evaluation
    histogram.py     scripts with the functions YOU write (marked with TODO)
    gaussian.py
    mixture.py
    pixels.py
lab02/ ...          the same structure for every following lab
```

## Setup

1. Fork the repository on github
2. Clone it to your computer `git clone <repository URL>` (or download it as a zip file).
3. Create a Python environment in whatever way you prefer (venv, conda, ...) and install the packages in `requirements.txt`.  
   For PyTorch you may want to use the specific installation corresponding to your GPU situation and CUDA version.  
   Nothing else is installed - the folder `udl/` and the lab folders are used directly from the repository.
3. Start Jupyter (or open the notebook in VS Code) **from the root folder of the repository**,
   then open `lab01/lab.ipynb`.

The early labs will run on a laptop CPU but you will need GPU later. 
You can use Google Colab or request access to the student GPU server by sending me an email.

## How to work on a lab

1. Read the instructions in the notebook, from top to bottom.
2. **Write your code in the scripts** (`labNN/*.py`), not in the notebook. Every function has a docstring
   that says what goes in and what comes out; replace each `# TODO` with your code.  
   The notebook is only for presenting: it imports your functions, runs checks, draws plots and evaluates your models.
3. Save the script and rerun the notebook cell - your changes are loaded automatically.
4. Run the `check(...)` lines. A passing check does not prove that your code is correct but catches major errors.
   Always look at the plots and other outputs too.
5. Answer the written questions in the cells marked **Answer** in the notebook.

Work step by step: the parts of a lab are designed so that an unfinished part does not stop you from doing the others.

## Getting new labs

New material appears in the repository during the semester - sync the fork.
Your own code lives in the files you edit, so to avoid conflicts **keep a copy of your finished scripts 
in a branch** and do not change files that you did not have to edit.

## Goals and ground rules
 
This is an **elective course** - nobody has to take it. If you decide to take it, the goal is that you
**understand the main generative models well enough to implement them yourself**, at a small scale, and to
judge what they do and where they fail. The lectures give you the ideas. The labs are where you really learn them,
and that only works if you do the work yourself.
 
### Recommendations
- **Do the labs alone.** You can discuss with friends but try to work on your own to really learn.
- **Switch off code assistants** such as Copilot (code completion and code generation) while you work on a lab.
- **Do not copy and paste code** you find somewhere - from the web, other repositories, colleagues or AI answers.
- Feel free to search, read documentation, papers and books, and to chat with an AI to understand a concept,
  a formula or an error message. **But the final implementation shall be yours**: written by you that you fully understand every line of it.
 
Why so strict? Code you did not write but only read and used gives you the feeling that you understand it. That feeling is wrong:
it only shows when you have to produce something similar from scratch - in the project, at the exam, or later
in your work. The labs are small on purpose, so that you can afford to do them properly.
A simple test: **if you could not write your solution again tomorrow without looking at it, you have not learned it yet.**

## Questions

Ask in the lab sessions or write to magda.gregorova@thws.de.

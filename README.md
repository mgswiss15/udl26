# Unsupervised Deep Learning - lab repository

Materials for the hands-on part of the course *Unsupervised Deep Learning* (Master in AI, THWS).
Lecturer: Magda Gregorova (magda.gregorova@thws.de).

## What is this repository?

Every week of the course has a **lab**: you implement the models from the lecture yourself, run them on
small datasets, look at what they generate and evaluate how good they are. This repository contains
everything you need for the labs:

- one folder per week (`week01/`, `week02/`, ...) with the lab notebook and the scripts you work in
- a helper folder `udl/` with code we provide, so that you can concentrate on the models
- the lecture slides and lecture notes as PDF files

The labs are done **individually**. New weeks are added to this repository as the course goes on.

## Goals and ground rules

This is an **elective course** - nobody has to take it. If you decide to take it, the goal is that you
**understand the main generative models well enough to implement them yourself**, at a small scale, and to
judge what they do and where they fail. The lectures give you the ideas. The labs are where you really learn them,
and that only works if you do the work yourself:

- **Do the labs alone.**
- **Switch off code assistants** such as Copilot (code completion and code generation) while you work on a lab.
- **Do not copy and paste code** you find somewhere - from the web, other repositories, colleagues or AI answers.
- You are free to search, read documentation, papers and books, and to chat with an AI to understand a concept,
  a formula or an error message. **But the final implementation must be yours**: written by you, and you must
  be able to explain every line.

Why so strict? Code you did not write yourself gives you the feeling that you understand it. That feeling is wrong:
it only shows when you have to produce something similar from scratch - in the project, at the exam, or later
in your work. The labs are small on purpose, so that you can afford to do them properly.
A simple test: if you could not write your solution again tomorrow without looking at it, you have not learned it yet.

## What is in it?

```
requirements.txt     Python packages needed for the labs
udl/                 provided helpers: datasets, densities, plotting, checks - you do not edit these
week01/
    lab.ipynb        the lab notebook: instructions, checks, plots, evaluation
    histogram.py     scripts with the functions YOU write (marked with TODO)
    gaussian.py
    mixture.py
    pixels.py
    slides.pdf       slides of the week
week02/ ...          the same structure for every following week
LectureNotes.pdf     lecture notes
```

## Setup

1. Get the repository: `git clone <repository URL>` (or download it as a zip file).
2. Create a Python environment in whatever way you prefer (venv, conda, ...) and install the packages:

   ```
   pip install -r requirements.txt
   ```

   Nothing else is installed - the folder `udl/` and the week folders are used directly from the repository.
3. Start Jupyter (or open the notebook in VS Code) **from the root folder of the repository**,
   then open `week01/lab.ipynb`.

The labs of the first weeks run on a laptop CPU. You can also use Google Colab: open the notebook there
and run its first code cell, which fetches this repository automatically.

## How to work on a lab

1. Read the instructions in the notebook, from top to bottom.
2. **Write your code in the scripts** (`weekNN/*.py`), not in the notebook. Every function has a docstring
   that says what goes in and what comes out; replace each `# TODO` with your code.
   The notebook is only for presenting: it imports your functions, runs checks, draws plots and evaluates your models.
3. Save the script and rerun the notebook cell - your changes are loaded automatically.
4. Run the `check(...)` lines. A passing check does not prove that your code is correct, so always look at the plots too.
5. Answer the written questions in the cells marked **Answer** in the notebook.

Work step by step: the parts of a lab are designed so that an unfinished part does not stop you from doing the others.

## Getting new weeks

New material appears in the repository during the semester. Update with `git pull`.
Your own code lives in the files you edit, so to avoid conflicts **keep a copy of your finished scripts
elsewhere before you pull** (or work in your own git branch), and do not change files that you did not have to edit.

## Questions

Ask in the lab sessions or write to magda.gregorova@thws.de.

# 11697 HW6: Develop a RAG pipeline

***Note: This is an individual work (See below for exceptions).***

# Timeline

* **Part 0 (deadline): 10/31 11:59 PM (EDT) / 11/1 5:59 AM (CAT)**
* Part 1 (recommend): 11/7 11:59 PM (EDT) / 11/8 5:59 (CAT)  
* Part 2 (recommend): 11/14 11:59 PM (EDT) / 11/15 5:59 (CAT)
* Part 3 & 4 (recommend): 11/21 11:59 PM (EDT) / 11/22 5:59 (CAT)
* Part 5 (recommend): 11/25 11:59 PM (EDT) / 11/26 5:59 (CAT)
* Part 6 (recommend): 12/3 11:59 PM (EDT) / 12/4 5:59 (CAT)
* **All (deadline): 12/5 11:59 PM (EDT) / 12/6 5:59 (CAT)**

# Repository structure
This is a sample expected repository structure.
```
11697-hw6-<your andrewID>
├── README.md
├── data
│   ├── answer.tsv
│   ├── evidence.tsv
│   └── question.tsv
├── notebook  # optional
│   └── example_notebook.ipynb  
├── output
│   ├── evaluation
│   │   ├── ...
│   │   └── xxx_yyy.tsv
│   └── prediction
│       ├── ...
│       └── xxx_yyy.tsv
├── src
│   ├── ...
│   └── example_rag.py
└── topic.txt
```

# Instruction

## Part 0: GitHub repo creation. 
1. Create a private GitHub repo, naming `11697-hw6-<andrewID>`  
2. Give access to instructors and TAs: `EricNyberg`, `teruko`, `Leonard250`, `kimihiroh`  
3. Submit the URL for your GitHub repo on Canvas

## Part 1: Topic selection
Select your own topic.   

### Requirements:
* While you can use the same topic you chose for HW4, you cannot choose the following topics: topics we have already used as examples (CMU, Prof. Eric Nyberg, Pittsburgh, etc), reasoning-intensive topics (Math), etc.  
* Please check with us in advance if you are not sure if your chosen topic is suitable for this assignment. 

### Tips: 
* Think of a kind of real-life problem that your system would be solving.  
* A good topic is one that you can easily create questions that would require retrieval.   
* A good topic is broad enough so that you can have like a thousand questions about it, which are not easily answerable by the model itself without a RAG system.

### Deliverable: 
Create a `topic.txt` file, and include your topic.   

### Format: 
`topic.txt`
```txt
<topic: a few words/sentences>
```
### Ref: 
HW4

## Part 2: QA curation & Document collection 
Curate a set of at least 100 QA pairs and 50 supporting documents. Your main target questions are the ones that models cannot answer without retrieval.   
You can manually create all of them or synthesize them with LLMs.   

### Requirement: 
* Include the following question types at least 20 QA pairs for each: multiple choice, factoid (words/short phrase), list (multiple possible answers/instructions)  
* Try to create questions that cannot be answered without retrieval, as much as possible.  
* Create a diverse set of questions so that they need as many distinct documents as possible in your document collection.

### Deliverable: 
Create a data folder and create files, `question.tsv`, `answer.tsv`, `evidence.tsv`. 

### Format: 
question.tsv (one question per line)
```
<question 1>\t<question type>
```
answer.tsv (comma-separated answers, each line for the corresponding question)
```
<anwer 1-1>\t<answer 1-2>\t...
```
evidence.tsv
```
<original url 1>\t<filename 1>
```

### Tips: 
* Draft them in Excel/spreadsheet and download as `.tsv`.

### Ref: 
HW 4, In-class exercise (2025/10/08)

## Part 3: Develop Retrieval \+ Generator
Develop one no-retrieval QA system and RAG-based QA systems. Make sure to tune your prompts so that LLMs generate answers in the expected format.  

### Requirements
* At least 1 no-retrieval system  
* At least 4 RAG systems
  * 2 variations of retrieval parts: both API-based and open-weight model  
  * 2 variations of the generation parts: both API-based and open-weight model  
  * 2 x 2 \= 4 combinations
* Make sure that all systems can be run with command lines, e.g., `bash rag.py --retriever XXX --generator YYY` 

### Deliverable: 
* `.py` file(s) under `src` folder and execution commands  
* `<retriever/None>_<generator>.tsv` for each system prediction (at least 5 files) under `output/prediction`

### Format: 
`<retriever/None>_<generator>.tsv`
```
<prediction>\t<additional info: optional>
```

### Ref: 
HW1, HW3, HW4, HW5

## Part 4: Evaluation
Evaluate the quality of the generated output.  

### Requirements:
* At least two metrics: one is LLM-as-a-judge, and the other is your choice
* Make sure that all systems can be run with command lines, e.g., `bash evaluate.py --file XXX`  

### Deliverable: 
* `.py` file under `src` folder and execution commands
* `<retriever/None>_<generator>.tsv` for each eval result under `output/evaluation`

### Format: 
`<retriever/None>_<generator>.tsv` 
```
<score by metric 1>\t<score by metric 2>\t...
```

### Ref: 
HW2, HW5

## Part 5: Advanced
You can explore any part of this assignment based on your interest. Just as a starting point, we share some possible ideas below:

### Possible ideas
* Extensive analysis   
  * Why models with RAG do not work: Retriever? Generator? Format? Eval Metrics? etc…  
  * Include noisy evidence to see how good your retriever is.  
  * More eval metrics & calibration of eval metrics (eg, IAA with human evaluation)  
  * More combinations for the RAG system
  * Visualization 
* QA quality analysis  
  * E.g., nugget voting with other students, IAA calculation  
  * Ref: In-class exercise (2025/09/10)
  * Note: You can collaborate with other students if you do this. 
* Upgrade your system  
  * Integrate external search tools to handle queries that fall outside the RAG’s knowledge context — for example, automatically performing a Google search instead of relying solely on the fixed dataset (Ref: TavilySearch).  
  * Integrate systems capable of providing real-time data responses — such as current weather, temperature, and other live information — by connecting to external APIs and enabling dynamic query handling within the platform.  
* Synthetic QA generation (+ verification)
* How calibrated are model predictions? (eg, analyzing confidence score)
* Make a UI for a chat interface  
* etc…

### Deliverable: 
Description in your report and corresponding output files if any. 

### Format: 
N/A (As long as we understand how to replicate, that is ok)

### Note 
* This is not a comprehensive list. Other directions can satisfy this aspect. If you are not sure about your idea, feel free to reach out to us.   
* You can focus on one direction. Or, you can choose multiple directions.   
* You can collaborate with other students only for this part.

## Part 6: Report
Describe your work in a report.

### Format & Requirements
* PDF
* Use [ARR Format](https://github.com/acl-org/acl-style-files)
* 4-8 pages \+ unlimited reference & appendix
* Make sure to have the following sections:
  * Sec 1 Introduction (incl. Topic and your GitHub repo)
  * Sec 2 Dataset (incl. QA/Document)
  * Sec 3 Method (e.g., Retrieval+generator)
  * Sec 4 Experiment (incl. Eval)
  * Sec 5 Result & discussion 
  * Sec 6 Qualitative analysis (i.e., pick a few actual representative cases and analyze them)

## Submission
Submit your GitHub repo URL and your report on Canvas.

## Grading Rubric

* System development: 5 pts  
  * GitHub repo: 1  
  * topic: 1  
  * QA/Evidence: 1  
  * system: 1  
  * evaluation: 1  
* Report: 6 pts  
  * Sec 1 Introduction (incl. Topic and your GitHub repo): 1  
  * Sec 2 Dataset (incl. QA/Document): 1  
  * Sec 3 Method (e.g., Retrieval+generator): 1  
  * Sec 4 Experiment (incl. Eval): 1
  * Sec 5 Result & discussion: 1  
  * Sec 6 Qualitative error analysis: 1  
* Advanced point: 5 pts (subjective & relative)  
  * Impressing to us: 5  
  * Reasonable amount of additional work: 3  
  * Some trivial prompt engineering: 1  
  * In the report, you can add another section, or you can add subsections to the corresponding ones for your additional explorations. 

## Note: 
* We assume you have both API credits and AWS credits. Contact us as soon as possible if you do not have them.   
* Make sure to push your updates into the GitHub repo.  
* We may contact you based on the expected checkpoint dates for each part to encourage steady progress. 

## Acknowledgments:
This assignment is inspired by HW2 in 11711 Advanced NLP \[[ref](https://github.com/neubig/nlp-from-scratch-assignment-spring2024/)\]

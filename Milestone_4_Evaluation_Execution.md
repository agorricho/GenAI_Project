# Milestone 4 Evaluation Execution 
    Execution of the Milestone 4 Validation and Evaluation instructions for the GenAI Project. I have provided the instructions to my team . Unfortunately, I do not trust they'll be able to fill out the csv table in time. So we'll ahve to do it ourselves. Please develop quick solutution with tech writer agent to write the 47 missing papers and entries into the desired format for the execution of evaluation for Milestone 4. The requirements consist of these steps.

1. Copy Paste each unique abstract into the the Reference_answer column of the eval_template.csv
2. Copy paste each unqiue correpsonding title 
3. Execute Party Mode session with bmad-agent-sm[Bob], bmad-agent-analyst[Mary] and bmad-agent-tech-writer to fill in no deterministic columns using an advanced LLM. tHE COLUMNS ARE
    Question
    Generation_mode
    Difficulty
    Esg_category
    Is_in_corpus

## References 
Documentation:/home/agorricho1/AI_Scientist2/GenAI_Project/PROJECT_DOCUMENTATION.md
    Contains the documentation and code specifications of the project
ReadMe: /home/agorricho1/AI_Scientist2/GenAI_Project/README.md
    Contains the ReadMe for overall set up of the proiject, dependencies, how to run the Gen AI DAIS Pipeline
Validation Instructions: /home/agorricho1/AI_Scientist2/GenAI_Project/Validation_Instructions.md
    Contains instructions of how run the Milestone 4 Validations, to the letter
Unique Abstract:/home/agorricho1/AI_Scientist2/GenAI_Project/validation_chunks.json
    Json file containing de-duplicated papers and abstracts with metadata keys. Organize by their index key value entry. abstracts are stored under abstract key. Title is stored under title key
Evaluation Template:
    /home/agorricho1/AI_Scientist2/GenAI_Project/Milestone3/eval/eval_template.csv
    Format from which 


## Workflow

### 1. Generate quick plan to copy past the abstract from the validation chunks into the eval template.
 I am unfaamiliar with how to trigger this workflow with the tech writer. If we where implementing a coding solution, the equivalent coding command is bmad quick dev. Please provide relevant command for this task. Generate Plan, Then Execute
### 2. Commuicate PLan to Dev Agent Amelia via [PM] party mode to execute bmad-quick-dev workflow of solution
Generate plan and execut via bmad-quickdev
### 3. Write Non Deterministic columns with advanced LLM FOR THE judgements of abstracts 
Use LLM Claude Opus 4.7 Fill non deterministic columns

## GUARDRAILS
1. Respect existing code as much as possible. no need to change the working of the GenAI pipeline 
2. Respect Instruction on how to fill in the eval_template.csv poutline by /home/agorricho1/AI_Scientist2/GenAI_Project/Validation_Instructions.md to the letter



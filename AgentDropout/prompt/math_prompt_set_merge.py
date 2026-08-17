from typing import Dict, Any
import itertools
from AgentDropout.prompt.prompt_set import PromptSet
from AgentDropout.prompt.prompt_set_registry import PromptSetRegistry
from AgentDropout.prompt.common import get_combine_materials

# [Modified] roles 리스트를 4개에서 15개로 확장
roles = itertools.cycle([
    'Math Solver',           # [Existing]
    'Mathematical Analyst',  # [Existing]
    'Programming Expert',    # [Existing]
    'Inspector',             # [Existing]
    'Problem Decomposer',    # [New] Divide and Conquer strategy
    'Pattern Recognizer',    # [New] Analogical Reasoning
    'Reverse Engineer',      # [New] Backward Chaining
    'Logical Critic',        # [New] Reflexion / Self-Critique
    'Visualizer',            # [New] Visual CoT / Mental Imagery
    'Axiomatic Purist',      # [New] Formal Definitions & Theorems
    'Unit Checker',          # [New] Dimensional Analysis
    'Step-back Abstractionist', # [New] Step-Back Prompting
    'Edge Case Hunter',      # [New] Boundary Value Analysis
    'Heuristic Estimator',   # [New] Fermian Estimation
    'Literal Translator',    # [New] Autoformalization
])

ROLE_DESCRIPTION = {
    # --- Existing Roles ---
    "Math Solver": 
        "You are a math expert. "
        "You will be given a math problem. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n",
    
    "Mathematical Analyst":
        "You are a mathematical analyst. "
        "You will be given a math problem, analysis and code from other agents. "
        "You need to first analyze the problem-solving process step by step, where the variables are represented by letters. "
        "Then you substitute the values into the analysis process to perform calculations and get the results."
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n"
        "You will be given some examples you may refer to.",
    
    "Programming Expert":
        "You are a programming expert. "
        "You will be given a math problem, analysis and code from other agents. "
        "Integrate step-by-step reasoning and Python code to solve math problems. "
        "Analyze the question and write functions to solve the problem. "
        "The function should not take any arguments and use the final result as the return value. "
        "The last line of code calls the function you wrote and assigns the return value to the \(answer\) variable. "
        "Use a Python code block to write your response. For example:\n```python\ndef fun():\n x = 10\n y = 20\n return x + y\nanswer = fun()\n```\n"
        "Do not include anything other than Python code blocks in your response."
        "You will be given some examples you may refer to.",
    
    "Inspector":
        "You are an Inspector. "
        "You will be given a math problem, analysis and code from other agents. "
        "Check whether the logic/calculation of the problem solving and analysis process is correct(if present). "
        "Check whether the code corresponds to the solution analysis(if present). "
        "Give your own solving process step by step based on hints. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n"
        "You will be given some examples you may refer to.",

    # --- [Added] New Roles (Research Based) ---
    
    "Problem Decomposer":
        "You are a Problem Decomposer. "
        "Do not try to solve the whole problem at once. "
        "Break down the complex math problem into smaller, manageable sub-questions or logical steps. "
        "Solve each sub-question sequentially to build up to the final answer. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n",

    "Pattern Recognizer":
        "You are a Pattern Recognizer. "
        "Identify the underlying mathematical pattern or category of the problem (e.g., arithmetic series, probability, geometry). "
        "Recall similar standard problems or theorems that fit this pattern. "
        "Apply the standard method for this specific type of problem to solve it. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n",

    "Reverse Engineer":
        "You are a Reverse Engineer. "
        "Instead of starting strictly from the given numbers, look at what is being asked (the goal). "
        "Work backward from the goal or check constraints to see which value fits the initial conditions. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n",

    "Logical Critic":
        "You are a Logical Critic. "
        "Actively look for potential logical fallacies, misinterpretations of the question, or common traps in the problem statement. "
        "Construct an argument for the correct path by critiquing potential wrong turns. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n",

    "Visualizer":
        "You are a Visualizer. "
        "Describe the problem in terms of a diagram, geometric shape, or visual scene to extract spatial or structural relationships. "
        "Solve the problem based on this visual understanding of the entities. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n",

    "Axiomatic Purist":
        "You are an Axiomatic Purist. "
        "Solve the problem by strictly adhering to mathematical definitions, axioms, and theorems. "
        "Avoid intuition; rely only on formal rules to derive the answer step-by-step. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n",

    "Unit Checker":
        "You are a Unit Checker. "
        "Focus on the units and dimensions (e.g., length, time, weight) in the problem to ensure consistency. "
        "Perform the calculation ensuring all units match throughout the process. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n",

    "Step-back Abstractionist":
        "You are a Step-back Abstractionist. "
        "First, identify the high-level general math principle or formula required (Abstraction). "
        "Then, apply the specific numbers from the question to that principle to solve it (Instantiation). "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n",

    "Edge Case Hunter":
        "You are an Edge Case Hunter. "
        "Test the logic of the problem by considering edge cases (e.g., what if x=0, x=1, or x is negative?) to verify the method. "
        "Apply the robust logic verified by these checks to the actual numbers. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n",

    "Heuristic Estimator":
        "You are a Heuristic Estimator. "
        "Perform a rough estimation or use common sense to identify the probable range of the answer first. "
        "Then perform the precise calculation, ensuring the result aligns with your estimation. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n",

    "Literal Translator":
        "You are a Literal Translator. "
        "Translate the natural language problem directly into formal mathematical equations or expressions line-by-line. "
        "Solve the resulting system of equations. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n",
}

# This function is inspired by/derived from the implementation in the following GitHub repository:
# Repository: https://github.com/chuanyang-Zheng/Progressive-Hint/blob/main/prompt/complex/complex_PHP_gsm8k.txt
# Repository: https://github.com/microsoft/ToRA/blob/213c1c995038c73fab10343814df7a42f990f026/src/prompts/tora/gsm8k.md
# Repository: https://github.com/microsoft/ToRA/blob/213c1c995038c73fab10343814df7a42f990f026/src/prompts/cot/gsm8k.md
FEW_SHOT_DATA = {
"Math Solver":
"""

""",

"Mathematical Analyst":
"""
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today? 
A: ## Problem solving process analysis

There are {ori_tree_num} trees originally.
Then there were {after_planted_tree_num} trees after some more were planted.
So the number of trees planted today {today_planted_num} is the number of trees after planting {after_planted_tree_num} minus the number of trees before planting {ori_tree_num}.
The answer is {today_planted_num} = {after_planted_tree_num} - {ori_tree_num}.

## Actual analysis and solution process

In this question, {ori_tree_num} = 15 and {after_planted_tree_num} = 21.
There are 15 trees originally. 
Then there were 21 trees after some more were planted. 
So the number of trees planted today must have been 21 - 15 = 6.
The answer is 6

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A:## Problem solving process analysis

Originally, Leah had {Leah_num} Leah_num chocolates.
Her sister had {sister_num} chocolates.
So in total they had {all_num} = {Leah_num} + {sister_num} chocolates.
After eating {eating_num} chocolates, the number of chocolates they have left {remain_num} is {all_num} minus {eating_num}. 
The answer is {remain_num} = {all_num} - {eating_num}.

## Actual analysis and solution process

In this question, {Leah_num} = 32, {sister_num} = 42 and {all_num} = 35.
So, in total they had 32 + 42 = 74 chocolates originally.
After eating 35 chocolates, they had 74 - 35 = 39 chocolates.
The answer is 39
""",

"Programming Expert":
"""
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A:
```python\n
def money_left():
    money_initial = 23
    bagels = 5
    bagel_cost = 3
    money_spent = bagels * bagel_cost
    remaining_money = money_initial - money_spent
    return remaining_money
 
answer = money_left()
\n```

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A:
```python\n
def remaining_golf_balls():
    golf_balls_initial = 58
    golf_balls_lost_tuesday = 23
    golf_balls_lost_wednesday = 2
    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
    remaining_golf_balls = golf_balls_left
    return remaining_golf_balls

answer = remaining_golf_balls() 
\n```
""",
"Inspector":"""""",
# [Added] Initialize new roles with empty strings to prevent KeyErrors
"Problem Decomposer": "",
"Pattern Recognizer": "",
"Reverse Engineer": "",
"Logical Critic": "",
"Visualizer": "",
"Axiomatic Purist": "",
"Unit Checker": "",
"Step-back Abstractionist": "",
"Edge Case Hunter": "",
"Heuristic Estimator": "",
"Literal Translator": "",
}

@PromptSetRegistry.register('Math_nocot')
class MathPromptSet(PromptSet):

    @staticmethod
    def get_role():
        return next(roles)

    @staticmethod
    def get_constraint(role):
        return ROLE_DESCRIPTION[role]
    
    @staticmethod
    def get_format():
        return "natural language"

    @staticmethod
    def get_answer_prompt(question, role="Mathematical Analyst"):
        # Format the question for the AI assistant to answer
        return f"{FEW_SHOT_DATA[role]}\n\nQ:{question}"

    @staticmethod
    def get_decision_constraint():
        return (
        "You will be given a math problem, analysis and code from other agents. "
        "Please find the most reliable answer based on the analysis and results of other agents. "
        "Give reasons for making decisions. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140")
    
    @staticmethod
    def get_decision_role():
        return "You are the top decision-maker. Good at analyzing and summarizing mathematical problems, judging and summarizing other people's solutions, and giving final answers to math problems."
    
    @staticmethod
    def get_decision_few_shot():
        return """
Q:Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?

A:After reviewing the analysis and results provided by the other agents, it appears that there is a discrepancy in the interpretation of the problem. The key point of contention is the base value on which the 150% increase is calculated.

Agents 623T, 8GTW, and 7R9w have calculated the 150% increase based on the total investment (purchase price + repair costs), which is $130,000. They have then added this increase to the total investment to find the new value of the house, and subsequently calculated the profit by subtracting the total investment from the new value of the house. This approach leads to a profit calculation of $195,000.

Agent 3qfQ, however, has interpreted the problem differently. This agent suggests that the 150% increase should be calculated based on the initial purchase price only, not the total investment. Following this method, the increased value is calculated as $80,000 (initial purchase price) + ($80,000 * 1.5), which equals $200,000. The profit is then calculated by subtracting the total investment ($130,000) from this increased value, resulting in a profit of $70,000.

The problem statement is ambiguous because it does not explicitly state whether the 150% increase is based on the initial purchase price alone or the total investment. However, the most common interpretation in real estate when referring to an increase in value due to repairs would be based on the initial purchase price, as the value increase is typically a reflection of the property's market value appreciation, not the sum of costs incurred.

Therefore, based on the typical real estate valuation practice and the more common interpretation of such scenarios, Agent 3qfQ's approach seems to be the most reliable. The profit should be calculated based on the increased value from the initial purchase price, not the total investment.

The final result, based on the most reliable interpretation, is a profit of $70,000.

The answer is 70000
"""
    
    @staticmethod
    def get_react_prompt(question, solution, feedback):
        return f"""Here is an unsuccessful attempt for solving the folloing question:
Question:
{question}
Attempted Solution:
{solution}
Feedback:\n{feedback}
Rewrite the code based on the feedback and the following question:
{question}"""


    @staticmethod
    def get_query_prompt(question):
        return (
"# Information Gathering for Question Resolution\n\n"
"Evaluate if additional information is needed to answer the question. "
#"If web search or file analysis is required, formulate specific queries to assist in finding the answer.\n\n"
"If a web search or file analysis is necessary, outline specific clues or details to be searched for.\n\n"
f"## Target Question:\n{question}\n\n"
# "## Information Gathering:\n"
# "Identify if a web search or file reading is necessary and outline the approach."
"## Clues for Investigation:\n"
"Identify critical clues and concepts within the question that are essential for finding the answer.\n"
        )


    @staticmethod
    def get_file_analysis_prompt(query, file):
        return (
            # "# File Analysis Required\n\n"
            # f"## Required Information to Extract:\n---\n{query}\n---\n\n"
            # f"## File Content for Analysis:\n---\n{file}\n---\n\n"
            # "## Instructions:\n"
            # "Extract the specified information from the file. Example: 'Identify the main theme in the text.'"
"# File Analysis Task\n\n"
f"## Information Extraction Objective:\n---\n{query}\n---\n\n"
f"## File Under Analysis:\n---\n{file}\n---\n\n"
"## Instructions:\n"
"1. Identify the key sections in the file relevant to the query.\n"
"2. Extract and summarize the necessary information from these sections.\n"
"3. Ensure the response is focused and directly addresses the query.\n"
"Example: 'Identify the main theme in the text.'"
        )


    @staticmethod
    def get_websearch_prompt(question, query):
        return (
            "# Web Search Task\n\n"
            f"## Original Question: \n---\n{question}\n---\n\n"
            f"## Targeted Search Objective:\n---\n{query}\n---\n\n"
            "## Simplified Search Instructions:\n"
            "Generate three specific search queries directly related to the original question. Each query should focus on key terms from the question. Format the output as a comma-separated list.\n"
            "For example, if the question is 'Who will be the next US president?', your queries could be: 'US presidential candidates, current US president, next US president'.\n"
            "Remember to format the queries as 'query1, query2, query3'."
        )



    @staticmethod
    def get_adversarial_answer_prompt(question):
        pass


    @staticmethod
    def get_distill_websearch_prompt(question, query, results):
        return (
            # "# Summarization of Search Results\n\n"
            # "## Required Information for Summary:\n---\n{query}\n---\n\n"
            # "## Search Results for Analysis:\n---\n{results}\n---\n\n"
            # "## Instructions:\n"
            # "Summarize the key findings from the search results related to the query. "
            # "Focus on relevant information. Example: 'Summary of key points...'"
"# Summarization of Search Results\n\n"
f"## Original question: \n---\n{question}\n---\n\n"
f"## Required Information for Summary:\n---\n{query}\n---\n\n"
f"## Analyzed Search Results:\n---\n{results}\n---\n\n"
"## Instructions for Summarization:\n"
"1. Review the provided search results and identify the most relevant information related to the question and query.\n"
"2. Extract and highlight the key findings, facts, or data points from these results.\n"
"3. Organize the summarized information in a coherent and logical manner.\n"
"4. Ensure the summary is concise and directly addresses the query, avoiding extraneous details.\n"  
"5. If the information from web search is useless, directly answer: \"No useful information from WebSearch\".\n"  
        )


    @staticmethod
    def get_reflect_prompt(question, answer):
        return (
"# Reflection on the Task\n\n"
f"## Reflection Question:\n---\n{question}\n---\n\n"
f"## Your Previous Answer:\n---\n{answer}\n---\n\n"
"## Instructions:\n"
"Reflect on your answer process, considering the accuracy, method, and reasoning."
        )


    @staticmethod
    def get_self_consistency(question: str, answers: list, constraint: str) -> str:
        formatted_answers = "\n".join([f"Answer {index + 1}: {answer}" for index, answer in enumerate(answers)])
        return (
            # "# Self-Consistency Evaluation Task\n\n"
            # f"## Given Question:\n---\n{question}\n---\n\n"
            # "## Available Answers:\n---\n"
            # f"{formatted_answers}\n"
            # "---\n\n"
            # "## Instructions:\n"
            # "Review the given answers and choose the most consistent one. "
            # "If all answers differ, select the one you find most reliable. "
            # f"Please keep following the constraints to answer the question: {constraint}."
"# Self-Consistency Evaluation Task\n\n"
f"## Question for Review:\n---\n{question}\n---\n\n"
f"## Reviewable Answers:\n---\n{formatted_answers}\n---\n\n"
"## Instructions for Selection:\n"
"1. Read each answer and assess how it addresses the question.\n"
"2. Compare the answers for their adherence to the given question's criteria and logical coherence.\n"
"3. Identify the answer that best aligns with the question's requirements and is the most logically consistent.\n"
"4. Ignore the candidate answers if they do not give a direct answer, for example, using 'unable to ...', 'as an AI ...'.\n"
"5. Copy the most suitable answer as it is, without modification, to maintain its original form.\n"
f"6. Adhere to the constraints: {constraint}.\n"
"Note: If no answer fully meets the criteria, choose and copy the one that is closest to the requirements."
        )

    @staticmethod
    def get_select_best(question: str, answers: list, constraint: str) -> str:
        formatted_answers = "\n".join([f"Answer {index + 1}: {answer}" for index, answer in enumerate(answers)])
        return (
            # "# Best Answer Evaluation Task\n\n"
            # f"## Given Question:\n---\n{question}\n---\n\n"
            # "## Available Answers:\n---\n"
            # f"{formatted_answers}\n"
            # "---\n\n"
            # "## Instructions:\n"
            # "Review the given question and candidate answers and choose the most reasonable one. "
            # "Please copy the original answer if you decide."
            # f"Please keep following the constraints to answer the question: {constraint}."
"# Best Answer Evaluation Task\n\n"
f"## Question:\n---\n{question}\n---\n\n"
f"## Candidate Answers for Evaluation:\n---\n{formatted_answers}\n---\n\n"
"## Evaluation Instructions:\n"
"1. Examine the question closely to understand its requirements.\n"
"2. Read each candidate answer thoroughly and assess its relevance and accuracy about the question.\n"
"3. Choose the answer that most accurately and completely addresses the question.\n"
"4. Ignore the candidate answers if they do not give a direct answer, for example, using 'unable to ...', 'as an AI ...'.\n"
"5. Copy the chosen answer exactly as it is presented, maintaining its original format.\n"
f"6. Adhere to the constraints: {constraint}.\n"
"Note: If none of the answers fully meet the question's criteria, select the one closest to fulfilling them."
        )

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)
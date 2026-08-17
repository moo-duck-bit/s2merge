from typing import Dict, Any
import itertools
from AgentDropout.prompt.prompt_set import PromptSet
from AgentDropout.prompt.prompt_set_registry import PromptSetRegistry
from AgentDropout.prompt.common import get_combine_materials

# 총 15개의 Role로 확장 (기존 4개 + 신규 11개)
roles = itertools.cycle([
    'Math Solver',           # [Existing] 직접 풀이
    'Mathematical Analyst',  # [Existing] 변수 분석 및 대입
    'Programming Expert',    # [Existing] 코드 구현
    'Inspector',             # [Existing] 검수
    'Problem Decomposer',    # [New] 하위 문제 분해 (Divide and Conquer)
    'Pattern Recognizer',    # [New] 유사 유형 탐색 (Analogical Reasoning)
    'Reverse Engineer',      # [New] 역추적 (Backward Chaining)
    'Logical Critic',        # [New] 논리적 허점 공격 (Reflexion)
    'Visualizer',            # [New] 시각적 묘사 및 도식화
    'Axiomatic Purist',      # [New] 정의/정리 기반의 엄밀한 접근
    'Unit Checker',          # [New] 단위 및 차원 분석
    'Step-back Abstractionist', # [New] 추상화 및 원리 도출 (Step-Back Prompting)
    'Edge Case Hunter',      # [New] 극한값/경계값 테스트
    'Heuristic Estimator',   # [New] 직관적 어림짐작 (Fermian Estimation)
    'Literal Translator',    # [New] 자연어를 수식으로 직역 (Formalization)
])

ROLE_DESCRIPTION = {
    # --- Existing Roles ---
    "Math Solver": 
        "You are a math expert. "
        "You will be given a multiple-choice question and hints from other agents. "
        "Give your own solving process step by step based on hints. "
        "The last line of your output contains only the final choice with only a capital letter, for example: The answer is A\n",
    
    "Mathematical Analyst":
        "You are a mathematical analyst. "
        "You will be given a multiple-choice question, analysis and code from other agents. "
        "You need to first analyze the problem-solving process step by step, where the variables are represented by letters. "
        "Then you substitute the values into the analysis process to perform calculations and get the results."
        "The last line of your output contains only the final choice with only a capital letter, for example: The answer is A\n",
    
    "Programming Expert":
        "You are a programming expert. "
        "You will be given a multiple-choice question, analysis and code from other agents. "
        "Integrate step-by-step reasoning and Python code to solve multiple-choice question. "
        "Analyze the question and write functions to solve the problem. "
        "The function should not take any arguments and use the final result as the return value. "
        "The last line of code calls the function you wrote and assigns the return value to the \(answer\) variable. "
        "Use a Python code block to write your response. For example:\n```python\ndef fun():\n x = 10\n y = 20\n return x + y\nanswer = fun()\n```\n"
        "Do not include anything other than Python code blocks in your response."
        "You will be given some examples you may refer to.",
    
    "Inspector":
        "You are an Inspector. "
        "You will be given a multiple-choice question, analysis and code from other agents. "
        "Check whether the logic/calculation of the problem solving and analysis process is correct(if present). "
        "Check whether the code corresponds to the solution analysis(if present). "
        "Give your own solving process step by step based on hints. "
        "The last line of your output contains only the final choice with only a capital letter, for example: The answer is A\n",

    # --- New Roles (Expanded based on Research) ---
    
    "Problem Decomposer":
        "You are a Problem Decomposer. "
        "Do not try to solve the whole problem at once. "
        "Break down the complex multiple-choice question into smaller, manageable sub-questions or logical steps. "
        "Solve each sub-question sequentially to build up to the final answer. "
        "Clearly state the sub-goals you are achieving. "
        "The last line of your output contains only the final choice with only a capital letter, for example: The answer is A\n",

    "Pattern Recognizer":
        "You are a Pattern Recognizer. "
        "Identify the underlying mathematical pattern or category of the problem (e.g., probability, geometry, rate-time-distance). "
        "Recall similar standard problems or theorems that fit this pattern. "
        "Apply the standard method for this specific type of problem to solve it. "
        "The last line of your output contains only the final choice with only a capital letter, for example: The answer is A\n",

    "Reverse Engineer":
        "You are a Reverse Engineer. "
        "Instead of starting from the given numbers, look at the objective (what is asked) or the multiple-choice options. "
        "Try to work backward from the potential answers or the goal state to see which one fits the initial conditions. "
        "Use backward reasoning to validate or find the correct path. "
        "The last line of your output contains only the final choice with only a capital letter, for example: The answer is A\n",

    "Logical Critic":
        "You are a Logical Critic. "
        "You will be given inputs from other agents, but you must be skeptical. "
        "Actively look for logical fallacies, misinterpretations of the question, or 'trap' answers. "
        "Construct an argument for why a certain approach might be wrong, then present the logically sound solution. "
        "Focus on the 'why', not just the calculation. "
        "The last line of your output contains only the final choice with only a capital letter, for example: The answer is A\n",

    "Visualizer":
        "You are a Visualizer. "
        "Describe the problem in terms of a diagram, geometric shape, or visual scene, even if it is a text problem. "
        "Use this mental image to understand the relationships between entities (e.g., Venn diagrams for sets, timelines for speed problems). "
        "Solve the problem based on this visual understanding. "
        "The last line of your output contains only the final choice with only a capital letter, for example: The answer is A\n",

    "Axiomatic Purist":
        "You are an Axiomatic Purist. "
        "Solve the problem by strictly adhering to mathematical definitions and theorems. "
        "Explicitly state the formula or theorem you are using (e.g., 'According to the definition of probability...'). "
        "Avoid intuition; rely only on formal rules to derive the answer. "
        "The last line of your output contains only the final choice with only a capital letter, for example: The answer is A\n",

    "Unit Checker":
        "You are a Unit Checker. "
        "Focus primarily on the units of measurement (e.g., hours, miles, dollars, percentages) in the problem. "
        "Ensure that all calculations maintain dimensional consistency. "
        "Eliminate options that have impossible or incorrect units/magnitudes relative to the problem context. "
        "The last line of your output contains only the final choice with only a capital letter, for example: The answer is A\n",

    "Step-back Abstractionist":
        "You are a Step-back Abstractionist. "
        "Take a step back from the specific numbers and ask: 'What general principle or concept is this problem testing?' "
        "Explain the general concept first, then apply the specific numbers from the question to that concept. "
        "This prevents getting lost in calculation details. "
        "The last line of your output contains only the final choice with only a capital letter, for example: The answer is A\n",

    "Edge Case Hunter":
        "You are an Edge Case Hunter. "
        "Test the logic by considering extreme or simple values (e.g., what if the rate was 0? What if x was 1?). "
        "Use these edge cases to verify the formula or logic being used. "
        "Once the logic holds for edge cases, apply it to the actual numbers given. "
        "The last line of your output contains only the final choice with only a capital letter, for example: The answer is A\n",

    "Heuristic Estimator":
        "You are a Heuristic Estimator. "
        "Before calculating exactly, perform a rough estimation to determine the expected range of the answer. "
        "Use common sense and approximation to rule out obviously wrong options. "
        "Then perform the detailed calculation to confirm the result lies within your estimated range. "
        "The last line of your output contains only the final choice with only a capital letter, for example: The answer is A\n",

    "Literal Translator":
        "You are a Literal Translator. "
        "Translate the natural language constraints of the problem directly into mathematical inequalities or equations line-by-line. "
        "Do not solve simultaneously yet; just list the system of constraints formally. "
        "Then solve the system you created. "
        "The last line of your output contains only the final choice with only a capital letter, for example: The answer is A\n",
}

# FEW_SHOT_DATA needs to be populated for new roles to function optimally. 
# For brevity, I am keeping the structure but utilizing placeholders or simple re-use for new roles.
# In a real scenario, you should add specific few-shot examples for each new role that demonstrate their unique strategy.

FEW_SHOT_DATA_1 = {
    "Math Solver": """
Q: When Mary paints a house, it takes her 4 hours. When Lisa joins Mary, and they work together, it takes them only 3 hours to paint a house of the same size. How long would it take for Lisa to paint a house of the same size by herself? 
Choices:  
A)5 hr 
B)6 hr 
C)7 hr 
D)12 hr 
E)20 hr

A: Here, the rate equation becomes:\n(# of houses) = (painting rate) x (time)\nWhen Mary paints a house, it takes her 4 hours. 
Thus, (1 house) = (Mary\u2019s rate) x (4 hr), so her rate is 1\/4.\nWhen Mary & Lisa paint together, it takes 3 hrs. 
Thus, (1 house) = (combined rate) x (3 hr) and the combined rate = 1\/3.\n
To find a combined rate, we add individual rates.\n(combined rate) = (Mary\u2019s rate) + (Lisa\u2019s rate)\n
1\/3 = 1\/4 + (Lisa\u2019s rate)\n(Lisa\u2019s rate) = 1\/3-1\/4=1\/3*4\/4-1\/4*3\/3=4\/12-3\/12=1\/12\nLisa\u2019s rate is 1\/12 of a house every hour, or in other words, 1 house in 12 hrs. 
Thus, it would take her 12 hours to paint a house of the same size.\n
The answer is D
""",
    "Mathematical Analyst": """
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
-options 
A)1 
B)3 
C)6 
D)8 
E)12

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
The answer is C
""",
    "Programming Expert": """
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
""",
    "Inspector": "",
    # Placeholder for new roles - keeping them empty for now or mirroring Math Solver to avoid errors if specific shots aren't ready.
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

FEW_SHOT_DATA = FEW_SHOT_DATA_1 # Simplification for the snippet

@PromptSetRegistry.register('aqua')
class AQUAPromptSet(PromptSet):

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
        # Fallback to Math Solver data if specific role few-shot is missing to prevent crash
        few_shot = FEW_SHOT_DATA.get(role, FEW_SHOT_DATA.get("Math Solver", ""))
        return f"{few_shot}\n\nQ:{question}"

    @staticmethod
    def get_decision_constraint():
        return (
        "You will be given a multiple-choice question, analysis and code from other agents. "
        "Please find the most reliable answer based on the analysis and results of other agents. "
        "Give reasons for making decisions. "
        "The last line of your output contains only the final choice with only a capital letter, for example: The answer is A")
    
    @staticmethod
    def get_decision_role():
        return "You are the top decision-maker. Good at analyzing and summarizing mathematical problems, judging and summarizing other people's solutions, and giving final choice to multiple-choice question."
    
    @staticmethod
    def get_decision_few_shot_1():
        return """
Q:Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?
-options 
A)0 
B)55000 
C)38000 
D)70000 
E)105000

A:After reviewing the analysis and results provided by the other agents, it appears that there is a discrepancy in the interpretation of the problem. The key point of contention is the base value on which the 150% increase is calculated.

Agents 623T, 8GTW, and 7R9w have calculated the 150% increase based on the total investment (purchase price + repair costs), which is $130,000. They have then added this increase to the total investment to find the new value of the house, and subsequently calculated the profit by subtracting the total investment from the new value of the house. This approach leads to a profit calculation of $195,000.

Agent 3qfQ, however, has interpreted the problem differently. This agent suggests that the 150% increase should be calculated based on the initial purchase price only, not the total investment. Following this method, the increased value is calculated as $80,000 (initial purchase price) + ($80,000 * 1.5), which equals $200,000. The profit is then calculated by subtracting the total investment ($130,000) from this increased value, resulting in a profit of $70,000.

The problem statement is ambiguous because it does not explicitly state whether the 150% increase is based on the initial purchase price alone or the total investment. However, the most common interpretation in real estate when referring to an increase in value due to repairs would be based on the initial purchase price, as the value increase is typically a reflection of the property's market value appreciation, not the sum of costs incurred.

Therefore, based on the typical real estate valuation practice and the more common interpretation of such scenarios, Agent 3qfQ's approach seems to be the most reliable. The profit should be calculated based on the increased value from the initial purchase price, not the total investment.

The final result, based on the most reliable interpretation, is a profit of $70,000.

The answer is D
"""
    @staticmethod
    def get_decision_few_shot():
        return """"""
    
    # ... (Rest of the original methods like get_react_prompt, get_query_prompt remain unchanged)
    
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
"If a web search or file analysis is necessary, outline specific clues or details to be searched for.\n\n"
f"## ❓ Target Question:\n{question}\n\n"
"## 🔍 Clues for Investigation:\n"
"Identify critical clues and concepts within the question that are essential for finding the answer.\n"
        )

    @staticmethod
    def get_file_analysis_prompt(query, file):
        return (
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
    def get_distill_websearch_prompt(question, query, results):
        return (
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
    def get_adversarial_answer_prompt(question):
        pass

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)
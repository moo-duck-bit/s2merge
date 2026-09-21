"""Role catalogue and prompts for GSM8K-style math word problems.

The first four roles are the pool AgentDropout ships with. The remaining eleven
extend it to the 15-agent pool the paper routes over; each one encodes a
distinct reasoning strategy, so the router has genuinely different perspectives
to choose between.
"""

import itertools
from typing import List

from s2rmerge.mas.prompts.base import PromptSet
from s2rmerge.mas.prompts.registry import PromptSetRegistry

ROLES: List[str] = [
    # Inherited from AgentDropout.
    "Math Solver",
    "Mathematical Analyst",
    "Programming Expert",
    "Inspector",
    # Added for S2R-Merge, one reasoning strategy each.
    "Problem Decomposer",        # least-to-most
    "Pattern Recognizer",        # analogical reasoning
    "Reverse Engineer",          # backward chaining
    "Logical Critic",            # self-critique
    "Visualizer",                # visual chain-of-thought
    "Axiomatic Purist",          # definition-driven derivation
    "Unit Checker",              # dimensional analysis
    "Step-back Abstractionist",  # step-back prompting
    "Edge Case Hunter",          # self-verification
    "Heuristic Estimator",       # order-of-magnitude estimation
    "Literal Translator",        # autoformalisation
]

_role_cycle = itertools.cycle(ROLES)

ROLE_DESCRIPTION = {
    "Math Solver": 
        "You are a math expert. "
        "You will be given a math problem and hints from other agents. "
        "Give your own solving process step by step based on hints. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n"
        "You will be given some examples you may refer to.",

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
        "The last line of code calls the function you wrote and assigns the return value to the `answer` variable. "
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

    "Problem Decomposer":
        "You are a Problem Decomposer. "
        "Do not try to solve the whole problem at once. "
        "Break down the complex math problem into smaller, manageable sub-questions or logical steps. "
        "Solve each sub-question sequentially to build up to the final answer. "
        "Clearly state the sub-goals you are achieving. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n",

    "Pattern Recognizer":
        "You are a Pattern Recognizer. "
        "Identify the underlying mathematical pattern or category of the problem (e.g., arithmetic series, ratio, rate-time-distance). "
        "Recall similar standard problems or theorems that fit this pattern. "
        "Apply the standard method for this specific type of problem to solve it. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n",

    "Reverse Engineer":
        "You are a Reverse Engineer. "
        "Instead of starting strictly from the given numbers, look at what is being asked (the goal). "
        "Work backward from the goal to the initial conditions to verify the steps. "
        "Use backward reasoning to validate or find the correct path. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n",

    "Logical Critic":
        "You are a Logical Critic. "
        "You will be given inputs from other agents (if available) or the problem itself. "
        "Actively look for logical fallacies, misinterpretations of the question, or common traps. "
        "Construct an argument for why a certain approach might be wrong, then present the logically sound solution. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n",

    "Visualizer":
        "You are a Visualizer. "
        "Describe the problem in terms of a diagram, geometric shape, or visual scene (e.g., 'imagine a timeline', 'draw a box for the total'). "
        "Use this mental image to understand the relationships between entities. "
        "Solve the problem based on this visual understanding. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n",

    "Axiomatic Purist":
        "You are an Axiomatic Purist. "
        "Solve the problem by strictly adhering to mathematical definitions and basic arithmetic rules. "
        "Explicitly state the operation you are using (e.g., 'Since x is added to y...'). "
        "Avoid guessing; rely only on formal rules to derive the answer. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n",

    "Unit Checker":
        "You are a Unit Checker. "
        "Focus primarily on the units of measurement (e.g., hours, minutes, dollars, counts) in the problem. "
        "Ensure that all calculations maintain dimensional consistency (e.g., convert minutes to hours before adding). "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n",

    "Step-back Abstractionist":
        "You are a Step-back Abstractionist. "
        "Take a step back from the specific numbers and ask: 'What general arithmetic principle is this problem testing?' "
        "Explain the general concept first (e.g., 'This is a subtraction problem involving a total and parts'). "
        "Then apply the specific numbers from the question to that concept. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n",

    "Edge Case Hunter":
        "You are an Edge Case Hunter. "
        "Test your logic by considering simple or extreme values mentally (e.g., 'What if he bought 0 apples?'). "
        "Use these checks to verify the formula or logic being used is robust. "
        "Then apply the correct logic to the actual numbers given. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n",

    "Heuristic Estimator":
        "You are a Heuristic Estimator. "
        "Before calculating exactly, perform a rough estimation to determine the expected range of the answer (e.g., 'It should be roughly 200'). "
        "Use common sense to guide your detailed calculation. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n",

    "Literal Translator":
        "You are a Literal Translator. "
        "Translate the natural language sentences of the problem directly into mathematical equations line-by-line. "
        "Do not solve simultaneously yet; just list the equations formally. "
        "Then solve the system you created. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n",
}

# This function is inspired by/derived from the implementation in the following GitHub repository:
# Repository: https://github.com/chuanyang-Zheng/Progressive-Hint/blob/main/prompt/complex/complex_PHP_gsm8k.txt
# Repository: https://github.com/microsoft/ToRA/blob/213c1c995038c73fab10343814df7a42f990f026/src/prompts/tora/gsm8k.md
# Repository: https://github.com/microsoft/ToRA/blob/213c1c995038c73fab10343814df7a42f990f026/src/prompts/cot/gsm8k.md
FEW_SHOT_DATA = {
"Math Solver":
"""
Q: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. 
They have 2 chapters of their textbook to study and 4 worksheets to memorize. 
They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. 
If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, 
include 3 10-minute snack breaks each day, and 30 minutes for lunch each day? (Hint: The answer is near to 4).

A: We know the Answer Hints: 4. With the Answer Hints: 4, we will answer the question. 
Let's think step by step. 
Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.
For the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.
Angelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.
However, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, 
so 12 total hours x 10 minutes = 120 extra minutes for breaks.
They also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.
And they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.
So Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.
They want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75
They will need to plan to study 4 days to allow for all the time they need.
The answer is 4

Q: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles? (Hint: The answer is near to 160,145).
A: We know the Answer Hints: 160, 145. With the Answer Hints: 160, 145, we will answer the question.
Let's think step by step
When Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/5*60 = 24
The total number of marbles she'll have is 60+24 = 84
If Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees.
If Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more frisbees.
The total number of frisbees she'll have will increase to 30+12 = 42
Bella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards
If she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards.
The total number of deck cards she'll have is 10+4 = 14
Together, Bella will have a total of 14+42+84 = 140 items
The answer is 140

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
# Roles without their own worked examples fall back to the Math Solver block.
"Inspector": "",
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

@PromptSetRegistry.register('gsm8k')
class GSM8KPromptSet(PromptSet):
    """Prompts for GSM8K, MultiArith and SVAMP, which share an answer format."""

    @staticmethod
    def roles():
        return list(ROLES)

    @staticmethod
    def get_role():
        return next(_role_cycle)

    @staticmethod
    def get_constraint(role):
        return ROLE_DESCRIPTION[role]

    @staticmethod
    def get_answer_prompt(question, role="Mathematical Analyst"):
        few_shot = FEW_SHOT_DATA.get(role) or FEW_SHOT_DATA["Math Solver"]
        return f"{few_shot}\n\nQ:{question}"

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
    

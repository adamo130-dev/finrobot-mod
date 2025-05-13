from finrobot.agent import FinGPTAgent

agent = FinGPTAgent()

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = agent.chat(user_input)
    print(f"FinRobot: {response}")
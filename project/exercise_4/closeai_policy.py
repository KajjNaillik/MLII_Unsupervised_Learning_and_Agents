from agent import Agent
import numpy as np

def closeai_policy(agent: Agent) -> str:
    """
    Improved stochastic policy for the agent.
    Returns "left", "right", or "none" based on a stochastic strategy.
    """
    if not hasattr(closeai_policy, "flag"):
        closeai_policy.flag = 0

    # print(agent.position, " --- ", agent.known_rewards)

    if agent.known_rewards[agent.position] != 0:
        return "none"

    if agent.position == 0:
        closeai_policy.flag = 1
    elif agent.position == len(agent.known_rewards) - 1:
        closeai_policy.flag = 0

    if closeai_policy.flag == 0 and agent.position != 0:
        action = "left"
    elif closeai_policy.flag == 1 and agent.position != len(agent.known_rewards) - 1:
        action = "right"

    return action

from othello.agents.base import Agent


class RandomAgent(Agent):
    name = "random"

    def act(self, state):
        return self.rng.choice(self.env.legal_moves(state))

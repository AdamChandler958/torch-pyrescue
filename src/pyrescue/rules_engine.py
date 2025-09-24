import logging

from src.pyrescue.state_manager import StateManager

logger = logging.getLogger(__name__)


class RulesEngine:
    def __init__(self, state_manager: StateManager):
        self.rules = {
            "NaN_Detected": self.nan_rule,
        }
        self.state_manager = state_manager

    def update(self, flag_name: str, flag_value: bool):
        if flag_value:
            logger.info(f"Flag {flag_name} activated. Applying Rules...")

    def apply_rules(self, flag_name: str):
        rule_function = self.rules.get(flag_name, self.default_rule)
        rule_function()

    def nan_rule(self):
        logger.info("Applying learning rate decrease...")
        self.state_manager.load_state()
        self.state_manager.apply_lr_decrease()

        raise ValueError("NaN value detected")

    def default_rule():
        logger.info("No specific rule found. Executing a default action.")

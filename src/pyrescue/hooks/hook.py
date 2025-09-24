from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.pyrescue.rules_engine import RulesEngine


class FlagManager:
    def __init__(self):
        self._observers: list[RulesEngine] = []

    def register_observer(self, observer: RulesEngine):
        self._observers.append(observer)

    def notify_observers(self, flag_name: str, flag_value: bool):
        for observer in self._observers:
            observer.update(flag_name, flag_value)


class Hook(FlagManager):
    def __init__(self, name: str, is_forward_hook: bool):
        super().__init__()
        self.name = name
        self.is_forward_hook = is_forward_hook
        self._flag_status = False

    @property
    def flag_status(self):
        return self._flag_status

    @flag_status.setter
    def flag_status(self, value: bool):
        if self._flag_status != value:
            self._flag_status = value
            self.notify_observers(self.name, self._flag_status)

    def hook(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the hook method.")

from typing import Callable, Any
from .connection import Connection

class Node:
    def __init__(self, name: str, connection: Connection) -> None:
        super().__init__(name)
        self._children = {}
        self._executer = lambda _: True
        
    def send(self, data):
        # if this node decide to forward the data
        if self._executer(data):
            # broadcast to all possible handler
            for _, (node, criterion)  in self._children.items():
                if criterion(data):
                    node.send(data)      

    def recieve(self, buffer=None):
        return super().recieve(buffer)
    
    def add_child(
        self,
        node: "Node",
        forward_criterion: Callable[[Any], bool]
    ) -> None:
        self._children[node.name] = node, forward_criterion
        
    def set_executer(self, executer: Callable[[Any], bool]):
        self._executer = executer
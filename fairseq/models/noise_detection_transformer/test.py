import torch

class A:
    __name = "cls A"

    def __init__(self) -> None:
        self.__apple = "apple"

    def tt(self):
        print(exec("self.__apple"))
    @classmethod
    def t(cls):
        # print(cls.__name)
        print(getattr(cls, "__name"))


def ttt(self):
    print(self.__apple)
a = A()
# a.t()
# a.tt()
a.ttt = ttt
print(a.ttt())

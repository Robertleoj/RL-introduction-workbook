
from multiprocessing import Manager, Lock
from itertools import repeat


def prog_bar(curr, total):
    perc = ((curr) / total) * 100
    string = f"{int(perc):<3}% ["

    bar = [' ' if i > perc
                else "█" for i in range(100) ]
    string += "".join(bar) + ']'
    string += f" ({curr} / {total})"
    return string

def print_prog_bar(curr, total, start =1):
    print("\r" + prog_bar(curr, total), end='')
    if curr == total:
        print()



class CompleteTaskCounter:
    def __init__(self, tasks):
        self.m = Manager()
        self.count = self.m.Value('i', 0)
        self.lock = Lock()
        self.tasks = tasks

    def increment(self, val):
        with self.lock:
            val.value += 1
    
    def print_prog(self):
        print_prog_bar(self.count.value, self.tasks)

    def incrprint(self, val):
        self.increment(val)
        self.print_prog()

    def reset(self, tasks):
        self.__init__(tasks)

    def rep_gen(self):
        return repeat(self.count, self.tasks)



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
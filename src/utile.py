# from utile import cls, exit

class cls(object):
    def __repr__(self):
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        return ''

cls = cls()

class exit(object):
    exit = exit # original object
    def __repr__(self):
        self.exit() # call original
        return ''

quit = exit = exit()
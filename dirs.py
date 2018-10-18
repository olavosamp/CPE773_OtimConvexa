results = "../results/"
figures = "../figures/"

def make_folder(path):
    import os

    try:
        os.makedirs(path)
    except OSError:
        pass

make_folder(results)
make_folder(figures)
